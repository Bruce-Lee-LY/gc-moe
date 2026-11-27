#pragma once

#include "mma.cuh"

#include <cstddef>

namespace gc_moe {

// Pair the scales for the two rows handled by each lane group of the MMA.
__device__ __forceinline__ int paired_scale_row(int row) {
    return (row / 16) * 16 + (row % 8) * 2 + (row % 16) / 8;
}

// In each M16 panel, keep two adjacent K scales together for each row pair.
__device__ __forceinline__ int packed_gate_scale_index(int row_pair, int column) {
    return (column / 2 * 8 + row_pair) * 2 + column % 2;
}

template <int TileRows, int Chunk>
__device__ __forceinline__ void copy_swizzled_rows(uint8_t* dst, const uint8_t* src, int rows, int stride) {
    const uint32_t address = __cvta_generic_to_shared(dst);
    const uint64_t row_bytes = stride;
    const uint64_t bytes = row_bytes * rows;
    asm volatile("ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.LLC::128B.padz.swzl.b8 "
                 "[%0], [%1], {%2,%3,%4}, {%2,%5,%4}, {%6,%7}, {%8,%8,%8}, %9;"
                 :: "r"(address), "l"(src), "r"(Chunk), "r"(rows), "r"(1), "r"(TileRows),
                    "l"(row_bytes), "l"(bytes), "r"(0), "r"(Chunk == 128 ? 0 : 1) : "memory");
}

template <int TileRows, int Slices, int ScaleStrideRows = TileRows>
__device__ __forceinline__ void copy_a_scales(uint16_t* dst, const uint16_t* src, int live_rows) {
    const uint32_t address = __cvta_generic_to_shared(dst);
    constexpr uint64_t stride = ScaleStrideRows * sizeof(uint16_t);
    asm volatile("ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.padz.linear.b16 "
                 "[%0], [%1], {%2,%3,%4}, {%5,%3,%4}, {%6,%7}, {%8,%8,%8}, %8;"
                 :: "r"(address), "l"(src), "r"(live_rows), "r"(Slices), "r"(1), "r"(TileRows),
                    "l"(stride), "l"(stride * Slices), "r"(0) : "memory");
}

template <int Chunk>
__device__ __forceinline__ uint4 matrix_fragment(const uint8_t* src) {
    const uint64_t address = __cvta_generic_to_shared(src) / 16;
    uint4 result;
    asm volatile("ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 "
                 "{%0,%1,%2,%3}, [%4], %5, %6, %7;"
                 : "=r"(result.x), "=r"(result.y), "=r"(result.z), "=r"(result.w)
                 : "l"(address), "r"(1), "r"(Chunk / 2), "r"(Chunk == 128 ? 0 : 1) : "memory");
    return result;
}

template <int M, int N, int KBytes, bool FP4 = true, int ChunkBytes = (M == 16 ? 128 : 64),
          int StageCount = 3, bool PairedScales = false, int RowPanels = 1, bool CycleStages = false, bool MaskScales = false,
          bool SplitColumns = false, int ScaleStrideRows = M, int PacketWidth = 0, bool FP8_FP4 = false, int WarpN = 64, bool Gate = false, int AStride = KBytes, bool CachedAScales = false>
struct BlockProject {
    static constexpr int MPanels = RowPanels;
    static constexpr int NWarps = N / WarpN;
    // A packed weight group always covers 64 output columns. Large M
    // tiles can assign its two halves to separate compute warps.
    static constexpr int WeightWarps = N / 64;
    static constexpr int WeightPanels = 4;
    static_assert(WarpN == 64 || (FP8_FP4 && WarpN == 32));
    static constexpr int MWarps = M / (16 * RowPanels);
    static constexpr int Threads = NWarps * MWarps * 32;
    static constexpr int NPanels = N / (16 * NWarps);
    static constexpr int Stages = StageCount <= KBytes / ChunkBytes + 1 ? StageCount : KBytes / ChunkBytes + 1;
    static constexpr int Chunk = ChunkBytes;
    static constexpr int Slices = Chunk / 32;
    static_assert(!FP8_FP4 || (!FP4 && Chunk >= 64));
    static_assert(!CachedAScales || (FP4 && MaskScales));
    static constexpr int BWords = WeightPanels * (FP8_FP4 ? 16 : 32);
    static constexpr int ScaleWords = FP8_FP4 ? 4 : 8;

    struct alignas(128) Stage {
        uint8_t a[M * Chunk];
        uint4 packets[WeightWarps * Slices * PackedWeightWords<FP4, FP8_FP4>];
        uint16_t sa[Slices * M];
    };

    struct Result {
        Accumulator block[RowPanels][NPanels];
    };

    struct alignas(128) GroupStage {
        uint8_t a[M * Chunk];
        uint4 packets[2 * Slices * PackedWeightWords<FP4, FP8_FP4>];
        uint16_t sa[Slices * M];
    };

    struct GroupStorage {
        GroupStage groups[NWarps <= 2 ? 1 : NWarps / 2][StageCount];
    };

    static constexpr int ScratchBytes = SplitColumns ? sizeof(GroupStorage) : sizeof(Stage) * Stages;

    // Rebasing keeps all nonzero E2M1 values exactly representable in E5M2.
    __device__ __forceinline__ static uint2 e5m2_magnitudes(uint32_t bias) {
        return {(0x0a080400u + bias) & 0xffffff00u, 0x12100e0cu + bias};
    }

    __device__ __forceinline__ static uint2 expand_e5m2_pair(uint32_t codes, uint2 magnitudes) {
        constexpr uint32_t Sign = 0x80808080u;
        const uint32_t lo = __byte_perm(magnitudes.x, magnitudes.y, codes & 0x7777u);
        const uint32_t hi = __byte_perm(magnitudes.x, magnitudes.y, (codes >> 16) & 0x7777u);
        return {lo | (codes & Sign), hi | ((codes << 4) & Sign)};
    }

    template <bool UseLut = false>
    __device__ __forceinline__ static uint4 load_weight(const uint4* packet, int step, int panel,
                                                       int lane, uint2 scales, const uint2* lut = nullptr) {
        const auto* fragment = packet + step * BWords;
        if constexpr (FP8_FP4) {
            const uint2 packed = reinterpret_cast<const uint2*>(fragment)[panel * 32 + lane];
            uint2 lo, hi;
            if constexpr (UseLut) {
                const uint32_t pair = panel < 2 ? scales.x : scales.y;
                const int shift = 16 * (panel % 2);
                lo = lut[(pair >> (shift + 2)) & 31u];
                hi = lut[(pair >> (shift + 10)) & 31u];
            } else {
                const int selector = panel * 0x2222;
                lo = e5m2_magnitudes(__byte_perm(scales.x, scales.y, selector));
                hi = e5m2_magnitudes(__byte_perm(scales.x, scales.y, selector + 0x1111));
            }
            // Each word holds both K halves for one output column.
            const auto lower_column = expand_e5m2_pair(packed.x, lo);
            const auto upper_column = expand_e5m2_pair(packed.y, hi);
            return {lower_column.x, lower_column.y, upper_column.x, upper_column.y};
        } else {
            return fragment[panel * 32 + lane];
        }
    }

    // Each column group has a separate activation and weight pipeline.
    __device__ __forceinline__ static Result run_warp_groups(const uint8_t* a, const uint16_t* sa, const uint4* b,
                                                int rows, GroupStorage& storage) {
        static_assert(M % (16 * RowPanels) == 0);
        static_assert((NPanels == 4 || (FP8_FP4 && NPanels == 2)) && KBytes % Chunk == 0);
        constexpr int KGroups = KBytes / Chunk;
        const int warp = threadIdx.x / 32;
        const int warp_m = warp % MWarps;
        const int warp_n = warp / MWarps;
        const int lane = threadIdx.x % 32;
        constexpr int ThreadsPerGroup = MWarps * 2 * 32;
        const int group = warp_n / 2;
        const int group_warp = warp % (MWarps * 2);
        auto* stages = storage.groups[group];

        uint32_t scale_masks[RowPanels];
        if constexpr (MaskScales) {
            #pragma unroll
            for (int m = 0; m < RowPanels; ++m) {
                const int row = (warp_m * RowPanels + m) * 16 + lane / 4;
                scale_masks[m] = (row < rows ? 0xffffu : 0) | (row + 8 < rows ? 0xffff0000u : 0);
            }
        }
        auto issue = [&](int k, GroupStage& stage) {
            if (group_warp == 0 && lane == 0) {
                copy_swizzled_rows<M, Chunk>(stage.a, a + k * Chunk, rows, AStride);
                if constexpr (FP4 && !CachedAScales) {
                    const int scale_rows = PairedScales ? (rows + 15) / 16 * 16 : rows;
                    copy_a_scales<M, Slices, ScaleStrideRows>(stage.sa, sa + k * Slices * ScaleStrideRows, scale_rows);
                }
            }
            if (group_warp == 1 && lane == 0) {
                constexpr int packet_words = 2 * Slices * PackedWeightWords<FP4, FP8_FP4>;
                // Use the .rt request policy for streamed routed weight packets.
                constexpr int width = M == 32 && ScaleStrideRows == 32 ? 64 : 128;
                copy_linear_tile<packet_words * sizeof(uint4) / width, true, width>(
                    stage.packets, b + (group * KGroups + k) * packet_words);
            }
            asm volatile("ppu.cp.async.commit_group;" ::: "memory");
        };

        Result result{};
        constexpr int Ahead = Stages - 1;
        static_assert(Ahead >= 1 && Ahead <= KGroups);
        #pragma unroll
        for (int k = 0; k < Ahead; ++k)
            issue(k, stages[k]);
        auto compute = [&](const GroupStage& stage, int k) {
            #pragma unroll
            for (int slice = 0; slice < Slices; ++slice) {
                uint4 av[RowPanels];
                uint32_t asf[RowPanels]{};
                #pragma unroll
                for (int m = 0; m < RowPanels; ++m) {
                    const int row = (warp_m * RowPanels + m) * 16;
                    av[m] = matrix_fragment<Chunk>(stage.a + row * Chunk + slice * 32);
                    if constexpr (FP4) {
                        if constexpr (CachedAScales) {
                            const int k_index = k * Slices + slice;
                            const auto* scales = reinterpret_cast<const uint32_t*>(sa);
                            const int panel_base = (row / 16) * (KBytes / 32) * 8;
                            asf[m] = scales[panel_base + packed_gate_scale_index(lane / 4, k_index)];
                            asf[m] &= scale_masks[m];
                        } else {
                            const auto* scale = stage.sa + slice * M + row;
                            if constexpr (PairedScales) {
                                asf[m] = reinterpret_cast<const uint32_t*>(scale)[lane / 4];
                                if constexpr (MaskScales)
                                    asf[m] &= scale_masks[m];
                            } else {
                                asf[m] = scale[lane / 4] | (uint32_t(scale[lane / 4 + 8]) << 16);
                            }
                        }
                    }
                }
                const auto* packet = stage.packets + (slice / 2) * 4 * PackedWeightWords<FP4, FP8_FP4>;
                const int weight_step = (warp_n % 2) * 2 + slice % 2;
                uint4 scales{};
                uint2 routed_scales{};
                if constexpr (FP8_FP4) {
                    const auto* bases = packet + 4 * BWords + weight_step * ScaleWords;
                    routed_scales = reinterpret_cast<const uint2*>(bases)[lane / 4];
                } else if constexpr (FP4) {
                    scales = packet[4 * BWords + weight_step * ScaleWords + lane / 4];
                }
                const uint32_t bsf[] = {scales.x, scales.y, scales.z, scales.w};
                #pragma unroll
                for (int n = 0; n < NPanels; ++n) {
                    const uint4 bv = load_weight(packet, weight_step, n, lane, routed_scales);
                    #pragma unroll
                    for (int m = 0; m < RowPanels; ++m)
                        mma<FP4, FP8_FP4>(result.block[m][n], av[m], bv, asf[m], bsf[n]);
                }
            }
        };
        if constexpr (CycleStages) {
            // Index each stage at a fixed offset through the steady-state loop.
            constexpr int Steady = (KGroups - Ahead) / Stages * Stages;
            for (int base = 0; base < Steady; base += Stages) {
                #pragma unroll
                for (int slot = 0; slot < Stages; ++slot) {
                    asm volatile("ppu.cp.async.wait_group %0;" :: "n"(Ahead - 1) : "memory");
                    __ppu_barrier_sync(3 + group, ThreadsPerGroup);
                    issue(base + slot + Ahead, stages[(slot + Ahead) % Stages]);
                    compute(stages[slot], base + slot);
                }
            }
            #pragma unroll
            for (int k = Steady; k < KGroups; ++k) {
                const int slot = k % Stages;
                if (k + Ahead <= KGroups)
                    asm volatile("ppu.cp.async.wait_group %0;" :: "n"(Ahead - 1) : "memory");
                else
                    asm volatile("ppu.cp.async.wait_all;" ::: "memory");
                __ppu_barrier_sync(3 + group, ThreadsPerGroup);
                if (k + Ahead < KGroups)
                    issue(k + Ahead, stages[(slot + Ahead) % Stages]);
                compute(stages[slot], k);
            }
        } else {
            int slot = 0;
            for (int k = 0; k < KGroups; ++k) {
                if (k + Ahead <= KGroups)
                    asm volatile("ppu.cp.async.wait_group %0;" :: "n"(Ahead - 1) : "memory");
                else
                    asm volatile("ppu.cp.async.wait_all;" ::: "memory");
                // The barrier also releases the stage consumed in the preceding iteration.
                __ppu_barrier_sync(3 + group, ThreadsPerGroup);
                if (k + Ahead < KGroups)
                    issue(k + Ahead, stages[slot == 0 ? Stages - 1 : slot - 1]);
                compute(stages[slot], k);
                slot = slot == Stages - 1 ? 0 : slot + 1;
            }
        }
        return result;
    }

    struct InputTile {
        const uint8_t* activation;
        const uint16_t* scales;
        const uint4* weights;
        int rows;
    };

    template <bool RtWeights = false, bool StageReady = false, int CycleCount = 1,
              bool TilePrefetch = false, int StageOffset = 0, uint32_t WholeTileMask = 0,
              typename FollowingTile = std::nullptr_t>
    __device__ __forceinline__ static Result run(const uint8_t* a, const uint16_t* sa, const uint4* b,
                                                int rows, Stage* stages, const uint32_t* ready_columns = nullptr,
                                                bool seeded = false, FollowingTile next_tile = nullptr,
                                                const uint2* weight_magnitudes = nullptr) {
        static_assert(M % (16 * RowPanels) == 0);
        static_assert((NPanels == 4 || (FP8_FP4 && NPanels == 2)) && KBytes % Chunk == 0);
        constexpr int KGroups = KBytes / Chunk;
        static_assert(!TilePrefetch || (!StageReady && CycleStages && (KGroups % Stages == 0 || Stages == 2)));
        static_assert(StageOffset < Stages);
        auto slot_for = [&](int k) -> Stage& { return stages[(StageOffset + k) % Stages]; };
        const InputTile current{a, sa, b, rows};
        if constexpr (CachedAScales) {
            static_assert(!StageReady && !TilePrefetch);
            auto* scale_cache = reinterpret_cast<uint8_t*>(stages) + ScratchBytes;
            constexpr int ScaleBytes = M * (KBytes / 16);
            static_assert(ScaleBytes % 128 == 0);
            if (threadIdx.x == 0)
                copy_linear_tile<ScaleBytes / 128>(scale_cache, sa);
            sa = reinterpret_cast<const uint16_t*>(scale_cache);
        }
        const int warp = threadIdx.x / 32;
        const int warp_m = warp % MWarps;
        const int warp_n = warp / MWarps;
        const int lane = threadIdx.x % 32;

        uint32_t scale_masks[RowPanels];
        if constexpr (MaskScales) {
            #pragma unroll
            for (int m = 0; m < RowPanels; ++m) {
                const int row = (warp_m * RowPanels + m) * 16 + lane / 4;
                scale_masks[m] = (row < rows ? 0xffffu : 0) | (row + 8 < rows ? 0xffff0000u : 0);
            }
        }
        uint32_t observed = 0;
        auto issue = [&](int k, Stage& stage, const InputTile& input) {
            if (warp == 0 && lane == 0) {
                if constexpr (StageReady) {
                    // Weight requests can proceed while only the activation loader
                    // waits for the corresponding W1 stores to become visible.
                    const uint32_t mask = WholeTileMask ? WholeTileMask : 1u << k;
                    while ((observed & mask) != mask) {
                        asm volatile("ppu.ld.acquire.gpu.b32 %0, [%1];"
                                     : "=r"(observed) : "l"(ready_columns) : "memory");
                    }
                }
                copy_swizzled_rows<M, Chunk>(stage.a, input.activation + k * Chunk, input.rows, AStride);
                if constexpr (FP4 && !CachedAScales) {
                    const int scale_rows = PairedScales ? (input.rows + 15) / 16 * 16 : input.rows;
                    copy_a_scales<M, Slices, ScaleStrideRows>(stage.sa, input.scales + k * Slices * ScaleStrideRows, scale_rows);
                }
            }
            if (warp > 0 && warp <= WeightWarps / 2 && lane == 0) {
                const int pair = warp - 1;
                constexpr int packet_words = 2 * Slices * PackedWeightWords<FP4, FP8_FP4>;
                constexpr int width = PacketWidth ? PacketWidth : (M == 32 && RtWeights ? 64 : 128);
                copy_linear_tile<packet_words * sizeof(uint4) / width, RtWeights, width>(
                    stage.packets + pair * packet_words, input.weights + (pair * KGroups + k) * packet_words);
            }
            asm volatile("ppu.cp.async.commit_group;" ::: "memory");
        };

        Result result{};
        constexpr int Ahead = Stages - 1;
        static_assert(Ahead >= 1 && Ahead <= KGroups);
        if (!TilePrefetch || !seeded) {
            #pragma unroll
            for (int k = 0; k < Ahead; ++k)
                issue(k, slot_for(k), current);
        }
        auto compute = [&](const Stage& stage, int k) {
            #pragma unroll
            for (int slice = 0; slice < Slices; ++slice) {
                uint4 av[RowPanels];
                uint32_t asf[RowPanels]{};
                #pragma unroll
                for (int m = 0; m < RowPanels; ++m) {
                    const int row = (warp_m * RowPanels + m) * 16;
                    av[m] = matrix_fragment<Chunk>(stage.a + row * Chunk + slice * 32);
                    if constexpr (FP4) {
                        if constexpr (CachedAScales) {
                            const auto* scales = reinterpret_cast<const uint32_t*>(sa);
                            const int k_index = k * Slices + slice;
                            asf[m] = scales[((row / 16) * (KBytes / 32) + k_index) * 8 + lane / 4];
                            asf[m] &= scale_masks[m];
                        } else {
                            const auto* scale = stage.sa + slice * M + row;
                            if constexpr (PairedScales) {
                                asf[m] = reinterpret_cast<const uint32_t*>(scale)[lane / 4];
                                if constexpr (MaskScales)
                                    asf[m] &= scale_masks[m];
                            } else {
                                asf[m] = scale[lane / 4] | (uint32_t(scale[lane / 4 + 8]) << 16);
                            }
                        }
                    }
                }
                const int weight_warp = warp_n / (64 / WarpN);
                const auto* packet = stage.packets + ((weight_warp / 2) * Slices / 2 + slice / 2) *
                                     4 * PackedWeightWords<FP4, FP8_FP4>;
                const int weight_step = (weight_warp % 2) * 2 + slice % 2;
                uint4 scales{};
                uint2 routed_scales{};
                if constexpr (FP8_FP4) {
                    const auto* bases = packet + 4 * BWords + weight_step * ScaleWords;
                    routed_scales = reinterpret_cast<const uint2*>(bases)[lane / 4];
                } else if constexpr (FP4) {
                    scales = packet[4 * BWords + weight_step * ScaleWords + lane / 4];
                }
                const uint32_t bsf[] = {scales.x, scales.y, scales.z, scales.w};
                #pragma unroll
                for (int n = 0; n < NPanels; ++n) {
                    int panel = n;
                    if constexpr (WarpN == 32)
                        panel = Gate ? warp_n % 2 + 2 * n : 2 * (warp_n % 2) + n;
                    // The narrower N32 warps already reuse each decoded fragment
                    // across more row panels, so retain arithmetic expansion there.
                    const uint4 bv = load_weight<FP8_FP4 && WarpN == 64>(
                        packet, weight_step, panel, lane, routed_scales, weight_magnitudes);
                    #pragma unroll
                    for (int m = 0; m < RowPanels; ++m)
                        mma<FP4, FP8_FP4>(result.block[m][n], av[m], bv, asf[m], bsf[n]);
                }
            }
        };
        if constexpr (CycleStages) {
            // Index each stage at a fixed offset through the steady-state loop.
            constexpr int Step = Stages * CycleCount;
            constexpr int Steady = (KGroups - Ahead) / Step * Step;
            for (int base = 0; base < Steady; base += Step) {
                #pragma unroll
                for (int slot = 0; slot < Step; ++slot) {
                    asm volatile("ppu.cp.async.wait_group %0;" :: "n"(Ahead - 1) : "memory");
                    __syncthreads();
                    issue(base + slot + Ahead, slot_for(slot + Ahead), current);
                    compute(slot_for(slot), base + slot);
                }
            }
            InputTile following{};
            #pragma unroll
            for (int k = Steady; k < KGroups; ++k) {
                const int slot = k % Stages;
                if constexpr (TilePrefetch) {
                    if (k + Ahead == KGroups)
                        next_tile(true);
                }
                if (k + Ahead <= KGroups || (TilePrefetch && following.activation))
                    asm volatile("ppu.cp.async.wait_group %0;" :: "n"(Ahead - 1) : "memory");
                else
                    asm volatile("ppu.cp.async.wait_all;" ::: "memory");
                __syncthreads();
                if constexpr (TilePrefetch) {
                    if (k + Ahead == KGroups)
                        following = next_tile(false);
                }
                if (k + Ahead < KGroups) {
                    issue(k + Ahead, slot_for(slot + Ahead), current);
                } else if constexpr (TilePrefetch) {
                    if (following.activation)
                        issue(k + Ahead - KGroups, slot_for(slot + Ahead), following);
                }
                compute(slot_for(slot), k);
            }
        } else {
            int slot = 0;
            for (int k = 0; k < KGroups; ++k) {
                if (k + Ahead <= KGroups)
                    asm volatile("ppu.cp.async.wait_group %0;" :: "n"(Ahead - 1) : "memory");
                else
                    asm volatile("ppu.cp.async.wait_all;" ::: "memory");
                // The barrier also releases the stage consumed in the preceding iteration.
                __syncthreads();
                if (k + Ahead < KGroups)
                    issue(k + Ahead, stages[slot == 0 ? Stages - 1 : slot - 1], current);
                compute(slot_for(slot), k);
                slot = slot == Stages - 1 ? 0 : slot + 1;
            }
        }
        return result;
    }
};

}  // namespace gc_moe
