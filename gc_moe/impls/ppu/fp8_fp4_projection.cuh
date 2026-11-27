#pragma once

#include "projection.cuh"

namespace gc_moe {

__device__ __forceinline__ uint8_t* fp8_fp4_mid(const Launch& p, int base) {
    return p.mid + 2ll * base * I;
}

template <typename Project>
__device__ __forceinline__ void fp8_fp4_gate(const Launch& p, const typename Project::Result& acc,
                                          const Tile& tile, int column) {
    constexpr int Panels = Project::MPanels;
    constexpr int GatePanels = Project::NPanels / 2;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row_base = warp % Project::MWarps * Panels * TileM;
    const int channel = (column * Project::NWarps + warp / Project::MWarps) * (GatePanels * 16);
    const auto* scales = reinterpret_cast<const float*>(p.grouped_sf) + tile.row_base;
    auto* out = reinterpret_cast<uint32_t*>(fp8_fp4_mid(p, tile.row_base));
    #pragma unroll
    for (int m = 0; m < Panels; ++m) {
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row = row_base + m * TileM + lane / 4 + r * 8;
            if (row >= tile.rows)
                continue;
            const float scale = scales[row] * p.w1_inv_scale;
            #pragma unroll
            for (int part = 0; part < GatePanels; ++part) {
                #pragma unroll
                for (int pair = 0; pair < 2; ++pair) {
                    const int index = 2 * r + 4 * pair;
                    auto activate = [&](int idx) {
                        const float gate = fminf(rounded(acc.block[m][part].v[idx] * scale), p.clamp);
                        const float up = fmaxf(-p.clamp, fminf(rounded(acc.block[m][part + GatePanels].v[idx] * scale), p.clamp));
                        return evaluate_silu<false>(p.silu_table, gate) * up;
                    };
                    const int col = channel + 16 * part + accumulator_col(lane, index);
                    out[(row * I + col) / 2] = bf16_pair(activate(index), activate(index + 1));
                }
            }
        }
    }
}

template <typename Project>
__device__ __forceinline__ void fp8_fp4_scale_down(const Launch& p, typename Project::Result& acc,
                                                const Tile& tile) {
    constexpr int Panels = Project::MPanels;
    const int lane = threadIdx.x % 32;
    const int warp_m = (threadIdx.x / 32) % Project::MWarps;
    const auto* scales = reinterpret_cast<const float*>(p.mid_sf) + tile.row_base;
    #pragma unroll
    for (int m = 0; m < Panels; ++m) {
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row = (warp_m * Panels + m) * TileM + lane / 4 + r * 8;
            if (row >= tile.rows)
                continue;
            const float scale = scales[row] * p.w2_inv_scale;
            #pragma unroll
            for (int n = 0; n < Project::NPanels; ++n) {
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    acc.block[m][n].v[2 * r + j % 2 + 4 * (j / 2)] *= scale;
            }
        }
    }
}

__device__ __forceinline__ void quantize_fp8_fp4_tile(const Launch& p, const Tile& tile) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const auto* act = reinterpret_cast<const uint32_t*>(fp8_fp4_mid(p, tile.row_base));
    auto* mid = reinterpret_cast<uint16_t*>(fp8_fp4_mid(p, tile.row_base));
    auto* scales = reinterpret_cast<float*>(p.mid_sf) + tile.row_base;
    for (int row = warp; row < tile.rows; row += blockDim.x / 32) {
        constexpr int Values = I / 64;
        uint32_t values[Values];
        float maximum = 1.e-10f;
        #pragma unroll
        for (int pair = 0; pair < Values; ++pair) {
            const uint32_t packed = act[row * (I / 2) + 32 * pair + lane];
            values[pair] = packed;
            const float lo = __uint_as_float(packed << 16);
            const float hi = __uint_as_float(packed & 0xffff0000u);
            maximum = fmaxf(maximum, fmaxf(fabsf(lo), fabsf(hi)));
        }
        for (int mask = 16; mask; mask /= 2)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, mask));
        const float scale = maximum * (1.f / 448.f);
        const float inverse = 1.f / scale;
        if (lane == 0)
            scales[row] = scale;
        // Each warp reuses the first I bytes of its row after loading BF16.
        #pragma unroll
        for (int pair = 0; pair < Values; ++pair) {
            const float lo = __uint_as_float(values[pair] << 16);
            const float hi = __uint_as_float(values[pair] & 0xffff0000u);
            mid[row * I + 32 * pair + lane] = fp8_pair(lo * inverse, hi * inverse);
        }
    }
}

template <int Rows, int NTile, typename Project>
__device__ __forceinline__ void store_down_warp_n32(const Launch& p, const typename Project::Result& acc,
                                                   const Tile& tile, int column, uint8_t* memory) {
    constexpr int Vectors = NTile / 8;
    constexpr int Group = std::min(Vectors, 32);
    static_assert(Project::ScratchBytes >= Rows * NTile * sizeof(bf16));
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp_n = tid / 32;
    // The two N32 warps produce the same N64 return layout used by combine.
    const int vector = (warp_n / 2) * 8 + (lane % 4) * 2 + warp_n % 2;
    auto* stage = reinterpret_cast<uint4*>(memory);
    __syncthreads();
    #pragma unroll
    for (int m = 0; m < Project::MPanels; ++m) {
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row = m * TileM + lane / 4 + r * 8;
            if (row >= tile.rows)
                continue;
            const auto& lo = acc.block[m][0].v;
            const auto& hi = acc.block[m][1].v;
            const int j = 2 * r;
            stage[row * Vectors + vector] = {
                bf16_pair(lo[j], lo[j + 1]), bf16_pair(lo[j + 4], lo[j + 5]),
                bf16_pair(hi[j], hi[j + 1]), bf16_pair(hi[j + 4], hi[j + 5])};
        }
    }
    __syncthreads();
    for (int row = tid / Group; row < tile.rows; row += Project::Threads / Group) {
        const int lane_in_group = lane % Group;
        auto* target = return_row<true>(p, tile.row_base + row) + column * Vectors;
        #pragma unroll
        for (int i = 0; i < Vectors; i += Group)
            __stbl(target + i + lane_in_group, stage[row * Vectors + i + lane_in_group]);
    }
}

template <bool First, int Rows, int NTile, int RowPanels, int Stages, int WarpN>
using FP8_FP4Project = BlockProject<Rows, NTile, First ? H : I, false, 128, Stages, false, RowPanels, true,
                                 false, false, Rows, LongGate && WarpN == 64 ? 256 : 0,
                                 true, WarpN, First, First ? H : 2 * I>;

template <bool First, int Rows, int NTile, int RowPanels, int Stages, int WarpN, int ComputeRows = Rows>
__device__ __forceinline__ void project_fp8_fp4_rows(const Launch& p, const Tile& tile,
                                                   int column, uint8_t* memory,
                                                   const uint2* weight_magnitudes) {
    constexpr int K = First ? H : I;
    constexpr int Columns = (First ? 2 * I : H) / NTile;
    constexpr int Panels = RowPanels * ComputeRows / Rows;
    using Project = FP8_FP4Project<First, ComputeRows, NTile, Panels, Stages, WarpN>;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const auto* a = First ? p.grouped_x + int64_t(tile.row_base) * H : fp8_fp4_mid(p, tile.row_base);
    const int64_t job = int64_t(tile.expert) * Columns + column;
    const auto* b = (First ? p.w1 : p.w2) + job * Project::WeightWarps * (K / 32) * PackedWeightWords<false, true>;
    auto acc = Project::template run<true>(a, nullptr, b, tile.rows,
        reinterpret_cast<typename Project::Stage*>(memory), nullptr, false, nullptr, weight_magnitudes);
    if constexpr (First) {
        fp8_fp4_gate<Project>(p, acc, tile, column);
    } else {
        fp8_fp4_scale_down<Project>(p, acc, tile);
        if constexpr (WarpN == 32)
            store_down_warp_n32<ComputeRows, NTile, Project>(p, acc, tile, column, memory);
        else if constexpr (Project::ScratchBytes >= ComputeRows * NTile * sizeof(bf16))
            routed_staged_store<ComputeRows, NTile, Project, Rows >= 64>(p, acc, tile.rows, tile.row_base, column, memory);
        else
            routed_store<Panels, Rows >= 64>(p, acc.block, tile.rows, tile.row_base,
                column * Project::NWarps + warp / Project::MWarps, lane, (warp % Project::MWarps) * Panels * TileM);
    }
}

template <bool First, int Rows, int NTile, int RowPanels, int Stages, int WarpN>
__device__ __forceinline__ void project_fp8_fp4_tile(const Launch& p, const Tile& tile,
                                                   int column, uint8_t* memory,
                                                   const uint2* weight_magnitudes) {
    constexpr int GroupRows = Rows / RowPanels;
    if constexpr (WarpN == 32) {
        if (tile.rows <= 64) {
            project_fp8_fp4_rows<First, Rows, NTile, RowPanels, Stages, WarpN, 64>(p, tile, column, memory, weight_magnitudes);
            return;
        }
        if constexpr (Rows == 96) {
            if (tile.rows <= 80) {
                project_fp8_fp4_rows<First, Rows, NTile, RowPanels, Stages, WarpN, 80>(p, tile, column, memory, weight_magnitudes);
                return;
            }
        } else if constexpr (Rows == 128) {
            if (tile.rows <= 96) {
                project_fp8_fp4_rows<First, Rows, NTile, RowPanels, Stages, WarpN, 96>(p, tile, column, memory, weight_magnitudes);
                return;
            }
            if (tile.rows <= 112) {
                project_fp8_fp4_rows<First, Rows, NTile, RowPanels, Stages, WarpN, 112>(p, tile, column, memory, weight_magnitudes);
                return;
            }
        }
    } else {
        if constexpr (RowPanels > 1) {
            if (tile.rows <= GroupRows) {
                project_fp8_fp4_rows<First, Rows, NTile, RowPanels, Stages, WarpN, GroupRows>(p, tile, column, memory, weight_magnitudes);
                return;
            }
        }
        if constexpr (RowPanels > 2) {
            if (tile.rows <= 2 * GroupRows) {
                project_fp8_fp4_rows<First, Rows, NTile, RowPanels, Stages, WarpN, 2 * GroupRows>(p, tile, column, memory, weight_magnitudes);
                return;
            }
        }
        if constexpr (RowPanels > 3) {
            if (tile.rows <= 3 * GroupRows) {
                project_fp8_fp4_rows<First, Rows, NTile, RowPanels, Stages, WarpN, 3 * GroupRows>(p, tile, column, memory, weight_magnitudes);
                return;
            }
        }
    }
    project_fp8_fp4_rows<First, Rows, NTile, RowPanels, Stages, WarpN>(p, tile, column, memory, weight_magnitudes);
}

template <int Rows, int NTile, int RowPanels, int Stages, int WarpN = 64>
__global__ void decode_fp8_fp4_pipeline(Launch p) {
    constexpr int GateColumns = 2 * I / NTile;
    constexpr int DownColumns = H / NTile;
    using Gate = FP8_FP4Project<true, Rows, NTile, RowPanels, Stages, WarpN>;
    constexpr int ScaleCount = 27;
    extern __shared__ __align__(128) uint8_t memory[];
    __shared__ __align__(128) uint2 weight_magnitudes[ScaleCount];
    __shared__ int job, down_tile;
    __shared__ bool last;
    const int tid = threadIdx.x;
    // Each rebased UE8M0 exponent selects the eight finite E2M1 magnitudes.
    // The first work-queue barrier publishes the table before projection.
    if constexpr (WarpN == 64)
        if (tid < ScaleCount)
            weight_magnitudes[tid] = Gate::e5m2_magnitudes(uint32_t(tid) * 0x04040404u);
    const int tiles = *p.tile_count;
    auto& control = header(p, p.rank);
    for (;;) {
        if (tid == 0)
            job = atomicAdd(&control.first_head, 1u);
        __syncthreads();
        const int gate = job;
        if (gate >= tiles * GateColumns)
            break;
        const int index = gate / GateColumns;
        const Tile tile = p.tiles[index];
        project_fp8_fp4_tile<true, Rows, NTile, RowPanels, Stages, WarpN>(p, tile, gate % GateColumns, memory, weight_magnitudes);
        __syncthreads();
        if (tid == 0) {
            // The last W1 column acquires the preceding BF16 stores for the page.
            uint32_t prior;
            asm volatile("ppu.atom.add.acq_rel.gpu.u32 %0, [%1], %2;"
                         : "=r"(prior) : "l"(p.finished_first + index), "r"(1u) : "memory");
            last = prior + 1 == GateColumns;
        }
        __syncthreads();
        if (last) {
            quantize_fp8_fp4_tile(p, tile);
            __syncthreads();
            if (tid == 0) {
                const int ready = atomicAdd(&control.down_tail, 1u);
                asm volatile("ppu.st.release.gpu.b32 [%0], %1;"
                             :: "l"(p.down_tiles + ready), "r"(index + 1) : "memory");
            }
        }
        __syncthreads();
    }

    __syncthreads();
    bool wrote_returns = false;
    for (;;) {
        if (tid == 0) {
            job = atomicAdd(&control.down_head, 1u);
            if (job < tiles * DownColumns) {
                // Quantization publishes the page after writing FP8 and its scale.
                int available;
                do {
                    asm volatile("ppu.ld.acquire.gpu.b32 %0, [%1];"
                                 : "=r"(available) : "l"(p.down_tiles + job / DownColumns) : "memory");
                } while (!available);
                down_tile = available - 1;
            }
        }
        __syncthreads();
        const int down = job;
        if (down >= tiles * DownColumns)
            break;
        wrote_returns = true;
        project_fp8_fp4_tile<false, Rows, NTile, RowPanels, Stages, WarpN>(
            p, p.tiles[down_tile], down % DownColumns, memory, weight_magnitudes);
        __syncthreads();
    }
    if (wrote_returns && tid % 32 == 0)
        __ppu_threadfence_system_uncache();
}

template <int Rows, int NTile, int RowPanels, int Stages, int WarpN = 64>
void launch_fp8_fp4_tiles(const Launch& p, hggcStream_t stream, int blocks = 0) {
    using Gate = FP8_FP4Project<true, Rows, NTile, RowPanels, Stages, WarpN>;
    using Down = FP8_FP4Project<false, Rows, NTile, RowPanels, Stages, WarpN>;
    constexpr int Scratch = std::max(Gate::ScratchBytes, Down::ScratchBytes);
    const int blocks_per_cu = blocks ? blocks : (p.bound * TopK <= p.units / 4 ? 2 : 4);
    decode_fp8_fp4_pipeline<Rows, NTile, RowPanels, Stages, WarpN>
        <<<p.units * blocks_per_cu, Gate::Threads, Scratch, stream>>>(p);
}

inline void launch_fp8_fp4(const Launch& p, hggcStream_t stream) {
    static constexpr bool WideN = (2 * I) % 512 == 0;
    // Widen the N tile when the expected number of jobs can cover the device.
    const int routes_per_rank = p.bound * TopK;
    const int active_experts = std::min(Experts / p.ranks, routes_per_rank);
    constexpr int SmallerProjection = std::min(2 * I, H);
    const bool wide_jobs = active_experts * (SmallerProjection / 512) >= p.units;
    switch (p.m_tiles) {
        case 1:
            if constexpr (LongGate) {
                // Long reductions benefit from more smaller CTAs. At a tiny
                // bound with an active SE, four wide CTAs per unit cover both streams.
                const bool compact_routed = p.ranks > 8 && !p.shared_x &&
                                            active_experts * (SmallerProjection / 256) <= 4 * p.units;
                if constexpr (WideN) {
                    const bool small_fused = p.shared_x && routes_per_rank <= p.units / 4;
                    const bool sparse_local = p.ranks <= 8 && !p.shared_x && p.bound > 2;
                    const bool sparse_multinode = p.ranks > 8 && !compact_routed && routes_per_rank > p.units / 4;
                    const bool sparse_batch = (sparse_local || sparse_multinode) && routes_per_rank <= p.units / 2;
                    if (wide_jobs && (small_fused || sparse_batch)) {
                        launch_fp8_fp4_tiles<16, 512, 1, 2>(p, stream, 4);
                        break;
                    }
                }
                const bool compact_grid = routes_per_rank <= p.units / 2 || compact_routed;
                launch_fp8_fp4_tiles<16, 256, 1, 2>(
                    p, stream, compact_grid ? 4 : 6);
            } else {
                if constexpr (WideN) {
                    if (wide_jobs) {
                        launch_fp8_fp4_tiles<16, 512, 1, 2>(p, stream);
                        break;
                    }
                }
                launch_fp8_fp4_tiles<16, 256, 1, 2>(p, stream);
            }
            break;
        case 2:
            if constexpr (WideN)
                launch_fp8_fp4_tiles<32, 512, 2, 2>(p, stream);
            else
                launch_fp8_fp4_tiles<32, 256, 2, 3>(p, stream);
            break;
        case 3:
            if constexpr (WideN)
                launch_fp8_fp4_tiles<48, 512, 3, 2>(p, stream);
            else
                launch_fp8_fp4_tiles<48, 256, 3, 3>(p, stream);
            break;
        case 4:
            if constexpr (WideN)
                launch_fp8_fp4_tiles<64, 512, 4, 2>(p, stream);
            else
                launch_fp8_fp4_tiles<64, 256, 2, 3>(p, stream);
            break;
        case 6: launch_fp8_fp4_tiles<96, 256, 6, 2, 32>(p, stream); break;
        case 8:
            if constexpr (CanUseM128)
                launch_fp8_fp4_tiles<128, 256, 8, 2, 32>(p, stream);
            break;
    }
}

}  // namespace gc_moe
