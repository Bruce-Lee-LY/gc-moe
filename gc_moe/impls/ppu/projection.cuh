#pragma once

#include "schedule.cuh"
#include "shared_expert.cuh"

#include <algorithm>
#include <type_traits>

namespace gc_moe {

template <int MTiles, int BlockRows = 0, bool PairedScales = false, bool ClampInTable = false>
__device__ __forceinline__ void routed_activation(const Launch& p, const Accumulator (&acc)[MTiles][4],
                                                  int rows, int base, int column, int lane, int row_offset = 0,
                                                  const float* cached_weights = nullptr) {
    #pragma unroll
    for (int m = 0; m < MTiles; ++m) {
        const int panel = row_offset + m * TileM;
        if constexpr (BlockRows >= 64) {
            if (panel >= rows)
                continue;
        }
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row = row_offset + m * TileM + lane / 4 + r * 8;
            if constexpr (BlockRows >= 64) {
                if (panel + r * 8 >= rows) {
                    if constexpr (PairedScales) {
                        if (lane % 4 == 0) {
                            const int scale_row = paired_scale_row(row);
                            p.mid_sf[base * (I / 32) + ((column / 2) * BlockRows + scale_row) * 2 + column % 2] = 0;
                        }
                    }
                    continue;
                }
            }
            const float weight = row < rows ? (cached_weights ? cached_weights[row] : p.grouped_weights[base + row]) : 0.f;
            float v[8];
            float maximum = 0.f;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int j = (i & 1) + (i & 2) * 2 + 2 * r;
                const int half = i / 4;
                const float gate = fminf(rounded(acc[m][half].v[j]), p.clamp);
                const float up = fmaxf(-p.clamp, fminf(rounded(acc[m][half + 2].v[j]), p.clamp));
                const float silu = ClampInTable ? p.silu_table[__float_as_uint(gate) >> 16]
                                                : evaluate_silu<false>(p.silu_table, gate);
                v[i] = (silu * up) * weight;
                maximum = fmaxf(maximum, fabsf(v[i]));
            }
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 1, 4));
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 2, 4));
            const int scale_row = PairedScales ? paired_scale_row(row) : row;
            const int scale_index = base * (I / 32) + ((column / 2) * BlockRows + scale_row) * 2 + column % 2;
            if (row >= rows) {
                if constexpr (PairedScales) {
                    if (lane % 4 == 0 && row_offset < rows)
                        p.mid_sf[scale_index] = 0;
                }
                continue;
            }
            const uint32_t bits = __float_as_uint(maximum);
            const int exponent = max(113, min(254, int(bits >> 23) + ((bits & 0x7fffff) != 0) - 2));
            const float inverse = __uint_as_float((254 - exponent) << 23);
            if constexpr (BlockRows) {
                uint8_t* dst = p.mid + (base + row) * (I / 2) + column * 16 + lane % 4;
                #pragma unroll
                for (int pair = 0; pair < 4; ++pair)
                    dst[4 * pair] = fp4_pair(v[2 * pair] * inverse, v[2 * pair + 1] * inverse);
                if (lane % 4 == 0)
                    p.mid_sf[scale_index] = exponent;
            } else {
                uint8_t* dst = p.mid + (base + row) * (I / 2) + (column / 2) * 32 + (column % 2) * 4 + lane % 4;
                #pragma unroll
                for (int pair = 0; pair < 4; ++pair)
                    dst[8 * pair] = fp4_pair(v[2 * pair] * inverse, v[2 * pair + 1] * inverse);
                if (lane % 4 == 0)
                    p.mid_sf[(base + row) * (I / 32) + column] = exponent;
            }
        }
    }
}

template <int MTiles>
__device__ __forceinline__ void routed_fc1(const Launch& p, Tile job, int base, int column, int lane) {
    constexpr int weight_steps = 2 * I / 16 * H / 64;
    Accumulator acc[MTiles][4];
    project<true, H, MTiles>(p.grouped_x + base * (H / 2), p.grouped_sf + base * (H / 32),
                           p.w1 + job.expert * weight_steps * (PackedWeightWords<true> / 4),
                           job.rows, column, lane, acc);
    routed_activation(p, acc, job.rows, base, column, lane);
}

template <int MTiles, bool Packed = false>
__device__ __forceinline__ void routed_store(const Launch& p, const Accumulator (&acc)[MTiles][4],
                                             int rows, int base, int column, int lane, int row_offset = 0) {
    #pragma unroll
    for (int m = 0; m < MTiles; ++m) {
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row = row_offset + m * TileM + lane / 4 + r * 8;
            if (row >= rows)
                continue;
            auto* vector = return_row<Packed>(p, base + row) + column * 8 + (lane % 4) * 2;
            const auto packed = pack_return_row(acc[m], r);
            __stbl(vector, packed.first);
            __stbl(vector + 1, packed.second);
        }
    }
}

template <int MTiles>
__device__ __forceinline__ void routed_fc2(const Launch& p, Tile job, int base, int column, int lane) {
    constexpr int weight_steps = H / 16 * I / 64;
    Accumulator acc[MTiles][4];
    project<true, I, MTiles>(p.mid + base * (I / 2), p.mid_sf + base * (I / 32),
                        p.w2 + job.expert * weight_steps * (PackedWeightWords<true> / 4),
                        job.rows, column, lane, acc);
    routed_store<MTiles, MTiles == 4>(p, acc, job.rows, base, column, lane);
}

enum class ProjectionRole { Mixed, Routed, Shared };

template <bool First, int MTiles, ProjectionRole Role>
__global__ void decode_experts(Launch p) {
    constexpr int Columns = First ? I / 32 : H / 64;
    const int lane = threadIdx.x % 32;
    const int local_warp = threadIdx.x / 32;
    const int first_job = blockIdx.x * Warps + local_warp;
    const int stride = gridDim.x * Warps;
    constexpr int tile_rows = MTiles * TileM;
    const int shared_jobs = Role != ProjectionRole::Routed && p.shared_x ? (p.tokens + tile_rows - 1) / tile_rows * Columns : 0;
    const int total = shared_jobs + (Role == ProjectionRole::Shared ? 0 : *p.tile_count * Columns);
    for (int job = first_job; job < total; job += stride) {
        const bool shared = Role == ProjectionRole::Shared || job < shared_jobs;
        const int local = shared ? job : job - shared_jobs;
        const int tile = local / Columns;
        const int base = tile * tile_rows;
        const int column = local % Columns;
        if (shared) {
            if constexpr (First)
                shared_fc1<MTiles>(p, base, column, lane);
            else
                shared_fc2<MTiles>(p, base, column, lane);
        } else {
            const Tile expert = p.tiles[tile];
            if constexpr (First)
                routed_fc1<MTiles>(p, expert, base, column, lane);
            else
                routed_fc2<MTiles>(p, expert, base, column, lane);
        }
    }
    if constexpr (!First && Role != ProjectionRole::Shared) {
        __syncwarp();
        if (lane == 0)
            __ppu_threadfence_system_uncache();
    }
}

template <int Rows, int NTile, typename Project, bool Packed>
__device__ __forceinline__ void routed_staged_store(const Launch& p, const typename Project::Result& acc,
                                                    int live_rows, int base, int column, uint8_t* memory) {
    constexpr int Vectors = NTile / 8;
    constexpr int StoreGroup = std::min(Vectors, 32);
    static_assert(Vectors % StoreGroup == 0);
    static_assert(Project::ScratchBytes >= Rows * NTile * sizeof(bf16));
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp % Project::MWarps;
    const int warp_n = warp / Project::MWarps;
    auto* staging = reinterpret_cast<uint4*>(memory);
    // The staging area overlays the AIU buffers after their last consumer.
    __syncthreads();
    #pragma unroll
    for (int m = 0; m < Project::MPanels; ++m) {
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row = (warp_m * Project::MPanels + m) * TileM + lane / 4 + r * 8;
            if (row >= live_rows)
                continue;
            const auto packed = pack_return_row(acc.block[m], r);
            auto* lane_values = staging + row * Vectors + warp_n * 8 + (lane % 4) * 2;
            lane_values[0] = packed.first;
            lane_values[1] = packed.second;
        }
    }
    __syncthreads();
    // Each group writes a contiguous column tile to one destination row.
    const int col = lane % StoreGroup;
    for (int row = tid / StoreGroup; row < live_rows; row += Project::Threads / StoreGroup) {
        auto* target = return_row<Packed>(p, base + row) + column * Vectors;
        #pragma unroll
        for (int i = 0; i < Vectors; i += StoreGroup)
            __stbl(target + col + i, staging[row * Vectors + col + i]);
    }
}

template <bool First, int Rows, int NTile, int Chunk, int StageCount = 3,
          bool PairedScales = false, bool StagedScatter = false, int RowPanels = 1,
          bool ClampInTable = false, bool PageLayout = false, bool SplitColumns = false, bool RtWeights = false, int ComputeRows = Rows, bool StageReady = false, int PacketWidth = 0, int CycleCount = 1, bool TilePrefetch = false>
__device__ __forceinline__ void run_routed_tile_compute(const Launch& p, int tile, int column, uint8_t* memory,
                                                        int* next_job = nullptr, int total = 0, bool seeded = false,
                                                        int* next_down_tile = nullptr, bool swap_slots = false) {
    constexpr int K = First ? H / 2 : I / 2;
    constexpr int Columns = (First ? 2 * I : H) / NTile;
    constexpr int Panels = ComputeRows * RowPanels / Rows;
    // Wider expert pages keep W1 scale rows in order, so AIU can zero-pad
    // exactly the missing rows. W1 still produces the paired W2 layout.
    constexpr bool CompactW1Scales = PageLayout && Rows >= 64 && First;
    using Project = BlockProject<ComputeRows, NTile, K, true, Chunk, StageCount, PairedScales && !CompactW1Scales, Panels,
                                 PairedScales || ClampInTable, PageLayout && First && !CompactW1Scales, SplitColumns, Rows, PacketWidth>;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int warp_m = warp % Project::MWarps;
    const int warp_n = warp / Project::MWarps;
    __shared__ Tile cached_tile;
    const Tile expert = TilePrefetch && seeded ? cached_tile : p.tiles[tile];
    const int base = PageLayout ? expert.row_base : tile * Rows;
    const auto* a = (First ? p.grouped_x : p.mid) + base * K;
    const auto* sa = reinterpret_cast<const uint16_t*>(First ? p.grouped_sf : p.mid_sf) + base * (K / 32);
    const int task = (expert.expert * Columns + column) * Project::NWarps;
    const auto* b = (First ? p.w1 : p.w2) + task * (K / 32) * PackedWeightWords<true>;
    const float* cached_weights = nullptr;
    if constexpr (First && TilePrefetch) {
        auto* weights = reinterpret_cast<float*>(memory + Project::ScratchBytes);
        if (threadIdx.x < expert.rows)
            weights[threadIdx.x] = p.grouped_weights[base + threadIdx.x];
        cached_weights = weights;
    }
    const auto acc = [&] {
        if constexpr (SplitColumns)
            return Project::run_warp_groups(a, sa, b, expert.rows, *reinterpret_cast<typename Project::GroupStorage*>(memory));
        else if constexpr (TilePrefetch) {
            static_assert(PageLayout && !StageReady);
            auto prepare_next = [&](bool claim) -> typename Project::InputTile {
                if (claim) {
                    if (threadIdx.x == 0) {
                        auto& schedule = header(p, p.rank);
                        if constexpr (First) {
                            *next_job = atomicAdd(&schedule.first_head, 1u);
                        } else {
                            const int next = atomicAdd(&schedule.down_head, 1u);
                            *next_job = next;
                            int available = 0;
                            if (next < total) {
                                asm volatile("ppu.ld.acquire.gpu.b32 %0, [%1];"
                                             : "=r"(available) : "l"(p.down_tiles + next / Columns) : "memory");
                            }
                            *next_down_tile = available - 1;
                        }
                        const int next = *next_job;
                        if (next < total && (First || *next_down_tile >= 0))
                            cached_tile = p.tiles[First ? next / Columns : *next_down_tile];
                    }
                    return {};
                }
                const int next = *next_job;
                if (next >= total || (!First && *next_down_tile < 0))
                    return {};
                // The existing stage barrier publishes the next descriptor.
                const Tile next_tile = cached_tile;
                const int next_task = (next_tile.expert * Columns + next % Columns) * Project::NWarps;
                const int row_base = next_tile.row_base;
                return {(First ? p.grouped_x : p.mid) + row_base * K,
                        reinterpret_cast<const uint16_t*>(First ? p.grouped_sf : p.mid_sf) + row_base * (K / 32),
                        (First ? p.w1 : p.w2) + next_task * (K / 32) * PackedWeightWords<true>, next_tile.rows};
            };
            if constexpr (!First && StageCount == 2 && (K / Chunk) % 2) {
                if (swap_slots)
                    return Project::template run<RtWeights, false, CycleCount, true, 1>(
                        a, sa, b, expert.rows, reinterpret_cast<typename Project::Stage*>(memory), nullptr, seeded, prepare_next);
            }
            return Project::template run<RtWeights, false, CycleCount, true>(
                a, sa, b, expert.rows, reinterpret_cast<typename Project::Stage*>(memory), nullptr, seeded, prepare_next);
        } else {
            constexpr uint32_t required_columns = [] {
                if constexpr (StageReady && Rows == 64)
                    return (1u << (2 * I / NTile)) - 1;
                return 0u;
            }();
            return Project::template run<RtWeights, StageReady, CycleCount, false, 0, required_columns>(
                a, sa, b, expert.rows, reinterpret_cast<typename Project::Stage*>(memory),
                StageReady ? p.finished_first + tile : nullptr);
        }
    }();
    const int logical_column = column * Project::NWarps + warp_n;
    const int row_offset = warp_m * TileM * Panels;
    if constexpr (First)
        routed_activation<Panels, Rows, PairedScales, ClampInTable>(p, acc.block, expert.rows, base, logical_column, lane, row_offset, cached_weights);
    else if constexpr (StagedScatter && Project::ScratchBytes >= ComputeRows * NTile * sizeof(bf16))
        routed_staged_store<ComputeRows, NTile, Project, Rows >= 64>(p, acc, expert.rows, base, column, memory);
    else
        routed_store<Panels, Rows >= 64>(p, acc.block, expert.rows, base, logical_column, lane, row_offset);
}

template <bool First, int Rows, int NTile, int Chunk, int StageCount = 3,
          bool PairedScales = false, bool StagedScatter = false, int RowPanels = 1,
          bool ClampInTable = false, bool PageLayout = false, bool SplitColumns = false, bool RtWeights = false, int NarrowMode = 0, bool StageReady = false, int PacketWidth = 0, int W1NarrowStages = StageCount, int CycleCount = 1, bool TilePrefetch = false>
__device__ __forceinline__ void run_routed_tile(const Launch& p, int tile, int column, uint8_t* memory,
                                                int* next_job = nullptr, int total = 0, bool seeded = false,
                                                int* next_down_tile = nullptr, bool swap_slots = false) {
    constexpr bool Narrow = NarrowMode & (First ? 1 : 2);
    if constexpr ((Rows >= 96 && Narrow) || (Rows == 64 && (Narrow || (First && SplitColumns)))) {
        const int rows = p.tiles[tile].rows;
        if (rows <= 32) {
            run_routed_tile_compute<First, Rows, NTile, Chunk, First ? W1NarrowStages : StageCount,
                                    PairedScales, StagedScatter, RowPanels, ClampInTable, PageLayout,
                                    SplitColumns, RtWeights, 32, StageReady, PacketWidth, CycleCount>(
                p, tile, column, memory);
            return;
        }
        if constexpr (Rows >= 96) {
            if (rows <= 64) {
                run_routed_tile_compute<First, Rows, NTile, Chunk, StageCount, PairedScales, StagedScatter,
                                        RowPanels, ClampInTable, PageLayout, SplitColumns, RtWeights,
                                        64, StageReady, PacketWidth, CycleCount>(p, tile, column, memory);
                return;
            }
        }
        if constexpr (Rows == 128) {
            if (rows <= 96) {
                run_routed_tile_compute<First, Rows, NTile, Chunk, StageCount, PairedScales, StagedScatter,
                                        RowPanels, ClampInTable, PageLayout, SplitColumns, RtWeights,
                                        96, StageReady, PacketWidth, CycleCount>(p, tile, column, memory);
                return;
            }
        }
    }
    run_routed_tile_compute<First, Rows, NTile, Chunk, StageCount, PairedScales, StagedScatter,
                            RowPanels, ClampInTable, PageLayout, SplitColumns, RtWeights, Rows, StageReady, PacketWidth, CycleCount, TilePrefetch>(
        p, tile, column, memory, next_job, total, seeded, next_down_tile, swap_slots);
}

template <bool First, int MTiles, ProjectionRole Role, int NTile, int Chunk, int StageCount,
          bool RtWeights = false, bool InterleaveM = false>
__global__ void decode_block_experts(Launch p) {
    constexpr int Rows = MTiles * TileM;
    constexpr int N = First ? 2 * I : H;
    constexpr int K = First ? H / 2 : I / 2;
    constexpr int Columns = N / NTile;
    using Project = BlockProject<Rows, NTile, K, true, Chunk>;
    using SharedProject = BlockProject<Rows, NTile, K * 2, false, Chunk>;
    extern __shared__ __align__(128) uint8_t memory[];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warp_m = warp % Project::MWarps;
    const int warp_n = warp / Project::MWarps;
    const int shared_rows = (p.tokens + Rows - 1) / Rows;
    const int shared_blocks = Role != ProjectionRole::Routed && p.shared_x ? shared_rows * Columns : 0;
    const int blocks = shared_blocks + (Role == ProjectionRole::Shared ? 0 : *p.tile_count * Columns);
    for (int job = blockIdx.x; job < blocks; job += gridDim.x) {
        if (Role == ProjectionRole::Shared || job < shared_blocks) {
            const int column = InterleaveM ? job / shared_rows : job % Columns;
            const int base = (InterleaveM ? job % shared_rows : job / Columns) * Rows;
            const int rows = min(Rows, p.tokens - base);
            const auto* a = (First ? p.shared_x : p.shared_mid) + base * (K * 2);
            const int task = column * Project::NWarps;
            const auto* b = (First ? p.shared_w1 : p.shared_w2) + task * (K * 2 / 32) * Project::BWords;
            const auto acc = SharedProject::run(a, nullptr, b, rows,
                                                reinterpret_cast<typename SharedProject::Stage*>(memory));
            const int logical_column = column * Project::NWarps + warp_n;
            const int row_offset = warp_m * TileM;
            if constexpr (First) {
                shared_activation(p, acc.block, rows, base, logical_column, lane, row_offset);
            } else {
                shared_store<1>(p, acc.block, rows, base, logical_column, lane, row_offset);
            }
        } else {
            const int index = job - shared_blocks;
            run_routed_tile<First, Rows, NTile, Chunk, StageCount, false, false, 1, false, false, false, RtWeights>(
                p, index / Columns, index % Columns, memory);
        }
        __syncthreads();
    }
    if constexpr (!First && Role != ProjectionRole::Shared) {
        __syncwarp();
        if (lane == 0)
            __ppu_threadfence_system_uncache();
    }
}

// Let a quarter of the CTAs take a completed down tile while other CTAs
// are claiming W1. A CTA starts waiting for W1 only after the W1 queue is
// fully claimed, so that work owned by another CTA can always make progress.
__global__ void decode_fp4_fp4_pipeline_m16_interleaved(Launch p) {
    constexpr int GateColumns = 2 * I / 128;
    constexpr int DownColumns = H / 128;
    extern __shared__ __align__(128) uint8_t memory[];
    __shared__ int job, down_tile;
    __shared__ bool down;
    const int tiles = *p.tile_count;
    const int gate_jobs = tiles * GateColumns, down_jobs = tiles * DownColumns;
    auto& schedule = header(p, p.rank);
    bool gate_claims_done = false;
    const int tid = threadIdx.x;
    for (;;) {
        if (tid == 0) {
            down = false;
            if (!gate_claims_done && !(blockIdx.x & 3)) {
                const unsigned candidate = __ldbl(&schedule.down_head);
                if (candidate < unsigned(down_jobs)) {
                    int tile;
                    asm volatile("ppu.ld.acquire.gpu.b32 %0, [%1];"
                                 : "=r"(tile) : "l"(p.down_tiles + candidate / DownColumns) : "memory");
                    if (tile && atomicCAS(&schedule.down_head, candidate, candidate + 1) == candidate) {
                        job = candidate;
                        down_tile = tile - 1;
                        down = true;
                    }
                }
            }
            if (!down && !gate_claims_done) {
                job = atomicAdd(&schedule.first_head, 1u);
                gate_claims_done = job >= gate_jobs;
            }
            if (!down && gate_claims_done) {
                job = atomicAdd(&schedule.down_head, 1u);
                if (job < down_jobs) {
                    int tile;
                    do {
                        asm volatile("ppu.ld.acquire.gpu.b32 %0, [%1];"
                                     : "=r"(tile) : "l"(p.down_tiles + job / DownColumns) : "memory");
                    } while (!tile);
                    down_tile = tile - 1;
                }
                down = true;
            }
        }
        __syncthreads();
        const int current = job;
        if (down) {
            if (current >= down_jobs)
                break;
            run_routed_tile<false, 16, 128, FullDownChunk, 2, true, false, 1, true, true, false, true>(
                p, down_tile, current % DownColumns, memory);
            __syncthreads();
        } else {
            const int tile = current / GateColumns;
            run_routed_tile<true, 16, 128, 64, 5, true, false, 1, true, true, false, true>(
                p, tile, current % GateColumns, memory);
            __syncthreads();
            if (tid == 0) {
                uint32_t finished;
                asm volatile("ppu.atom.add.acq_rel.gpu.u32 %0, [%1], %2;"
                             : "=r"(finished) : "l"(p.finished_first + tile), "r"(1u) : "memory");
                if (finished == GateColumns - 1) {
                    const int ready = atomicAdd(&schedule.down_tail, 1u);
                    asm volatile("ppu.st.release.gpu.b32 [%0], %1;"
                                 :: "l"(p.down_tiles + ready), "r"(tile + 1) : "memory");
                }
            }
        }
    }
    __syncwarp();
    if (tid % 32 == 0)
        __ppu_threadfence_system_uncache();
}

void launch_fp4_fp4_interleaved_m16(const Launch& p, hggcStream_t stream) {
    using Gate = BlockProject<16, 128, H / 2, true, 64, 5, true>;
    using Down = BlockProject<16, 128, I / 2, true, FullDownChunk, 2, true>;
    constexpr int Scratch = std::max(Gate::ScratchBytes, Down::ScratchBytes);
    decode_fp4_fp4_pipeline_m16_interleaved<<<p.units * 8, Gate::Threads, Scratch, stream>>>(p);
}

template <int Rows, int NTile, int Chunk, int StageCount, bool PairedScales, bool StagedScatter,
          int RowPanels, bool ClampInTable, bool PageLayout, bool SplitColumns, bool RtWeights, int DownStages, int DownChunk, int NarrowMode, bool StageReady, int W1NarrowStages, int DownCycles, bool PrefetchTiles, int GateCycles>
__global__ void decode_fp4_fp4_pipeline(Launch p) {
    constexpr int GateColumns = 2 * I / NTile;
    constexpr int DownColumns = H / NTile;
    // A W1 column tile produces half as many SwiGLU channels as its N width.
    static_assert(!StageReady || (2 * DownChunk == NTile / 2 && GateColumns <= 32 && !SplitColumns));
    // Short wide-page projections use smaller weight requests to balance
    // the activation and weight loaders.
    constexpr int PacketWidth = Rows == 64 && PageLayout && !SplitColumns && ClampInTable && ShortDown ? 64 : 0;
    extern __shared__ __align__(128) uint8_t memory[];
    __shared__ int job, down_tile, down_job;
    const int tid = threadIdx.x;
    const int tiles = *p.tile_count;
    auto& schedule = header(p, p.rank);
    constexpr bool TilePrefetch = PrefetchTiles && ClampInTable;
    static_assert(!PrefetchTiles || (Rows == 16 && NTile == 128 && PageLayout &&
                                     !StageReady && !StagedScatter && !SplitColumns && DownStages == 2));
    const int gate_jobs = tiles * GateColumns;
    bool seeded = false;
    if constexpr (TilePrefetch) {
        if (tid == 0)
            job = atomicAdd(&schedule.first_head, 1u);
        __syncthreads();
    }
    for (;;) {
        if constexpr (!TilePrefetch) {
            if (tid == 0)
                job = atomicAdd(&schedule.first_head, 1u);
            __syncthreads();
        }
        const int first = job;
        if (first >= gate_jobs)
            break;
        const int tile = first / GateColumns;
        run_routed_tile<true, Rows, NTile, Chunk, StageCount, PairedScales, false, RowPanels, ClampInTable, PageLayout, SplitColumns, RtWeights, NarrowMode, false, PacketWidth, W1NarrowStages, GateCycles, TilePrefetch>(
            p, tile, first % GateColumns, memory, &job, gate_jobs, seeded);
        __syncthreads();
        if constexpr (TilePrefetch)
            seeded = true;
        if (tid == 0) {
            uint32_t finished;
            // Publish this W1 column's stores. Each column contributes one bit
            // to the mask, or one count when W2 waits for the entire M tile.
            const uint32_t increment = StageReady ? (1u << (first % GateColumns)) : 1u;
            asm volatile("ppu.atom.add.acq_rel.gpu.u32 %0, [%1], %2;"
                         : "=r"(finished) : "l"(p.finished_first + tile), "r"(increment) : "memory");
            if (!StageReady && finished == GateColumns - 1) {
                const int ready = atomicAdd(&schedule.down_tail, 1u);
                asm volatile("ppu.st.release.gpu.b32 [%0], %1;"
                             :: "l"(p.down_tiles + ready), "r"(tile + 1) : "memory");
            }
        }
    }

    // Every unfinished W1 job is already owned by a resident CTA. W2 can
    // safely wait on individual K chunks without blocking unclaimed W1 jobs.
    if constexpr (!TilePrefetch)
        __syncthreads();
    if constexpr (TilePrefetch) {
        const int down_jobs = tiles * DownColumns;
        bool seeded_down = false;
        bool swap_slots = false;
        int* ticket = &down_job;
        const auto wait_for_row_tile = [&](int current) {
            int available;
            do {
                asm volatile("ppu.ld.acquire.gpu.b32 %0, [%1];"
                             : "=r"(available) : "l"(p.down_tiles + current / DownColumns) : "memory");
            } while (!available);
            down_tile = available - 1;
        };
        if (tid == 0) {
            *ticket = atomicAdd(&schedule.down_head, 1u);
            down_tile = -1;
            if (*ticket < down_jobs)
                wait_for_row_tile(*ticket);
        }
        __syncthreads();
        bool first_down = true;
        while (*ticket < down_jobs) {
            if (!seeded_down && !first_down) {
                if (tid == 0)
                    wait_for_row_tile(*ticket);
                __syncthreads();
            }
            run_routed_tile<false, Rows, NTile, DownChunk, DownStages, PairedScales, StagedScatter, RowPanels,
                            ClampInTable, PageLayout, SplitColumns, RtWeights, NarrowMode, false, PacketWidth,
                            DownStages, DownCycles, true>(
                p, down_tile, *ticket % DownColumns, memory, ticket, down_jobs, seeded_down, &down_tile, swap_slots);
            // For a seeded tile, the next first-stage barrier also retires
            // the previous epilogue. An unseeded tile needs a separate barrier
            // before its prologue can write the stage ring.
            seeded_down = down_tile >= 0;
            if (!seeded_down)
                __syncthreads();
            swap_slots = !swap_slots;
            first_down = false;
        }
    } else for (;;) {
        if (tid == 0) {
            job = atomicAdd(&schedule.down_head, 1u);
            if (!StageReady && job < tiles * DownColumns) {
                int tile;
                do {
                    asm volatile("ppu.ld.acquire.gpu.b32 %0, [%1];"
                                 : "=r"(tile) : "l"(p.down_tiles + job / DownColumns) : "memory");
                } while (!tile);
                down_tile = tile - 1;
            }
        }
        __syncthreads();
        const int down = job;
        if (down >= tiles * DownColumns)
            break;
        run_routed_tile<false, Rows, NTile, DownChunk, DownStages, PairedScales, StagedScatter, RowPanels, ClampInTable, PageLayout, SplitColumns, RtWeights, NarrowMode, StageReady, PacketWidth, DownStages, DownCycles>(
            p, StageReady ? down / DownColumns : down_tile, down % DownColumns, memory);
        __syncthreads();
    }
    __syncwarp();
    if (tid % 32 == 0)
        __ppu_threadfence_system_uncache();
}

template <bool First, ProjectionRole Role>
void launch_experts(const Launch& p, hggcStream_t stream) {
    const int blocks = p.units * 4;
    if (p.m_tiles == 3)
        decode_experts<First, 3, Role><<<blocks, Threads, 0, stream>>>(p);
    else if (p.m_tiles == 2)
        decode_experts<First, 2, Role><<<blocks, Threads, 0, stream>>>(p);
    else
        decode_experts<First, 1, Role><<<blocks, Threads, 0, stream>>>(p);
}

template <bool First, int MTiles, ProjectionRole Role, int NTile = BlockN,
          int Chunk = (MTiles == 1 ? FullDownChunk : 64)>
void launch_block_experts(const Launch& p, hggcStream_t stream) {
    constexpr int StageCount = MTiles >= 3 && Role == ProjectionRole::Routed && Chunk == 128 ? 2 : 3;
    constexpr int K = First ? H : I;
    using Project = BlockProject<MTiles * TileM, NTile, K / 2, true, Chunk, StageCount>;
    using SharedProject = BlockProject<MTiles * TileM, NTile, K, false, Chunk, StageCount>;
    constexpr int staging = Role == ProjectionRole::Routed ? Project::ScratchBytes
                            : std::max(Project::ScratchBytes, SharedProject::ScratchBytes);
    const bool sparse = Role != ProjectionRole::Shared && MTiles == 1 && p.bound * p.ranks * TopK <= 4 * Experts;
    // Extra CTAs let the device scheduler distribute individual M48 and M64 jobs.
    const bool large_ep8 = Role == ProjectionRole::Routed && MTiles >= 3 && Chunk == 128 && p.ranks == 8;
    const int blocks_per_unit = large_ep8 ? 56 : (sparse ? 8 : (MTiles == 3 ? 5 : 4));
    const int blocks = p.units * blocks_per_unit;
    if constexpr (Role == ProjectionRole::Shared) {
        if (p.ranks == MaxRanks) {
            // Adjacent M tiles request the same column of shared weights.
            constexpr int Rows = MTiles * TileM;
            constexpr int Columns = (First ? 2 * I : H) / NTile;
            const int jobs = (p.tokens + Rows - 1) / Rows * Columns;
            decode_block_experts<First, MTiles, Role, NTile, Chunk, StageCount, false, true>
                <<<jobs, Project::Threads, staging, stream>>>(p);
            return;
        }
    }
    if constexpr (!LongGate && Role == ProjectionRole::Routed) {
        if (!p.shared_x && p.ranks == MaxRanks) {
            decode_block_experts<First, MTiles, Role, NTile, Chunk, StageCount, true>
                <<<blocks, Project::Threads, staging, stream>>>(p);
            return;
        }
    }
    decode_block_experts<First, MTiles, Role, NTile, Chunk, StageCount><<<blocks, Project::Threads, staging, stream>>>(p);
}

template <bool First, ProjectionRole Role = ProjectionRole::Mixed>
void launch_projection(const Launch& p, hggcStream_t stream) {
    if (!p.aiu)
        launch_experts<First, Role>(p, stream);
    else if (p.m_tiles == 4) {
        launch_block_experts<First, 4, Role, 128, FullDownChunk>(p, stream);
    } else if (p.m_tiles == 3) {
        if (Role == ProjectionRole::Routed && (!LongGate || p.bound * p.ranks * TopK >= 29 * Experts))
            launch_block_experts<First, 3, Role, 128, FullDownChunk>(p, stream);
        else
            launch_block_experts<First, 3, Role>(p, stream);
    } else if (p.m_tiles == 2) {
        launch_block_experts<First, 2, Role, 256, 64>(p, stream);
    } else if (p.bound * p.ranks * TopK >= 8 * Experts) {
        launch_block_experts<First, 1, Role, 256, 64>(p, stream);
    } else if constexpr (Role == ProjectionRole::Shared) {
        launch_block_experts<First, 1, Role>(p, stream);
    } else {
        launch_block_experts<First, 1, Role, 128, 64>(p, stream);
    }
}

template <bool First>
void launch_shared_projection(const Launch& p, hggcStream_t stream) {
    if (p.ranks == MaxRanks && p.tokens > 128)
        launch_shared_tiles<First, 128, 3, 2>(p, stream);
    else if (!BalancedProjections && p.ranks == MaxRanks && p.tokens > 16 && p.tokens <= 32)
        launch_shared_tiles<First, 16, 4, 1>(p, stream);
    else if (p.ranks == MaxRanks && p.tokens > 32 &&
             (p.tokens > 64 || !BalancedProjections))
        launch_shared_tiles<First, 64, 3, 2>(p, stream);
    else if (p.tokens > 32)
        launch_block_experts<First, 4, ProjectionRole::Shared, 128, 64>(p, stream);
    else
        launch_projection<First, ProjectionRole::Shared>(p, stream);
}

template <int Rows, int NTile, int Chunk = 64, int StageCount = 3,
          bool PairedScales = false, bool StagedScatter = false, int RowPanels = 1,
          bool PageLayout = false, bool SplitColumns = false, int DownStages = StageCount, int DownChunk = Chunk, int NarrowMode = 0, bool StageReady = false, int W1NarrowStages = StageCount, int DownCycles = 1, bool PrefetchTiles = false, int GateCycles = 1>
void launch_fp4_fp4_pipeline(const Launch& p, hggcStream_t stream, bool clamp_in_table,
                            int blocks_per_unit_override = 0) {
    using Project = BlockProject<Rows, NTile, H / 2, true, Chunk, StageCount, PairedScales, RowPanels,
                                 false, false, SplitColumns>;
    using DownProject = BlockProject<Rows, NTile, I / 2, true, DownChunk, DownStages, PairedScales, RowPanels,
                                     false, false, SplitColumns>;
    using NarrowProject = BlockProject<32, NTile, H / 2, true, Chunk, W1NarrowStages, false>;
    constexpr int gate_bytes = PrefetchTiles ? Rows * sizeof(float) : 0;
    constexpr int staging = std::max({Project::ScratchBytes + gate_bytes, DownProject::ScratchBytes,
                                      W1NarrowStages != StageCount ? NarrowProject::ScratchBytes : 0});
    constexpr int blocks_per_unit = Rows > 32 ? 16 : (Rows == TileM && NTile == 128 ? 8 : 4);
    const bool share_wide_grid = BalancedProjections && Rows == 64 && p.shared_x &&
                                 p.ranks == MaxRanks && p.bound * p.ranks * TopK >= 28 * Experts;
    const int ctas_per_unit = blocks_per_unit_override ? blocks_per_unit_override
                                                      : share_wide_grid ? 8 : blocks_per_unit;
    const int blocks = p.units * ctas_per_unit;
    auto launch = [&](auto rt_weights) {
        constexpr bool RtWeights = decltype(rt_weights)::value;
        if (clamp_in_table) {
            decode_fp4_fp4_pipeline<Rows, NTile, Chunk, StageCount, PairedScales, StagedScatter,
                                   RowPanels, true, PageLayout, SplitColumns, RtWeights, DownStages, DownChunk, NarrowMode, StageReady, W1NarrowStages, DownCycles, PrefetchTiles, GateCycles>
                <<<blocks, Project::Threads, staging, stream>>>(p);
            return;
        }
        decode_fp4_fp4_pipeline<Rows, NTile, Chunk, StageCount, PairedScales, StagedScatter,
                               RowPanels, false, PageLayout, SplitColumns, RtWeights, DownStages, DownChunk, NarrowMode, false, W1NarrowStages, DownCycles, PrefetchTiles, GateCycles>
            <<<blocks, Project::Threads, staging, stream>>>(p);
    };
    if constexpr (!SplitColumns) {
        if ((!LongGate && !p.shared_x && p.ranks == MaxRanks) || clamp_in_table) {
            launch(std::true_type{});
            return;
        }
    }
    launch(std::false_type{});
}

inline void launch_fp4_fp4(const Launch& p, const DecodeSchedule& schedule, hggcStream_t stream) {
    if (schedule.pipeline) {
        if (schedule.small_pages) {
            const bool prefetch = schedule.clamp_in_table && (schedule.fused_dispatch || (LongGate && schedule.prefetch_returns));
            if (prefetch && (p.bound >= 2 || LongGate)) {
                if (GateStages > 4 && schedule.prefetch_returns && schedule.fused_dispatch) {
                    launch_fp4_fp4_pipeline<16, 128, 64, 4, true, false, 1, true,
                                           false, 2, FullDownChunk, 0, false, 4, 1, true>(p, stream, schedule.clamp_in_table);
                } else {
                    launch_fp4_fp4_pipeline<16, 128, 64, GateStages, true, false, 1, true,
                                           false, 2, FullDownChunk, 0, false, GateStages, 1, true>(p, stream, schedule.clamp_in_table);
                }
            } else if (LongGate && schedule.clamp_in_table && p.bound == 2) {
                launch_fp4_fp4_interleaved_m16(p, stream);
            } else if (LongGate && p.bound == 1) {
                launch_fp4_fp4_pipeline<16, 128, 128, 3, true, false, 1, true,
                                       false, 2, FullDownChunk>(p, stream, schedule.clamp_in_table);
            } else if (LongGate) {
                launch_fp4_fp4_pipeline<16, 128, 64, 5, true, false, 1, true,
                                       false, 2, FullDownChunk>(p, stream, schedule.clamp_in_table);
            } else {
                launch_fp4_fp4_pipeline<16, 128, 64, 5, true, false, 1, true,
                                       false, 4, 64>(p, stream, schedule.clamp_in_table);
            }
        } else if (schedule.tall_pages) {
            if constexpr (CanUseM128) {
                if (p.m_tiles == 8) {
                    launch_fp4_fp4_pipeline<128, 256, 64, 3, true, true, 4, true, false,
                                           3, 64, 3, false, 3, 2>(p, stream, schedule.clamp_in_table, 8);
                    return;
                }
            }
            launch_fp4_fp4_pipeline<96, 256, 64, 3, true, true, 3, true, false,
                                   3, 64, 3, false, 3, 2>(p, stream, schedule.clamp_in_table);
        } else if (schedule.wide_pages) {
            const bool compact = !LongGate && schedule.routes < 28 * Experts;
            if (schedule.large_routed) {
                if (!LongGate && schedule.routes >= 32 * Experts) {
                    launch_fp4_fp4_pipeline<64, 256, 64, 3, true, true, 2, true, false,
                                           3, 64, 3, ShortDown, 4, 2, false, 2>(p, stream, schedule.clamp_in_table, 10);
                } else {
                    launch_fp4_fp4_pipeline<64, 256, 64, 3, true, true, 2, true, false,
                                           3, 64, 3, ShortDown, 4, 2>(p, stream, schedule.clamp_in_table, LongGate ? 8 : 10);
                }
            } else if (compact) {
                launch_fp4_fp4_pipeline<64, 256, 64, 3, true, true, 2, true, true>(p, stream, schedule.clamp_in_table);
            } else if (!LongGate || schedule.routes >= 28 * Experts) {
                launch_fp4_fp4_pipeline<64, 256, 64, 3, true, true, 2, true, false,
                                       3, 64, 3, ShortDown, 4, 2>(p, stream, schedule.clamp_in_table);
            } else {
                launch_fp4_fp4_pipeline<64, 256, 64, 3, true, true, 2, true, false,
                                       3, 64, 3, ShortDown, 4>(p, stream, schedule.clamp_in_table);
            }
        } else if (schedule.medium_pages) {
            if (LongGate) {
                launch_fp4_fp4_pipeline<32, 256, 64, 4, false, true, 1, true, false,
                                       3, 64, 0, ShortDown>(p, stream, schedule.clamp_in_table);
            } else if (schedule.routes >= 15 * Experts) {
                launch_fp4_fp4_pipeline<32, 256, 64, 3, false, true, 1, true, true>(p, stream, schedule.clamp_in_table);
            } else {
                launch_fp4_fp4_pipeline<32, 256, 64, 4, false, false, 1, true, false,
                                       3>(p, stream, schedule.clamp_in_table);
            }
        } else if (schedule.paired_scales) {
            launch_fp4_fp4_pipeline<64, 256, 64, 3, true, true, 2, false, true>(p, stream, schedule.clamp_in_table);
        } else if (schedule.large_queue && p.m_tiles == 4) {
            launch_fp4_fp4_pipeline<64, 128, FullDownChunk, 2>(p, stream, schedule.clamp_in_table);
        } else if (schedule.large_queue) {
            launch_fp4_fp4_pipeline<48, 128, FullDownChunk, 2>(p, stream, schedule.clamp_in_table);
        } else if (p.m_tiles == 2) {
            launch_fp4_fp4_pipeline<32, 256>(p, stream, schedule.clamp_in_table);
        } else {
            launch_fp4_fp4_pipeline<16, 128>(p, stream, schedule.clamp_in_table);
        }
    } else if (schedule.overlap || schedule.routed_only) {
        launch_projection<true, ProjectionRole::Routed>(p, stream);
        launch_projection<false, ProjectionRole::Routed>(p, stream);
    } else {
        launch_projection<true>(p, stream);
        if (p.shared_x && p.tokens)
            decode_ptpc_quantize<<<p.tokens, Threads, 0, stream>>>(p);
        launch_projection<false>(p, stream);
    }
}

}  // namespace gc_moe
