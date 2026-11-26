#pragma once

#include "block_mma.cuh"
#include "epilogue.cuh"
#include "workspace.cuh"

#include <algorithm>

namespace gc_moe {

template <int MTiles>
__device__ __forceinline__ void shared_activation(const Launch& p, const Accumulator (&acc)[MTiles][4],
                                                  int rows, int base, int column, int lane, int row_offset = 0) {
    const int columns[] = {2 * column, 2 * column + 1, I / 16 + 2 * column, I / 16 + 2 * column + 1};
    const float clamp = rounded(p.clamp);
    #pragma unroll
    for (int m = 0; m < MTiles; ++m) {
        #pragma unroll
        for (int tile = 0; tile < 2; ++tile) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int row = row_offset + m * TileM + accumulator_row(lane, j);
                const int col = columns[tile] * 16 + accumulator_col(lane, j);
                if (row < rows) {
                    const float scale = p.shared_x_sf[base + row];
                    const float gate = fminf(rounded(acc[m][tile].v[j] * scale * p.shared_w1_sf[col]), clamp);
                    const float up = fmaxf(-clamp, fminf(rounded(acc[m][tile + 2].v[j] * scale * p.shared_w1_sf[I + col]), clamp));
                    const float silu = evaluate_silu<true>(p.silu_table, gate);
                    p.shared_act[(base + row) * I + col] = bf16(silu * up);
                }
            }
        }
    }
}

template <int MTiles>
__device__ __forceinline__ void shared_fc1(const Launch& p, int base, int column, int lane) {
    const int rows = min(TileM * MTiles, p.tokens - base);
    Accumulator acc[MTiles][4];
    project<false, H, MTiles>(p.shared_x + base * H, nullptr, p.shared_w1,
                            rows, column, lane, acc);
    shared_activation(p, acc, rows, base, column, lane);
}

template <int BlockThreads>
__device__ __forceinline__ void quantize_shared_row(const Launch& p, int row) {
    constexpr int Pairs = (I + 2 * BlockThreads - 1) / (2 * BlockThreads);
    constexpr int BlockWarps = BlockThreads / 32;
    __shared__ float maxima[BlockWarps];
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    float v[2 * Pairs];
    float maximum = 1.e-10f;
    #pragma unroll
    for (int i = 0; i < Pairs; ++i) {
        const int column = 2 * (i * BlockThreads + tid);
        if (column >= I)
            continue;
        v[2 * i] = float(p.shared_act[row * I + column]);
        v[2 * i + 1] = float(p.shared_act[row * I + column + 1]);
        maximum = fmaxf(maximum, fmaxf(fabsf(v[2 * i]), fabsf(v[2 * i + 1])));
    }
    for (int mask = 16; mask; mask /= 2)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, mask));
    if (lane == 0)
        maxima[tid / 32] = maximum;
    __syncthreads();
    maximum = maxima[0];
    #pragma unroll
    for (int i = 1; i < BlockWarps; ++i)
        maximum = fmaxf(maximum, maxima[i]);
    const float scale = maximum * (1.f / 448.f);
    const float inverse = 1.f / scale;
    if (tid == 0)
        p.shared_mid_sf[row] = scale;
    auto* mid = reinterpret_cast<uint16_t*>(p.shared_mid + row * I);
    #pragma unroll
    for (int i = 0; i < Pairs; ++i) {
        if (i * BlockThreads + tid < I / 2)
            mid[i * BlockThreads + tid] = fp8_pair(v[2 * i] * inverse, v[2 * i + 1] * inverse);
    }
}

__global__ void decode_ptpc_quantize(Launch p) {
    quantize_shared_row<Threads>(p, blockIdx.x);
}

template <int MTiles>
__device__ __forceinline__ void shared_store(const Launch& p, const Accumulator (&acc)[MTiles][4],
                                             int rows, int base, int column, int lane, int row_offset = 0) {
    #pragma unroll
    for (int m = 0; m < MTiles; ++m) {
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row = row_offset + m * TileM + lane / 4 + r * 8;
            if (row >= rows)
                continue;
            const float scale = p.shared_mid_sf[base + row];
            uint32_t pairs[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int index = (j & 1) * 4 + 2 * r;
                const int col = column * 64 + j * 8 + (lane % 4) * 2;
                const float lo = acc[m][j / 2].v[index] * scale * p.shared_w2_sf[col];
                const float hi = acc[m][j / 2].v[index + 1] * scale * p.shared_w2_sf[col + 1];
                pairs[j] = bf16_pair(lo, hi);
            }
            auto* dst = reinterpret_cast<uint4*>(p.shared_out);
            dst += (base + row) * (H / 8) + column * 8 + (lane % 4) * 2;
            dst[0] = {pairs[0], pairs[1], pairs[2], pairs[3]};
            dst[1] = {pairs[4], pairs[5], pairs[6], pairs[7]};
        }
    }
}

template <int MTiles>
__device__ __forceinline__ void shared_fc2(const Launch& p, int base, int column, int lane) {
    const int rows = min(TileM * MTiles, p.tokens - base);
    Accumulator acc[MTiles][4];
    project<false, I, MTiles>(p.shared_mid + base * I, nullptr, p.shared_w2,
                            rows, column, lane, acc);
    shared_store(p, acc, rows, base, column, lane);
}

template <bool First, int Rows, int NTile, typename Project, int Cycles>
__device__ __forceinline__ void run_shared_tile(const Launch& p, int tile, int column, uint8_t* memory) {
    constexpr int K = First ? H : I;
    const int base = tile * Rows;
    const int rows = min(Rows, p.tokens - base);
    const auto* a = (First ? p.shared_x : p.shared_mid) + base * K;
    const int task = column * Project::NWarps;
    const auto* b = (First ? p.shared_w1 : p.shared_w2) + task * (K / 32) * Project::BWords;
    const auto acc = Project::template run<false, false, Cycles>(
        a, nullptr, b, rows, reinterpret_cast<typename Project::Stage*>(memory));
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int logical_column = task + warp / Project::MWarps;
    const int row_offset = (warp % Project::MWarps) * TileM * Project::MPanels;
    if constexpr (First)
        shared_activation(p, acc.block, rows, base, logical_column, lane, row_offset);
    else
        shared_store<Project::MPanels>(p, acc.block, rows, base, logical_column, lane, row_offset);
}

__device__ __forceinline__ void quantize_tiny_shared_warp(const Launch& p, int row) {
    constexpr int Pairs = I / 64;
    const int lane = threadIdx.x % 32;
    const auto* src = p.shared_act + row * I;
    float values[2 * Pairs];
    float maximum = 1.e-10f;
    #pragma unroll
    for (int i = 0; i < Pairs; ++i) {
        const int col = (i * 32 + lane) * 2;
        values[2 * i] = float(src[col]);
        values[2 * i + 1] = float(src[col + 1]);
        maximum = fmaxf(maximum, fmaxf(fabsf(values[2 * i]), fabsf(values[2 * i + 1])));
    }
    for (int mask = 16; mask; mask /= 2)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, mask));
    const float scale = maximum * (1.f / 448.f);
    const float inverse = 1.f / scale;
    if (lane == 0)
        p.shared_mid_sf[row] = scale;
    auto* dst = reinterpret_cast<uint16_t*>(p.shared_mid + row * I);
    #pragma unroll
    for (int i = 0; i < Pairs; ++i)
        dst[i * 32 + lane] = fp8_pair(values[2 * i] * inverse, values[2 * i + 1] * inverse);
}

enum TinySharedCounter { GateHead, QuantHead, DownHead, GateDone, QuantDone, TinySharedWords };

__device__ __forceinline__ unsigned* tiny_shared_schedule(const Launch& p) {
    return reinterpret_cast<unsigned*>(p.shared_mid + p.capacity * I);
}

__device__ __forceinline__ unsigned publish_shared_job(unsigned* ready, unsigned increment = 1) {
    unsigned previous;
    asm volatile("ppu.atom.add.acq_rel.gpu.u32 %0, [%1], %2;"
                 : "=r"(previous) : "l"(ready), "r"(increment) : "memory");
    return previous;
}

__device__ __forceinline__ void wait_shared_jobs(const unsigned* ready, unsigned expected) {
    unsigned observed;
    do {
        asm volatile("ppu.ld.acquire.gpu.b32 %0, [%1];"
                     : "=r"(observed) : "l"(ready) : "memory");
    } while (observed < expected);
}

template <int Rows, bool FuseQuantize>
using SharedQueueGate = BlockProject<Rows, 128, H, false, Rows == 64 ? 64 : 128,
                                     (Rows == 64 || FuseQuantize) ? 3 : 2, false,
                                     Rows == 64 ? 2 : 1, true>;

template <int Rows, bool FuseQuantize>
using SharedQueueDown = BlockProject<Rows, 128, I, false, Rows == 64 ? 64 : 128,
                                     (Rows == 64 || FuseQuantize) ? 3 : 2, false,
                                     Rows == 64 ? 2 : 1, true>;

template <bool FuseQuantize, int Rows>
__global__ void decode_shared_tiny(Launch p) {
    constexpr int NTile = 128;
    constexpr int GateColumns = 2 * I / NTile, DownColumns = H / NTile;
    constexpr int Cycles = FuseQuantize ? 2 : 1;
    using Gate = SharedQueueGate<Rows, FuseQuantize>;
    using Down = SharedQueueDown<Rows, FuseQuantize>;
    auto* schedule = tiny_shared_schedule(p);
    const int tid = threadIdx.x;
    extern __shared__ __align__(128) uint8_t memory[];
    __shared__ int job;
    __shared__ bool last_gate;
    for (;;) {
        if (tid == 0)
            job = atomicAdd(schedule + GateHead, 1u);
        __syncthreads();
        const int current = job;
        if (current >= GateColumns)
            break;
        run_shared_tile<true, Rows, NTile, Gate, Cycles>(p, 0, current, memory);
        __syncthreads();
        if constexpr (FuseQuantize) {
            if (tid == 0)
                last_gate = publish_shared_job(schedule + GateDone) == GateColumns - 1;
            __syncthreads();
            if (last_gate) {
                for (int row = tid / 32; row < p.tokens; row += Gate::Threads / 32)
                    quantize_tiny_shared_warp(p, row);
                __syncthreads();
                if (tid == 0)
                    publish_shared_job(schedule + QuantDone, p.tokens);
            }
        } else if (tid == 0) {
            publish_shared_job(schedule + GateDone);
        }
    }
    __syncthreads();
    // Excess gate tickets are unique. Every earlier ticket has already been
    // claimed, so these CTAs can wait without occupying slots needed by quantizers.
    if constexpr (BalancedProjections) {
        const int worker = job - GateColumns;
        if constexpr (!FuseQuantize) {
            if (worker < p.tokens) {
                if (tid == 0)
                    wait_shared_jobs(schedule + GateDone, GateColumns);
                __syncthreads();
                auto quantize_row = [&](int row) {
                    quantize_shared_row<Gate::Threads>(p, row);
                    __syncthreads();
                    if (tid == 0)
                        publish_shared_job(schedule + QuantDone);
                };
                if constexpr (Rows > DownColumns) {
                    for (int row = worker; row < p.tokens; row += DownColumns)
                        quantize_row(row);
                } else {
                    quantize_row(worker);
                }
            }
        }
        if (worker < DownColumns) {
            if (tid == 0)
                wait_shared_jobs(schedule + QuantDone, p.tokens);
            __syncthreads();
            run_shared_tile<false, Rows, NTile, Down, Cycles>(p, 0, worker, memory);
        }
        return;
    }
    // A phase waits only after the earlier work has been claimed by active CTAs.
    if constexpr (!FuseQuantize) {
        for (;;) {
            if (tid == 0) {
                job = atomicAdd(schedule + QuantHead, 1u);
                if (job < p.tokens)
                    wait_shared_jobs(schedule + GateDone, GateColumns);
            }
            __syncthreads();
            const int row = job;
            if (row >= p.tokens)
                break;
            quantize_shared_row<Gate::Threads>(p, row);
            __syncthreads();
            if (tid == 0)
                publish_shared_job(schedule + QuantDone);
        }
        __syncthreads();
    }
    for (;;) {
        if (tid == 0) {
            job = atomicAdd(schedule + DownHead, 1u);
            if (job < DownColumns)
                wait_shared_jobs(schedule + QuantDone, p.tokens);
        }
        __syncthreads();
        const int current = job;
        if (current >= DownColumns)
            break;
        run_shared_tile<false, Rows, NTile, Down, Cycles>(p, 0, current, memory);
        __syncthreads();
    }
}

template <bool FuseQuantize, int Rows = 16>
void launch_tiny_shared(const Launch& p, hggcStream_t stream) {
    using Gate = SharedQueueGate<Rows, FuseQuantize>;
    using Down = SharedQueueDown<Rows, FuseQuantize>;
    constexpr int Scratch = std::max(Gate::ScratchBytes, Down::ScratchBytes);
    constexpr int ColumnJobs = std::max(2 * I, H) / 128;
    const int blocks = FuseQuantize || BalancedProjections ? ColumnJobs : p.units;
    decode_shared_tiny<FuseQuantize, Rows><<<blocks, Gate::Threads, Scratch, stream>>>(p);
}

template <bool First, int Rows, int Stages, int Panels>
__global__ void decode_shared_tiles(Launch p) {
    constexpr int NTile = 128;
    using Project = BlockProject<Rows, NTile, First ? H : I, false, 64, Stages, false, Panels, true>;
    extern __shared__ __align__(128) uint8_t memory[];
    const int tiles = (p.tokens + Rows - 1) / Rows;
    run_shared_tile<First, Rows, NTile, Project, 1>(p, blockIdx.x % tiles, blockIdx.x / tiles, memory);
}

template <bool First, int Rows, int Stages, int Panels>
void launch_shared_tiles(const Launch& p, hggcStream_t stream) {
    constexpr int NTile = 128;
    using Project = BlockProject<Rows, NTile, First ? H : I, false, 64, Stages, false, Panels, true>;
    const int blocks = ((p.tokens + Rows - 1) / Rows) * ((First ? 2 * I : H) / NTile);
    // Neighboring CTAs reuse the same weight columns for different row tiles.
    decode_shared_tiles<First, Rows, Stages, Panels>
        <<<blocks, Project::Threads, Project::ScratchBytes, stream>>>(p);
}

}  // namespace gc_moe
