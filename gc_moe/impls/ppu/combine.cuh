#pragma once

#include "schedule.cuh"
#include "shared_expert.cuh"

#include <type_traits>

namespace gc_moe {

template <bool ReleaseSignal = false>
__global__ void decode_rank_barrier(Launch p) {
    auto& local = header(p, p.rank);
    const int peer = threadIdx.x;
    const auto epoch = __ldbl(local.published + p.rank);
    if constexpr (MaxRanks > 32) {
        for (int i = peer; i < p.ranks; i += 32)
            publish_return_epoch<ReleaseSignal>(header(p, i).outputs_ready + p.rank, epoch);
        __syncwarp();
        for (int i = peer; i < p.ranks; i += 32)
            while (acquire(local.outputs_ready + i) != epoch) {}
    } else if (peer < p.ranks) {
        publish_return_epoch<ReleaseSignal>(header(p, peer).outputs_ready + p.rank, epoch);
        while (acquire(local.outputs_ready + peer) != epoch) {}
    }
}

__device__ __forceinline__ float unpack_bf16(uint32_t pair, int half) {
    return __uint_as_float((pair << (half ? 0 : 16)) & 0xffff0000u);
}

template <bool ReleaseSignal>
__device__ __forceinline__ void announce_returns(const Launch& p, uint32_t epoch) {
    if (threadIdx.x >= 32)
        return;
    auto& local = header(p, p.rank);
    const int lane = threadIdx.x;
    int elected = 0;
    if (lane == 0 && __ldbl(&local.return_announced) != epoch)
        elected = atomicExch(&local.return_announced, epoch) != epoch;
    if constexpr (MaxRanks > 32) {
        if (__shfl_sync(0xffffffff, elected, 0))
            for (int peer = lane; peer < p.ranks; peer += 32)
                publish_return_epoch<ReleaseSignal>(header(p, peer).outputs_ready + p.rank, epoch);
    } else if (__shfl_sync(0xffffffff, elected, 0) && lane < p.ranks) {
        publish_return_epoch<ReleaseSignal>(header(p, lane).outputs_ready + p.rank, epoch);
    }
}

template <int BlockThreads, bool WaitForReturns = false, bool DenseReturns = false,
          bool BlockWait = false, bool ReadRoutesEarly = false, bool ResetTinyShared = false,
          bool ReleaseSignal = false, bool ReadSharedEarly = false, int TokensPerBlock = 1>
__global__ void decode_combine(Launch p) {
    constexpr int Values = 8;
    constexpr int TokenThreads = BlockThreads / TokensPerBlock;
    constexpr int Columns = TokenThreads * Values;
    static_assert(H % Columns == 0);
    static_assert(TokensPerBlock == 1 || (!WaitForReturns && TokenThreads % 32 == 0));
    static_assert(!ReadRoutesEarly || WaitForReturns);
    static_assert(!ReadSharedEarly || ReadRoutesEarly);
    if constexpr (ResetTinyShared) {
        if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x < TinySharedWords) {
            auto* counters = tiny_shared_schedule(p);
            counters[threadIdx.x] = 0;
        }
    }
    const int lane = threadIdx.x % 32;
    uint32_t epoch = 0;
    auto wait_for_returns = [&] {
        // All peers must finish reading this dispatch epoch before reuse.
        if constexpr (MaxRanks > 32) {
            if (!BlockWait || threadIdx.x < 32)
                for (int peer = lane; peer < p.ranks; peer += 32)
                    while (acquire(header(p, p.rank).outputs_ready + peer) != epoch) {}
        } else if (lane < p.ranks && (!BlockWait || threadIdx.x < 32)) {
            while (acquire(header(p, p.rank).outputs_ready + lane) != epoch) {}
        }
        if constexpr (BlockWait)
            __syncthreads();
        else
            __syncwarp();
    };
    if constexpr (WaitForReturns) {
        epoch = __ldbl(header(p, p.rank).published + p.rank);
        announce_returns<ReleaseSignal>(p, epoch);
        if constexpr (!ReadRoutesEarly) {
            wait_for_returns();
            if (blockIdx.x >= p.tokens)
                return;
        }
    }
    const auto* peer = p.peers[p.rank];
    const auto* ids = reinterpret_cast<const int64_t*>(peer + p.id_offset);
    const auto* values = reinterpret_cast<const uint4*>(peer + p.return_offset);
    const int token = blockIdx.x * TokensPerBlock + threadIdx.x / TokenThreads;
    if constexpr (TokensPerBlock > 1) {
        if (token >= p.tokens)
            return;
    }
    const int column = blockIdx.y * TokenThreads + threadIdx.x % TokenThreads;
    const int first_route = token * TopK;
    const bool live_token = !ReadRoutesEarly || token < p.tokens;
    const uint64_t expert = lane < TopK && live_token ? uint64_t(ids[first_route + lane]) : uint64_t(Experts);
    const uint32_t valid = __ballot_sync(0xffffffff, expert < Experts);
    const auto* returns = values + first_route * (H / Values) + column;
    const int offset = token * (H / Values) + column;
    uint4 shared{};
    if constexpr (ReadSharedEarly)
        if (p.shared_x && live_token)
            shared = __ldbl(reinterpret_cast<const uint4*>(p.shared_out) + offset);
    if constexpr (ReadRoutesEarly) {
        wait_for_returns();
        if (!live_token)
            return;
    }
    float routed[Values]{};
    auto add_route = [&](uint4 v) {
        const uint32_t pairs[] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int i = 0; i < Values; ++i)
            routed[i] += unpack_bf16(pairs[i / 2], i % 2);
    };
    const bool dense = DenseReturns && valid == (1u << TopK) - 1;
    if (dense) {
        // Issue the loads together, then accumulate in top-k order.
        uint4 payloads[TopK];
        #pragma unroll
        for (int slot = 0; slot < TopK; ++slot)
            payloads[slot] = __ldbl(returns + slot * (H / Values));
        #pragma unroll
        for (int slot = 0; slot < TopK; ++slot)
            add_route(payloads[slot]);
    } else {
        #pragma unroll
        for (int slot = 0; slot < TopK; ++slot) {
            if (valid & (1u << slot))
                add_route(__ldbl(returns + slot * (H / Values)));
        }
    }
    if constexpr (!ReadSharedEarly)
        if (p.shared_x)
            shared = reinterpret_cast<const uint4*>(p.shared_out)[offset];
    const uint32_t pairs[] = {shared.x, shared.y, shared.z, shared.w};
    uint32_t result[Values / 2];
    #pragma unroll
    for (int i = 0; i < Values / 2; ++i) {
        const float lo = rounded(routed[2 * i]) + unpack_bf16(pairs[i], 0);
        const float hi = rounded(routed[2 * i + 1]) + unpack_bf16(pairs[i], 1);
        result[i] = bf16_pair(lo, hi);
    }
    // Down projections write BF16 pairs in accumulator order.
    const int out_pair = token * (H / 2) + (column / 8) * 32 + (column % 2) * 16 + column / 2 % 4;
    const bool aligned = reinterpret_cast<uintptr_t>(p.out) % alignof(uint32_t) == 0;
    #pragma unroll
    for (int i = 0; i < Values / 2; ++i) {
        const int dest = out_pair + i * 4;
        if (aligned) {
            reinterpret_cast<uint32_t*>(p.out)[dest] = result[i];
        } else {
            auto* dst = reinterpret_cast<uint16_t*>(p.out) + 2 * dest;
            dst[0] = result[i];
            dst[1] = result[i] >> 16;
        }
    }
}

template <int BlockThreads, bool WaitForReturns = false, bool BlockWait = false, bool ReadRoutesEarly = false,
          bool ResetTinyShared = false, bool ReleaseSignal = false, bool ReadSharedEarly = false>
void launch_combine(const Launch& p, dim3 grid, bool dense_returns, hggcStream_t stream) {
    if (dense_returns) {
        decode_combine<BlockThreads, WaitForReturns, true, BlockWait, ReadRoutesEarly, ResetTinyShared, ReleaseSignal, ReadSharedEarly><<<grid, BlockThreads, 0, stream>>>(p);
        return;
    }
    decode_combine<BlockThreads, WaitForReturns, false, BlockWait, ReadRoutesEarly, ResetTinyShared, ReleaseSignal, ReadSharedEarly><<<grid, BlockThreads, 0, stream>>>(p);
}

inline hggcError_t launch_finalize(const Launch& p, const DecodeSchedule& schedule,
                                  hggcStream_t stream, hggcEvent_t done) {
    if (schedule.large_routed) {
        decode_rank_barrier<true><<<1, 32, 0, stream>>>(p);
        if (p.tokens) {
            constexpr int TokensPerBlock = 2;
            const dim3 grid((p.tokens + TokensPerBlock - 1) / TokensPerBlock, H / (8 * CombineThreads));
            decode_combine<CombineThreads * TokensPerBlock, false, true, false, false, false, false, false,
                           TokensPerBlock><<<grid, CombineThreads * TokensPerBlock, 0, stream>>>(p);
        }
        return hggcGetLastError();
    }
    const bool dense_returns = schedule.paged_dispatch && p.bound <= 128;
    if (schedule.tiny_shared) {
        if (schedule.overlap)
            if (const auto status = hggcStreamWaitEvent(stream, done, 0); status != hggcSuccess)
                return status;
        const bool release_signal = LongGate || p.bound <= 4;
        if ((p.bound > 4 && p.bound <= 8) || (LongGate && p.bound <= 2)) {
            auto launch = [&](auto release) {
                constexpr bool Release = decltype(release)::value;
                if (p.tokens < 8) {
                    const dim3 grid(max(1, p.tokens), p.tokens ? H / (8 * 32) : 1);
                    launch_combine<32, true, false, true, true, Release, true>(p, grid, dense_returns, stream);
                } else {
                    const dim3 grid(p.tokens, H / (8 * CombineThreads));
                    launch_combine<CombineThreads, true, true, true, true, Release, true>(p, grid, dense_returns, stream);
                }
            };
            if (release_signal)
                launch(std::true_type{});
            else
                launch(std::false_type{});
            return hggcGetLastError();
        }
        if (LongGate && p.bound > 2 && p.bound <= 4) {
            const dim3 grid(max(1, p.tokens), p.tokens ? H / (8 * 64) : 1);
            launch_combine<64, true, true, true, true, true>(p, grid, dense_returns, stream);
        } else if (p.tokens < 8) {
            const dim3 grid(max(1, p.tokens), p.tokens ? H / (8 * 32) : 1);
            if (release_signal)
                launch_combine<32, true, false, false, true, true>(p, grid, dense_returns, stream);
            else
                launch_combine<32, true, false, false, true>(p, grid, dense_returns, stream);
        } else {
            const dim3 grid(p.tokens, H / (8 * CombineThreads));
            if (release_signal)
                launch_combine<CombineThreads, true, true, false, true, true>(p, grid, dense_returns, stream);
            else
                launch_combine<CombineThreads, true, true, false, true>(p, grid, dense_returns, stream);
        }
        return hggcGetLastError();
    }
    if (schedule.prefetch_returns && !schedule.fused_dispatch) {
        const dim3 grid(max(1, p.tokens), H / (8 * CombineThreads));
        launch_combine<CombineThreads, true, true, true>(p, grid, dense_returns, stream);
        return hggcGetLastError();
    }
    const bool overlap_returns_with_shared = schedule.fused_dispatch && (schedule.prefetch_returns || (BalancedProjections && p.bound > 16));
    if (schedule.paged_dispatch && p.bound > 8 && p.bound <= 32 && !overlap_returns_with_shared &&
        (!p.shared_x || schedule.overlap || !p.tokens)) {
        const dim3 grid(max(1, p.tokens), H / (8 * CombineThreads));
        if (schedule.overlap)
            if (const auto status = hggcStreamWaitEvent(stream, done, 0); status != hggcSuccess)
                return status;
        launch_combine<CombineThreads, true, true>(p, grid, dense_returns, stream);
        return hggcGetLastError();
    }
    if ((schedule.routed_only || schedule.fused_dispatch) && p.ranks == 8 && p.bound <= 8) {
        // A peer cannot write the next epoch's returns until every rank
        // has combined this epoch and published its next request.
        if (schedule.overlap)
            if (const auto status = hggcStreamWaitEvent(stream, done, 0); status != hggcSuccess)
                return status;
        if (p.tokens < 8) {
            const dim3 grid(p.tokens ? p.tokens : 1, p.tokens ? H / (8 * 32) : 1);
            launch_combine<32, true>(p, grid, dense_returns, stream);
        } else {
            launch_combine<CombineThreads, true>(p, dim3(p.tokens, H / (8 * CombineThreads)), dense_returns, stream);
        }
        return hggcGetLastError();
    }
    if (schedule.fused_dispatch && p.bound > 2 * TileM && p.bound <= 4 * TileM)
        decode_rank_barrier<true><<<1, 32, 0, stream>>>(p);
    else
        decode_rank_barrier<><<<1, 32, 0, stream>>>(p);
    if (schedule.overlap) {
        if (const auto status = hggcStreamWaitEvent(stream, done, 0); status != hggcSuccess)
            return status;
    }
    if (schedule.queue_shared_m64)
        launch_combine<CombineThreads, false, false, false, true>(
            p, dim3(p.tokens, H / (8 * CombineThreads)), dense_returns, stream);
    else if (p.tokens && p.tokens < 8)
        launch_combine<32>(p, dim3(p.tokens, H / (8 * 32)), dense_returns, stream);
    else if (p.tokens)
        launch_combine<CombineThreads>(p, dim3(p.tokens, H / (8 * CombineThreads)), dense_returns, stream);
    return hggcGetLastError();
}

}  // namespace gc_moe
