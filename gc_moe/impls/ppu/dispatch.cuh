#pragma once

#include "block_mma.cuh"
#include "schedule.cuh"

#include <algorithm>

namespace gc_moe {

__device__ __forceinline__ void publish_epoch(const Launch& p, int peer, bool pipeline) {
    auto* published = header(p, p.rank).published + peer;
    const uint32_t epoch = __ldbl(published) + 1;
    __stbl(published, epoch);
    if (peer == p.rank) {
        *p.tile_count = 0;
        if (pipeline) {
            auto& local = header(p, p.rank);
            local.first_head = local.down_head = local.down_tail = 0;
        }
    }
    __stbl(header(p, peer).inputs_ready + p.rank, ((epoch & 0x7fffff) << 9) | p.tokens);
    __ppu_threadfence_system_uncache();
}

// The final CTA sees the writes completed by the other CTAs before it publishes
// their epoch or builds the tile list.
__device__ __forceinline__ bool finish_stripe(uint32_t* counter, uint32_t stripes) {
    uint32_t completed;
    asm volatile("ppu.atom.add.acq_rel.gpu.u32 %0, [%1], %2;"
                 : "=r"(completed) : "l"(counter), "r"(1u) : "memory");
    if (completed + 1 != stripes)
        return false;
    *counter = 0;
    return true;
}

__device__ __forceinline__ int expert_tile_end(int tiles) {
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    int end = tiles;
    if (tid < ((PageExperts + 31) / 32) * 32) {
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int prior = __shfl_up_sync(0xffffffff, end, offset);
            if (lane >= offset)
                end += prior;
        }
    }
    if constexpr (PageExperts > 32) {
        __shared__ int first_warp_total;
        if (tid == 31)
            first_warp_total = end;
        __syncthreads();
        if (tid >= 32)
            end += first_warp_total;
    }
    return end;
}

template <bool SingleStripe, bool Bitmap = false>
__global__ void decode_publish(Launch p, bool pipeline) {
    constexpr int Stripe = Bitmap ? BitmapStripe : PublishTokens;
    const int peer = blockIdx.x;
    const int lane = threadIdx.x % 32;
    const auto* local = p.peers[p.rank];
    const auto* src_x = reinterpret_cast<const uint4*>(local + p.x_offset);
    const auto* src_sf = local + p.sf_offset;
    const auto* src_ids = reinterpret_cast<const int64_t*>(local + p.id_offset);
    const auto* src_weights = reinterpret_cast<const float*>(local + p.weight_offset);
    auto* target = p.peers[peer];
    auto* inbox = reinterpret_cast<InboxRow*>(target + p.inbox_offset) + p.rank * p.capacity;
    auto* inbox_ids = reinterpret_cast<uint32_t*>(target + p.inbox_ids_offset) + p.rank * p.capacity * TopK;
    const int experts = Experts / p.ranks;
    __shared__ uint32_t masks[Bitmap ? 32 * BitmapWords : 1];
    if constexpr (Bitmap) {
        for (int i = threadIdx.x; i < 32 * BitmapWords; i += blockDim.x)
            masks[i] = 0;
        __syncthreads();
    }
    const int start = blockIdx.y * Stripe;
    const int end = min(start + Stripe, p.tokens);
    for (int token = start + threadIdx.x / 32; token < end; token += Warps) {
        const int slot = token * TopK + lane;
        const int64_t id = lane < TopK ? src_ids[slot] : -1;
        const bool selected = id >= peer * experts && id < (peer + 1) * experts;
        if constexpr (!Bitmap) {
            if (lane < TopK)
                __stbl(inbox_ids + slot, selected ? uint32_t(id) : ~0u);
        }
        if (selected) {
            if constexpr (Bitmap) {
                const int bit = (token - start) * TopK + lane;
                const int expert = int(id) - peer * 32;
                atomicOr(masks + expert * BitmapWords + bit / 32, 1u << (bit % 32));
            }
            if constexpr (!RoutedFP8_FP4)
                __stbl(reinterpret_cast<uint32_t*>(&inbox[token].weights[lane]), __float_as_uint(src_weights[slot]));
        }
        if (__ballot_sync(0xffffffff, selected)) {
            constexpr int Vectors = RoutedActivationBytes / sizeof(uint4);
            for (int i = lane; i < Vectors; i += 32)
                __stbl(inbox[token].x + i, src_x[token * Vectors + i]);
            if constexpr (RoutedFP8_FP4) {
                if (lane == 0)
                    __stbl(inbox[token].sf, reinterpret_cast<const uint32_t*>(src_sf)[token]);
            } else if (lane < H / 512) {
                auto* target_sf = reinterpret_cast<uint4*>(inbox[token].sf);
                __stbl(target_sf + lane, reinterpret_cast<const uint4*>(src_sf)[token * (H / 512) + lane]);
            }
        }
    }
    if constexpr (Bitmap) {
        __syncthreads();
        const int stripes = (p.bound + Stripe - 1) / Stripe;
        for (int i = threadIdx.x; i < 32 * BitmapWords; i += blockDim.x)
            __stbl(inbox_ids + (i / BitmapWords * stripes + blockIdx.y) * BitmapWords + i % BitmapWords, masks[i]);
    }
    __syncwarp();
    if (lane == 0)
        __ppu_threadfence_system_uncache();
    if constexpr (SingleStripe) {
        __syncthreads();
        if (threadIdx.x == 0)
            publish_epoch(p, peer, pipeline);
    } else if constexpr (Bitmap) {
        __shared__ bool last;
        __syncthreads();
        if (threadIdx.x == 0)
            last = finish_stripe(header(p, p.rank).published_stripes + peer, gridDim.y);
        __syncthreads();
        if (last && threadIdx.x == 0)
            publish_epoch(p, peer, pipeline);
    }
}

// All payload CTAs finish their remote stores before the epoch is published.
template <bool Indexed>
__global__ void decode_inputs_ready(Launch p, bool pipeline) {
    const int peer = threadIdx.x;
    if constexpr (Indexed) {
        for (int expert = threadIdx.x; expert < Experts / p.ranks; expert += blockDim.x)
            p.route_counts[expert] = 0;
    }
    if (peer < p.ranks)
        publish_epoch(p, peer, pipeline);
}

template <bool ReleaseSignal = false, bool LocalEpoch = false>
__device__ __forceinline__ uint32_t publish_pull_epoch(const Launch& p, int peer,
                                                       bool single_stripe = false) {
    auto& local = header(p, p.rank);
    // The previous call publishes its return epoch before this stream can reenter dispatch.
    const uint32_t prior = LocalEpoch ? __ldbl(local.outputs_ready + p.rank) : acquire(local.outputs_ready + p.rank);
    const uint32_t epoch = prior + 1;
    auto* published = local.published + peer;
    if (single_stripe)
        __stbl(published, epoch);
    if (single_stripe || (__ldbl(published) != epoch && atomicCAS(published, prior, epoch) == prior)) {
        auto* notice = header(p, peer).inputs_ready + p.rank;
        const uint32_t value = ((epoch & 0x7fffff) << 9) | p.tokens;
        if constexpr (ReleaseSignal) {
            // published_slots acquires this system-scope notification.
            asm volatile("ppu.st.release.sys.global.b32 [%0], %1;"
                         :: "l"(notice), "r"(value) : "memory");
        } else {
            __stbl(notice, value);
            __ppu_threadfence_system_uncache();
        }
    }
    return epoch;
}

template <bool PullIds = false>
__global__ void decode_index_routes(Launch p, bool pipeline = false) {
    __shared__ int slots;
    const int peer = blockIdx.x;
    const int experts = Experts / p.ranks;
    auto& local = header(p, p.rank);
    auto* counts = p.route_counts + (PullIds ? experts : 0);
    if (threadIdx.x == 0) {
        if constexpr (PullIds) {
            slots = published_slots(p, peer, publish_pull_epoch(p, peer));
            if (blockIdx.y == 0)
                local.indexed_slots[peer] = slots;
        } else {
            slots = published_slots(p, peer);
        }
    }
    __syncthreads();

    constexpr int StripeSlots = (PullIds ? PullIndexTokens : PublishTokens) * TopK;
    const int slot = blockIdx.y * StripeSlots + threadIdx.x;
    if (threadIdx.x < StripeSlots && slot < slots) {
        const int source = peer * p.capacity * TopK + slot;
        uint64_t id;
        if constexpr (PullIds) {
            const auto* peer_ids = reinterpret_cast<const int64_t*>(p.peers[peer] + p.id_offset);
            id = uint64_t(__ldbl(peer_ids + slot));
        } else {
            const auto* inbox_ids = reinterpret_cast<const uint32_t*>(p.peers[p.rank] + p.inbox_ids_offset);
            id = __ldbl(inbox_ids + source);
        }
        const uint64_t expert = id - p.rank * experts;
        if constexpr (PullIds) {
            // This path owns the local inbox metadata until dispatch completes.
            // Keep the row index wide enough for every source route to target one expert.
            auto* token_routes = reinterpret_cast<uint32_t*>(p.peers[p.rank] + p.inbox_ids_offset);
            uint32_t route = ~0u;
            if (expert < experts) {
                const uint32_t row = atomicAdd(counts + expert, 1u);
                route = (uint32_t(expert) << RouteIndexBits) | row;
            }
            token_routes[source] = route;
        } else if (expert < experts) {
            const int index = atomicAdd(counts + expert, 1u);
            p.route_index[expert * (p.capacity * p.ranks * TopK) + index] = source;
        }
    }
    if constexpr (PullIds) {
        __shared__ bool last;
        __syncthreads();
        if (threadIdx.x == 0) {
            last = finish_stripe(&local.indexed_stripes, gridDim.x * gridDim.y);
            if (last && pipeline)
                local.first_head = local.down_head = local.down_tail = 0;
        }
        __syncthreads();
        if (last) {
            const int expert = threadIdx.x;
            const int tile_rows = TileM * p.m_tiles;
            const bool valid = expert < PageExperts;
            const int rows = valid ? counts[expert] : 0;
            const int tiles = (rows + tile_rows - 1) / tile_rows;
            const int end = expert_tile_end(tiles);
            if (valid) {
                counts[expert] = 0;
                if (p.stats)
                    p.stats[expert] += rows;
                p.route_counts[expert] = end - tiles;
                if (expert == PageExperts - 1)
                    *p.tile_count = end;
                for (int i = 0; i < tiles; ++i) {
                    const int tile = end - tiles + i;
                    p.tiles[tile] = {expert, min(rows - i * tile_rows, tile_rows), tile * tile_rows};
                    p.finished_first[tile] = 0;
                    p.down_tiles[tile] = 0;
                }
            }
        }
    }
}

__device__ __forceinline__ void gather_aiu_row(const Launch& p, const InboxRow& token, int source,
                                              int dest, int tile_rows, int lane) {
    constexpr int Vectors = RoutedActivationBytes / sizeof(uint4);
    auto* grouped = reinterpret_cast<uint4*>(p.grouped_x) + dest * Vectors;
    for (int i = lane; i < Vectors; i += 32)
        grouped[i] = __ldbl(token.x + i);
    if constexpr (RoutedFP8_FP4) {
        if (lane == 0)
            reinterpret_cast<uint32_t*>(p.grouped_sf)[dest] = __ldbl(token.sf);
    } else {
        auto* transposed = reinterpret_cast<uint16_t*>(p.grouped_sf);
        const int tile_row = dest % tile_rows;
        transposed += dest / tile_rows * (H / 64) * tile_rows + tile_row;
        for (int i = lane; i < H / 128; i += 32) {
            const uint32_t pair = __ldbl(token.sf + i);
            transposed[2 * i * tile_rows] = pair;
            transposed[(2 * i + 1) * tile_rows] = pair >> 16;
        }
    }
    if (lane == 0) {
        p.return_rows[dest] = encode_return_slot(p, source);
        if constexpr (!RoutedFP8_FP4)
            p.grouped_weights[dest] = __uint_as_float(__ldbl(reinterpret_cast<const uint32_t*>(
                &token.weights[source % TopK])));
    }
}

template <bool Indexed, bool Bitmap = false>
__global__ void decode_dispatch(Launch p, bool pipeline) {
    extern __shared__ int routes[];
    __shared__ int starts[Warps + 1], source_slots[MaxRanks], tile_start;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int experts = Experts / p.ranks;
    const int tile_rows = TileM * p.m_tiles;
    const int expert = p.rank * experts + blockIdx.x;
    const auto* inbox = reinterpret_cast<const InboxRow*>(p.peers[p.rank] + p.inbox_offset);
    const auto* inbox_ids = reinterpret_cast<const uint32_t*>(p.peers[p.rank] + p.inbox_ids_offset);
    int* warp_routes = nullptr;
    int count = 0;
    if constexpr (!Indexed) {
        if (tid < p.ranks)
            source_slots[tid] = published_slots(p, tid);
        __syncthreads();

        // Each source slot is unique, including repeated experts within one token.
        const int warp_stride = (p.ranks + Warps - 1) / Warps * p.bound * TopK;
        warp_routes = routes + warp * warp_stride;
        if constexpr (Bitmap) {
            const int stripes = (p.bound + BitmapStripe - 1) / BitmapStripe;
            const int words = stripes * BitmapWords;
            // Bitmap dispatch is selected only below the indexed threshold, so
            // a single warp can read every route word for this expert and peer.
            for (int peer = warp; peer < p.ranks; peer += Warps) {
                const int first = (lane / BitmapWords) * BitmapStripe * TopK;
                const int offset = first + lane % BitmapWords * 32;
                uint32_t mask = 0;
                if (lane < words && first < source_slots[peer])
                    mask = __ldbl(inbox_ids + peer * p.capacity * TopK + blockIdx.x * words + lane);
                const int local = __popc(mask);
                int scan = local;
                #pragma unroll
                for (int distance = 1; distance < 32; distance <<= 1) {
                    const int prior = __shfl_up_sync(0xffffffff, scan, distance);
                    if (lane >= distance)
                        scan += prior;
                }
                int out = count + scan - local;
                while (mask) {
                    warp_routes[out++] = peer * p.capacity * TopK + offset + __ffs(mask) - 1;
                    mask &= mask - 1;
                }
                count += __shfl_sync(0xffffffff, scan, 31);
            }
        } else {
            for (int peer = warp; peer < p.ranks; peer += Warps) {
                const int slots = source_slots[peer];
                const int peer_base = peer * p.capacity * TopK;
                for (int base = 0; base < slots; base += 32) {
                    const int slot = base + lane;
                    const bool match = slot < slots && int(__ldbl(inbox_ids + peer_base + slot)) == expert;
                    const uint32_t mask = __ballot_sync(0xffffffff, match);
                    if (match)
                        warp_routes[count + __popc(mask & ((1u << lane) - 1))] = peer_base + slot;
                    count += __popc(mask);
                }
            }
        }
        if (lane == 0)
            starts[warp + 1] = count;
        __syncthreads();
    }
    if (tid == 0) {
        if constexpr (Indexed) {
            starts[Warps] = p.route_counts[blockIdx.x];
        } else {
            starts[0] = 0;
            #pragma unroll
            for (int i = 1; i <= Warps; ++i)
                starts[i] += starts[i - 1];
        }
        const int rows = starts[Warps];
        if (p.stats)
            p.stats[blockIdx.x] += rows;
        tile_start = rows ? atomicAdd(p.tile_count, (rows + tile_rows - 1) / tile_rows) : 0;
    }
    __syncthreads();

    const int rows = starts[Warps];
    for (int i = tid; i * tile_rows < rows; i += blockDim.x) {
        const int tile = tile_start + i;
        p.tiles[tile] = {int(blockIdx.x), min(rows - i * tile_rows, tile_rows), tile * tile_rows};
        if (pipeline) {
            p.finished_first[tile] = 0;
            p.down_tiles[tile] = 0;
        }
    }
    const int gather_warps = Indexed ? blockDim.x / 32 : Warps;
    if constexpr (Indexed)
        count = (rows + gather_warps - 1 - warp) / gather_warps;
    const auto* indexed_routes = Indexed ? p.route_index + blockIdx.x * (p.capacity * p.ranks * TopK) : nullptr;
    for (int row = 0; row < count; ++row) {
        const int position = Indexed ? row * gather_warps + warp : starts[warp] + row;
        const int source = Indexed ? indexed_routes[position] : warp_routes[row];
        const auto& token = inbox[source / TopK];
        const int dest = tile_start * tile_rows + position;
        if (p.aiu || RoutedFP8_FP4) {
            gather_aiu_row(p, token, source, dest, tile_rows, lane);
            continue;
        }
        auto* grouped = reinterpret_cast<uint4*>(p.grouped_x) + dest * (H / 32);
        auto* scales = reinterpret_cast<uint32_t*>(p.grouped_sf) + dest * (H / 128);
        const auto* token_scales = reinterpret_cast<const uint32_t*>(token.sf);
        for (int i = lane; i < H / 64; i += 32) {
            const auto lo = __ldbl(token.x + 2 * i);
            const auto hi = __ldbl(token.x + 2 * i + 1);
            grouped[2 * i] = {lo.x, hi.x, lo.y, hi.y};
            grouped[2 * i + 1] = {lo.z, hi.z, lo.w, hi.w};
        }
        for (int i = lane; i < H / 128; i += 32)
            scales[i] = __ldbl(token_scales + i);
        if (lane == 0) {
            p.return_rows[dest] = encode_return_slot(p, source);
            p.grouped_weights[dest] = __uint_as_float(__ldbl(reinterpret_cast<const uint32_t*>(
                &token.weights[source % TopK])));
        }
    }
}

template <bool PairedScales, bool PadScales = true, bool DirectActivation = false, bool PanelMajorScales = false, bool BulkActivation = false>
__device__ __forceinline__ void store_pulled_routes(const Launch& p, const uint4* stage, uint32_t scale,
                                                    int dest, uint32_t mask, int tile_rows, int lane, uint32_t tail_scale = 0, int owner = 0) {
    static_assert(!PanelMajorScales || (DirectActivation && !RoutedFP8_FP4));
    if constexpr (!DirectActivation) {
        asm volatile("ppu.cp.async.wait_all;" ::: "memory");
        __syncwarp();
    }
    constexpr int Vectors = RoutedActivationBytes / sizeof(uint4);
    constexpr int Chunks = (Vectors + 31) / 32;
    uint4 x[Chunks];
    bool use_bulk = false;
    if constexpr (BulkActivation && Vectors % 32 == 0) {
        const uint64_t address = reinterpret_cast<uint64_t>(stage);
        const uint64_t low = __ppu_read_firstlane(uint32_t(address));
        const uint64_t high = __ppu_read_firstlane(uint32_t(address >> 32));
        use_bulk = !((low | reinterpret_cast<uint64_t>(p.grouped_x)) & 511u);
        if (use_bulk) {
            // Bulk transfers consume a warp-aligned base and one vector per lane.
            const auto* src = reinterpret_cast<const uint4*>((high << 32) | low);
            __syncwarp();
            if (p.rank == owner) {
                #pragma unroll
                for (int i = 0; i < Chunks; ++i)
                    x[i] = __ppu_global_ldcg_bulk_b32x4(src + i * 32 + lane);
            } else {
                #pragma unroll
                for (int i = 0; i < Chunks; ++i) {
                    x[i] = __ppu_remote_load_bulk_b32x4(src + i * 32 + lane);
                }
            }
            __ppu_wait_vldcnt(0);
        }
    }
    if (!use_bulk) {
        #pragma unroll
        for (int i = 0; i < Chunks; ++i) {
            const int index = i * 32 + lane;
            if (index < Vectors) {
                if constexpr (DirectActivation)
                    x[i] = __ldbl(stage + index);
                else
                    x[i] = stage[index];
            }
        }
    }
    while (mask) {
        const int slot = __ffs(mask) - 1;
        int row = __shfl_sync(0xffffffff, dest, slot);
        if constexpr (BulkActivation)
            row = __ppu_read_firstlane(uint32_t(row));
        auto* out = reinterpret_cast<uint4*>(p.grouped_x) + row * Vectors;
        #pragma unroll
        for (int i = 0; i < Chunks; ++i) {
            const int index = i * 32 + lane;
            if (index < Vectors) {
                if constexpr (BulkActivation && Vectors % 32 == 0) {
                    if (use_bulk)
                        __ppu_global_stbl_bulk_b32x4(out + index, x[i]);
                    else
                        out[index] = x[i];
                } else {
                    out[index] = x[i];
                }
            }
        }
        if constexpr (RoutedFP8_FP4) {
            if (lane == 0)
                reinterpret_cast<uint32_t*>(p.grouped_sf)[row] = scale;
        } else if constexpr (PanelMajorScales) {
            static_assert(!PairedScales && !PadScales);
            // Adjacent halves of each scale word belong to rows eight apart.
            const int panel = row / TileM;
            const int scale_row = paired_scale_row(row % TileM);
            auto* scales = reinterpret_cast<uint16_t*>(p.grouped_sf) + panel * TileM * (H / 64);
            #pragma unroll
            for (int pair = lane; pair < H / 128; pair += 32) {
                const uint32_t value = pair < 32 ? scale : tail_scale;
                const int index = 2 * packed_gate_scale_index(scale_row / 2, 2 * pair) + scale_row % 2;
                scales[index] = value;
                scales[index + 2] = value >> 16;
            }
        } else {
            const int offset = row % tile_rows;
            const int scale_row = PairedScales ? paired_scale_row(offset) : offset;
            auto* scales = reinterpret_cast<uint16_t*>(p.grouped_sf) + row / tile_rows * tile_rows * (H / 64);
            #pragma unroll
            for (int pair = lane; pair < H / 128; pair += 32) {
                const uint32_t value = pair < 32 ? scale : tail_scale;
                scales[2 * pair * tile_rows + scale_row] = value;
                scales[(2 * pair + 1) * tile_rows + scale_row] = value >> 16;
            }
            if constexpr (PairedScales && PadScales) {
                const int live = p.tiles[row / tile_rows].rows;
                if (offset + 1 == live) {
                    const int padded = (live + TileM - 1) / TileM * TileM;
                    for (int missing = live; missing < padded; ++missing) {
                        const int missing_scale = paired_scale_row(missing);
                        #pragma unroll
                        for (int pair = lane; pair < H / 128; pair += 32) {
                            scales[2 * pair * tile_rows + missing_scale] = 0;
                            scales[(2 * pair + 1) * tile_rows + missing_scale] = 0;
                        }
                    }
                }
            }
        }
        mask &= mask - 1;
    }
    if constexpr (BulkActivation)
        __ppu_wait_vstcnt(0);
}

template <bool PairedScales>
__global__ void decode_pull_tokens(Launch p) {
    extern __shared__ __align__(128) uint8_t storage[];
    __shared__ int tokens;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int owner = blockIdx.x;
    if (threadIdx.x == 0)
        tokens = header(p, p.rank).indexed_slots[owner] / TopK;
    __syncthreads();
    const int token = blockIdx.y * GatherWarps + warp;
    if (token >= tokens)
        return;
    auto* stage = reinterpret_cast<uint4*>(storage) + warp * (RoutedActivationBytes / sizeof(uint4));
    const int tile_rows = PairedScales ? 4 * TileM : p.m_tiles * TileM;
    const int origin = (owner * p.capacity + token) * TopK;
    const auto* token_routes = reinterpret_cast<const uint32_t*>(p.peers[p.rank] + p.inbox_ids_offset);
    const uint32_t dest_code = lane < TopK ? token_routes[origin + lane] : ~0u;
    uint32_t mask = __ballot_sync(0xffffffff, dest_code != ~0u);
    if (!mask)
        return;
    const auto* peer = p.peers[owner];
    const auto* weights = reinterpret_cast<const uint32_t*>(peer + p.weight_offset);
    const int expert = dest_code >> RouteIndexBits;
    const int index = dest_code & RouteIndexMask;
    int dest = 0;
    if (dest_code != ~0u)
        dest = p.route_counts[expert] * tile_rows + index;
    // The same source row can feed several local experts.
    if (lane == 0)
        copy_linear_tile<RoutedActivationBytes / 128>(stage, peer + p.x_offset + token * RoutedActivationBytes);
    asm volatile("ppu.cp.async.commit_group;" ::: "memory");
    const auto* sf = reinterpret_cast<const uint32_t*>(peer + p.sf_offset + token * RoutedScaleBytes);
    const uint32_t scale = lane < ScaleWordsPerRow ? __ldbl(sf + lane) : 0;
    uint32_t tail_scale = 0;
    if constexpr (ScaleWordsPerRow > 32)
        if (lane < H / 128 - 32)
            tail_scale = __ldbl(sf + lane + 32);
    if (dest_code != ~0u) {
        p.return_rows[dest] = encode_return_row(p, owner, token * TopK + lane);
        if constexpr (!RoutedFP8_FP4)
            p.grouped_weights[dest] = __uint_as_float(__ldbl(weights + token * TopK + lane));
    }
    store_pulled_routes<PairedScales>(p, stage, scale, dest, mask, tile_rows, lane, tail_scale);
}

template <int PageRows>
__device__ __forceinline__ int overflow_page_stride(const Launch& p) {
    return (p.capacity * MaxRanks * TopK + PageRows - 1) / PageRows;
}

template <int Bytes>
__device__ __forceinline__ void copy_page_metadata(void* dst, const void* src) {
    const uint32_t address = __cvta_generic_to_shared(dst);
    asm volatile("ppu.cp.async.ca.shared.global.LLC::128B [%0], [%1], %2;"
                 :: "r"(address), "l"(src), "n"(Bytes) : "memory");
}

template <int BlockWarps, int PageRows, int MetaMode, bool DirectActivation, bool PanelMajorScales, bool BulkActivation>
__device__ __forceinline__ void gather_into_pages(const Launch& p, int owner, int tokens, uint4* storage) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int token = blockIdx.y * BlockWarps + warp;
    if (token >= tokens)
        return;
    constexpr bool PrefetchMetadata = DirectActivation || PageRows == TileM || MetaMode == 3;
    constexpr bool SpeculateActivation = PageRows == TileM && !DirectActivation;
    constexpr int StageWords = (DirectActivation ? 0 : RoutedActivationBytes / sizeof(uint4)) + (PrefetchMetadata ? PageMetadataBytes / 16 : 0);
    const auto* peer = p.peers[owner];
    const auto* ids = reinterpret_cast<const int64_t*>(peer + p.id_offset);
    const auto* weights = reinterpret_cast<const uint32_t*>(peer + p.weight_offset);
    const auto* sf = reinterpret_cast<const uint32_t*>(peer + p.sf_offset + token * RoutedScaleBytes);
    const int source = token * TopK + lane;
    auto* stage = storage + warp * StageWords;
    uint64_t id;
    uint32_t scale, tail_scale = 0, weight = 0;
    if constexpr (PrefetchMetadata) {
        auto* scales = reinterpret_cast<uint32_t*>(stage + (DirectActivation ? 0 : RoutedActivationBytes / sizeof(uint4)));
        auto* staged_ids = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(scales) + PageScaleBytes);
        auto* staged_weights = reinterpret_cast<uint32_t*>(staged_ids + AlignedTopK);
        if constexpr (!RoutedFP8_FP4 && MetaMode && H % 1024 == 0) {
            if (lane == 0)
                copy_linear_tile<H / 1024, false, 32>(scales, sf);
        } else {
            #pragma unroll
            for (int pair = lane; pair < ScaleWordsPerRow; pair += 32)
                copy_page_metadata<4>(scales + pair, sf + pair);
        }
        if constexpr (MetaMode >= 2 && TopK % 2 == 0) {
            if (lane < TopK / 2) {
                const int pair = token * TopK + 2 * lane;
                copy_page_metadata<16>(staged_ids + 2 * lane, ids + pair);
                if constexpr (!RoutedFP8_FP4)
                    copy_page_metadata<8>(staged_weights + 2 * lane, weights + pair);
            }
        } else {
            if (lane < TopK) {
                copy_page_metadata<8>(staged_ids + lane, ids + source);
                if constexpr (!RoutedFP8_FP4)
                    copy_page_metadata<4>(staged_weights + lane, weights + source);
            }
        }
        // On M16 pages, hide route resolution under the speculative activation copy.
        // Larger pages resolve expert IDs before copying staged activations.
        asm volatile("ppu.cp.async.commit_group;" ::: "memory");
        if constexpr (SpeculateActivation) {
            if (lane == 0)
                copy_linear_tile<RoutedActivationBytes / 128>(stage, peer + p.x_offset + token * RoutedActivationBytes);
            asm volatile("ppu.cp.async.commit_group;" ::: "memory");
            asm volatile("ppu.cp.async.wait_group %0;" :: "n"(1) : "memory");
        } else {
            asm volatile("ppu.cp.async.wait_all;" ::: "memory");
        }
        __syncwarp();
        id = lane < TopK ? uint64_t(staged_ids[lane]) : ~0ull;
        scale = lane < ScaleWordsPerRow ? scales[lane] : 0;
        if constexpr (ScaleWordsPerRow > 32)
            if (lane < H / 128 - 32)
                tail_scale = scales[lane + 32];
        if constexpr (!RoutedFP8_FP4)
            weight = lane < TopK ? staged_weights[lane] : 0;
    } else {
        id = lane < TopK ? uint64_t(__ldbl(ids + source)) : ~0ull;
    }
    const uint64_t expert = id - p.rank * PageExperts;
    const uint32_t active = __ballot_sync(0xffffffff, expert < PageExperts);
    if (!active) {
        if constexpr (SpeculateActivation)
            asm volatile("ppu.cp.async.wait_all;" ::: "memory");
        return;
    }
    if constexpr (!SpeculateActivation && !DirectActivation) {
        if (lane == 0)
            copy_linear_tile<RoutedActivationBytes / 128>(stage, peer + p.x_offset + token * RoutedActivationBytes);
        asm volatile("ppu.cp.async.commit_group;" ::: "memory");
    }
    if constexpr (!PrefetchMetadata) {
        scale = lane < ScaleWordsPerRow ? __ldbl(sf + lane) : 0;
        if constexpr (ScaleWordsPerRow > 32)
            if (lane < H / 128 - 32)
                tail_scale = __ldbl(sf + lane + 32);
    }

    uint32_t* overflow = nullptr;
    uint32_t offset = 0;
    uint32_t page = uint32_t(expert);
    if (expert < PageExperts) {
        auto* counts = p.route_counts + PageExperts;
        offset = atomicAdd(counts + expert, 1u);
        // Each expert owns one fixed page. A full page reserves its successor on demand.
        if (offset >= PageRows) {
            overflow = counts + PageExperts + expert * overflow_page_stride<PageRows>(p) + offset / PageRows;
            if (offset % PageRows == 0) {
                page = PageExperts + atomicAdd(&header(p, p.rank).overflow_tail, 1u);
                asm volatile("ppu.st.release.gpu.b32 [%0], %1;" :: "l"(overflow), "r"(page) : "memory");
            }
        }
    }
    // The first row of each overflow page publishes its address before peers use it.
    __syncwarp();
    if (overflow && offset % PageRows != 0) {
        do {
            asm volatile("ppu.ld.acquire.gpu.b32 %0, [%1];" : "=r"(page) : "l"(overflow) : "memory");
        } while (!page);
    }

    int dest = 0;
    if (expert < PageExperts) {
        dest = page * PageRows + offset % PageRows;
        p.return_rows[dest] = encode_return_row(p, owner, source);
        if constexpr (!RoutedFP8_FP4) {
            if constexpr (!PrefetchMetadata)
                weight = __ldbl(weights + source);
            p.grouped_weights[dest] = __uint_as_float(weight);
        }
    }
    const uint4* activation = stage;
    if constexpr (DirectActivation)
        activation = reinterpret_cast<const uint4*>(peer + p.x_offset + token * RoutedActivationBytes);
    store_pulled_routes<PageRows == TileM, false, DirectActivation, PanelMajorScales, BulkActivation>(
        p, activation, scale, dest, active, PageRows, lane, tail_scale, owner);
}

template <int BlockWarps, int PageRows = TileM, int MetaMode = 0, bool DirectActivation = false, bool PanelMajorScales = false, bool OrderedPull = false>
__global__ void decode_page_dispatch(Launch p) {
    extern __shared__ __align__(128) uint8_t storage[];
    __shared__ int tokens;
    __shared__ bool last;
    const int tid = threadIdx.x;
    const int owner = OrderedPull ? (blockIdx.x + p.rank) % MaxRanks : blockIdx.x;
    auto& local = header(p, p.rank);
    if (tid == 0) {
        const bool single_stripe = PageRows == TileM && gridDim.y == 1;
        const uint32_t epoch = publish_pull_epoch<PageRows != 32, OrderedPull>(p, owner, single_stripe);
        tokens = OrderedPull && owner == p.rank ? p.tokens : published_slots(p, owner, epoch) / TopK;
    }
    __syncthreads();
    gather_into_pages<BlockWarps, PageRows, MetaMode, DirectActivation, PanelMajorScales, OrderedPull>(
        p, owner, tokens, reinterpret_cast<uint4*>(storage));
    __syncthreads();
    if (tid == 0) {
        last = finish_stripe(&local.indexed_stripes, gridDim.x * gridDim.y);
        if (last)
            local.first_head = local.down_head = local.down_tail = local.overflow_tail = 0;
    }
    __syncthreads();
    if (!last)
        return;

    const bool valid = tid < PageExperts;
    const int rows = valid ? p.route_counts[PageExperts + tid] : 0;
    const int tiles = (rows + PageRows - 1) / PageRows;
    const int end = expert_tile_end(tiles);
    if (valid) {
        auto* counts = p.route_counts + PageExperts;
        auto* overflow = counts + PageExperts + tid * overflow_page_stride<PageRows>(p);
        counts[tid] = 0;
        if (p.stats)
            p.stats[tid] += rows;
        if (tid == PageExperts - 1)
            *p.tile_count = end;
        int physical = tid;
        for (int i = 0; i < tiles; ++i) {
            if (i) {
                physical = overflow[i];
                overflow[i] = 0;
            }
            const int logical = end - tiles + i;
            p.tiles[logical] = {tid, min(rows - i * PageRows, PageRows), physical * PageRows};
            p.finished_first[logical] = 0;
            p.down_tiles[logical] = 0;
        }
    }
}

template <int BlockWarps, int PageRows = TileM>
void launch_page_dispatch(const Launch& p, hggcStream_t stream) {
    constexpr int WarpsPerBlock = std::max(BlockWarps, PageMinWarps);
    const dim3 grid(p.ranks, (p.bound + WarpsPerBlock - 1) / WarpsPerBlock);
    constexpr bool PrefetchWideMetadata = (LongGate || PageRows == 64) && PageRows != TileM;
    constexpr int Scratch = WarpsPerBlock * (RoutedActivationBytes + ((PageRows == TileM || PrefetchWideMetadata) ? PageMetadataBytes : 0));
    if constexpr (PrefetchWideMetadata) {
        decode_page_dispatch<WarpsPerBlock, PageRows, 3><<<grid, WarpsPerBlock * 32, Scratch, stream>>>(p);
        return;
    }
    if constexpr (LongGate && PageRows == TileM) {
        if (p.bound == 1) {
            decode_page_dispatch<WarpsPerBlock, PageRows, 2><<<grid, WarpsPerBlock * 32, Scratch, stream>>>(p);
            return;
        }
        if (p.bound == 2) {
            decode_page_dispatch<WarpsPerBlock, PageRows, 1><<<grid, WarpsPerBlock * 32, Scratch, stream>>>(p);
            return;
        }
    }
    decode_page_dispatch<WarpsPerBlock, PageRows><<<grid, WarpsPerBlock * 32, Scratch, stream>>>(p);
}

template <int BlockWarps, int PageRows, bool PanelMajorScales = false, bool OrderedPull = false>
void launch_direct_page_dispatch(const Launch& p, hggcStream_t stream) {
    const dim3 grid(p.ranks, (p.bound + BlockWarps - 1) / BlockWarps);
    decode_page_dispatch<BlockWarps, PageRows, 3, true, PanelMajorScales, OrderedPull>
        <<<grid, BlockWarps * 32, BlockWarps * PageMetadataBytes, stream>>>(p);
}

inline void launch_dispatch(const Launch& p, const DecodeSchedule& schedule, hggcStream_t stream) {
    const int bitmap_stripes = (p.bound + BitmapStripe - 1) / BitmapStripe;
    const bool small_bitmap = schedule.paged_dispatch && schedule.routed_only && PageExperts == 32 && !schedule.page_dispatch && !schedule.paired_scales &&
                              p.bound >= TileM && bitmap_stripes * BitmapWords <= 32 &&
                              32 * bitmap_stripes * BitmapWords <= p.capacity * TopK;
    const bool pull_dispatch = schedule.paged_dispatch && p.bound > 0 && !small_bitmap;
    const bool index_routes = pull_dispatch || (p.route_index && schedule.routes >= 3072);
    if (!schedule.page_dispatch && small_bitmap) {
        const int stripes = max(1, (p.tokens + BitmapStripe - 1) / BitmapStripe);
        decode_publish<false, true><<<dim3(p.ranks, stripes), Threads, 0, stream>>>(p, schedule.pipeline);
    } else if (!schedule.page_dispatch && !pull_dispatch) {
        if (p.bound <= PublishTokens) {
            decode_publish<true><<<p.ranks, Threads, 0, stream>>>(p, schedule.pipeline);
        } else {
            if (p.tokens)
                decode_publish<false><<<dim3(p.ranks, (p.tokens + PublishTokens - 1) / PublishTokens), Threads, 0, stream>>>(p, schedule.pipeline);
            if (index_routes)
                decode_inputs_ready<true><<<1, Threads, 0, stream>>>(p, schedule.pipeline);
            else
                decode_inputs_ready<false><<<1, MaxRanks > 32 ? 64 : 32, 0, stream>>>(p, schedule.pipeline);
        }
    }
    if constexpr (RoutedFP8_FP4) {
        if (schedule.page_dispatch && p.m_tiles == 3) {
            launch_direct_page_dispatch<8, 48>(p, stream);
            return;
        }
    }
    if (schedule.tall_pages) {
        if constexpr (CanUseM128) {
            if (p.m_tiles == 8) {
                if (!RoutedFP8_FP4 && schedule.large_routed)
                    launch_direct_page_dispatch<16, 128, !RoutedFP8_FP4, true>(p, stream);
                else
                    launch_direct_page_dispatch<16, 128, !RoutedFP8_FP4>(p, stream);
                return;
            }
        }
        launch_direct_page_dispatch<32, 96>(p, stream);
    } else if (schedule.wide_pages) {
        if (schedule.large_routed) {
            if constexpr (LongGate)
                launch_direct_page_dispatch<8, 64>(p, stream);
            else
                launch_direct_page_dispatch<16, 64>(p, stream);
        } else if (schedule.fused_dispatch && schedule.routes >= 28 * Experts) {
            if constexpr (!LongGate)
                launch_direct_page_dispatch<8, 64>(p, stream);
            else if (schedule.routes >= 30 * Experts)
                launch_direct_page_dispatch<32, 64>(p, stream);
            else
                launch_direct_page_dispatch<16, 64>(p, stream);
        } else {
            launch_page_dispatch<16, 64>(p, stream);
        }
    } else if (schedule.medium_pages) {
        if (schedule.large_routed)
            launch_direct_page_dispatch<8, 32>(p, stream);
        else
            launch_page_dispatch<8, 32>(p, stream);
    } else if (schedule.small_pages) {
        // Keep the metadata scan to one or two warps at small bounds. The
        // activation's staging footprint sets the larger dispatch grid.
        if (schedule.large_routed && LongGate && schedule.routes >= 8 * Experts)
            launch_direct_page_dispatch<32, TileM>(p, stream);
        else if (schedule.fused_dispatch && LongGate && p.bound > 2 && p.bound <= 4)
            launch_direct_page_dispatch<2, TileM>(p, stream);
        else if (schedule.fused_dispatch && LongGate && p.bound > 2 * TileM)
            launch_direct_page_dispatch<16, TileM>(p, stream);
        else if (p.bound <= TileM)
            launch_page_dispatch<2>(p, stream);
        else if (LongGate)
            launch_page_dispatch<8>(p, stream);
        else
            launch_page_dispatch<4>(p, stream);
    } else if (index_routes) {
        const int stripes = (p.bound + PublishTokens - 1) / PublishTokens;
        if (pull_dispatch) {
            const dim3 index(p.ranks, (p.bound + PullIndexTokens - 1) / PullIndexTokens);
            decode_index_routes<true><<<index, PullIndexTokens * TopK, 0, stream>>>(p, schedule.pipeline);
            const dim3 grid(p.ranks, (p.bound + GatherWarps - 1) / GatherWarps);
            constexpr int Scratch = GatherWarps * RoutedActivationBytes;
            if (schedule.paired_scales)
                decode_pull_tokens<true><<<grid, GatherWarps * 32, Scratch, stream>>>(p);
            else
                decode_pull_tokens<false><<<grid, GatherWarps * 32, Scratch, stream>>>(p);
        } else {
            constexpr int IndexThreads = std::max(Threads, (PublishTokens * TopK + 31) / 32 * 32);
            decode_index_routes<false><<<dim3(p.ranks, stripes), IndexThreads, 0, stream>>>(p);
            decode_dispatch<true><<<Experts / p.ranks, 2 * Threads, 0, stream>>>(p, schedule.pipeline);
        }
    } else {
        const auto route_bytes = (p.ranks + Warps - 1) / Warps * Warps * p.bound * TopK * sizeof(int);
        if (small_bitmap)
            decode_dispatch<false, true><<<Experts / p.ranks, Threads, route_bytes, stream>>>(p, schedule.pipeline);
        else
            decode_dispatch<false><<<Experts / p.ranks, Threads, route_bytes, stream>>>(p, schedule.pipeline);
    }
}

}  // namespace gc_moe
