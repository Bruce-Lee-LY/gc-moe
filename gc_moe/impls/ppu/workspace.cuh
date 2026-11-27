#pragma once

#include "mma.cuh"

namespace gc_moe {

constexpr int H = GC_MOE_HIDDEN, I = GC_MOE_INTERMEDIATE;
#ifndef GC_MOE_ROUTED_FP8_FP4
#define GC_MOE_ROUTED_FP8_FP4 0
#endif
constexpr bool RoutedFP8_FP4 = GC_MOE_ROUTED_FP8_FP4;
constexpr int RoutedActivationBytes = RoutedFP8_FP4 ? H : H / 2;
constexpr int RoutedScaleBytes = RoutedFP8_FP4 ? sizeof(float) : H / 32;
#ifndef GC_MOE_MAX_RANKS
#define GC_MOE_MAX_RANKS 8
#endif
constexpr int Experts = GC_MOE_EXPERTS, TopK = GC_MOE_TOPK, MaxRanks = GC_MOE_MAX_RANKS;
constexpr int RouteIndexBits = MaxRanks <= 8 ? 16 : 24;
constexpr uint32_t RouteIndexMask = (1u << RouteIndexBits) - 1;
static_assert(256 * MaxRanks * TopK <= RouteIndexMask + 1);
using ReturnRow = uint64_t;
// A per-rank return area at maximum capacity needs fewer than 24 bits
// to index its BF16 vectors.
constexpr int ReturnPeerShift = 24;
static_assert(256 * TopK * H / 8 < (1 << ReturnPeerShift));
constexpr int TileM = 16, Warps = 4, Threads = Warps * 32;
constexpr int CombineThreads = H % (Threads * 8) == 0 ? Threads : Threads / 2;
constexpr int PublishTokens = 4 * Warps;
constexpr int PullIndexTokens = TopK <= 8 ? 64 : 32;
constexpr int BitmapStripe = 8;
constexpr int BitmapWords = (BitmapStripe * TopK + 31) / 32;
constexpr int GatherWarps = 8;
constexpr int BlockN = 128;
constexpr int PageExperts = Experts / MaxRanks;
constexpr bool CanPage = Experts % MaxRanks == 0 && PageExperts <= 64;
constexpr int PageMinWarps = CanPage ? (PageExperts + 31) / 32 : 1;
constexpr int ScaleWordsPerRow = RoutedScaleBytes / sizeof(uint32_t);
constexpr int PageScaleBytes = (RoutedScaleBytes + 7) / 8 * 8;
constexpr int AlignedTopK = (TopK + 1) & ~1;
constexpr int PageMetadataBytes = (PageScaleBytes + AlignedTopK * 8 + TopK * 4 + 127) / 128 * 128;
constexpr int FullDownChunk = I % 256 == 0 ? 128 : 64;
constexpr int GateStages = H / 128 % 5 == 0 ? 5 : 4;
constexpr bool LongGate = H / 128 > 32;
constexpr bool BalancedProjections = H <= 2 * I;
constexpr bool CanUseM128 = MaxRanks > 8 && BalancedProjections;
constexpr bool ShortDown = I / 128 <= 20;

struct Tile {
    int expert;
    int rows;
    int row_base;
};

struct Launch {
    uint8_t* const* peers;
    uint8_t* grouped_x;
    uint8_t* grouped_sf;
    ReturnRow* return_rows;
    Tile* tiles;
    int* tile_count;
    uint8_t* mid;
    uint8_t* mid_sf;
    const uint4* w1;
    const uint4* w2;
    const uint8_t* shared_x;
    const float* shared_x_sf;
    const uint4* shared_w1;
    const float* shared_w1_sf;
    const uint4* shared_w2;
    const float* shared_w2_sf;
    bf16* shared_act;
    uint8_t* shared_mid;
    float* shared_mid_sf;
    bf16* shared_out;
    bf16* out;
    int* stats;
    float* grouped_weights;
    const float* silu_table;
    uint32_t* route_index;
    uint32_t* route_counts;
    uint32_t* finished_first;
    uint32_t* down_tiles;
    int rank, ranks, capacity, tokens, bound, units;
    int x_offset, sf_offset, id_offset, weight_offset, return_offset, inbox_offset, inbox_ids_offset, aiu;
    int m_tiles;
    float clamp, w1_inv_scale, w2_inv_scale;
};

struct Header {
    uint32_t published[MaxRanks];
    uint32_t inputs_ready[MaxRanks];
    uint32_t outputs_ready[MaxRanks];
    uint32_t first_head, down_head, down_tail;
    uint32_t return_announced;
    uint32_t indexed_stripes, overflow_tail;
    uint32_t indexed_slots[MaxRanks];
    uint32_t published_stripes[MaxRanks];
};
static_assert(sizeof(Header) == (5 * MaxRanks + 6) * sizeof(uint32_t));

struct alignas(256) InboxRow {
    uint4 x[RoutedActivationBytes / sizeof(uint4)];
    uint32_t sf[ScaleWordsPerRow];
    float weights[TopK];
};
static_assert(sizeof(InboxRow) == (RoutedActivationBytes + RoutedScaleBytes + TopK * sizeof(float) + 255) / 256 * 256);

__device__ __forceinline__ Header& header(const Launch& p, int rank) {
    return *reinterpret_cast<Header*>(p.peers[rank]);
}

__device__ __forceinline__ uint32_t acquire(const uint32_t* address) {
    uint32_t result;
    asm volatile("ppu.ld.acquire.sys.global.b32 %0, [%1];"
                 : "=r"(result) : "l"(address) : "memory");
    return result;
}

__device__ __forceinline__ int published_slots(const Launch& p, int peer, uint32_t epoch) {
    const auto& control = header(p, p.rank);
    uint32_t arrival;
    do {
        arrival = acquire(control.inputs_ready + peer);
    } while ((arrival >> 9) != (epoch & 0x7fffff));
    return min(p.bound, int(arrival & 511)) * TopK;
}

__device__ __forceinline__ int published_slots(const Launch& p, int peer) {
    return published_slots(p, peer, acquire(header(p, p.rank).published + p.rank));
}

template <bool ReleaseSignal>
__device__ __forceinline__ void publish_return_epoch(uint32_t* notice, uint32_t epoch) {
    if constexpr (ReleaseSignal) {
        asm volatile("ppu.st.release.sys.global.b32 [%0], %1;"
                     :: "l"(notice), "r"(epoch) : "memory");
    } else {
        __stbl(notice, epoch);
        __ppu_threadfence_system_uncache();
    }
}

__device__ __forceinline__ ReturnRow encode_return_row(const Launch& p, int owner, int slot) {
    const int vector = slot * (H / 8);
    if (p.m_tiles >= 4)
        return (owner << ReturnPeerShift) | vector;
    const auto* row = reinterpret_cast<const uint4*>(p.peers[owner] + p.return_offset) + vector;
    return reinterpret_cast<uint64_t>(row);
}

__device__ __forceinline__ ReturnRow encode_return_slot(const Launch& p, int source) {
    const int stride = p.capacity * TopK;
    return encode_return_row(p, source / stride, source % stride);
}

template <bool Packed = false>
__device__ __forceinline__ uint4* return_row(const Launch& p, int row) {
    const ReturnRow entry = p.return_rows[row];
    if constexpr (Packed) {
        const uint32_t descriptor = entry;
        const int owner = descriptor >> ReturnPeerShift;
        const int vector = descriptor & ((1 << ReturnPeerShift) - 1);
        return reinterpret_cast<uint4*>(p.peers[owner] + p.return_offset) + vector;
    }
    return reinterpret_cast<uint4*>(entry);
}

}  // namespace gc_moe
