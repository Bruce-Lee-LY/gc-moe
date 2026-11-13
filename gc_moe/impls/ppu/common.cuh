#pragma once

#include <cstdint>

namespace role12 {
// Pointer and integer slots shared by the host pipeline and role12 launcher.
// Keep these in sync with the ctypes bindings in runtime/ppu_gc_moe.py.
enum Ptr {
    XP, SF, TOPK_IDX, TOPK_W, SYM, CURSORS, DONE, COUNTS, POOL, POOL_SF,
    REC_T, REC_SLOT, REC_SRC, REC_W, BUCKET, STARTS, RN, COMPACT_T,
    COMPACT_SLOT, COMPACT_SRC, W1, SFW1, W2, SFW2, X_STARTS, ACT, ACT_SF,
    OUT, W1_CNT, SCRATCH, UNUSED, STATS, SE_W1, SE_SFW1, SE_W2, SE_SFW2, STREAM, NPTR
};
enum Int {
    RANKS, MY, T, TOPK, H, I, E, CAP, CAP_E, POOL_OFF, SF_OFF, REC_OFF,
    CNT_OFF, GEN_OFF, NCOUNTERS, GENERATION, NCTA, ROLE1_CTAS, ROUTE_I64, CLAMP_BITS, NINT
};
}  // namespace role12

// A warp reads one 512 B chunk from each slot. Keep additions in FP32
// and in the original slot order, regardless of the load depth.
template<int LoadAhead, bool WithBias = false>
__device__ __forceinline__ void sum_combine_slots(const char* src, int row_bytes,
                                                  int topk, float (&acc)[8],
                                                  const char* bias = nullptr) {
    if constexpr (WithBias) {
        const uint4 words = __ppu_global_ldcg_bulk_b32x4(bias);
        acc[0] = __uint_as_float(words.x << 16);
        acc[1] = __uint_as_float(words.y << 16);
        acc[2] = __uint_as_float(words.z << 16);
        acc[3] = __uint_as_float(words.w << 16);
        acc[4] = __uint_as_float(words.x & 0xffff0000u);
        acc[5] = __uint_as_float(words.y & 0xffff0000u);
        acc[6] = __uint_as_float(words.z & 0xffff0000u);
        acc[7] = __uint_as_float(words.w & 0xffff0000u);
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = 0.f;
    }
    auto add = [&](uint4 words) {
        acc[0] += __uint_as_float(words.x << 16);
        acc[1] += __uint_as_float(words.y << 16);
        acc[2] += __uint_as_float(words.z << 16);
        acc[3] += __uint_as_float(words.w << 16);
        acc[4] += __uint_as_float(words.x & 0xffff0000u);
        acc[5] += __uint_as_float(words.y & 0xffff0000u);
        acc[6] += __uint_as_float(words.z & 0xffff0000u);
        acc[7] += __uint_as_float(words.w & 0xffff0000u);
    };
    int slot = 0;
    if constexpr (LoadAhead > 1) {
        for (; slot + LoadAhead <= topk; slot += LoadAhead) {
            uint4 words[LoadAhead];
#pragma unroll
            for (int i = 0; i < LoadAhead; ++i)
                words[i] = __ppu_global_ldcg_bulk_b32x4(src + size_t(slot + i) * row_bytes);
#pragma unroll
            for (int i = 0; i < LoadAhead; ++i) add(words[i]);
        }
    }
    for (; slot < topk; ++slot)
        add(__ppu_global_ldcg_bulk_b32x4(src + size_t(slot) * row_bytes));
}
