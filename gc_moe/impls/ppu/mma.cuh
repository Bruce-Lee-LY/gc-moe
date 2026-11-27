#pragma once

#include <hggc_bf16.h>
#include <hggc_runtime.h>
#include <cstdint>

namespace gc_moe {

using bf16 = __ppu_bfloat16;

struct Accumulator {
    float v[8]{};
};

template <bool FP4, bool FP8_FP4 = false>
constexpr int PackedWeightWords = FP8_FP4 ? 68 : 128 + 8 * FP4;

template <int Rows, bool Rt = false, int Width = 128>
__device__ __forceinline__ void copy_linear_tile(void* dst, const void* src) {
    const uint32_t address = uint32_t(__cvta_generic_to_shared(dst));
    if constexpr (Rt) {
        asm volatile("ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.rt.LLC::128B.padz.linear.b8 "
                     "[%0], [%1], {%2,%3,%4}, {%2,%3,%4}, {%5,%6}, {%7,%7,%7}, %7;"
                     :: "r"(address), "l"(src), "r"(Width), "r"(Rows), "r"(1),
                        "l"(uint64_t(Width)), "l"(uint64_t(Width) * Rows), "r"(0) : "memory");
    } else {
        asm volatile("ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.padz.linear.b8 "
                     "[%0], [%1], {%2,%3,%4}, {%2,%3,%4}, {%5,%6}, {%7,%7,%7}, %7;"
                     :: "r"(address), "l"(src), "r"(Width), "r"(Rows), "r"(1),
                        "l"(uint64_t(Width)), "l"(uint64_t(Width) * Rows), "r"(0) : "memory");
    }
}

// The two instructions consume the same four packed words per lane. Each
// nibble in the MXFP4 operand corresponds to one byte in the E4M3 operand.
template <bool FP4, bool FP8_FP4 = false>
__device__ __forceinline__ void mma(Accumulator& acc, uint4 a, uint4 b,
                                    uint32_t a_scale = 0, uint32_t b_scale = 0) {
    auto& c = acc.v;
    if constexpr (FP4) {
        asm("ppu.tc02.mma.mx.sync.aligned.m16n16k64.row.col.f32.f4.f4.f32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, {%16,%17}, %18, %18, 0;"
            : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]),
              "+r"(c[4]), "+r"(c[5]), "+r"(c[6]), "+r"(c[7])
            : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w),
              "r"(b.x), "r"(b.y), "r"(b.z), "r"(b.w),
              "r"(a_scale), "r"(b_scale), "r"(0u));
    } else {
        #define GC_MOE_FP8_MMA(format) \
        asm("ppu.tc02.mma.sync.aligned.m16n16k32.row.col.f32.e4m3." format ".f32 " \
            "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, " \
            "{%0,%1,%2,%3,%4,%5,%6,%7};" \
            : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]), \
              "+r"(c[4]), "+r"(c[5]), "+r"(c[6]), "+r"(c[7]) \
            : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), \
              "r"(b.x), "r"(b.y), "r"(b.z), "r"(b.w))
        if constexpr (FP8_FP4)
            GC_MOE_FP8_MMA("e5m2");
        else
            GC_MOE_FP8_MMA("e4m3");
        #undef GC_MOE_FP8_MMA
    }
}

__device__ __forceinline__ int accumulator_row(int lane, int index) {
    return lane / 4 + ((index & 2) ? 8 : 0);
}

__device__ __forceinline__ int accumulator_col(int lane, int index) {
    return (lane % 4) * 2 + (index & 1) + ((index & 4) ? 8 : 0);
}

// Each row panel traverses K in its original order. Reuse B across panels
// before advancing the stage or writing either accumulator.
template <bool FP4, int K, int MTiles, int Tiles>
__device__ __forceinline__ void project(const uint8_t* x, const uint8_t* x_sf,
                                        const uint4* weights, int rows, int task, int lane,
                                        Accumulator (&out)[MTiles][Tiles]) {
    static_assert(Tiles == 4);
    constexpr int kTiles = K / (FP4 ? 64 : 32);
    constexpr int rowBytes = K / (FP4 ? 2 : 1);
    constexpr int weightWords = 32 * Tiles;
    const int row = lane / 4;
    constexpr int packet_words = 4 * PackedWeightWords<FP4>;
    weights += (task / 2) * (kTiles / 2) * packet_words;

    // Tail rows have no observable accumulators. Read the last valid row for
    // them, keeping all lane loads uniform and within the activation tile.
    const uint32_t* a_lo[MTiles];
    const uint32_t* a_hi[MTiles];
    const uint16_t* sf_lo[MTiles];
    const uint16_t* sf_hi[MTiles];
    #pragma unroll
    for (int m = 0; m < MTiles; ++m) {
        const int lo = min(row + 16 * m, rows - 1);
        const int hi = min(row + 16 * m + 8, rows - 1);
        const int word = (lane % 4) * (FP4 ? 2 : 1);
        a_lo[m] = reinterpret_cast<const uint32_t*>(x + lo * rowBytes) + word;
        a_hi[m] = reinterpret_cast<const uint32_t*>(x + hi * rowBytes) + word;
        if constexpr (FP4) {
            const auto* data = reinterpret_cast<const uint16_t*>(x_sf);
            sf_lo[m] = data + lo * kTiles;
            sf_hi[m] = data + hi * kTiles;
        }
    }

    #pragma unroll 2
    for (int k = 0; k < kTiles; ++k) {
        const int word = k * 8;
        uint4 a[MTiles];
        uint32_t sa[MTiles]{};
        #pragma unroll
        for (int m = 0; m < MTiles; ++m) {
            if constexpr (FP4) {
                const auto lo = *reinterpret_cast<const uint2*>(a_lo[m] + word);
                const auto hi = *reinterpret_cast<const uint2*>(a_hi[m] + word);
                a[m] = {lo.x, hi.x, lo.y, hi.y};
                sa[m] = sf_lo[m][k] | (uint32_t(sf_hi[m][k]) << 16);
            } else {
                a[m] = {a_lo[m][word], a_hi[m][word], a_lo[m][word + 4], a_hi[m][word + 4]};
            }
        }
        const auto* packet = weights + (k / 2) * packet_words;
        const int weight_step = (task % 2) * 2 + k % 2;
        const auto* src = packet + weight_step * weightWords;
        uint4 scales{};
        if constexpr (FP4)
            scales = packet[4 * weightWords + weight_step * 8 + row];
        const uint32_t sb[] = {scales.x, scales.y, scales.z, scales.w};
        #pragma unroll
        for (int i = 0; i < Tiles; ++i) {
            const uint4 b = src[i * 32 + lane];
            #pragma unroll
            for (int m = 0; m < MTiles; ++m)
                mma<FP4>(out[m][i], a[m], b, sa[m], sb[i]);
        }
    }
}

__device__ __forceinline__ uint8_t fp4_pair(float lo, float hi) {
    // Match the scalar encoder's handling of signed zero and NaN.
    lo = fabsf(lo) > 0.f ? lo : 0.f;
    hi = fabsf(hi) > 0.f ? hi : 0.f;
    uint8_t packed;
    asm("ppu.cvt.rtte.satfinite.e2m1x2.f32 %0, %1, %2;"
        : "=c"(packed) : "f"(hi), "f"(lo));
    return packed;
}

__device__ __forceinline__ uint16_t fp8_pair(float lo, float hi) {
    uint16_t result;
    asm("ppu.cvt.rtte.satfinite.e4m3x2.f32 %0, %1, %2;"
        : "=h"(result) : "f"(hi), "f"(lo));
    return result;
}

}  // namespace gc_moe
