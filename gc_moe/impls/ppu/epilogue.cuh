#pragma once

#include "mma.cuh"

namespace gc_moe {

__device__ __forceinline__ float rounded(float value) {
    return float(bf16(value));
}

template <bool Shared>
__device__ __forceinline__ float silu_exact(float gate) {
    if constexpr (Shared)
        return rounded(gate * (1.f / (1.f + expf(-gate))));
    else
        return gate / (1.f + expf(-gate));
}

template <bool Shared>
__device__ __forceinline__ float evaluate_silu(const float* table, float gate) {
    const uint32_t bits = __float_as_uint(gate);
    if (Shared || !(bits & 0xffffu))
        return table[(bits >> 16) + (Shared ? 65536 : 0)];
    return silu_exact<Shared>(gate);
}

__global__ void make_silu_table(float* table) {
    const uint32_t bits = blockIdx.x * blockDim.x + threadIdx.x;
    const float gate = __uint_as_float(bits << 16);
    table[bits] = silu_exact<false>(gate);
    table[bits + 65536] = silu_exact<true>(gate);
}

__device__ __forceinline__ uint32_t bf16_pair(float lo, float hi) {
    uint32_t result;
    asm("ppu.cvt.rtte.bf16x2.f32 %0, %1, %2;" : "=r"(result) : "f"(hi), "f"(lo));
    return result;
}

struct ReturnVectors {
    uint4 first, second;
};

__device__ __forceinline__ ReturnVectors pack_return_row(const Accumulator (&acc)[4], int row_half) {
    uint32_t pairs[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int index = (j & 1) * 4 + 2 * row_half;
        pairs[j] = bf16_pair(acc[j / 2].v[index], acc[j / 2].v[index + 1]);
    }
    return {{pairs[0], pairs[1], pairs[2], pairs[3]}, {pairs[4], pairs[5], pairs[6], pairs[7]}};
}

}  // namespace gc_moe
