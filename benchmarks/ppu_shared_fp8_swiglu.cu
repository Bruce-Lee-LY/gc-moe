#include <hggc.h>
#include <hggc_bf16.h>
#include <hggc_fp8.h>

namespace {

constexpr int threads = 256;
constexpr int elements_per_vector = 8;

__device__ __forceinline__ __ppu_bfloat162 swiglu_pair(
    __ppu_bfloat162 gate, __ppu_bfloat162 up, float limit) {
    const float g0 = fminf(__bfloat162float(__low2bfloat16(gate)), limit);
    const float g1 = fminf(__bfloat162float(__high2bfloat16(gate)), limit);
    const float u0 = fmaxf(-limit, fminf(__bfloat162float(__low2bfloat16(up)), limit));
    const float u1 = fmaxf(-limit, fminf(__bfloat162float(__high2bfloat16(up)), limit));
    const float s0 = g0 * (1.f / (1.f + expf(-g0)));
    const float s1 = g1 * (1.f / (1.f + expf(-g1)));
    return __hmul2(__floats2bfloat162_rn(u0, u1), __floats2bfloat162_rn(s0, s1));
}

__global__ void swiglu_quantize(const uint4* gate_up, __hg_fp8_e4m3* output,
                                float* scales, int width, float clamp) {
    extern __shared__ __align__(16) unsigned char storage[];
    auto* products = reinterpret_cast<uint4*>(storage);
    __shared__ float warp_max[threads / 32];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int vectors = width / elements_per_vector;
    const auto* gate = gate_up + size_t(row) * 2 * vectors;
    const auto* up = gate + vectors;
    const float limit = __bfloat162float(__float2bfloat16(clamp));
    float maximum = 0.f;
    for (int col = tid; col < vectors; col += threads) {
        const uint4 gates = gate[col];
        const uint4 ups = up[col];
        uint4 result;
        const auto* gate_pairs = reinterpret_cast<const __ppu_bfloat162*>(&gates);
        const auto* up_pairs = reinterpret_cast<const __ppu_bfloat162*>(&ups);
        auto* pairs = reinterpret_cast<__ppu_bfloat162*>(&result);
#pragma unroll
        for (int j = 0; j < elements_per_vector / 2; ++j) {
            const auto pair = swiglu_pair(gate_pairs[j], up_pairs[j], limit);
            pairs[j] = pair;
            maximum = fmaxf(maximum, fabsf(__bfloat162float(__low2bfloat16(pair))));
            maximum = fmaxf(maximum, fabsf(__bfloat162float(__high2bfloat16(pair))));
        }
        products[col] = result;
    }
    for (int delta = 16; delta; delta /= 2)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, delta));
    if (tid % 32 == 0) warp_max[tid / 32] = maximum;
    __syncthreads();
    maximum = 0.f;
#pragma unroll
    for (int warp = 0; warp < threads / 32; ++warp)
        maximum = fmaxf(maximum, warp_max[warp]);
    // Match the scalar reciprocal multiply and rounding in the PTPC reference.
    const float scale = fmaxf(maximum, 1.e-10f) * (1.f / 448.f);
    const float inverse = 1.f / scale;
    if (tid == 0) scales[row] = scale;
    auto* dest = reinterpret_cast<uint2*>(output + size_t(row) * width);
    for (int col = tid; col < vectors; col += threads) {
        const uint4 product = products[col];
        const auto* pairs = reinterpret_cast<const __ppu_bfloat162*>(&product);
        uint2 packed;
        auto* words = reinterpret_cast<uint16_t*>(&packed);
#pragma unroll
        for (int j = 0; j < elements_per_vector / 2; ++j) {
            const float low = __bfloat162float(__low2bfloat16(pairs[j])) * inverse;
            const float high = __bfloat162float(__high2bfloat16(pairs[j])) * inverse;
            words[j] = __hg_fp8x2_e4m3(make_float2(low, high)).__x;
        }
        dest[col] = packed;
    }
}

}  // namespace

extern "C" int gc_moe_shared_swiglu_fp8(const void* gate_up, void* output, float* scales,
                                        int rows, int width, float clamp, hggcStream_t stream) {
    swiglu_quantize<<<rows, threads, width * sizeof(__ppu_bfloat16), stream>>>(
        static_cast<const uint4*>(gate_up), static_cast<__hg_fp8_e4m3*>(output),
        scales, width, clamp);
    return hggcGetLastError();
}
