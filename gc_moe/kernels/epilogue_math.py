"""Element math for the unified routed/shared L1 epilogue.

These functions operate on values loaded directly from the worker's TMEM
accumulator. The weight and BF16 rounding order follows the reference GC
worker; the caller owns pairwise amax reductions and FP8 stores into L2.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Uint32

from ..common.math import (
    bf16x2_high_f32_bits,
    bf16x2_max,
    bf16x2_min,
    bits_float,
    f32_to_bf16_bits,
    f32x2_to_bf16x2_bits,
    fast_exp_neg_f32,
    fast_rcp,
)


@cute.jit
def swiglu_pair(gate0_accum: Float32, gate1_accum: Float32,
                up0_accum: Float32, up1_accum: Float32,
                weight0: Float32, weight1: Float32,
                clamp_value: Float32, alpha: Float32,
                clamp_enabled: cutlass.Constexpr,
                oai_enabled: cutlass.Constexpr,
                fast_math: cutlass.Constexpr):
    """Reference W1 float2 SwiGLU, preserving the BF16x2 and FP32 order."""
    gate_bits = f32x2_to_bf16x2_bits(gate0_accum, gate1_accum)
    up_bits = f32x2_to_bf16x2_bits(up0_accum, up1_accum)
    if cutlass.const_expr(clamp_enabled):
        bound = Uint32(f32_to_bf16_bits(clamp_value))
        bound_pair = bound | (bound << Uint32(16))
        gate_bits = bf16x2_min(gate_bits, bound_pair)
        up_bits = bf16x2_max(up_bits, bound_pair ^ Uint32(0x80008000))
        up_bits = bf16x2_min(up_bits, bound_pair)
    gate0 = bits_float((gate_bits & Uint32(0xFFFF)) << Uint32(16))
    gate1 = bits_float(bf16x2_high_f32_bits(gate_bits))
    sigmoid0, sigmoid1 = gate0, gate1
    if cutlass.const_expr(oai_enabled):
        sigmoid0, sigmoid1 = cute.arch.mul_packed_f32x2(
            (gate0, gate1), (alpha, alpha), rnd='rn', ftz=False)
    if cutlass.const_expr(fast_math):
        exp0 = fast_exp_neg_f32(sigmoid0)
        exp1 = fast_exp_neg_f32(sigmoid1)
    else:
        exp0 = cute.math.exp(-sigmoid0, fastmath=False)
        exp1 = cute.math.exp(-sigmoid1, fastmath=False)
    denom0, denom1 = cute.arch.add_packed_f32x2(
        (Float32(1), Float32(1)), (exp0, exp1), rnd='rn', ftz=False)
    if cutlass.const_expr(fast_math):
        gate0, gate1 = cute.arch.mul_packed_f32x2(
            (gate0, gate1), (fast_rcp(denom0), fast_rcp(denom1)), rnd='rn', ftz=False)
    else:
        gate0 = cute.math.div(
            gate0, denom0, fastmath=False, ftz=False,
            rounding=cute.math.RoundingMode.NEAREST_EVEN)
        gate1 = cute.math.div(
            gate1, denom1, fastmath=False, ftz=False,
            rounding=cute.math.RoundingMode.NEAREST_EVEN)
    up0 = bits_float((up_bits & Uint32(0xFFFF)) << Uint32(16))
    up1 = bits_float(bf16x2_high_f32_bits(up_bits))
    if cutlass.const_expr(oai_enabled):
        up0, up1 = cute.arch.add_packed_f32x2(
            (up0, up1), (Float32(1), Float32(1)), rnd='rn', ftz=False)
    activation0, activation1 = cute.arch.mul_packed_f32x2(
        (gate0, gate1), (up0, up1), rnd='rn', ftz=False)
    return cute.arch.mul_packed_f32x2(
        (activation0, activation1), (weight0, weight1), rnd='rn', ftz=False)
