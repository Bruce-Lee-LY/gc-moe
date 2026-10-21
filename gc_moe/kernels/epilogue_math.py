"""Element math for the unified routed/shared L1 epilogue.

These functions operate on values loaded directly from the worker's TMEM
accumulator. The weight and BF16 rounding order follows the reference GC
worker; the caller owns pairwise amax reductions and FP8 stores into L2.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Uint16, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def _sync_wg_id(barrier_idx: Uint32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [barrier_idx.ir_value(loc=loc, ip=ip)],
                    'bar.sync $0, 128;', 'r',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@cute.jit
def sync_epilogue_warpgroup(warpgroup: Uint32) -> None:
    """Reference WG-local barrier ID 3 + warpgroup, with 128 callers."""
    _sync_wg_id(Uint32(3) + warpgroup)


@dsl_user_op
def round_bf16_f32(value: Float32, *, loc=None, ip=None) -> Float32:
    rounded = llvm.inline_asm(
        T.f32(), [value.ir_value(loc=loc, ip=ip)],
        '{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; cvt.f32.bf16 $0, h; }',
        '=f,f', has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return Float32(rounded)


@dsl_user_op
def float_bits(value: Float32, *, loc=None, ip=None) -> Uint32:
    bits = llvm.inline_asm(
        T.i32(), [value.ir_value(loc=loc, ip=ip)],
        'mov.b32 $0, $1;', '=r,f', has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint32(bits)


@dsl_user_op
def bits_float(bits: Uint32, *, loc=None, ip=None) -> Float32:
    value = llvm.inline_asm(
        T.f32(), [bits.ir_value(loc=loc, ip=ip)],
        'mov.b32 $0, $1;', '=f,r', has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return Float32(value)


@dsl_user_op
def fast_rcp(value: Float32, *, loc=None, ip=None) -> Float32:
    reciprocal = llvm.inline_asm(
        T.f32(), [value.ir_value(loc=loc, ip=ip)],
        'rcp.approx.ftz.f32 $0, $1;', '=f,f', has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return Float32(reciprocal)


@dsl_user_op
def abs_f32(value: Float32, *, loc=None, ip=None) -> Float32:
    absolute = llvm.inline_asm(
        T.f32(), [value.ir_value(loc=loc, ip=ip)],
        'abs.f32 $0, $1;', '=f,f', has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return Float32(absolute)


@dsl_user_op
def f32_to_bf16_bits(value: Float32, *, loc=None, ip=None) -> Uint16:
    bits = llvm.inline_asm(
        T.i16(), [value.ir_value(loc=loc, ip=ip)],
        'cvt.rn.bf16.f32 $0, $1;', '=h,f', has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint16(bits)


@dsl_user_op
def f32x2_to_bf16x2_bits(low: Float32, high: Float32,
                         *, loc=None, ip=None) -> Uint32:
    """Reference ``__float22bfloat162_rn``: low/high FP32 to BF16x2."""
    bits = llvm.inline_asm(
        T.i32(), [high.ir_value(loc=loc, ip=ip),
                  low.ir_value(loc=loc, ip=ip)],
        'cvt.rn.bf16x2.f32 $0, $1, $2;', '=r,f,f',
        has_side_effects=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint32(bits)


@dsl_user_op
def bf16x2_min(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    value = llvm.inline_asm(
        T.i32(), [a.ir_value(loc=loc, ip=ip),
                  b.ir_value(loc=loc, ip=ip)],
        'min.bf16x2 $0, $1, $2;', '=r,r,r',
        has_side_effects=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint32(value)


@dsl_user_op
def bf16x2_max(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    value = llvm.inline_asm(
        T.i32(), [a.ir_value(loc=loc, ip=ip),
                  b.ir_value(loc=loc, ip=ip)],
        'max.bf16x2 $0, $1, $2;', '=r,r,r',
        has_side_effects=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint32(value)


@dsl_user_op
def f32x2_to_e4m3x2_bits(low: Float32, high: Float32,
                         *, loc=None, ip=None) -> Uint16:
    """CUDA FP8x2 constructor: low byte from `low`, high from `high`."""
    pair = llvm.inline_asm(
        T.i16(), [low.ir_value(loc=loc, ip=ip),
                  high.ir_value(loc=loc, ip=ip)],
        '{ cvt.rn.satfinite.e4m3x2.f32 $0, $2, $1; }',
        '=h,f,f', has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint16(pair)


@dsl_user_op
def _fast_expf_reference(sigmoid_arg: Float32, *, loc=None, ip=None) -> Float32:
    """Evaluate the fast negative-gate exponential for the W1 epilogue."""
    result = llvm.inline_asm(
        T.f32(), [sigmoid_arg.ir_value(loc=loc, ip=ip)],
        '{ .reg .f32 exponent; '
        'mul.f32 exponent, $1, 0fBFB8AA3B; '
        'ex2.approx.f32 $0, exponent; }',
        '=f,f', has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return Float32(result)


@dsl_user_op
def _bf16x2_high_f32_bits(packed: Uint32, *, loc=None, ip=None) -> Uint32:
    """Place the high BF16 lane in FP32 bits with one byte permutation."""
    bits = llvm.inline_asm(
        T.i32(), [packed.ir_value(loc=loc, ip=ip)],
        'prmt.b32 $0, $1, 0, 0x3254;', '=r,r',
        has_side_effects=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint32(bits)


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
    gate1 = bits_float(_bf16x2_high_f32_bits(gate_bits))
    sigmoid0, sigmoid1 = gate0, gate1
    if cutlass.const_expr(oai_enabled):
        sigmoid0, sigmoid1 = cute.arch.mul_packed_f32x2(
            (gate0, gate1), (alpha, alpha), rnd='rn', ftz=False)
    if cutlass.const_expr(fast_math):
        exp0 = _fast_expf_reference(sigmoid0)
        exp1 = _fast_expf_reference(sigmoid1)
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
    up1 = bits_float(_bf16x2_high_f32_bits(up_bits))
    if cutlass.const_expr(oai_enabled):
        up0, up1 = cute.arch.add_packed_f32x2(
            (up0, up1), (Float32(1), Float32(1)), rnd='rn', ftz=False)
    activation0, activation1 = cute.arch.mul_packed_f32x2(
        (gate0, gate1), (up0, up1), rnd='rn', ftz=False)
    return cute.arch.mul_packed_f32x2(
        (activation0, activation1), (weight0, weight1), rnd='rn', ftz=False)


@cute.jit
def ue8m0_fp8_scale_exp(amax: Float32) -> Uint32:
    """Compute the UE8M0 scale exponent for FP8 E4M3 quantization."""
    bits = float_bits(amax)
    # 0x600000 is 1.75 in FP32 mantissa units; min exponent 105 keeps
    # E4M3 values finite. Preserve the uint32 addition/shift order.
    rounded_exp = (bits + Uint32(0x7fffff) - Uint32(0x600000)) >> 23
    rounded_exp = rounded_exp if rounded_exp > Uint32(113) else Uint32(113)
    return rounded_exp - Uint32(8)


@cute.jit
def ue8m0_fp8_inv(sf_exp: Uint32) -> Float32:
    return bits_float((Uint32(254) - sf_exp) << 23)
