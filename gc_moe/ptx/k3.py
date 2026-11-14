"""K3 BF16 shared流水线的同步、重排与MMA指令。"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Uint32, Uint64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm


@dsl_user_op
def st_async_release_gpu_u64(ptr: cute.Pointer, value: Uint64, *, loc=None, ip=None):
    llvm.inline_asm(None, [ptr.toint(loc=loc, ip=ip).ir_value(), value.ir_value(loc=loc, ip=ip)],
                    'st.async.release.gpu.global.u64 [$0], $1;', 'l,l,~{memory}',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def fence_proxy_async_global(*, loc=None, ip=None):
    llvm.inline_asm(None, [], 'fence.proxy.async.global;', '~{memory}',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def tensormap_acquire_gpu(ptr: cute.Pointer, *, loc=None, ip=None):
    llvm.inline_asm(None, [ptr.toint(loc=loc, ip=ip).ir_value()],
                    'fence.proxy.tensormap::generic.acquire.gpu [$0], 128;', 'l,~{memory}',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def tanh_approx_f32(value: Float32, *, loc=None, ip=None) -> Float32:
    result = llvm.inline_asm(
        T.f32(), [value.ir_value(loc=loc, ip=ip)],
        'tanh.approx.f32 $0, $1;', '=f,f', has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return Float32(result)


@dsl_user_op
def transpose_w2_bf16x8(p0: Uint32, p1: Uint32,
                        p2: Uint32, p3: Uint32, lane: Uint32,
                        *, loc=None, ip=None):
    # 16dp读出每lane含2个token×4个通道；先BF16 RN，再只交换位，不改数值。
    # 三次位交换：lane bit2↔半字，lane bit3↔寄存器bit0，lane bit4↔寄存器bit1。
    # 得到每lane同一token的连续8个BF16；全程warp内，不需要CD或跨warp同步。
    values = llvm.inline_asm(
        ir.Type.parse('!llvm.struct<(i32,i32,i32,i32)>'),
        [p0.ir_value(), p1.ir_value(), p2.ir_value(), p3.ir_value(), lane.ir_value()],
        r'''{
          .reg .b32 a<4>, b<4>, s, x, selector, bits;
          .reg .pred pbit2, pbit3, pbit4;
          and.b32 bits, $8, 4;
          setp.ne.u32 pbit2, bits, 0;
          selp.b32 selector, 0x3276, 0x5410, pbit2;
          shfl.sync.bfly.b32 x, $4, 4, 31, -1;
          prmt.b32 a0, $4, x, selector;
          shfl.sync.bfly.b32 x, $5, 4, 31, -1;
          prmt.b32 a1, $5, x, selector;
          shfl.sync.bfly.b32 x, $6, 4, 31, -1;
          prmt.b32 a2, $6, x, selector;
          shfl.sync.bfly.b32 x, $7, 4, 31, -1;
          prmt.b32 a3, $7, x, selector;
          and.b32 bits, $8, 8;
          setp.ne.u32 pbit3, bits, 0;
          selp.b32 s, a0, a1, pbit3;
          shfl.sync.bfly.b32 x, s, 8, 31, -1;
          selp.b32 b0, x, a0, pbit3;
          selp.b32 b1, a1, x, pbit3;
          selp.b32 s, a2, a3, pbit3;
          shfl.sync.bfly.b32 x, s, 8, 31, -1;
          selp.b32 b2, x, a2, pbit3;
          selp.b32 b3, a3, x, pbit3;
          and.b32 bits, $8, 16;
          setp.ne.u32 pbit4, bits, 0;
          selp.b32 s, b0, b2, pbit4;
          shfl.sync.bfly.b32 x, s, 16, 31, -1;
          selp.b32 $0, x, b0, pbit4;
          selp.b32 $2, b2, x, pbit4;
          selp.b32 s, b1, b3, pbit4;
          shfl.sync.bfly.b32 x, s, 16, 31, -1;
          selp.b32 $1, x, b1, pbit4;
          selp.b32 $3, b3, x, pbit4;
        }''',
        '=r,=r,=r,=r,r,r,r,r,r', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return tuple(Uint32(llvm.extractvalue(T.i32(), values, [i]))
                 for i in range(4))


@dsl_user_op
def _mma_bf16_shared_2cta(desc_a: Uint64, desc_b: Uint64,
                        accum_tmem: Uint32, accumulate: Uint32,
                        n_tokens: cutlass.Constexpr, m_channels: cutlass.Constexpr,
                        *, loc=None, ip=None) -> None:
    # swapAB：A=权重通道；B=激活token。M128时每CTA64通道，N分布在DP上下半部。
    descriptor = Uint32(((m_channels // 16) << 24) | ((n_tokens // 8) << 17)
                        | (1 << 10) | (1 << 7) | (1 << 4))
    llvm.inline_asm(
        None,
        [x.ir_value(loc=loc, ip=ip) for x in
         (accum_tmem, desc_a, desc_b, descriptor, accumulate, Uint32(0))],
        '{ .reg .pred p; setp.ne.b32 p, $4, 0; '
        'tcgen05.mma.cta_group::2.kind::f16 [$0], $1, $2, $3, '
        '{$5, $5, $5, $5, $5, $5, $5, $5}, p; }',
        'r,l,l,r,r,r', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
