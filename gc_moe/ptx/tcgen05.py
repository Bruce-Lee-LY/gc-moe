"""Mixed FP8/FP4 two-CTA TCGEN05 operations needed by the GC worker.

CuTe's public blockscaled MmaMXF8Op currently fixes A and B to the same
FP8 dtype. The reference GC worker uses the mixed mxf8f6f4 PTX encoding;
these user ops preserve its operand order while CuTe owns the surrounding
SMEM/TMEM tensors and schedule. They are not a stand-alone GEMM path.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Uint16, Uint32, Uint64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass._mlir import ir


@cute.jit
def blockscaled_instr_desc_hi(umma_n: Uint32, weight_format: Uint32,
                              weight_sf_byte: Uint32,
                              activation_sf_byte: Uint32) -> Uint32:
    """Encode the reference swapped-operand MXF8F6F4 instruction descriptor.

    The descriptor's A operand is the *weight* (E2M1=5 or E4M3=0) and B is
    the FP8 activation. `umma_n` is align(valid_m, 16), not BLOCK_M. SF byte
    selectors map to bits 29:30 for weight A and 4:5 for activation B.
    See the CUTLASS InstrDescriptorBlockScaled bitfield definition.
    """
    return (Uint32(16) << 24) | (Uint32(1) << 23) | \
        ((umma_n >> 3) << 17) | (weight_format << 7) | \
        (weight_sf_byte << 29) | (activation_sf_byte << 4)


@cute.jit
def kmajor_smem_desc(smem_ptr: cute.Pointer, stride_bytes: Uint32,
                     swizzle_128: cutlass.Constexpr,
                     byte_offset: Uint32) -> Uint64:
    """SM100 descriptor for reference K-major 128B data or UTCCP SF tile."""
    address = Uint64(smem_ptr.toint()) + Uint64(byte_offset)
    layout = Uint64(2) if cutlass.const_expr(swizzle_128) else Uint64(0)
    return ((address >> 4) & Uint64(0x3fff)) | \
        (Uint64(stride_bytes >> 4) << 32) | (Uint64(1) << 46) | \
        (layout << 61)


@cute.jit
def kmajor_smem_desc_from_start(start_address_16b: Uint32,
                                stride_bytes: Uint32,
                                swizzle_128: cutlass.Constexpr) -> Uint64:
    """Keep descriptor flags fixed and replace only the reference low address."""
    layout = Uint64(2) if cutlass.const_expr(swizzle_128) else Uint64(0)
    return Uint64(start_address_16b & Uint32(0x3fff)) | \
        (Uint64(stride_bytes >> 4) << 32) | (Uint64(1) << 46) | \
        (layout << 61)


@dsl_user_op
def mma_mxf8f6f4_2cta(desc_a: Uint64, desc_b: Uint64,
                      accum_tmem: Uint32, accumulate: Uint32,
                      instr_desc_hi: Uint32, sfa_tmem: Uint32,
                      sfb_tmem: Uint32, *, loc=None, ip=None) -> None:
    """Issue reference 2CTA FP8×FP4 blockscaled MMA into FP32 TMEM."""
    operands = [accum_tmem, desc_a, desc_b, instr_desc_hi,
                accumulate, sfa_tmem, sfb_tmem]
    llvm.inline_asm(
        None, [x.ir_value(loc=loc, ip=ip) for x in operands],
        '{ .reg .pred p; '
        'setp.ne.b32 p, $4, 0; '
        'tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale '
        '[$0], $1, $2, $3, [$5], [$6], p; }',
        'r,l,l,r,r,r,r', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def utccp_sf_4x32_2cta(smem_desc: Uint64, tmem_col: Uint32,
                       *, loc=None, ip=None) -> None:
    """Copy 128 packed-int32 UE8M0 elements (512 bytes) to TMEM."""
    llvm.inline_asm(
        None, [tmem_col.ir_value(loc=loc, ip=ip),
               smem_desc.ir_value(loc=loc, ip=ip)],
        'tcgen05.cp.cta_group::2.32x128b.warpx4 [$0], $1;',
        'r,l', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def tcgen05_commit_2cta(mbar_ptr: cute.Pointer, cta_mask: Uint16,
                        *, loc=None, ip=None) -> None:
    """Commit to both CTAs' barriers; the GC worker uses multicast mask 3."""
    llvm.inline_asm(
        None, [mbar_ptr.toint(loc=loc, ip=ip).ir_value(),
               cta_mask.ir_value(loc=loc, ip=ip)],
        'tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.'
        'multicast::cluster.b64 [$0], $1;',
        'r,h', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def tcgen05_after_thread_sync(*, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [], 'tcgen05.fence::after_thread_sync;', '',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def tcgen05_before_thread_sync(*, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [], 'tcgen05.fence::before_thread_sync;', '',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def tmem_load_16dp256b4(address: Uint32, *, loc=None, ip=None) -> tuple[Float32, Float32, Float32, Float32]:
    """Reference GC epilogue TMEM atom: four FP32 registers per lane."""
    values = llvm.inline_asm(
        ir.Type.parse('!llvm.struct<(i32,i32,i32,i32)>'),
        [address.ir_value(loc=loc, ip=ip)],
        'tcgen05.ld.sync.aligned.16x256b.x1.b32 '
        '{$0, $1, $2, $3}, [$4];',
        '=r,=r,=r,=r,r', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return tuple(Float32(llvm.bitcast(T.f32(),
                                    llvm.extractvalue(T.i32(), values, [i])))
                 for i in range(4))


@dsl_user_op
def tmem_wait_load(*, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [], 'tcgen05.wait::ld.sync.aligned;', '',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def tmem_free_2cta(address: Uint32, columns: Uint32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [address.ir_value(loc=loc, ip=ip),
                           columns.ir_value(loc=loc, ip=ip)],
                    'tcgen05.dealloc.cta_group::2.sync.aligned.b32 $0, $1;',
                    'r,r', has_side_effects=True,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def stmatrix_fp8x4_trans(dst_smem: cute.Pointer, packed_fp8x4: Uint32,
                        *, loc=None, ip=None) -> None:
    """Reference SM100_U8x4_STSM_T: one packed E4M3x4 word per lane.

    The caller supplies each lane's 16-byte-aligned shared address in the
    reference 64B-swizzled output tile. Warp participation and the subsequent
    shared-memory barrier/TMA fence belong to the epilogue scheduler.
    """
    llvm.inline_asm(
        None, [dst_smem.toint(loc=loc, ip=ip).ir_value(),
               packed_fp8x4.ir_value(loc=loc, ip=ip)],
        'stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [$0], {$1};',
        'r,r', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def stmatrix_fp8x4_trans_late(base_smem: cute.Pointer, byte_offset: Uint32,
                              packed_fp8x4: Uint32, *, loc=None, ip=None) -> None:
    """Form the W1 swizzled shared address in the STSM instruction region.

    The reference forms ``smem_ptr`` inside the FP8 store loop. Keeping the
    add and STSM in one PTX block keeps the summed address local to that
    block. The base and byte offset use the same shared address
    units as ``stmatrix_fp8x4_trans``.
    """
    llvm.inline_asm(
        None, [base_smem.toint(loc=loc, ip=ip).ir_value(),
               byte_offset.ir_value(loc=loc, ip=ip),
               packed_fp8x4.ir_value(loc=loc, ip=ip)],
        '{ .reg .b32 addr; add.u32 addr, $0, $1; '
        'stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [addr], {$2}; }',
        'r,r,r', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def stmatrix_bf16x8_trans(dst_smem: cute.Pointer,
                         packed_bf16x2_0: Uint32, packed_bf16x2_1: Uint32,
                         packed_bf16x2_2: Uint32, packed_bf16x2_3: Uint32,
                         *, loc=None, ip=None) -> None:
    """Reference SM90_U32x4_STSM_T: four packed BF16x2 words per lane.

    These are the reference's four register pairs after rounding and packing;
    the atom's lane/register-to-output bijection is established by the caller.
    """
    llvm.inline_asm(
        None, [dst_smem.toint(loc=loc, ip=ip).ir_value(),
               packed_bf16x2_0.ir_value(loc=loc, ip=ip),
               packed_bf16x2_1.ir_value(loc=loc, ip=ip),
               packed_bf16x2_2.ir_value(loc=loc, ip=ip),
               packed_bf16x2_3.ir_value(loc=loc, ip=ip)],
        'stmatrix.sync.aligned.x4.m8n8.shared.b16.trans '
        '[$0], {$1, $2, $3, $4};',
        'r,r,r,r,r', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def ld_shared_u32x4(src_smem: cute.Pointer, *, loc=None, ip=None) -> tuple[Uint32, Uint32, Uint32, Uint32]:
    """Read one 16-byte BF16 peer-store vector from the swizzled L2 tile."""
    values = llvm.inline_asm(
        ir.Type.parse('!llvm.struct<(i32,i32,i32,i32)>'),
        [src_smem.toint(loc=loc, ip=ip).ir_value()],
        'ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];',
        '=r,=r,=r,=r,r', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return tuple(Uint32(llvm.extractvalue(T.i32(), values, [i]))
                 for i in range(4))


@dsl_user_op
def st_global_u32x4(dst_gmem: cute.Pointer, value_0: Uint32,
                    value_1: Uint32, value_2: Uint32, value_3: Uint32,
                    *, loc=None, ip=None) -> None:
    """Write 16 BF16 bytes to an already-mapped, 16-byte-aligned peer slot.

    The caller owns rank mapping, row/channel offsets, and the release/acquire
    publication after all vector writes complete.
    """
    llvm.inline_asm(
        None, [dst_gmem.toint(loc=loc, ip=ip).ir_value(),
               value_0.ir_value(loc=loc, ip=ip),
               value_1.ir_value(loc=loc, ip=ip),
               value_2.ir_value(loc=loc, ip=ip),
               value_3.ir_value(loc=loc, ip=ip)],
        'st.global.v4.b32 [$0], {$1, $2, $3, $4};',
        'l,r,r,r,r', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)


@cute.jit
def warpgroup_reg_dealloc(reg_count: cutlass.Constexpr) -> None:
    """Reference warpgroup_reg_dealloc<N>, with a compile-time register count."""
    cute.arch.setmaxregister_decrease(reg_count)


@cute.jit
def warpgroup_reg_alloc(reg_count: cutlass.Constexpr) -> None:
    """Reference warpgroup_reg_alloc<N>, with a compile-time register count."""
    cute.arch.setmaxregister_increase(reg_count)
