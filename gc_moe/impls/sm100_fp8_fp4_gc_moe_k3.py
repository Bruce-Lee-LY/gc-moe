"""CuTeDSL kernels for the K3 front, router and persistent workers."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import math
from cutlass import BFloat16, Float32, Int32, Uint8, Uint16, Uint32, Uint64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm

from ..comm.gc_k3_barrier import (
    DEFAULT_NVLINK_BARRIER_TIMEOUT_SECS,
    grid_sync,
)
from ..common.math import (
    bf16x2_high_f32_bits,
    bf16x2_max,
    bf16x2_min,
    bits_float,
    f32_to_bf16_bits,
    float_bits,
    f32x2_to_bf16x2_bits,
    fast_exp_neg_f32,
    fast_rcp,
    abs_f32,
    f32x2_to_e4m3x2_bits,
    fmax_f32,
    pack_bf16_pair,
    ue8m0_fp8_inv,
    ue8m0_fp8_scale_exp,
    warp_group4_max,
    warp_inclusive_sum,
)
from ..common.math import fma_rn, tanh_approx
from ..layout.gc_moe_k3 import (
    ACCUM_BARRIER_STAGES,
    COMBINE_WARPS,
    COMPUTE_ROLE,
    DISPATCH_ROLE,
    FINALIZE_ROLE,
    WORKER_ROLES,
    FINALIZE_THREADS,
    PROJECTION_CHANNELS_PER_CTA,
    PROJECTION_K_TILE,
    align,
    GCK3BufferLayout,
    CANDIDATE_BLOCK_M,
    MAX_EXPERTS,
    MAX_EXPERTS_PER_RANK,
    MAX_RANKS,
    SIGNAL_OFFSETS,
    SMALL_LOCAL_EXPERT_TABLE_MAX,
)
from ..layout.sym_buffer import mapped_peer_base
from ..mma.sm100 import (
    MixedMmaTile,
    bf16_instr_desc_hi,
    blockscaled_instr_desc_hi,
    issue_mixed_k128,
    kmajor_smem_desc,
    kmajor_smem_desc_from_start,
    select_weight_format,
)
from ..ptx.ld_st import (
    EVICT_FIRST,
    EVICT_LAST,
    EVICT_NORMAL,
    atom_add_u32,
    atom_add_u64,
    atom_shared_add_u32,
    atom_sys_add_u64,
    bulk_zero_shared,
    ld_acquire_gpu_u32,
    ld_acquire_gpu_u64,
    ld_acquire_sys_u64,
    ld_global_u32x3,
    ld_global_u32x4,
    ld_global_u64,
    ld_shared_u32x4,
    ld_volatile_u32,
    ld_volatile_u64,
    red_gpu_add_u32,
    red_release_gpu_add_u32,
    red_release_gpu_xor_u64,
    st_global_u32x4,
    st_global_u64,
    st_release_gpu_u64,
    st_release_sys_u64,
    stmatrix_bf16x4_trans,
    stmatrix_bf16x8_trans,
    stmatrix_fp8x4_trans_late,
)
from ..ptx.tcgen05 import (
    mma_f16_2cta,
    tcgen05_commit_2cta,
    tcgen05_after_thread_sync,
    tcgen05_before_thread_sync,
    tmem_free_2cta,
    tmem_load_16dp256b4,
)
from ..ptx.tma import (
    bulk_g2s_1d,
    bulk_s2g_1d,
    generic_tensormap_ptr,
    prefetch_tensormap,
    tile_s2g_2d,
    tile_g2s_2cta_2d,
    tile_g2s_2cta_4d,
    tma_store_arrive,
    tma_store_fence,
    tma_store_wait_0,
    tma_store_wait_1,
)
from ..ptx.utils import (
    assume_u32_lt,
    device_trap,
    elect_one_sync,
    fence_acq_rel_sys,
    read_clock64,
    sync_aligned,
    undefined_u32,
    warp_nth_set_bit,
    warpgroup_reg_alloc,
    warpgroup_reg_dealloc,
)
from ..scheduler.gc_moe_k3 import (
    GCK3BlockPhase,
    TASK_EMPTY_CONSUMERS,
    advance_expert,
    release_task_info,
    schedule_tasks,
    claim_combine_row,
)

# Zero-row ranks still clear the routed rings and publish peer completion.
ZERO_TOKEN_FINALIZE_SMS = 64


PROJECTION_EPILOGUE_WARPS = 4


@dsl_user_op
def _load_accumulator(address: Uint32, *, loc=None, ip=None):
    values = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32,i32,i32,i32,i32,i32,i32,i32)>"),
        [address.ir_value(loc=loc, ip=ip)],
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 {$0,$1,$2,$3,$4,$5,$6,$7}, [$8]; "
        "tcgen05.wait::ld.sync.aligned;",
        "=r,=r,=r,=r,=r,=r,=r,=r,r",
        has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return tuple(Float32(llvm.bitcast(T.f32(), llvm.extractvalue(T.i32(), values, [i])))
                 for i in range(8))


@dsl_user_op
def _fence_projection_loads(*, loc=None, ip=None):
    llvm.inline_asm(None, [], 'fence.proxy.async.global;', '~{memory}',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT,
                    loc=loc, ip=ip)


@dsl_user_op
def _try_wait_finalize_phase(barrier: cute.Pointer, phase: Uint32, *, loc=None, ip=None) -> Uint32:
    complete = llvm.inline_asm(
        T.i32(), [barrier.toint(loc=loc, ip=ip).ir_value(), phase.ir_value(loc=loc, ip=ip)],
        '{ .reg .pred p; mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 '
        'p, [$1], $2; selp.u32 $0, 1, 0, p; }',
        '=r,r,r,~{memory}', has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip)
    return Uint32(complete)


@cute.jit
def _wait_finalize_phase(barrier: cute.Pointer, phase: Uint32) -> None:
    # The MMA leader waits on CTA0's empty barrier after both epilogues release it.
    while _try_wait_finalize_phase(barrier, phase) == Uint32(0):
        pass


@cute.jit
def _arrive_finalize_phase(barrier: cute.Pointer, peer: cutlass.Constexpr) -> None:
    cute.arch.mbarrier_arrive(barrier, peer_cta_rank_in_cluster=peer,
                              scope=nvvm.MemScopeKind.CLUSTER)


@cute.jit
def _projection_tile_geometry(projection: Uint32, channel_tile: Uint32,
                              cta_rank: Uint32,
                              k_tiles: cutlass.Constexpr, tile_rows: cutlass.Constexpr,
                              data_stages: cutlass.Constexpr, channels_per_cta: cutlass.Constexpr):
    data_phase = (projection * Uint32(k_tiles // data_stages)) & Uint32(1)
    accum_stage = projection & Uint32(1)
    accum_phase = (projection // Uint32(2)) & Uint32(1)
    channel = (channel_tile * Uint32(2) + cta_rank) * Uint32(channels_per_cta)
    return channel, data_phase, accum_stage, accum_phase, accum_stage * Uint32(tile_rows)


@cute.jit
def _issue_projection_weight(weight_smem: cute.Pointer, weight_map: cute.Pointer,
                             data_full: cute.Pointer, data_empty: cute.Pointer,
                             stage: Uint32, parity: Uint32, k: Uint32, channel: Uint32,
                             cta_rank: Uint32, weight_bytes: cutlass.Constexpr,
                             latent_bytes: cutlass.Constexpr, hint: Uint64):
    cute.arch.mbarrier_wait(data_empty + stage, parity ^ Uint32(1))
    if cta_rank == Uint32(0):
        cute.arch.mbarrier_arrive_and_expect_tx(data_full + stage, 2 * (weight_bytes + latent_bytes))
    else:
        cute.arch.mbarrier_arrive(data_full + stage, peer_cta_rank_in_cluster=0)
    tile_g2s_2cta_2d(weight_smem + stage * weight_bytes, weight_map, data_full + stage,
                     k * Uint32(PROJECTION_K_TILE), channel, hint)


@cute.jit
def _project_tile_2cta(
    slab: cute.Pointer, data_full: cute.Pointer, data_empty: cute.Pointer,
    accum_full: cute.Pointer, accum_empty: cute.Pointer, tmem_ptr: cute.Pointer,
    row_ready: cute.Pointer,
    weight_map: cute.Pointer, latent_map: cute.Pointer, out: cute.Tensor,
    shared_output: cute.Tensor, prefix: cute.Tensor, token_start: Uint32, channel_tile: Uint32,
    num_tokens: Uint32, ready_epoch: Uint64, projection: Uint32,
    cta_rank: Uint32, warp: Uint32, lane: Uint32, tile_rows: cutlass.Constexpr,
    hidden: cutlass.Constexpr, add3: cutlass.Constexpr, data_stages: cutlass.Constexpr,
    channels_per_cta: cutlass.Constexpr, parts: cute.Pointer,
    split: Uint32, splits: cutlass.Constexpr, out_hidden: cutlass.Constexpr,
    full_split_k_tile: cutlass.Constexpr,
):
    """Pipeline one tile through the TMA ring and the selected TMEM slot."""
    weight_bytes = channels_per_cta * PROJECTION_K_TILE * 2
    latent_bytes = tile_rows // 2 * PROJECTION_K_TILE * 2
    weight_smem = slab
    latent_smem = slab + data_stages * weight_bytes
    k_tiles = hidden // (PROJECTION_K_TILE * splits)
    k_start = split * Uint32(k_tiles)
    if cutlass.const_expr(full_split_k_tile):
        weight_hint = Uint64(EVICT_NORMAL)
        if num_tokens > Uint32(tile_rows):
            weight_hint = Uint64(EVICT_LAST)
    else:
        weight_hint = Uint64(EVICT_LAST if tile_rows > 32 else EVICT_FIRST)
        if cutlass.const_expr(tile_rows == 64 and splits == 2):
            weight_hint = Uint64(EVICT_FIRST)
    if warp == Uint32(0):
        first_channel, start_phase, _, _, _ = _projection_tile_geometry(
            projection, channel_tile, cta_rank, k_tiles, tile_rows, data_stages, channels_per_cta)
        # Prefill the weight stages while the combine warps produce this tile.
        # The transaction count also reserves the latent bytes, so no MMA
        # sees a full stage before the matching activation arrives.
        with cute.arch.elect_one():
            for stage in cutlass.range_constexpr(data_stages):
                _issue_projection_weight(
                    weight_smem, weight_map, data_full, data_empty,
                    Uint32(stage), start_phase, k_start + Uint32(stage), first_channel, cta_rank,
                    weight_bytes, latent_bytes, weight_hint)

        row = token_start + cta_rank * Uint32(tile_rows // 2) + lane
        for half in cutlass.range_constexpr((tile_rows + 63) // 64):
            tile_row = row + Uint32(half * 32)
            if lane + Uint32(half * 32) < Uint32(tile_rows // 2) and tile_row < num_tokens:
                while ld_acquire_gpu_u64(row_ready + tile_row) != ready_epoch:
                    pass
        cute.arch.sync_warp()
        _fence_projection_loads()
        with cute.arch.elect_one():
            for k in cutlass.range(k_tiles):
                stage = Uint32(k % data_stages)
                if k >= data_stages:
                    parity = start_phase ^ Uint32((k // data_stages) % 2)
                    _issue_projection_weight(
                        weight_smem, weight_map, data_full, data_empty, stage, parity,
                        k_start + Uint32(k), first_channel, cta_rank, weight_bytes, latent_bytes, weight_hint)
                tile_g2s_2cta_2d(latent_smem + stage * latent_bytes, latent_map,
                                 data_full + stage, (k_start + Uint32(k)) * Uint32(PROJECTION_K_TILE),
                                 token_start + cta_rank * Uint32(tile_rows // 2), Uint64(EVICT_LAST))
    elif warp == Uint32(1) and cta_rank == Uint32(0):
        instr = bf16_instr_desc_hi(Uint32(tile_rows), 2 * channels_per_cta)
        _, start_phase, accum_stage, accum_phase, accum_col = _projection_tile_geometry(
            projection, channel_tile, cta_rank, k_tiles, tile_rows, data_stages, channels_per_cta)
        accum_addr = tmem_ptr[0] + accum_col
        with cute.arch.elect_one():
            _wait_finalize_phase(accum_empty + accum_stage, accum_phase ^ Uint32(1))
            tcgen05_after_thread_sync()
            for k in cutlass.range(k_tiles):
                stage = Uint32(k % data_stages)
                parity = start_phase ^ Uint32((k // data_stages) % 2)
                cute.arch.mbarrier_wait(data_full + stage, parity)
                tcgen05_after_thread_sync()
                for kk in cutlass.range_constexpr(PROJECTION_K_TILE // 16):
                    w = kmajor_smem_desc(weight_smem, Uint32(1024), True,
                                         stage * Uint32(weight_bytes) + Uint32(kk * 32))
                    x = kmajor_smem_desc(latent_smem, Uint32(1024), True,
                                         stage * Uint32(latent_bytes) + Uint32(kk * 32))
                    mma_f16_2cta(w, x, accum_addr, Uint32(k > 0 or kk > 0), instr)
                tcgen05_commit_2cta(data_empty + stage, Uint16(3))
                if k == k_tiles - 1:
                    tcgen05_commit_2cta(accum_full + accum_stage, Uint16(3))
    elif warp >= Uint32(4) and warp < Uint32(4 + PROJECTION_EPILOGUE_WARPS):
        first_channel, _, accum_stage, accum_phase, accum_col = _projection_tile_geometry(
            projection, channel_tile, cta_rank, k_tiles, tile_rows, data_stages, channels_per_cta)
        accum_addr = tmem_ptr[0] + accum_col
        cute.arch.mbarrier_wait(accum_full + accum_stage, accum_phase)
        tcgen05_after_thread_sync()
        epilogue_warp = warp - Uint32(4)
        address = accum_addr + (epilogue_warp << Uint32(21))
        channel = first_channel + epilogue_warp * Uint32(32) + lane
        first_row = token_start
        rows_per_warp = tile_rows
        if cutlass.const_expr(channels_per_cta == 64):
            # tcgen05 Layout B uses two channel groups and two token
            # groups in each CTA's four TMEM datapath groups.
            channel = first_channel + (epilogue_warp & Uint32(1)) * Uint32(32) + lane
            first_row += (epilogue_warp >> Uint32(1)) * Uint32(tile_rows // 2)
            rows_per_warp = tile_rows // 2
        for offset in cutlass.range(0, rows_per_warp, 8):
            values = _load_accumulator(address + Uint32(offset))
            if offset == rows_per_warp - 8:
                # The last load already waited for TMEM. The next MMA
                # may now reuse the slot while we finish the final stores.
                tcgen05_before_thread_sync()
                cute.arch.sync_warp()
                if lane == Uint32(0):
                    _arrive_finalize_phase(accum_empty + accum_stage, 0)
            for index in cutlass.range_constexpr(8):
                row = first_row + Uint32(offset + index)
                if row < num_tokens:
                    val = values[index]
                    if cutlass.const_expr(splits > 1):
                        parts[(row * Uint32(splits) + split) * Uint32(out_hidden) + channel] = val
                    else:
                        shared = shared_output[row, channel].to(Float32)
                        if cutlass.const_expr(add3):
                            val = (val.to(BFloat16).to(Float32) + shared).to(BFloat16).to(Float32)
                            val += prefix[row, channel].to(Float32)
                        else:
                            val += shared
                        out[row, channel] = val.to(BFloat16)
        cute.arch.sync_warp()


class GCK3ProjectionReduceKernel:
    def __init__(self, layout, rows, has_prefix):
        self.rows = rows
        self.width = layout.shared_hidden
        self.has_prefix = has_prefix
        self.partials_offset = layout.views['projection_partials'].offset
        self.shared_offset = layout.views['shared_output'].offset
        # Each thread reduces four adjacent channels of an aligned BF16 row.
        per_block = 256 * 4
        self.grid = (rows * self.width + per_block - 1) // per_block

    @cute.kernel
    def kernel(self, workspace: cute.Tensor, out: cute.Tensor, prefix: cute.Tensor):
        block, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        element = Uint32((block * 256 + thread) * 4)
        if element < Uint32(self.rows * self.width):
            row = element // Uint32(self.width)
            col = element % Uint32(self.width)
            partials = cute.recast_ptr(workspace.iterator + Uint64(self.partials_offset), dtype=Float32)
            shared = cute.recast_ptr(workspace.iterator + Uint64(self.shared_offset), dtype=BFloat16)
            first = row * Uint32(2 * self.width) + col
            left = ld_global_u32x4(cute.recast_ptr(partials + first, dtype=Uint32))
            right = ld_global_u32x4(cute.recast_ptr(
                partials + first + Uint32(self.width), dtype=Uint32))
            se = ld_global_u64(cute.recast_ptr(shared + element, dtype=Uint64))
            pref = Uint64(0)
            if cutlass.const_expr(self.has_prefix):
                pref = ld_global_u64(cute.recast_ptr(prefix.iterator + element, dtype=Uint64))
            packed = Uint64(0)
            for i in cutlass.range_constexpr(4):
                addend = bits_float(Uint32((se >> (16*i)) & Uint64(0xffff)) << 16)
                val = bits_float(left[i]) + bits_float(right[i])
                if cutlass.const_expr(self.has_prefix):
                    p = bits_float(Uint32((pref >> (16*i)) & Uint64(0xffff)) << 16)
                    val = (val.to(BFloat16).to(Float32) + addend).to(BFloat16).to(Float32)
                    val += p
                else:
                    val += addend
                packed |= Uint64(f32_to_bf16_bits(val)) << (16*i)
            st_global_u64(cute.recast_ptr(out.iterator + element, dtype=Uint64), packed)

    @cute.jit
    def __call__(self, workspace: cute.Tensor, out: cute.Tensor, prefix: cute.Tensor):
        self.kernel(workspace, out, prefix).launch(grid=(self.grid, 1, 1), block=(256, 1, 1))


FRONT_DATA_STAGES = 8


# The same swapAB 2CTA mapping used by the final projection puts one
# contiguous K32 FP8 scale group in each epilogue warp.
class GCK3FrontKernel:
    def __init__(self, layout, tokens, *, available_sms, smem_budget):
        self.tokens = tokens
        self.experts = layout.num_experts
        self.hidden = layout.hidden
        self.input_hidden = layout.shared_hidden
        self.rows = 16 if tokens <= 32 else 32 if tokens <= 64 else 64 if tokens <= 192 else 128
        self.channels_per_cta = PROJECTION_CHANNELS_PER_CTA
        output_columns = self.experts + self.hidden
        narrow_channels = PROJECTION_CHANNELS_PER_CTA // 2
        narrow_grid = 2 * ((output_columns + 2 * narrow_channels - 1) // (2 * narrow_channels))
        # A single minimum-M tile can use narrower channel tiles when the
        # narrow grid fits. Larger M tiles must be more than half full.
        row_tile_choices = (16, 32, 64, 128)
        for rows in row_tile_choices:
            row_tiles = (tokens + rows - 1) // rows
            single_minimum_tile = row_tiles == 1 and rows == row_tile_choices[0]
            if ((row_tiles == 1 or rows == row_tile_choices[-1])
                    and narrow_grid * row_tiles <= available_sms
                    and (single_minimum_tile or 2 * tokens > row_tiles * rows)):
                self.rows = rows
                self.channels_per_cta = narrow_channels
                break
        self.row_tiles = (tokens + self.rows - 1) // self.rows
        channels = 2 * self.channels_per_cta
        self.grid = 2 * self.row_tiles * ((self.experts + self.hidden + channels - 1) // channels)
        self.logits_offset = layout.views['front_logits'].offset
        self.x_offset = layout.views['x'].offset
        self.sf_offset = layout.views['x_sf'].offset
        self.weight_bytes = self.channels_per_cta * PROJECTION_K_TILE * 2
        self.input_bytes = self.rows // 2 * PROJECTION_K_TILE * 2
        self.early_shared = False
        k_slices = self.input_hidden // PROJECTION_K_TILE
        # Each stage adds one full and one empty barrier. Reserve another
        # barrier and an aligned TMEM pointer after the data ring.
        bytes_per_stage = self.weight_bytes + self.input_bytes + 16
        stages_fit = min(k_slices, (smem_budget - 16) // bytes_per_stage)
        self.stages = min(FRONT_DATA_STAGES, stages_fit)
        # Deepen a single row tile or fully occupied tiles if the grid fits
        # in one wave. Cap the increased depth at half the K slices so
        # the deeper ring can complete at least two traversals.
        if self.grid <= available_sms and (self.row_tiles == 1 or tokens % self.rows == 0):
            self.stages = max(self.stages, min(16, stages_fit, k_slices // 2))
        if self.stages < 2:
            raise ValueError('K3 Front needs shared memory for at least two TMA stages')
        self.bar_offset = self.stages * (self.weight_bytes + self.input_bytes)
        self.smem_bytes = 16 + self.stages * bytes_per_stage

    @cute.kernel
    def kernel(self, workspace: cute.Tensor, weight_map: cute.Tensor, input_map: cute.Tensor):
        row_tiles = self.row_tiles
        rows = self.rows
        smem_bytes = self.smem_bytes
        bar_offset = self.bar_offset
        weight_bytes = self.weight_bytes
        input_hidden = self.input_hidden
        input_bytes = self.input_bytes
        experts = self.experts
        hidden = self.hidden
        tokens = self.tokens
        logits_offset = self.logits_offset
        x_offset = self.x_offset
        sf_offset = self.sf_offset
        block, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_in_cluster_idx()
        cta = Uint32(cta)
        warp, lane = Uint32(tid // 32), Uint32(tid % 32)
        job = Uint32(block // 2)
        token_start = (job % Uint32(row_tiles)) * Uint32(rows)
        channel = (job // Uint32(row_tiles) * Uint32(2) + cta) * Uint32(self.channels_per_cta)
        allocator = cutlass.utils.SmemAllocator()
        slab = cute.recast_ptr(allocator.allocate(smem_bytes, 1024), dtype=Uint8)
        data_full = cute.recast_ptr(slab + bar_offset, dtype=Uint64)
        data_empty = data_full + self.stages
        accumulator_full = data_empty + self.stages
        tmem = cute.recast_ptr(accumulator_full + 1, dtype=Uint32)
        activation = slab + self.stages * weight_bytes
        if warp == 0:
            if lane < self.stages:
                if cta == 0:
                    cute.arch.mbarrier_init(data_full + lane, 2)
                cute.arch.mbarrier_init(data_empty + lane, 1)
            if lane == 0:
                cute.arch.mbarrier_init(accumulator_full, 1)
            cute.arch.mbarrier_init_fence()
        if warp == 2:
            cute.arch.alloc_tmem(2 * rows, tmem, is_two_cta=True)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        k_tiles = input_hidden // PROJECTION_K_TILE
        hint = Uint64(EVICT_FIRST)
        if cutlass.const_expr(row_tiles > 1):
            hint = Uint64(EVICT_LAST)
        if warp == 0:
            with cute.arch.elect_one():
                for k in cutlass.range(k_tiles):
                    stage = Uint32(k % self.stages)
                    phase = Uint32((k // self.stages) & 1)
                    _issue_projection_weight(
                        slab, weight_map.iterator, data_full, data_empty, stage, phase,
                        Uint32(k), channel, cta, weight_bytes, input_bytes, hint)
                    tile_g2s_2cta_2d(activation + stage * input_bytes,
                        input_map.iterator, data_full + stage, Uint32(k * PROJECTION_K_TILE),
                        token_start + cta * Uint32(rows // 2), Uint64(EVICT_LAST))
                    if cutlass.const_expr(self.early_shared):
                        # Request early Compute launch once this CTA has
                        # issued two rounds of TMA loads into the ring.
                        if k == 2 * self.stages - 1:
                            cute.arch.griddepcontrol_launch_dependents()
        elif warp == 1 and cta == 0:
            with cute.arch.elect_one():
                instr = bf16_instr_desc_hi(Uint32(rows), 2 * self.channels_per_cta)
                acc = tmem[0]
                for k in cutlass.range(k_tiles):
                    stage = Uint32(k % self.stages)
                    phase = Uint32((k // self.stages) & 1)
                    cute.arch.mbarrier_wait(data_full + stage, phase)
                    tcgen05_after_thread_sync()
                    for kk in cutlass.range_constexpr(PROJECTION_K_TILE // 16):
                        w = kmajor_smem_desc(slab, Uint32(1024), True,
                            stage * Uint32(weight_bytes) + Uint32(kk * 32))
                        x = kmajor_smem_desc(activation, Uint32(1024), True,
                            stage * Uint32(input_bytes) + Uint32(kk * 32))
                        mma_f16_2cta(w, x, acc, Uint32(k > 0 or kk > 0), instr)
                    tcgen05_commit_2cta(data_empty + stage, Uint16(3))
                tcgen05_commit_2cta(accumulator_full, Uint16(3))
        elif warp >= 4:
            cute.arch.mbarrier_wait(accumulator_full, Uint32(0))
            tcgen05_after_thread_sync()
            acc = tmem[0] + ((warp - Uint32(4)) << Uint32(21))
            epilogue_warp = warp - Uint32(4)
            group_channel = channel + epilogue_warp * Uint32(32)
            row_base = token_start
            rows_per_warp = rows
            if cutlass.const_expr(self.channels_per_cta == 64):
                # Dense M128 2CTA layout places the second token half in
                # datapath groups 2 and 3 for the same channels.
                group_channel = channel + (epilogue_warp & Uint32(1)) * Uint32(32)
                row_base += (epilogue_warp >> Uint32(1)) * Uint32(rows // 2)
                rows_per_warp = rows // 2
            ch = group_channel + lane
            base = workspace.iterator
            if group_channel < Uint32(experts):
                logits = cute.recast_ptr(base + logits_offset, dtype=Float32)
                for first in cutlass.range(0, rows_per_warp, 8):
                    values = _load_accumulator(acc + Uint32(first))
                    first_row = row_base + Uint32(first)
                    first_index = first_row * Uint32(experts) + ch
                    for i in cutlass.range_constexpr(8):
                        row = first_row + Uint32(i)
                        if cutlass.const_expr(tokens % rows == 0) or row < Uint32(tokens):
                            logits[first_index + Uint32(i * experts)] = values[i]
            elif group_channel < Uint32(experts + hidden):
                ch_in_x = ch - Uint32(experts)
                for first in cutlass.range(0, rows_per_warp, 8):
                    values = _load_accumulator(acc + Uint32(first))
                    first_row = row_base + Uint32(first)
                    first_x = first_row * Uint32(hidden) + ch_in_x
                    first_sf = first_row * Uint32(hidden // 32) + ch_in_x // Uint32(32)
                    for i in cutlass.range_constexpr(8):
                        row = first_row + Uint32(i)
                        if cutlass.const_expr(tokens % rows == 0) or row < Uint32(tokens):
                            val = values[i]
                            bf = val.to(BFloat16).to(Float32)
                            amax = abs_f32(bf)
                            for step in cutlass.range_constexpr(5):
                                amax = fmax_f32(amax, cute.arch.shuffle_sync_bfly(amax, 16 >> step))
                            raw_sf = fmax_f32(amax, Float32(1e-10)) * Float32(1.0 / 448.0)
                            sf = (float_bits(raw_sf) + Uint32(0x7fffff)) >> Uint32(23)
                            fp8 = f32x2_to_e4m3x2_bits(bf * ue8m0_fp8_inv(sf), Float32(0))
                            (base + x_offset)[first_x + Uint32(i * hidden)] = Uint8(fp8)
                            if lane == 0:
                                (base + sf_offset)[first_sf + Uint32(i * (hidden // 32))] = Uint8(sf)
            tcgen05_before_thread_sync()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        if warp == 2:
            tmem_free_2cta(tmem[0], Uint32(2 * rows))

    @cute.jit
    def __call__(self, workspace: cute.Tensor, weight_map: cute.Tensor, input_map: cute.Tensor):
        self.kernel(workspace, weight_map, input_map).launch(
            grid=(self.grid, 1, 1), block=(256, 1, 1), cluster=(2, 1, 1), min_blocks_per_mp=1)


@cute.jit
def _router_sigmoid(logit: Float32) -> Float32:
    return Float32(1.0) / (Float32(1.0) + cute.math.exp(-logit, fastmath=False))


@cute.jit
def _router_select(keys: cute.Tensor, ids: cute.Tensor, count: cutlass.Constexpr):
    # Initialize from a key rather than a literal zero. The float-sort
    # encoding uses bit 31, so every local comparison must stay unsigned.
    local_key = keys[0]
    for i in cutlass.range_constexpr(1, count):
        other = keys[i]
        if other > local_key:
            local_key = other
    key = cute.arch.warp_redux_sync(local_key, 'max')
    local_id = Uint32(0xffffffff)
    for i in cutlass.range_constexpr(count):
        if keys[i] == key:
            local_id = min(local_id, ids[i])
    expert = cute.arch.warp_redux_sync(local_id, 'min')
    for i in cutlass.range_constexpr(count):
        if ids[i] == expert:
            keys[i] = Uint32(0)
            ids[i] = Uint32(0xffffffff)
    return expert


class GCK3RouterKernel:
    def __init__(self, layout, tokens, renormalize, padded):
        self.tokens = tokens
        self.experts = layout.num_experts
        self.topk = layout.topk
        self.per_lane = (self.experts + 255) // 256
        self.logits_offset = layout.views['front_logits'].offset
        self.idx_offset = layout.views['topk_idx'].offset
        self.weight_offset = layout.views['topk_weights'].offset
        self.renormalize = renormalize
        self.padded = padded
        self.smem_bytes = (8 * 2 + 2) * self.topk * 4

    @cute.kernel
    def kernel(self, workspace: cute.Tensor, router_bias: cute.Tensor, valid_token_count):
        logits_offset = self.logits_offset
        experts = self.experts
        idx_offset = self.idx_offset
        topk = self.topk
        weight_offset = self.weight_offset
        smem_bytes = self.smem_bytes
        padded = self.padded
        per_lane = self.per_lane
        renormalize = self.renormalize
        row, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        warp, lane = Uint32(tid // 32), Uint32(tid % 32)
        base = workspace.iterator
        logits = cute.recast_ptr(base + logits_offset, dtype=Float32) + Uint32(row * experts)
        out_idx = cute.recast_ptr(base + idx_offset, dtype=Uint64) + Uint32(row * topk)
        out_weight = cute.recast_ptr(base + weight_offset, dtype=Float32) + Uint32(row * topk)
        allocator = cutlass.utils.SmemAllocator()
        candidate_keys = cute.recast_ptr(allocator.allocate(smem_bytes, 16), dtype=Uint32)
        candidate_ids = candidate_keys + 8 * topk
        selected_ids = candidate_ids + 8 * topk
        selected_weights = cute.recast_ptr(selected_ids + topk, dtype=Float32)
        valid = True
        if cutlass.const_expr(padded):
            valid = Int32(row) < valid_token_count[0]
        if valid:
            keys = cute.make_rmem_tensor((per_lane,), Uint32)
            ids = cute.make_rmem_tensor((per_lane,), Uint32)
            # Each thread reads the same position in successive 256-expert panels.
            for i in cutlass.range_constexpr(per_lane):
                expert = Uint32(i * 256) + Uint32(tid)
                keys[i] = Uint32(0)
                ids[i] = Uint32(0xffffffff)
                if expert < Uint32(experts):
                    score = _router_sigmoid(logits[expert]) + router_bias[expert]
                    if score != score:
                        score = Float32(-1e30)
                    # Canonicalize signed zero before building the integer sort key.
                    if score == Float32(0):
                        score = Float32(0)
                    bits = float_bits(score)
                    if (bits & Uint32(0x80000000)) != 0:
                        keys[i] = ~bits
                    else:
                        keys[i] = bits | Uint32(0x80000000)
                    ids[i] = expert
            # Each lane loads experts in increasing ID order. Swapping only on a
            # higher score preserves that order for ties.
            for end in cutlass.range_constexpr(per_lane - 1):
                for off in cutlass.range_constexpr(per_lane - 1 - end):
                    ak = keys[off]
                    ai = ids[off]
                    bk = keys[off + 1]
                    bi = ids[off + 1]
                    take = bk > ak
                    keys[off] = bk if take else ak
                    keys[off + 1] = ak if take else bk
                    ids[off] = bi if take else ai
                    ids[off + 1] = ai if take else bi
            for i in cutlass.range(topk):
                key = cute.arch.warp_redux_sync(keys[0], 'max')
                cand = ids[0] if keys[0] == key else Uint32(0xffffffff)
                eid = cute.arch.warp_redux_sync(cand, 'min')
                if ids[0] == eid:
                    for off in cutlass.range_constexpr(per_lane - 1):
                        keys[off] = keys[off + 1]
                        ids[off] = ids[off + 1]
                    keys[per_lane - 1] = Uint32(0)
                    ids[per_lane - 1] = Uint32(0xffffffff)
                if lane == 0:
                    candidate_keys[warp * Uint32(topk) + Uint32(i)] = key
                    candidate_ids[warp * Uint32(topk) + Uint32(i)] = eid
            cute.arch.sync_threads()
            if warp == 0:
                keys2 = cute.make_rmem_tensor(((8 * topk + 31) // 32,), Uint32)
                ids2 = cute.make_rmem_tensor(((8 * topk + 31) // 32,), Uint32)
                for i in cutlass.range_constexpr((8 * topk + 31) // 32):
                    offset = Uint32(i * 32) + lane
                    keys2[i] = Uint32(0)
                    ids2[i] = Uint32(0xffffffff)
                    if offset < 8 * topk:
                        keys2[i] = candidate_keys[offset]
                        ids2[i] = candidate_ids[offset]
                for i in cutlass.range(topk):
                    eid = _router_select(keys2, ids2, (8 * topk + 31) // 32)
                    if lane == Uint32(i):
                        selected_ids[i] = eid
                cute.arch.sync_warp()
                if lane < topk:
                    eid = selected_ids[lane]
                    slot = Uint32(0)
                    for i in cutlass.range_constexpr(topk):
                        slot += Uint32(selected_ids[i] < eid)
                    out_idx[slot] = Uint64(eid)
                    selected_weights[slot] = _router_sigmoid(logits[eid])
                cute.arch.sync_warp()
                if lane < topk:
                    weight = selected_weights[lane]
                    if cutlass.const_expr(renormalize):
                        total = Float32(0)
                        for i in cutlass.range_constexpr(topk):
                            total += selected_weights[i]
                        if total > Float32(0):
                            weight /= total
                    out_weight[lane] = weight
        elif tid < topk:
            out_idx[tid] = Uint64(0xffffffffffffffff)
            out_weight[tid] = Float32(0)

    @cute.jit
    def __call__(self, workspace: cute.Tensor, router_bias: cute.Tensor, valid_token_count=None):
        self.kernel(workspace, router_bias, valid_token_count).launch(
            grid=(self.tokens, 1, 1), block=(256, 1, 1))


@cute.jit
def _situ_pair(
    gate0_accum: Float32, gate1_accum: Float32,
    up0_accum: Float32, up1_accum: Float32,
    clamp_value: Float32, clamp_enabled: cutlass.Constexpr,
    beta: cutlass.Constexpr, linear_beta: cutlass.Constexpr, fast_math: cutlass.Constexpr,
):
    """W1 BF16 rounding followed by explicit FP32 SiTU arithmetic."""
    gb = f32x2_to_bf16x2_bits(gate0_accum, gate1_accum)
    ub = f32x2_to_bf16x2_bits(up0_accum, up1_accum)
    if cutlass.const_expr(clamp_enabled):
        bound = Uint32(f32_to_bf16_bits(clamp_value))
        bounds = bound | (bound << Uint32(16))
        gb = bf16x2_min(gb,bounds)
        ub = bf16x2_min(bf16x2_max(ub,bounds ^ Uint32(0x80008000)),bounds)
    g0 = bits_float((gb & Uint32(0xffff)) << Uint32(16))
    g1 = bits_float(bf16x2_high_f32_bits(gb))
    u0 = bits_float((ub & Uint32(0xffff)) << Uint32(16))
    u1 = bits_float(bf16x2_high_f32_bits(ub))
    e0, e1 = Float32(0), Float32(0)
    if cutlass.const_expr(fast_math):
        e0, e1 = fast_exp_neg_f32(g0), fast_exp_neg_f32(g1)
    else:
        e0, e1 = cute.math.exp(-g0,fastmath=False), cute.math.exp(-g1,fastmath=False)
    d0,d1 = cute.arch.add_packed_f32x2((Float32(1),Float32(1)),(e0,e1),rnd='rn',ftz=False)
    sig0,sig1 = Float32(0),Float32(0)
    if cutlass.const_expr(fast_math):
        sig0,sig1 = fast_rcp(d0),fast_rcp(d1)
    else:
        sig0 = cute.math.div(Float32(1),d0,fastmath=False,ftz=False,rounding=cute.math.RoundingMode.NEAREST_EVEN)
        sig1 = cute.math.div(Float32(1),d1,fastmath=False,ftz=False,rounding=cute.math.RoundingMode.NEAREST_EVEN)
    z0 = cute.math.div(g0,Float32(beta),fastmath=False,ftz=False,rounding=cute.math.RoundingMode.NEAREST_EVEN)
    z1 = cute.math.div(g1,Float32(beta),fastmath=False,ftz=False,rounding=cute.math.RoundingMode.NEAREST_EVEN)
    t0,t1 = Float32(0),Float32(0)
    if cutlass.const_expr(fast_math):
        t0,t1 = tanh_approx(z0),tanh_approx(z1)
    else:
        t0,t1 = cute.math.tanh(z0,fastmath=False),cute.math.tanh(z1,fastmath=False)
    g0,g1 = cute.arch.mul_packed_f32x2((t0,t1),(Float32(beta),Float32(beta)),rnd='rn',ftz=False)
    if cutlass.const_expr(linear_beta is not None):
        z0 = cute.math.div(u0,Float32(linear_beta),fastmath=False,ftz=False,rounding=cute.math.RoundingMode.NEAREST_EVEN)
        z1 = cute.math.div(u1,Float32(linear_beta),fastmath=False,ftz=False,rounding=cute.math.RoundingMode.NEAREST_EVEN)
        if cutlass.const_expr(fast_math):
            t0,t1 = tanh_approx(z0),tanh_approx(z1)
        else:
            t0,t1 = cute.math.tanh(z0,fastmath=False),cute.math.tanh(z1,fastmath=False)
        u0,u1 = cute.arch.mul_packed_f32x2((t0,t1),(Float32(linear_beta),Float32(linear_beta)),rnd='rn',ftz=False)
    g0,g1 = cute.arch.mul_packed_f32x2((sig0,sig1),(g0,g1),rnd='rn',ftz=False)
    return cute.arch.mul_packed_f32x2((g0,g1),(u0,u1),rnd='rn',ftz=False)


@cute.jit
def sync_epilogue_warpgroup(warpgroup: Uint32) -> None:
    """Reference WG-local barrier ID 3 + warpgroup, with 128 callers."""
    sync_aligned(128, Uint32(3) + warpgroup)


@cute.jit
def _load_w2_atom(raw_values: cute.Tensor, buffer: cutlass.Constexpr,
                  atom: cutlass.Constexpr, address: Uint32) -> None:
    """Keep both 16dp halves of one 8-row W2 atom in registers."""
    first = tmem_load_16dp256b4(address)
    second = tmem_load_16dp256b4(address | Uint32(0x00100000))
    for register in cutlass.range_constexpr(4):
        raw_values[buffer, atom, register] = first[register]
        raw_values[buffer, atom, register + 4] = second[register]


@cute.jit
def _release_w2_tmem(accum_empty: cute.Pointer, stage: Uint32) -> None:
    tcgen05_before_thread_sync()
    cute.arch.mbarrier_arrive(accum_empty + stage, peer_cta_rank_in_cluster=0)


@cute.jit
def _load_w1_atom(raw_values: cute.Tensor, buffer: cutlass.Constexpr,
                  atom: cutlass.Constexpr, address: Uint32) -> None:
    """Source-contract 16dp W1 atom: gate/up for two post channels."""
    first = tmem_load_16dp256b4(address)
    second = tmem_load_16dp256b4(address | Uint32(0x00100000))
    for register in cutlass.range_constexpr(4):
        raw_values[buffer, atom, register] = first[register]
        raw_values[buffer, atom, register + 4] = second[register]


@cute.jit
def _release_w1_tmem(accum_empty: cute.Pointer, stage: Uint32) -> None:
    tcgen05_before_thread_sync()
    cute.arch.mbarrier_arrive(accum_empty + stage, peer_cta_rank_in_cluster=0)


@cute.jit
def _w1_pack_fp8x4(a: Float32, b: Float32,
                    c: Float32, d: Float32) -> Uint32:
    """Pack reference channel0/token0, channel0/token1, channel1/token0/1."""
    ab = f32x2_to_e4m3x2_bits(a, b)
    cd = f32x2_to_e4m3x2_bits(c, d)
    return Uint32(ab) | (Uint32(cd) << Uint32(16))


@cute.jit
def _load_w2_source_metadata(base: cute.Pointer, row_in_block: Uint32,
                              valid_m: Uint32, pool_token: Uint32,
                              rank_idx: Uint32, is_shared,
                              topk: cutlass.Constexpr,
                              has_shared: cutlass.Constexpr):
    """Read valid rows with fresh undefined values for inactive rows."""
    src_rank = undefined_u32()
    src_token = undefined_u32()
    topk_slot = undefined_u32()
    if row_in_block < valid_m:
        if cutlass.const_expr(has_shared):
            src_rank = rank_idx
            src_token = pool_token
            topk_slot = Uint32(topk)
            if not is_shared:
                src_rank, src_token, topk_slot = ld_global_u32x3(
                    base + pool_token * Uint32(3))
        else:
            src_rank, src_token, topk_slot = ld_global_u32x3(
                base + pool_token * Uint32(3))
    return src_rank, src_token, topk_slot


@cute.jit
def _load_k3_tma(activation_map: cute.Pointer,
                        activation_sf_map: cute.Pointer,
                        weight_map: cute.Pointer,
                        weight_sf_map: cute.Pointer,
                        activation_smem: cute.Pointer,
                        activation_sf_smem: cute.Pointer,
                        weight_smem: cute.Pointer,
                        weight_sf_smem: cute.Pointer,
                        full_bar: cute.Pointer, empty_bar: cute.Pointer,
                        k_block: Uint32, stage_phase: Uint32,
                        block_idx: Uint32, n_block: Uint32,
                        local_expert: Uint32, valid_m: Uint32,
                        activation_rows: Uint32,
                        shape_k: Uint32, is_shared: Uint32,
                        weight_n: Uint32, weight_k_base: Uint32,
                        cta_rank: Uint32,
                        loader_warp: cutlass.Constexpr,
                        block_m: cutlass.Constexpr,
                        sf_block_m: cutlass.Constexpr,
                        has_shared: cutlass.Constexpr,
                        routed_weight_linger: cutlass.Constexpr,
                        shared_weight_linger: cutlass.Constexpr = False,
                        shared_block_m: cutlass.Constexpr = 16,
                        activation_payload_hint: Uint64 = None,
                        repeated_weight: Uint32 = None) -> None:
    """K3 routed K128 / shared BF16 K64 transfers with independent M tiles.

    Both CTAs issue their own TMA transfers into local SMEM. The leader's
    full barrier counts two arrivals per CTA and source-byte transaction
    counts for both CTAs. The caller waits it before issuing mixed MMA.
    Routed FP8/FP4 tasks use K128 with scale factors; shared BF16 tasks
    use K64 without scale factors. The caller selects the stage storage.
    """
    cute.arch.mbarrier_wait(empty_bar, stage_phase ^ Uint32(1))
    k_step = Uint32(128)
    if cutlass.const_expr(has_shared):
        if is_shared != Uint32(0):
            k_step = Uint32(64)
    k_idx = k_block * k_step
    if cutlass.const_expr(loader_warp == 0):
        aligned_m = (valid_m + Uint32(15)) & Uint32(~15)
        logical_m = Uint32(block_m)
        if cutlass.const_expr(has_shared):
            if is_shared != Uint32(0):
                logical_m = Uint32(shared_block_m)
        m_idx = block_idx * logical_m
        if cta_rank != Uint32(0):
            m_idx += aligned_m // Uint32(2)
        with cute.arch.elect_one():
            tile_g2s_2cta_2d(activation_smem, activation_map, full_bar,
                             k_idx, m_idx, activation_payload_hint)
            sf_bytes = Uint32(sf_block_m * 4)
            load_sf = True
            if cutlass.const_expr(has_shared):
                if is_shared != Uint32(0):
                    sf_bytes = Uint32(0)
                    load_sf = False
            if load_sf:
                tile_g2s_2cta_2d(activation_sf_smem, activation_sf_map, full_bar,
                                 block_idx * Uint32(sf_block_m), k_block, Uint64(EVICT_LAST))
            if cta_rank == Uint32(0):
                cute.arch.mbarrier_arrive_and_expect_tx(
                    full_bar, 2 * (activation_rows * Uint32(128) + sf_bytes))
            else:
                cute.arch.mbarrier_arrive(full_bar, peer_cta_rank_in_cluster=0)
    else:
        n_idx = n_block * Uint32(128)
        # Keep weights at normal priority for likely reuse across routed
        # M-blocks. A full block may have a successor, and a later block
        # can reuse its predecessor's weights. Shared tasks use their own policy.
        routed_cache_hint = (Uint64(EVICT_NORMAL) if routed_weight_linger
                             else Uint64(EVICT_FIRST))
        weight_cache_hint = routed_cache_hint
        if cutlass.const_expr(repeated_weight is not None):
            if repeated_weight != Uint32(0):
                weight_cache_hint = Uint64(EVICT_NORMAL)
        shared_task = False
        if cutlass.const_expr(has_shared):
            shared_task = is_shared != Uint32(0)
            if cutlass.const_expr(repeated_weight is not None):
                if shared_task:
                    weight_cache_hint = routed_cache_hint
        if cutlass.const_expr(shared_weight_linger):
            # Two shared M-blocks are paired on adjacent clusters; the
            # second concurrent reader rereads these weight rows within
            # microseconds. EVICT_FIRST would demote them before it arrives.
            if shared_task:
                weight_cache_hint = Uint64(EVICT_LAST)
        with cute.arch.elect_one():
            if shared_task:
                tile_g2s_2cta_4d(weight_smem, weight_map, full_bar,
                                 Uint32(0), weight_n, weight_k_base + k_block,
                                 Uint32(0), weight_cache_hint)
            else:
                tile_g2s_2cta_4d(weight_smem, weight_map, full_bar,
                                 Uint32(0), weight_n, weight_k_base + k_block,
                                 local_expert, weight_cache_hint)
                tile_g2s_2cta_2d(weight_sf_smem, weight_sf_map, full_bar,
                                 n_idx, local_expert * (shape_k // Uint32(128)) + k_block,
                                 weight_cache_hint)
            weight_bytes = Uint32(128 * 128 // 2 + 128 * 4)
            if shared_task:
                weight_bytes = Uint32(128 * 128)
            if cta_rank == Uint32(0):
                cute.arch.mbarrier_arrive_and_expect_tx(full_bar, weight_bytes * Uint32(2))
            else:
                cute.arch.mbarrier_arrive(full_bar, peer_cta_rank_in_cluster=0)


class GCK3MoERoleKernel:
    """Keep the role on the Python object so CuTe erases other role paths."""

    def __init__(self, role: int, dispatch_sms: int = 1, compute_sms: int = 2,
                 block_m: int = 16, hidden: int = 256,
                 intermediate: int = 128, num_tokens: int = 16,
                 num_ranks: int = 1, local_experts: int = 1,
                 shared_experts: int = 0, capacity: int = 1920,
                 shared_hidden: int = 0, shared_intermediate: int = 0,
                 activation_beta: float = 4.0,
                 activation_linear_beta: float | None = 25.0,
                 ring_tokens: int = 1920,
                 sf_ring_tokens: int = 30720,
                 topk: int = 1,
                 activation_clamp: float = math.inf,
                 fast_math: bool = True,
                 nvlink_timeout_secs: int = DEFAULT_NVLINK_BARRIER_TIMEOUT_SECS,
                 *, role_config, per_source_ready: bool, single_cta_routes: bool,
                 combine_geometry: tuple[int, int, int] | None = None,
                 routed_weight_linger: bool = False, routed_weight_reuse: bool = False,
                 full_split_k_tile: bool = False,
                 shared_weight_linger: bool = False,
                 shared_block_m: int | None = None, epilogue_config=None,
                 projection_tile_rows: int = 0, projection_stages: int = 0,
                 projection_channels: int = PROJECTION_CHANNELS_PER_CTA,
                 has_prefix: bool = False, projection_splits: int = 1,
                 early_shared: bool = False, routed_assist: bool = False,
                 routed_midpoint_min_tokens: int = 0):
        self.routed_assist = routed_assist
        self.routed_midpoint_min_tokens = routed_midpoint_min_tokens
        shared_block_m = block_m if shared_block_m is None else shared_block_m
        storage_block_m = max(block_m, shared_block_m) if role == COMPUTE_ROLE else block_m
        if role_config.role != role or role_config.block_m != storage_block_m:
            raise ValueError('kernel role/block M differs from its host configuration')
        math_stages = role_config.num_stages
        if role not in WORKER_ROLES:
            raise ValueError('role must be dispatch, compute or finalize')
        if local_experts < 1 or local_experts > MAX_EXPERTS_PER_RANK or \
                num_ranks * local_experts > MAX_EXPERTS or topk < 1 or \
                topk > min(32, num_ranks * local_experts) or shared_experts < 1:
            raise NotImplementedError('device scheduler requires 1..512 local experts, <=2048 global experts, 1<=topk<=min(32,num_experts); K3 requires shared experts')
        if num_ranks < 1 or num_ranks > MAX_RANKS or not 0 <= num_tokens <= capacity:
            raise NotImplementedError('dispatch routing requires 1..72 ranks within capacity')
        if block_m not in CANDIDATE_BLOCK_M or shared_block_m not in CANDIDATE_BLOCK_M:
            raise ValueError('invalid K3 routed/shared block M')
        if math_stages < 2 or math_stages > 32:
            raise ValueError('GC math pipeline needs 2..32 TMA stages')
        if not 0 < nvlink_timeout_secs <= 2_147_483_647:
            raise ValueError('NVLink barrier timeout must fit positive INT_MAX seconds')
        self.role = role
        self.dispatch_sms = dispatch_sms
        self.compute_sms = compute_sms
        self.block_m = block_m
        self.shared_block_m = shared_block_m
        # The accumulator pitch is shared across all task formats.
        self.storage_block_m = storage_block_m
        self.num_tokens = num_tokens
        self.hidden = hidden
        self.intermediate = intermediate
        self.num_ranks = num_ranks
        self.local_experts = local_experts
        self.topk = topk
        # Host selection is shared by Dispatch and Finalize. Only traversal
        # may vary with the rank's local token count.
        self.per_source_ready = per_source_ready
        self.single_cta_routes = single_cta_routes
        self.has_activation_clamp = not math.isinf(activation_clamp)
        self.fast_math = fast_math
        self.shared_experts = shared_experts
        self.shared_hidden = shared_hidden
        self.shared_intermediate = shared_intermediate
        # Each shared W2 M tile tracks one bit per completed W1 K64 slice.
        # Use the existing cleared bank only when all of its tile masks fit.
        self.shared_w2_mask_words = (shared_intermediate // 64 + 63) // 64
        shared_tiles = (num_tokens + shared_block_m - 1) // shared_block_m
        self.shared_w2_stage_ready = (
            role == COMPUTE_ROLE and shared_block_m >= 16
            and (capacity + 7) // 8 >= 2 * self.shared_w2_mask_words * shared_tiles
        )
        self.activation_beta = activation_beta
        self.activation_linear_beta = activation_linear_beta
        self.capacity = capacity
        self.math_stages = math_stages
        routed_payload = role_config.shared_layout.routed_payload
        self.routed_math_stages = routed_payload.stages if routed_payload else 0
        self.data_barrier_stages = math_stages + self.routed_math_stages
        if self.data_barrier_stages > 32:
            raise ValueError('GC data barrier banks exceed one initialization warp')
        self.nvlink_timeout_secs = nvlink_timeout_secs
        # The default routed hint estimates reuse from the launch geometry.
        # The weight loader can also detect likely reuse from each task's M block.
        # BF16 shared weights with paired M blocks use their own EVICT_LAST hint.
        self.routed_weight_linger = routed_weight_linger
        self.routed_weight_reuse = routed_weight_reuse
        self.full_split_k_tile = full_split_k_tile
        self.shared_weight_linger = shared_weight_linger
        # The compact dual-ring layout sizes scratch from the routed epilogue.
        if routed_payload is not None and epilogue_config is None:
            raise ValueError("GC dual data rings require an explicit epilogue_config")
        epilogue_config = role_config if epilogue_config is None else epilogue_config
        self.store_m_l1 = epilogue_config.store_m_l1
        self.store_m_l2 = epilogue_config.store_m_l2
        self.cd_l1_bytes = 2 * 2 * self.store_m_l1 * 128
        self.cd_l2_bytes = 2 * self.store_m_l2 * 128 * 2
        self.cd_bytes = max(self.cd_l1_bytes, self.cd_l2_bytes)
        # float2[kNumEpilogueWarps][STORE_BLOCK_M_L1 / 2].
        self.amax_float_count = 8 * self.store_m_l1
        small_expert_table = local_experts <= SMALL_LOCAL_EXPERT_TABLE_MAX
        self.dispatch_registers = 48 if small_expert_table else 96
        self.non_epi_registers = 40 if small_expert_table else 72
        self.epi_registers = 208 if small_expert_table else 168
        tile = MixedMmaTile(storage_block_m, hidden, intermediate)
        self.l1_clusters = tile.l1_clusters
        self.l2_clusters = tile.l2_clusters
        self.l1_full_mask = (1 << tile.l1_n_blocks) - 1
        self.tmem_cols = tile.tmem_cols
        self.accum_stages = tile.epilogue_stages
        if self.accum_stages > ACCUM_BARRIER_STAGES:
            raise ValueError('epilogue stages exceed the reserved accumulator barriers')
        self.sf_block_m = role_config.sf_block_m
        self.sfa_tmem_cols = tile.sfa_tmem_cols
        self.dispatch_warps = role_config.routing_warps
        role_registers = (
            self.dispatch_warps * 32 * self.dispatch_registers if role == DISPATCH_ROLE else
            128 * self.non_epi_registers + 256 * self.epi_registers if role == COMPUTE_ROLE else
            128 * self.dispatch_registers + (COMBINE_WARPS * 32) * self.epi_registers)
        if role_registers > 64512:
            raise ValueError('role register allocation exceeds the SM budget')
        self.pull_bytes = role_config.pull_bytes
        self.pull_chunks = hidden // self.pull_bytes
        self.projection_tile_rows = projection_tile_rows
        self.projection_stages = projection_stages
        self.projection_channels = projection_channels
        self.projection_splits = projection_splits
        self.early_shared = early_shared
        self.has_prefix = has_prefix
        if role == FINALIZE_ROLE:
            if projection_tile_rows:
                if combine_geometry is None:
                    raise ValueError('projecting finalize requires host combine geometry')
                (self.combine_chunks, self.combine_chunk_bytes,
                 self.combine_uint4_per_lane) = combine_geometry
                self.projection_smem_offset = align(COMBINE_WARPS * 4, 1024)
                self.finalize_sms = compute_sms
            else:
                self.finalize_sms = min(compute_sms, ZERO_TOKEN_FINALIZE_SMS)
        self.shared_layout = role_config.shared_layout
        layout = GCK3BufferLayout(num_ranks, num_ranks * local_experts, capacity, topk,
                                hidden, intermediate, ring_tokens, sf_ring_tokens,
                                shared_experts, shared_hidden, shared_intermediate)
        if projection_splits > 1:
            self.projection_partial_offset = layout.views['projection_partials'].offset
        self.topk_offset_words = layout.views['topk_idx'].offset // 4
        self.input_x_words = layout.views['x'].offset // 4
        self.input_sf_words = layout.views['x_sf'].offset // 4
        self.input_weight_words = layout.views['topk_weights'].offset // 4
        self.l1_x_words = layout.views['l1_acts'].offset // 4
        self.l1_sf_words = layout.views['l1_acts_sf'].offset // 4
        self.l2_x_words = layout.views['l2_acts'].offset // 4
        self.l2_sf_words = layout.views['l2_acts_sf'].offset // 4
        self.shared_l2_x_words = layout.views['shared_l2_acts'].offset // 4
        self.combine_words = layout.views['combine_tokens'].offset // 4
        self.metadata_words = layout.workspace_offsets['token_src_metadata'] // 4
        self.ring_tokens = layout.ring_tokens
        self.sf_ring_tokens = layout.sf_ring_tokens
        self.ring_blocks = layout.ring_tokens // block_m
        sf_block_m = self.sf_block_m
        sf_row_max = max((t & ~127) + (t & 31) * 4 + ((t >> 5) & 3)
                         for t in range(block_m))
        sf_even_row_max = max((t & ~127) + (t & 31) * 4 + ((t >> 5) & 3)
                              for t in range(0, block_m, 2))
        last_ring_sf_row = (self.ring_blocks - 1) * sf_block_m
        l1_sf_local_max = ((hidden // 128 - 1) * self.sf_ring_tokens +
                           last_ring_sf_row + sf_row_max)
        routed_w1_sf_local_max = 4 * (
            (intermediate // 128 - 1) * self.sf_ring_tokens +
            last_ring_sf_row + sf_even_row_max) + 19
        self.wide_l1_sf = (l1_sf_local_max > 0xffffffff or
                           self.l1_sf_words + l1_sf_local_max > 0xffffffff)
        self.wide_routed_w1_sf = routed_w1_sf_local_max > 0xffffffff

    @cute.jit
    def _initialize_shared_storage(self, slab: cute.Pointer,
                                    physical_warp: Uint32, lane: Uint32) -> None:
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()
        if physical_warp == 0:
            if elect_one_sync():
                if cutlass.const_expr(self.role == DISPATCH_ROLE):
                    bulk_zero_shared(
                        slab + self.shared_layout.expert_counts,
                        Uint32(((self.num_ranks * self.local_experts * 4 + 1023) //
                                1024) * 1024))
        elif physical_warp == 1:
            if cutlass.const_expr(self.role == DISPATCH_ROLE):
                bars = cute.recast_ptr(slab + self.shared_layout.dispatch_bars, dtype=Uint64)
                for index in cutlass.range(
                        Int32(lane), Int32(self.dispatch_warps), Int32(32), unroll_full=True):
                    cute.arch.mbarrier_init(bars + index, 1)
            cute.arch.mbarrier_init_fence()
        elif physical_warp == 2:
            # CuTe 4.8.1's single-thread-per-CTA docstring is conservative
            # for disjoint barriers; the wrapper does not elect a lane.
            # Each object below has exactly one initializer. Every such
            # lane executes the init fence before the cluster rendezvous
            # publishes initialization to consumers. The constructor caps
            # the combined stage count at 32 so one warp initializes both banks.
            if cutlass.const_expr(self.role == COMPUTE_ROLE):
                cta_rank, _, _ = cute.arch.block_in_cluster_idx()
                data_full = cute.recast_ptr(slab + self.shared_layout.data_full, dtype=Uint64)
                data_empty = cute.recast_ptr(slab + self.shared_layout.data_empty, dtype=Uint64)
                if lane < self.data_barrier_stages:
                    if cta_rank == 0:
                        cute.arch.mbarrier_init(data_full + lane, 4)
                    cute.arch.mbarrier_init(data_empty + lane, 1)
                accum_full = cute.recast_ptr(slab + self.shared_layout.accum_full, dtype=Uint64)
                accum_empty = cute.recast_ptr(slab + self.shared_layout.accum_empty, dtype=Uint64)
                if lane < self.accum_stages:
                    cute.arch.mbarrier_init(accum_full + lane, 1)
                    if cta_rank == 0:
                        cute.arch.mbarrier_init(accum_empty + lane, TASK_EMPTY_CONSUMERS)
                full = cute.recast_ptr(slab + self.shared_layout.task_full, dtype=Uint64)
                empty = cute.recast_ptr(slab + self.shared_layout.task_empty, dtype=Uint64)
                if lane < 2:
                    cute.arch.mbarrier_init(full + lane, 1)
                    if cta_rank == 0:
                        cute.arch.mbarrier_init(empty + lane, TASK_EMPTY_CONSUMERS)
            cute.arch.mbarrier_init_fence()
        elif physical_warp == 3:
            if cutlass.const_expr(self.role == COMPUTE_ROLE):
                tptr = cute.recast_ptr(slab + self.shared_layout.tmem_ptr, dtype=Uint32)
                cute.arch.alloc_tmem(self.tmem_cols, tptr, is_two_cta=True)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        # Early Compute starts with independent shared work and waits at
        # exit. Routed tasks still wait for Dispatch's counts and FP8 payloads.
        if cutlass.const_expr(not self.early_shared):
            cute.arch.griddepcontrol_wait()

    @cute.jit
    def _data_ring(self, slab: cute.Pointer, is_shared: Uint32):
        b_offset = Uint32(self.shared_layout.data_b)
        sfa_offset = Uint32(self.shared_layout.data_sfa)
        sfb_offset = Uint32(self.shared_layout.data_sfb)
        stages = Uint32(self.math_stages)
        activation_pitch = Uint32(self.storage_block_m // 2 * 128)
        bank = Uint32(0)
        if cutlass.const_expr(self.routed_math_stages > 0):
            if is_shared == Uint32(0):
                b_offset = Uint32(self.shared_layout.routed_payload.data_b)
                sfa_offset = Uint32(self.shared_layout.routed_payload.data_sfa)
                sfb_offset = Uint32(self.shared_layout.routed_payload.data_sfb)
                stages = Uint32(self.routed_math_stages)
                activation_pitch = Uint32(self.shared_layout.routed_payload.activation_pitch)
            else:
                bank = Uint32(self.routed_math_stages)
        # Select byte offsets first so both branches have the same CuTe type
        # even when the physical planes have different additional alignment.
        data_b = slab + b_offset
        data_sfa = cute.recast_ptr(slab + sfa_offset, dtype=Uint32)
        data_sfb = cute.recast_ptr(slab + sfb_offset, dtype=Uint32)
        return data_b, data_sfa, data_sfb, stages, activation_pitch, bank

    @cute.jit
    def _switch_data_ring(self, data_empty: cute.Pointer, previous_shared: Uint32,
                          is_shared: Uint32, stage: Uint32, phase: Uint32,
                          inactive_stage: Uint32, inactive_phase: Uint32,
                          drain: cutlass.Constexpr):
        if previous_shared != is_shared:
            if cutlass.const_expr(drain):
                if previous_shared != Uint32(2):
                    old_stages = Uint32(self.math_stages)
                    old_bank = Uint32(self.routed_math_stages)
                    if previous_shared == Uint32(0):
                        old_stages = Uint32(self.routed_math_stages)
                        old_bank = Uint32(0)
                    wrapped = stage == Uint32(0)
                    last_stage = old_stages - Uint32(1) if wrapped else stage - Uint32(1)
                    last_phase = phase ^ Uint32(wrapped)
                    # The final old commit covers the preceding MMA and SF
                    # reads. Both loaders wait on their own CTA's old bank
                    # before either writes into the overlapping payload.
                    cute.arch.mbarrier_wait(data_empty + old_bank + last_stage,
                                            last_phase)
            stage, inactive_stage = inactive_stage, stage
            phase, inactive_phase = inactive_phase, phase
        return stage, phase, inactive_stage, inactive_phase

    @cute.jit
    def _retire_workspace(self, signals: cute.Tensor, peer_ptrs: cute.Tensor,
                        rank_idx: Uint32, tid: Uint32,
                        block: Uint32, ready_epoch: Uint64) -> None:
        # Publish completed peer FC2 stores even with no routed tokens.
        # Epochs distinguish late old publications without resetting ready.
        if cutlass.const_expr(not self.projection_tile_rows):
            if block == 0 and tid < self.num_ranks:
                dst = cute.make_ptr(
                    Uint64, mapped_peer_base(peer_ptrs, Uint32(tid), self.num_ranks) +
                    Uint64(SIGNAL_OFFSETS['combine_ready_grid_idx']) +
                    Uint64(rank_idx * 8), cute.AddressSpace.gmem,
                    assumed_align=8)
                st_release_sys_u64(dst, ready_epoch)
        # dispatch acquired every expert sum, including zero-count experts,
        # before publishing this summary. Normal graph completion dependencies
        # also retire every compute ring access before this footer executes.
        pool_blocks = signals[SIGNAL_OFFSETS['routed_pool_blocks'] // 4]
        # All CTAs use the immutable dispatch epoch, so block0 may flip parity
        # without a grid-wide cache-completion barrier. The next dispatch
        # clears this bank only after the entire local finalize grid ends.
        # That clear precedes count publication; all-rank sum acquires in
        # the intervening call order it before any peer can publish into
        # the same bank again. Zero-count experts must join too.
        if block == Uint32(0) and tid == Uint32(0):
            red_gpu_add_u32(
                signals.iterator + SIGNAL_OFFSETS['recv_parity'] // 4,
                Uint32(1))
        if cutlass.const_expr(not self.projection_tile_rows):
            warpgroup_reg_dealloc(self.dispatch_registers)

        if block == 0:
            if cutlass.const_expr(not self.single_cta_routes):
                # Only the multi-CTA traversal writes these local counters.
                send_off = SIGNAL_OFFSETS['expert_send_count'] // 4
                for expert_iter in cutlass.range(
                        Uint32(tid), Uint32(self.num_ranks * self.local_experts),
                        128, unroll_full=True):
                    expert_idx = Uint32(expert_iter)
                    st_global_u64(signals.iterator + send_off + expert_idx * Uint32(2),
                                  Uint64(0))
            if tid // Uint32(32) == Uint32(0):
                if elect_one_sync():
                    for locality in cutlass.range(2, unroll_full=True):
                        signals[SIGNAL_OFFSETS['l1_task_count'] // 4 + locality] = Uint32(0)
                        signals[SIGNAL_OFFSETS['l2_task_count'] // 4 + locality] = Uint32(0)
                        signals[SIGNAL_OFFSETS['shared_l1_task_count'] // 4 + locality] = Uint32(0)
                        signals[SIGNAL_OFFSETS['shared_l2_task_count'] // 4 + locality] = Uint32(0)
            cute.arch.sync_warp()
            shared_block = Uint32(tid)
            while shared_block < Uint32((self.capacity + 7) // 8):
                signals[SIGNAL_OFFSETS['shared_l2_full_count'] // 4 +
                        shared_block] = Uint32(0)
                shared_block += Uint32(128)
            cute.arch.sync_warp()

        # The padded expert ranges concatenate to [0,P). Modulo R, their
        # used slots are exactly [0,min(P,R)); clear each of those slots once.
        ring_block = block * Uint32(128) + tid
        while ring_block < min(pool_blocks, Uint32(self.ring_blocks)):
            signals[SIGNAL_OFFSETS['l1_full_count'] // 4 + ring_block] = Uint32(0)
            signals[SIGNAL_OFFSETS['l1_empty_count'] // 4 + ring_block] = Uint32(0)
            st_global_u64(
                signals.iterator + SIGNAL_OFFSETS['l2_full_mask'] // 4 +
                ring_block * Uint32(2), Uint64(0))
            signals[SIGNAL_OFFSETS['l2_empty_count'] // 4 + ring_block] = Uint32(0)
            ring_block += Uint32(self.finalize_sms * 128)

        # The former tail NVLink barrier (tag 2) is gone: the receive
        # signals are dual-buffered by recv_parity, so peers' next publish
        # targets the alternate generation and ordering is carried by the
        # all-expert sum acquires in the intervening call. Selected-peer
        # ready acquires protect each source's single-buffer inputs/indices.

    @cute.jit
    def _normalize_combine_row(self, reduced: cute.Tensor, scratch: cute.Pointer,
                               routed_norm: cute.Tensor, norm_eps: Float32, routed_scale: Float32,
                               epi_warp: Uint32, lane) -> None:
        partials = cute.make_tensor(cute.recast_ptr(
            scratch, dtype=Float32), cute.make_layout((COMBINE_WARPS,)))
        square_sum = Float32(0)
        for i in cutlass.range_constexpr(self.combine_uint4_per_lane * 8):
            # The standalone RMSNorm consumes a BF16 combine result.
            x = bits_float(Uint32(f32_to_bf16_bits(reduced[i])) << Uint32(16))
            if routed_scale != Float32(1.0):
                x = (x * routed_scale).to(BFloat16).to(Float32)
            reduced[i] = x
            square_sum = fma_rn(x, x, square_sum)
        for shift in cutlass.range_constexpr(5):
            square_sum += cute.arch.shuffle_sync_bfly(square_sum, 16 >> shift)
        if lane == 0:
            partials[epi_warp] = square_sum
        sync_aligned(COMBINE_WARPS * 32, 2)
        row_sum = Float32(0)
        if lane == 0:
            for i in cutlass.range_constexpr(COMBINE_WARPS):
                row_sum += partials[i]
        row_sum = cute.arch.shuffle_sync(row_sum, 0)
        scale = cute.math.rsqrt(
            row_sum * Float32(1.0 / self.hidden) + norm_eps, approx=True)
        # Retire reads before the next row reuses the partials.
        sync_aligned(COMBINE_WARPS * 32, 2)
        if epi_warp < Uint32(self.combine_chunks):
            for i in cutlass.range_constexpr(self.combine_uint4_per_lane * 8):
                col = (epi_warp * Uint32(self.combine_chunk_bytes // 2) +
                       Uint32((i // 8) * 256 + i % 8) + Uint32(lane) * Uint32(8))
                reduced[i] = reduced[i] * scale * Float32(routed_norm[col])

    @cute.jit
    def _accumulate_combine_routes(self, reduced: cute.Tensor, combine: cute.Pointer,
                                   routing_weights: cute.Pointer, token: Uint32, lane: Uint32,
                                   chunk_offset: Uint32, mask: Uint32, all_valid: cutlass.Constexpr):
        # Both paths visit slots in the same order and preserve the FP32 FMA chain.
        for g in cutlass.range_constexpr(self.topk):
            if cutlass.const_expr(all_valid) or ((mask >> Uint32(g)) & Uint32(1)) != Uint32(0):
                source = combine + (
                    Uint64(g) * Uint64(self.capacity) + Uint64(token)) * Uint64(self.hidden * 2) + Uint64(chunk_offset)
                weight = routing_weights[token * Uint32(self.topk) + Uint32(g)]
                for j in cutlass.range_constexpr(self.combine_uint4_per_lane):
                    words = ld_global_u32x4(source + Uint64((j * 32 + lane) * 16))
                    for q in cutlass.range_constexpr(4):
                        packed = words[q]
                        reduced[j * 8 + q * 2] = \
                            fma_rn(bits_float((packed & Uint32(0xffff)) << Uint32(16)),
                                weight,
                                reduced[j * 8 + q * 2])
                        reduced[j * 8 + q * 2 + 1] = \
                            fma_rn(bits_float(packed & Uint32(0xffff0000)),
                                weight,
                                reduced[j * 8 + q * 2 + 1])

    @cute.jit
    def _combine_normalize_row(self, signals: cute.Tensor, rank_idx: Uint32,
                               num_tokens: Uint32, y_out: cute.Tensor,
                               physical_warp: Uint32, lane: Uint32,
                               combine_smem: cute.Pointer,
                               ready_epoch: Uint64, routed_norm: cute.Tensor,
                               norm_eps: Float32, routed_scale: Float32, token: Uint32) -> None:
        combine_chunks = self.combine_chunks
        chunk_bytes = self.combine_chunk_bytes
        u4_per_lane = self.combine_uint4_per_lane
        epi_warp = Uint32(physical_warp - 4)
        ready = signals.iterator + SIGNAL_OFFSETS['combine_ready_grid_idx'] // 4
        # Per-warp vector loads stay in registers through TopK reduction.
        combine = cute.recast_ptr(
            signals.iterator, dtype=Uint8) + \
            Uint64(self.combine_words) * Uint64(4)
        output = cute.recast_ptr(y_out.iterator, dtype=Uint8)
        routes = cute.recast_ptr(
            signals.iterator + self.topk_offset_words, dtype=Uint64)
        routing_weights = cute.recast_ptr(signals.iterator + self.input_weight_words,dtype=Float32)
        if token < num_tokens:
            chunk = epi_warp
            chunk_offset = chunk * Uint32(chunk_bytes)
            lane_valid = False
            routed_peer = Uint32(0)
            active_chunk = chunk < Uint32(combine_chunks)
            if active_chunk and lane < self.topk:
                routed_expert = routes[token * Uint32(self.topk) + lane]
                lane_valid = routed_expert < \
                    Uint64(self.num_ranks * self.local_experts)
                if lane_valid:
                    routed_peer = Uint32(routed_expert) // Uint32(self.local_experts)
            # All lanes vote; invalid routes never form a peer pointer.
            # Duplicate owners may acquire the same publication twice.
            peer_ready = not lane_valid
            wait_start = read_clock64()
            while cute.arch.vote_ballot_sync(peer_ready) != Uint32(0xffffffff):
                if lane_valid and not peer_ready:
                    peer_ready = ld_acquire_sys_u64(
                        ready + routed_peer * Uint32(2)) == ready_epoch
                if Uint64(read_clock64() - wait_start) >= Uint64(
                        self.nvlink_timeout_secs * 2000000000):
                    if not peer_ready:
                        cute.printf('GC token readiness timeout rank={} peer={}',
                                    rank_idx, routed_peer)
                        device_trap()
            # Make selected-lane acquires visible to the vector loaders.
            cute.arch.sync_warp()
            mask = Uint32(cute.arch.vote_ballot_sync(lane_valid))
            reduced = cute.make_rmem_tensor((u4_per_lane * 8,), Float32)
            for index in cutlass.range_constexpr(u4_per_lane * 8):
                reduced[index] = Float32(0)
            if mask == (Uint32(0xffffffff) >> Uint32(32 - self.topk)):
                self._accumulate_combine_routes(reduced, combine, routing_weights, token, lane, chunk_offset, mask, True)
            else:
                self._accumulate_combine_routes(reduced, combine, routing_weights, token, lane, chunk_offset, mask, False)
            self._normalize_combine_row(reduced, combine_smem, routed_norm,
                                        norm_eps, routed_scale, epi_warp, lane)
            if active_chunk:
                dst = output + Uint64(token) * Uint64(self.hidden * 2) + Uint64(chunk_offset)
                for j in cutlass.range_constexpr(u4_per_lane):
                    words = cute.make_rmem_tensor((4,), Uint32)
                    for q in cutlass.range_constexpr(4):
                        words[q] = pack_bf16_pair(reduced[j * 8 + q * 2], reduced[j * 8 + q * 2 + 1])
                    st_global_u32x4(dst + Uint64((j * 32 + lane) * 16),
                                    words[0], words[1], words[2], words[3])
            # Carry every chunk's vector stores to the row publisher.
            sync_aligned(COMBINE_WARPS * 32, 2)
            if epi_warp == Uint32(0) and lane == 0:
                ready_row = cute.recast_ptr(
                    signals.iterator + SIGNAL_OFFSETS['projection_row_ready'] // 4,
                    dtype=Uint64)
                st_release_gpu_u64(ready_row + token, ready_epoch | (Uint64(1) << 32))

    @cute.jit
    def _finalize_output(self, signals: cute.Tensor, peer_ptrs: cute.Tensor, rank_idx: Uint32,
                    num_tokens: Uint32, y_out: cute.Tensor,
                    routed_norm: cute.Tensor, norm_eps: Float32, routed_scale: Float32, out: cute.Tensor,
                    shared_output: cute.Tensor, prefix: cute.Tensor, weight_map: cute.Pointer, latent_map: cute.Pointer,
                    slab: cute.Pointer, ready_epoch: Uint64,
                    block: Uint32, tid: Uint32, warp: Uint32, lane: Uint32) -> None:
        data_full = cute.recast_ptr(slab + self.shared_layout.data_full, dtype=Uint64)
        data_empty = cute.recast_ptr(slab + self.shared_layout.data_empty, dtype=Uint64)
        accum_full = cute.recast_ptr(slab + self.shared_layout.accum_full, dtype=Uint64)
        accum_empty = cute.recast_ptr(slab + self.shared_layout.accum_empty, dtype=Uint64)
        row_slot = cute.recast_ptr(slab + self.shared_layout.tasks, dtype=Uint32)
        tmem_ptr = cute.recast_ptr(slab + self.shared_layout.tmem_ptr, dtype=Uint32)
        cta_rank, _, _ = cute.arch.block_in_cluster_idx()
        channels_per_pair = 2 * self.projection_channels
        channel_tiles = self.shared_hidden // channels_per_pair
        tile_count = cute.ceil_div(num_tokens, self.projection_tile_rows)
        jobs = Uint32(tile_count * channel_tiles * self.projection_splits)
        pair = block // Uint32(2)
        if warp == Uint32(0):
            if lane < Uint32(self.projection_stages):
                if cta_rank == 0:
                    cute.arch.mbarrier_init(data_full + lane, 2)
                cute.arch.mbarrier_init(data_empty + lane, 1)
            if lane < Uint32(2):
                cute.arch.mbarrier_init(accum_full + lane, 1)
                if cta_rank == 0:
                    cute.arch.mbarrier_init(accum_empty + lane, 2 * PROJECTION_EPILOGUE_WARPS)
            cute.arch.mbarrier_init_fence()
        if warp == Uint32(3) and pair < jobs:
            cute.arch.alloc_tmem(2 * self.projection_tile_rows, tmem_ptr, is_two_cta=True)
        # Any admitted pair can publish, even when block zero has yet to run.
        if cta_rank == 0 and warp == Uint32(2):
            elected = Uint32(1)
            if lane == Uint32(0):
                elected = atom_add_u32(
                    signals.iterator + SIGNAL_OFFSETS['combine_peer_publisher'] // 4,
                    Uint32(1))
            elected = cute.arch.shuffle_sync(elected, 0)
            if elected == Uint32(0):
                for peer in cutlass.range(Int32(lane), Int32(self.num_ranks), 32):
                    dst = cute.make_ptr(
                        Uint64, mapped_peer_base(peer_ptrs, Uint32(peer), self.num_ranks) +
                        Uint64(SIGNAL_OFFSETS['combine_ready_grid_idx']) + Uint64(rank_idx * 8),
                        cute.AddressSpace.gmem, assumed_align=8)
                    st_release_sys_u64(dst, ready_epoch)
        if warp < Uint32(4):
            self._retire_workspace(signals, peer_ptrs, rank_idx, tid, block, ready_epoch)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        local_epoch = ready_epoch | (Uint64(1) << 32)
        row_ready = cute.recast_ptr(
            signals.iterator + SIGNAL_OFFSETS['projection_row_ready'] // 4, dtype=Uint64)
        if warp >= Uint32(4):
            # Any admitted CTA can claim the next combine row. A projection
            # can therefore wait for its own rows even when some pairs have
            # not started. Other warps use a disjoint SMEM ring to prefill W.
            row = claim_combine_row(signals, row_slot, warp, lane, num_tokens)
            while row < num_tokens:
                self._combine_normalize_row(signals, rank_idx, num_tokens, y_out, warp, lane,
                                            slab, ready_epoch, routed_norm, norm_eps, routed_scale, row)
                row = claim_combine_row(signals, row_slot, warp, lane, num_tokens)

        split_count = Uint32(self.projection_splits)
        parts = cute.recast_ptr(signals.iterator, dtype=Float32)
        if cutlass.const_expr(self.projection_splits > 1):
            parts = parts + Uint64(self.projection_partial_offset // 4)
        projection = Uint32(0)
        job = pair
        while job < jobs:
            # Channel-major assignment spreads the first tiles over all pairs.
            split = job % split_count
            logical_job = job // split_count
            row = (logical_job % tile_count) * Uint32(self.projection_tile_rows)
            channel = logical_job // tile_count
            _project_tile_2cta(
                slab + self.projection_smem_offset, data_full, data_empty,
                accum_full, accum_empty, tmem_ptr, row_ready,
                weight_map, latent_map, out, shared_output, prefix,
                row, channel, num_tokens, local_epoch, projection,
                Uint32(cta_rank), warp, lane, self.projection_tile_rows,
                self.hidden, self.has_prefix, self.projection_stages, self.projection_channels,
                parts, split, self.projection_splits, self.shared_hidden, self.full_split_k_tile)
            projection += Uint32(1)
            job += Uint32(self.finalize_sms // 2)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        if warp == Uint32(3) and pair < jobs:
            tmem_free_2cta(tmem_ptr[0], Uint32(2 * self.projection_tile_rows))

    @cute.kernel
    def kernel(self, signals: cute.Tensor, peer_ptrs: cute.Tensor,
               rank_idx: Uint32, num_tokens: Uint32,
               tensor_maps=None, y_out=None, activation_clamp=None,
               tensor_map_1=None, tensor_map_2=None, tensor_map_3=None,
               tensor_map_4=None, tensor_map_5=None, tensor_map_6=None,
               tensor_map_7=None, tensor_map_8=None, tensor_map_9=None,
               tensor_map_10=None, tensor_map_11=None, tensor_map_12=None,
               tensor_map_13=None, tensor_map_14=None, tensor_map_15=None,
               tensor_map_16=None, tensor_map_17=None, valid_token_count=None,
               routed_norm=None, norm_eps=None, routed_scale=None, final_out=None, shared_out=None, prefix_sum=None,
               projection_weight_map=None, projection_latent_map=None):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        physical_warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = tid % 32

        if cutlass.const_expr(self.role == DISPATCH_ROLE):
            # First count local routes, reserve a distinct source-index span,
            # then publish the receive count only after writing that span.
            # Route each global expert to its owner rank and local expert.
            # Each local expert has an independent receive count and a padded
            # pool-block prefix; ring slots remain shared by all experts.
            dispatch_smem = cutlass.utils.SmemAllocator()
            slab = cute.recast_ptr(
                dispatch_smem.allocate(self.shared_layout.field_end, 1024), dtype=Uint8)
            expert_token_count = cute.recast_ptr(
                slab + self.shared_layout.expert_counts, dtype=Uint32)
            pull = cute.recast_ptr(slab + self.shared_layout.pull, dtype=Uint32)
            pull_bar = cute.recast_ptr(slab + self.shared_layout.dispatch_bars, dtype=Uint64)
            self._initialize_shared_storage(slab, Uint32(physical_warp), Uint32(lane))
            warpgroup_reg_dealloc(self.dispatch_registers)
            # Dual-buffered receive signals: replay N publishes and reads
            # generation parity & 1; finalize flips the persistent parity
            # word after consuming each replay, and the replay that retires
            # a generation clears it here, so no cross-rank tail barrier is
            # needed.  finalize reads the epoch published below instead of a
            # grid barrier.
            parity = ld_volatile_u32(
                signals.iterator + SIGNAL_OFFSETS['recv_parity'] // 4)
            retired = (parity + Uint32(1)) & Uint32(1)
            gen = parity & Uint32(1)
            if block == Uint32(0):
                if tid == Uint32(0):
                    # The normal dispatch -> finalize graph completion dependency
                    # makes this plain store visible; finalize never polls it
                    # concurrently with this dispatch.
                    signals[SIGNAL_OFFSETS['finalize_epoch'] // 4] = parity + Uint32(1)
                if cutlass.const_expr(self.projection_tile_rows > 0):
                    if tid == Uint32(0):
                        signals[SIGNAL_OFFSETS['combine_peer_publisher'] // 4] = Uint32(0)
                        signals[SIGNAL_OFFSETS['combine_next_row'] // 4] = Uint32(0)
                    row_ready = cute.recast_ptr(
                        signals.iterator + SIGNAL_OFFSETS['projection_row_ready'] // 4, dtype=Uint64)
                    for row in cutlass.range(
                            Int32(tid), Int32(num_tokens), self.dispatch_warps * 32):
                        st_global_u64(row_ready + row, Uint64(0))
                retired_sum = signals.iterator + \
                    SIGNAL_OFFSETS['expert_recv_count_sum'] // 4 + \
                    retired * Uint32(self.local_experts * 2)
                for retired_iter in cutlass.range(
                        Uint32(tid), Uint32(self.local_experts),
                        self.dispatch_warps * 32, unroll_full=True):
                    st_global_u64(
                        retired_sum + Uint32(retired_iter) * Uint32(2), Uint64(0))
                retired_recv = signals.iterator + \
                    SIGNAL_OFFSETS['expert_recv_count'] // 4 + \
                    retired * Uint32(self.num_ranks * self.local_experts * 2)
                for retired_iter in cutlass.range(
                        Uint32(tid), Uint32(self.num_ranks * self.local_experts),
                        self.dispatch_warps * 32, unroll_full=True):
                    st_global_u64(
                        retired_recv + Uint32(retired_iter) * Uint32(2), Uint64(0))
            recv_ptr = signals.iterator + SIGNAL_OFFSETS['expert_recv_count_sum'] // 4 + \
                gen * Uint32(self.local_experts * 2)
            send_ptr = signals.iterator + SIGNAL_OFFSETS['expert_send_count'] // 4
            routes = cute.make_tensor(
                cute.recast_ptr(signals.iterator + self.topk_offset_words, dtype=Uint64),
                cute.make_layout((self.capacity * self.topk,)))
            source_idx = cute.make_tensor(
                cute.recast_ptr(signals.iterator + SIGNAL_OFFSETS['size'] // 4,
                                dtype=Uint32),
                cute.make_layout((self.num_ranks * self.local_experts * self.capacity,)))
            # Reference read_topk_idx: adjacent lanes own adjacent topk
            # entries, so one warp processes floor(32 / topk) tokens per
            # iteration. Lanes past that whole-token group stay inactive.
            tokens_per_warp = 32 // self.topk
            active_route_lanes = tokens_per_warp * self.topk
            route_stride = self.dispatch_sms * self.dispatch_warps * tokens_per_warp
            if cutlass.const_expr(not self.single_cta_routes):
                route_base = Uint32((block * self.dispatch_warps + physical_warp) *
                                    tokens_per_warp)
                for route_iter in cutlass.range(
                        route_base, num_tokens, route_stride, unroll_full=True):
                    route_base = Uint32(route_iter)
                    route_token = route_base + Uint32(lane // self.topk)
                    token_topk = route_base * Uint32(self.topk) + Uint32(lane)
                    if lane < active_route_lanes and route_token < num_tokens:
                        expert = routes[token_topk]
                        if expert < Uint64(self.num_ranks * self.local_experts):
                            atom_shared_add_u32(expert_token_count + Uint32(expert),
                                                Uint32(1))
                    cute.arch.sync_warp()
                cute.arch.sync_threads()
                for expert_iter in cutlass.range(
                        Uint32(tid), Uint32(self.num_ranks * self.local_experts),
                        self.dispatch_warps * 32, unroll_full=True):
                    expert_idx = Uint32(expert_iter)
                    old_send = atom_add_u64(
                        send_ptr + expert_idx * Uint32(2),
                        (Uint64(1) << 32) | Uint64(expert_token_count[expert_idx]))
                    expert_token_count[expert_idx] = Uint32(old_send)
                cute.arch.sync_threads()
            if cutlass.const_expr(self.single_cta_routes):
                # Block zero assigns each source index and increments its
                # shared expert count in the same pass. Other CTAs only pull.
                route_base = Uint32(num_tokens)
                if block == Uint32(0):
                    route_base = Uint32(physical_warp * tokens_per_warp)
                route_stride = self.dispatch_warps * tokens_per_warp
            else:
                route_base = Uint32((block * self.dispatch_warps + physical_warp) *
                                    tokens_per_warp)
            for route_iter in cutlass.range(
                    route_base, num_tokens, route_stride, unroll_full=True):
                route_base = Uint32(route_iter)
                route_token = route_base + Uint32(lane // self.topk)
                token_topk = route_base * Uint32(self.topk) + Uint32(lane)
                if lane < active_route_lanes and route_token < num_tokens:
                    expert = routes[token_topk]
                    if expert < Uint64(self.num_ranks * self.local_experts):
                        dst_rank = Uint32(expert) // Uint32(self.local_experts)
                        dst_local = Uint32(expert) % Uint32(self.local_experts)
                        slot = atom_shared_add_u32(
                            expert_token_count + Uint32(expert), Uint32(1))
                        # One source/token may contribute at most one slot to
                        # an expert. Reject overflow before indexing a peer buffer.
                        if slot < Uint32(self.capacity):
                            dst_base = mapped_peer_base(peer_ptrs, dst_rank, self.num_ranks)
                            dst_index = cute.make_ptr(
                                Uint32, dst_base + Uint64(SIGNAL_OFFSETS['size']) +
                                Uint64(((dst_local * Uint32(self.num_ranks) + rank_idx) *
                                        Uint32(self.capacity) +
                                        slot) *
                                       Uint32(4)), cute.AddressSpace.gmem)
                            dst_index[0] = token_topk
                        else:
                            device_trap()
                cute.arch.sync_warp()
            if cutlass.const_expr(not self.single_cta_routes):
                grid_sync(signals, Uint32(block), Uint32(tid), self.dispatch_sms, 0)
            if block == 0:
                if cutlass.const_expr(self.single_cta_routes):
                    # Carry this CTA's remote index writes to its publisher.
                    cute.arch.sync_threads()
                for expert_iter in cutlass.range(
                        Uint32(tid), Uint32(self.num_ranks * self.local_experts),
                        self.dispatch_warps * 32, unroll_full=True):
                    expert_idx = Uint32(expert_iter)
                    dst_rank = expert_idx // Uint32(self.local_experts)
                    dst_local = expert_idx % Uint32(self.local_experts)
                    dst_base = mapped_peer_base(peer_ptrs, dst_rank, self.num_ranks)
                    status = Uint64(0)
                    if cutlass.const_expr(self.single_cta_routes):
                        status = Uint64(expert_token_count[expert_idx])
                    else:
                        status = ld_global_u64(
                            signals.iterator + SIGNAL_OFFSETS['expert_send_count'] // 4 +
                            expert_idx * Uint32(2))
                    remote_count = cute.make_ptr(
                        Uint64, dst_base + Uint64(SIGNAL_OFFSETS['expert_recv_count']) +
                        Uint64(gen * Uint32(self.num_ranks * self.local_experts * 8)) +
                        Uint64((rank_idx * Uint32(self.local_experts) + dst_local) *
                               Uint32(8)), cute.AddressSpace.gmem,
                        assumed_align=8)
                    st_global_u64(remote_count, status & Uint64(0xffffffff))
                if cutlass.const_expr(self.per_source_ready):
                    cute.arch.sync_threads()
                    flag_offset = Uint64(SIGNAL_OFFSETS['dispatch_source_ready']) + Uint64(gen * MAX_RANKS * 8)
                    epoch = Uint64(parity) + Uint64(1)
                    if tid < self.num_ranks:
                        peer = mapped_peer_base(peer_ptrs, Uint32(tid), self.num_ranks)
                        dst = cute.make_ptr(Uint64, peer + flag_offset + Uint64(rank_idx * 8),
                                            cute.AddressSpace.gmem, assumed_align=8)
                        st_release_sys_u64(dst, epoch)
                    if tid < self.num_ranks:
                        ready = cute.recast_ptr(
                            cute.recast_ptr(signals.iterator, dtype=Uint8) + flag_offset, dtype=Uint64)
                        wait_start = read_clock64()
                        while ld_acquire_sys_u64(ready + tid) != epoch:
                            if read_clock64() - wait_start >= Uint64(self.nvlink_timeout_secs * 2000000000):
                                cute.printf('GC dispatch publication timeout rank={} source={}', rank_idx, tid)
                                device_trap()
                    # Import every source's route indices and counts into the
                    # CTA before publishing local expert sums to the grid.
                    cute.arch.sync_threads()
                    local_counts = signals.iterator + SIGNAL_OFFSETS['expert_recv_count'] // 4 + \
                        gen * Uint32(self.num_ranks * self.local_experts * 2)
                    for expert in cutlass.range(
                            Uint32(tid), Uint32(self.local_experts), self.dispatch_warps * 32):
                        total = Uint64(0)
                        for source in cutlass.range_constexpr(self.num_ranks):
                            total += ld_global_u64(local_counts + (Uint32(source * self.local_experts) + expert) * 2)
                        completed = (Uint64(self.dispatch_sms * self.num_ranks) << 32) | total
                        st_release_gpu_u64(recv_ptr + expert * 2, completed)
                else:
                    # The grid acquire carries all route writes. Transfer
                    # them, plus every per-rank count store,
                    # to every publisher before the system release fence.
                    cute.arch.sync_threads()
                    fence_acq_rel_sys()
                    for expert_iter in cutlass.range(
                            Uint32(tid), Uint32(self.num_ranks * self.local_experts),
                            self.dispatch_warps * 32, unroll_full=True):
                        expert_idx = Uint32(expert_iter)
                        dst_rank = expert_idx // Uint32(self.local_experts)
                        dst_local = expert_idx % Uint32(self.local_experts)
                        dst_base = mapped_peer_base(peer_ptrs, dst_rank, self.num_ranks)
                        status = ld_global_u64(
                            signals.iterator + SIGNAL_OFFSETS['expert_send_count'] // 4 +
                            expert_idx * Uint32(2))
                        remote_sum = cute.make_ptr(
                            Uint64, dst_base + Uint64(SIGNAL_OFFSETS['expert_recv_count_sum']) +
                            Uint64(gen * Uint32(self.local_experts * 8)) +
                            Uint64(dst_local * Uint32(8)),
                            cute.AddressSpace.gmem, assumed_align=8)
                        atom_sys_add_u64(remote_sum, status)
            cute.arch.sync_threads()
            # Each per-expert acquire below imports every rank's route/count
            # stores, replacing the dispatch rank barrier. Include zero-token
            # experts: this join orders prior combine reads before new FC2
            # stores, and generation N's clear before N+2 can publish.
            # GCMoEScheduler::fetch_expert_recv_count: each lane caches
            # expert wave*32+lane, including zero for padded lanes.
            expert_counts = cute.make_rmem_tensor(
                ((self.local_experts + 31) // 32,), Uint32)
            for wave in cutlass.range_constexpr(
                    (self.local_experts + 31) // 32):
                expert_idx = Uint32(wave * 32) + lane
                observed = Uint64(0)
                if expert_idx < Uint32(self.local_experts):
                    wait_start = read_clock64()
                    while Uint32(observed >> 32) != Uint32(
                            self.dispatch_sms * self.num_ranks):
                        observed = ld_volatile_u64(
                            recv_ptr + expert_idx * Uint32(2))
                        if Uint64(read_clock64() - wait_start) >= Uint64(
                                self.nvlink_timeout_secs * 2000000000):
                            if block == Uint32(0):
                                cute.printf(
                                    'GC Dispatch recv-count timeout rank={} expert={} gen={} observed_hi={} expected={}',
                                    rank_idx, expert_idx, gen, Uint32(observed >> 32),
                                    self.dispatch_sms * self.num_ranks)
                            device_trap()
                    observed = ld_acquire_sys_u64(
                        recv_ptr + expert_idx * Uint32(2))
                expert_counts[wave] = Uint32(observed)
            cute.arch.sync_warp()

            if block == Uint32(0) and physical_warp == Uint32(0):
                own_pool_blocks = Uint32(0)
                for wave in cutlass.range_constexpr(
                        (self.local_experts + 31) // 32):
                    own_pool_blocks += (expert_counts[wave] +
                                        Uint32(self.block_m - 1)) // Uint32(self.block_m)
                total_pool_blocks = cute.arch.warp_redux_sync(own_pool_blocks, 'add')
                if lane == Uint32(0):
                    signals[SIGNAL_OFFSETS['routed_pool_blocks'] // 4] = total_pool_blocks

            # One global warp cursor traverses concatenated, unpadded expert
            # tokens; the pool-block prefix advances by each expert's padded
            # M-block count. Only the TMA and final L1 release elect a lane.
            token_idx = Uint32(block) * Uint32(self.dispatch_warps) + Uint32(physical_warp)
            current_expert = Int32(-1)
            expert_start = Uint32(0)
            expert_end = Uint32(0)
            pool_block_offset = Uint32(0)
            pull_phase = Uint32(0)
            rank_waves = (self.num_ranks + 31) // 32
            current_expert, expert_start, expert_end, pool_block_offset = \
                advance_expert(
                    expert_counts, token_idx, current_expert, expert_start,
                    expert_end, pool_block_offset, lane, self.local_experts,
                    self.block_m)
            while current_expert < Int32(self.local_experts):
                received = expert_end - expert_start
                pull_token = token_idx - expert_start
                pool_block = pool_block_offset + pull_token // self.block_m
                pool_token = pool_block * Uint32(self.block_m) + pull_token % Uint32(self.block_m)
                ring_block = pool_block % self.ring_blocks
                generation = pool_block // self.ring_blocks

                # Each lane owns one source rank in each 32-rank wave.
                # Reduce across its register waves before the warp reduction.
                remaining = cute.make_rmem_tensor((rank_waves,), Uint32)
                # Reload per token to isolate the immutable source count from
                # the destructively peeled loop-carried remainder.
                for wave in cutlass.range_constexpr(rank_waves):
                    source_rank = Uint32(wave * 32) + lane
                    remaining[wave] = Uint32(0)
                    if source_rank < Uint32(self.num_ranks):
                        count_ptr = signals.iterator + \
                            SIGNAL_OFFSETS['expert_recv_count'] // 4 + \
                            gen * Uint32(self.num_ranks * self.local_experts * 2) + \
                            (source_rank * Uint32(self.local_experts) +
                             Uint32(current_expert)) * Uint32(2)
                        remaining[wave] = Uint32(
                            ld_global_u64(count_ptr))
                src_rank = Uint32(0)
                src_slot = Uint32(0)
                use_interval = False
                if cutlass.const_expr(rank_waves == 1):
                    active_sources = Uint32(cute.arch.popc(Uint32(
                        cute.arch.vote_ballot_sync(remaining[0] > Uint32(0)))))
                    # Concatenating sparse sources takes one warp prefix scan.
                    # With more active peers, spread the first pulls round-robin.
                    use_interval = active_sources <= Uint32(8)
                if use_interval:
                    inclusive = warp_inclusive_sum(remaining[0], Uint32(lane))
                    mask = Uint32(cute.arch.vote_ballot_sync(inclusive <= pull_token))
                    src_rank = Uint32(cute.arch.popc(mask))
                    start = cute.arch.shuffle_sync(inclusive - remaining[0], Int32(src_rank))
                    src_slot = pull_token - start
                else:
                    peel_slot = Uint32(pull_token)
                    peel_offset = Uint32(0)
                    selected = False
                    while not selected:
                        lane_active = Uint32(0)
                        lane_minimum = Uint32(0xffffffff)
                        for wave in cutlass.range_constexpr(rank_waves):
                            count = remaining[wave]
                            lane_active += Uint32(1) if count > Uint32(0) else Uint32(0)
                            if count > Uint32(0):
                                lane_minimum = min(lane_minimum, count)
                        active = cute.arch.warp_redux_sync(lane_active, 'add')
                        minimum = cute.arch.warp_redux_sync(lane_minimum, 'min')
                        round_tokens = active * minimum
                        if peel_slot < round_tokens:
                            position = peel_slot % active
                            seen = Uint32(0)
                            for wave in cutlass.range_constexpr(rank_waves):
                                mask = Uint32(cute.arch.vote_ballot_sync(
                                    remaining[wave] > Uint32(0)))
                                wave_active = Uint32(cute.arch.popc(mask))
                                if position >= seen and position < seen + wave_active:
                                    src_rank = Uint32(wave * 32) + warp_nth_set_bit(
                                        mask, position - seen + Uint32(1))
                                seen += wave_active
                            src_slot = peel_offset + peel_slot // active
                            selected = True
                        else:
                            peel_slot -= round_tokens
                            peel_offset += minimum
                            for wave in cutlass.range_constexpr(rank_waves):
                                remaining[wave] -= min(remaining[wave], minimum)
                token_topk = source_idx[(Uint32(current_expert) * Uint32(self.num_ranks) +
                                         src_rank) * Uint32(self.capacity) + src_slot]
                src_token = token_topk // Uint32(self.topk)
                topk_slot = token_topk % Uint32(self.topk)
                if generation > 0:
                    empty_ptr = signals.iterator + \
                        SIGNAL_OFFSETS['l1_empty_count'] // 4 + ring_block
                    target = generation * (2 * self.intermediate // 128)
                    while ld_acquire_gpu_u32(empty_ptr) < target:
                        pass
                src_base = mapped_peer_base(peer_ptrs, src_rank, self.num_ranks)
                src_x = cute.make_ptr(
                    Uint32, src_base + Uint64(self.input_x_words * 4) +
                    Uint64(src_token) * Uint64(self.hidden),
                    cute.AddressSpace.gmem, assumed_align=16)
                dst_x = cute.recast_ptr(
                    cute.recast_ptr(signals.iterator, dtype=Uint8) +
                    Uint64(self.l1_x_words) * Uint64(4) +
                    Uint64(pool_token % self.ring_tokens) * Uint64(self.hidden),
                    dtype=Uint32)
                warp_pull = pull + physical_warp * (self.pull_bytes // 4)
                warp_bar = pull_bar + physical_warp
                if elect_one_sync():
                    for chunk in cutlass.range_constexpr(self.pull_chunks):
                        byte_words = chunk * (self.pull_bytes // 4)
                        bulk_g2s_1d(warp_pull, src_x + byte_words,
                                     Uint32(self.pull_bytes), warp_bar)
                        cute.arch.mbarrier_arrive_and_expect_tx(warp_bar, self.pull_bytes)
                        if chunk + 1 < self.pull_chunks:
                            cute.arch.mbarrier_wait(warp_bar, pull_phase)
                            pull_phase ^= Uint32(1)
                            bulk_s2g_1d(dst_x + byte_words, warp_pull,
                                         Uint32(self.pull_bytes))
                            cute.arch.cp_async_bulk_commit_group()
                            cute.arch.cp_async_bulk_wait_group(0)
                cute.arch.sync_warp()

                sf_row = ring_block * ((self.block_m + 127) // 128 * 128) + \
                    ((pull_token % self.block_m) & ~127) + \
                    ((pull_token % self.block_m) & 31) * 4 + \
                    (((pull_token % self.block_m) >> 5) & 3)
                remote_sf_base = src_base + Uint64(self.input_sf_words * 4) + \
                    Uint64(src_token) * Uint64(self.hidden // 128 * 4)
                if cutlass.const_expr((self.hidden // 128) % 4 == 0):
                    sf_vectors = self.hidden // 512
                    for q_round in cutlass.range(
                            (sf_vectors + 31) // 32, unroll_full=True):
                        q = q_round * 32 + lane
                        if q < sf_vectors:
                            remote_vec = cute.make_ptr(
                                Uint32, remote_sf_base + Uint64(q * Uint32(16)),
                                cute.AddressSpace.gmem, assumed_align=16)
                            values = ld_global_u32x4(remote_vec)
                            for component in cutlass.range_constexpr(4):
                                sf_col = q * 4 + component
                                if cutlass.const_expr(self.wide_l1_sf):
                                    l1_sf_bytes = cute.recast_ptr(
                                        signals.iterator, dtype=Uint8) + \
                                        Uint64(self.l1_sf_words) * Uint64(4)
                                    sf_word = Uint64(sf_col) * \
                                        Uint64(self.sf_ring_tokens) + Uint64(sf_row)
                                    cute.recast_ptr(
                                        l1_sf_bytes + sf_word * Uint64(4),
                                        dtype=Uint32)[0] = values[component]
                                else:
                                    signals[self.l1_sf_words +
                                            sf_col * self.sf_ring_tokens + sf_row] = \
                                        values[component]
                else:
                    for sf_round in cutlass.range(
                            (self.hidden // 128 + 31) // 32, unroll_full=True):
                        sf_col = sf_round * 32 + lane
                        if sf_col < self.hidden // 128:
                            remote_word = cute.make_ptr(
                                Uint32, remote_sf_base + Uint64(sf_col * Uint32(4)),
                                cute.AddressSpace.gmem, assumed_align=4)
                            if cutlass.const_expr(self.wide_l1_sf):
                                l1_sf_bytes = cute.recast_ptr(
                                    signals.iterator, dtype=Uint8) + \
                                    Uint64(self.l1_sf_words) * Uint64(4)
                                sf_word = Uint64(sf_col) * \
                                    Uint64(self.sf_ring_tokens) + Uint64(sf_row)
                                cute.recast_ptr(
                                    l1_sf_bytes + sf_word * Uint64(4),
                                    dtype=Uint32)[0] = remote_word[0]
                            else:
                                signals[self.l1_sf_words +
                                        sf_col * self.sf_ring_tokens + sf_row] = \
                                    remote_word[0]
                cute.arch.sync_warp()
                if elect_one_sync():
                    signals[self.metadata_words + pool_token * 3] = src_rank
                    signals[self.metadata_words + pool_token * 3 + 1] = src_token
                    signals[self.metadata_words + pool_token * 3 + 2] = topk_slot
                    # The last remote G2S overlapped SF/metadata work. Drain
                    # it, store the chunk, then publish W1 after S2G finishes.
                    cute.arch.mbarrier_wait(warp_bar, pull_phase)
                    pull_phase ^= Uint32(1)
                    final_byte_words = (self.pull_chunks - 1) * (self.pull_bytes // 4)
                    bulk_s2g_1d(dst_x + final_byte_words, warp_pull,
                                 Uint32(self.pull_bytes))
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0)
                    is_last_token = token_idx == expert_end - Uint32(1)
                    full_increment = Uint32(self.block_m) - (pull_token % self.block_m) \
                        if is_last_token else Uint32(1)
                    red_release_gpu_add_u32(
                        signals.iterator + SIGNAL_OFFSETS['l1_full_count'] // 4 + ring_block,
                        Uint32(full_increment))
                cute.arch.sync_warp()
                token_idx += Uint32(self.dispatch_sms * self.dispatch_warps)
                current_expert, expert_start, expert_end, pool_block_offset = \
                    advance_expert(
                        expert_counts, token_idx, current_expert, expert_start,
                        expert_end, pool_block_offset, lane, self.local_experts,
                        self.block_m)
        elif cutlass.const_expr(self.role == COMPUTE_ROLE):
            if physical_warp == 0:
                # Each descriptor has its own aligned by-value kernel
                # parameter, in the configuration's map order.
                prefetch_tensormap(tensor_maps.iterator)
                prefetch_tensormap(tensor_map_1.iterator)
                prefetch_tensormap(tensor_map_2.iterator)
                prefetch_tensormap(tensor_map_3.iterator)
                prefetch_tensormap(tensor_map_4.iterator)
                prefetch_tensormap(tensor_map_5.iterator)
                prefetch_tensormap(tensor_map_6.iterator)
                prefetch_tensormap(tensor_map_7.iterator)
                prefetch_tensormap(tensor_map_8.iterator)
                if cutlass.const_expr(not self.routed_assist):
                    prefetch_tensormap(tensor_map_9.iterator)
                    prefetch_tensormap(tensor_map_10.iterator)
                    prefetch_tensormap(tensor_map_11.iterator)
                    prefetch_tensormap(tensor_map_12.iterator)
                    prefetch_tensormap(tensor_map_13.iterator)
                    prefetch_tensormap(tensor_map_14.iterator)
                    prefetch_tensormap(tensor_map_15.iterator)
                    prefetch_tensormap(tensor_map_16.iterator)
                    prefetch_tensormap(tensor_map_17.iterator)
            smem = cutlass.utils.SmemAllocator()
            slab = cute.recast_ptr(
                smem.allocate(self.shared_layout.field_end, 1024), dtype=Uint8)
            full = cute.recast_ptr(slab + self.shared_layout.task_full, dtype=Uint64)
            empty = cute.recast_ptr(slab + self.shared_layout.task_empty, dtype=Uint64)
            task = cute.recast_ptr(slab + self.shared_layout.tasks, dtype=Uint32)
            cd_smem = slab + self.shared_layout.cd
            data_a = slab + self.shared_layout.data_a
            data_b = slab + self.shared_layout.data_b
            epi_amax = cute.recast_ptr(slab + self.shared_layout.amax, dtype=Float32)
            data_full = cute.recast_ptr(slab + self.shared_layout.data_full, dtype=Uint64)
            data_empty = cute.recast_ptr(slab + self.shared_layout.data_empty, dtype=Uint64)
            accum_full = cute.recast_ptr(slab + self.shared_layout.accum_full, dtype=Uint64)
            accum_empty = cute.recast_ptr(slab + self.shared_layout.accum_empty, dtype=Uint64)
            tptr = cute.recast_ptr(slab + self.shared_layout.tmem_ptr, dtype=Uint32)
            cta_rank, _, _ = cute.arch.block_in_cluster_idx()
            self._initialize_shared_storage(slab, Uint32(physical_warp), Uint32(lane))
            # Warpgroup-uniform deallocation/allocation. The first WG
            # runs load/MMA/scheduler; the other two run epilogue.
            if physical_warp < 4:
                warpgroup_reg_dealloc(self.non_epi_registers)
            else:
                warpgroup_reg_alloc(self.epi_registers)
            iteration = Uint32(0)
            # Dual-buffered receive sums: poll the generation selected by
            # the persistent parity word (volatile read, fresh from L2).
            gen = ld_volatile_u32(
                signals.iterator + SIGNAL_OFFSETS['recv_parity'] // 4) & Uint32(1)
            recv_ptr = signals.iterator + SIGNAL_OFFSETS['expert_recv_count_sum'] // 4 + \
                gen * Uint32(self.local_experts * 2)
            l1_count_ptr = signals.iterator + SIGNAL_OFFSETS['l1_task_count'] // 4
            l2_count_ptr = signals.iterator + SIGNAL_OFFSETS['l2_task_count'] // 4
            shared_l1_count_ptr = signals.iterator + SIGNAL_OFFSETS['shared_l1_task_count'] // 4
            shared_l2_count_ptr = signals.iterator + SIGNAL_OFFSETS['shared_l2_task_count'] // 4
            expert_counts = cute.make_rmem_tensor(
                ((self.local_experts + 31) // 32,), Uint32)
            if cta_rank == 0 and physical_warp == 3:
                schedule_tasks(
                    task, full, empty, recv_ptr, l1_count_ptr, l2_count_ptr,
                    shared_l1_count_ptr, shared_l2_count_ptr, expert_counts,
                    iteration, Int32(lane), Int32(block), num_tokens,
                    self.block_m,
                    self.compute_sms,
                    self.dispatch_sms,
                    self.hidden,
                    self.intermediate,
                    self.l1_clusters,
                    self.l2_clusters,
                    self.local_experts,
                    self.num_ranks,
                    0 if self.routed_assist else self.shared_experts, self.shared_hidden, self.shared_intermediate,
                    self.shared_block_m, self.shared_w2_stage_ready,
                    self.ring_blocks, self.routed_assist,
                    self.routed_midpoint_min_tokens, valid_token_count,
                )
            # The same two-slot task stream drives both TMA producer warps
            # and the leader MMA warp. All three consume every valid phase.
            if physical_warp == 0:
                task_words = cute.make_tensor(task, cute.make_layout((16,)))
                task_iter = Uint32(0)
                data_stage = Uint32(0)
                data_phase = Uint32(0)
                if cutlass.const_expr(self.routed_math_stages > 0):
                    inactive_stage = Uint32(0)
                    inactive_phase = Uint32(0)
                    previous_shared = Uint32(2)
                running = True
                while running:
                    task_stage = task_iter % Uint32(2)
                    cute.arch.mbarrier_wait(full + task_stage,
                                            (task_iter // Uint32(2)) % Uint32(2))
                    task_phase = task_words[task_stage * 8]
                    running = task_phase != Uint32(GCK3BlockPhase.None_)
                    if running:
                        pool_block = Uint32(task_words[task_stage * 8 + 4])
                        valid_m = Uint32(task_words[task_stage * 8 + 5])
                        shape_k = Uint32(task_words[task_stage * 8 + 7])
                        n_cluster = Uint32(task_words[task_stage * 8 + 3])
                        local_expert = Uint32(task_words[task_stage * 8 + 1])
                        is_shared = Uint32(0)
                        if cutlass.const_expr(self.shared_experts > 0):
                            is_shared = Uint32(1) if task_phase >= \
                                Uint32(GCK3BlockPhase.SharedLinear1) else Uint32(0)
                        if cutlass.const_expr(self.routed_math_stages > 0):
                            data_stage, data_phase, inactive_stage, inactive_phase = \
                                self._switch_data_ring(
                                    data_empty, previous_shared, is_shared,
                                    data_stage, data_phase, inactive_stage,
                                    inactive_phase, drain=True)
                            previous_shared = is_shared
                        task_b, task_sfa, task_sfb, data_stages, activation_pitch, data_bank = \
                            self._data_ring(slab, is_shared)
                        block_idx = pool_block % Uint32(self.ring_blocks)
                        if cutlass.const_expr(self.shared_experts > 0):
                            if is_shared != Uint32(0):
                                block_idx = pool_block
                        activation_map = tensor_maps.iterator
                        activation_sf_map = tensor_map_1.iterator
                        weight_map = tensor_map_2.iterator
                        weight_sf_map = tensor_map_3.iterator
                        if task_phase == Uint32(GCK3BlockPhase.Linear2):
                            activation_map = tensor_map_5.iterator
                            activation_sf_map = tensor_map_6.iterator
                            weight_map = tensor_map_7.iterator
                            weight_sf_map = tensor_map_8.iterator
                        if cutlass.const_expr(self.shared_experts > 0):
                            if task_phase == Uint32(GCK3BlockPhase.SharedLinear1):
                                activation_map = tensor_map_9.iterator
                                activation_sf_map = tensor_map_10.iterator
                                weight_map = tensor_map_11.iterator
                                weight_sf_map = tensor_map_12.iterator
                            elif task_phase == Uint32(GCK3BlockPhase.SharedLinear2):
                                activation_map = tensor_map_14.iterator
                                activation_sf_map = tensor_map_15.iterator
                                weight_map = tensor_map_16.iterator
                                weight_sf_map = tensor_map_17.iterator
                        # The MMA B operand consumes N/2 rows from each CTA.
                        # Keep the original stage pitch and full shared/fallback maps.
                        activation_rows = Uint32(self.block_m // 2)
                        if cutlass.const_expr(self.shared_experts > 0):
                            if is_shared != Uint32(0):
                                activation_rows = Uint32(self.shared_block_m // 2)
                        # Keep this pair generic for the entire K loop.
                        # The conversion is full-warp, before elected TMA.
                        activation_map = generic_tensormap_ptr(activation_map)
                        activation_sf_map = generic_tensormap_ptr(activation_sf_map)
                        if task_phase == Uint32(GCK3BlockPhase.Linear1):
                            l1_full_ptr = signals.iterator + \
                                SIGNAL_OFFSETS['l1_full_count'] // 4 + block_idx
                            required = self.block_m * \
                                (pool_block // Uint32(self.ring_blocks) + Uint32(1))
                            while ld_acquire_gpu_u32(l1_full_ptr) != required:
                                pass
                        if cutlass.const_expr(self.shared_experts > 0 and
                                              not self.shared_w2_stage_ready):
                            if task_phase == Uint32(GCK3BlockPhase.SharedLinear2):
                                shared_full_ptr = signals.iterator + \
                                    SIGNAL_OFFSETS['shared_l2_full_count'] // 4 + block_idx
                                while ld_acquire_gpu_u32(shared_full_ptr) != \
                                        self.shared_intermediate // 128 * 2:
                                    pass
                        # GCL2KBlockDependency belongs to this task, so
                        # the observed XOR-normalized mask survives all
                        # K128 waits and is reset for the next task.
                        mask_ptr = signals.iterator + \
                            SIGNAL_OFFSETS['l2_full_mask'] // 4 + \
                            (pool_block % Uint32(self.ring_blocks)) * 2
                        generation = pool_block // Uint32(self.ring_blocks)
                        expected = Uint64(0) if (generation & Uint32(1)) != Uint32(0) \
                            else Uint64(self.l1_full_mask)
                        pending = Uint64(self.l1_full_mask)
                        k_block = Uint32(0)
                        k_step = Uint32(128)
                        if cutlass.const_expr(self.shared_experts > 0):
                            if is_shared != Uint32(0):
                                k_step = Uint32(64)
                        if cutlass.const_expr(self.shared_w2_stage_ready):
                            shared_mask = cute.recast_ptr(
                                signals.iterator + SIGNAL_OFFSETS['shared_l2_full_count'] // 4,
                                dtype=Uint64) + block_idx * Uint32(self.shared_w2_mask_words)
                            shared_ready = Uint64(0)
                        # Keep the scale resident. A full shared M tile twice as wide
                        # as the routed tile also needs L2 space, so stream the routed
                        # W2 payload at normal priority in that geometry.
                        activation_payload_hint = Uint64(EVICT_LAST)
                        if cutlass.const_expr(self.num_tokens == self.shared_block_m and
                                              self.shared_block_m >= 2 * self.block_m):
                            if task_phase == Uint32(GCK3BlockPhase.Linear2):
                                activation_payload_hint = Uint64(EVICT_NORMAL)
                        while k_block < shape_k // k_step:
                            if task_phase == Uint32(GCK3BlockPhase.Linear2):
                                required_mask = Uint64(3) << (k_block * Uint32(2))
                                while (pending & required_mask) != Uint64(0):
                                    pending = ld_acquire_gpu_u64(mask_ptr) ^ expected
                            if cutlass.const_expr(self.shared_w2_stage_ready):
                                if task_phase == Uint32(GCK3BlockPhase.SharedLinear2):
                                    if (k_block & Uint32(63)) == Uint32(0):
                                        shared_ready = Uint64(0)
                                    stage_bit = Uint64(1) << (k_block & Uint32(63))
                                    while (shared_ready & stage_bit) == Uint64(0):
                                        shared_ready = ld_acquire_gpu_u64(
                                            shared_mask + k_block // Uint32(64))
                            _load_k3_tma(
                                activation_map, activation_sf_map,
                                weight_map, weight_sf_map,
                                data_a + data_stage * activation_pitch,
                                task_sfa + data_stage * self.sf_block_m,
                                task_b + data_stage * (128 * 128),
                                task_sfb + data_stage * 128,
                                data_full + data_bank + data_stage,
                                data_empty + data_bank + data_stage,
                                k_block, data_phase, block_idx,
                                n_cluster * Uint32(2) + cta_rank,
                                local_expert, valid_m, activation_rows, shape_k,
                                is_shared, Uint32(0), Uint32(0), cta_rank, 0,
                                self.block_m, self.sf_block_m,
                                self.shared_experts > 0,
                                self.routed_weight_linger,
                                self.shared_weight_linger,
                                shared_block_m=self.shared_block_m,
                                activation_payload_hint=activation_payload_hint)
                            data_stage = Uint32(0) if data_stage == \
                                data_stages - Uint32(1) else data_stage + Uint32(1)
                            data_phase ^= Uint32(data_stage == Uint32(0))
                            k_block += Uint32(1)
                    task_iter += Uint32(1)
            elif physical_warp == 1:
                task_words = cute.make_tensor(task, cute.make_layout((16,)))
                task_iter = Uint32(0)
                data_stage = Uint32(0)
                data_phase = Uint32(0)
                if cutlass.const_expr(self.routed_math_stages > 0):
                    inactive_stage = Uint32(0)
                    inactive_phase = Uint32(0)
                    previous_shared = Uint32(2)
                running = True
                while running:
                    task_stage = task_iter % Uint32(2)
                    cute.arch.mbarrier_wait(full + task_stage,
                                            (task_iter // Uint32(2)) % Uint32(2))
                    task_phase = task_words[task_stage * 8]
                    running = task_phase != Uint32(GCK3BlockPhase.None_)
                    if running:
                        pool_block = Uint32(task_words[task_stage * 8 + 4])
                        valid_m = Uint32(task_words[task_stage * 8 + 5])
                        shape_k = Uint32(task_words[task_stage * 8 + 7])
                        n_cluster = Uint32(task_words[task_stage * 8 + 3])
                        local_expert = Uint32(task_words[task_stage * 8 + 1])
                        is_shared = Uint32(0)
                        if cutlass.const_expr(self.shared_experts > 0):
                            is_shared = Uint32(1) if task_phase >= \
                                Uint32(GCK3BlockPhase.SharedLinear1) else Uint32(0)
                        if cutlass.const_expr(self.routed_math_stages > 0):
                            data_stage, data_phase, inactive_stage, inactive_phase = \
                                self._switch_data_ring(
                                    data_empty, previous_shared, is_shared,
                                    data_stage, data_phase, inactive_stage,
                                    inactive_phase, drain=True)
                            previous_shared = is_shared
                        task_b, task_sfa, task_sfb, data_stages, activation_pitch, data_bank = \
                            self._data_ring(slab, is_shared)
                        block_idx = pool_block % Uint32(self.ring_blocks)
                        if cutlass.const_expr(self.shared_experts > 0):
                            if is_shared != Uint32(0):
                                block_idx = pool_block
                        activation_map = tensor_maps.iterator
                        activation_sf_map = tensor_map_1.iterator
                        weight_map = tensor_map_2.iterator
                        weight_sf_map = tensor_map_3.iterator
                        if task_phase == Uint32(GCK3BlockPhase.Linear2):
                            activation_map = tensor_map_5.iterator
                            activation_sf_map = tensor_map_6.iterator
                            weight_map = tensor_map_7.iterator
                            weight_sf_map = tensor_map_8.iterator
                        if cutlass.const_expr(self.shared_experts > 0):
                            if task_phase == Uint32(GCK3BlockPhase.SharedLinear1):
                                activation_map = tensor_map_9.iterator
                                activation_sf_map = tensor_map_10.iterator
                                weight_map = tensor_map_11.iterator
                                weight_sf_map = tensor_map_12.iterator
                            elif task_phase == Uint32(GCK3BlockPhase.SharedLinear2):
                                activation_map = tensor_map_14.iterator
                                activation_sf_map = tensor_map_15.iterator
                                weight_map = tensor_map_16.iterator
                                weight_sf_map = tensor_map_17.iterator
                        weight_map = generic_tensormap_ptr(weight_map)
                        weight_sf_map = generic_tensormap_ptr(weight_sf_map)
                        k_block = Uint32(0)
                        k_step = Uint32(128)
                        if cutlass.const_expr(self.shared_experts > 0):
                            if is_shared != Uint32(0):
                                k_step = Uint32(64)
                        k_tiles = shape_k // k_step
                        n_block = n_cluster * Uint32(2) + cta_rank
                        weight_n = cta_rank * Uint32(128)
                        weight_k_base = n_cluster * k_tiles
                        repeated_weight = None
                        if cutlass.const_expr(self.routed_weight_reuse):
                            expert_block = task_words[task_stage * 8 + 2]
                            repeated_weight = Uint32(
                                valid_m == Uint32(self.block_m) or expert_block != Uint32(0))
                        while k_block < k_tiles:
                            _load_k3_tma(
                                activation_map, activation_sf_map,
                                weight_map, weight_sf_map,
                                data_a + data_stage * activation_pitch,
                                task_sfa + data_stage * self.sf_block_m,
                                task_b + data_stage * (128 * 128),
                                task_sfb + data_stage * 128,
                                data_full + data_bank + data_stage,
                                data_empty + data_bank + data_stage,
                                k_block, data_phase, block_idx,
                                n_block, local_expert, valid_m, Uint32(0), shape_k,
                                is_shared, weight_n, weight_k_base, cta_rank, 1,
                                self.block_m, self.sf_block_m,
                                self.shared_experts > 0,
                                self.routed_weight_linger,
                                self.shared_weight_linger,
                                shared_block_m=self.shared_block_m,
                                repeated_weight=repeated_weight)
                            data_stage = Uint32(0) if data_stage == \
                                data_stages - Uint32(1) else data_stage + Uint32(1)
                            data_phase ^= Uint32(data_stage == Uint32(0))
                            k_block += Uint32(1)
                    task_iter += Uint32(1)
            if cta_rank == 0 and physical_warp == 2:
                task_words = cute.make_tensor(task, cute.make_layout((16,)))
                # Create the two swapped K-major data descriptor bases
                # once for the persistent leader MMA warp.
                weight_desc_base = kmajor_smem_desc(
                    data_b, Uint32(1024), True, Uint32(0))
                activation_desc_base = kmajor_smem_desc(
                    data_a, Uint32(1024), True, Uint32(0))
                data_desc_hi = (weight_desc_base >> Uint64(32)) << Uint64(32)
                sf_desc_hi = kmajor_smem_desc_from_start(
                    Uint32(0), Uint32(128), False)
                task_iter = Uint32(0)
                data_stage = Uint32(0)
                data_phase = Uint32(0)
                if cutlass.const_expr(self.routed_math_stages > 0):
                    inactive_stage = Uint32(0)
                    inactive_phase = Uint32(0)
                    previous_shared = Uint32(2)
                running = True
                while running:
                    task_stage = task_iter % Uint32(2)
                    cute.arch.mbarrier_wait(full + task_stage,
                                            (task_iter // Uint32(2)) % Uint32(2))
                    task_phase = task_words[task_stage * 8]
                    running = task_phase != Uint32(GCK3BlockPhase.None_)
                    if running:
                        valid_m = Uint32(task_words[task_stage * 8 + 5])
                        shape_k = Uint32(task_words[task_stage * 8 + 7])
                        is_shared = Uint32(0)
                        if cutlass.const_expr(self.shared_experts > 0):
                            is_shared = Uint32(1) if task_phase >= \
                                Uint32(GCK3BlockPhase.SharedLinear1) else Uint32(0)
                        if cutlass.const_expr(self.routed_math_stages > 0):
                            data_stage, data_phase, inactive_stage, inactive_phase = \
                                self._switch_data_ring(
                                    data_empty, previous_shared, is_shared,
                                    data_stage, data_phase, inactive_stage,
                                    inactive_phase, drain=False)
                            previous_shared = is_shared
                        task_b, task_sfa, task_sfb, data_stages, activation_pitch, data_bank = \
                            self._data_ring(slab, is_shared)
                        k_step = Uint32(128)
                        if cutlass.const_expr(self.shared_experts > 0):
                            if is_shared != Uint32(0):
                                k_step = Uint32(64)
                        num_k_blocks = shape_k // k_step
                        weight_start = Uint32(weight_desc_base)
                        if cutlass.const_expr(self.routed_math_stages > 0):
                            weight_start = Uint32(kmajor_smem_desc(
                                task_b, Uint32(1024), True, Uint32(0)))
                        weight_format = Uint32(5)
                        if cutlass.const_expr(self.shared_experts > 0):
                            weight_format = select_weight_format(is_shared)
                        aligned_n = (valid_m + Uint32(15)) & Uint32(~15)
                        instr_desc_base = blockscaled_instr_desc_hi(
                            aligned_n, weight_format, Uint32(0), Uint32(0))
                        if cutlass.const_expr(self.shared_experts > 0):
                            if is_shared != Uint32(0):
                                instr_desc_base = bf16_instr_desc_hi(aligned_n)
                        accum_stage = task_iter % Uint32(self.accum_stages)
                        accum_phase = (task_iter // Uint32(self.accum_stages)) % Uint32(2)
                        cute.arch.mbarrier_wait(accum_empty + accum_stage,
                                                accum_phase ^ Uint32(1))
                        tcgen05_after_thread_sync()
                        # Unroll K128 MMA stages by the configured group size.
                        for k_iter in cutlass.range(
                                Int32(0), Int32(num_k_blocks),
                                Int32(1), unroll=4):
                            k_block = Uint32(k_iter)
                            cute.arch.mbarrier_wait(data_full + data_bank + data_stage,
                                                    data_phase)
                            # Keep descriptor layout bits fixed and
                            # advance only the 16-byte SMEM address field.
                            weight_desc_start = Uint32(
                                weight_start + data_stage * Uint32(1024))
                            activation_desc_start = Uint32(
                                Uint32(activation_desc_base) + data_stage *
                                (activation_pitch >> Uint32(4)))
                            issue_mixed_k128(
                                weight_desc_start, activation_desc_start,
                                task_sfb, task_sfa, data_stage,
                                data_desc_hi, sf_desc_hi,
                                accum_stage * Uint32(self.storage_block_m),
                                Uint32(self.accum_stages * self.storage_block_m +
                                       self.sfa_tmem_cols),
                                Uint32(self.accum_stages * self.storage_block_m),
                                data_empty + data_bank + data_stage, accum_full + accum_stage,
                                instr_desc_base, k_block,
                                Uint32(k_block == num_k_blocks - Uint32(1)),
                                is_shared, self.sf_block_m,
                                32,
                                self.shared_experts > 0,
                                reuse_descriptors=False, shared_bf16=True)
                            data_stage = Uint32(0) if data_stage == \
                                data_stages - Uint32(1) else data_stage + Uint32(1)
                            data_phase ^= Uint32(data_stage == Uint32(0))
                    task_iter += Uint32(1)
                # The terminal None task increments task_iter once. Drain
                # the final valid accumulator stage before leaving the
                # producer warp, as the reference does after its loop.
                if task_iter > Uint32(1):
                    last_valid_iter = task_iter - Uint32(2)
                    last_stage = last_valid_iter % Uint32(self.accum_stages)
                    last_phase = (last_valid_iter // Uint32(self.accum_stages)) % Uint32(2)
                    cute.arch.mbarrier_wait(accum_empty + last_stage,
                                            last_phase)
            if physical_warp >= 4:
                if tptr[0] != Uint32(0):
                    device_trap()
                task_words = cute.make_tensor(task, cute.make_layout((16,)))
                iteration = Uint32(0)
                running = True
                while running:
                    stage = iteration % 2
                    generation_phase = (iteration // 2) % 2
                    cute.arch.mbarrier_wait(full + stage, generation_phase)
                    task_phase = task_words[stage * 8]
                    n_cluster = Uint32(task_words[stage * 8 + 3])
                    pool_block = Uint32(task_words[stage * 8 + 4])
                    valid_m = Uint32(task_words[stage * 8 + 5])
                    if task_phase != Uint32(GCK3BlockPhase.None_):
                        accum_stage = iteration % Uint32(self.accum_stages)
                        accum_phase = (iteration // Uint32(self.accum_stages)) % Uint32(2)
                        cute.arch.mbarrier_wait(accum_full + accum_stage,
                                                accum_phase)
                        tcgen05_after_thread_sync()
                        release_task_info(empty, Int32(stage))
                        # Match the reference's lane-0 exchange so valid-M
                        # controls the epilogue uniformly within each warp.
                        valid_m = cute.arch.shuffle_sync(valid_m, 0)
                        if physical_warp == 4 and \
                                task_phase == Uint32(GCK3BlockPhase.Linear2):
                            if elect_one_sync():
                                # The final MMA commit completed every K128
                                # input read. The BF16 peer scatter below uses
                                # TMEM, so L1 may now reuse the input ring.
                                ring_block = pool_block % Uint32(self.ring_blocks)
                                red_gpu_add_u32(
                                    signals.iterator +
                                    SIGNAL_OFFSETS['l2_empty_count'] // 4 + ring_block,
                                    Uint32(1))
                        if task_phase == Uint32(GCK3BlockPhase.Linear2):
                            # The reference joins every epilogue warp after
                            # the elected routed L2 ring release.
                            cute.arch.sync_warp()
                        is_l1 = task_phase == Uint32(GCK3BlockPhase.Linear1)
                        if cutlass.const_expr(self.shared_experts > 0):
                            is_l1 = is_l1 or task_phase == Uint32(GCK3BlockPhase.SharedLinear1)
                        if is_l1:
                            # W1 epilogue: the 16dp register mapping below
                            # predicates stores with this task's valid rows.
                            is_shared_l1 = Uint32(0)
                            if cutlass.const_expr(self.shared_experts > 0):
                                is_shared_l1 = Uint32(1) if task_phase == \
                                    Uint32(GCK3BlockPhase.SharedLinear1) else Uint32(0)
                            ring_block = pool_block % Uint32(self.ring_blocks)
                            generation = pool_block // Uint32(self.ring_blocks)
                            if not is_shared_l1:
                                l2_empty_ptr = signals.iterator + \
                                    SIGNAL_OFFSETS['l2_empty_count'] // 4 + ring_block
                                while ld_acquire_gpu_u32(l2_empty_ptr) != \
                                        generation * Uint32(self.l2_clusters * 2):
                                    pass
                            output_block = ring_block
                            if is_shared_l1:
                                output_block = pool_block
                            epi_warp = Uint32(physical_warp - 4)
                            warpgroup = epi_warp // Uint32(4)
                            warp_in_wg = epi_warp % Uint32(4)
                            logical_m = Uint32(self.block_m)
                            if is_shared_l1:
                                logical_m = Uint32(self.shared_block_m)
                            wg_start_m = warpgroup * (logical_m // Uint32(2))
                            valid_wg_rows = Uint32(0)
                            if valid_m > wg_start_m:
                                valid_wg_rows = valid_m - wg_start_m
                                if valid_wg_rows > logical_m // Uint32(2):
                                    valid_wg_rows = logical_m // Uint32(2)
                            valid_stores = (valid_wg_rows + Uint32(self.store_m_l1 - 1)) // \
                                Uint32(self.store_m_l1)
                            n_block = n_cluster * Uint32(2) + cta_rank
                            output_row_bytes = Uint32(64)
                            if is_shared_l1:
                                output_row_bytes = Uint32(128)
                            cd_wg = cd_smem + warpgroup * Uint32(2*self.store_m_l1) * output_row_bytes
                            raw_values = cute.make_rmem_tensor(
                                (2 if self.block_m == 240 else 1,
                                 self.store_m_l1 // 8, 8), Float32)
                            if valid_stores == Uint32(0):
                                _release_w1_tmem(accum_empty, accum_stage)
                            if cutlass.const_expr(self.block_m == 240):
                                if valid_stores > Uint32(0):
                                    for atom in cutlass.range_constexpr(
                                            self.store_m_l1 // 8):
                                        base = accum_stage * Uint32(self.storage_block_m) + \
                                            wg_start_m + Uint32(atom * 8)
                                        _load_w1_atom(raw_values, 0, atom, base)
                                    if valid_stores == Uint32(1):
                                        cute.arch.fence_view_async_tmem_load()
                                        _release_w1_tmem(accum_empty, accum_stage)
                            for store in cutlass.range_constexpr(
                                    (self.storage_block_m // 2) // self.store_m_l1):
                                if Uint32(store) < valid_stores:
                                    if cutlass.const_expr(self.block_m != 240):
                                        for atom in cutlass.range_constexpr(
                                                self.store_m_l1 // 8):
                                            base = accum_stage * Uint32(self.storage_block_m) + \
                                                wg_start_m + Uint32(store * self.store_m_l1 + atom * 8)
                                            _load_w1_atom(raw_values, 0, atom, base)
                                            if atom == self.store_m_l1 // 8 - 1 and \
                                                    Uint32(store + 1) == valid_stores:
                                                cute.arch.fence_view_async_tmem_load()
                                                _release_w1_tmem(accum_empty, accum_stage)
                                    # BM240 overlaps the next register
                                    # block's loads with this block's math.
                                    cute.arch.fence_view_async_tmem_load()
                                    if cutlass.const_expr(self.block_m == 240):
                                        if Uint32(store + 1) < valid_stores:
                                            for atom in cutlass.range_constexpr(
                                                    self.store_m_l1 // 8):
                                                next_base = accum_stage * Uint32(self.storage_block_m) + \
                                                    wg_start_m + Uint32(
                                                        (store + 1) * self.store_m_l1 + atom * 8)
                                                _load_w1_atom(
                                                    raw_values, (store + 1) % 2,
                                                    atom, next_base)
                                            if Uint32(store + 2) == valid_stores:
                                                cute.arch.fence_view_async_tmem_load()
                                                _release_w1_tmem(accum_empty, accum_stage)
                                    raw_buffer = store % 2 if self.block_m == 240 else 0
                                    activation_values = cute.make_rmem_tensor(
                                        (self.store_m_l1 // 8, 4), Float32)
                                    reduced_amax = cute.make_rmem_tensor(
                                        (self.store_m_l1 // 8, 2), Float32)
                                    for atom in cutlass.range_constexpr(
                                            self.store_m_l1 // 8):
                                        # Lanes with the same low two bits own one row pair. A padded
                                        # routed pair still writes zero payloads and publishes its scale.
                                        atom_row = Uint32(store * self.store_m_l1 + atom * 8)
                                        compute_situ = (is_shared_l1 != Uint32(0) or
                                                        atom_row + Uint32(lane % 4) * 2 < valid_wg_rows)
                                        value0 = value1 = value2 = value3 = Float32(0)
                                        if compute_situ:
                                            value0, value1 = _situ_pair(
                                                raw_values[raw_buffer, atom, 0],
                                                raw_values[raw_buffer, atom, 1],
                                                raw_values[raw_buffer, atom, 2],
                                                raw_values[raw_buffer, atom, 3],
                                                activation_clamp, self.has_activation_clamp,
                                                self.activation_beta,
                                                self.activation_linear_beta, self.fast_math)
                                            value2, value3 = _situ_pair(
                                                raw_values[raw_buffer, atom, 4],
                                                raw_values[raw_buffer, atom, 5],
                                                raw_values[raw_buffer, atom, 6],
                                                raw_values[raw_buffer, atom, 7],
                                                activation_clamp, self.has_activation_clamp,
                                                self.activation_beta,
                                                self.activation_linear_beta, self.fast_math)
                                        activation_values[atom, 0] = value0
                                        activation_values[atom, 1] = value1
                                        activation_values[atom, 2] = value2
                                        activation_values[atom, 3] = value3
                                        amax0 = warp_group4_max(
                                            fmax_f32(
                                                fmax_f32(Float32(0), abs_f32(value0)),
                                                abs_f32(value2)))
                                        amax1 = warp_group4_max(
                                            fmax_f32(
                                                fmax_f32(Float32(0), abs_f32(value1)),
                                                abs_f32(value3)))
                                        reduced_amax[atom, 0] = amax0
                                        reduced_amax[atom, 1] = amax1
                                        if lane < 4:
                                            amax_offset = epi_warp * \
                                                Uint32(self.store_m_l1) + \
                                                Uint32(atom * 8 + lane * 2)
                                            epi_amax[amax_offset] = amax0
                                            epi_amax[amax_offset + 1] = amax1
                                        cute.arch.sync_warp()
                                    tma_store_wait_1()
                                    sync_epilogue_warpgroup(warpgroup)
                                    tma_stage = store % 2
                                    cd_stage = cd_wg + Uint32(
                                        tma_stage * self.store_m_l1) * output_row_bytes
                                    for atom in cutlass.range_constexpr(
                                            self.store_m_l1 // 8):
                                        if is_shared_l1:
                                            row = Uint32(lane % 8)
                                            col = warp_in_wg * Uint32(2) + Uint32(lane // 8)
                                            offset = (Uint32(atom*8)+row)*Uint32(128) + (col ^ row)*Uint32(16)
                                            stmatrix_bf16x4_trans(cd_stage + offset,
                                                pack_bf16_pair(activation_values[atom,0],activation_values[atom,1]),
                                                pack_bf16_pair(activation_values[atom,2],activation_values[atom,3]))
                                        else:
                                            partner_offset = (epi_warp ^ Uint32(1)) * \
                                                Uint32(self.store_m_l1) + \
                                                Uint32(atom * 8) + \
                                                (Uint32(lane) % Uint32(4)) * Uint32(2)
                                            sf_amax0 = fmax_f32(
                                                reduced_amax[atom, 0],
                                                epi_amax[partner_offset])
                                            sf_amax1 = fmax_f32(
                                                reduced_amax[atom, 1],
                                                epi_amax[partner_offset + 1])
                                            sf_exp0 = ue8m0_fp8_scale_exp(sf_amax0)
                                            sf_exp1 = ue8m0_fp8_scale_exp(sf_amax1)
                                            inv0 = ue8m0_fp8_inv(sf_exp0)
                                            inv1 = ue8m0_fp8_inv(sf_exp1)
                                            upper0, upper1 = cute.arch.mul_packed_f32x2(
                                                (activation_values[atom, 0],
                                                 activation_values[atom, 1]), (inv0, inv1), rnd='rn', ftz=False)
                                            lower0, lower1 = cute.arch.mul_packed_f32x2(
                                                (activation_values[atom, 2],
                                                 activation_values[atom, 3]), (inv0, inv1), rnd='rn', ftz=False)
                                            packed = _w1_pack_fp8x4(
                                                upper0, upper1, lower0, lower1)
                                            row = Uint32(lane)
                                            smem_offset = Uint32(atom * 8 * 64) + \
                                                row * Uint32(64) + \
                                                (warp_in_wg ^ (row // Uint32(2))) * Uint32(16)
                                            stmatrix_fp8x4_trans_late(
                                                cd_stage, smem_offset, packed)
                                            if warp_in_wg % Uint32(2) == Uint32(0) and lane < 4:
                                                output_sf_words = Uint64(self.l2_sf_words)
                                                sf_stride = Uint32(self.sf_ring_tokens)
                                                l2_sf_bytes = cute.recast_ptr(
                                                    signals.iterator, dtype=Uint8) + \
                                                    output_sf_words * Uint64(4)
                                                k_sf = n_block * Uint32(2) + \
                                                    warp_in_wg // Uint32(2)
                                                token_base = wg_start_m + Uint32(
                                                    store * self.store_m_l1 + atom * 8)
                                                assume_u32_lt(
                                                    token_base, Uint32(self.block_m))
                                                sf_token = output_block * Uint32(self.sf_block_m) + \
                                                    (token_base & Uint32(~127)) + \
                                                    (token_base & Uint32(31)) * Uint32(4) + \
                                                    ((token_base >> Uint32(5)) & Uint32(3)) + \
                                                    Uint32(lane * 8)
                                                if cutlass.const_expr(
                                                        self.wide_routed_w1_sf):
                                                    sf_addr = Uint64(k_sf // Uint32(4)) * \
                                                        (Uint64(sf_stride) * Uint64(4)) + \
                                                        Uint64(sf_token) * Uint64(4) + \
                                                        Uint64(k_sf % Uint32(4))
                                                    l2_sf_bytes[sf_addr] = Uint8(sf_exp0)
                                                    l2_sf_bytes[sf_addr + Uint64(16)] = \
                                                        Uint8(sf_exp1)
                                                else:
                                                    sf_addr = (k_sf // Uint32(4)) * \
                                                        (sf_stride * Uint32(4)) + \
                                                        sf_token * Uint32(4) + k_sf % Uint32(4)
                                                    l2_sf_bytes[sf_addr] = Uint8(sf_exp0)
                                                    l2_sf_bytes[sf_addr + Uint32(16)] = \
                                                        Uint8(sf_exp1)
                                        cute.arch.sync_warp()
                                    sync_epilogue_warpgroup(warpgroup)
                                    if warp_in_wg == Uint32(0):
                                        with cute.arch.elect_one():
                                            output_map = tensor_map_4.iterator
                                            if is_shared_l1:
                                                output_map = tensor_map_13.iterator
                                            tma_store_fence()
                                            tile_s2g_2d(
                                                output_map, cd_stage,
                                                n_block * Uint32(64),
                                                output_block * logical_m +
                                                wg_start_m + Uint32(store * self.store_m_l1))
                                            tma_store_arrive()
                                    cute.arch.sync_warp()
                            tma_store_wait_0()
                        else:
                            # Source-contract W2 vector epilogue. The
                            # 16dp register-to-value bijection still
                            # requires validM>8 device verification.
                            is_shared_l2 = Uint32(0)
                            if cutlass.const_expr(self.shared_experts > 0):
                                is_shared_l2 = Uint32(
                                    task_phase == Uint32(GCK3BlockPhase.SharedLinear2))
                            epi_warp = Uint32(physical_warp - 4)
                            warpgroup = epi_warp // Uint32(4)
                            warp_in_wg = epi_warp % Uint32(4)
                            logical_m = Uint32(self.block_m)
                            if is_shared_l2:
                                logical_m = Uint32(self.shared_block_m)
                            wg_start_m = warpgroup * (logical_m // Uint32(2))
                            valid_wg_rows = Uint32(0)
                            if valid_m > wg_start_m:
                                valid_wg_rows = valid_m - wg_start_m
                                if valid_wg_rows > logical_m // Uint32(2):
                                    valid_wg_rows = logical_m // Uint32(2)
                            valid_stores = (valid_wg_rows + Uint32(self.store_m_l2 - 1)) // \
                                Uint32(self.store_m_l2)
                            n_block = n_cluster * Uint32(2) + cta_rank
                            cd_wg = cd_smem + warpgroup * Uint32(
                                self.store_m_l2 * 128 * 2)
                            raw_values = cute.make_rmem_tensor(
                                (2 if self.block_m == 240 else 1,
                                 self.store_m_l2 // 8, 8), Float32)
                            if valid_stores == Uint32(0):
                                _release_w2_tmem(accum_empty, accum_stage)
                            if cutlass.const_expr(self.block_m == 240):
                                if valid_stores > Uint32(0):
                                    base = accum_stage * Uint32(self.storage_block_m) + \
                                        wg_start_m
                                    _load_w2_atom(raw_values, 0, 0, base)
                                    if valid_stores == Uint32(1):
                                        cute.arch.fence_view_async_tmem_load()
                                        _release_w2_tmem(accum_empty, accum_stage)
                            for store in cutlass.range_constexpr(
                                    (self.storage_block_m // 2) // self.store_m_l2):
                                if Uint32(store) < valid_stores:
                                    if cutlass.const_expr(self.block_m != 240):
                                        for atom in cutlass.range_constexpr(
                                                self.store_m_l2 // 8):
                                            base = accum_stage * Uint32(self.storage_block_m) + \
                                                wg_start_m + Uint32(store * self.store_m_l2 + atom * 8)
                                            _load_w2_atom(raw_values, 0, atom, base)
                                            if atom == self.store_m_l2 // 8 - 1 and \
                                                    Uint32(store + 1) == valid_stores:
                                                cute.arch.fence_view_async_tmem_load()
                                                _release_w2_tmem(accum_empty, accum_stage)
                                    # Wait for the current atom before
                                    # issuing the BM240 next-store prefetch.
                                    cute.arch.fence_view_async_tmem_load()
                                    if cutlass.const_expr(self.block_m == 240):
                                        if Uint32(store + 1) < valid_stores:
                                            next_base = accum_stage * Uint32(self.storage_block_m) + \
                                                wg_start_m + Uint32((store + 1) * 8)
                                            _load_w2_atom(
                                                raw_values, (store + 1) % 2,
                                                0, next_base)
                                            if Uint32(store + 2) == valid_stores:
                                                cute.arch.fence_view_async_tmem_load()
                                                _release_w2_tmem(accum_empty, accum_stage)
                                    # One metadata record for each lane's
                                    # row in every 8-row atom.
                                    src_metadata = ()
                                    for atom in cutlass.range_constexpr(
                                            self.store_m_l2 // 8):
                                        token_in_block = wg_start_m + \
                                            Uint32(store * self.store_m_l2 + atom * 8) + \
                                            warp_in_wg * Uint32(2) + Uint32(lane // 16)
                                        pool_token = pool_block * logical_m + \
                                            token_in_block
                                        record = _load_w2_source_metadata(
                                            signals.iterator + self.metadata_words,
                                            token_in_block, valid_m, pool_token,
                                            rank_idx, is_shared_l2, self.topk,
                                            self.shared_experts > 0)
                                        src_metadata += (record,)
                                    cute.arch.sync_warp()
                                    for atom in cutlass.range_constexpr(
                                            self.store_m_l2 // 8):
                                        if atom == 0 and store > 0:
                                            sync_epilogue_warpgroup(warpgroup)
                                        row = Uint32(lane % 8)
                                        col = (epi_warp % Uint32(2)) * Uint32(4) + \
                                            Uint32(lane // 8)
                                        smem_store_offset = (warp_in_wg // Uint32(2)) * \
                                            Uint32(self.store_m_l2 * 128) + \
                                            Uint32(atom * 8 * 128) + row * Uint32(128) + \
                                            (col ^ row) * Uint32(16)
                                        buffer = store % 2 if self.block_m == 240 else 0
                                        stmatrix_bf16x8_trans(
                                            cd_wg + smem_store_offset,
                                            pack_bf16_pair(raw_values[buffer, atom, 0],
                                                            raw_values[buffer, atom, 1]),
                                            pack_bf16_pair(raw_values[buffer, atom, 2],
                                                            raw_values[buffer, atom, 3]),
                                            pack_bf16_pair(raw_values[buffer, atom, 4],
                                                            raw_values[buffer, atom, 5]),
                                            pack_bf16_pair(raw_values[buffer, atom, 6],
                                                            raw_values[buffer, atom, 7]))
                                    sync_epilogue_warpgroup(warpgroup)
                                    for atom in cutlass.range_constexpr(
                                            self.store_m_l2 // 8):
                                        row_in_atom = warp_in_wg * Uint32(2) + \
                                            Uint32(lane // 16)
                                        row_in_store = Uint32(atom * 8) + row_in_atom
                                        token_in_block = wg_start_m + \
                                            Uint32(store * self.store_m_l2) + row_in_store
                                        if token_in_block < valid_m:
                                            bank_group = Uint32(lane % 8)
                                            half = (Uint32(lane) % Uint32(16)) // Uint32(8)
                                            smem_load_offset = half * \
                                                Uint32(self.store_m_l2 * 128) + \
                                                row_in_store * Uint32(128) + \
                                                (bank_group ^ row_in_atom) * Uint32(16)
                                            words = ld_shared_u32x4(
                                                cd_wg + smem_load_offset)
                                            dst = cute.make_ptr(Uint32,signals.iterator.toint(),cute.AddressSpace.gmem,assumed_align=16)
                                            if is_shared_l2:
                                                # finalize consumes the BF16 shared
                                                # output directly from the workspace.
                                                local_token = pool_block*logical_m+token_in_block
                                                channel_bytes = n_block*Uint32(256) + (Uint32(lane)%Uint32(16))*Uint32(16)
                                                dst = cute.make_ptr(Uint32,
                                                    Uint64(y_out.iterator.toint()) +
                                                    Uint64(local_token)*Uint64(self.shared_hidden*2) +
                                                    Uint64(channel_bytes),
                                                    cute.AddressSpace.gmem,assumed_align=16)
                                            else:
                                                token_bytes = Uint64(self.hidden * 2)
                                                slot_bytes = Uint64(self.capacity) * token_bytes
                                                channel_bytes = n_block * Uint32(256) + \
                                                    (Uint32(lane) % Uint32(16)) * Uint32(16)
                                                dst_bytes = (Uint64(src_metadata[atom][2]) * slot_bytes +
                                                             Uint64(src_metadata[atom][1]) * token_bytes +
                                                             Uint64(channel_bytes))
                                                dst = cute.make_ptr(
                                                    Uint32, mapped_peer_base(
                                                        peer_ptrs, src_metadata[atom][0],
                                                        self.num_ranks) +
                                                    Uint64(self.combine_words * 4) + \
                                                    dst_bytes,
                                                    cute.AddressSpace.gmem,
                                                    assumed_align=16)
                                            st_global_u32x4(
                                                dst, words[0], words[1],
                                                words[2], words[3])
                        sync_aligned(256, 2)
                    is_w1_release = task_phase == Uint32(GCK3BlockPhase.Linear1)
                    if cutlass.const_expr(self.shared_experts > 0):
                        is_w1_release = is_w1_release or \
                            task_phase == Uint32(GCK3BlockPhase.SharedLinear1)
                    if is_w1_release:
                        if physical_warp == 4:
                            if elect_one_sync():
                                ring_block = pool_block % self.ring_blocks
                                n_block = n_cluster * 2 + cta_rank
                                if task_phase == Uint32(GCK3BlockPhase.Linear1):
                                    red_release_gpu_xor_u64(
                                        signals.iterator + SIGNAL_OFFSETS['l2_full_mask'] // 4 +
                                        ring_block * 2,
                                        Uint64(1) << n_block)
                                    red_gpu_add_u32(
                                        signals.iterator + SIGNAL_OFFSETS['l1_empty_count'] // 4 +
                                        ring_block, Uint32(1))
                                else:
                                    if cutlass.const_expr(self.shared_experts > 0):
                                        if task_phase == Uint32(GCK3BlockPhase.SharedLinear1):
                                            shared_ready = signals.iterator + \
                                                SIGNAL_OFFSETS['shared_l2_full_count'] // 4
                                            if cutlass.const_expr(self.shared_w2_stage_ready):
                                                # The W1 stores and warpgroup join above
                                                # retire this CTA's 64 output channels.
                                                shared_mask = cute.recast_ptr(
                                                    shared_ready, dtype=Uint64) + pool_block * Uint32(self.shared_w2_mask_words)
                                                red_release_gpu_xor_u64(
                                                    shared_mask + n_block // Uint32(64),
                                                    Uint64(1) << (n_block & Uint32(63)))
                                            else:
                                                red_release_gpu_add_u32(shared_ready + pool_block,
                                                                        Uint32(1))
                        cute.arch.sync_warp()
                    running = task_phase != Uint32(GCK3BlockPhase.None_)
                    iteration = Uint32(iteration + Uint32(1))
                # The same full epilogue warp in both CTAs retires TMEM.
                if physical_warp == 4:
                    tmem_free_2cta(tptr[0], Uint32(self.tmem_cols))
            if cutlass.const_expr(self.early_shared):
                # All threads complete the Front dependency before returning.
                cute.arch.griddepcontrol_wait()
        else:
            # After both graph workers finish, positive-token ranks
            # combine, normalize, project and add. Zero-token ranks still
            # publish peer completion and retire the workspace generation.
            combine_alloc = cutlass.utils.SmemAllocator()
            combine_smem = cute.recast_ptr(
                combine_alloc.allocate(self.shared_layout.field_end, 1024), dtype=Uint8)
            cute.arch.griddepcontrol_wait()
            # Dispatch's epoch remains immutable throughout this Finalize launch.
            # Each warp caches it independently; the footer advances recv_parity.
            ready_epoch = Uint64(ld_volatile_u32(
                signals.iterator + SIGNAL_OFFSETS['finalize_epoch'] // 4))
            if cutlass.const_expr(self.projection_tile_rows > 0):
                self._finalize_output(signals, peer_ptrs, rank_idx, num_tokens, y_out,
                                 routed_norm, norm_eps, routed_scale, final_out, shared_out, prefix_sum,
                                 projection_weight_map.iterator, projection_latent_map.iterator,
                                 combine_smem, ready_epoch,
                                 Uint32(block), Uint32(tid), Uint32(physical_warp), Uint32(lane))
            elif physical_warp < 4:
                self._retire_workspace(signals, peer_ptrs, rank_idx,
                                     Uint32(tid), Uint32(block), ready_epoch)

    @cute.jit
    def __call__(self, signals: cute.Tensor, peer_ptrs: cute.Tensor,
                 rank_idx: Uint32, num_tokens: Uint32,
                 tensor_maps=None, y_out=None, activation_clamp=None,
                 tensor_map_1=None, tensor_map_2=None, tensor_map_3=None,
                 tensor_map_4=None, tensor_map_5=None, tensor_map_6=None,
                 tensor_map_7=None, tensor_map_8=None, tensor_map_9=None,
                 tensor_map_10=None, tensor_map_11=None, tensor_map_12=None,
                 tensor_map_13=None, tensor_map_14=None, tensor_map_15=None,
                 tensor_map_16=None, tensor_map_17=None, valid_token_count=None,
                 routed_norm=None, norm_eps=None, routed_scale=None, final_out=None, shared_out=None, prefix_sum=None,
               projection_weight_map=None, projection_latent_map=None):
        if cutlass.const_expr(self.role == DISPATCH_ROLE):
            self.kernel(signals, peer_ptrs, rank_idx, num_tokens).launch(
                grid=(self.dispatch_sms, 1, 1),
                block=(self.dispatch_warps * 32, 1, 1),
                max_number_threads=(self.dispatch_warps * 32, 1, 1),
                min_blocks_per_mp=1)
        elif cutlass.const_expr(self.role == COMPUTE_ROLE):
            self.kernel(signals, peer_ptrs, rank_idx, num_tokens, tensor_maps,
                        y_out, activation_clamp,
                        tensor_map_1, tensor_map_2, tensor_map_3,
                        tensor_map_4, tensor_map_5, tensor_map_6,
                        tensor_map_7, tensor_map_8, tensor_map_9,
                        tensor_map_10, tensor_map_11, tensor_map_12,
                        tensor_map_13, tensor_map_14, tensor_map_15,
                        tensor_map_16, tensor_map_17, valid_token_count).launch(
                grid=(self.dispatch_sms if self.routed_assist else self.compute_sms, 1, 1), block=(384, 1, 1),
                max_number_threads=(384, 1, 1),
                cluster=(2, 1, 1), min_blocks_per_mp=1)
        else:
            self.kernel(signals, peer_ptrs, rank_idx, num_tokens, y_out=y_out,
                        routed_norm=routed_norm, norm_eps=norm_eps, routed_scale=routed_scale,
                        final_out=final_out, shared_out=shared_out, prefix_sum=prefix_sum,
                        projection_weight_map=projection_weight_map, projection_latent_map=projection_latent_map).launch(
                grid=(self.finalize_sms, 1, 1),
                block=(FINALIZE_THREADS, 1, 1),
                max_number_threads=(FINALIZE_THREADS, 1, 1),
                cluster=(2, 1, 1) if cutlass.const_expr(self.projection_tile_rows > 0) else (1, 1, 1),
                min_blocks_per_mp=1)
