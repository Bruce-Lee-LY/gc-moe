"""Compile-time geometry for the single persistent 2CTA GC math worker.

The worker consumes `GCTaskInfo` for routed W1/W2 and shared W1/W2 in one
cluster. These values size its SMEM/TMEM stages and encode per-K-block L2
dependencies; they are not a separate dense-kernel launch configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Uint16, Uint32, Uint64
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import arith

from ..ptx.tcgen05 import (
    mma_mxf8f6f4_2cta,
    tcgen05_after_thread_sync,
    tcgen05_commit_2cta,
    utccp_sf_4x32_2cta,
)
from ..ptx.tma import EVICT_FIRST, tile_g2s_2cta_2d, tile_g2s_2cta_4d


@dataclass(frozen=True)
class MixedMmaTile:
    block_m: int
    hidden: int
    intermediate: int
    block_k: int = 128
    block_n: int = 128
    epilogue_stages: int = 2

    def __post_init__(self) -> None:
        if self.block_m not in (8, 16, 32, 64, 96, 128, 192, 240) or self.block_m % 16:
            raise ValueError('2CTA mixed MMA requires a 16-aligned routed BM')
        if self.block_n != 128 or self.block_k != 128:
            raise ValueError('GC mixed MMA uses BN128 and BK128')
        if self.hidden % 256 or self.intermediate % 128:
            raise ValueError('GC two-CTA N geometry is not integral')
        if self.l1_n_blocks > 64:
            raise ValueError('L2 dependency mask exceeds 64 bits')

    @property
    def sf_block_m(self) -> int:
        return (self.block_m + 127) // 128 * 128

    @property
    def l1_n_blocks(self) -> int:
        return 2 * self.intermediate // self.block_n

    @property
    def l2_n_blocks(self) -> int:
        return self.hidden // self.block_n

    @property
    def l1_clusters(self) -> int:
        return self.l1_n_blocks // 2

    @property
    def l2_clusters(self) -> int:
        return self.l2_n_blocks // 2

    @property
    def accum_tmem_cols(self) -> int:
        return self.block_m * self.epilogue_stages

    @property
    def sfa_tmem_cols(self) -> int:
        return self.sf_block_m // 32

    @property
    def sfb_tmem_cols(self) -> int:
        return self.block_n // 32

    @property
    def tmem_cols(self) -> int:
        needed = self.accum_tmem_cols + self.sfa_tmem_cols + self.sfb_tmem_cols
        result = max(32, 1 << (needed - 1).bit_length())
        if result > 512:
            raise ValueError('GC TMEM allocation exceeds 512 columns')
        return result

    @property
    def smem_a_stage_bytes(self) -> int:
        return self.block_m // 2 * self.block_k  # FP8 activation

    @property
    def smem_b_stage_bytes(self) -> int:
        return self.block_n * self.block_k  # packed FP4 TMA-unpacked to 8b SMEM

    @property
    def smem_sfa_stage_bytes(self) -> int:
        return self.sf_block_m * (self.block_k // 128) * 4

    @property
    def smem_sfb_stage_bytes(self) -> int:
        return self.block_n * (self.block_k // 128) * 4

    def l2_k_mask(self, k_block: int) -> int:
        """Two completed W1 N128 tiles feed one W2 K128 tile."""
        width = self.block_k // (self.block_n // 2)
        if k_block < 0 or (k_block + 1) * width > self.l1_n_blocks:
            raise ValueError('invalid L2 K block')
        return ((1 << width) - 1) << (k_block * width)


@cute.jit
def load_mixed_k128_tma(activation_map: cute.Pointer,
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
                        shape_n: Uint32, shape_k: Uint32,
                        is_shared: Uint32, cta_rank: Uint32,
                        loader_warp: cutlass.Constexpr,
                        block_m: cutlass.Constexpr,
                        sf_block_m: cutlass.Constexpr,
                        shared_weight_gran_k: cutlass.Constexpr,
                        has_shared: cutlass.Constexpr) -> None:
    """Warp 0/1 stage the same K128 task through 2CTA TensorMaps.

    Both CTAs issue their own TMA transfers into local SMEM. The leader's
    full barrier counts two arrivals per CTA and source-byte transaction
    counts for both CTAs. The caller waits it before issuing mixed MMA.
    """
    cute.arch.mbarrier_wait(empty_bar, stage_phase ^ Uint32(1))
    k_idx = k_block * Uint32(128)
    cache_hint = Uint64(0x1000000000000000)
    if cutlass.const_expr(loader_warp == 0):
        aligned_m = (valid_m + Uint32(15)) & Uint32(~15)
        m_idx = block_idx * Uint32(block_m)
        if cta_rank != Uint32(0):
            m_idx += aligned_m // Uint32(2)
        with cute.arch.elect_one():
            tile_g2s_2cta_2d(activation_smem, activation_map, full_bar,
                             k_idx, m_idx, cache_hint)
            tile_g2s_2cta_2d(activation_sf_smem, activation_sf_map, full_bar,
                             block_idx * Uint32(sf_block_m), k_block,
                             cache_hint)
            if cta_rank == Uint32(0):
                cute.arch.mbarrier_arrive_and_expect_tx(
                    full_bar, 2 * ((block_m // 2) * 128 + sf_block_m * 4))
            else:
                cute.arch.mbarrier_arrive(full_bar, peer_cta_rank_in_cluster=0)
    else:
        n_idx = n_block * Uint32(128)
        half_n = shape_n // Uint32(2)
        domain = Uint32(1) if n_idx >= half_n else Uint32(0)
        half_n_idx = n_idx - domain * half_n
        # Evict routed FP4 weights first. Shared FP8 weights and
        # both scale-factor maps retain the normal cache hint.
        weight_cache_hint = Uint64(EVICT_FIRST)
        gran = Uint32(32)
        if cutlass.const_expr(has_shared):
            weight_cache_hint = (Uint64(EVICT_FIRST) if is_shared == Uint32(0)
                                 else cache_hint)
            if cutlass.const_expr(shared_weight_gran_k == 128):
                if is_shared != Uint32(0):
                    gran = Uint32(128)
        sf_stages = gran // Uint32(32)
        sf_index = k_block // sf_stages
        sf_packs_per_expert = (shape_k + gran * Uint32(4) - Uint32(1)) // \
            (gran * Uint32(4))
        load_sf = k_block % sf_stages == Uint32(0)
        weight_bytes = Uint32(128 * 128 // 2)
        if cutlass.const_expr(has_shared):
            if is_shared != Uint32(0):
                weight_bytes = Uint32(128 * 128)
        sf_bytes = Uint32(128 * 4) if load_sf else Uint32(0)
        with cute.arch.elect_one():
            tile_g2s_2cta_4d(weight_smem, weight_map, full_bar,
                             k_idx, half_n_idx, domain, local_expert,
                             weight_cache_hint)
            if load_sf:
                tile_g2s_2cta_2d(weight_sf_smem, weight_sf_map, full_bar,
                                 n_idx, local_expert * sf_packs_per_expert + sf_index,
                                 cache_hint)
            if cta_rank == Uint32(0):
                cute.arch.mbarrier_arrive_and_expect_tx(
                    full_bar, (weight_bytes + sf_bytes) * Uint32(2))
            else:
                cute.arch.mbarrier_arrive(full_bar, peer_cta_rank_in_cluster=0)
    cute.arch.sync_warp()


@dsl_user_op
def _select_sfb_id(is_shared: Uint32, shared_id: Uint32, routed_id: Uint32,
                   *, loc=None, ip=None) -> Uint32:
    """Pure reference ternary for the shared/routed SFB byte ID."""
    is_shared_pred = (is_shared != Uint32(0)).ir_value(loc=loc, ip=ip)
    return Uint32(arith.select(
        is_shared_pred, shared_id.ir_value(loc=loc, ip=ip),
        routed_id.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@cute.jit
def issue_mixed_k128(weight_desc_start: Uint32, activation_desc_start: Uint32,
                     weight_sf_smem: cute.Pointer,
                     activation_sf_smem: cute.Pointer, stage_idx: Uint32,
                     data_desc_hi: Uint64, sf_desc_hi: Uint64,
                     accum_tmem: Uint32, weight_sf_tmem: Uint32,
                     activation_sf_tmem: Uint32,
                     stage_empty_bar: cute.Pointer, accum_full_bar: cute.Pointer,
                     instr_desc_base: Uint32, k_block: Uint32,
                     last_k_block: Uint32, is_shared: Uint32,
                     sf_block_m: cutlass.Constexpr,
                     shared_weight_gran_k: cutlass.Constexpr,
                     has_shared: cutlass.Constexpr) -> None:
    """Issue the reference swapped-A/B 2CTA mixed MMA for one K128 stage.

    The caller has already completed the stage's TMA G2S and mbarrier wait.
    It gives this function the per-stage A/B descriptor bases, the two SF
    shared-memory bases, and descriptor high fields retained by the leader
    warp across tasks and K stages.
    A=weight, B=activation in the instruction;
    routed FP4 uses format 5, shared FP8 format 0. Every K128 commits to the
    stage-empty barrier; the last K128 also commits to accumulator-full.
    """
    tcgen05_after_thread_sync()
    # The leader warp has already selected one task slot and pipeline stage.
    # Only stage A/B descriptor bases are prepared in the full warp. The
    # reference computes SF addresses, SF selection and K32 descriptors in
    # the single elected issuer after the stage-full wait.
    weight_start = weight_desc_start
    activation_start = activation_desc_start

    with cute.arch.elect_one():
        activation_sf_start = Uint32(
            (Uint32((activation_sf_smem + stage_idx * sf_block_m).toint()) >>
             Uint32(4)))
        activation_sf_desc = sf_desc_hi | Uint64(activation_sf_start)
        utccp_sf_4x32_2cta(activation_sf_desc, activation_sf_tmem)
        if cutlass.const_expr(sf_block_m > 128):
            activation_sf_desc = sf_desc_hi | Uint64(
                activation_sf_start + Uint32(32))
            utccp_sf_4x32_2cta(activation_sf_desc,
                               activation_sf_tmem + Uint32(4))
        copy_weight_sf = True
        if cutlass.const_expr(has_shared and shared_weight_gran_k == 128):
            copy_weight_sf = (is_shared == Uint32(0) or
                              k_block % Uint32(4) == Uint32(0))
        if copy_weight_sf:
            weight_sf_start = Uint32(
                (Uint32((weight_sf_smem + stage_idx * 128).toint()) >>
                 Uint32(4)))
            weight_sf_desc = sf_desc_hi | Uint64(weight_sf_start)
            utccp_sf_4x32_2cta(weight_sf_desc, weight_sf_tmem)

        # FP4 16U4_ALIGN16B TMA expands sixteen nibbles into an 8-byte
        # payload and 8-byte padding. K32 advances each descriptor low by
        # two 16-byte address units. Keep every mutation in the elected lane.
        for k in cutlass.range_constexpr(4):
            weight_sf_id = Uint32(k)
            if cutlass.const_expr(has_shared and shared_weight_gran_k == 128):
                weight_sf_id = _select_sfb_id(
                    is_shared, k_block % Uint32(4), Uint32(k))
            runtime_instr_desc = (
                instr_desc_base | (weight_sf_id << Uint32(29)) |
                Uint32(16 * k))
            weight_desc = data_desc_hi | Uint64(
                weight_start + Uint32(2 * k))
            activation_desc = data_desc_hi | Uint64(
                activation_start + Uint32(2 * k))
            accumulate = (Uint32(1) if k > 0 else
                          Uint32(1) if k_block > Uint32(0) else Uint32(0))
            mma_mxf8f6f4_2cta(
                weight_desc, activation_desc, accum_tmem, accumulate,
                runtime_instr_desc, weight_sf_tmem, activation_sf_tmem)
        tcgen05_commit_2cta(stage_empty_bar, Uint16(3))
        if last_k_block != Uint32(0):
            tcgen05_commit_2cta(accum_full_bar, Uint16(3))
    cute.arch.sync_warp()
