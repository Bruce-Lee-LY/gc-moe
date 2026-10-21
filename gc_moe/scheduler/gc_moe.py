"""Two-CTA GC task handoff and the four task phase identifiers.

The 32-byte task lives in each CTA's SMEM. The scheduler warp in the leader
CTA publishes both copies with transaction-counted DSM stores; the two
epilogue warpgroups per CTA retire a slot through the leader empty barrier.
"""

from __future__ import annotations

from enum import IntEnum

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from ..comm.gc_barrier import atom_add_u32, elect_one_sync


class GCBlockPhase(IntEnum):
    None_ = 0
    Linear1 = 1
    Linear2 = 2
    SharedLinear1 = 3
    SharedLinear2 = 4


TASK_WORDS = 8
TASK_BYTES = 32
SCHEDULE_STAGES = 2
TASK_EMPTY_CONSUMERS = 512


@dsl_user_op
def mapa_shared(ptr: cute.Pointer, peer_cta: Int32, *, loc=None, ip=None) -> Int32:
    address = ptr.toint(loc=loc, ip=ip).ir_value()
    mapped = llvm.inline_asm(
        T.i32(), [address, peer_cta.ir_value(loc=loc, ip=ip)],
        'mapa.shared::cluster.u32 $0, $1, $2;', '=r,r,r',
        has_side_effects=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Int32(mapped)


@dsl_user_op
def store_task_v4(task_ptr: cute.Pointer, mbar_ptr: cute.Pointer,
                  peer_cta: Int32, word_offset: Int32,
                  a: Uint32, b: Uint32, c: Uint32, d: Uint32,
                  *, loc=None, ip=None) -> None:
    task_address = mapa_shared(task_ptr + word_offset, peer_cta, loc=loc, ip=ip)
    mbar_address = mapa_shared(mbar_ptr, peer_cta, loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [task_address.ir_value(loc=loc, ip=ip),
         a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip),
         c.ir_value(loc=loc, ip=ip), d.ir_value(loc=loc, ip=ip),
         mbar_address.ir_value(loc=loc, ip=ip)],
        'st.async.shared::cluster.mbarrier::complete_tx::bytes.u32.v4 '
        '[$0], {$1, $2, $3, $4}, [$5];',
        'r,r,r,r,r,r', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)


@cute.jit
def publish_task32(task_ptr: cute.Pointer, full_ptr: cute.Pointer,
                   stage: Int32, peer_cta: Int32,
                   block_phase: Uint32, local_expert: Uint32,
                   m_block: Uint32, n_cluster: Uint32,
                   pool_block: Uint32, valid_m: Uint32,
                   shape_n: Uint32, shape_k: Uint32) -> None:
    cute.arch.mbarrier_arrive_and_expect_tx(full_ptr + stage, TASK_BYTES, peer_cta)
    base = stage * TASK_WORDS
    store_task_v4(task_ptr, full_ptr + stage, peer_cta, base,
                  block_phase, local_expert, m_block, n_cluster)
    store_task_v4(task_ptr, full_ptr + stage, peer_cta, base + 4,
                  pool_block, valid_m, shape_n, shape_k)


@cute.jit
def release_task_info(empty_ptr: cute.Pointer, stage: Int32) -> None:
    cute.arch.fence_acq_rel_cta()
    cute.arch.mbarrier_arrive(empty_ptr + stage, peer_cta_rank_in_cluster=0)


@cute.jit
def claim_task(task_count_ptr: cute.Pointer, lane: Int32) -> Uint32:
    task_idx = Uint32(0)
    if elect_one_sync():
        task_idx = atom_add_u32(task_count_ptr, Uint32(1))
    return cute.arch.shuffle_sync(task_idx, 0)


@cute.jit
def wait_task_slot(empty_ptr: cute.Pointer, iteration: Uint32) -> None:
    """Wait for the next task slot before entering its claim loop."""
    stage = Int32(iteration % 2)
    parity = Int32((iteration // 2) % 2)
    cute.arch.mbarrier_wait(empty_ptr + stage, parity ^ 1)


@cute.jit
def publish_iteration(task_ptr: cute.Pointer, full_ptr: cute.Pointer,
                      iteration: Uint32,
                      lane: Int32, block_phase: Uint32,
                      local_expert: Uint32,
                      m_block: Uint32, n_cluster: Uint32,
                      pool_block: Uint32, valid_m: Uint32,
                      shape_n: Uint32, shape_k: Uint32) -> None:
    stage = Int32(iteration % 2)
    if lane < 2:
        publish_task32(task_ptr, full_ptr, stage, lane,
                       block_phase, local_expert, m_block, n_cluster,
                       pool_block, valid_m, shape_n, shape_k)
    cute.arch.sync_warp()


@cute.jit
def publish_shared_phase(task_ptr: cute.Pointer, full_ptr: cute.Pointer,
                         empty_ptr: cute.Pointer, counter_ptr: cute.Pointer,
                         iteration: Uint32, lane: Int32,
                         cluster_idx: Uint32,
                         phase: Uint32, num_tokens: Uint32,
                         block_m: cutlass.Constexpr,
                         n_clusters: cutlass.Constexpr,
                         num_compute_sms: cutlass.Constexpr,
                         shape_n: cutlass.Constexpr,
                         shape_k: cutlass.Constexpr) -> Uint32:
    """Reference shared_mainloop: static one task/cluster, else global claims."""
    m_blocks = (num_tokens + Uint32(block_m - 1)) // Uint32(block_m)
    total = m_blocks * Uint32(n_clusters)
    if total <= Uint32(num_compute_sms // 2):
        wait_task_slot(empty_ptr, iteration)
        if cluster_idx < Uint32(total):
            m_block = cluster_idx // Uint32(n_clusters)
            n_cluster = cluster_idx % Uint32(n_clusters)
            valid_m = min(Uint32(block_m), Uint32(num_tokens) - m_block * Uint32(block_m))
            publish_iteration(task_ptr, full_ptr, iteration,
                              lane, phase, Uint32(0), m_block, n_cluster, m_block,
                              valid_m, Uint32(shape_n), Uint32(shape_k))
            iteration += Uint32(1)
            # Match the reference's wait before the static path's failed claim.
            wait_task_slot(empty_ptr, iteration)
    else:
        scheduling = True
        while scheduling:
            wait_task_slot(empty_ptr, iteration)
            task_idx = claim_task(counter_ptr, lane)
            if task_idx >= Uint32(total):
                scheduling = False
            else:
                m_block = task_idx // Uint32(n_clusters)
                n_cluster = task_idx % Uint32(n_clusters)
                valid_m = min(Uint32(block_m), Uint32(num_tokens) - m_block * Uint32(block_m))
                publish_iteration(task_ptr, full_ptr, iteration,
                                  lane, phase, Uint32(0), m_block, n_cluster, m_block,
                                  valid_m, Uint32(shape_n), Uint32(shape_k))
                iteration += Uint32(1)
    return iteration
