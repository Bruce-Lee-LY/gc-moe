"""Thread-group, grid and cross-rank synchronization protocols."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Uint32, Uint64

from ..layout.gc_moe import SIGNAL_OFFSETS
from ..layout.sym_buffer import mapped_peer_base
from ..ptx.ld_st import (
    atom_release_gpu_add_u32,
    ld_acquire_gpu_u32,
    ld_acquire_sys_u32,
    ld_acquire_sys_u64,
    red_gpu_add_u32,
    red_release_sys_add_u32,
)
from ..ptx.utils import (
    device_trap,
    read_clock64,
    sync_aligned,
)


DEFAULT_NVLINK_BARRIER_TIMEOUT_SECS = 3600
_WAIT_TIMEOUT_CYCLES = 60 * 2_000_000_000


@cute.jit
def sync_dispatch_threads(participants: cutlass.Constexpr = 0) -> None:
    """Dispatch barrier id 0; zero preserves the existing whole-CTA scope.

    The caller must have exactly ``participants`` threads enter each nonzero
    scope: role3 dispatch uses 128; role1 uses its full 256/384-thread block.
    """
    if cutlass.const_expr(participants == 128):
        sync_aligned(128, 0)
    elif cutlass.const_expr(participants == 256):
        sync_aligned(256, 0)
    elif cutlass.const_expr(participants == 384):
        sync_aligned(384, 0)
    else:
        cute.arch.sync_threads()


@cute.jit
def wait_sys_u64_eq(ptr: cute.Pointer, target: Uint64,
                    rank_idx: Uint32, peer: Uint32) -> None:
    """Reference 60-second per-peer combine readiness wait."""
    start_clock = read_clock64()
    observed = ld_acquire_sys_u64(ptr)
    while observed != target:
        if Uint64(read_clock64() - start_clock) >= Uint64(_WAIT_TIMEOUT_CYCLES):
            cute.printf('GC MoE combine readiness timeout: rank={} peer={} ready={}',
                        rank_idx, peer, ld_acquire_sys_u64(ptr))
            device_trap()
        observed = ld_acquire_sys_u64(ptr)


@cute.jit
def grid_sync(signals: cute.Tensor, block: Uint32, tid: Uint32,
              num_sms: cutlass.Constexpr, counter_index: cutlass.Constexpr,
              sync_participants: cutlass.Constexpr = 0) -> None:
    """Reference high-bit tagged grid barrier; the counter survives replays."""
    sync_dispatch_threads(sync_participants)
    if tid == Uint32(0):
        counter = signals.iterator + SIGNAL_OFFSETS['grid_sync_count'] // 4 + counter_index
        increment = (Uint32(0x80000000) - Uint32(num_sms - 1)) \
            if block == Uint32(0) else Uint32(1)
        previous = atom_release_gpu_add_u32(counter, increment)
        current = previous
        start_clock = read_clock64()
        while ((current ^ previous) & Uint32(0x80000000)) == Uint32(0):
            current = ld_acquire_gpu_u32(counter)
            if ((current ^ previous) & Uint32(0x80000000)) == Uint32(0) and \
                    Uint64(read_clock64() - start_clock) >= Uint64(_WAIT_TIMEOUT_CYCLES):
                cute.printf(
                    'GC MoE grid sync timeout: sm={} thread={} grid_sync_idx={} '
                    'old={} current={} expected_tag={}',
                    block, tid, counter_index, previous, current,
                    previous ^ Uint32(0x80000000))
                device_trap()
    sync_dispatch_threads(sync_participants)


@cute.jit
def nvlink_barrier(signals: cute.Tensor, peer_ptrs: cute.Tensor,
                   block: Uint32, tid: Uint32,
                   num_ranks: cutlass.Constexpr, num_sms: cutlass.Constexpr,
                   pre_grid: cutlass.Constexpr,
                   post_grid: cutlass.Constexpr,
                   rank_idx: Uint32, barrier_tag: cutlass.Constexpr,
                   timeout_secs: cutlass.Constexpr,
                   sync_participants: cutlass.Constexpr = 0) -> None:
    """Two-phase rank barrier with system release/acquire peer signals."""
    if cutlass.const_expr(pre_grid):
        grid_sync(signals, block, tid, num_sms, 0, sync_participants)
    if block == Uint32(0):
        full_counter = Uint32(signals[SIGNAL_OFFSETS['nvl_barrier_counter'] // 4])
        status = full_counter & Uint32(3)
        signal_phase = status & Uint32(1)
        decrement = (status & Uint32(2)) != Uint32(0)
        if tid < Uint32(num_ranks):
            peer_signal = cute.make_ptr(
                Uint32, mapped_peer_base(peer_ptrs, tid, num_ranks) +
                Uint64(SIGNAL_OFFSETS['nvl_barrier_signals']) +
                Uint64(signal_phase * Uint32(4)), cute.AddressSpace.gmem,
                assumed_align=4)
            red_release_sys_add_u32(peer_signal,
                                    Uint32(0xffffffff) if decrement else Uint32(1))
        sync_dispatch_threads(sync_participants)
        if tid == Uint32(0):
            signal = signals.iterator + SIGNAL_OFFSETS['nvl_barrier_signals'] // 4 + \
                signal_phase
            red_gpu_add_u32(signals.iterator +
                            SIGNAL_OFFSETS['nvl_barrier_counter'] // 4, Uint32(1))
            target = Uint32(0) if decrement else Uint32(num_ranks)
            start_clock = read_clock64()
            seen = ld_acquire_sys_u32(signal)
            while seen != target:
                if Uint64(read_clock64() - start_clock) >= \
                        Uint64(timeout_secs * 2000000000):
                    cute.printf(
                        'GC MoE NVLink barrier timeout ({}s): rank={} block={} '
                        'thread={} counter={} signal={} target={} phase={} sign={} tag={}',
                        timeout_secs, rank_idx, block, tid,
                        Uint32(signals[SIGNAL_OFFSETS['nvl_barrier_counter'] // 4]),
                        ld_acquire_sys_u32(signal), target, signal_phase,
                        status >> Uint32(1), barrier_tag)
                    device_trap()
                seen = ld_acquire_sys_u32(signal)
    if cutlass.const_expr(post_grid):
        grid_sync(signals, block, tid, num_sms, 0, sync_participants)
