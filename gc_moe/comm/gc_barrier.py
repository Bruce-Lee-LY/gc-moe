"""GPU/system scope primitives shared by the CuTe GC role kernel.

The full grid/NVLink algorithms are implemented where the symmetric buffer
and rank mapping are available.  These operations retain the reference scopes.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32, Uint32, Uint64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm, nvvm

from ..layout.gc_moe import SIGNAL_OFFSETS


@dsl_user_op
def elect_one_sync(*, loc=None, ip=None) -> Boolean:
    """Select one issuer with all threads of the active warp participating."""
    return Boolean(nvvm.elect_sync(loc=loc, ip=ip))


@dsl_user_op
def ld_volatile_u32(ptr: cute.Pointer, *, loc=None, ip=None) -> Uint32:
    address = ptr.toint(loc=loc, ip=ip).ir_value()
    value = llvm.inline_asm(T.i32(), [address],
                            'ld.volatile.global.b32 $0, [$1];', '=r,l',
                            has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint32(value)


@dsl_user_op
def ld_volatile_u64(ptr: cute.Pointer, *, loc=None, ip=None) -> Uint64:
    address = ptr.toint(loc=loc, ip=ip).ir_value()
    value = llvm.inline_asm(T.i64(), [address],
                            'ld.volatile.global.b64 $0, [$1];', '=l,l',
                            has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint64(value)


@dsl_user_op
def ld_global_u64(ptr: cute.Pointer, *, loc=None, ip=None) -> Uint64:
    """Ordinary 64-bit load for source counts after the rank rendezvous."""
    address = ptr.toint(loc=loc, ip=ip).ir_value()
    value = llvm.inline_asm(T.i64(), [address],
                            'ld.global.u64 $0, [$1];', '=l,l',
                            has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint64(value)


@dsl_user_op
def ld_peer_param_u64(ptr: cute.Pointer, *, loc=None, ip=None) -> Uint64:
    """Read one field of the by-value SymBuffer kernel parameter."""
    address = ptr.toint(loc=loc, ip=ip).ir_value()
    value = llvm.inline_asm(
        T.i64(), [address],
        '{ .reg .b64 p; cvta.to.param::entry.u64 p, $1; '
        'ld.param.u64 $0, [p]; }',
        '=l,l', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return Uint64(value)


@cute.jit
def mapped_peer_base(peer_info: cute.Tensor, rank: Uint32,
                     num_ranks: cutlass.Constexpr) -> Uint64:
    """Reference SymBuffer::map(base, rank) using signed relative bytes."""
    base = ld_peer_param_u64(peer_info.iterator)
    if cutlass.const_expr(num_ranks == 1):
        return base
    delta = ld_peer_param_u64(peer_info.iterator + Uint32(1) + rank)
    return base + delta


@dsl_user_op
def ld_acquire_gpu_u32(ptr: cute.Pointer, *, loc=None, ip=None) -> Uint32:
    address = ptr.toint(loc=loc, ip=ip).ir_value()
    value = llvm.inline_asm(T.i32(), [address],
                            'ld.acquire.gpu.global.u32 $0, [$1];', '=r,l,~{memory}',
                            has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint32(value)


@dsl_user_op
def ld_acquire_gpu_u64(ptr: cute.Pointer, *, loc=None, ip=None) -> Uint64:
    address = ptr.toint(loc=loc, ip=ip).ir_value()
    value = llvm.inline_asm(T.i64(), [address],
                            'ld.acquire.gpu.global.u64 $0, [$1];', '=l,l,~{memory}',
                            has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint64(value)


@dsl_user_op
def ld_acquire_sys_u64(ptr: cute.Pointer, *, loc=None, ip=None) -> Uint64:
    address = ptr.toint(loc=loc, ip=ip).ir_value()
    value = llvm.inline_asm(T.i64(), [address],
                            'ld.acquire.sys.global.u64 $0, [$1];', '=l,l,~{memory}',
                            has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint64(value)


@dsl_user_op
def ld_acquire_sys_u32(ptr: cute.Pointer, *, loc=None, ip=None) -> Uint32:
    address = ptr.toint(loc=loc, ip=ip).ir_value()
    value = llvm.inline_asm(T.i32(), [address],
                            'ld.acquire.sys.global.u32 $0, [$1];', '=r,l',
                            has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint32(value)


@dsl_user_op
def red_release_gpu_add_u32(ptr: cute.Pointer, value: Uint32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [ptr.toint(loc=loc, ip=ip).ir_value(), value.ir_value(loc=loc, ip=ip)],
                    'red.release.gpu.global.add.u32 [$0], $1;', 'l,r',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def red_gpu_add_u32(ptr: cute.Pointer, value: Uint32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [ptr.toint(loc=loc, ip=ip).ir_value(), value.ir_value(loc=loc, ip=ip)],
                    'red.gpu.global.add.u32 [$0], $1;', 'l,r',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def red_gpu_add_s32(ptr: cute.Pointer, value: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [ptr.toint(loc=loc, ip=ip).ir_value(), value.ir_value(loc=loc, ip=ip)],
                    'red.gpu.global.add.s32 [$0], $1;', 'l,r',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def red_release_gpu_xor_u64(ptr: cute.Pointer, value: Uint64, *, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [ptr.toint(loc=loc, ip=ip).ir_value(), value.ir_value(loc=loc, ip=ip)],
                    'red.release.gpu.global.xor.b64 [$0], $1;', 'l,l,~{memory}',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def red_release_sys_add_u64(ptr: cute.Pointer, value: Uint64, *, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [ptr.toint(loc=loc, ip=ip).ir_value(), value.ir_value(loc=loc, ip=ip)],
                    'red.release.sys.global.add.u64 [$0], $1;', 'l,l,~{memory}',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def red_release_sys_add_u32(ptr: cute.Pointer, value: Uint32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [ptr.toint(loc=loc, ip=ip).ir_value(), value.ir_value(loc=loc, ip=ip)],
                    'red.release.sys.global.add.u32 [$0], $1;', 'l,r',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def atom_add_u32(ptr: cute.Pointer, value: Uint32, *, loc=None, ip=None) -> Uint32:
    previous = llvm.inline_asm(
        T.i32(), [ptr.toint(loc=loc, ip=ip).ir_value(), value.ir_value(loc=loc, ip=ip)],
        'atom.global.add.u32 $0, [$1], $2;', '=r,l,r',
        has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint32(previous)


@dsl_user_op
def atom_add_u64(ptr: cute.Pointer, value: Uint64, *, loc=None, ip=None) -> Uint64:
    previous = llvm.inline_asm(
        T.i64(), [ptr.toint(loc=loc, ip=ip).ir_value(), value.ir_value(loc=loc, ip=ip)],
        'atom.global.add.u64 $0, [$1], $2;', '=l,l,l',
        has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint64(previous)


@dsl_user_op
def atom_sys_add_u64(ptr: cute.Pointer, value: Uint64, *, loc=None, ip=None) -> Uint64:
    previous = llvm.inline_asm(
        T.i64(), [ptr.toint(loc=loc, ip=ip).ir_value(), value.ir_value(loc=loc, ip=ip)],
        'atom.sys.global.add.u64 $0, [$1], $2;', '=l,l,l',
        has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint64(previous)


@dsl_user_op
def atom_release_gpu_add_u32(ptr: cute.Pointer, value: Uint32, *, loc=None, ip=None) -> Uint32:
    previous = llvm.inline_asm(
        T.i32(), [ptr.toint(loc=loc, ip=ip).ir_value(), value.ir_value(loc=loc, ip=ip)],
        'atom.release.gpu.global.add.u32 $0, [$1], $2;', '=r,l,r,~{memory}',
        has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint32(previous)


@dsl_user_op
def atom_shared_add_u32(ptr: cute.Pointer, value: Uint32, *, loc=None, ip=None) -> Uint32:
    previous = llvm.inline_asm(
        T.i32(), [ptr.toint(loc=loc, ip=ip).ir_value(), value.ir_value(loc=loc, ip=ip)],
        'atom.shared.add.u32 $0, [$1], $2;', '=r,r,r',
        has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint32(previous)


@dsl_user_op
def st_release_sys_u64(ptr: cute.Pointer, value: Uint64, *, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [ptr.toint(loc=loc, ip=ip).ir_value(), value.ir_value(loc=loc, ip=ip)],
                    'st.release.sys.global.u64 [$0], $1;', 'l,l,~{memory}',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def st_global_u64(ptr: cute.Pointer, value: Uint64, *, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [ptr.toint(loc=loc, ip=ip).ir_value(), value.ir_value(loc=loc, ip=ip)],
                    'st.global.u64 [$0], $1;', 'l,l',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def read_clock64(*, loc=None, ip=None) -> Uint64:
    """Read the per-SM cycle counter used by the reference wait deadlines."""
    return Uint64(nvvm.read_ptx_sreg_clock64(T.i64(), loc=loc, ip=ip))


@dsl_user_op
def device_trap(*, loc=None, ip=None) -> None:
    """A timed-out barrier must fail the kernel, never resume as success."""
    llvm.inline_asm(None, [], 'trap;', '', has_side_effects=True,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def sync_epilogue_threads(*, loc=None, ip=None) -> None:
    """Join the eight epilogue warps in one CTA before ring publication."""
    llvm.inline_asm(None, [], 'bar.sync 2, 256;', '',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def _sync_dispatch_128(*, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [], 'bar.sync 0, 128;', '',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def _sync_dispatch_256(*, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [], 'bar.sync 0, 256;', '',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def _sync_dispatch_384(*, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [], 'bar.sync 0, 384;', '',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@cute.jit
def sync_dispatch_threads(participants: cutlass.Constexpr = 0) -> None:
    """Dispatch barrier id 0; zero preserves the existing whole-CTA scope.

    The caller must have exactly ``participants`` threads enter each nonzero
    scope: role3 dispatch uses 128; role1 uses its full 256/384-thread block.
    """
    if cutlass.const_expr(participants == 128):
        _sync_dispatch_128()
    elif cutlass.const_expr(participants == 256):
        _sync_dispatch_256()
    elif cutlass.const_expr(participants == 384):
        _sync_dispatch_384()
    else:
        cute.arch.sync_threads()


@cute.jit
def wait_sys_u64_eq(ptr: cute.Pointer, target: Uint64,
                    rank_idx: Uint32, peer: Uint32) -> None:
    """Reference 60-second per-peer combine readiness wait."""
    start_clock = read_clock64()
    observed = ld_acquire_sys_u64(ptr)
    while observed != target:
        if Uint64(read_clock64() - start_clock) >= Uint64(120000000000):
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
                    Uint64(read_clock64() - start_clock) >= Uint64(120000000000):
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
