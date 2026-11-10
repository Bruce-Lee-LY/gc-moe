"""SM120 role preparation and pointer ABI; compiler utilities are shared with GC MoE."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
import re

import torch
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Uint8, Uint32, Uint64
from cutlass.cute.runtime import make_ptr

from ...impls.sm120_fp8_fp4_gc_moe import SM120GCMoERoleKernel, SM120SharedExpertGemm
from ...runtime.sm120_gc_moe import RoleBinary, get_context
from ...runtime.jit import (
    COMPILE_LOCK,
    CompilerIdentity,
    assemble_ptx,
    compile_kernel,
    compiler_identity,
)
from ..heuristics.gc_moe import get_block_m_for_gc_moe
from ..heuristics.sm120_gc_moe import get_sm120_role_config
from .runtime_utils import TmaDescriptorSpec, make_tma_2d_desc, make_tma_weights_3d_desc


@dataclass(frozen=True)
class _Prepared:
    binaries: tuple[RoleBinary, ...]


_CACHE: dict[tuple, _Prepared] = {}
_BINARY_CACHE: dict[tuple, tuple[RoleBinary, ...]] = {}
# Encoded TensorMaps plus every tensor whose address they embed.
_MAP_CACHE: dict[tuple, tuple] = {}
_SHARED_BINARIES: dict[tuple, tuple[RoleBinary, ...]] = {}
_SHARED_PREPARED: dict[tuple, tuple[RoleBinary, ...]] = {}
_SHARED_BINDINGS: dict[tuple, tuple] = {}


def _compiler_identity(target: str) -> CompilerIdentity:
    return compiler_identity(target, SM120GCMoERoleKernel.__call__,
                             Path(__file__).resolve().parents[2], 'gc_moe_sm120_role')


def _assemble_sm120_ptx(ptx, dump_dir, basename, identity):
    # CuTe emits a 9.4 header for these kernels. CUDA 13.2 accepts the SM120
    # instruction stream under PTX 9.2 and checks every instruction itself.
    if re.search(r'\brelease[ \t]+13\.2(?=[,\s]|$)', identity.tool_version):
        ptx = re.sub(r'(?m)^\.version[ \t]+9\.4[ \t]*$', '.version 9.2', ptx, count=1)
    return assemble_ptx(ptx, dump_dir, basename, target='sm_120a',
                        ptxas_options=identity.ptxas_options)


def _assemble_role_ptx(ptx, role, dump_dir, identity, reg_cap=0, threads=0):
    ptx = _clean_cute_ptx(ptx)
    formals = _entry_formals(ptx)
    if len(formals) != 15:
        raise RuntimeError('SM120 CuTe launch ABI must have twelve pointers and three scalars')
    for ordinal, formal in enumerate(formals):
        # Map formals 7-10 need 128-byte alignment for tensormap prefetch.
        align = 128 if 7 <= ordinal <= 10 else 16
        kind = (r'\.u64\s+\.ptr\s+\.global\s+\.align\s+' + str(align) if ordinal < 12
                else r'\.u32' if ordinal < 14 else r'\.f32')
        if re.search(kind + r'\s+\S+_param_' + str(ordinal) + r',?\s*$', formal) is None:
            raise RuntimeError(f'SM120 CuTe launch formal {ordinal} changed type or order')
    if role == 2:
        if 'mma.sync.aligned.kind::mxf8f6f4' not in ptx:
            raise RuntimeError('SM120 CuTe compute lost its block-scaled MMA')
        if 'cp.async.bulk.tensor' not in ptx:
            raise RuntimeError('SM120 CuTe compute lost its TensorMap loads')
    if role == 3 and threads == 384:
        # Match the two-resident-CTA finalize grid used by the scheduler.
        ptx, rewritten = re.subn(
            r'(?m)^\.minnctapersm\s+4\s*$',
            '.minnctapersm 2', ptx, count=1)
        if rewritten != 1:
            raise RuntimeError('SM120 CuTe combine PTX occupancy directive changed')
    if reg_cap:
        # The warp-specialized producer adds enough live values to tip the
        # 160-thread role2 CTA past the 65,536-register file once the
        # allocator's 16-wide per-thread rounding lands (197 -> 208): both
        # the block limit and the driver's L1/shared carveout then collapse
        # to a single CTA per SM for a ~1.5x slower sweep. Rewrite the
        # CuTe-emitted minimum-occupancy directive to the co-residency the
        # SMEM budget intends and pin a 16-aligned register ceiling.
        resident = 65536 // (threads * reg_cap) if threads else 2
        ptx, rewritten = re.subn(
            r'(?m)^\.minnctapersm\s+1\s*$',
            f'.minnctapersm {resident}\n.maxnreg {reg_cap}', ptx, count=1)
        if rewritten != 1:
            raise RuntimeError('SM120 CuTe compute PTX lost its occupancy directive')
    return _assemble_sm120_ptx(ptx, dump_dir, f'sm120_role{role}', identity)


def _clean_cute_ptx(ptx: str) -> str:
    if ptx.endswith("\x00"):
        ptx = ptx[:-1]
    if "\x00" in ptx:
        raise RuntimeError('unexpected NUL inside CuTe PTX')
    return ptx

def _entry_formals(ptx: str) -> list[str]:
    """Read only the launch ABI, excluding function and call-local params."""
    entries = list(re.finditer(
        r"(?m)^[ \t]*\.visible[ \t]+\.entry[ \t]+[A-Za-z0-9_.$]+\([ \t]*$",
        ptx))
    if len(entries) != 1:
        raise RuntimeError('CuTe kernel entry count changed')
    start = entries[0].end()
    closing = re.search(r"(?m)^\)[ \t]*$", ptx[start:])
    if closing is None:
        raise RuntimeError('CuTe kernel entry signature is unterminated')
    lines = [line for line in ptx[start:start + closing.start()].splitlines()
             if line.strip()]
    if not lines or any(re.fullmatch(r"[ \t]*\.param\b[^\n]*", line) is None
                        for line in lines):
        raise RuntimeError('CuTe kernel entry formals changed')
    return lines

_POINTER_TYPES = (Uint8, Uint64, BFloat16, Uint8, Uint32,
                  Uint8, Uint32, Uint8, Uint8, Uint8, Uint8, Uint32)


def _role_specs(config, layout, schedule, has_stats, primary_ctx):
    specs = []
    for role in (1, 2, 3):
        kernel = SM120GCMoERoleKernel(config, layout, role, schedule, has_stats,
                                    launch_dependents=primary_ctx and role == 1)
        specs.append((kernel, (role, schedule.smem[role - 1],
                               schedule.threads[role - 1],
                               schedule.grids[role - 1])))
    return tuple(specs)


def _compile_roles(specs, identity):
    pointers = []
    for ordinal, dtype in enumerate(_POINTER_TYPES):
        align = 128 if 7 <= ordinal <= 10 else 16
        pointers.append(make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=align))
    pointers = tuple(pointers)
    binaries = []
    for kernel, launch in specs:
        role = launch[0]
        cubin = compile_kernel(
            kernel, (*pointers, Uint32(0), Uint32(0), Float32(0)), identity,
            lambda ptx, dump_dir, reg_cap=kernel.reg_cap,
                   threads=kernel.threads:
                _assemble_role_ptx(ptx, role, dump_dir, identity,
                                   reg_cap, threads))
        binaries.append(RoleBinary(role, cubin, *launch[1:]))
    return tuple(binaries)


def _binary_key(identity, specs):
    # Pointer compile arguments have fixed types. Kernel attributes and launch
    # resources cover every constexpr/layout field, without CUDA addresses.
    return (identity, tuple((tuple(sorted(vars(kernel).items())), launch)
                            for kernel, launch in specs))


def _get_or_compile_binaries(key, identity, specs):
    with COMPILE_LOCK:
        binaries = _BINARY_CACHE.get(key)
        if binaries is None:
            binaries = _compile_roles(specs, identity)
            expected = (1, 2, 3)
            if tuple(item.role for item in binaries) != expected:
                raise RuntimeError('SM120 compilation did not produce all role binaries')
            if _compiler_identity('sm_120a') != identity:
                raise RuntimeError('SM120 compiler/source identity changed while compiling')
            _BINARY_CACHE[key] = binaries
        return binaries


def _make_compact_weight_desc(payload):
    experts, k_tiles, rows, segment = payload.shape
    # The shared encoder maps E4M3FN to CUDA UINT8. Request byte transport
    # for the packed FP4 payload, which the SM120 kernel unpacks itself.
    return TmaDescriptorSpec(
        payload.data_ptr(), torch.float8_e4m3fn, (segment, rows, k_tiles, experts),
        (payload.stride(2), payload.stride(1), payload.stride(0)),
        (segment, 64, 1, 1), 64)


def _get_map_storage(config, buffer, w1, w2, device, tile_n, compact_w=False):
    """Encode the four role2 TensorMaps once per live pointer set.

    The descriptors embed global base addresses, so the cache pins the
    workspace and weight tensors to keep those addresses stable.
    """
    layout = buffer.layout
    local, rows = layout.local, layout.rows
    key = (buffer.buffer.data_ptr(), w1[0].data_ptr(), w1[1].data_ptr(),
           w2[0].data_ptr(), w2[1].data_ptr(), config, tile_n, compact_w)
    cached = _MAP_CACHE.get(key)
    if cached is not None:
        return cached[0]
    l1_payload, l2_payload = w1[0], w2[0]
    if compact_w:
        w1_desc = _make_compact_weight_desc(l1_payload)
        w2_desc = _make_compact_weight_desc(l2_payload)
    else:
        w1_desc = make_tma_weights_3d_desc(l1_payload, 128, 64, swizzle=128)
        w2_desc = make_tma_weights_3d_desc(l2_payload, 128, 64, swizzle=128)
    # The X box carries only the CTA token tile's rows, so the TMA moves
    # N x 128 E4M3 bytes and expect_tx matches the kernel's stage budget.
    specs = (
        w1_desc,
        make_tma_2d_desc(buffer.l1_acts.view(local * rows, config.hidden),
                         config.hidden, local * rows, 128, tile_n,
                         config.hidden, 128),
        w2_desc,
        make_tma_2d_desc(buffer.l2_acts.view(local * rows, config.intermediate),
                         config.intermediate, local * rows, 128, tile_n,
                         config.intermediate, 128),
    )
    payloads = []
    for spec in specs:
        descriptor = spec.encode()
        payload = ctypes.string_at(descriptor.getPtr(), 128)
        if len(payload) != 128:
            raise RuntimeError('SM120 TensorMap is not 128 bytes')
        payloads.append(payload)
    map_storage = torch.empty(512, dtype=torch.uint8, device=device)
    map_storage.copy_(torch.frombuffer(bytearray(b''.join(payloads)), dtype=torch.uint8))
    _MAP_CACHE[key] = (map_storage, buffer, w1, w2)
    return map_storage


def prepare_role_binaries(config, y, w1, w2, buffer, stats=None, context=None):
    if torch.cuda.get_device_capability(y.device) != (12, 0):
        raise ValueError('SM120 CuTe GC requires compute capability 12.0')
    if context is None:
        context = get_context()
    schedule = get_sm120_role_config(config, context, buffer.layout)
    key = (config, buffer.layout, schedule, stats is not None, context.primary_ctx_roles)
    prepared = _CACHE.get(key)
    if prepared is None:
        with COMPILE_LOCK:
            prepared = _CACHE.get(key)
            if prepared is None:
                identity = _compiler_identity('sm_120a')
                specs = _role_specs(config, buffer.layout, schedule, stats is not None,
                                    context.primary_ctx_roles)
                binaries = _get_or_compile_binaries(_binary_key(identity, specs), identity, specs)
                prepared = _Prepared(binaries)
                _CACHE[key] = prepared
    tile_n = schedule.tile_n
    map_storage = _get_map_storage(config, buffer, w1, w2, y.device, tile_n,
                                    schedule.compact_w)
    map_base = map_storage.data_ptr()
    addresses = [buffer.buffer.data_ptr(), buffer.peer_table.data_ptr(), y.data_ptr(),
                 w1[0].data_ptr(), w1[1].data_ptr(),
                 w2[0].data_ptr(), w2[1].data_ptr(),
                 map_base, map_base + 128, map_base + 256, map_base + 384,
                 0 if stats is None else stats.data_ptr()]
    arguments = [*(ctypes.c_void_p(p) for p in addresses),
                 ctypes.c_uint32(buffer.rank_idx), ctypes.c_uint32(config.tokens),
                 ctypes.c_float(config.clamp)]
    return prepared.binaries, arguments


def launch_sm120_gc_moe(config, y, w1, w2, buffer, stats,
                       shared_w1=None, shared_w2=None):
    context = get_context()
    binaries, arguments = prepare_role_binaries(
        config, y, w1, w2, buffer, stats, context)
    parallel_shared = False
    if config.shared_experts:
        shared_binaries, shared_arguments = _prepare_shared_binaries(
            config, buffer, shared_w1, shared_w2)
        schedule = get_sm120_role_config(config, context, buffer.layout)
        parallel_shared = schedule.parallel_shared
        binaries = (binaries[0], *shared_binaries, binaries[1], binaries[2])
        arguments = (arguments, *shared_arguments, arguments, arguments)
    context.launch(binaries, arguments, parallel_shared=parallel_shared)


def release_workspace(buffer):
    from ...runtime.sm120_gc_moe import release_workspace_graphs

    address = buffer.buffer.data_ptr()
    release_workspace_graphs(address)
    for key in tuple(_MAP_CACHE):
        if key[0] == address:
            _MAP_CACHE.pop(key, None)
    for key, binding in tuple(_SHARED_BINDINGS.items()):
        if binding[0] is buffer:
            _SHARED_BINDINGS.pop(key, None)


def _prepare_shared_binaries(config, buffer, w1, w2):
    tile_n = (8 if config.tokens <= 8 else 16 if config.tokens <= 16
              else 32 if config.tokens <= 32 else 64)
    shared_i = config.intermediate * config.shared_experts
    block_m = get_block_m_for_gc_moe(
        config.ranks, config.experts, config.capacity, config.tokens, config.topk)
    input_sf_rows = buffer.shared_l1_acts_sf.shape[0]
    packed = tuple(pair[1].dtype == torch.int32 for pair in (w1, w2))
    k128 = tuple(is_packed and pair[1].shape[1] != pair[0].shape[0]
                 for is_packed, pair in zip(packed, (w1, w2)))
    shape_key = (config.hidden, shared_i, config.tokens,
                 config.fast_math, tile_n, packed, k128, block_m, input_sf_rows)
    binaries = _SHARED_PREPARED.get(shape_key)
    if binaries is None:
        binaries = _compile_shared_binaries(shape_key)
    tensors = ((*w1, buffer.x, buffer.shared_l1_acts_sf, buffer.shared_l2_acts, buffer.shared_l2_acts_sf),
               (*w2, buffer.shared_l2_acts, buffer.shared_l2_acts_sf,
                buffer.shared_y, buffer.shared_l2_acts_sf))
    # Captured child nodes embed these addresses; retain their allocations.
    pointer_key = tuple(t.data_ptr() for role in tensors for t in role)
    binding = _SHARED_BINDINGS.get(pointer_key)
    if binding is None:
        maps = tuple(_make_weight_map(role[0]) if is_packed else None
                     for role, is_packed in zip(tensors, packed))
        binding = (buffer, w1, w2, maps)
        _SHARED_BINDINGS[pointer_key] = binding
    arguments = tuple([*(ctypes.c_void_p(t.data_ptr()) for t in role),
                        ctypes.c_void_p(0 if desc is None else desc.data_ptr()),
                        ctypes.c_float(config.clamp)] for role, desc in zip(tensors, binding[3]))
    return binaries, arguments


def _make_weight_map(payload):
    blocks, rows, _ = payload.shape
    desc = make_tma_2d_desc(payload, 128, blocks * rows, 128, 64, 128, 128).encode()
    storage = torch.empty(128, dtype=torch.uint8, device=payload.device)
    storage.copy_(torch.frombuffer(bytearray(ctypes.string_at(desc.getPtr(), 128)), dtype=torch.uint8))
    return storage


def _compile_shared_binaries(shape_key):
    hidden, intermediate, tokens, fast_math, tile_n, packed, k128, block_m, input_sf_rows = shape_key
    identity = compiler_identity('sm_120a', SM120SharedExpertGemm.__call__,
                                  Path(__file__).resolve().parents[2],
                                  'gc_moe_sm120_shared')
    key = (identity, *shape_key)
    with COMPILE_LOCK:
        binaries = _SHARED_BINARIES.get(key)
        if binaries is None:
            pointers = tuple(make_ptr(Uint8, 0, cute.AddressSpace.gmem, assumed_align=16)
                             for _ in range(7))
            compiled = []
            for index, role in enumerate((4, 5)):
                first = index == 0
                kernel = SM120SharedExpertGemm(hidden, intermediate,
                    tokens, first=first, tile_n=tile_n, fast_math=fast_math,
                    ue8m0_weights=packed[index], k128_weights=k128[index],
                    input_block_m=block_m, input_sf_rows=input_sf_rows)
                cubin = compile_kernel(kernel, (*pointers, Float32(0)), identity,
                    lambda ptx, out: _assemble_sm120_ptx(
                        _clean_cute_ptx(ptx), out, 'sm120_shared', identity))
                compiled.append(RoleBinary(role, cubin, kernel.smem, 128, kernel.grid))
            binaries = tuple(compiled)
            _SHARED_BINARIES[key] = binaries
        _SHARED_PREPARED[shape_key] = binaries
    return binaries
