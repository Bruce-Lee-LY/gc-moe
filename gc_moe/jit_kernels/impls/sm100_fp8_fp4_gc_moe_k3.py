"""Prepare CuTeDSL cubins and TensorMaps for the K3 child graph.

The supported H/I and token ranges are checked before graph launch. Each
role's PTX/cubin export is owned by its compile, including concurrent ranks.
"""

from __future__ import annotations

from collections import OrderedDict
import ctypes
from dataclasses import dataclass
from math import inf
import os
from pathlib import Path
import re

from cutlass import Float32, Uint32
from cutlass.cute.runtime import from_dlpack
import torch
from cuda.bindings import driver as cu

from ...comm.gc_k3_barrier import DEFAULT_NVLINK_BARRIER_TIMEOUT_SECS
from ...impls.sm100_fp8_fp4_gc_moe_k3 import GCK3ProjectionReduceKernel, GCK3FrontKernel, GCK3MoERoleKernel, GCK3RouterKernel
from ...layout.gc_moe_k3 import (
    COMBINE_WARPS,
    COMPUTE_ROLE,
    DISPATCH_ROLE,
    FINALIZE_ROLE,
    WORKER_ROLES,
    FRONT_ROLE,
    ROUTER_ROLE,
    ASSIST_WORKER_ROLES,
    ROUTED_ASSIST_ROLE,
    VALID_ROLE_ORDERS,
    PROJECTION_REDUCE_ROLE,
    CANDIDATE_BLOCK_M,
    MAX_EXPERTS,
    MAX_EXPERTS_PER_RANK,
    MAX_RANKS,
    PROJECTION_CHANNELS_PER_CTA,
    PROJECTION_K_TILE,
    PROJECTION_ROW_TILES,
    align,
    ceil_div,
)
from ...runtime.gc_moe_k3 import RoleBinary, get_context
from ...runtime.jit import (
    COMPILE_LOCK,
    PTXAS_OPTIONS,
    CompilerIdentity,
    assemble_ptx,
    compile_kernel,
    compiler_identity,
    device_compile_target,
    ptx_target,
)
from ..heuristics.gc_moe_k3 import (
    SMEM_CAPACITY,
    get_combine_geometry,
    get_role_config,
    get_shared_block_m,
    num_max_live_pool_blocks,
)
from .runtime_utils import (
    TmaDescriptorSpec,
    make_tma_2d_desc,
    make_tma_sf_desc,
)


ROUTED_TMA_MAP_COUNT = 9
TMA_MAP_COUNT = 18
# CuTe emits the first TensorMap, workspace pointer and clamp before maps 1..17.
_COMPUTE_MAP_ORDINALS = (4, *range(7, TMA_MAP_COUNT + 6))


@dataclass(frozen=True)
class GCTmaPlan:
    """Fixed eighteen-map ABI for routed FP8/FP4 and independent BF16 shared."""

    l1_acts: TmaDescriptorSpec
    l1_acts_sf: TmaDescriptorSpec
    l1_weights: TmaDescriptorSpec
    l1_weights_sf: TmaDescriptorSpec
    l1_output: TmaDescriptorSpec
    l2_acts: TmaDescriptorSpec
    l2_acts_sf: TmaDescriptorSpec
    l2_weights: TmaDescriptorSpec
    l2_weights_sf: TmaDescriptorSpec
    shared_l1_acts: TmaDescriptorSpec
    shared_l1_acts_sf: TmaDescriptorSpec
    shared_l1_weights: TmaDescriptorSpec
    shared_l1_weights_sf: TmaDescriptorSpec
    shared_l1_output: TmaDescriptorSpec
    shared_l2_acts: TmaDescriptorSpec
    shared_l2_acts_sf: TmaDescriptorSpec
    shared_l2_weights: TmaDescriptorSpec
    shared_l2_weights_sf: TmaDescriptorSpec

    def ordered(self) -> tuple[TmaDescriptorSpec, ...]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__)


def _k3_weight_map(tensor, block_k, load_n, swizzle, *, shared=False):
    experts = 1 if shared else tensor.shape[0]

    # Collapse the adjacent N-slice and K-tile axes into one map dimension.
    # Both CTAs still load their own 128-row box from the same 256-row slice.
    slices, kb, rows, segment = tensor.shape[-4:]
    element_bytes = tensor.element_size()
    row_bytes = segment * element_bytes
    tile_bytes = rows * row_bytes
    return TmaDescriptorSpec(
        tensor.data_ptr(), tensor.dtype, (block_k, rows, slices * kb, experts),
        (row_bytes, tile_bytes, slices * kb * tile_bytes),
        (swizzle // element_bytes if swizzle else block_k, load_n, 1, 1), swizzle)


def build_gc_tma_plan(config, sym_buffer, l1_weights, l2_weights,
                      shared_l1_weights, shared_l2_weights, *,
                      shared_x, shared_block_m: int) -> GCTmaPlan:
    """Describe routed/shared tensors; encode with live pointers at launch.

    Each weight tuple's [0] tensor is the tile-repacked payload prepared by
    the caller; scale factors and activations use their original storage.
    """
    bk, bn = config.block_k, config.block_n
    h, i = sym_buffer.hidden, sym_buffer.intermediate_hidden
    sf_m = config.sf_block_m
    ring, sf_ring = sym_buffer.layout.ring_tokens, sym_buffer.layout.sf_ring_tokens
    shared_i = sym_buffer.shared_intermediate_hidden
    shared_h = sym_buffer.shared_hidden

    def acts(t, inner, outer, block_inner, block_outer, swizzle):
        return make_tma_2d_desc(t, inner, outer, block_inner, block_outer,
                         t.stride(-2), swizzle)

    def scales(t, mn, k, block_mn, gran, groups):
        return make_tma_sf_desc(t, mn, k, block_mn, gran, groups,
                     (bk + gran * 4 - 1) // (gran * 4), t.stride(-1))

    routed = (
        acts(sym_buffer.l1_acts, h, ring, bk, config.load_m, config.swizzle_acts),
        scales(sym_buffer.l1_acts_sf, sf_ring, h, sf_m, 32, 1),
        _k3_weight_map(l1_weights[0], bk, bn, config.swizzle_weights),
        scales(l1_weights[1], 2 * i, h, bn, 32,
               sym_buffer.num_experts // sym_buffer.group.size()),
        acts(sym_buffer.l2_acts, i, ring, bn // 2,
             config.store_m_l1, config.swizzle_output),
        acts(sym_buffer.l2_acts, i, ring, bk, config.load_m, config.swizzle_acts),
        scales(sym_buffer.l2_acts_sf, sf_ring, i, sf_m, 32, 1),
        _k3_weight_map(l2_weights[0], bk, bn, config.swizzle_weights),
        scales(l2_weights[1], h, i, bn, 32,
               sym_buffer.num_experts // sym_buffer.group.size()),
    )
    shared_load_m = shared_block_m // 2
    shared = (
        acts(shared_x, shared_h, shared_x.shape[0],64,shared_load_m,128),
        routed[1],
        _k3_weight_map(shared_l1_weights, 64, bn,
                                   swizzle=config.swizzle_weights, shared=True),
        routed[3],
        acts(sym_buffer.shared_l2_acts,shared_i,sym_buffer.num_max_tokens_per_rank,
             64,config.store_m_l1,128),
        acts(sym_buffer.shared_l2_acts,shared_i,sym_buffer.num_max_tokens_per_rank,
             64,shared_load_m,128),
        routed[6],
        _k3_weight_map(shared_l2_weights, 64, bn,
                                   swizzle=config.swizzle_weights, shared=True),
        routed[8],
    )
    return GCTmaPlan(*routed, *shared)


def _get_gc_moe_nvlink_barrier_timeout_secs() -> int:
    """Read a positive int32 timeout. Empty selects the default; leading whitespace is accepted."""
    name = 'GC_MOE_NVLINK_BARRIER_TIMEOUT_SECS'
    value = os.getenv(name)
    if value is None or value == '':
        return DEFAULT_NVLINK_BARRIER_TIMEOUT_SECS
    if re.fullmatch(r"[ \t\n\v\f\r]*[+-]?[0-9]+", value, re.ASCII) is None:
        raise ValueError(f"Invalid {name}: {value}")
    try:
        seconds = int(value, 10)
    except ValueError as exc:
        raise ValueError(f"Invalid {name}: {value}") from exc
    if not 0 < seconds <= 2_147_483_647:
        raise ValueError(f"Invalid {name}: {value}")
    return seconds


def _routed_weight_linger(config, num_topk: int) -> bool:
    """Linger when the uniform-routing estimate gives an expert two or more m-blocks.

    The gate compares average tokens per expert against the tile, so skewed
    routing can leave some experts below one m-block and others above it.
    """
    tokens_per_expert = ceil_div(config.tokens * num_topk, config.local_experts)
    return tokens_per_expert >= 2 * config.block_m


def _compiler_identity(target: str) -> CompilerIdentity:
    return compiler_identity(target, GCK3MoERoleKernel.__call__,
                             Path(__file__).resolve().parents[2], 'gc_moe_role')


@dataclass
class _Prepared:
    maps: bytes
    projection_maps: bytes
    binaries: tuple[RoleBinary, ...]
    front_maps: bytes


# Address-bound descriptors contain bytes only. Allocation ownership belongs
# to the CUDA user object on each child graph, not this bounded host cache.
_PREPARED_CACHE_CAPACITY = 16
_CACHE: OrderedDict[tuple, _Prepared] = OrderedDict()
_BINARY_CACHE: dict[tuple, tuple[RoleBinary, ...]] = {}


def _get_or_compile_binaries(key: tuple, identity: CompilerIdentity,
                             target: str, compile_roles) -> tuple[RoleBinary, ...]:
    """Publish a complete cubin set once, after compiler identity is stable."""
    with COMPILE_LOCK:
        binaries = _BINARY_CACHE.get(key)
        if binaries is None:
            binaries = tuple(compile_roles())
            if tuple(item.role for item in binaries) not in VALID_ROLE_ORDERS:
                raise RuntimeError('GC role compilation did not produce the required nodes')
            if _compiler_identity(target) != identity:
                raise RuntimeError('compiler identity changed while compiling role binaries')
            _BINARY_CACHE[key] = binaries
        return binaries


class _SymBufferValue(ctypes.Structure):
    """592-byte by-value symmetric peer-buffer launch argument."""
    _fields_ = (('base', ctypes.c_int64),
                ('offsets', ctypes.c_int64 * 72),
                ('rank_idx', ctypes.c_uint32))


def _sym_buffer_value(sym_buffer) -> _SymBufferValue:
    addresses = tuple(int(ptr) for ptr in sym_buffer.peer_addresses)
    rank = int(sym_buffer.rank_idx)
    if (not 1 <= len(addresses) <= 72 or not 0 <= rank < len(addresses) or
            any(ptr <= 0 or ptr >= 1 << 64 or ptr % 16 for ptr in addresses)):
        raise ValueError('invalid symmetric peer address set')
    base = int(sym_buffer.buffer.data_ptr())
    if addresses[rank] != base:
        raise ValueError('local symmetric buffer differs from rendezvous base')
    value = _SymBufferValue()
    value.base = base if base < 1 << 63 else base - (1 << 64)
    for i, address in enumerate(addresses):
        delta = address - base
        if not -(1 << 63) <= delta < (1 << 63):
            raise ValueError('symmetric peer offset exceeds signed int64')
        value.offsets[i] = delta
    value.rank_idx = rank
    if (ctypes.sizeof(value) != 592 or
            (_SymBufferValue.base.offset, _SymBufferValue.offsets.offset,
             _SymBufferValue.rank_idx.offset) != (0, 8, 584)):
        raise AssertionError('SymBuffer by-value ABI changed')
    return value


def _clean_cute_ptx(ptx: str) -> str:
    if ptx.endswith("\x00"):
        ptx = ptx[:-1]
    if "\x00" in ptx:
        raise RuntimeError('unexpected NUL inside CuTe PTX')
    return ptx


def _rewrite_value_formal(ptx: str, ordinal: int, value_bytes: int,
                          alignment: int, *, raw_param_address: bool = False) -> str:
    """Replace one audited 8B CuTe pointer formal with one value formal."""
    declaration = re.compile(
        r"(?m)^([ \t]*\.param[ \t]+\.align[ \t]+8[ \t]+\.b8[ \t]+)"
        rf"([A-Za-z0-9_.$]+_param_{ordinal})\[8\](,?)[ \t]*$")
    matches = list(declaration.finditer(ptx))
    if len(matches) != 1:
        raise RuntimeError(f"CuTe value formal {ordinal} changed")
    name = matches[0].group(2)
    uses = re.findall(r"(?<![A-Za-z0-9_.$])" + re.escape(name) +
                      r"(?![A-Za-z0-9_.$])", ptx)
    if len(uses) != 2:
        raise RuntimeError(f"CuTe value formal {ordinal} has unexpected uses")
    load = re.compile(
        r"(?m)^([ \t]*)ld\.param\.(?:u64|b64)[ \t]+(%rd[0-9]+),[ \t]*\[" +
        re.escape(name) + r"\];[ \t]*$")
    if len(list(load.finditer(ptx))) != 1:
        raise RuntimeError(f"CuTe value formal {ordinal} pointer load changed")
    ptx = declaration.sub(
        lambda m: re.sub(r"\.align[ \t]+8\b", f".align {alignment}",
                         m.group(1)) + f"{name}[{value_bytes}]{m.group(3)}",
        ptx, count=1)
    # The reference keeps by-value TensorMap addresses in param space across
    # task-phase selection. The peer value is accessed through a generic
    # address, so only the compute maps use the raw form here.
    address_op = 'mov.b64' if raw_param_address else 'cvta.param::entry.u64'
    ptx = load.sub(
        lambda m: f"{m.group(1)}{address_op} {m.group(2)}, {name};",
        ptx, count=1)
    return ptx


def _convert_compute_map_uses(ptx: str, has_shared: bool, routed_assist: bool) -> str:
    """Convert early prefetches and W1 output stores at PTX instruction uses.

    CuTe exposes map parameters as pointers. After the by-value formal rewrite
    those registers contain *param-space* addresses, as in the reference.
    Loader warps convert their two selected maps once per task before the K
    loop. This bridge converts only the earlier prefetches and the per-store
    W1 output map, which the reference also selects inside its elected store.
    Every conversion is scalar; no elected lane executes a warp collective.

    Fail closed on changes to CuTe's generated descriptor instruction forms.
    This bridge deliberately does not depend on PTX register IDs or labels.
    """
    raw_map_regs = []
    for ordinal in _COMPUTE_MAP_ORDINALS:
        raw = re.findall(
            r"(?m)^[ \t]*mov\.b64[ \t]+(%rd[0-9]+),[ \t]*"
            r"[A-Za-z0-9_.$]+_param_" + str(ordinal) + r";[ \t]*$", ptx)
        if len(raw) != 1:
            raise RuntimeError(f"compute raw TensorMap {ordinal} changed")
        raw_map_regs.append(raw[0])
    if len(set(raw_map_regs)) != TMA_MAP_COUNT:
        raise RuntimeError('compute TensorMap address registers alias')

    register_decl = re.compile(
        r"(?m)^([ \t]*\.reg[ \t]+\.b64[ \t]+%rd<)([0-9]+)(>;[ \t]*)$")
    declarations = list(register_decl.finditer(ptx))
    if len(declarations) != 1:
        raise RuntimeError('compute 64-bit PTX register declaration changed')
    next_reg = int(declarations[0].group(2))
    prefetch = re.compile(
        r"(?m)^([ \t]*prefetch\.tensormap[ \t]+\[)(%rd[0-9]+)"
        r"(\];[ \t]*)$")
    g2s = re.compile(
        r"(?m)^([ \t]*cp\.async\.bulk\.tensor\.(?:2d|4d)\.cta_group::2"
        r"[^\n]*?\[[^\n]*?\],[ \t]*\[)(%rd[0-9]+)"
        r"(,[ \t]*\{[^\n]+)$")
    s2g = re.compile(
        r"(?m)^([ \t]*cp\.async\.bulk\.tensor\.2d\.global\.shared"
        r"[^\n]*?\[)(%rd[0-9]+)(,[ \t]*\{[^\n]+)$")
    prefetch_regs = [m.group(2) for m in prefetch.finditer(ptx)]
    expected_prefetch = raw_map_regs[:ROUTED_TMA_MAP_COUNT] if routed_assist else raw_map_regs
    if sorted(prefetch_regs) != sorted(expected_prefetch):
        raise RuntimeError('compute TensorMap prefetch set changed')
    tensor_op_count = len(re.findall(
        r"(?m)^[ \t]*cp\.async\.bulk\.tensor\.", ptx))
    g2s_count = len(list(g2s.finditer(ptx)))
    s2g_count = len(list(s2g.finditer(ptx)))
    if g2s_count != (23 if has_shared else 16) or s2g_count < 1 or tensor_op_count != \
            g2s_count + s2g_count:
        raise RuntimeError(f'compute TensorMap load/store PTX forms changed: g2s={g2s_count}, s2g={s2g_count}, all={tensor_op_count}, shared={has_shared}')
    selected_conversions = re.findall(
        r"(?m)^[ \t]*cvta\.param\.u64[ \t]+(%rd[0-9]+),[ \t]*"
        r"(%rd[0-9]+);[ \t]*$", ptx)
    loader_regs = {m.group(2) for m in g2s.finditer(ptx)}
    if (len(selected_conversions) != 4 or
            len(loader_regs) != 4 or
            loader_regs != {dst for dst, _ in selected_conversions} or
            len({src for _, src in selected_conversions}) != 4 or
            any(src in raw_map_regs for _, src in selected_conversions)):
        raise RuntimeError('compute loader selected-map conversions changed')
    if has_shared:
        # The current frontend keeps one activation pair and emits a four-way
        # weight loop plus its three-position remainder. Check each path,
        # including tile-repacked BF16, routed packed FP4 and routed weight
        # scales. Both shared and routed weight ops load 4D tiles now.
        counts = {reg: sum(m.group(2) == reg for m in g2s.finditer(ptx))
                  for reg in loader_regs}
        weight_4d = [m.group(2) for m in g2s.finditer(ptx)
                     if '.tensor.4d.' in m.group(0)]
        if (sorted(counts.values()) != [1, 1, 7, 14] or
                len(weight_4d) != 14 or len(set(weight_4d)) != 1 or
                counts[weight_4d[0]] != 14):
            raise RuntimeError(f'compute BF16/routed loader paths changed: expected counts=[1,1,7,14], weight_4d_count=14, weight_4d_unique=1, weight_uses=14; observed counts={counts}, weight_4d={weight_4d}')
    source_cvta_count = len(selected_conversions)

    def converted_use(match: re.Match[str]) -> str:
        nonlocal next_reg
        prefix, raw_addr, suffix = match.groups()
        generic = f"%rd{next_reg}"
        next_reg += 1
        indent = re.match(r"[ \t]*", prefix).group(0)
        return (f"{indent}cvta.param.u64 {generic}, {raw_addr};\n"
                f"{prefix}{generic}{suffix}")

    ptx = prefetch.sub(converted_use, ptx)
    ptx = s2g.sub(converted_use, ptx)
    ptx = register_decl.sub(
        lambda m: m.group(1) + str(next_reg) + m.group(3), ptx, count=1)
    if len(re.findall(r"\bcvta\.param\.u64\b", ptx)) != \
            source_cvta_count + len(expected_prefetch) + s2g_count:
        raise RuntimeError('compute TensorMap conversion count changed')
    return ptx


def _device_compile_target(device: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) not in ((10, 0), (10, 3)):
        raise NotImplementedError('SM100 GC MoE K3 requires compute capability 10.0 or 10.3')
    return device_compile_target((major, minor))


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


def _order_compute_value_formals(ptx: str) -> str:
    """Keep ordinary scalars ahead of the configuration-selected value maps."""
    formals = _entry_formals(ptx)
    if len(formals) != TMA_MAP_COUNT + 6:
        raise RuntimeError(f'compute value-formal count changed: expected={TMA_MAP_COUNT + 6}, observed={len(formals)}')
    for index, line in enumerate(formals):
        if re.search(r"\b[A-Za-z0-9_.$]+_param_" + str(index) +
                     r"(?:\[[0-9]+\])?,?[ \t]*$", line) is None:
            raise RuntimeError(f'compute value-formal order changed: expected param_{index}, observed={line!r}')
    # Put the clamp and the shared workspace pointer before the value maps.
    order = (0, 1, 2, 3, 6, 5, *_COMPUTE_MAP_ORDINALS)
    ordered = [formals[i].rstrip().rstrip(',') for i in order]
    original = '\n'.join(formals)
    if ptx.count(original) != 1:
        raise RuntimeError('compute value-formal signature changed')
    return ptx.replace(original, ',\n'.join(ordered), 1)


def _assemble_role_value_params(ptx: str, role: int,
                                dump_dir: str, *, target: str,
                                has_shared: bool = False, routed_assist: bool = False,
                                with_projection: bool = False,
                                ptxas_options: tuple[str, ...] = PTXAS_OPTIONS) -> tuple[str, bytes, str]:
    """Build each role with relative peer value, plus compute's value maps."""
    ptx = _clean_cute_ptx(ptx)
    if ptx_target(ptx) != target:
        raise RuntimeError('CuTe generated a different device target')
    common = ('ptr', 'ptr', 'u32', 'u32')
    expected = {
        DISPATCH_ROLE: common,
        COMPUTE_ROLE: common + ('ptr', 'ptr', 'f32') + ('ptr',) * (TMA_MAP_COUNT - 1),
        FINALIZE_ROLE: common + ('ptr',),
    }[role]
    if with_projection:
        expected += ('ptr', 'f32', 'f32') + ('ptr',) * 5
    formals = _entry_formals(ptx)
    if len(formals) != len(expected):
        raise RuntimeError(f"role{role} CuTe parameter count changed: expected={len(expected)}, observed={len(formals)}")
    for index, (line, kind) in enumerate(zip(formals, expected)):
        form = (r"\.align[ \t]+8[ \t]+\.b8[ \t]+\S+_param_" +
                str(index) + r"\[8\]" if kind == 'ptr' else
                r"\." + kind + r"[ \t]+\S+_param_" + str(index))
        if re.search(form + r",?[ \t]*$", line) is None:
            raise RuntimeError(f"role{role} CuTe formal {index} changed: expected={kind}, observed={line!r}")
    if 'cvta.to.param::entry.u64' not in ptx or 'ld.param.u64' not in ptx:
        raise RuntimeError(f"role{role} peer parameter load path is missing")
    ptx = _rewrite_value_formal(ptx, 1, 592, 8)
    if role == COMPUTE_ROLE:
        for map_ordinal in _COMPUTE_MAP_ORDINALS:
            ptx = _rewrite_value_formal(
                ptx, map_ordinal, 128, 128, raw_param_address=True)
        ptx = _convert_compute_map_uses(ptx, has_shared, routed_assist)
        ptx = _order_compute_value_formals(ptx)
    elif with_projection:
        for ordinal in range(len(expected) - 2, len(expected)):
            ptx = _rewrite_value_formal(ptx, ordinal, 128, 128)
    return assemble_ptx(ptx, dump_dir, f"role{role}_value_params",
                                target=target, ptxas_options=ptxas_options)


def _capture_key(config, sym_buffer, green_context, y, l1_weights, l2_weights,
                 shared_l1_weights, shared_l2_weights,
                 nvlink_timeout_secs: int, routed_weight_linger: bool,
                 role_resources: tuple,
                 combine_resources: tuple, compiler_identity: CompilerIdentity,
                 shared_x=None, shared_y=None, routed_norm=None, projection_tensors=(), front=None) -> tuple:
    status, cuda_context = cu.cuCtxGetCurrent()
    if status != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cannot query CUDA context: {status}")
    layout = sym_buffer.layout
    weights = (*l1_weights, *l2_weights, *(x for x in
        (shared_l1_weights, shared_l2_weights, shared_x, shared_y, routed_norm,
         *projection_tensors, *(front[:3] if front else ())) if x is not None))
    return (int(cuda_context), compiler_identity, green_context.primary_ctx_roles,
            green_context.dispatch_sms,
            green_context.compute_sms,
            sym_buffer.buffer.data_ptr(),
            tuple(sym_buffer.peer_addresses), sym_buffer.rank_idx,
            sym_buffer.group.size(), sym_buffer.num_experts,
            sym_buffer.num_max_tokens_per_rank,
            config.tokens, config.block_m, config.hidden, config.intermediate,
            config.local_experts, config.shared_experts,
            config.shared_hidden, config.shared_intermediate,
            config.beta, config.linear_beta, config.clamp != inf,
            config.fast_math, sym_buffer.num_topk,
            nvlink_timeout_secs,
            routed_weight_linger,
            role_resources, combine_resources, None if front is None else front[3],
            layout.nbytes, layout.ring_tokens, layout.sf_ring_tokens,
            y.data_ptr(), _tensor_metadata(y),
            *(item for tensor in weights
              for item in (tensor.data_ptr(), tuple(tensor.shape),
                           tuple(tensor.stride()), str(tensor.dtype))))


def _tensor_metadata(tensor: torch.Tensor) -> tuple:
    return (tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype))


def _binary_key(compiler_identity: CompilerIdentity, role_kernels: tuple,
                role_resources: tuple, combine_resources: tuple,
                sym_buffer, y: torch.Tensor, weights: tuple) -> tuple:
    """Identify code generation without retaining any live CUDA address.

    Every role kernel attribute is included so new constexpr or derived layout
    fields cannot silently reuse an older cubin. Tensor metadata covers the
    CuTe compile argument types; actual addresses remain in _capture_key.
    """
    key = (compiler_identity,
           tuple(tuple(sorted(vars(kernel).items())) for kernel in role_kernels),
           role_resources, combine_resources,
           sym_buffer.layout.nbytes,
           _tensor_metadata(sym_buffer.buffer), _tensor_metadata(y),
           tuple(_tensor_metadata(tensor) for tensor in weights))
    return key


def _check_supported(config, sym_buffer) -> None:
    if (not 1 <= sym_buffer.group.size() <= MAX_RANKS or
            not 1 <= config.local_experts <= MAX_EXPERTS_PER_RANK or
            sym_buffer.group.size() * config.local_experts > MAX_EXPERTS or
            not 0 < sym_buffer.num_topk or
            sym_buffer.num_topk > min(32,sym_buffer.num_experts) or
            config.shared_experts < 1):
        raise NotImplementedError('current CuTe device path needs 1..512 experts/rank, <=2048 total experts, 1<=topk<=min(32,num_experts); K3 requires shared experts')
    if (config.hidden <= 0 or config.intermediate <= 0 or
            config.hidden % 256 != 0 or config.intermediate % 128 != 0 or
            (2 * config.intermediate) // 128 > 64 or
            config.block_m not in CANDIDATE_BLOCK_M or
            config.tokens < 0):
        raise NotImplementedError('current CuTe device math slice needs 2CTA H/I geometry, <=64 W1 N blocks, supported BM and nonnegative T')


def _check_ring_capacity(config, sym_buffer, compute_sms: int) -> None:
    """Recheck the full-device ring against the actual Green compute partition."""
    layout = sym_buffer.layout
    block_m = config.block_m
    if layout.ring_tokens % block_m:
        raise ValueError('GC ring capacity must contain whole M blocks')
    max_routed = (sym_buffer.num_max_tokens_per_rank * sym_buffer.group.size() *
                  min(sym_buffer.num_topk, config.local_experts))
    max_pool_blocks = ceil_div(max_routed, block_m) + config.local_experts
    live_blocks = num_max_live_pool_blocks(
        max_pool_blocks, compute_sms, config.hidden, config.intermediate)
    ring_blocks = layout.ring_tokens // block_m
    if live_blocks > ring_blocks:
        raise ValueError('GC ring capacity is insufficient for Green compute SMs')
    if layout.sf_ring_tokens < ring_blocks * align(block_m, 128):
        raise ValueError('GC SF ring capacity is insufficient for block M')


def _encode_tensor_maps(specs: tuple[TmaDescriptorSpec, ...]) -> bytes:
    payloads = []
    for spec in specs:
        # getPtr() borrows memory from the descriptor. Keep its owner alive
        # until string_at has copied the by-value kernel argument.
        descriptor = spec.encode()
        payloads.append(ctypes.string_at(descriptor.getPtr(), 128))
    return b"".join(payloads)


def prepare_role_binaries(config, y: torch.Tensor,
                          l1_weights: tuple[torch.Tensor, torch.Tensor],
                          l2_weights: tuple[torch.Tensor, torch.Tensor],
                          sym_buffer, shared_l1_weights, shared_l2_weights,
                          context=None, *, shared_x, shared_y,
                          routed_norm=None, norm_eps=1e-6, routed_scale=1.0, projection=None, front=None):
    """Encode live maps once per pointer set and compile each role once/shape."""
    _check_supported(config, sym_buffer)
    if config.shared_experts != sym_buffer.num_shared_experts:
        raise ValueError('configuration and symmetric-buffer shared experts disagree')
    if context is None:
        context = get_context(num_ranks=sym_buffer.group.size(),
                              requested_dispatch_sms=sym_buffer._requested_dispatch_sms)
    if context.dispatch_sms != sym_buffer._dispatch_sms:
        raise ValueError('K3 context dispatch SM count must match the workspace')
    projection_tensors = projection or ()
    device = sym_buffer.buffer.device
    weights = (*l1_weights, *l2_weights, *(x for x in
        (shared_l1_weights, shared_l2_weights, shared_x, shared_y, routed_norm,
         *projection_tensors, *(front[:3] if front else ())) if x is not None))
    tensors = (y, *weights)
    if (device.type != 'cuda' or torch.cuda.current_device() != device.index or
            int(context.device) != device.index or any(tensor.device != device for tensor in tensors)):
        raise ValueError('role tensors and Green contexts must use the current CUDA device')
    _check_ring_capacity(config, sym_buffer, context.compute_sms)
    compiler_target = _device_compile_target(sym_buffer.buffer.device)
    compiler_identity = _compiler_identity(compiler_target)
    nvlink_timeout_secs = _get_gc_moe_nvlink_barrier_timeout_secs()
    routed_weight_linger = _routed_weight_linger(config, sym_buffer.num_topk)
    shared_block_m = get_shared_block_m(config.block_m, config.tokens)
    # Two or more shared M-blocks pair their weight reads on adjacent
    # clusters; the shared stream then requests L2 residency.
    shared_weight_linger = ceil_div(config.tokens, shared_block_m) >= 2
    storage_block_m = max(config.block_m, shared_block_m)
    routed_block_m = config.block_m if config.block_m < shared_block_m <= 128 else None
    epilogue_config = get_role_config(COMPUTE_ROLE, config.block_m, sym_buffer.num_experts,
                                      config.hidden, local_experts=config.local_experts)
    role_configs = tuple(get_role_config(
        role, storage_block_m if role == COMPUTE_ROLE else config.block_m,
        sym_buffer.num_experts, config.hidden, local_experts=config.local_experts,
        routed_block_m=routed_block_m if role == COMPUTE_ROLE else None,
        route_entries=config.tokens * sym_buffer.num_topk if role == DISPATCH_ROLE else 0,
    ) for role in WORKER_ROLES)
    # The source summary aggregates at most one warp worth of ranks.
    # Select its protocol from group-wide geometry, including zero-token ranks.
    per_source_ready = sym_buffer.group.size() <= 32
    tokens_per_pass = role_configs[0].routing_warps * (32 // sym_buffer.num_topk)
    # At most four routing iterations on the publishing CTA.
    single_cta_routes = per_source_ready and config.tokens <= 4 * tokens_per_pass
    role_resources = tuple((item.num_stages, item.dynamic_smem, item.threads,
                            item.shared_layout.routed_payload)
                           for item in role_configs)
    finalize_config = role_configs[2]
    finalize_layout = finalize_config.shared_layout
    combine_reusable_bytes = finalize_layout.reusable_bytes
    if finalize_layout.field_end > finalize_config.dynamic_smem:
        raise ValueError('finalize shared storage exceeds its launch allocation')
    combine_geometry = None
    tile_rows = projection_stages = 0
    projection_channels = PROJECTION_CHANNELS_PER_CTA
    projection_splits = 1
    if projection:
        clusters = context.compute_sms // 2
        channels = config.shared_hidden // (2 * projection_channels)
        packed_rows = next((m for m in PROJECTION_ROW_TILES if config.tokens <= m), 0)
        tile_rows = next((m for m in PROJECTION_ROW_TILES if config.tokens <= 2 * m),
                         PROJECTION_ROW_TILES[-1])
        if packed_rows and config.tokens > packed_rows // 2 and 2 * channels <= clusters:
            tile_rows = packed_rows
            if 'projection_partials' in sym_buffer.layout.views:
                projection_splits = 2
            else:
                projection_channels //= 2
        elif 'projection_partials' in sym_buffer.layout.views:
            rows_per_wave = ceil_div(config.tokens, tile_rows)
            partial_rows = sym_buffer.layout.views['projection_partials'].shape[0]
            # A two-tile tail may split K when both halves can occupy at
            # most one wave each and the partial allocation covers all rows.
            if (rows_per_wave == 2 and config.tokens <= partial_rows
                    and rows_per_wave * channels <= clusters):
                projection_splits = 2
        combine_geometry = get_combine_geometry(config.hidden)
        combine_end = align(COMBINE_WARPS * 4, 1024)
        k_slices = config.hidden // (PROJECTION_K_TILE * projection_splits)
        # Each projection job restarts at stage zero. Multiple jobs on a
        # cluster must leave the ring at that slot with a well-defined phase.
        for rows in reversed(PROJECTION_ROW_TILES):
            if rows > tile_rows:
                continue
            jobs = (ceil_div(config.tokens, rows)
                    * (config.shared_hidden // (2 * projection_channels)) * projection_splits)
            stage_bytes = PROJECTION_K_TILE * 2 * (projection_channels + rows // 2)
            projection_stages = next((stages for stages in (10, 8, 7, 6, 4, 2)
                if stages <= min(k_slices, finalize_config.num_stages)
                and combine_end + stages * stage_bytes <= combine_reusable_bytes
                and (jobs <= clusters or k_slices % stages == 0)), 0)
            if projection_stages:
                tile_rows = rows
                break
        if not projection_stages:
            raise ValueError('K3 combine and projection exceed the available shared memory')
    # Enable the measured weight hints for one fully occupied, widest
    # split-K tile. A live row count can change on each graph replay.
    full_split_k_tile = (projection_splits > 1 and tile_rows == PROJECTION_ROW_TILES[-1]
                         and config.tokens == tile_rows and (front is None or front[2] is None))
    combine_resources = (combine_reusable_bytes, combine_geometry,
                         tile_rows, projection_stages, projection_channels, projection_splits,
                         full_split_k_tile)
    key = _capture_key(config, sym_buffer, context, y, l1_weights, l2_weights,
                       shared_l1_weights, shared_l2_weights,
                       nvlink_timeout_secs, routed_weight_linger,
                       role_resources, combine_resources,
                       compiler_identity, shared_x, shared_y, routed_norm, projection_tensors, front)
    with COMPILE_LOCK:
        prepared = _CACHE.get(key)
        if prepared is not None:
            _CACHE.move_to_end(key)
    if prepared is None:
        with COMPILE_LOCK:
            prepared = _CACHE.get(key)
            if prepared is None:
                layout = sym_buffer.layout
                tma_plan = build_gc_tma_plan(epilogue_config, sym_buffer,
                                             l1_weights, l2_weights,
                                             shared_l1_weights, shared_l2_weights,
                                             shared_x=shared_x, shared_block_m=shared_block_m)
                maps = _encode_tensor_maps(tma_plan.ordered())
                projection_maps = b""
                if projection:
                    up = projection[0]
                    projection_specs = (
                        make_tma_2d_desc(up, config.hidden, config.shared_hidden,
                                         PROJECTION_K_TILE, projection_channels, up.stride(0), 128),
                        make_tma_2d_desc(y, config.hidden, config.tokens,
                                         PROJECTION_K_TILE, tile_rows // 2, y.stride(0), 128),
                    )
                    projection_maps = _encode_tensor_maps(projection_specs)
                front_kernels = ()
                front_maps = b''
                if front:
                    front_weight, _, valid_token_count, renormalize = front
                    available_sms = context.compute_sms
                    if context.primary_ctx_roles:
                        available_sms += context.dispatch_sms
                    status, smem_budget = cu.cuDeviceGetAttribute(
                        cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                        context.device)
                    if status != cu.CUresult.CUDA_SUCCESS:
                        raise RuntimeError(f'cannot query K3 Front shared memory limit: {status}')
                    front_gemm = GCK3FrontKernel(
                        layout, config.tokens, available_sms=available_sms,
                        smem_budget=min(smem_budget, SMEM_CAPACITY))
                    shared_w1_columns = config.shared_intermediate // 128
                    shared_w1_jobs = ceil_div(config.tokens, shared_block_m) * shared_w1_columns
                    free_clusters = (context.compute_sms - front_gemm.grid) // 2
                    k_slices = front_gemm.input_hidden // PROJECTION_K_TILE
                    # Compute can issue a partial shared wave before Front
                    # completes. Require space for half of one W1 row tile's
                    # tasks. With several row tiles the remaining tasks queue
                    # normally and still wait on their existing readiness.
                    overlap_jobs = min(shared_w1_jobs, shared_w1_columns, context.compute_sms // 2)
                    front_gemm.early_shared = (
                        not context.primary_ctx_roles
                        and overlap_jobs > 0 and free_clusters >= ceil_div(overlap_jobs, 2)
                        and k_slices > 2 * front_gemm.stages)
                    router = GCK3RouterKernel(layout, config.tokens, renormalize,
                                             valid_token_count is not None)
                    front_kernels = (front_gemm, router)
                    front_maps = _encode_tensor_maps((
                        make_tma_2d_desc(front_weight, config.shared_hidden,
                            sym_buffer.num_experts + config.hidden, PROJECTION_K_TILE,
                            front_gemm.channels_per_cta, front_weight.stride(0), 128),
                        make_tma_2d_desc(shared_x, config.shared_hidden, config.tokens,
                            PROJECTION_K_TILE, front_gemm.rows // 2, shared_x.stride(0), 128),
                    ))
                local_entries = config.tokens * sym_buffer.num_topk
                route_entries = local_entries * sym_buffer.group.size()
                store_rows = max(epilogue_config.store_m_l1, epilogue_config.store_m_l2)
                # Below the two-store-groups-per-expert threshold, estimate whether local
                # routes could fill a store group on a distinct expert for each 2-CTA cluster.
                full_expert_blocks = min(config.local_experts, local_entries // store_rows)
                worker_clusters = (context.compute_sms + context.dispatch_sms) // 2
                assist = (bool(front) and not context.primary_ctx_roles and context.dispatch_sms % 2 == 0
                          and (route_entries >= 2 * sym_buffer.num_experts * store_rows
                               or full_expert_blocks >= worker_clusters))
                worker_roles = ASSIST_WORKER_ROLES if assist else WORKER_ROLES
                role_kernels = []
                for node_role in worker_roles:
                    routed_assist = node_role == ROUTED_ASSIST_ROLE
                    role = COMPUTE_ROLE if routed_assist else node_role
                    index = WORKER_ROLES.index(role)
                    role_kernel = GCK3MoERoleKernel(
                        role, context.dispatch_sms, context.compute_sms,
                        config.block_m, config.hidden, config.intermediate,
                        config.tokens, sym_buffer.group.size(), config.local_experts,
                        config.shared_experts,
                        sym_buffer.num_max_tokens_per_rank,
                        ring_tokens=layout.ring_tokens,
                        sf_ring_tokens=layout.sf_ring_tokens,
                        topk=sym_buffer.num_topk,
                        shared_hidden=config.shared_hidden,
                        shared_intermediate=config.shared_intermediate,
                        activation_beta=config.beta,
                        activation_linear_beta=config.linear_beta,
                        # Only the clamp-presence flag specializes the binary. The
                        # actual FP32 value is a kernel argument on every replay.
                        activation_clamp=0.0 if config.clamp != inf else inf,
                        fast_math=config.fast_math,
                        nvlink_timeout_secs=nvlink_timeout_secs,
                        routed_weight_linger=routed_weight_linger,
                        full_split_k_tile=full_split_k_tile and role in (COMPUTE_ROLE, FINALIZE_ROLE),
                        shared_weight_linger=shared_weight_linger,
                        role_config=role_configs[index],
                        per_source_ready=per_source_ready,
                        single_cta_routes=single_cta_routes,
                        shared_block_m=shared_block_m if role == COMPUTE_ROLE else config.block_m,
                        epilogue_config=epilogue_config if role == COMPUTE_ROLE else None,
                        combine_geometry=combine_geometry if role == FINALIZE_ROLE else None,
                        projection_tile_rows=tile_rows if role != COMPUTE_ROLE else 0,
                        projection_stages=projection_stages if role == FINALIZE_ROLE else 0,
                        projection_channels=projection_channels,
                        projection_splits=projection_splits if role == FINALIZE_ROLE else 1,
                        early_shared=bool(front_kernels and front_kernels[0].early_shared
                                          and role == COMPUTE_ROLE and not routed_assist),
                        routed_assist=routed_assist,
                        has_prefix=bool(role == FINALIZE_ROLE and projection and projection[1] is not None))
                    role_kernels.append(role_kernel)
                binary_key = _binary_key(
                    compiler_identity, (*front_kernels, *role_kernels), role_resources,
                    combine_resources, sym_buffer, y, weights)
                def compile_roles():
                    signal_view = from_dlpack(sym_buffer.buffer.view(torch.int32))
                    peer_view = from_dlpack(sym_buffer.buffer.view(torch.int64).flatten()[:74])
                    # These views supply CuTe's pointer type/shape only. Every
                    # TensorMap pointer load becomes a host-supplied value param.
                    map_storage = sym_buffer.buffer.view(torch.uint8).flatten()
                    map_views = tuple(from_dlpack(map_storage[i * 128:(i + 1) * 128])
                                      for i in range(TMA_MAP_COUNT))
                    y_view = from_dlpack(y)
                    shared_y_view = from_dlpack(shared_y)
                    common_views = (signal_view, peer_view, Uint32(0), Uint32(0))
                    compiled_binaries = []
                    if front_kernels:
                        gemm_policy, router_policy = front_kernels
                        workspace_view = from_dlpack(map_storage)
                        valid_view = None if valid_token_count is None else from_dlpack(valid_token_count.view(1))

                        def assemble_front(ptx, dump_dir):
                            ptx = _clean_cute_ptx(ptx)
                            if len(_entry_formals(ptx)) != 3:
                                raise RuntimeError('K3 front GEMM ABI changed')
                            for ordinal in (1, 2):
                                ptx = _rewrite_value_formal(ptx, ordinal, 128, 128)
                            return assemble_ptx(ptx, dump_dir, 'k3_front_gemm',
                                target=compiler_target, ptxas_options=compiler_identity.ptxas_options)

                        def assemble_router(ptx, dump_dir):
                            ptx = _clean_cute_ptx(ptx)
                            if len(_entry_formals(ptx)) != 2 + int(valid_view is not None):
                                raise RuntimeError('K3 router ABI changed')
                            return assemble_ptx(ptx, dump_dir, 'k3_router',
                                target=compiler_target, ptxas_options=compiler_identity.ptxas_options)

                        gemm_cubin = compile_kernel(gemm_policy, (workspace_view, *map_views[:2]),
                                         compiler_identity, assemble_front)
                        router_cubin = compile_kernel(router_policy,
                            (workspace_view, from_dlpack(front[1]), valid_view),
                            compiler_identity, assemble_router)
                        compiled_binaries.extend((
                            RoleBinary(FRONT_ROLE, gemm_cubin, gemm_policy.smem_bytes, 256, gemm_policy.grid, (2, 1, 1)),
                            RoleBinary(ROUTER_ROLE, router_cubin, router_policy.smem_bytes, 256, router_policy.tokens),
                        ))
                    for node_role, role_kernel in zip(worker_roles, role_kernels):
                        role = role_kernel.role
                        index = WORKER_ROLES.index(role)
                        compile_args = common_views
                        if role == COMPUTE_ROLE:
                            compile_args += (map_views[0], shared_y_view, Float32(0.0), *map_views[1:])
                        elif role == FINALIZE_ROLE:
                            compile_args += (None, y_view, None)
                            if projection:
                                _, prefix, out = projection
                                compile_args += ((None,) * (TMA_MAP_COUNT - 1) +
                                    (from_dlpack(routed_norm), Float32(1e-6), Float32(1.0),
                                     from_dlpack(out), shared_y_view,
                                     from_dlpack(out if prefix is None else prefix), *map_views[:2]))
                        cubin = compile_kernel(
                            role_kernel, compile_args, compiler_identity,
                            lambda ptx, dump_dir: _assemble_role_value_params(
                                ptx, role, dump_dir, target=compiler_target,
                                has_shared=config.shared_experts > 0,
                                routed_assist=role_kernel.routed_assist,
                                with_projection=role == FINALIZE_ROLE and bool(projection),
                                ptxas_options=compiler_identity.ptxas_options))
                        role_config = role_configs[index]
                        compiled_binaries.append(RoleBinary(
                            node_role, cubin, role_config.dynamic_smem,
                            role_config.threads,
                            context.dispatch_sms if node_role in (DISPATCH_ROLE, ROUTED_ASSIST_ROLE) else
                            context.compute_sms if role == COMPUTE_ROLE else
                            role_kernel.finalize_sms,
                            cluster=(2, 1, 1) if role == COMPUTE_ROLE or (role == FINALIZE_ROLE and tile_rows)
                            else (1, 1, 1),
                            programmatic_dependencies=(FRONT_ROLE,) if role_kernel.early_shared else ()))
                    if projection_splits > 1:
                        _, prefix, out = projection
                        reduce = GCK3ProjectionReduceKernel(
                            layout, config.tokens, prefix is not None)

                        def assemble_reduce(ptx, dump_dir):
                            ptx = _clean_cute_ptx(ptx)
                            if len(_entry_formals(ptx)) != 3:
                                raise RuntimeError('K3 projection reduction ABI changed')
                            return assemble_ptx(ptx, dump_dir, 'k3_projection_reduce',
                                target=compiler_target, ptxas_options=compiler_identity.ptxas_options)

                        cubin = compile_kernel(reduce, (from_dlpack(map_storage), from_dlpack(out),
                            from_dlpack(out if prefix is None else prefix)), compiler_identity, assemble_reduce)
                        compiled_binaries.append(RoleBinary(PROJECTION_REDUCE_ROLE, cubin, 0, 256, reduce.grid))
                    return compiled_binaries

                binaries = _get_or_compile_binaries(
                    binary_key, compiler_identity, compiler_target, compile_roles)
                prepared = _Prepared(maps, projection_maps, binaries, front_maps)
                if _compiler_identity(compiler_target) != compiler_identity:
                    raise RuntimeError('compiler identity changed while preparing role binaries')
                _CACHE[key] = prepared
                while len(_CACHE) > _PREPARED_CACHE_CAPACITY:
                    _CACHE.popitem(last=False)
            _CACHE.move_to_end(key)
    if len(prepared.maps) != TMA_MAP_COUNT * 128:
        raise RuntimeError('cached TensorMap payload disagrees with its configuration')
    signal_arg = ctypes.c_void_p(sym_buffer.buffer.data_ptr())
    peer_arg = _sym_buffer_value(sym_buffer)
    rank_arg = ctypes.c_uint32(sym_buffer.rank_idx)
    token_arg = ctypes.c_uint32(config.tokens)
    clamp_arg = ctypes.c_float(config.clamp)
    map_args = tuple((ctypes.c_uint8 * 128).from_buffer_copy(
        prepared.maps[i * 128:(i + 1) * 128]) for i in range(TMA_MAP_COUNT))
    # compute writes shared output to the workspace for finalize.
    shared_y_arg = ctypes.c_void_p(shared_y.data_ptr())
    common_args = (signal_arg, peer_arg, rank_arg, token_arg)
    arguments = (common_args,
                 (*common_args, clamp_arg, shared_y_arg, *map_args),
                 (*common_args, ctypes.c_void_p(y.data_ptr())))
    if projection:
        if len(prepared.projection_maps) != 256:
            raise RuntimeError('cached finalize TensorMap payload disagrees with its configuration')
        _, prefix, out = projection
        prefix = out if prefix is None else prefix
        projection_maps = tuple((ctypes.c_uint8 * 128).from_buffer_copy(
            prepared.projection_maps[offset:offset + 128]) for offset in (0, 128))
        arguments = (*arguments[:2], (*arguments[2], ctypes.c_void_p(routed_norm.data_ptr()),
                                      ctypes.c_float(norm_eps), ctypes.c_float(routed_scale),
                                      ctypes.c_void_p(out.data_ptr()), shared_y_arg,
                                      ctypes.c_void_p(prefix.data_ptr()), *projection_maps))
    if any(b.role == ROUTED_ASSIST_ROLE for b in prepared.binaries):
        dispatch_args, compute_args, finalize_args = arguments
        arguments = (dispatch_args, compute_args, compute_args, finalize_args)
    if projection_splits > 1:
        arguments = (*arguments, (signal_arg, ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(prefix.data_ptr())))
    if front:
        if len(prepared.front_maps) != 256:
            raise RuntimeError('cached front TensorMap payload disagrees with its configuration')
        maps = tuple((ctypes.c_uint8 * 128).from_buffer_copy(
            prepared.front_maps[offset:offset + 128]) for offset in (0, 128))
        route_args = (signal_arg, ctypes.c_void_p(front[1].data_ptr()))
        if front[2] is not None:
            route_args += (ctypes.c_void_p(front[2].data_ptr()),)
        arguments = ((signal_arg, *maps), route_args, *arguments)
    return list(prepared.binaries), arguments

def launch_gc_moe_k3(config, y, l1_weights, l2_weights, sym_buffer,
                  shared_l1_weights, shared_l2_weights, *, shared_x, shared_y,
                  routed_norm=None, norm_eps=1e-6, routed_scale=1.0, projection=None, front=None) -> None:
    context = get_context(num_ranks=sym_buffer.group.size(),
                          requested_dispatch_sms=sym_buffer._requested_dispatch_sms)
    binaries, arguments = prepare_role_binaries(
        config, y, l1_weights, l2_weights, sym_buffer,
        shared_l1_weights, shared_l2_weights, context,
        shared_x=shared_x, shared_y=shared_y, routed_norm=routed_norm, norm_eps=norm_eps,
        routed_scale=routed_scale, projection=projection, front=front)
    # Snapshot the allocation/IPC handle, since a wrapper may later destroy
    # or detach its views. CUDA owns these references through graph clones
    # and in-flight executions; the prepared descriptor cache owns none.
    keepalive = (*l1_weights, *l2_weights, y,
                 sym_buffer.buffer, sym_buffer.handle,
                 shared_l1_weights, shared_l2_weights, shared_x, shared_y, routed_norm,
                 *(projection or ()), *(front[:3] if front else ()))
    context.launch(binaries, arguments, keepalive=keepalive)
