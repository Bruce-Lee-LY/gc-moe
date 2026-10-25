"""Prepare the three CuTe GC role cubins and live TensorMaps for a call.

The supported H/I and token ranges are checked before graph launch. Each
role's PTX/cubin export is owned by its compile, including concurrent ranks.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import inspect
from math import inf
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import threading

from cutlass import Float32, Uint32
import cutlass.cute as cute
from cutlass.base_dsl.dsl import BaseDSL
from cutlass._mlir._mlir_libs import _cutlass_ir
from cutlass.base_dsl.compiler import CompileOptions, DeviceTarget, ExtraCompilerOpts, GPUArch, LinkLibraries, OptLevel, PtxasOptions, RDC
from cutlass.cute.runtime import from_dlpack
import torch
from cuda.bindings import driver as cu

from ...comm.gc_barrier import DEFAULT_NVLINK_BARRIER_TIMEOUT_SECS
from ...kernels.sm100_fp8_fp4_gc_moe import GCMoERoleKernel
from ...layout.gc_moe import (
    MAX_RANKS,
    align,
    ceil_div,
    get_role_shared_layout,
)
from ...runtime.gc_moe import RoleBinary
from ..heuristics.gc_moe import (
    get_combine_geometry,
    get_role_config,
    num_max_live_pool_blocks,
)
from .tma_descriptors import build_gc_tma_plan


def _get_gc_moe_nvlink_barrier_timeout_secs() -> int:
    """Read the positive signed-decimal int32 barrier timeout."""
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


_SYSTEM_PTXAS = '/usr/local/cuda/bin/ptxas'
_PTXAS_OPTIONS = ('-O3', '--register-usage-level=10', '--fmad=true')
_COMPILE_LOCK = threading.RLock()


@dataclass(frozen=True)
class _CompilerIdentity:
    target: str
    compile_options: str
    ptxas_options: tuple[str, ...]
    tool_realpath: str
    tool_stamp: tuple[int, ...]
    tool_version: str
    tool_sha256: str
    environment: str
    effective_options: str
    frontend_sha256: str
    source_sha256: str
    native_library: tuple
    bridge_revision: int = 2


@lru_cache(maxsize=16)
def _tool_identity(realpath: str, stamp: tuple[int, ...]) -> tuple[str, str]:
    result = subprocess.run([_SYSTEM_PTXAS, '--version'], check=False,
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('cannot query the system PTX assembler')
    return ((result.stdout + result.stderr).strip(),
            hashlib.sha256(Path(realpath).read_bytes()).hexdigest())


@lru_cache(maxsize=16)
def _binary_identity(realpath: str, stamp: tuple[int, ...]) -> str:
    return hashlib.sha256(Path(realpath).read_bytes()).hexdigest()


@lru_cache(maxsize=16)
def _source_identity(stamps: tuple) -> str:
    digest = hashlib.sha256()
    for name, *_ in stamps:
        path = Path(name)
        digest.update(str(path.relative_to(Path(__file__).resolve().parents[2])).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _compiler_identity(target: str) -> _CompilerIdentity:
    tool = Path(_SYSTEM_PTXAS)
    if not tool.is_file() or not os.access(tool, os.X_OK):
        raise RuntimeError(f"an executable system PTX assembler is required at {tool}")
    info = tool.stat()
    stamp = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    realpath = str(tool.resolve())
    version, digest = _tool_identity(realpath, stamp)
    assembly_options = ' '.join(_PTXAS_OPTIONS)
    options = (f"--gpu-arch {target} --opt-level 3 "
               f"--ptxas-options {shlex.quote(assembly_options)}")
    function = inspect.unwrap(GCMoERoleKernel.__call__)
    with _COMPILE_LOCK:
        BaseDSL._lazy_initialize_dsl(function)
        settings = function._dsl_object.envar
    effective = CompileOptions((GPUArch(target), OptLevel(3),
                                PtxasOptions(assembly_options)))
    effective.apply_envar_settings(settings, 'gc_moe_role')
    if (effective.options[GPUArch].value != target or
            effective.options[OptLevel].value != 3):
        raise RuntimeError('compiler environment overrides the required target or optimization level')
    if (effective.options[RDC].value or effective.options[DeviceTarget].value or
            effective.options[LinkLibraries].value):
        raise RuntimeError('GC roles require standalone kernel compilation without device linking')
    extra = effective.options[ExtraCompilerOpts].value
    if re.search(r'cubin-chip|gpu-arch|opt-level|ptx-options|fast.?math|unsafe.?fp|'
                 r'ftz|fmad|contract|prec.?div|prec.?sqrt|device-debug', extra, re.IGNORECASE):
        raise RuntimeError('raw compiler overrides conflict with the role compilation policy')
    normalized = effective.to_str()
    assembly = tuple(shlex.split(effective.options[PtxasOptions].value.strip("'")))
    allowed = {*_PTXAS_OPTIONS, '-v', '--verbose', '--warn-on-spills',
               '--warn-on-local-memory-usage', '--generate-line-info'}
    if (any(assembly.count(option) != 1 for option in _PTXAS_OPTIONS) or
            any(option not in allowed for option in assembly)):
        raise RuntimeError('assembler options conflict with the role compilation policy')
    if effective.generate_line_info and '--generate-line-info' not in assembly:
        assembly += ('--generate-line-info',)
    source_root = Path(__file__).resolve().parents[2]
    stamps = []
    for path in sorted(source_root.rglob('*.py')):
        stat = path.stat()
        stamps.append((str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns))
    native = Path(_cutlass_ir.__file__).resolve()
    native_stat = native.stat()
    native_stamp = (native_stat.st_dev, native_stat.st_ino, native_stat.st_size,
                    native_stat.st_mtime_ns)
    native_identity = (str(native), native_stamp, _binary_identity(str(native), native_stamp))
    return _CompilerIdentity(
        target, options, assembly, realpath, stamp, version, digest,
        settings.cache_key_str(), normalized,
        function._dsl_object.get_version().hexdigest(), _source_identity(tuple(stamps)),
        native_identity)


@dataclass
class _Prepared:
    maps: bytes
    binaries: tuple[RoleBinary, RoleBinary, RoleBinary]


_CACHE: dict[tuple, _Prepared] = {}


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
    # address, so only the eighteen role2 maps use the raw form here.
    address_op = 'mov.b64' if raw_param_address else 'cvta.param::entry.u64'
    ptx = load.sub(
        lambda m: f"{m.group(1)}{address_op} {m.group(2)}, {name};",
        ptx, count=1)
    return ptx


def _convert_role2_map_uses(ptx: str) -> str:
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
    map_ordinals = (4, *range(8, 25))
    raw_map_regs = []
    for ordinal in map_ordinals:
        raw = re.findall(
            r"(?m)^[ \t]*mov\.b64[ \t]+(%rd[0-9]+),[ \t]*"
            r"[A-Za-z0-9_.$]+_param_" + str(ordinal) + r";[ \t]*$", ptx)
        if len(raw) != 1:
            raise RuntimeError(f"role2 raw TensorMap {ordinal} changed")
        raw_map_regs.append(raw[0])
    if len(set(raw_map_regs)) != 18:
        raise RuntimeError('role2 TensorMap address registers alias')

    register_decl = re.compile(
        r"(?m)^([ \t]*\.reg[ \t]+\.b64[ \t]+%rd<)([0-9]+)(>;[ \t]*)$")
    declarations = list(register_decl.finditer(ptx))
    if len(declarations) != 1:
        raise RuntimeError('role2 64-bit PTX register declaration changed')
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
    if len(prefetch_regs) != 18 or set(prefetch_regs) != set(raw_map_regs):
        raise RuntimeError('role2 TensorMap prefetch set changed')
    tensor_op_count = len(re.findall(
        r"(?m)^[ \t]*cp\.async\.bulk\.tensor\.", ptx))
    g2s_count = len(list(g2s.finditer(ptx)))
    s2g_count = len(list(s2g.finditer(ptx)))
    if g2s_count != 16 or s2g_count < 1 or tensor_op_count != \
            g2s_count + s2g_count:
        raise RuntimeError('role2 TensorMap load/store PTX forms changed')
    selected_conversions = re.findall(
        r"(?m)^[ \t]*cvta\.param\.u64[ \t]+(%rd[0-9]+),[ \t]*"
        r"(%rd[0-9]+);[ \t]*$", ptx)
    loader_regs = {m.group(2) for m in g2s.finditer(ptx)}
    if (len(selected_conversions) != 4 or
            len(loader_regs) != 4 or
            loader_regs != {dst for dst, _ in selected_conversions} or
            len({src for _, src in selected_conversions}) != 4 or
            any(src in raw_map_regs for _, src in selected_conversions)):
        raise RuntimeError('role2 loader selected-map conversions changed')
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
            source_cvta_count + 18 + s2g_count:
        raise RuntimeError('role2 TensorMap conversion count changed')
    return ptx


def _device_compile_target(device: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) != (10, 3):
        raise NotImplementedError('current three-role device path requires compute capability 10.3')
    suffix = 'f' if major > 9 else 'a' if major == 9 else ''
    return f"sm_{major * 10 + minor}{suffix}"


def _ptx_target(ptx: str) -> str:
    targets = re.findall(r"(?m)^\.target[ \t]+(sm_[0-9]+[af]?)(?:[ \t]*,.*)?[ \t]*$", ptx)
    if len(targets) != 1:
        raise RuntimeError('PTX target header is missing or ambiguous')
    return targets[0]


def _assemble_value_ptx(ptx: str, dump_dir: str,
                        basename: str, *, target: str,
                        ptxas_options: tuple[str, ...] = _PTXAS_OPTIONS) -> tuple[str, bytes, str]:
    if _ptx_target(ptx) != target:
        raise RuntimeError('PTX target differs from the selected device target')
    ptx_version = re.search(r"(?m)^\.version[ \t]+([0-9]+)\.([0-9]+)[ \t]*$",
                            ptx)
    if ptx_version is None:
        raise RuntimeError('PTX version header is missing')
    minimum_release = ((13, 2) if
                       (int(ptx_version.group(1)), int(ptx_version.group(2)))
                       >= (9, 1) else (13, 0))
    # Assemble every role with the system CUDA toolchain.
    tool = _SYSTEM_PTXAS
    required = (f"CUDA {minimum_release[0]}.{minimum_release[1]} or newer "
                f"ptxas is required at {tool} for these by-value PTX parameters")
    if not Path(tool).is_file() or not os.access(tool, os.X_OK):
        raise RuntimeError(required)
    result = subprocess.run([tool, '--version'], check=False,
                            capture_output=True, text=True)
    version = (result.stdout + result.stderr).strip()
    release = re.search(r"release[ \t]+([0-9]+)\.([0-9]+)", version)
    if (result.returncode != 0 or release is None or
            (int(release.group(1)), int(release.group(2))) < minimum_release):
        raise RuntimeError(required)
    source = Path(dump_dir) / f"{basename}.ptx"
    output = Path(dump_dir) / f"{basename}.cubin"
    source.write_text(ptx)
    assembled = subprocess.run([tool, f'-arch={target}', *ptxas_options,
                                '-o', str(output), str(source)],
                               capture_output=True, text=True)
    if assembled.returncode:
        raise RuntimeError(f"{basename} assembly failed: " +
                           assembled.stderr)
    return ptx, output.read_bytes(), version


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


def _order_role2_value_formals(ptx: str) -> str:
    """Keep ordinary scalars ahead of the eighteen contiguous value maps."""
    formals = _entry_formals(ptx)
    if len(formals) != 25:
        raise RuntimeError('role2 value-formal count changed')
    for index, line in enumerate(formals):
        if re.search(r"\b[A-Za-z0-9_.$]+_param_" + str(index) +
                     r"(?:\[[0-9]+\])?,?[ \t]*$", line) is None:
            raise RuntimeError('role2 value-formal order changed')
    order = (0, 1, 2, 3, 5, 6, 7, 4, *range(8, 25))
    ordered = [formals[i].rstrip().rstrip(',') for i in order]
    original = '\n'.join(formals)
    if ptx.count(original) != 1:
        raise RuntimeError('role2 value-formal signature changed')
    return ptx.replace(original, ',\n'.join(ordered), 1)


def _assemble_role_value_params(ptx: str, role: int,
                                dump_dir: str, *, target: str,
                                ptxas_options: tuple[str, ...] = _PTXAS_OPTIONS) -> tuple[str, bytes, str]:
    """Build each role with relative peer value, plus role2's value maps."""
    ptx = _clean_cute_ptx(ptx)
    if _ptx_target(ptx) != target:
        raise RuntimeError('CuTe generated a different device target')
    expected = {1: ('ptr', 'ptr', 'u32', 'u32', 'f32', 'f32'),
                2: (('ptr', 'ptr', 'u32', 'u32', 'ptr', 'ptr', 'f32', 'f32') +
                    ('ptr',) * 17),
                3: ('ptr', 'ptr', 'u32', 'u32', 'ptr', 'ptr', 'ptr', 'ptr',
                    'f32', 'f32')}[role]
    formals = _entry_formals(ptx)
    if len(formals) != len(expected):
        raise RuntimeError(f"role{role} CuTe parameter count changed")
    for index, (line, kind) in enumerate(zip(formals, expected)):
        form = (r"\.align[ \t]+8[ \t]+\.b8[ \t]+\S+_param_" +
                str(index) + r"\[8\]" if kind == 'ptr' else
                r"\." + kind + r"[ \t]+\S+_param_" + str(index))
        if re.search(form + r",?[ \t]*$", line) is None:
            raise RuntimeError(f"role{role} CuTe formal {index} changed")
    if 'cvta.to.param::entry.u64' not in ptx or 'ld.param.u64' not in ptx:
        raise RuntimeError(f"role{role} peer parameter load path is missing")
    ptx = _rewrite_value_formal(ptx, 1, 592, 8)
    if role == 2:
        for map_ordinal in (4, *range(8, 25)):
            ptx = _rewrite_value_formal(
                ptx, map_ordinal, 128, 128, raw_param_address=True)
        ptx = _convert_role2_map_uses(ptx)
        ptx = _order_role2_value_formals(ptx)
    return _assemble_value_ptx(ptx, dump_dir, f"role{role}_value_params",
                                target=target, ptxas_options=ptxas_options)


def _capture_key(config, sym_buffer, green_context, l1_weights, l2_weights,
                 shared_l1_weights, shared_l2_weights, stats,
                 nvlink_timeout_secs: int, role_resources: tuple,
                 combine_resources: tuple, compiler_identity: _CompilerIdentity) -> tuple:
    status, cuda_context = cu.cuCtxGetCurrent()
    if status != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cannot query CUDA context: {status}")
    layout = sym_buffer.layout
    weights = (*l1_weights, *l2_weights,
               *(shared_l1_weights or ()), *(shared_l2_weights or ()))
    return (int(cuda_context), compiler_identity, green_context.dispatch_sms,
            green_context.compute_sms,
            sym_buffer.buffer.data_ptr(),
            tuple(sym_buffer.peer_addresses), sym_buffer.rank_idx,
            sym_buffer.group.size(), sym_buffer.num_experts,
            sym_buffer.num_max_tokens_per_rank,
            config.tokens, config.block_m, config.hidden, config.intermediate,
            config.local_experts, config.shared_experts,
            config.shared_weight_gran_k, config.clamp != inf,
            config.fast_math, sym_buffer.num_topk,
            nvlink_timeout_secs,
            stats is not None,
            role_resources, combine_resources,
            layout.nbytes, layout.ring_tokens, layout.sf_ring_tokens,
            *(item for tensor in weights
              for item in (tensor.data_ptr(), tuple(tensor.shape),
                           tuple(tensor.stride()), str(tensor.dtype))))


def _check_supported(config, sym_buffer) -> None:
    if (not 1 <= sym_buffer.group.size() <= MAX_RANKS or
            not 1 <= config.local_experts <= 512 or
            sym_buffer.group.size() * config.local_experts > 2048 or
            not 0 < sym_buffer.num_topk or
            sym_buffer.num_topk + bool(config.shared_experts) > 32 or
            config.shared_experts < 0):
        raise NotImplementedError('current CuTe device path needs 1..512 experts/rank, <=2048 total experts, topk>=1 and topk+shared slot<=32')
    if (config.hidden <= 0 or config.intermediate <= 0 or
            config.hidden % 256 != 0 or config.intermediate % 128 != 0 or
            (2 * config.intermediate) // 128 > 64 or
            config.block_m not in (16, 32, 64, 96, 128, 192, 240) or
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


def prepare_role_binaries(config, y: torch.Tensor,
                          l1_weights: tuple[torch.Tensor, torch.Tensor],
                          l2_weights: tuple[torch.Tensor, torch.Tensor],
                          sym_buffer, shared_l1_weights=None,
                          shared_l2_weights=None, stats=None,
                          context=None):
    """Encode live maps once per pointer set and compile each role once/shape."""
    _check_supported(config, sym_buffer)
    if context is None:
        from ...runtime.gc_moe import get_context
        context = get_context()
    device = sym_buffer.buffer.device
    tensors = (y, *l1_weights, *l2_weights, *(shared_l1_weights or ()),
               *(shared_l2_weights or ()), *((stats,) if stats is not None else ()))
    if (device.type != 'cuda' or torch.cuda.current_device() != device.index or
            int(context.device) != device.index or any(tensor.device != device for tensor in tensors)):
        raise ValueError('role tensors and Green contexts must use the current CUDA device')
    _check_ring_capacity(config, sym_buffer, context.compute_sms)
    compiler_target = _device_compile_target(sym_buffer.buffer.device)
    compiler_identity = _compiler_identity(compiler_target)
    nvlink_timeout_secs = _get_gc_moe_nvlink_barrier_timeout_secs()
    pull_bytes = config.hidden
    while pull_bytes > 8192:
        pull_bytes //= 2
    role_configs = tuple(get_role_config(role, config.block_m,
                                         sym_buffer.num_experts,
                                         config.hidden, pull_bytes)
                         for role in (1, 2, 3))
    role_resources = tuple((item.num_stages, item.dynamic_smem, item.threads)
                           for item in role_configs)
    role3_config = role_configs[2]
    role3_layout = get_role_shared_layout(
        3, config.block_m, sym_buffer.num_experts, pull_bytes, role3_config.num_stages)
    combine_reusable_bytes = role3_layout.reusable_bytes
    if role3_layout.field_end > role3_config.dynamic_smem:
        raise ValueError('role3 shared storage exceeds its launch allocation')
    combine_chunks, combine_chunk_bytes, _, _ = get_combine_geometry(
        config.hidden, combine_reusable_bytes, role3_config.dynamic_smem)
    combine_resources = (combine_reusable_bytes, combine_chunks,
                         combine_chunk_bytes)
    key = _capture_key(config, sym_buffer, context, l1_weights, l2_weights,
                       shared_l1_weights, shared_l2_weights, stats,
                       nvlink_timeout_secs, role_resources, combine_resources,
                       compiler_identity)
    prepared = _CACHE.get(key)
    if prepared is None:
        with _COMPILE_LOCK:
            prepared = _CACHE.get(key)
            if prepared is None:
                layout = sym_buffer.layout
                tma_plan = build_gc_tma_plan(role_configs[1], sym_buffer,
                                             l1_weights, l2_weights,
                                             shared_l1_weights, shared_l2_weights,
                                             config.shared_weight_gran_k)
                payloads = []
                for i, spec in enumerate(tma_plan.ordered()):
                    descriptor = spec.encode()
                    payload = ctypes.string_at(descriptor.getPtr(), 128)
                    if len(payload) != 128:
                        raise RuntimeError(f"TensorMap {i} is not 128 bytes")
                    payloads.append(payload)
                if len(payloads) != 18:
                    raise RuntimeError('GC role2 needs exactly 18 ordered TensorMaps')
                maps = b"".join(payloads)
                signal_view = from_dlpack(sym_buffer.buffer.view(torch.int32))
                peer_view = from_dlpack(sym_buffer.buffer.view(torch.int64).flatten()[:74])
                # These distinct views supply CuTe's pointer type/shape at compile
                # time only. Every pointer load is rewritten to its own 128B
                # host-supplied TensorMap value parameter.
                map_storage = sym_buffer.buffer.view(torch.uint8).flatten()
                map_views = tuple(from_dlpack(map_storage[i * 128:(i + 1) * 128])
                                  for i in range(18))
                role3_map_view = from_dlpack(map_storage[:2304])
                y_view = from_dlpack(y)
                stats_view = from_dlpack(stats) if stats is not None else signal_view
                binaries = []
                for index, role in enumerate((1, 2, 3)):
                    role_kernel = GCMoERoleKernel(
                        role, context.dispatch_sms, context.compute_sms,
                        config.block_m, config.hidden, config.intermediate,
                        config.tokens, sym_buffer.group.size(), config.local_experts,
                        config.shared_experts,
                        sym_buffer.num_max_tokens_per_rank,
                        math_debug=role == 2,
                        finalize_output=role == 3,
                        ring_tokens=layout.ring_tokens,
                        sf_ring_tokens=layout.sf_ring_tokens,
                        topk=sym_buffer.num_topk,
                        shared_weight_gran_k=config.shared_weight_gran_k,
                        # Only the clamp-presence flag specializes the binary. The
                        # actual FP32 value is a kernel argument on every replay.
                        activation_clamp=0.0 if config.clamp != inf else inf,
                        fast_math=config.fast_math,
                        has_stats=stats is not None,
                        nvlink_timeout_secs=nvlink_timeout_secs,
                        math_stages=role_configs[index].num_stages,
                        combine_reusable_bytes=(combine_reusable_bytes if role == 3
                                                else None),
                        combine_launch_bytes=(role3_config.dynamic_smem if role == 3
                                              else None))
                    rank_view = Uint32(0)
                    token_view = Uint32(0)
                    clamp_view = Float32(0.0)
                    alpha_view = Float32(1.0)
                    compile_args = ((signal_view, peer_view, rank_view, token_view,
                                     map_views[0], signal_view, None, None,
                                     clamp_view, alpha_view, *map_views[1:])
                                    if role == 2 else
                                    (signal_view, peer_view, rank_view, token_view,
                                     role3_map_view, signal_view, y_view, stats_view,
                                     clamp_view, alpha_view)
                                    if role == 3 else
                                    (signal_view, peer_view, rank_view, token_view,
                                     None, None, None, None,
                                     clamp_view, alpha_view))
                    # JitFunctionArtifacts reads named PTX/cubin files after compile.
                    # Its basename does not encode all of this object's constexpr
                    # fields, so concurrent ranks with different shapes can overwrite
                    # each other's files. Own the files per role compile; the CuTe JIT
                    # cache and the compiled bytes retained below are unaffected.
                    with tempfile.TemporaryDirectory(prefix='gc_moe_cute_role_') as dump_dir:
                        compiled = cute.compile(
                            role_kernel, *compile_args,
                            options=(compiler_identity.compile_options +
                                     f" --keep-cubin --keep-ptx --dump-dir {dump_dir}"))
                        ptx = compiled.artifacts.PTX
                        cubin = compiled.artifacts.CUBIN
                        _, cubin, _ = _assemble_role_value_params(
                            ptx, role, dump_dir, target=compiler_target,
                            ptxas_options=compiler_identity.ptxas_options)
                    role_config = role_configs[index]
                    binaries.append(RoleBinary(
                        role, cubin, role_config.dynamic_smem,
                        role_config.threads,
                        context.dispatch_sms if role == 1 else
                        context.compute_sms if role == 2 else
                        role_kernel.finalize_sms))
                prepared = _Prepared(maps, tuple(binaries))
                if _compiler_identity(compiler_target) != compiler_identity:
                    raise RuntimeError('compiler identity changed while preparing role binaries')
                _CACHE[key] = prepared
    signal_arg = ctypes.c_void_p(sym_buffer.buffer.data_ptr())
    peer_arg = _sym_buffer_value(sym_buffer)
    rank_arg = ctypes.c_uint32(sym_buffer.rank_idx)
    token_arg = ctypes.c_uint32(config.tokens)
    clamp_arg = ctypes.c_float(config.clamp)
    alpha_arg = ctypes.c_float(config.alpha)
    map_args = tuple((ctypes.c_uint8 * 128).from_buffer_copy(
        prepared.maps[i * 128:(i + 1) * 128]) for i in range(18))
    # Role3 retains its original 8-byte unused CuTe pointer formal. Role2
    # carries the eighteen separate 128-byte descriptor values.
    role3_map_arg = ctypes.c_void_p(sym_buffer.buffer.data_ptr())
    y_arg = ctypes.c_void_p(y.data_ptr())
    stats_arg = ctypes.c_void_p(stats.data_ptr() if stats is not None else
                               sym_buffer.buffer.data_ptr())
    arguments = ((signal_arg, peer_arg, rank_arg, token_arg,
                  clamp_arg, alpha_arg),
                 (signal_arg, peer_arg, rank_arg, token_arg, signal_arg,
                  clamp_arg, alpha_arg, *map_args),
                 (signal_arg, peer_arg, rank_arg, token_arg, role3_map_arg, signal_arg,
                  y_arg, stats_arg, clamp_arg, alpha_arg))
    return list(prepared.binaries), arguments
