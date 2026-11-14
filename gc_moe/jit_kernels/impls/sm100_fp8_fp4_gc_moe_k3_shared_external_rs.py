"""K3 role2的PTX组装、TensorMap构建、二进制缓存及启动参数持有。"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from math import inf
from pathlib import Path
import re
import subprocess

from cutlass import Float32, Uint32
from cutlass.cute.runtime import from_dlpack
import torch

from ...apis.k3_gc_moe import K3_SITU_CONTRACT
from ...gc.k3_shared import K3SharedWorkspace
from ...impls.sm100_fp8_fp4_gc_moe_k3_shared import K3GCMoERoleKernel
from ...runtime.gc_moe import RoleBinary
from ...runtime.jit import (
    COMPILE_LOCK,
    PTXAS_OPTIONS,
    SYSTEM_PTXAS,
    compile_kernel,
    compiler_identity,
    ptx_target,
)
from ..heuristics.gc_moe import get_role_config
from .runtime_utils import make_tma_2d_desc
from .sm100_fp8_fp4_gc_moe import (
    _clean_cute_ptx,
    _device_compile_target,
    _entry_formals,
    _get_gc_moe_nvlink_barrier_timeout_secs,
    _rewrite_value_formal,
    _routed_weight_linger,
    prepare_role_binaries,
)


def _assemble_k3_role2(ptx, dump_dir, *, target, ptxas_options=PTXAS_OPTIONS):
    """按启动ABI编码18个routed值参数TensorMap和全局shared TensorMap指针。"""
    ptx = _clean_cute_ptx(ptx)
    if ptx_target(ptx) != target:
        raise RuntimeError('K3 role2 device target differs from compiler identity')
    formals = _entry_formals(ptx)
    expected = ('ptr', 'ptr', 'u32', 'u32', 'ptr', 'f32', 'f32') + ('ptr',) * 19
    if len(formals) != len(expected):
        raise RuntimeError(f'K3 role2 expected 26 CuTe formals, got {len(formals)}')
    for index, (line, kind) in enumerate(zip(formals, expected)):
        pattern = (rf'\.align\s+8\s+\.b8\s+\S+_param_{index}\[8\]'
                   if kind == 'ptr' else rf'\.{kind}\s+\S+_param_{index}')
        if re.search(pattern + r',?\s*$', line) is None:
            raise RuntimeError(f'K3 role2 parameter {index} changed: {line}')
    ptx = _rewrite_value_formal(ptx, 1, 592, 8)
    map_ordinals = (4, *range(7, 24))
    raw_regs = set()
    for ordinal in map_ordinals:
        ptx = _rewrite_value_formal(ptx, ordinal, 128, 128, raw_param_address=True)
        found = re.findall(r'(?m)^\s*mov\.b64\s+(%rd[0-9]+),\s*'
                           rf'[A-Za-z0-9_.$]+_param_{ordinal};\s*$', ptx)
        if len(found) != 1:
            raise RuntimeError(f'K3 role2 routed map parameter {ordinal} address changed')
        raw_regs.add(found[0])
    decl = re.compile(r'(?m)^([ \t]*\.reg\s+\.b64\s+%rd<)([0-9]+)(>;[ \t]*)$')
    match = list(decl.finditer(ptx))
    if len(match) != 1:
        raise RuntimeError('K3 role2 PTX register declaration changed')
    next_reg = int(match[0].group(2))
    converted_prefetches = 0
    converted_stores = 0

    def convert(match):
        nonlocal next_reg, converted_prefetches, converted_stores
        prefix, register, suffix = match.groups()
        if register not in raw_regs:
            return match.group(0)
        generic = f'%rd{next_reg}'
        next_reg += 1
        if 'prefetch.tensormap' in prefix:
            converted_prefetches += 1
        else:
            converted_stores += 1
        indent = re.match(r'[ \t]*', prefix).group(0)
        return f'{indent}cvta.param.u64 {generic}, {register};\n{prefix}{generic}{suffix}'

    prefetch = re.compile(r'(?m)^([ \t]*prefetch\.tensormap\s+\[)(%rd[0-9]+)(\];[ \t]*)$')
    store = re.compile(r'(?m)^([ \t]*cp\.async\.bulk\.tensor\.2d\.global\.shared[^\n]*?\[)'
                       r'(%rd[0-9]+)(,\s*\{[^\n]+)$')
    ptx = prefetch.sub(convert, ptx)
    ptx = store.sub(convert, ptx)
    if converted_prefetches != 18:
        raise RuntimeError(f'K3 role2 routed-map prefetch count changed: {converted_prefetches}')
    if converted_stores < 1:
        raise RuntimeError('K3 role2 routed output-map conversion was not found')
    ptx = decl.sub(lambda match: match.group(1) + str(next_reg) + match.group(3), ptx, count=1)
    # 启动ABI：6个标量/值对象、18个routed TensorMap、shared状态和TensorMap指针。
    formals = _entry_formals(ptx)
    order = (0, 1, 2, 3, 5, 6, 4, *range(7, 26))
    replacement = ',\n'.join(formals[index].rstrip().rstrip(',') for index in order)
    ptx = ptx.replace('\n'.join(formals), replacement, 1)
    return _assemble_k3_checked(ptx, dump_dir, target=target, ptxas_options=ptxas_options)


def _assemble_k3_checked(ptx, dump_dir, *, target, ptxas_options):
    # setmaxnreg在CTA初始寄存器池内重分配；ptxas的分配必须覆盖各warp预算之和。
    version_result = subprocess.run([SYSTEM_PTXAS, '--version'], capture_output=True, text=True)
    version = (version_result.stdout + version_result.stderr).strip()
    release = re.search(r'release\s+([0-9]+)\.([0-9]+)', version)
    if (version_result.returncode or release is None or
            tuple(map(int, release.groups())) < (13, 2)):
        raise RuntimeError('K3 shared requires CUDA 13.2 or newer ptxas')
    source = Path(dump_dir) / 'k3_role2_value_params.ptx'
    output = Path(dump_dir) / 'k3_role2_value_params.cubin'
    source.write_text(ptx)
    options = tuple(ptxas_options)
    if '-v' not in options and '--verbose' not in options:
        options += ('-v',)
    result = subprocess.run([SYSTEM_PTXAS, f'-arch={target}', *options,
                             '--warn-on-spills', '--warn-on-local-memory-usage',
                             '-o', str(output), str(source)], capture_output=True, text=True)
    diagnostics = result.stdout + result.stderr
    (Path(dump_dir) / 'k3-role2-ptxas.log').write_text(diagnostics)
    if result.returncode:
        raise RuntimeError('K3 shared ptxas failed: ' + diagnostics)
    register_counts = re.findall(r'Used\s+([0-9]+)\s+registers', diagnostics)
    if len(register_counts) != 1:
        raise RuntimeError('cannot establish K3 shared CTA register allocation: ' + diagnostics)
    registers = int(register_counts[0])
    # 每warp按256个寄存器取整，20warp的总分配须在SM的65536个寄存器内。
    allocated = ((registers + 7) // 8 * 8) * 640
    if registers * 640 < 61440 or allocated > 65536:
        raise RuntimeError(f'K3 shared register allocation {registers}/thread '
                           'cannot support the 61440-register warp budgets safely')
    return ptx, output.read_bytes(), version


def make_k3_shared_role2(config, sym_buffer, context, layout, has_stats=False, *, shared_rows):
    # shared按连续token tile配置两级输入缓冲，SMEM占用决定routed使用9级或10级。
    return K3GCMoERoleKernel(
        2, context.dispatch_sms, context.compute_sms, config.block_m,
        config.hidden, config.intermediate, config.tokens, sym_buffer.group.size(),
        config.local_experts, 0, sym_buffer.num_max_tokens_per_rank,
        finalize_output=False, ring_tokens=sym_buffer.layout.ring_tokens,
        sf_ring_tokens=sym_buffer.layout.sf_ring_tokens, topk=sym_buffer.num_topk,
        shared_weight_gran_k=config.shared_weight_gran_k,
        activation_clamp=inf,
        fast_math=config.fast_math, has_stats=has_stats,
        nvlink_timeout_secs=_get_gc_moe_nvlink_barrier_timeout_secs(),
        routed_weight_linger=_routed_weight_linger(config, sym_buffer.num_topk),
        role_config=get_role_config(
            2, config.block_m, sym_buffer.num_experts, config.hidden,
            local_experts=config.local_experts),
        k3_layout=layout, shared_rows=shared_rows)


@dataclass
class K3PreparedCall:
    context: object
    binaries: tuple
    arguments: tuple
    workspace: K3SharedWorkspace
    keepalive: tuple
    shared_rows: int

    @property
    def shared_partial(self):
        # 返回与输入行序一致的[n,H]视图，与workspace.partial共享存储。
        return self.workspace.partial[:self.shared_rows]

    def launch(self, stream=None):
        self.context.launch(self.binaries, self.arguments, stream=stream)


_K3_BINARY_CACHE = {}


def prepare_k3_shared_binaries(config, y, l1_weights, l2_weights, sym_buffer,
                               gathered_x, shared_w13, shared_w2, workspace, *,
                               stats, context, shared_w13_interleaved,
                               shared_rows):
    """构建四张shared TensorMap、缓存role2二进制并持有启动参数及其底层存储。"""
    device = sym_buffer.buffer.device
    intermediate = workspace.layout.intermediate
    binaries, arguments = prepare_role_binaries(
        config, y, l1_weights, l2_weights, sym_buffer, None, None, stats, context)
    # W13按相邻gate/up行排列；已交错时复用输入，否则只重排行，保持BF16位值。
    w13 = shared_w13 if shared_w13_interleaved else torch.stack(
        (shared_w13[:intermediate], shared_w13[intermediate:]), dim=1).reshape(
            2 * intermediate, workspace.layout.hidden)
    token_tile = min(workspace.layout.max_token_tile, (shared_rows + 15) // 16 * 16)
    cta_rows = token_tile // 2
    # 输入和ACT的bound都是实际n；最后一个tile的越界行由TMA填零。
    activations = workspace.activations[:shared_rows]
    tensors = (gathered_x, w13, activations, shared_w2)
    payloads = []
    for index, tensor in enumerate(tensors):
        rows, inner = tensor.shape
        outer_tile = cta_rows if index in (0, 2) else 64
        spec = make_tma_2d_desc(tensor, inner, rows, 64, outer_tile, inner, 128)
        encoded = spec.encode()
        payloads.append(ctypes.string_at(encoded.getPtr(), 128))
    maps = torch.tensor(list(b''.join(payloads)), dtype=torch.uint8, device=device).view(4, 128)
    kernel = make_k3_shared_role2(
        config, sym_buffer, context, workspace.layout, stats is not None, shared_rows=shared_rows)
    target = _device_compile_target(device)
    identity = compiler_identity(target, K3GCMoERoleKernel.__call__,
                                  Path(__file__).resolve().parents[2],
                                  f'gc_moe_k3_situ_shared_flat_m128_s2_n{shared_rows}_i{intermediate}_role2_v3')
    key = (K3_SITU_CONTRACT, identity, config, context.dispatch_sms, context.compute_sms,
           sym_buffer.num_max_tokens_per_rank, sym_buffer.layout.ring_tokens,
           sym_buffer.layout.sf_ring_tokens, workspace.layout, shared_rows, stats is not None,
           kernel.routed_weight_linger, kernel.nvlink_timeout_secs)
    with COMPILE_LOCK:
        binary = _K3_BINARY_CACHE.get(key)
        if binary is None:
            signal_view = from_dlpack(sym_buffer.buffer.view(torch.int32))
            peer_view = from_dlpack(sym_buffer.buffer.view(torch.int64).flatten()[:74])
            storage = sym_buffer.buffer.view(torch.uint8).flatten()
            routed_maps = tuple(from_dlpack(storage[i * 128:(i + 1) * 128]) for i in range(18))
            compile_args = (signal_view, peer_view, Uint32(0), Uint32(0), routed_maps[0],
                            None, None, Float32(0.0), Float32(1.0), *routed_maps[1:],
                            from_dlpack(workspace.buffer.view(torch.uint8)), from_dlpack(maps))
            cubin = compile_kernel(
                kernel, compile_args, identity,
                lambda ptx, dump_dir: _assemble_k3_role2(
                    ptx, dump_dir, target=target, ptxas_options=identity.ptxas_options))
            binary = RoleBinary(2, cubin, kernel.dynamic_smem, 640, context.compute_sms)
            _K3_BINARY_CACHE[key] = binary
    combined_binaries = (binaries[0], binary, binaries[2])
    shared_args = (ctypes.c_void_p(workspace.buffer.data_ptr()),
                   ctypes.c_void_p(maps.data_ptr()))
    combined_args = (arguments[0], (*arguments[1], *shared_args), arguments[2])
    result = K3PreparedCall(context, combined_binaries, combined_args, workspace,
                           (sym_buffer, gathered_x, shared_w13, shared_w2, w13, maps, y,
                            l1_weights, l2_weights, stats, kernel, activations),
                           shared_rows=shared_rows)
    workspace._prepared.append(result)
    return result
