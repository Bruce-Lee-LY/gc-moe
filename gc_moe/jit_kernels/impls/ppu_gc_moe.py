"""Build and load the native Sail decode module for a model geometry."""

import ctypes
import fcntl
import functools
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch


class Launch(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in (
        'peers', 'grouped_x', 'grouped_sf', 'return_rows', 'tiles', 'tile_count',
        'mid', 'mid_sf', 'w1', 'w2', 'shared_x', 'shared_x_sf',
        'shared_w1', 'shared_w1_sf', 'shared_w2', 'shared_w2_sf', 'shared_act',
        'shared_mid', 'shared_mid_sf', 'shared_out', 'out', 'stats', 'grouped_weights',
        'silu_table', 'route_index', 'route_counts', 'finished_first', 'down_tiles',
    )] + [(name, ctypes.c_int) for name in (
        'rank', 'ranks', 'capacity', 'tokens', 'bound', 'units',
        'x_offset', 'sf_offset', 'id_offset', 'weight_offset', 'return_offset',
        'inbox_offset', 'inbox_ids_offset', 'aiu', 'm_tiles',
    )] + [(name, ctypes.c_float) for name in ('clamp', 'w1_inv_scale', 'w2_inv_scale')]


def build_ppu_library(cache_dir, source, *, defines=(), headers=()):
    sdk_compiler = Path(os.environ.get('PPU_SDK', '/usr/local/PPU_SDK')) / 'bin/hgcc'
    compiler_name = os.environ.get('GC_MOE_HGCC') or (str(sdk_compiler) if sdk_compiler.is_file() else 'hgcc')
    compiler = shutil.which(compiler_name)
    if not compiler:
        raise RuntimeError(f'cannot find executable {compiler_name!r}; activate Sail SDK 2.2.0 or set GC_MOE_HGCC')
    flags = ('-std=c++17', '-arch=ppu_15', '-O3', '-lineinfo', '-Xcompiler=-fPIC', '-shared')
    flags += defines
    compiler_path = Path(compiler).resolve()
    key = hashlib.sha256(repr((flags, compiler_path, compiler_path.stat().st_mtime_ns)).encode())
    for path in (source, *headers):
        key.update(path.name.encode())
        key.update(path.read_bytes())
    cache_dir.mkdir(parents=True, exist_ok=True)
    library = cache_dir / f'{source.stem}-{key.hexdigest()[:20]}.so'
    with library.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.exists():
            with tempfile.TemporaryDirectory(dir=cache_dir) as tmp:
                temporary = Path(tmp) / library.name
                subprocess.run([compiler, *flags, str(source), '-o', str(temporary)], check=True)
                temporary.replace(library)
    return ctypes.CDLL(str(library))


@functools.lru_cache(maxsize=None)
def load(cache_dir, hidden, intermediate, experts, topk, ranks=8, routed_precision='fp4_fp4'):
    source = Path(__file__).resolve().parents[2] / 'impls' / 'ppu' / 'decode.cu'
    defines = (f'-DGC_MOE_HIDDEN={hidden}', f'-DGC_MOE_INTERMEDIATE={intermediate}',
               f'-DGC_MOE_EXPERTS={experts}', f'-DGC_MOE_TOPK={topk}')
    if ranks > 8:
        defines += (f'-DGC_MOE_MAX_RANKS={ranks}',)
    if routed_precision == 'fp8_fp4':
        defines += ('-DGC_MOE_ROUTED_FP8_FP4=1',)
    lib = build_ppu_library(cache_dir, source, defines=defines, headers=sorted(source.parent.glob('*.cuh')))
    lib.gc_moe_decode.argtypes = [ctypes.POINTER(Launch)] + [ctypes.c_void_p] * 4
    lib.gc_moe_decode.restype = ctypes.c_int
    lib.gc_moe_init_silu.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.gc_moe_init_silu.restype = ctypes.c_int
    if ranks > 8:
        lib.gc_moe_probe_peers.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        lib.gc_moe_probe_peers.restype = ctypes.c_int
    return lib


def launch_gc_moe(config, y, workspace, stats) -> None:
    # Assign the ABI by field name so workspace and launch members can be
    # reviewed independently of their storage order.
    grouped_x, grouped_sf, return_rows, tiles, tile_count, mid, mid_sf = workspace._scratch
    x_offset, sf_offset, id_offset, weight_offset, return_offset, inbox_offset, inbox_ids_offset = workspace._offsets
    route_index, route_counts = workspace._route_index
    routed_w1, routed_w2 = config.routed_weights
    w1_inv_scale, w2_inv_scale = config.routed_inverse_scales
    args = Launch(
        peers=workspace._peers.data_ptr(),
        grouped_x=grouped_x.data_ptr(),
        grouped_sf=grouped_sf.data_ptr(),
        return_rows=return_rows.data_ptr(),
        tiles=tiles.data_ptr(),
        tile_count=tile_count.data_ptr(),
        mid=mid.data_ptr(),
        mid_sf=mid_sf.data_ptr(),
        w1=routed_w1.data_ptr(),
        w2=routed_w2.data_ptr(),
        out=y.data_ptr(),
        stats=stats.data_ptr() if stats is not None else None,
        grouped_weights=workspace._grouped_weights.data_ptr() if workspace._grouped_weights is not None else None,
        silu_table=workspace._silu_table.data_ptr(),
        route_index=route_index.data_ptr() if route_index is not None else None,
        route_counts=route_counts.data_ptr() if route_counts is not None else None,
        finished_first=workspace._finished_first.data_ptr(),
        down_tiles=workspace._down_tiles.data_ptr(),
        rank=workspace.rank_idx,
        ranks=workspace.layout.ranks,
        capacity=workspace.num_max_tokens_per_rank,
        tokens=config.tokens,
        bound=config.bound,
        units=workspace._units,
        x_offset=x_offset,
        sf_offset=sf_offset,
        id_offset=id_offset,
        weight_offset=weight_offset,
        return_offset=return_offset,
        inbox_offset=inbox_offset,
        inbox_ids_offset=inbox_ids_offset,
        aiu=workspace._aiu,
        m_tiles=workspace._m_tiles,
        clamp=config.clamp,
        w1_inv_scale=w1_inv_scale,
        w2_inv_scale=w2_inv_scale,
    )
    if config.shared_fp8_fp8_ptpc_weights is not None:
        (shared_w1, shared_w1_sf), (shared_w2, shared_w2_sf) = config.shared_fp8_fp8_ptpc_weights
        act, shared_mid, shared_mid_sf, shared_out = workspace._shared_scratch
        args.shared_x = workspace.shared_x.data_ptr()
        args.shared_x_sf = workspace.shared_x_sf.data_ptr()
        args.shared_w1 = shared_w1.data_ptr()
        args.shared_w1_sf = shared_w1_sf.data_ptr()
        args.shared_w2 = shared_w2.data_ptr()
        args.shared_w2_sf = shared_w2_sf.data_ptr()
        args.shared_act = act.data_ptr()
        args.shared_mid = shared_mid.data_ptr()
        args.shared_mid_sf = shared_mid_sf.data_ptr()
        args.shared_out = shared_out.data_ptr()

    stream = torch.cuda.current_stream(workspace._device)
    pipeline = (None, None, None)
    if workspace._se_pipeline is not None:
        side, start, done = workspace._se_pipeline
        pipeline = (side.cuda_stream, start.cuda_event, done.cuda_event)
    status = workspace._library.gc_moe_decode(ctypes.byref(args), stream.cuda_stream, *pipeline)
    if status:
        raise RuntimeError(f'PPU GC MoE launch failed with HGGC status {status}')
