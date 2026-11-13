"""Compile packaged PPU sources into a cache on the target host."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


_SOURCE_DIR = Path(__file__).resolve().parents[2] / 'impls' / 'ppu'


def _compile(cache, digest, compiler, name, source, flags):
    import fcntl

    key = digest.copy()
    key.update(json.dumps((source, flags)).encode())
    output = cache / key.hexdigest()[:24] / f'{name}.so'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not output.is_file():
            log = output.with_suffix('.log')
            with TemporaryDirectory(dir=output.parent) as tmp, log.open('w') as stream:
                library = Path(tmp) / output.name
                command = [compiler, str(_SOURCE_DIR / source), *flags, '-shared', '-o', str(library)]
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode or not library.is_file():
                    raise RuntimeError(f'Failed to build {output}. See {log}')
                os.replace(library, output)
    return output


def _schedule_flags(spec, schedule, config, shared_expert):
    direct_chunked = schedule == 'm48_direct_chunked'
    tile_schedule = 'm64_grouped_chunked' if direct_chunked else schedule
    if schedule in ('m16_small_batch', 'm16_short_decode'):
        tile_schedule = 'm16_deep_ring'
    flags = list(spec['schedules'][tile_schedule])
    if direct_chunked:
        flags.remove('-DFROLE12_EARLY_METADATA')
        flags[flags.index('-DFTILE_ROWS=64')] = '-DFTILE_ROWS=48'
        flags.extend(('-DFROLE12_DIRECT_ROUTES', '-DFROLE12_DIRECT_METADATA',
                      '-DFROLE12_DIRECT_ONE_ROUTE_PER_WARP', '-DFROLE12_DISPATCH_PREFETCH=2',
                      '-DFROLE12_DIRECT_METADATA_PREFETCH', '-DFROLE12_EXPERT_PAYLOAD'))
        if config.topk >= 8 and not shared_expert:
            flags.append('-DFROLE12_POLL_NS=0')
    if tile_schedule in ('m64_grouped_chunked', 'm80_grouped'):
        flags.append('-DFROLE12_W1_REFILL_ADDRESS')
    if not shared_expert and tile_schedule == 'm16_deep_ring':
        flags[flags.index('-DFROLE12_DISPATCH_PREFETCH=6')] = '-DFROLE12_DISPATCH_PREFETCH=2'
        flags.extend(('-DFROLE12_DIRECT_ONE_ROUTE_PER_WARP', '-DFROLE12_DIRECT_METADATA_PREFETCH'))
        if 4 < config.topk <= 8:
            flags.append('-DFROLE12_TOKEN_FANOUT=4')
    if schedule == 'm16_small_batch':
        if shared_expert:
            flags.extend(('-DFROLE12_TOKEN_FANOUT=4', '-DFROLE12_DIRECT_ONE_ROUTE_PER_WARP'))
        else:
            flags.append('-DFROLE12_PREFETCH_FC2_FROM_FC1')
            flags[flags.index('-DFROLE12_ASF_CHUNK=8')] = '-DFROLE12_ASF_CHUNK=16'
    if schedule in ('m16_small_batch', 'm16_short_decode'):
        if shared_expert:
            flags.extend(('-DFROLE12_A_TSM_OFFSET', '-DFROLE12_B_TSM_OFFSET'))
        else:
            flags.append('-DFROLE12_PACK_FC2_STORES')
    if (tile_schedule == 'm16_deep_ring'
            or (schedule == 'm16_direct' and not shared_expert)
            or (schedule == 'm16_two_stage_receiver' and config.topk > 4)):
        # Give gate and up to separate issuers, leaving activation and scale
        # copies on their own warps.
        flags = [flag for flag in flags if flag not in ('-DFROLE12_SPLIT_W1_COPY', '-DFL1_GATE_PAIR')]
        flags.append('-DFROLE12_W1_SCALE_PAIR')
    if config.topk > 4 and schedule in ('m16_direct', 'm16_two_stage'):
        flags.append('-DFROLE12_PACK_W1_TSM_SCALES')
    if tile_schedule == 'm64_grouped_chunked' and config.topk < 8:
        flags.extend(('-DFROLE12_A_TSM_OFFSET', '-DFROLE12_B_TSM_OFFSET'))
        if direct_chunked:
            flags.append('-DFROLE12_W2_FULL_RING')
    if schedule == 'm16_two_stage_receiver':
        if shared_expert:
            flags[flags.index('-DFROLE12_POLL_NS=16')] = '-DFROLE12_POLL_NS=8'
            flags[flags.index('-DFROLE12_COMBINE=6')] = '-DFROLE12_COMBINE=4'
        else:
            flags.extend(('-DFROLE12_DIRECT_METADATA', '-DFROLE12_DIRECT_METADATA_PREFETCH'))
        if config.topk > 4:
            flags[flags.index('-DFROLE12_POLL_NS=8' if shared_expert else '-DFROLE12_POLL_NS=16')] = '-DFROLE12_POLL_NS=0'
            if shared_expert:
                flags.append('-DFROLE12_DIRECT_METADATA')
            else:
                flags.append('-DFROLE12_DIRECT_ONE_ROUTE_PER_WARP')
            flags.extend(('-DFROLE12_EXPERT_PAYLOAD', '-DFROLE12_PREFETCH_FC2_FROM_FC1'))
    if shared_expert and schedule == 'm80_grouped':
        # The local shared expert can end in a single m16 block. Pack the
        # FC2 stores for the full tiles without changing the BF16 wire format.
        flags.extend(('-DFROLE12_ADAPT_M16', '-DFROLE12_PACK_FC2_STORES', '-DFROLE12_PREFILL_W1_WEIGHTS'))
        flags.extend(('-DFROLE12_CAP_SHARED_FC1', '-DFROLE12_PREFETCH_FC2_FROM_FC1'))
        # Two full-width copies keep the shared FC1 prefetch on two issuers.
        flags.remove('-DFROLE12_SPLIT_W1_COPY')
        flags.remove('-DFROLE12_INTERLEAVE_W1_ISSUERS')
    if config.local_experts < 4 and '-DFROLE12_W1_GROUP_RECORDS' in flags:
        flags.remove('-DFROLE12_W1_GROUP_RECORDS')
    if direct_chunked and '-DFROLE12_W1_GROUP_RECORDS' in flags:
        flags.append('-DFROLE12_PREFETCH_W1_TAIL')
    if ((config.local_experts + int(shared_expert)) * config.intermediate * config.hidden >= 2**32
            and '-DFROLE12_W1_OFFSET32' in flags):
        flags.remove('-DFROLE12_W1_OFFSET32')

    # Dispatch prefetch cannot exceed the number of 512-byte FP4 tiles.
    prefetch = '-DFROLE12_DISPATCH_PREFETCH='
    for index, flag in enumerate(flags):
        if flag.startswith(prefetch):
            flags[index] = f'{prefetch}{min(int(flag[len(prefetch):]), config.hidden // 1024)}'
            break
    else:
        if config.hidden < 5120:
            flags.append(f'{prefetch}{config.hidden // 1024}')
    return flags


def build_ppu_libraries(config, token_bounds, *, cache_dir=None, shared_expert=False,
                        activation_clamp=False, fast_math=True, k_major_w2=False):
    """Return the communication, pipeline and selected role libraries."""
    if sys.platform != 'linux':
        raise RuntimeError('PPU GC MoE JIT requires the SAIL SDK on Linux')
    compiler = shutil.which('hgcc')
    if compiler is None:
        raise RuntimeError('hgcc is required to build PPU GC MoE')
    if cache_dir is None:
        default = Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache') / 'gc_moe'
        cache_dir = Path(os.environ.get('GC_MOE_JIT_CACHE_DIR') or default) / 'ppu'
    cache = Path(cache_dir).expanduser().resolve()

    # A cache can be shared by ranks and mounted on different target hosts.
    compiler_path = Path(compiler).resolve()
    compiler_stat = compiler_path.stat()
    toolchain = (sys.platform, platform.machine(), platform.libc_ver(),
                 str(compiler_path), compiler_stat.st_size, compiler_stat.st_mtime_ns)
    digest = hashlib.sha256(json.dumps(toolchain).encode())
    for source in sorted(_SOURCE_DIR.iterdir()):
        if source.suffix in ('.cu', '.cuh', '.cpp', '.h'):
            digest.update(source.name.encode())
            digest.update(hashlib.sha256(source.read_bytes()).digest())

    spec = json.loads((_SOURCE_DIR / 'build.json').read_text())
    common = spec['compiler_flags']
    jobs = {name: (source, [*common, *flags])
            for name, (source, flags) in spec['launchers'].items()}
    geometry_flags = [
        f'-DFCONST_H={config.hidden}', f'-DFCONST_I={config.intermediate}',
        f'-DFCONST_E={config.local_experts}', f'-DFCONST_RANKS={config.ranks}',
        f'-DFCONST_TOPK={config.topk}', '-DFROLE12_PRECLEARED_DISPATCH',
    ]
    schedules = {bound: config.schedule(bound, shared_expert=shared_expert) for bound in token_bounds}
    for schedule in dict.fromkeys(schedules.values()):
        flags = [*common, *spec['role_flags'], *_schedule_flags(spec, schedule, config, shared_expert),
                 *geometry_flags]
        if activation_clamp:
            flags.append('-DFROLE12_ACTIVATION_CLAMP')
        if k_major_w2:
            flags.append('-DFROLE12_K_MAJOR_W2')
        if not fast_math:
            flags.append('-DFROLE12_PRECISE_MATH')
        if shared_expert:
            flags.append('-DFROLE12_SHARED_EXPERT')
            if not any(flag.startswith('-DFROLE12_COMBINE=') for flag in flags):
                flags.append('-DFROLE12_COMBINE=4')
        jobs[schedule] = ('gc_moe.cu', flags)

    def build(name):
        source, flags = jobs[name]
        return _compile(cache, digest, compiler, name, source, flags)

    with ThreadPoolExecutor(max_workers=4) as pool:
        libraries = dict(zip(jobs, pool.map(build, jobs)))
    return (libraries['libgc_comm'], libraries['libpipeline_host'],
            {bound: libraries[schedule] for bound, schedule in schedules.items()})
