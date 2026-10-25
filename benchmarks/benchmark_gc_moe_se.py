"""Multi-rank Mega MoE and GC MoE timing with fused shared experts."""
from __future__ import annotations
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'tests'))
os.environ['DG_PRINT_CONFIGS'] = '0'
import numpy as np
import torch
import torch.distributed as dist
import deep_gemm
import gc_moe
assert Path(gc_moe.__file__).resolve().is_relative_to(_ROOT), gc_moe.__file__
from test_gc_moe_se import (
    DEFAULT_TOKENS_PER_RANK, ROUTED_RECIPE, allocate_buffer, build_tokens, build_weights,
    per_token_cast_to_fp8, prepare_shared_sf, stage_inputs,
)

_L2_FLUSH_BUF: Optional[torch.Tensor] = None


def _flush_l2(device: torch.device) -> None:
    global _L2_FLUSH_BUF
    if _L2_FLUSH_BUF is None or _L2_FLUSH_BUF.device != device:
        l2_bytes = torch.cuda.get_device_properties(device).L2_cache_size
        _L2_FLUSH_BUF = torch.empty(max(l2_bytes * 2, 256 * 1024 * 1024),
                                    dtype=torch.int8, device=device)
    _L2_FLUSH_BUF.zero_()


def _bench_gpu_time_ms(
    fn: Callable[[], None],
    *,
    warmup: int,
    iters: int,
    inner_iters: int,
    device: torch.device,
    barrier_fn: Optional[Callable[[], None]] = None,
    inter_iter_sleep_s: float = 0.005,
) -> Dict[str, float]:
    """Event-window latency per eager call, including CPU submission gaps."""
    if inner_iters < 1:
        raise ValueError("inner_iters must be >= 1")

    for _ in range(warmup):
        if barrier_fn is not None:
            barrier_fn()
        _flush_l2(device)
        torch.cuda.synchronize(device)
        time.sleep(inter_iter_sleep_s)
        for _ in range(inner_iters):
            fn()
    torch.cuda.synchronize(device)

    times = []
    for _ in range(iters):
        if barrier_fn is not None:
            barrier_fn()
        _flush_l2(device)
        torch.cuda.synchronize(device)
        time.sleep(inter_iter_sleep_s)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner_iters):
            fn()
        end.record()
        torch.cuda.synchronize(device)
        times.append(start.elapsed_time(end) / inner_iters)

    arr = np.asarray(times, dtype=np.float64)
    return {
        "windows_ms": arr.tolist(),
        "median_ms": float(np.median(arr)),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def _bench_cuda_graph_ms(
    fn: Callable[[], None],
    *,
    warmup: int,
    iters: int,
    inner_iters: int,
    device: torch.device,
    barrier_fn: Optional[Callable[[], None]] = None,
    inter_iter_sleep_s: float = 0.005,
) -> Dict[str, float]:
    """Capture one call, time inner_iters replays, and divide the event time."""
    if inner_iters < 1:
        raise ValueError("inner_iters must be >= 1")

    # JIT-warm the kernel before capture; stream capture forbids any
    # host-side work (NVRTC compile, allocator growth, ...).
    for _ in range(3):
        fn()
    torch.cuda.synchronize(device)

    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(capture_stream):
        with torch.cuda.graph(g, stream=capture_stream):
            fn()
    torch.cuda.current_stream(device).wait_stream(capture_stream)
    torch.cuda.synchronize(device)

    for _ in range(warmup):
        if barrier_fn is not None:
            barrier_fn()
        _flush_l2(device)
        torch.cuda.synchronize(device)
        time.sleep(inter_iter_sleep_s)
        for _ in range(inner_iters):
            g.replay()
    torch.cuda.synchronize(device)

    times = []
    for _ in range(iters):
        if barrier_fn is not None:
            barrier_fn()
        _flush_l2(device)
        torch.cuda.synchronize(device)
        time.sleep(inter_iter_sleep_s)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner_iters):
            g.replay()
        end.record()
        torch.cuda.synchronize(device)
        times.append(start.elapsed_time(end) / inner_iters)

    # Release the captured graph before the next sweep point reuses the
    # backing buffers.
    del g
    arr = np.asarray(times, dtype=np.float64)
    return {
        "windows_ms": arr.tolist(),
        "median_ms": float(np.median(arr)),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hidden', type=int, default=6144)
    p.add_argument('--intermediate-hidden', type=int, default=2048)
    p.add_argument('--num-experts', type=int, default=256)
    p.add_argument('--num-topk', type=int, default=8)
    p.add_argument('--num-shared-experts', type=int, default=1)
    p.add_argument('--shared-weight-gran-k', type=int, choices=[32, 128], default=128)
    p.add_argument('--num-tokens-per-rank-list', default='')
    p.add_argument('--num-max-tokens-per-rank', type=int, default=0)
    p.add_argument('--activation', choices=['swiglu'], default='swiglu')
    p.add_argument('--activation-alpha', type=float, default=1.0)
    p.add_argument('--activation-clamp', type=float, default=10.0)
    p.add_argument('--fast-math', type=int, default=1)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--iters', type=int, default=30)
    p.add_argument('--inner-iters', type=int, default=100)
    p.add_argument('--cuda-graph-max-tokens-per-rank', type=int, default=256)
    p.add_argument('--output-json', default='results/_performance_results_gc_se.json')
    args = p.parse_args()
    assert args.num_shared_experts >= 1
    sweep = [int(t) for t in args.num_tokens_per_rank_list.split(',') if t.strip()] or DEFAULT_TOKENS_PER_RANK
    capacity = args.num_max_tokens_per_rank or max(sweep)
    assert min(sweep) >= 1 and max(sweep) <= capacity
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    assert torch.cuda.get_device_capability(device) in [(10, 0), (10, 3)]
    assert args.num_experts % world == 0, 'num_experts must be divisible by world_size'
    mr, gr, original_shared, ms, gs, recipe = build_weights(args, rank, world, device)
    del original_shared
    mb = allocate_buffer(args, capacity, 'base')
    gb = allocate_buffer(args, capacity, 'gc')
    torch.cuda.empty_cache()
    results = []
    for tokens in sweep:
        x, indices, weights = build_tokens(tokens, args.hidden, args.num_experts, args.num_topk, rank, device)
        x8, xs = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        msf, mbm = prepare_shared_sf(args, xs, mb, tokens, world, 'base')
        gsf, gbm = prepare_shared_sf(args, xs, gb, tokens, world, 'gc')
        stage_inputs(mb, x8, xs, indices, weights, msf)
        stage_inputs(gb, x8, xs, indices, weights, gsf)
        base_y = torch.empty((tokens, args.hidden), dtype=torch.bfloat16, device=device)
        gc_y = torch.empty_like(base_y)
        common = dict(recipe=ROUTED_RECIPE, shared_recipe=recipe, activation=args.activation,
                      activation_clamp=args.activation_clamp, activation_alpha=args.activation_alpha,
                      fast_math=bool(args.fast_math))

        def base_fn():
            deep_gemm.fp8_fp4_mega_moe(base_y, *mr, mb,
                shared_l1_weights=ms[0], shared_l2_weights=ms[1], **common)

        def candidate_fn():
            gc_moe.fp8_fp4_gc_moe(gc_y, *gr, gb,
                shared_l1_weights=gs[0], shared_l2_weights=gs[1], **common)

        graph_mode = args.cuda_graph_max_tokens_per_rank > 0 and tokens <= args.cuda_graph_max_tokens_per_rank
        bench = _bench_cuda_graph_ms if graph_mode else _bench_gpu_time_ms
        order = [('baseline', base_fn), ('gc', candidate_fn)]
        if len(results) % 2:
            order.reverse()
        operator_stats = {}
        for name, fn in order:
            measured = bench(fn, warmup=args.warmup, iters=args.iters, inner_iters=args.inner_iters,
                             device=device, barrier_fn=dist.barrier)
            measured.update(rank=rank, mode='cuda_graph' if graph_mode else 'eager')
            all_ranks = [None] * world
            dist.all_gather_object(all_ranks, measured)
            medians = [row['median_ms'] * 1000 for row in all_ranks]
            operator_stats[name] = dict(operator='deep_gemm.fp8_fp4_mega_moe' if name == 'baseline'
                                       else 'gc_moe.fp8_fp4_gc_moe', ranks=all_ranks,
                                       per_rank_median_us=medians, slowest_rank_median_us=max(medians))
        base = operator_stats['baseline']['slowest_rank_median_us']
        candidate = operator_stats['gc']['slowest_rank_median_us']
        results.append(dict(num_tokens_per_rank=tokens, num_tokens_global=tokens * world,
                            mode='cuda_graph' if graph_mode else 'eager', baseline_us=base, gc_us=candidate,
                            speedup=base / candidate, gc_block_m=gbm, mega_block_m=mbm,
                            operator_stats=operator_stats))
        if rank == 0:
            print(f'COMPARE tokens_per_rank={tokens} total_tokens={tokens * world} mode={results[-1]["mode"]} Mega_SE={base:.6f}us '
                  f'CuTe_GC_SE={candidate:.6f}us speedup={base / candidate:.6f}', flush=True)
        del base_y, gc_y, x, x8, xs, indices, weights, msf, gsf
        gc.collect()
        torch.cuda.empty_cache()
    if rank == 0:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(world_size=world, config=vars(args), requested_capacity=capacity,
                                      aligned_capacity={'baseline': mb.num_max_tokens_per_rank,
                                                        'candidate': gb.num_max_tokens_per_rank},
                                      routed_recipe=ROUTED_RECIPE,
                                      shared_recipe=recipe,
                                      shared_weight_recipe=(1, args.shared_weight_gran_k, args.shared_weight_gran_k),
                                      activation_gran_k=32, warmup=args.warmup,
                                      iters=args.iters, inner_iters=args.inner_iters,
                                      latency='maximum of per-rank medians over CUDA-event windows',
                                      operators={'baseline': 'deep_gemm.fp8_fp4_mega_moe', 'candidate': 'gc_moe.fp8_fp4_gc_moe'},
                                      modules={'baseline': deep_gemm.__file__, 'candidate': gc_moe.__file__},
                                      results=results), indent=2) + '\n')
        print('Wrote', str(out), flush=True)
    torch.cuda.synchronize(device)
    gb.destroy()
    mb.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
