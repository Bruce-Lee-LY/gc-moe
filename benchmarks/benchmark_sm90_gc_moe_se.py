"""SM90 complete-operator timing: gc_unfused, mega_unfused and gc_se.

Throughput shares the SM90 routed-traffic and transformed-weight model from
benchmark_sm90_gc_moe. Shared-expert math and weight storage are included, but
shared experts remain local and therefore add no NVLink payload.
"""
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
from test_sm90_gc_moe_se import (
    OPERATORS,
    DEFAULT_TOKENS_PER_RANK,
    ROUTED_RECIPE,
    SHARED_RECIPE,
    DENSE_SHARED_RECIPE,
    allocate_buffer,
    build_tokens,
    build_weights,
    prepare_inputs,
    stage_inputs,
    make_calls,
    dependency_audit,
)

# Support direct script execution and `python -m benchmarks...`.
if __package__:
    from .benchmark_sm90_gc_moe import (
        _collect_throughput_workload,
        _collect_weight_storage,
        _common_weight_storage,
        _print_throughput_metrics,
        _throughput_definitions,
        _throughput_rates,
        _weight_storage,
    )
else:
    from benchmark_sm90_gc_moe import (
        _collect_throughput_workload,
        _collect_weight_storage,
        _common_weight_storage,
        _print_throughput_metrics,
        _throughput_definitions,
        _throughput_rates,
        _weight_storage,
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
    if args.activation_alpha != 1.0:
        p.error('SM90 swiglu uses activation_alpha=1')
    assert args.num_shared_experts >= 1
    sweep = [int(t) for t in args.num_tokens_per_rank_list.split(',') if t.strip()] or DEFAULT_TOKENS_PER_RANK
    capacity = args.num_max_tokens_per_rank or max(sweep)
    assert min(sweep) >= 1 and max(sweep) <= capacity
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    assert torch.cuda.get_device_capability(device) == (9, 0)
    assert args.num_experts % world == 0, 'num_experts must be divisible by world_size'
    routed, mega_routed, shared, fused_shared, recipe, quantized = build_weights(
        args, rank, world, device, reference=True)
    local_experts = args.num_experts // world
    routed_storage = _common_weight_storage(
        _weight_storage(mega_routed, local_experts),
        _weight_storage(routed, local_experts),
        _weight_storage(quantized, local_experts),
    )
    shared_storage = _common_weight_storage(
        _weight_storage(shared),
        _weight_storage(fused_shared),
        _weight_storage(shared),
    )
    del quantized
    mb = allocate_buffer(args, capacity, False)
    gb = allocate_buffer(args, capacity, True)
    mega = allocate_buffer(args, capacity, False, mega=True)
    assert len({b.buffer.data_ptr() for b in (mb, mega, gb)}) == 3
    torch.cuda.empty_cache()
    weight_storage_by_rank = _collect_weight_storage(routed_storage, shared_storage)
    results = []
    for tokens in sweep:
        x, indices, weights = build_tokens(tokens, args.hidden, args.num_experts, args.num_topk, rank, device)
        routed_input, shared_input = prepare_inputs(x)
        stage_inputs(mb, routed_input, shared_input, indices, weights)
        stage_inputs(gb, routed_input, shared_input, indices, weights)
        stage_inputs(mega, routed_input, shared_input, indices, weights)
        outputs = {name: torch.empty((tokens, args.hidden), dtype=torch.bfloat16, device=device)
                   for name in OPERATORS}
        calls, standalone = make_calls(args, tokens, tokens, routed, mega_routed, shared, fused_shared,
                                      mb, mega, gb, shared_input, outputs)
        graph_mode = args.cuda_graph_max_tokens_per_rank > 0 and tokens <= args.cuda_graph_max_tokens_per_rank
        bench = _bench_cuda_graph_ms if graph_mode else _bench_gpu_time_ms
        order = list(calls.items())
        # Rotate order so each complete operator appears first across the sweep.
        offset = len(results) % len(order)
        order = order[offset:] + order[:offset]
        operator_stats = {}
        for name, fn in order:
            measured = bench(fn, warmup=args.warmup, iters=args.iters, inner_iters=args.inner_iters,
                             device=device, barrier_fn=dist.barrier)
            measured.update(rank=rank, mode='cuda_graph' if graph_mode else 'eager')
            all_ranks = [None] * world
            dist.all_gather_object(all_ranks, measured)
            medians = [row['median_ms'] * 1000 for row in all_ranks]
            operator_stats[name] = dict(operator=OPERATORS[name], ranks=all_ranks,
                                       per_rank_median_us=medians, slowest_rank_median_us=max(medians))
        workload = _collect_throughput_workload(
            indices, args.num_experts, args.hidden, args.intermediate_hidden,
            weight_storage_by_rank,
            args.intermediate_hidden * args.num_shared_experts,
        )
        throughput = {
            name: _throughput_rates(workload, measured['slowest_rank_median_us'])
            for name, measured in operator_stats.items()
        }
        latencies = {name + '_us': row['slowest_rank_median_us'] for name, row in operator_stats.items()}
        speedups = {name + '_over_gc_se': latencies[name + '_us'] / latencies['gc_se_us']
                    for name in ('gc_unfused', 'mega_unfused')}
        results.append(dict(num_tokens_per_rank=tokens, num_tokens_global=tokens * world,
                            mode='cuda_graph' if graph_mode else 'eager', **latencies, **speedups,
                            operator_stats=operator_stats, workload=workload,
                            throughput=throughput))
        if rank == 0:
            print(f'COMPARE tokens_per_rank={tokens} total_tokens={tokens * world} mode={results[-1]["mode"]} '
                  + ' '.join(f'{k}={v:.6f}' for k, v in {**latencies, **speedups}.items()), flush=True)
            _print_throughput_metrics(tokens, throughput)
        del calls, standalone, outputs, x, routed_input, shared_input, indices, weights, fn, order
        gc.collect()
        torch.cuda.empty_cache()
    rank_audits = [None] * world
    dist.all_gather_object(rank_audits, dict(rank=rank, **dependency_audit()))
    assert len({a['gc']['source_sha256'] for a in rank_audits}) == 1
    if rank == 0:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(world_size=world, config=vars(args), requested_capacity=capacity,
                                      aligned_capacity={'gc_unfused': mb.num_max_tokens_per_rank,
                                                        'mega_unfused': mega.num_max_tokens_per_rank,
                                                        'gc_se': gb.num_max_tokens_per_rank},
                                      routed_recipe=ROUTED_RECIPE,
                                      shared_recipe=recipe, shared_raw_recipe=recipe,
                                      shared_dense_recipe=DENSE_SHARED_RECIPE,
                                      shared_fused_recipe=SHARED_RECIPE,
                                      activation_gran_k=32, warmup=args.warmup,
                                      iters=args.iters, inner_iters=args.inner_iters,
                                      latency='maximum of per-rank medians over CUDA-event windows',
                                      audit=rank_audits[0], rank_audits=rank_audits,
                                      operators=OPERATORS,
                                      modules={'mega_unfused': deep_gemm.__file__, 'gc_unfused': gc_moe.__file__, 'gc_se': gc_moe.__file__},
                                      throughput_metrics=_throughput_definitions(),
                                      weight_storage_by_rank=weight_storage_by_rank,
                                      decode_mean_speedups={name + '_over_gc_se': float(np.mean([r[name + '_over_gc_se'] for r in results]))
                                                            for name in ('gc_unfused', 'mega_unfused')},
                                      results=results), indent=2) + '\n')
        print('Wrote', str(out), flush=True)
    torch.cuda.synchronize(device)
    gb.destroy()
    mb.destroy()
    mega.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
