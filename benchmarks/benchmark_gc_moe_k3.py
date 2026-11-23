"""Kimi K3 decode graph timing against the selected Mega MoE K3 baseline."""
from __future__ import annotations
import os
import sys
from pathlib import Path
import time
from typing import Optional, Callable, Dict
import numpy as np
import torch
import torch.distributed as dist
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'tests'))
os.environ['DG_PRINT_CONFIGS']='0'

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
        "median_ms": float(np.median(arr)),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "windows_ms": arr.tolist(),
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
        "median_ms": float(np.median(arr)),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "windows_ms": arr.tolist(),
    }


def workload(a, states, routes):
    local=dict(routes=routes.cpu().tolist(),tokens=states.shape[0])
    rows=[None]*a.world;dist.all_gather_object(rows,local)
    active=set();assign=remote=0
    for rank,row in enumerate(rows):
        for token in row['routes']:
            for e in token:
                if 0<=e<a.num_experts:
                    assign+=1;active.add(e);remote+=e//(a.num_experts//a.world)!=rank
    n = sum(row['tokens'] for row in rows)
    nz = sum(row['tokens'] > 0 for row in rows)
    return dict(global_tokens=n,routed_assignments=assign,remote_assignments=remote,
        unique_active_experts=len(active),shared_active_ranks=nz,
        effective_matmul_flops=6*assign*a.hidden*a.intermediate_hidden+ 6*n*a.shared_hidden*a.shared_intermediate_hidden + 2*n*a.shared_hidden*(a.num_experts+2*a.hidden),
        hbm_weight_scale_working_set_bytes=len(active)*51*a.hidden*a.intermediate_hidden//32+ nz*(6*a.shared_hidden*a.shared_intermediate_hidden + 2*a.hidden*(a.shared_hidden + 1) + 2*(a.num_experts+a.hidden)*a.shared_hidden),
        nvlink_useful_payload_bytes=remote*(3*a.hidden+a.hidden//32))


def rates(w,us,world):
    agg=dict(effective_tflops=w['effective_matmul_flops']/(us*1e6),
        hbm_weight_scale_working_set_gbps=w['hbm_weight_scale_working_set_bytes']/(us*1e3),
        nvlink_useful_payload_gbps=w['nvlink_useful_payload_bytes']/(us*1e3))
    return dict(denominator_slowest_rank_median_us=us,aggregate=agg,
                per_gpu_average={k:v/world for k,v in agg.items()})


def run():
    import test_gc_moe_k3 as test

    a = test.arguments()
    test.initialize(a)
    pairs, references, shared_weights = test.build_weights(a)
    del references
    ops = test.Operators(a, pairs, shared_weights)
    results = []
    operators = (a.baseline_key, "gc_moe_k3")
    for index, vector in enumerate(a.vectors):
        if len(set(vector)) != 1:
            raise ValueError("decode benchmark requires symmetric tokens")
        t = vector[a.rank]
        data = test.build_tokens(a, t)
        prefix = ops.prefix(t)
        if a.baseline == "sglang":
            ops.baseline.global_num_tokens = sum(vector)
        benchmark = (_bench_cuda_graph_ms if 0 < t <= a.cuda_graph_max_tokens_per_rank
                     else _bench_gpu_time_ms)
        work = workload(a, data, ops.reference_input(data)[2])
        stats = {}
        for j in (index % 2, 1 - index % 2):
            out = ops.output(j, t)
            ops.warm_graph(j, out, data, prefix)
            torch.cuda.synchronize()
            dist.barrier()
            timing = benchmark(
                lambda: ops.call(j, out, data, prefix), warmup=a.warmup,
                iters=a.iters, inner_iters=a.inner_iters, device=a.device,
                barrier_fn=dist.barrier, inter_iter_sleep_s=a.inter_iter_sleep_s,
            )
            gathered = [None] * a.world
            dist.all_gather_object(gathered, timing)
            medians = [x["median_ms"] * 1000 for x in gathered]
            timing.update(per_rank_median_us=medians, slowest_rank_median_us=max(medians))
            stats[operators[j]] = timing

        baseline, gc = (stats[name]["slowest_rank_median_us"] for name in operators)
        results.append(dict(
            num_tokens_per_rank=t, num_tokens_global=sum(vector), prefix=True,
            **{f"{a.baseline_key}_us": baseline},
            gc_moe_k3_us=gc, speedup=baseline / gc,
            workload=work, operator_stats=stats,
            throughput={a.baseline_key: rates(work, baseline, a.world),
                        "gc_moe_k3": rates(work, gc, a.world)},
            timing="cuda_graph" if benchmark is _bench_cuda_graph_ms else "eager",
        ))
        if a.rank == 0:
            print(f"tokens_per_rank={t:4d} {test.BASELINE_NAMES[a.baseline]}={baseline:.3f}us "
                  f"GC MoE K3={gc:.3f}us", flush=True)
    report = test.evidence(a, ops)
    report.update(
        results=results,
        timing_config=dict(
            warmup=a.warmup, iters=a.iters, inner_iters=a.inner_iters,
            inputs="post-attention BF16 hidden states, including gate, TopK and down projection/quantization",
            output="post-RMSNorm projection plus shared and prefix",
            prefix_mode="with", denominator="maximum of per-rank median event latencies",
        ),
    )
    test.write_result(a, report)
    torch.cuda.synchronize()
    ops.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
