"""
Multi-GPU decode benchmark comparing GC MoE with Mega MoE.

Decode captures one operator call in an outer CUDA graph and replays it
inner_iters times per event window. A zero graph cutoff selects eager timing.
The GC operator uses an internal three-node graph in both modes. Event times
include any CPU submission gaps; no launch-overhead subtraction is applied.
Inputs are staged once before timing. Both operators use the same warmup,
L2 flush, rank barriers and iteration counts.

Throughput uses actual routes and a common slowest-rank latency per operator.
HBM is a weight+scale working-set MODEL; NVLink is useful assignment payload,
not physical traffic or single-link bandwidth. See throughput_metrics in JSON.

Usage: /opt/conda310/bin/python -m torch.distributed.run --nproc_per_node=8 \
    benchmarks/benchmark_gc_moe.py
"""

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional

_REPO_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tests'))

import numpy as np
import torch
import torch.distributed as dist

# Host configuration printing inside an eager timing window adds I/O latency.
os.environ["DG_PRINT_CONFIGS"] = "0"
import deep_gemm
import gc_moe
assert Path(gc_moe.__file__).resolve().is_relative_to(Path(_REPO_ROOT).resolve()), gc_moe.__file__
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8


DEFAULT_TOKENS_PER_RANK: List[int] = [
    1, 2, 4, 8, 16, 32, 64, 128, 256,
]
FP4_BLOCK = 32

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


def _weight_storage(weights, local_experts=None):
    """Read tensor metadata only; count both layers' numel() * element_size().

    Routed payloads and scales have an equal-size leading expert axis in both
    transforms. Check that each expert occupies its own dense storage slab
    before dividing bytes. Shared tensors contain all shared experts together.
    Tensor-visible packing/padding is included; allocator slack is excluded.
    """
    if len(weights) != 2 or any(len(pair) != 2 for pair in weights):
        raise ValueError("expected two (payload, scale) weight pairs")
    layers = []
    for pair in weights:
        layer = {}
        for name, tensor in zip(("payload", "scale"), pair):
            shape, strides = list(tensor.shape), list(tensor.stride())
            numel, itemsize = tensor.numel(), tensor.element_size()
            if local_experts is not None:
                if local_experts <= 0 or not shape or shape[0] != local_experts:
                    raise ValueError("routed weights need a leading local-expert axis")
                slab = numel // local_experts
                dense_span = 1
                for stride, size in sorted(zip(strides[1:], shape[1:])):
                    if size > 1:
                        if stride != dense_span:
                            raise ValueError("expert storage must be dense and nonoverlapping")
                        dense_span *= size
                if numel % local_experts or dense_span != slab or strides[0] != slab:
                    raise ValueError("expert storage slabs must have equal size")
            layer[name] = dict(shape=shape, stride=strides, dtype=str(tensor.dtype),
                               numel=numel, element_size=itemsize, bytes=numel * itemsize)
        layers.append(layer)
    total = sum(t["bytes"] for layer in layers for t in layer.values())
    return dict(layers=layers, total_bytes=total,
                bytes_per_expert=total // local_experts if local_experts is not None else None)


def _common_weight_storage(baseline, candidate, canonical):
    """Use one common numerator, falling back to original quantized inputs.

    All arguments are plain metadata dictionaries from _weight_storage.
    The current Mega/GC permutations preserve component bytes; record the
    runtime check rather than assuming scale dtype or physical layout.
    """
    equal = all(baseline["layers"][i][kind]["bytes"] == candidate["layers"][i][kind]["bytes"]
                for i in range(2) for kind in ("payload", "scale"))
    selected = baseline if equal else canonical
    return dict(baseline=baseline, gc=candidate, canonical_quantized_input=canonical,
                transformed_component_bytes_equal=equal,
                numerator_basis="matching_transformed_tensor_storage" if equal else "canonical_quantized_input",
                effective_total_bytes=selected["total_bytes"],
                effective_bytes_per_expert=selected["bytes_per_expert"])


def _local_route_summary(routes, num_experts):
    """CPU histogram of actual top-k assignments; never deduplicate routes."""
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    histogram = [0] * num_experts
    for token_routes in routes:
        for expert in token_routes:
            if not isinstance(expert, int) or not 0 <= expert < num_experts:
                raise ValueError("route expert must be in [0, num_experts)")
            histogram[expert] += 1
    return dict(local_tokens=len(routes), expert_route_counts=histogram)


def _throughput_workload(route_summaries, weight_storage_by_rank, hidden, intermediate, shared_intermediate=0):
    """Pure CPU counts; ranks own contiguous equal-size blocks of experts.

    HBM is a weight+scale working-set MODEL, not physical HBM traffic. Count
    each hit expert once on its owner and shared weights once per nonempty
    source rank. NVLink counts every remote assignment, including multiple
    experts on the same destination for one token. Shared work stays local.
    """
    world = len(route_summaries)
    if len(weight_storage_by_rank) != world:
        raise ValueError("weight storage must match route summary ranks")
    if not world or hidden <= 0 or hidden % 128 or intermediate <= 0 or shared_intermediate < 0:
        raise ValueError("need ranks, positive H divisible by 128, I > 0 and shared I >= 0")
    experts = len(route_summaries[0]["expert_route_counts"])
    if not experts or experts % world:
        raise ValueError("experts must divide evenly across ranks")
    local_experts = experts // world
    histograms = [row["expert_route_counts"] for row in route_summaries]
    for row, hist in zip(route_summaries, histograms):
        if (len(hist) != experts or any(not isinstance(n, int) or n < 0 for n in hist)
                or row["local_tokens"] < 0):
            raise ValueError("invalid route counts")
    incoming = [sum(hist[e] for hist in histograms) for e in range(experts)]
    per_rank = []
    for rank, row in enumerate(route_summaries):
        storage = weight_storage_by_rank[rank]
        start, end = rank * local_experts, (rank + 1) * local_experts
        active = [e for e in range(start, end) if incoming[e]]
        received = sum(incoming[start:end])
        source_remote = sum(histograms[rank]) - sum(histograms[rank][start:end])
        received_remote = received - sum(histograms[rank][start:end])
        tokens = row["local_tokens"]
        routed_flops = 6 * received * hidden * intermediate
        shared_flops = 6 * tokens * hidden * shared_intermediate
        routed_bytes = len(active) * storage["routed_weight_storage"]["effective_bytes_per_expert"]
        shared_bytes = (storage["shared_weight_storage"]["effective_total_bytes"]
                        if shared_intermediate and tokens else 0)
        per_rank.append(dict(
            rank=rank, local_tokens=tokens, incoming_route_assignments=received,
            owned_expert_route_counts=incoming[start:end], unique_active_experts=active,
            source_remote_assignments=source_remote, received_remote_assignments=received_remote,
            routed_matmul_flops=routed_flops, shared_matmul_flops=shared_flops,
            effective_matmul_flops=routed_flops + shared_flops,
            routed_weight_scale_working_set_bytes=routed_bytes,
            shared_weight_scale_working_set_bytes=shared_bytes,
            hbm_weight_scale_working_set_bytes=routed_bytes + shared_bytes,
            nvlink_injected_tx_payload_bytes=source_remote * (hidden + hidden // 32) + received_remote * (2 * hidden),
        ))
    remote = sum(row["source_remote_assignments"] for row in per_rank)
    return dict(
        world_size=world, global_tokens=sum(row["local_tokens"] for row in per_rank),
        routed_assignments=sum(incoming), remote_assignments=remote,
        unique_active_experts=sum(len(row["unique_active_experts"]) for row in per_rank),
        shared_intermediate=shared_intermediate,
        shared_active_ranks=sum(row["local_tokens"] > 0 for row in per_rank) if shared_intermediate else 0,
        routed_matmul_flops=sum(row["routed_matmul_flops"] for row in per_rank),
        shared_matmul_flops=sum(row["shared_matmul_flops"] for row in per_rank),
        effective_matmul_flops=sum(row["effective_matmul_flops"] for row in per_rank),
        hbm_weight_scale_working_set_bytes=sum(row["hbm_weight_scale_working_set_bytes"] for row in per_rank),
        nvlink_useful_payload_bytes=remote * (3 * hidden + hidden // 32),
        per_rank=per_rank,
    )


def _throughput_rates(workload, latency_us):
    """All rates use this operator's original slowest-rank median latency."""
    if not math.isfinite(latency_us) or latency_us <= 0:
        raise ValueError("throughput latency_us must be finite and > 0")
    def rates(row, payload_key):
        return dict(effective_tflops=row["effective_matmul_flops"] / (latency_us * 1e6),
                    hbm_weight_scale_working_set_gbps=row["hbm_weight_scale_working_set_bytes"] / (latency_us * 1e3),
                    nvlink_useful_payload_gbps=row[payload_key] / (latency_us * 1e3))
    aggregate = rates(workload, "nvlink_useful_payload_bytes")
    per_rank = []
    for row in workload["per_rank"]:
        rate = rates(row, "nvlink_injected_tx_payload_bytes")
        rate["nvlink_injected_tx_payload_gbps"] = rate.pop("nvlink_useful_payload_gbps")
        per_rank.append(dict(rank=row["rank"], **rate))
    return dict(denominator_slowest_rank_median_us=latency_us, aggregate=aggregate,
                per_gpu_average={key: value / workload["world_size"] for key, value in aggregate.items()},
                per_rank=per_rank)


def _collect_weight_storage(routed_storage, shared_storage=None):
    """Gather plain weight metadata once before the token sweep, in rank order."""
    local = dict(routed_weight_storage=routed_storage, shared_weight_storage=shared_storage)
    weight_storage_by_rank = [None] * dist.get_world_size()
    dist.all_gather_object(weight_storage_by_rank, local)
    return weight_storage_by_rank


def _collect_throughput_workload(indices, num_experts, hidden, intermediate,
                                 weight_storage_by_rank, shared_intermediate=0):
    """Gather routes after BOTH operator timings; reuse the CPU weight metadata."""
    local = _local_route_summary(indices.detach().cpu().tolist(), num_experts)
    route_summaries = [None] * dist.get_world_size()
    dist.all_gather_object(route_summaries, local)
    return _throughput_workload(route_summaries, weight_storage_by_rank,
                                hidden, intermediate, shared_intermediate)


def _print_throughput_metrics(tokens, throughput):
    for operator, rate in throughput.items():
        total, average = rate["aggregate"], rate["per_gpu_average"]
        print(f"METRICS tokens_per_rank={tokens} operator={operator} "
              f"slowest_rank_median_us={rate['denominator_slowest_rank_median_us']:.6f} "
              f"effective_TFLOPS_aggregate={total['effective_tflops']:.6f} "
              f"effective_TFLOPS_per_gpu_average={average['effective_tflops']:.6f} "
              f"HBM_weight_scale_working_set_MODEL_GBps_aggregate={total['hbm_weight_scale_working_set_gbps']:.6f} "
              f"HBM_weight_scale_working_set_MODEL_GBps_per_gpu_average={average['hbm_weight_scale_working_set_gbps']:.6f} "
              f"NVLink_useful_payload_GBps_aggregate={total['nvlink_useful_payload_gbps']:.6f} "
              f"NVLink_useful_payload_GBps_per_gpu_average={average['nvlink_useful_payload_gbps']:.6f}", flush=True)


def _throughput_definitions():
    return dict(
        denominator="Each operator's unchanged operator_stats[operator].slowest_rank_median_us; same global latency for all its rates/ranks",
        units=dict(flops="FLOP", storage="byte", latency="us", effective_tflops="10^12 FLOP/s",
                   hbm_weight_scale_working_set_gbps="decimal GB/s (10^9 byte/s)",
                   nvlink_useful_payload_gbps="decimal GB/s (10^9 byte/s)",
                   nvlink_injected_tx_payload_gbps="decimal GB/s (10^9 byte/s)"),
        rate_formulas=dict(effective_tflops="FLOPs / (latency_us * 1e6)", gbps="bytes / (latency_us * 1e3)"),
        aggregation="aggregate is N GPUs combined; per_gpu_average = aggregate / N; per_rank uses actual received routes/local shared tokens and the same global latency",
        effective_matmul_flops="routed: 6 * actual global route assignments * H * I; shared: 6 * global source tokens * H * shared_intermediate; shared_intermediate = I * num_shared_experts",
        hbm_weight_scale_working_set_model="Sum FP4 payload+scale tensor bytes for unique hit experts once on each owner; add all local shared FP8 payload+scale bytes once on each nonempty rank",
        weight_storage_by_rank="Top-level weight_storage_by_rank[rank] records routed_weight_storage for that expert owner and shared_weight_storage for that source rank (null without shared experts), once for the entire sweep; results[].workload.per_rank[].rank indexes this list. Each layer records actual transformed shape/stride/dtype/numel/element_size and numel * element_size bytes. Leading routed expert axis and equal dense slabs are checked. Tensor-visible packing/padding included; stride gaps/allocator slack excluded. Component bytes checked Mega vs GC; equal -> common transformed bytes, unequal -> common original quantized input bytes",
        nvlink_useful_payload="remote assignments * (H + H/32 + 2H): GC role1 pulls FP8 H and packed UE8M0 H/32 bytes, role2 returns BF16 2H bytes; H divisible by 128. No token-destination deduplication, no local/shared payload, no TX+RX double count",
        nvlink_per_rank_injected_tx="source_remote * (H + H/32) + received_remote * 2H; sum equals global one-way useful payload",
        comparison="Mega and GC share identical workload numerators; NVLink is the common GC assignment-payload model, not measured implementation-specific traffic",
        estimate_caveats=[
            "HBM is a weight+scale working-set effective bandwidth MODEL, not actual physical HBM traffic, cache misses or evidence of HBM saturation",
            "HBM excludes activations, outputs, intermediates, metadata, padding work, repeated CTA loads and cache-hit effects",
            "FLOPs exclude padding, SwiGLU, quantization/dequantization and repeated math",
            "NVLink excludes route metadata, routing weights, control, barriers and protocol overhead; aggregate/per-GPU average are not single-link bandwidth or measured peak",
            "Rates use end-to-end operator latency; no hardware profiling or additional tracing",
        ],
    )


def cast_grouped_weights_to_fp4(bf16_weights: torch.Tensor):
    L, N, K = bf16_weights.shape
    w = torch.empty((L, N, K // 2), device=bf16_weights.device, dtype=torch.int8)
    w_sf = torch.empty(
        (L, N, K // FP4_BLOCK), device=bf16_weights.device, dtype=torch.float,
    )
    for i in range(L):
        w[i], w_sf[i] = per_token_cast_to_fp4(
            bf16_weights[i], use_ue8m0=True, gran_k=FP4_BLOCK,
        )
    w_sf = deep_gemm.transform_sf_into_required_layout(
        w_sf, N, K, (1, FP4_BLOCK), L,
    )
    return w, w_sf


def build_local_weights(
    Eloc: int, intermediate: int, hidden: int, rank: int, device: torch.device,
):
    g = torch.Generator(device=device).manual_seed(7 + rank)
    w1_local = torch.randn(
        Eloc, 2 * intermediate, hidden,
        dtype=torch.bfloat16, device=device, generator=g,
    ) * 0.05
    w2_local = torch.randn(
        Eloc, hidden, intermediate,
        dtype=torch.bfloat16, device=device, generator=g,
    ) * 0.05
    return w1_local, w2_local


def build_tokens(
    T: int, hidden: int, num_experts: int, num_topk: int,
    rank: int, device: torch.device,
):
    g_x = torch.Generator(device=device).manual_seed(42 + rank * 7919)
    x = torch.randn(
        T, hidden, dtype=torch.bfloat16, device=device, generator=g_x,
    ) * 0.5
    g_r = torch.Generator(device=device).manual_seed(1000 + rank * 31)
    scores = torch.randn(
        T, num_experts, dtype=torch.float32, device=device, generator=g_r,
    )
    topk_w, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    topk_w = torch.softmax(topk_w, dim=-1)
    topk_idx = topk_idx.to(torch.int64)
    return x, topk_idx, topk_w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--intermediate-hidden", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-tokens-per-rank-list", type=str, default="")
    parser.add_argument("--num-max-tokens-per-rank", type=int, default=0)
    parser.add_argument("--activation", choices=("swiglu",), default="swiglu")
    parser.add_argument("--activation-alpha", type=float, default=1.0)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--fast-math", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument(
        "--inner-iters", type=int, default=100,
        help="Kernel calls per timed window; amortizes per-call launch "
             "overhead by pipelining launches behind GPU execution.",
    )
    parser.add_argument(
        "--cuda-graph-max-tokens-per-rank", type=int, default=256,
        help="Use CUDA Graph timing up to this many tokens per rank (default: 256). "
             "Set 0 to use eager timing for all points.",
    )
    parser.add_argument("--output-json", type=str, default="results/_performance_results_gc.json")
    args = parser.parse_args()

    sweep = DEFAULT_TOKENS_PER_RANK if not args.num_tokens_per_rank_list else \
        [int(t) for t in args.num_tokens_per_rank_list.split(",") if t.strip()]
    num_max_tokens_per_rank = args.num_max_tokens_per_rank or max(sweep)

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()

    if rank == 0:
        print(f"[benchmarks/benchmark_gc_moe] world_size={num_ranks} device={device}")

    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) not in [(10, 0), (10, 3)]:
        if rank == 0:
            print(f"Skipping: SM100/SM103 only, got SM{major}{minor}")
        dist.destroy_process_group()
        raise RuntimeError("SM100/SM103 hardware is required; no timings were measured")

    if args.num_experts % num_ranks != 0:
        if rank == 0:
            print(f"Skipping: num_experts={args.num_experts} not divisible by "
                  f"num_ranks={num_ranks}")
        dist.destroy_process_group()
        return

    Eloc = args.num_experts // num_ranks
    if rank == 0:
        print(f"Config: hidden={args.hidden} intermediate={args.intermediate_hidden} "
              f"experts={args.num_experts} topk={args.num_topk} "
              f"experts_per_rank={Eloc} activation={args.activation} "
              f"alpha={args.activation_alpha} clamp={args.activation_clamp}")
        print(f"num_max_tokens_per_rank={num_max_tokens_per_rank} sweep={sweep}")
        print(f"warmup={args.warmup} iters={args.iters} inner_iters={args.inner_iters}")

    w1_local, w2_local = build_local_weights(
        Eloc, args.intermediate_hidden, args.hidden, rank, device,
    )
    if rank == 0:
        print("Quantizing weights to FP4 ...")
    l1_packed = cast_grouped_weights_to_fp4(w1_local)
    l2_packed = cast_grouped_weights_to_fp4(w2_local)
    base_weights = deep_gemm.transform_weights_for_mega_moe(
        l1_packed, l2_packed, activation=args.activation,
    )
    candidate_weights = gc_moe.transform_weights_for_gc_moe(
        l1_packed, l2_packed, activation=args.activation,
    )
    routed_storage = _common_weight_storage(
        _weight_storage(base_weights, Eloc),
        _weight_storage(candidate_weights, Eloc),
        _weight_storage((l1_packed, l2_packed), Eloc),
    )
    base_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        dist.group.WORLD, args.num_experts, num_max_tokens_per_rank,
        args.num_topk, args.hidden, args.intermediate_hidden,
        activation=args.activation,
    )
    candidate_buffer = gc_moe.get_symm_buffer_for_gc_moe(
        dist.group.WORLD, args.num_experts, num_max_tokens_per_rank,
        args.num_topk, args.hidden, args.intermediate_hidden,
        activation=args.activation,
    )
    if rank == 0:
        print(f"Symmetric buffers: Mega {base_buffer.buffer.nbytes / 2 ** 30:.3f} GiB, GC {candidate_buffer.buffer.nbytes / 2 ** 30:.3f} GiB")

    weight_storage_by_rank = _collect_weight_storage(routed_storage)
    results = []
    for T in sweep:
        if T > num_max_tokens_per_rank:
            if rank == 0:
                print(f"  Skipping T={T} (> num_max_tokens_per_rank)")
            continue

        x_bf16, topk_idx, topk_w = build_tokens(
            T, args.hidden, args.num_experts, args.num_topk, rank, device,
        )
        x_fp8, x_sf = per_token_cast_to_fp8(
            x_bf16, use_ue8m0=True, gran_k=FP4_BLOCK, use_packed_ue8m0=True,
        )
        # Stage inputs into the symmetric buffer ONCE — the timed window
        # below contains only the fused MoE kernel call.
        for buffer in (base_buffer, candidate_buffer):
            buffer.x[:T].copy_(x_fp8)
            buffer.x_sf[:T].copy_(x_sf)
            buffer.topk_idx[:T].copy_(topk_idx)
            buffer.topk_weights[:T].copy_(topk_w)
        torch.cuda.synchronize(device)
        dist.barrier()

        outputs = {
            "baseline": torch.empty((T, args.hidden), dtype=torch.bfloat16, device=device),
            "gc": torch.empty((T, args.hidden), dtype=torch.bfloat16, device=device),
        }
        operators = {
            "baseline": (deep_gemm.fp8_fp4_mega_moe, base_weights, base_buffer),
            "gc": (gc_moe.fp8_fp4_gc_moe, candidate_weights, candidate_buffer),
        }
        def invoke(name):
            operator, weights, buffer = operators[name]
            operator(outputs[name], *weights, buffer,
                     recipe=(1, 1, FP4_BLOCK), activation=args.activation,
                     activation_clamp=args.activation_clamp, activation_alpha=args.activation_alpha,
                     fast_math=bool(args.fast_math))

        use_cuda_graph = (
            args.cuda_graph_max_tokens_per_rank > 0
            and T <= args.cuda_graph_max_tokens_per_rank
        )
        bench_fn = _bench_cuda_graph_ms if use_cuda_graph else _bench_gpu_time_ms
        operator_stats = {}
        # Equal inputs, capacity, warmup/iterations/L2 flush/timing on both paths.
        # Alternate the operator order across points to reduce drift bias.
        order = ["baseline", "gc"]
        if len(results) % 2:
            order.reverse()
        for name in order:
            measured = bench_fn(
                lambda name=name: invoke(name), warmup=args.warmup, iters=args.iters,
                inner_iters=args.inner_iters, device=device, barrier_fn=dist.barrier)
            measured["operator"] = ("deep_gemm.fp8_fp4_mega_moe" if name == "baseline"
                                    else "gc_moe.fp8_fp4_gc_moe")
            measured["rank"] = rank
            rank_windows = [None] * num_ranks
            dist.all_gather_object(rank_windows, measured)
            measured["ranks"] = rank_windows
            measured["per_rank_median_us"] = [float(row["median_ms"]) * 1e3 for row in rank_windows]
            measured["slowest_rank_median_us"] = max(measured["per_rank_median_us"])
            operator_stats[name] = measured
        workload = _collect_throughput_workload(
            topk_idx, args.num_experts, args.hidden, args.intermediate_hidden,
            weight_storage_by_rank,
        )
        throughput = {name: _throughput_rates(workload, measured["slowest_rank_median_us"])
                      for name, measured in operator_stats.items()}
        stats = operator_stats["gc"]
        stats["mode"] = "cuda_graph" if use_cuda_graph else "eager"
        baseline_us = operator_stats["baseline"]["slowest_rank_median_us"]
        gc_us = operator_stats["gc"]["slowest_rank_median_us"]
        speedup = baseline_us / gc_us

        if rank == 0:
            print(
                f"  T={T:6d}  mode={stats['mode']:10s}  "
                f"median={stats['median_ms']*1e3:8.1f} us  "
                f"min={stats['min_ms']*1e3:8.1f}  max={stats['max_ms']*1e3:8.1f}  "
                f"std={stats['std_ms']*1e3:5.1f}  "
                f"slowest_rank_median={gc_us:8.1f} us"
            )

        results.append({
            "num_tokens_per_rank": T,
            "num_tokens_global": T * num_ranks,
            "baseline_us": baseline_us,
            "gc_us": gc_us,
            "speedup": speedup,
            "operator_stats": operator_stats,
            "workload": workload,
            "throughput": throughput,
            "mode": stats["mode"],
            "median_us": stats["median_ms"] * 1e3,
            "mean_us": stats["mean_ms"] * 1e3,
            "std_us": stats["std_ms"] * 1e3,
            "min_us": stats["min_ms"] * 1e3,
            "max_us": stats["max_ms"] * 1e3,
            "slowest_rank_median_us": gc_us,
            "fastest_rank_median_us": min(stats["per_rank_median_us"]),
        })

        if rank == 0:
            print(f"COMPARE tokens_per_rank={T} total_tokens={T * num_ranks} mode={stats['mode']} mega={baseline_us:.3f}us gc={gc_us:.3f}us speedup={speedup:.5f}")
            _print_throughput_metrics(T, throughput)
        del outputs, x_bf16, x_fp8, x_sf, topk_idx, topk_w
        gc.collect()
        torch.cuda.empty_cache()

    if rank == 0:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump({
                "world_size": num_ranks,
                "throughput_metrics": _throughput_definitions(),
                "weight_storage_by_rank": weight_storage_by_rank,
                "config": {
                    "hidden": args.hidden,
                    "intermediate_hidden": args.intermediate_hidden,
                    "num_experts": args.num_experts,
                    "num_topk": args.num_topk,
                    "activation": args.activation,
                    "activation_alpha": args.activation_alpha,
                    "activation_clamp": args.activation_clamp,
                    "fast_math": bool(args.fast_math),
                    "num_max_tokens_per_rank": num_max_tokens_per_rank,
                },
                "requested_capacity": num_max_tokens_per_rank,
                "aligned_capacity": {"baseline": base_buffer.num_max_tokens_per_rank, "candidate": candidate_buffer.num_max_tokens_per_rank},
                "operators": {"baseline": "deep_gemm.fp8_fp4_mega_moe", "candidate": "gc_moe.fp8_fp4_gc_moe"},
                "modules": {"baseline": deep_gemm.__file__, "candidate": gc_moe.__file__},
                "warmup": args.warmup,
                "iters": args.iters,
                "inner_iters": args.inner_iters,
                "cuda_graph_max_tokens_per_rank": args.cuda_graph_max_tokens_per_rank,
                "results": results,
            }, f, indent=2)
        print(f"\nWrote {args.output_json}")

    candidate_buffer.destroy()
    base_buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
