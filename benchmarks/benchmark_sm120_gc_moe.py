"""
Multi-GPU decode benchmark comparing SM120 GC MoE with the flashinfer
sm120 MXFP4xMXFP8 MegaMoE baseline.

Decode captures one operator call in an outer CUDA graph and replays it
inner_iters times per event window. A zero graph cutoff selects eager timing.
The GC operator uses an internal three-node graph in both modes. Event times
include any CPU submission gaps; no launch-overhead subtraction is applied.
Inputs are pre-quantized (MXFP8 activations + packed E2M1 weights) and staged
once before timing on both arms. Both operators use the same warmup, L2
flush, rank barriers and iteration counts.

Usage: /opt/conda310/bin/python -m torch.distributed.run --nproc_per_node=8 \
    benchmarks/benchmark_sm120_gc_moe.py
"""

import argparse
import gc
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

_REPO_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import torch.distributed as dist

from benchmarks.benchmark_gc_moe import _collect_throughput_workload, _collect_weight_storage
from tests.test_sm120_gc_moe import (
    build_shared_weights, build_tokens, cast_grouped_weights_to_fp4,
    cast_tokens_to_fp8, make_shared_input_sf, pack_ue8m0_weight_scales,
)

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


def _make_baseline_layer(
    rank: int, world_size: int, *, experts: int, capacity: int,
    hidden: int, intermediate: int, topk: int, activation_clamp: float,
    fast_math: bool, routed,
):
    """FlashInfer SM120 MegaMoE with prequantized inputs."""
    from flashinfer.moe_ep import (
        BootstrapConfig, FleetParams, MegaConfig, MoEEpLayer, MoEWeightPack,
        Sm120_Mxfp4_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
    )

    weights = MoEWeightPack(*routed)
    layer = MoEEpLayer(
        bootstrap=BootstrapConfig(world_size=world_size, rank=rank),
        fleet_params=FleetParams(
            num_experts=experts,
            max_tokens_per_rank=capacity,
            token_hidden_size=hidden,
        ),
        weights=weights,
        backend=MegaConfig(
            megakernel=Sm120_Mxfp4_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
                intermediate_size=intermediate,
                top_k=topk,
                activation_clamp=activation_clamp,
                fast_math=fast_math,
            ),
            quantize_input=False,
            preprocess_weights=True,
        ),
    )
    return layer


def _make_cake_layer(
    rank: int, world_size: int, *, experts: int, capacity: int,
    hidden: int, intermediate: int, topk: int, device: torch.device,
    swiglu_limit: int, transport_table: str, routed, shared, shared_gran_k: int,
):
    """FlashInfer Cake with the same prequantized payloads as GC."""
    from flashinfer.moe_ep import (
        BootstrapConfig, FleetParams, MegaConfig, MoEEpLayer, MoEWeightPack,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake import (
        CakePrequantizedMoEWeightsWithSharedExpert,
        Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig,
    )

    if shared is None:
        weights = MoEWeightPack(*routed)
    else:
        (s_w13, scale1), (s_w2q, scale2) = shared
        # Cake consumes per-row/K32 UE8M0 even when checkpoint scales are
        # shared across N128/K128. Expand the exact powers of two only.
        block_n = 128 if shared_gran_k == 128 else 1
        s_w13_sf, s_w2_sf = (
            (scale.contiguous().view(torch.int32) >> 23).to(torch.uint8)
            .repeat_interleave(block_n, 0).repeat_interleave(shared_gran_k // 32, 1)
            for scale in (scale1, scale2)
        )
        weights = CakePrequantizedMoEWeightsWithSharedExpert(
            *routed,
            shared_w13=s_w13,
            shared_w2=s_w2q,
            shared_w13_scale=s_w13_sf.view(torch.float8_e8m0fnu),
            shared_w2_scale=s_w2_sf.view(torch.float8_e8m0fnu),
        )
    layer = MoEEpLayer(
        bootstrap=BootstrapConfig(world_size=world_size, rank=rank,
                                  device=device.index),
        fleet_params=FleetParams(
            num_experts=experts,
            max_tokens_per_rank=capacity,
            token_hidden_size=hidden,
        ),
        weights=weights,
        backend=MegaConfig(
            megakernel=Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig(
                intermediate_size=intermediate,
                top_k=topk,
                swiglu_limit=swiglu_limit,
                shared_expert="fp8_in_kernel" if shared is not None else None,
                transport_table=transport_table,
            ),
            quantize_input=False,
            preprocess_weights=True,
        ),
    )
    return layer


def main(*, shared=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--intermediate-hidden", type=int, default=2304)
    parser.add_argument("--num-experts", type=int, default=384)
    parser.add_argument("--num-topk", type=int, default=6)
    parser.add_argument("--num-tokens-per-rank-list", type=str, default="")
    parser.add_argument("--num-max-tokens-per-rank", type=int, default=0)
    parser.add_argument("--activation", choices=("swiglu",), default="swiglu")
    parser.add_argument("--activation-alpha", type=float, default=1.0)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--fast-math", type=int, default=1)
    parser.add_argument(
        "--arms", default="cake,gc",
        help="Comma-separated operator arms to time (baseline, cake, gc). "
             "The cake arm needs flashinfer>=0.7.3 and NCCL 2.30.7; the gc "
             "arm requires the SM120 GC MoE kernels to be functional.")
    parser.add_argument(
        "--cake-transport-table", default="all_lsa",
        help="Cake backend transport_table (all_gin, all_lsa, "
             "lsa_where_reachable, auto or a flat tuple string).")
    parser.add_argument(
        "--cake-shared-expert", "--shared-expert", action="store_true",
        help="Enable matching FP8 shared experts in both Cake and GC arms.")
    parser.add_argument('--num-shared-experts', type=int,
                        help='Number of fused shared experts (Cake supports one).')
    parser.add_argument('--shared-weight-gran-k', type=int, choices=(32, 128), default=32)
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
    parser.add_argument("--output-json", type=str,
                        default=("results/_performance_results_sm120_gc_se.json" if shared
                                 else "results/_performance_results_sm120_gc.json"))
    args = parser.parse_args()

    num_shared = args.num_shared_experts
    if num_shared is None:
        num_shared = int(shared or args.cake_shared_expert)
    if num_shared < 0 or (num_shared == 0 and (shared or args.cake_shared_expert)):
        parser.error('invalid --num-shared-experts')
    arms = list(dict.fromkeys(name.strip() for name in args.arms.split(",") if name.strip()))
    if not arms or any(name not in ("baseline", "cake", "gc") for name in arms):
        parser.error("--arms must be a subset of baseline,cake,gc")
    if num_shared and 'baseline' in arms:
        parser.error('the cutedsl baseline arm has no matching shared expert')
    if num_shared > 1 and 'cake' in arms:
        parser.error('use --arms gc to time more than one shared expert')
    if 'cake' in arms and (not args.fast_math or not args.activation_clamp.is_integer()):
        parser.error('Cake comparison requires fast math and an integer clamp')

    flashinfer = None
    if 'baseline' in arms or 'cake' in arms:
        import flashinfer
        from flashinfer.moe_ep import MoEEpTensors
        from flashinfer.moe_ep.kernel_src.sm120.split_cutedsl_megakernel import (
            select_graph_compile_bucket,
        )

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
        print(f"[benchmarks/benchmark_sm120_gc_moe] world_size={num_ranks} "
              f"device={device}")
        if flashinfer is not None:
            print(f"flashinfer {flashinfer.__version__}")

    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) != (12, 0):
        if rank == 0:
            print(f"Skipping: SM120 only, got SM{major}{minor}")
        dist.destroy_process_group()
        raise RuntimeError("SM120 hardware is required; no timings were measured")

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
        print(f"arms={arms}")

    if args.activation_alpha != 1.0:
        raise ValueError("SM120 swiglu uses activation_alpha=1")

    from deep_gemm.utils import per_block_cast_to_fp8

    caster_source = Path(inspect.getfile(per_block_cast_to_fp8))
    routed = tuple(cast_grouped_weights_to_fp4(w) for w in build_local_weights(
        Eloc, args.intermediate_hidden, args.hidden, rank, device))
    flashinfer_routed = None
    if flashinfer is not None:
        flashinfer_routed = tuple(w.view(torch.float4_e2m1fn_x2) for w, _ in routed) + tuple(
            sf.view(torch.float8_e8m0fnu) for _, sf in routed)
    shared = None
    if num_shared:
        gran = args.shared_weight_gran_k
        shared = tuple(per_block_cast_to_fp8(w, use_ue8m0=True, gran_k=gran) for w in build_shared_weights(
            num_shared * args.intermediate_hidden, args.hidden, device))
        if gran == 32:
            shared = tuple((payload, scale.repeat_interleave(gran, 0)) for payload, scale in shared)

    routed_storage = dict(effective_bytes_per_expert=sum(
        tensor.nbytes for pair in routed for tensor in pair) // Eloc)
    shared_storage = None
    if shared is not None:
        # Compare operators using the common per-row UE8M0 weight footprint.
        # Four K-group exponents occupy one packed int32.
        shared_storage = dict(effective_total_bytes=sum(
            payload.nbytes + payload.size(0) * ((payload.size(1) + 4 * gran - 1) // (4 * gran)) * 4
            for payload, _ in shared))
    weight_storage_by_rank = _collect_weight_storage(routed_storage, shared_storage)

    baseline_layer = None
    cake_layer = None
    gc_weights = gc_buffer = None
    if "baseline" in arms:
        if rank == 0:
            print("Building flashinfer MegaMoE layer ...")
        baseline_layer = _make_baseline_layer(
            rank, num_ranks, experts=args.num_experts,
            capacity=num_max_tokens_per_rank, hidden=args.hidden,
            intermediate=args.intermediate_hidden, topk=args.num_topk,
            activation_clamp=args.activation_clamp, fast_math=bool(args.fast_math),
            routed=flashinfer_routed)
    if "cake" in arms:
        if rank == 0:
            print(f"Building flashinfer cake MegaMoE layer "
                  f"(transport_table={args.cake_transport_table}, "
                  f"shared_expert={bool(num_shared)}) ...")
        cake_layer = _make_cake_layer(
            rank, num_ranks, experts=args.num_experts,
            capacity=num_max_tokens_per_rank, hidden=args.hidden,
            intermediate=args.intermediate_hidden, topk=args.num_topk,
            device=device, swiglu_limit=int(args.activation_clamp),
            transport_table=args.cake_transport_table,
            routed=flashinfer_routed, shared=shared, shared_gran_k=args.shared_weight_gran_k)
    shared_kwargs = {}
    if "gc" in arms:
        import gc_moe
        if rank == 0:
            print("Building GC MoE operator ...")
        packed_routed = tuple((payload, pack_ue8m0_weight_scales(codes))
                              for payload, codes in routed)
        gc_weights = gc_moe.transform_weights_for_gc_moe_sm120(*packed_routed)
        if shared is not None:
            shared_recipe = (1, 128, 128) if args.shared_weight_gran_k == 128 else (1, 1, 32)
            packed_shared = tuple(
                (payload, pack_ue8m0_weight_scales(
                    ((scale.contiguous().view(torch.int32) >> 23).to(torch.uint8))
                    .repeat_interleave(shared_recipe[1], dim=0)))
                for payload, scale in shared
            )
            shared_weights = gc_moe.transform_weights_for_gc_moe_sm120(
                *packed_shared, shared_recipe=shared_recipe)
            shared_kwargs = dict(shared_l1_weights=shared_weights[0],
                                 shared_l2_weights=shared_weights[1], shared_recipe=shared_recipe)
        gc_buffer = gc_moe.get_symm_buffer_for_gc_moe_sm120(
            dist.group.WORLD, args.num_experts, num_max_tokens_per_rank,
            args.num_topk, args.hidden, args.intermediate_hidden,
            activation=args.activation,
            num_shared_experts=num_shared,
        )
        if rank == 0:
            print(f"Symmetric buffer: GC "
                  f"{gc_buffer.buffer.nbytes / 2 ** 30:.3f} GiB")
    del routed, flashinfer_routed, shared
    torch.cuda.empty_cache()

    results = []
    for T in sweep:
        if T > num_max_tokens_per_rank:
            if rank == 0:
                print(f"  Skipping T={T} (> num_max_tokens_per_rank)")
            continue

        x_bf16, topk_idx, topk_w = build_tokens(
            T, args.hidden, args.num_experts, args.num_topk, rank, device,
        )
        x_fp8, x_sf = cast_tokens_to_fp8(x_bf16)
        # Flashinfer pre-quantized staging wants (T, H//32) per-K32 E8M0
        # bytes; DeepGEMM packs four K32 exponents per word.
        x_sf_bytes = x_sf.view(torch.uint8).view(T, args.hidden // 32)
        # Stage inputs ONCE — the timed window below contains only the fused
        # MoE kernel call.
        if baseline_layer is not None or cake_layer is not None:
            tensors = MoEEpTensors(
                hidden_states=x_fp8,
                topk_ids=topk_idx,
                topk_weights=topk_w,
                scales=x_sf_bytes.view(torch.float8_e8m0fnu),
            )
        if baseline_layer is not None:
            baseline_layer.stage_inputs(
                tensors,
                compile_tokens_per_rank=select_graph_compile_bucket(
                    T, num_max_tokens_per_rank))
        if cake_layer is not None:
            # Same bucketed staging as the cutedsl baseline: the bound keeps
            # compute_staged free of host synchronization so it captures.
            cake_layer.stage_inputs(
                tensors,
                compile_tokens_per_rank=select_graph_compile_bucket(
                    T, num_max_tokens_per_rank))
        if gc_buffer is not None:
            gc_buffer.x[:T].copy_(x_fp8)
            gc_buffer.x_sf[:T].copy_(x_sf)
            if num_shared:
                block_m = gc_moe.get_block_m_for_gc_moe(
                    num_ranks, args.num_experts, gc_buffer.num_max_tokens_per_rank, T, args.num_topk)
                make_shared_input_sf(x_sf, gc_buffer.shared_l1_acts_sf, block_m)
            gc_buffer.topk_idx[:T].copy_(topk_idx)
            gc_buffer.topk_weights[:T].copy_(topk_w)
        torch.cuda.synchronize(device)
        dist.barrier()

        outputs = {}
        if baseline_layer is not None:
            outputs["baseline"] = torch.empty(
                (T, args.hidden), dtype=torch.bfloat16, device=device)
        if cake_layer is not None:
            outputs["cake"] = torch.empty(
                (T, args.hidden), dtype=torch.bfloat16, device=device)
        if gc_buffer is not None:
            outputs["gc"] = torch.empty(
                (T, args.hidden), dtype=torch.bfloat16, device=device)

        def invoke(name):
            if name == "baseline":
                baseline_layer.compute_staged(output=outputs["baseline"])
            elif name == "cake":
                cake_layer.compute_staged(output=outputs["cake"])
            else:
                import gc_moe
                gc_moe.sm120_fp8_fp4_gc_moe(
                    outputs[name], *gc_weights, gc_buffer,
                    recipe=(1, 1, FP4_BLOCK), activation=args.activation,
                    activation_clamp=args.activation_clamp,
                    fast_math=bool(args.fast_math),
                    num_tokens_bound=max(1, T), **shared_kwargs)

        use_cuda_graph = (
            args.cuda_graph_max_tokens_per_rank > 0
            and T <= args.cuda_graph_max_tokens_per_rank
        )
        bench_fn = _bench_cuda_graph_ms if use_cuda_graph else _bench_gpu_time_ms
        operator_stats = {}
        # Equal inputs, capacity, warmup/iterations/L2 flush/timing on both
        # paths. Alternate the operator order across points to reduce drift
        # bias.
        order = [name for name in ("baseline", "cake", "gc") if name in arms]
        if len(results) % 2:
            order.reverse()
        for name in order:
            measured = bench_fn(
                lambda name=name: invoke(name), warmup=args.warmup,
                iters=args.iters, inner_iters=args.inner_iters, device=device,
                barrier_fn=dist.barrier)
            measured["operator"] = {
                "baseline": "flashinfer.moe_ep.Sm120_Mxfp4_Mxfp8_Bf16"
                            "_Cutedsl_MegaMoe",
                "cake": "flashinfer.moe_ep.Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoe"
                        f"(transport_table={args.cake_transport_table},"
                        f" shared_expert={bool(num_shared)})",
                "gc": "gc_moe.sm120_fp8_fp4_gc_moe",
            }[name]
            measured["rank"] = rank
            rank_windows = [None] * num_ranks
            dist.all_gather_object(rank_windows, measured)
            measured["ranks"] = rank_windows
            measured["per_rank_median_us"] = [
                float(row["median_ms"]) * 1e3 for row in rank_windows]
            measured["slowest_rank_median_us"] = max(
                measured["per_rank_median_us"])
            operator_stats[name] = measured
        if cake_layer is not None:
            # Host-side check outside the timed region: a set protocol_error
            # invalidates the whole point (codec overflow etc.).
            err = cake_layer._kernel.protocol_error(cake_layer._workspace)
            if err != 0:
                raise RuntimeError(
                    f"cake protocol_error={err} at T={T}; timings invalid")

        row = {
            "num_tokens_per_rank": T,
            "num_tokens_global": T * num_ranks,
            "operator_stats": operator_stats,
            "mode": "cuda_graph" if use_cuda_graph else "eager",
        }
        workload = _collect_throughput_workload(
            topk_idx, args.num_experts, args.hidden, args.intermediate_hidden,
            weight_storage_by_rank, num_shared * args.intermediate_hidden)
        workload["inter_gpu_useful_payload_bytes"] = workload.pop("nvlink_useful_payload_bytes")
        for rank_workload in workload["per_rank"]:
            rank_workload["inter_gpu_injected_tx_payload_bytes"] = rank_workload.pop(
                "nvlink_injected_tx_payload_bytes")
        row["workload"] = workload
        if "gc" in outputs:
            candidate = outputs["gc"].float().flatten()
            for arm in ("baseline", "cake"):
                if arm not in outputs:
                    continue
                comparison = outputs[arm].float().flatten()
                if candidate.numel():
                    agreement = torch.nn.functional.cosine_similarity(comparison, candidate, dim=0)
                    max_abs = (comparison - candidate).abs().max()
                else:
                    agreement, max_abs = candidate.new_tensor(1.), candidate.new_tensor(0.)
                dist.all_reduce(agreement, op=dist.ReduceOp.MIN)
                dist.all_reduce(max_abs, op=dist.ReduceOp.MAX)
                row[f"gc_{arm}_cos_min"] = float(agreement)
                row[f"gc_{arm}_max_abs"] = float(max_abs)
                if num_shared and arm == "cake" and not row["gc_cake_cos_min"] >= .99:
                    raise AssertionError(f'SE output mismatch at T={T}: {row["gc_cake_cos_min"]}')
        if "baseline" in operator_stats:
            row["baseline_us"] = operator_stats["baseline"]["slowest_rank_median_us"]
        if "cake" in operator_stats:
            row["cake_us"] = operator_stats["cake"]["slowest_rank_median_us"]
        if "gc" in operator_stats:
            row["gc_us"] = operator_stats["gc"]["slowest_rank_median_us"]
        if "baseline" in operator_stats and "gc" in operator_stats:
            row["speedup"] = row["baseline_us"] / row["gc_us"]
        if "cake" in operator_stats and "gc" in operator_stats:
            row["cake_over_gc"] = row["cake_us"] / row["gc_us"]
        stats = operator_stats[order[0]]

        if rank == 0:
            print(
                f"  T={T:6d}  mode={row['mode']:10s}  "
                f"median={stats['median_ms']*1e3:8.1f} us  "
                f"min={stats['min_ms']*1e3:8.1f}  max={stats['max_ms']*1e3:8.1f}  "
                f"std={stats['std_ms']*1e3:5.1f}  "
                f"slowest_rank_median={stats['slowest_rank_median_us']:8.1f} us"
            )
            compare = ""
            if "baseline" in row and "gc" in row:
                compare = (f" flashinfer={row['baseline_us']:.3f}us "
                           f"gc={row['gc_us']:.3f}us "
                           f"speedup={row['speedup']:.5f}")
            print(f"COMPARE tokens_per_rank={T} total_tokens={T * num_ranks} "
                  f"mode={row['mode']}{compare}", flush=True)

        results.append(row)
        outputs.clear()
        del x_bf16, x_fp8, x_sf, x_sf_bytes, topk_idx, topk_w
        gc.collect()
        torch.cuda.empty_cache()

    if rank == 0:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump({
                "world_size": num_ranks,
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
                    "shared_expert": bool(num_shared),
                    "num_shared_experts": num_shared,
                    "shared_weight_gran_k": args.shared_weight_gran_k,
                },
                "arms": arms,
                "quantizer": {"source": str(caster_source),
                              "sha256": hashlib.sha256(caster_source.read_bytes()).hexdigest()},
                "operators": {
                    "baseline": "flashinfer.moe_ep.MoEEpLayer"
                                "(Sm120_Mxfp4_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,"
                                " quantize_input=False)",
                    "cake": "flashinfer.moe_ep.MoEEpLayer"
                            "(Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig, "
                            f"transport_table={args.cake_transport_table}, "
                            f"shared_expert={bool(num_shared)}, "
                            "quantize_input=False)",
                    "gc": "gc_moe.sm120_fp8_fp4_gc_moe",
                },
                "modules": {
                    "baseline": flashinfer.__file__,
                    "flashinfer_version": flashinfer.__version__,
                } if flashinfer is not None else {},
                "warmup": args.warmup,
                "iters": args.iters,
                "inner_iters": args.inner_iters,
                "cuda_graph_max_tokens_per_rank": args.cuda_graph_max_tokens_per_rank,
                "results": results,
            }, f, indent=2)
        print(f"\nWrote {args.output_json}")

    if baseline_layer is not None:
        baseline_layer.destroy()
    if cake_layer is not None:
        # Collective: releases the NCCL transport session on every rank.
        cake_layer.destroy()
    if gc_buffer is not None:
        gc_buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
