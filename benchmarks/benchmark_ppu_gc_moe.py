"""Time PPU GC MoE against PPU MegaMoE using eager device events.

Both arms use staged inputs. Each MegaMoE request completes pre-dispatch
outside its timed interval. GC uses the full public API. The unfused SE
comparisons time the routed operator, serial shared expert and output addition.
"""

import argparse
import json
import math
import sys
from functools import partial
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
import triton
import triton.language as tl

import gc_moe
from tests.test_ppu_gc_moe import (
    DEFAULT_TOKENS_PER_RANK, GCMoE, SharedExpert, _gather, add_workload_args,
    build_workload, configure_workload, init_dist, launch_ranks,
)


# The serial SE baseline uses one SwiGLU and MXFP4 conversion kernel.
@triton.jit
def _swiglu_quantize(x, packed, scales, H: tl.constexpr, BLOCK: tl.constexpr,
                     CLAMP: tl.constexpr, FAST_MATH: tl.constexpr):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(x + row * (2 * H) + offsets).to(tl.float32)
    up = tl.load(x + row * (2 * H) + H + offsets).to(tl.float32)
    gate = tl.minimum(gate, CLAMP)
    up = tl.minimum(tl.maximum(up, -CLAMP), CLAMP)
    denominator = 1.0 + tl.exp(-gate)
    if FAST_MATH:
        silu = gate / denominator
    else:
        silu = tl.div_rn(gate, denominator)
    value = silu * up
    grouped = tl.reshape(value, (BLOCK // 32, 32))
    amax = tl.max(tl.abs(grouped), axis=1)
    bits = amax.to(tl.uint32, bitcast=True)
    exponent = ((bits >> 23) & 255).to(tl.int32) - 2
    exponent += ((bits & 0x7fffff) != 0).to(tl.int32)
    exponent = tl.minimum(tl.maximum(exponent, 113), 254)
    inverse = ((254 - exponent).to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    q = tl.reshape(grouped * inverse[:, None], (BLOCK,))
    a = tl.abs(q)
    code = ((a > .25).to(tl.int32) + (a >= .75).to(tl.int32)
            + (a > 1.25).to(tl.int32) + (a >= 1.75).to(tl.int32)
            + (a > 2.5).to(tl.int32) + (a >= 3.5).to(tl.int32)
            + (a > 5).to(tl.int32))
    code |= (q < 0).to(tl.int32) << 3
    lo, hi = tl.split(tl.reshape(code, (BLOCK // 2, 2)))
    tl.store(packed + row * (H // 2) + block * (BLOCK // 2) + tl.arange(0, BLOCK // 2), lo | (hi << 4))
    tl.store(scales + row * (H // 32) + block * (BLOCK // 32) + tl.arange(0, BLOCK // 32), exponent)


class SerialSharedExpert(SharedExpert):
    def __init__(self, deep_gemm, w1, w1_sf, w2, w2_sf, capacity):
        super().__init__(deep_gemm, w1, w1_sf, w2, w2_sf, capacity)
        self.mid = w1.new_empty((capacity, self.inter // 2))
        self.mid_sf = w1.new_empty((capacity, self.inter // 32))

    def bind(self, tokens, packed, scales, *, activation_clamp, fast_math):
        inputs = packed[:tokens], scales[:tokens]
        gate_up = self.gate_up[:tokens]
        mid = self.mid[:tokens], self.mid_sf[:tokens]
        output = self.output[:tokens]
        block = min(1024, self.inter & -self.inter)
        swiglu = _swiglu_quantize[(tokens, self.inter // block)]

        def launch():
            self.gemm(inputs, self.w1, None, gate_up)
            swiglu(gate_up, *mid, H=self.inter, BLOCK=block, CLAMP=activation_clamp,
                   FAST_MATH=fast_math, num_warps=4)
            self.gemm(mid, self.w2, None, output)
            return output

        return launch


class MegaMoE:
    def __init__(self, deep_gemm, group, experts, topk, output, w1, w2):
        capacity, hidden = output.shape
        inter = 2 * w2[0].shape[-1]
        self.deep_gemm = deep_gemm
        self.w1, self.w2 = deep_gemm.transform_weights_for_mega_moe(w1, w2)
        self.workspace = deep_gemm.get_symm_buffer_for_mega_moe(
            group, experts, capacity, topk, hidden, inter)
        self.output = output

    def prepare(self, x, ids, weights):
        workspace = self.workspace
        self.deep_gemm.mega_moe_pre_dispatch(
            x, ids, weights, workspace.x, workspace.x_sf,
            workspace.topk_idx, workspace.topk_weights,
            num_tokens=x.shape[0], group_size=32, mma_type="mxf4xmxf4")

    def bind(self, output, *, activation, activation_clamp, fast_math):
        return partial(
            self.deep_gemm.fp4_fp4_mega_moe,
            y=output, l1_weights=self.w1, l2_weights=self.w2,
            sym_buffer=self.workspace, activation=activation,
            activation_clamp=activation_clamp, fast_math=fast_math)


def _compare_latency(steps, flush, *, warmup, iters, inner_iters, ranks):
    # Compile each arm and initialize its launch state before any timed request.
    for prepare, step in steps:
        if prepare is not None:
            prepare()
        torch.cuda.synchronize()
        dist.barrier()
        step()
        torch.cuda.synchronize()

    samples = [[] for _ in steps]
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for window in range(warmup + iters):
        measure = window >= warmup
        for offset in range(len(steps)):
            index = (window + offset) % len(steps)
            prepare, step = steps[index]
            flush.zero_()
            elapsed = 0.0
            for _ in range(inner_iters):
                if prepare is not None:
                    prepare()
                # Finish preprocessing on every rank before starting the timer.
                torch.cuda.synchronize()
                dist.barrier()
                if measure:
                    begin.record()
                step()
                if measure:
                    end.record()
                    end.synchronize()
                    elapsed += begin.elapsed_time(end)
            if measure:
                samples[index].append(elapsed * 1000 / inner_iters)
    medians = [median(series) for series in samples]
    rank_medians = [None] * ranks
    dist.all_gather_object(rank_medians, medians)
    return tuple(map(max, zip(*rank_medians)))


def _bind_gc(gc, tokens, operator_options):
    return partial(
        gc_moe.ppu_fp4_fp4_gc_moe, gc.output[:tokens], *gc.weights,
        gc.workspace, num_tokens_bound=tokens, **gc.shared_options, **operator_options)


def run_performance(args, deep_gemm, group, rank, ranks, x, gc, ids, weights, routed_weights,
                     shared_ref, output):
    _, _, experts, topk = args.shape
    all_ids = _gather(ids, ranks)
    if rank == 0:
        local_experts = experts // ranks
        source = torch.arange(ranks, device="cpu")[:, None, None]
        for tokens in args.tokens:
            routes = all_ids[:, :tokens]
            remote = (routes // local_experts != source).sum().item()
            print(f"[{args.model}] T={tokens} PPU_WORKLOAD active_experts={routes.unique().numel()} "
                  f"routed_assignments={routes.numel()} remote_assignments={remote}", flush=True)
    mega = None
    if not args.gc_only:
        mega = MegaMoE(deep_gemm, group, experts, topk, torch.empty_like(output), *routed_weights)
    packed, scales = gc.workspace.x, gc.workspace.x_sf
    operator_options = dict(activation=args.activation, activation_clamp=args.activation_clamp,
                            fast_math=bool(args.fast_math))

    routed = None
    if shared_ref is not None and mega is not None:
        routed = GCMoE(args, group, torch.empty_like(output), routed_weights)
        routed.workspace.x.copy_(packed)
        routed.workspace.x_sf.copy_(scales)
        routed.workspace.topk_idx.copy_(ids)
        routed.workspace.topk_weights.copy_(weights)

    flush = torch.empty(8 << 30, dtype=torch.uint8, device="cuda")
    results = []
    try:
        for tokens in args.tokens:
            steps = [(None, _bind_gc(gc, tokens, operator_options))]
            if mega is not None:
                prepare_mega = partial(mega.prepare, x[:tokens], ids[:tokens], weights[:tokens])
                mega_output = mega.output[:tokens]
                baseline = mega.bind(mega_output, **operator_options)
                if routed is not None:
                    shared_launch = shared_ref.bind(tokens, packed, scales, activation_clamp=args.activation_clamp,
                                                    fast_math=bool(args.fast_math))
                    mega_combined = torch.empty_like(mega_output)
                    routed_launch = _bind_gc(routed, tokens, operator_options)
                    routed_output = routed.output[:tokens]
                    gc_combined = torch.empty_like(routed_output)

                    def serial_mega():
                        baseline()
                        torch.add(mega_output, shared_launch(), out=mega_combined)

                    def serial_gc():
                        routed_launch()
                        torch.add(routed_output, shared_launch(), out=gc_combined)

                    steps.append((prepare_mega, serial_mega))
                    steps.append((None, serial_gc))
                    steps.append((None, routed_launch))
                else:
                    steps.append((prepare_mega, baseline))
            values = _compare_latency(steps, flush, warmup=args.warmup, iters=args.iters,
                                      inner_iters=args.inner_iters, ranks=ranks)
            gc_us = values[0]
            if mega is None:
                if rank == 0:
                    name = "GC_SE" if args.shared_expert else "GC"
                    print(f"[{args.model}] T={tokens} {name}={gc_us:.6f} us", flush=True)
                results.append(dict(num_tokens_per_rank=tokens, num_tokens_global=tokens * ranks,
                                    mode="eager", gc_us=gc_us))
                continue
            mega_us = values[1]
            row = dict(num_tokens_per_rank=tokens, num_tokens_global=tokens * ranks, mode="eager",
                       gc_us=gc_us, baseline_us=mega_us, speedup=mega_us / gc_us)
            if routed is not None:
                unfused_us, routed_us = values[2:]
                row["gc_unfused_us"] = unfused_us
                row["gc_routed_us"] = routed_us
                if rank == 0:
                    print(f"[{args.model}] T={tokens} GC_SE={gc_us:.6f} us MegaMoE_SE={mega_us:.6f} us "
                          f"GC_UNFUSED={unfused_us:.6f} us GC_ROUTED={routed_us:.6f} us "
                          f"speedup={mega_us / gc_us:.6f}x speedup_vs_gc_unfused={unfused_us / gc_us:.6f}x "
                          f"overhead={gc_us / routed_us - 1:+.6%}", flush=True)
            elif rank == 0:
                print(f"[{args.model}] T={tokens} GC={gc_us:.6f} us MegaMoE={mega_us:.6f} us "
                      f"speedup={mega_us / gc_us:.6f}x", flush=True)
            results.append(row)
        if rank == 0 and mega is not None:
            gc_total = sum(row["gc_us"] for row in results)
            mega_total = sum(row["baseline_us"] for row in results)
            print(f"[{args.model}] speedup by mean latency={mega_total / gc_total:.4f}x", flush=True)
            if routed is not None:
                unfused_total = sum(row["gc_unfused_us"] for row in results)
                routed_total = sum(row["gc_routed_us"] for row in results)
                print(f"[{args.model}] speedup vs unfused GC by mean latency={unfused_total / gc_total:.4f}x",
                      flush=True)
                print(f"[{args.model}] SE overhead by mean latency={gc_total / routed_total - 1:+.2%}",
                      flush=True)
        return results
    finally:
        if mega is not None:
            mega.workspace.destroy()
        if routed is not None:
            routed.destroy()


def main(*, shared=False):
    parser = argparse.ArgumentParser(description=__doc__)
    add_workload_args(parser, shared=shared, capacity=0)
    parser.add_argument("--activation", choices=("swiglu",), default="swiglu")
    parser.add_argument("--activation-alpha", type=float, choices=(1.0,), default=1.0,
                        help="PPU SwiGLU uses alpha=1.")
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--fast-math", type=int, choices=(0, 1), default=1)
    parser.add_argument("--shared-weight-gran-k", type=int, choices=(32, 128), default=32)
    parser.add_argument("--warmup", type=int, default=10, help="Warmup windows per arm.")
    parser.add_argument("--iters", "--windows", type=int, default=30,
                        help="Measured windows per token count.")
    parser.add_argument("--inner-iters", type=int, default=100, help="Requests per window.")
    parser.add_argument("--cuda-graph-max-tokens-per-rank", type=int, choices=(0,), default=0,
                        help="PPU communication requires eager submission.")
    parser.add_argument("--arms", default="baseline,gc", metavar="ARMS",
                        help="Time MegaMoE and GC MoE, or GC MoE alone.")
    parser.add_argument("--gc-only", action="store_true", help="Time only GC MoE.")
    args = parser.parse_args()
    try:
        args.tokens = (tuple(int(t) for t in args.num_tokens_per_rank_list.split(",") if t.strip())
                       if args.num_tokens_per_rank_list else DEFAULT_TOKENS_PER_RANK)
        if not args.tokens:
            raise ValueError("expected at least one token count")
        args.num_max_tokens_per_rank = args.num_max_tokens_per_rank or max(args.tokens)
        if not all(0 < tokens <= args.num_max_tokens_per_rank for tokens in args.tokens):
            raise ValueError("token counts must be positive and fit workspace capacity")
        if args.warmup < 0 or args.iters < 1 or args.inner_iters < 1:
            raise ValueError("expected nonnegative warmup and positive iters and inner-iters")
        if math.isnan(args.activation_clamp) or args.activation_clamp < 0:
            raise ValueError("activation clamp must be nonnegative")
    except ValueError as exc:
        parser.error(str(exc))
    configure_workload(parser, args, shared=shared)
    arms = {arm.strip() for arm in args.arms.split(",")}
    if "gc" not in arms or arms - {"baseline", "gc"}:
        parser.error("--arms must be baseline,gc or gc")
    args.gc_only = args.gc_only or arms == {"gc"}
    if args.shared_expert and any(dim % args.shared_weight_gran_k for dim in args.shape[:2]):
        parser.error("hidden and intermediate dimensions must be divisible by shared-weight-gran-k")
    if args.output_json is None:
        suffix = "_se" if args.shared_expert else ""
        args.output_json = Path(f"results/_performance_results_ppu_gc{suffix}.json")

    local_rank, ranks = launch_ranks(args)
    init_dist(local_rank)
    try:
        import deep_gemm

        group = dist.group.WORLD
        rank = dist.get_rank()
        granularity = args.shared_weight_gran_k
        x, packed, scales, ids, weights, routed_weights, shared_weights, output = build_workload(
            args, rank, ranks, shared_weight_gran_k=granularity)
        shared_op = None
        if shared_weights and not args.gc_only:
            (w1, s1), (w2, s2) = shared_weights
            if granularity == 128:
                s1 = s1.repeat_interleave(4, dim=-1)
                s2 = s2.repeat_interleave(4, dim=-1)
            shared_op = SerialSharedExpert(deep_gemm, w1, s1, w2, s2, args.num_max_tokens_per_rank)
        gc = GCMoE(args, group, output, routed_weights, shared_weights, shared_weight_gran_k=granularity)
        try:
            workspace = gc.workspace
            workspace.x.copy_(packed)
            workspace.x_sf.copy_(scales)
            workspace.topk_idx.copy_(ids)
            workspace.topk_weights.copy_(weights)
            rows = run_performance(args, deep_gemm, group, rank, ranks, x, gc, ids, weights,
                                   routed_weights, shared_op, output)
        finally:
            gc.destroy()

        if rank == 0:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps({
                "world_size": ranks,
                "config": {
                    "model": args.model,
                    "hidden": args.shape[0],
                    "intermediate_hidden": args.shape[1],
                    "num_experts": args.shape[2],
                    "num_topk": args.shape[3],
                    "num_max_tokens_per_rank": args.num_max_tokens_per_rank,
                    "num_shared_experts": args.num_shared_experts,
                    "shared_weight_gran_k": granularity if args.shared_expert else None,
                    "activation": args.activation,
                    "activation_alpha": args.activation_alpha,
                    "activation_clamp": args.activation_clamp if math.isfinite(args.activation_clamp) else None,
                    "fast_math": bool(args.fast_math),
                },
                "warmup": args.warmup,
                "iters": args.iters,
                "inner_iters": args.inner_iters,
                "cuda_graph_max_tokens_per_rank": args.cuda_graph_max_tokens_per_rank,
                "performance_timing": {
                    "mode": "eager",
                    "gc": "public_packed_input_api",
                    "mega": None if args.gc_only else "operator_after_deepgemm_pre_dispatch",
                    **({"gc_unfused": "public_packed_input_api_then_serial_shared_expert",
                        "gc_routed": "public_packed_input_api"} if args.shared_expert and not args.gc_only else {}),
                },
                "results": rows,
            }, indent=2) + "\n")
            print(f"Wrote {args.output_json}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
