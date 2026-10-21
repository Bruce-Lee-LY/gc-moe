"""
GLM5.2 accuracy comparison: GC MoE vs Mega MoE and BF16.

Compares the fused FP8/FP4 GC MoE kernel against a BF16 reference forward
pass (ported from ``rtp_moe/src/common/common.py::reference_moe_forward``) and
reports cosine similarity per token-count.

Highlights:
  * Cross-rank dispatch/combine happens entirely
    inside ``fp8_fp4_gc_moe`` via PyTorch symmetric memory.
  * Sweeps tokens per rank: 1, 2, 4, ..., 256.
  * GLM5.2 (H6144/I2048/E256/topk8) and DSV4.1 Flash (H5120/I2304/E384/topk6).
  * Uses one rank per GPU; NUM_EXPERTS must be divisible by NUM_RANKS.

USAGE:
    torchrun --nproc_per_node=4 tests/test_gc_moe.py
    torchrun --nproc_per_node=8 tests/test_gc_moe.py
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

_REPO_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _REPO_ROOT)

import torch
import torch.distributed as dist

import deep_gemm
import gc_moe
assert Path(gc_moe.__file__).resolve().is_relative_to(Path(_REPO_ROOT).resolve()), gc_moe.__file__
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8


DEFAULT_TOKENS_PER_RANK: List[int] = [
    1, 2, 4, 8, 16, 32, 64, 128, 256,
]
FP4_BLOCK = 32


# ---------------------------------------------------------------------------
# BF16 reference (mirrors reference_moe_forward; per-expert vectorization to
# stay tractable at T=32k).
# ---------------------------------------------------------------------------

def reference_moe_forward(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    w1_bf16_full: torch.Tensor,
    w2_bf16_full: torch.Tensor,
    activation: str = "swiglu",
    activation_clamp: Optional[float] = None,
    activation_alpha: float = 1.0,
) -> torch.Tensor:
    T, hidden = x.shape
    E, two_n, _ = w1_bf16_full.shape
    intermediate = two_n // 2

    out = torch.zeros((T, hidden), dtype=torch.float32, device=x.device)
    x_f = x.float()

    for e in range(E):
        sel = (topk_idx == e)
        if not bool(sel.any()):
            continue
        rows, slots = sel.nonzero(as_tuple=True)
        xs = x_f[rows]
        w1_e = w1_bf16_full[e].float()
        w2_e = w2_bf16_full[e].float()
        l1 = xs @ w1_e.T
        gate, up = l1[:, :intermediate], l1[:, intermediate:]
        if activation_clamp is not None:
            gate = gate.clamp(max=activation_clamp)
            up = up.clamp(-activation_clamp, activation_clamp)
        if activation == "swiglu":
            silu_gate = gate * torch.sigmoid(gate)
            mid = silu_gate * up
        elif activation == "swiglu_oai":
            mid = gate * torch.sigmoid(activation_alpha * gate) * (up + 1)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        l2 = mid @ w2_e.T
        wgts = topk_weights[rows, slots].float().unsqueeze(1)
        out.index_add_(0, rows, wgts * l2)
    return out.to(x.dtype)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0:
        return 1.0 if b.numel() == 0 else 0.0
    a = a.flatten().float()
    b = b.flatten().float()
    denom = a.norm() * b.norm()
    if denom.item() == 0:
        return float(torch.equal(a, b))
    return float((a @ b / denom).item())


# ---------------------------------------------------------------------------
# Weight quantization
# ---------------------------------------------------------------------------

def cast_grouped_weights_to_fp4(
    bf16_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
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


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def build_full_weights(
    num_experts: int, intermediate: int, hidden: int, device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(7)
    w1_full = torch.randn(
        num_experts, 2 * intermediate, hidden,
        dtype=torch.bfloat16, device=device, generator=g,
    ) * 0.05
    w2_full = torch.randn(
        num_experts, hidden, intermediate,
        dtype=torch.bfloat16, device=device, generator=g,
    ) * 0.05
    return w1_full, w2_full


def cast_tokens_to_fp8(x: torch.Tensor):
    if x.size(0) == 0:
        return x.to(torch.float8_e4m3fn), torch.empty(
            (0, x.size(1) // 128), dtype=torch.int32, device=x.device)
    return per_token_cast_to_fp8(
        x, use_ue8m0=True, gran_k=FP4_BLOCK, use_packed_ue8m0=True)


def build_tokens(
    T: int, hidden: int, num_experts: int, num_topk: int,
    rank: int, device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--intermediate-hidden", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--case-category", choices=("default", "boundary", "asymmetric"), default="default")
    parser.add_argument("--num-tokens-per-rank-list", type=str, default="",
                        help="Comma-separated list of per-rank token counts. "
                             "Empty (default) uses 1,2,4,8,16,32,64,128,256 tokens per rank.")
    parser.add_argument("--per-rank-tokens-list", type=str, default="",
                        help="Semicolon-separated list of per-rank token "
                             "vectors. Each vector is comma-separated with "
                             "exactly world_size values; the i-th element is "
                             "the T used by rank i in that iteration. When "
                             "set, this OVERRIDES --num-tokens-per-rank-list and the "
                             "test runs one asymmetric iteration per vector.")
    parser.add_argument("--num-max-tokens-per-rank", type=int, default=0,
                        help="Override max tokens per rank (default = max "
                             "of all T values across sweep / per-rank vectors).")
    parser.add_argument("--activation", choices=("swiglu",), default="swiglu")
    parser.add_argument("--activation-alpha", type=float, default=1.0)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--fast-math", type=int, default=1)
    parser.add_argument("--cosine-threshold", type=float, default=0.97,
                        help="Uniform minimum cosine for independent-reference and GC-to-Mega checks (0 to 1).")
    parser.add_argument("--gc-replays", type=int, default=5)
    parser.add_argument("--output-json", type=str, default="results/_accuracy_results_gc.json")
    args = parser.parse_args()
    if not 0.0 <= args.cosine_threshold <= 1.0:
        parser.error("--cosine-threshold must be between 0 and 1 for all cosine checks")

    # Parse per-rank vectors (asymmetric mode) first; world_size is validated
    # after `dist.init_process_group` below.
    per_rank_vectors: List[List[int]] = []
    if args.per_rank_tokens_list:
        for vec_str in args.per_rank_tokens_list.split(";"):
            vec_str = vec_str.strip()
            if not vec_str:
                continue
            vec = [int(t) for t in vec_str.split(",") if t.strip()]
            per_rank_vectors.append(vec)

    sweep = DEFAULT_TOKENS_PER_RANK if not args.num_tokens_per_rank_list else \
        [int(t) for t in args.num_tokens_per_rank_list.split(",") if t.strip()]
    if per_rank_vectors:
        max_T_across_vecs = max(max(v) for v in per_rank_vectors)
        num_max_tokens_per_rank = args.num_max_tokens_per_rank or max(1, max_T_across_vecs)
    else:
        num_max_tokens_per_rank = args.num_max_tokens_per_rank or max(1, max(sweep))

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()

    if rank == 0:
        print(f"[tests/test_gc_moe] world_size={num_ranks} device={device}")

    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) not in [(10, 0), (10, 3)]:
        if rank == 0:
            print(f"Skipping: SM100/SM103 only, got SM{major}{minor}")
        dist.destroy_process_group()
        raise RuntimeError("SM100/SM103 hardware is required; this is not a passed test")

    if args.num_experts % num_ranks != 0:
        if rank == 0:
            print(f"Skipping: num_experts={args.num_experts} not divisible by "
                  f"num_ranks={num_ranks}")
        dist.destroy_process_group()
        return

    # Validate per-rank vectors against the live world_size now that the
    # process group is up.
    if per_rank_vectors:
        for i, vec in enumerate(per_rank_vectors):
            if len(vec) != num_ranks:
                if rank == 0:
                    print(f"ERROR: per-rank vector #{i} has length "
                          f"{len(vec)}, expected world_size={num_ranks}: {vec}")
                dist.destroy_process_group()
                sys.exit(2)
            for t in vec:
                if t < 0:
                    if rank == 0:
                        print(f"ERROR: per-rank vector #{i} has T={t} < 0: {vec}")
                    dist.destroy_process_group()
                    sys.exit(2)
                if t > num_max_tokens_per_rank:
                    if rank == 0:
                        print(f"ERROR: per-rank vector #{i} has T={t} > "
                              f"num_max_tokens_per_rank={num_max_tokens_per_rank}: "
                              f"{vec}")
                    dist.destroy_process_group()
                    sys.exit(2)

    if rank == 0:
        print(f"Config: hidden={args.hidden} intermediate={args.intermediate_hidden} "
              f"experts={args.num_experts} topk={args.num_topk} "
              f"activation={args.activation} alpha={args.activation_alpha} "
              f"clamp={args.activation_clamp} fast_math={bool(args.fast_math)}")
        if per_rank_vectors:
            print(f"num_max_tokens_per_rank={num_max_tokens_per_rank}")
            print(f"asymmetric per-rank vectors ({len(per_rank_vectors)}):")
            for i, vec in enumerate(per_rank_vectors):
                print(f"  vec[{i}] = {vec}")
        else:
            print(f"num_max_tokens_per_rank={num_max_tokens_per_rank} sweep={sweep}")

    # Full-set BF16 weights, deterministic on every rank; slice the local set.
    w1_full, w2_full = build_full_weights(
        args.num_experts, args.intermediate_hidden, args.hidden, device,
    )
    Eloc = args.num_experts // num_ranks
    w1_local = w1_full[rank * Eloc:(rank + 1) * Eloc].contiguous()
    w2_local = w2_full[rank * Eloc:(rank + 1) * Eloc].contiguous()

    # FP4-quantize and apply each operator's weight layout.
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

    def run_kernel(T: int, operator, weights, buffer) -> torch.Tensor:
        y = torch.empty((T, args.hidden), dtype=torch.bfloat16, device=device)
        operator(
            y, *weights, buffer,
            recipe=(1, 1, FP4_BLOCK),
            activation=args.activation,
            activation_clamp=args.activation_clamp,
            activation_alpha=args.activation_alpha,
            fast_math=bool(args.fast_math),
        )
        return y

    results = []
    all_pass = True
    threshold = args.cosine_threshold

    # Build iteration plan. Symmetric mode: every rank uses the same T from
    # the sweep. Asymmetric mode: each iteration uses one per-rank vector,
    # and this rank picks vec[rank] as its own T. The cross-rank cosine
    # all-reduce works identically in both modes (each rank contributes
    # its own scalar).
    if per_rank_vectors:
        iter_plan = [
            ("asym", i, vec, vec[rank]) for i, vec in enumerate(per_rank_vectors)
        ]
    else:
        iter_plan = []
        for T in sweep:
            if T > num_max_tokens_per_rank:
                if rank == 0:
                    print(f"  Skipping T={T} (> num_max_tokens_per_rank)")
                continue
            iter_plan.append(("sym", T, None, T))

    for mode, label_key, vec, my_T in iter_plan:
        x_bf16, topk_idx, topk_w = build_tokens(
            my_T, args.hidden, args.num_experts, args.num_topk, rank, device,
        )

        # Reference (BF16 dense) — full-rank computation.
        ref = reference_moe_forward(
            x_bf16, topk_idx, topk_w, w1_full, w2_full,
            activation=args.activation,
            activation_clamp=args.activation_clamp,
            activation_alpha=args.activation_alpha,
        )

        # Stage inputs into the symmetric buffer.
        x_fp8, x_sf = cast_tokens_to_fp8(x_bf16)
        for buffer in (base_buffer, candidate_buffer):
            buffer.x[:my_T].copy_(x_fp8)
            buffer.x_sf[:my_T].copy_(x_sf)
            buffer.topk_idx[:my_T].copy_(topk_idx)
            buffer.topk_weights[:my_T].copy_(topk_w)
        torch.cuda.synchronize(device)
        dist.barrier()

        baseline = run_kernel(my_T, deep_gemm.fp8_fp4_mega_moe, base_weights, base_buffer)
        torch.cuda.synchronize(device)
        y = run_kernel(my_T, gc_moe.fp8_fp4_gc_moe, candidate_weights, candidate_buffer)
        torch.cuda.synchronize(device)
        # Repeated eager launches and captured replay exercise workspace generations.
        def candidate_call():
            gc_moe.fp8_fp4_gc_moe(
                y, *candidate_weights, candidate_buffer,
                recipe=(1, 1, FP4_BLOCK), activation=args.activation,
                activation_clamp=args.activation_clamp,
                activation_alpha=args.activation_alpha, fast_math=bool(args.fast_math))
        for _ in range(args.gc_replays):
            candidate_call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            candidate_call()
        for _ in range(args.gc_replays):
            graph.replay()
        torch.cuda.synchronize(device)
        del graph
        cos_baseline = cosine(baseline, ref)
        cos_comparison = cosine(y, baseline)
        comparison = torch.tensor([cos_baseline, cos_comparison], dtype=torch.float64, device=device)
        gathered = [torch.empty_like(comparison) for _ in range(num_ranks)]
        dist.all_gather(gathered, comparison)
        baseline_cos_per_rank = [float(v[0].item()) for v in gathered]
        comparison_cos_per_rank = [float(v[1].item()) for v in gathered]
        finite = torch.tensor([int(torch.isfinite(y).all())], device=device)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        cos = cosine(y, ref)
        cos_t = torch.tensor([cos], dtype=torch.float64, device=device)
        cos_min_t = cos_t.clone()
        # Gather per-rank cos values so we can report which ranks failed.
        cos_all = torch.zeros(num_ranks, dtype=torch.float64, device=device)
        cos_all[rank] = cos
        dist.all_reduce(cos_all, op=dist.ReduceOp.SUM)
        dist.all_reduce(cos_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(cos_min_t, op=dist.ReduceOp.MIN)
        cos_avg = float(cos_t.item() / num_ranks)
        cos_min = float(cos_min_t.item())
        per_rank_cos = [float(v) for v in cos_all.cpu().tolist()]
        ok = (cos_min >= threshold and min(baseline_cos_per_rank) >= threshold
              and min(comparison_cos_per_rank) >= threshold and bool(finite.item()))
        all_pass = all_pass and ok

        if rank == 0:
            status = "OK" if ok else "FAIL"
            per_rank_str = "[" + ",".join(f"{v:.4f}" for v in per_rank_cos) + "]"
            if mode == "sym":
                print(f"  tokens_per_rank={my_T:4d} total_tokens={my_T * num_ranks:5d} cos[r0]={cos:.4f}  cos_avg={cos_avg:.4f}  "
                      f"cos_min={cos_min:.4f}  per_rank={per_rank_str}  [{status}]")
            else:
                T_total = sum(vec)
                print(f"  asym#{label_key} vec={vec} T_total={T_total}  "
                      f"cos[r0]={cos:.4f}  cos_avg={cos_avg:.4f}  "
                      f"cos_min={cos_min:.4f}  per_rank={per_rank_str}  [{status}]")
        entry = {
            "num_tokens_per_rank": my_T,
            "baseline_cos_per_rank": baseline_cos_per_rank,
            "gc_vs_baseline_cos_per_rank": comparison_cos_per_rank,
            "baseline_cos_min": min(baseline_cos_per_rank),
            "gc_vs_baseline_cos_min": min(comparison_cos_per_rank),
            "finite": bool(finite.item()),
            "num_tokens_global": int(sum(vec)) if vec is not None else my_T * num_ranks,
            "gc_replays_eager_and_graph": args.gc_replays,
            "cos_rank0": cos,
            "cos_avg": cos_avg,
            "cos_min": cos_min,
            "cos_per_rank": per_rank_cos,
            "passed": ok,
        }
        if mode == "asym":
            entry["asym_vector"] = vec
            entry["asym_index"] = label_key
            entry["num_tokens_global"] = int(sum(vec))
        results.append(entry)

        del ref, y, baseline, x_bf16, x_fp8, x_sf, topk_idx, topk_w
        torch.cuda.empty_cache()

    if rank == 0:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump({
                "world_size": num_ranks,
                "case_category": args.case_category,
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
                "mode": "asym" if per_rank_vectors else "sym",
                "per_rank_vectors": per_rank_vectors,
                "threshold": threshold,
                "requested_capacity": num_max_tokens_per_rank,
                "aligned_capacity": {"baseline": base_buffer.num_max_tokens_per_rank, "candidate": candidate_buffer.num_max_tokens_per_rank},
                "operators": {"baseline": "deep_gemm.fp8_fp4_mega_moe", "candidate": "gc_moe.fp8_fp4_gc_moe"},
                "modules": {"baseline": deep_gemm.__file__, "candidate": gc_moe.__file__},
                "all_pass": all_pass,
                "results": results,
            }, f, indent=2)
        print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'} "
              f"(threshold={threshold}). Wrote {args.output_json}")

    candidate_buffer.destroy()
    base_buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
