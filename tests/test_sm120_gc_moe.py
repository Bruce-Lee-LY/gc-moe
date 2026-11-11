"""
SM120 accuracy comparison: GC MoE vs BF16 and processed references.

Compares the SM120 FP8/FP4 GC MoE kernel against a BF16 reference forward
pass and a quantization-exact processed reference, reporting cosine
similarity per token count and checking workspace reuse.

Highlights:
  * Cross-rank dispatch/combine happens entirely
    inside ``sm120_fp8_fp4_gc_moe`` via PyTorch symmetric memory.
  * Sweeps tokens per rank: 1, 2, 4, ..., 256.
  * DSV4.1 Flash (H5120/I2304/E384/topk6) by default.
  * Uses one rank per GPU; NUM_EXPERTS must be divisible by NUM_RANKS.

USAGE:
    torchrun --nproc_per_node=4 tests/test_sm120_gc_moe.py
    torchrun --nproc_per_node=8 tests/test_sm120_gc_moe.py
"""

import argparse
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

_REPO_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _REPO_ROOT)

import torch
import torch.distributed as dist

import gc_moe


DEFAULT_TOKENS_PER_RANK: List[int] = [
    1, 2, 4, 8, 16, 32, 64, 128, 256,
]
FP4_BLOCK = 32
E4M3_MAX = 448.0
FP4_CODES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


# ---------------------------------------------------------------------------
# Test fixtures and reference math for quantized inputs.
# ---------------------------------------------------------------------------

def cast_tokens_to_fp8(x: torch.Tensor):
    from deep_gemm.utils import per_token_cast_to_fp8

    if x.size(0) == 0:
        return x.to(torch.float8_e4m3fn), torch.empty(
            (0, x.size(1) // 128), dtype=torch.int32, device=x.device)
    return per_token_cast_to_fp8(
        x, use_ue8m0=True, gran_k=FP4_BLOCK, use_packed_ue8m0=True)


def cast_grouped_weights_to_fp4(weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    from deep_gemm.utils import per_token_cast_to_fp4

    experts, rows, k = weights.shape
    payload = torch.empty((experts, rows, k // 2), dtype=torch.int8, device=weights.device)
    codes = torch.empty((experts, rows, k // FP4_BLOCK), dtype=torch.uint8, device=weights.device)
    for expert in range(experts):
        payload[expert], scales = per_token_cast_to_fp4(
            weights[expert], use_ue8m0=True, gran_k=FP4_BLOCK)
        # SM120 consumes the UE8M0 bytes before its own weight layout transform.
        codes[expert] = (scales.contiguous().view(torch.int32) >> 23).to(torch.uint8)
    return payload, codes


def pack_ue8m0_weight_scales(codes: torch.Tensor) -> torch.Tensor:
    """Pack four per-row scale bytes into each int32 word."""
    padded = torch.nn.functional.pad(codes, (0, -codes.size(-1) % 4), value=127)
    return padded.contiguous().view(torch.int32)


def dequantize_mxfp4(packed: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """Exact real-arithmetic dequantization of the packed MXFP4 weights."""
    p = packed.view(torch.uint8)
    half_k = p.shape[-1]
    lo, hi = p & 15, p >> 4
    lut = torch.tensor(FP4_CODES, device=p.device)
    nib = torch.stack((lo, hi), dim=-1).reshape(*p.shape[:-1], half_k * 2)
    values = lut[(nib & 7).long()]
    values = torch.where((nib & 8) != 0, -values, values)
    scale = torch.exp2(codes.float() - 127.0)
    return (values * scale.repeat_interleave(32, dim=-1)).to(torch.bfloat16)


def dequantize_activation(fp8: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    tokens, hidden = fp8.shape
    # A packed word carries four K32 exponents, which may be identical.
    scale = torch.exp2(sf.contiguous().view(torch.uint8).reshape(tokens, hidden // 32).float() - 127.)
    return (fp8.float().reshape(tokens, hidden // 32, 32)
            * scale.unsqueeze(-1)).reshape(tokens, hidden)


def make_shared_input_sf(
    source: torch.Tensor, destination: torch.Tensor, block_m: int
) -> torch.Tensor:
    destination.zero_()
    tokens = source.size(0)
    if tokens == 0:
        return destination
    token_idx = torch.arange(tokens, dtype=torch.long, device=source.device)
    within_block = token_idx % block_m
    transformed = (
        token_idx // block_m * ((block_m + 127) // 128 * 128)
        + (within_block // 128) * 128
        + (within_block % 32) * 4
        + (within_block % 128) // 32
    )
    destination[transformed] = source
    return destination


def build_shared_weights(intermediate, hidden, device, *, seed=31337):
    g = torch.Generator(device=device).manual_seed(seed)
    return (torch.randn(2 * intermediate, hidden, dtype=torch.bfloat16,
                        device=device, generator=g) * .05,
            torch.randn(hidden, intermediate, dtype=torch.bfloat16,
                        device=device, generator=g) * .05)


def reference_shared_forward(x, weights, clamp, gran_k=None):
    if gran_k is None:
        w1, w2 = (w.float() for w in weights)
    else:
        w1, w2 = (data.float() * scale.repeat_interleave(data.shape[0] // scale.shape[0], 0).repeat_interleave(gran_k, 1)
                  for data, scale in weights)
    gate, up = (x.float() @ w1.T).to(torch.bfloat16).float().chunk(2, -1)
    gate, up = gate.clamp(max=clamp), up.clamp(-clamp, clamp)
    mid = gate * torch.sigmoid(gate) * up
    if gran_k is not None:
        blocks = mid.reshape(mid.shape[0], mid.shape[1] // 32, 32)
        scale = torch.exp2(torch.ceil(torch.log2((blocks.abs().amax(-1) / 448).clamp_min(1.e-10))))
        mid = ((blocks / scale[..., None]).to(torch.float8_e4m3fn).float()
               * scale[..., None]).reshape_as(mid)
    return (mid @ w2.T).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# BF16 reference (per-expert vectorization).
# ---------------------------------------------------------------------------

def reference_moe_forward(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    w1_bf16_full: torch.Tensor,
    w2_bf16_full: torch.Tensor,
    activation: str = "swiglu",
    activation_clamp: Optional[float] = None,
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
        if activation != "swiglu":
            raise ValueError(f"Unsupported activation: {activation}")
        mid = gate * torch.sigmoid(gate) * up
        l2 = mid @ w2_e.T
        wgts = topk_weights[rows, slots].float().unsqueeze(1)
        out.index_add_(0, rows, wgts * l2)
    return out.to(x.dtype)


def reference_processed_moe_forward(x_fp8, x_sf, indices, route_weights,
                                    packed_l1, codes_l1, packed_l2, codes_l2,
                                    intermediate, clamp):
    """Independent reference for BF16 FC1 rounding and BF16 routed return."""
    tokens, hidden = x_fp8.shape
    x = dequantize_activation(x_fp8, x_sf)
    out = torch.zeros((tokens, hidden), dtype=torch.float32, device=x.device)
    for e in range(packed_l1.shape[0]):
        rows, slots = (indices == e).nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        w1 = dequantize_mxfp4(packed_l1[e], codes_l1[e]).float()
        linear = x[rows] @ w1.T
        gate, up = linear.to(torch.bfloat16).float().split(intermediate, dim=-1)
        if clamp is not None:
            gate, up = gate.clamp(max=clamp), up.clamp(-clamp, clamp)
        mid = gate * torch.sigmoid(gate) * up * route_weights[rows, slots, None]
        # Kernel requantizes intermediate per (token, K32) to e4m3/UE8M0.
        grouped = mid.reshape(-1, intermediate // 32, 32)
        amax = grouped.abs().amax(-1)
        scale = torch.exp2(torch.ceil(torch.log2((amax / 448.).clamp_min(2 ** -22))))
        quant = (grouped / scale.unsqueeze(-1)).clamp(-E4M3_MAX, E4M3_MAX)
        restored = (quant.to(torch.float8_e4m3fn).float() * scale.unsqueeze(-1)).reshape(-1, intermediate)
        w2 = dequantize_mxfp4(packed_l2[e], codes_l2[e]).float()
        route = (restored @ w2.T).to(torch.bfloat16).float()
        out.index_add_(0, rows, route)
    return out.to(torch.bfloat16)


def dequantize_reference(pair, gran_k, fp4=False):
    """Decode the pre-layout values and UE8M0 exponents using only Torch."""
    values, packed_sf = pair
    sf = (packed_sf.contiguous().view(torch.uint8).to(torch.int32) << 23).view(torch.float32)
    if fp4:
        codes = torch.stack((values & 15, (values >> 4) & 15), dim=-1).flatten(-2)
        table = torch.tensor(FP4_CODES, device=values.device)
        decoded = table[(codes & 7).long()]
        decoded = torch.where((codes & 8) != 0, -decoded, decoded)
    else:
        decoded = values.float()
    columns = torch.arange(decoded.size(-1), device=values.device) // gran_k
    return decoded * sf[..., columns]


def reference_linear_pair(x, l1, l2, clamp, routing_weights=None):
    # Both kernels round the L1 accumulator to BF16 before SwiGLU and apply
    # routed weights before the per-token K32 FP8 intermediate quantization.
    if x.size(0) == 0:
        return torch.empty((0, l2.size(0)), dtype=torch.float32, device=x.device)
    gate, up = (x @ l1.T).to(torch.bfloat16).float().chunk(2, dim=-1)
    gate = gate.clamp(max=clamp)
    up = up.clamp(-clamp, clamp)
    mid = gate * torch.sigmoid(gate) * up
    if routing_weights is not None:
        mid = mid * routing_weights[:, None]
    groups = mid.reshape(mid.size(0), -1, FP4_BLOCK)
    amax = groups.abs().amax(dim=-1)
    scale = torch.exp2(torch.ceil(torch.log2((amax / 448.).clamp_min(1.e-10))))
    mid8 = (groups / scale[..., None]).to(torch.float8_e4m3fn)
    restored = (mid8.float() * scale[..., None]).reshape_as(mid)
    return (restored @ l2.T).to(torch.bfloat16).float()


def run_independent(output, routed_weights, shared_weights, x_pair, indices,
                    routing_weights, vector, rank, activation_clamp, shared_recipe):
    """Distributed external dequant reference; neither MoE operator is called."""
    world = len(vector)
    tokens, hidden = x_pair[0].shape
    x = dequantize_reference(x_pair, FP4_BLOCK)
    capacity = max(vector)
    def gather(tensor):
        padded = torch.zeros((capacity, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        padded[:tokens].copy_(tensor)
        peers = [torch.empty_like(padded) for _ in range(world)]
        dist.all_gather(peers, padded)
        return torch.cat([peer[:count] for peer, count in zip(peers, vector)])
    all_x, all_idx, all_weights = gather(x), gather(indices), gather(routing_weights)
    routed_full = torch.zeros_like(all_x)
    local_experts = routed_weights[0][0].size(0)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for expert in range(local_experts):
            rows, slots = (all_idx == rank * local_experts + expert).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            l1, l2 = [dequantize_reference((weight[expert], sf[expert]), 32, fp4=True)
                      for weight, sf in routed_weights]
            contribution = reference_linear_pair(all_x[rows], l1, l2, activation_clamp,
                                                  all_weights[rows, slots])
            routed_full.index_add_(0, rows, contribution)
        dist.all_reduce(routed_full)
        start = sum(vector[:rank])
        routed = routed_full[start:start + tokens]
        shared = None
        if shared_weights is not None:
            l1, l2 = [dequantize_reference(pair, shared_recipe[2]) for pair in shared_weights]
            shared = reference_linear_pair(x, l1, l2, activation_clamp)
        output.copy_((routed if shared is None else routed + shared).to(torch.bfloat16))
        return routed, shared
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0:
        return 1.0 if b.numel() == 0 else 0.0
    a = a.flatten().float()
    b = b.flatten().float()
    denom = a.norm() * b.norm()
    if denom.item() == 0:
        return float(torch.equal(a, b))
    return float((a @ b / denom).item())


def diff(actual, reference):
    x, y = actual.double(), reference.double()
    denominator = (x.square() + y.square()).sum()
    if denominator.item() == 0:
        return 0.0
    return float((1 - 2 * (x * y).sum() / denominator).item())


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def build_full_weights(
    num_experts: int, intermediate: int, hidden: int, device: torch.device, *, seed=7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    w1_full = torch.randn(
        num_experts, 2 * intermediate, hidden,
        dtype=torch.bfloat16, device=device, generator=g,
    ) * 0.05
    w2_full = torch.randn(
        num_experts, hidden, intermediate,
        dtype=torch.bfloat16, device=device, generator=g,
    ) * 0.05
    return w1_full, w2_full


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

def main(*, shared=False, readiness=False):
    from deep_gemm.utils import per_block_cast_to_fp8, per_token_cast_to_fp4, per_token_cast_to_fp8

    parser = argparse.ArgumentParser()
    parser.add_argument('--compare-cake', action='store_true',
                        help='Compare with FlashInfer Cake using the same prequantized inputs.')
    parser.add_argument('--require-cake-exact', action='store_true',
                        help='Require zero absolute output difference on every Cake comparison.')
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--intermediate-hidden", type=int, default=2304)
    parser.add_argument("--num-experts", type=int, default=384)
    parser.add_argument("--num-topk", type=int, default=6)
    parser.add_argument('--case-category', choices=('default', 'boundary', 'asymmetric'), default='default')
    parser.add_argument('--routing-pattern', choices=('random', 'rank-hotspot', 'zero-weight'),
                        default='random', help='Exercise full receive capacity or exact-zero BF16 returns.')
    parser.add_argument("--num-tokens-per-rank-list", default="",
                        help="Comma-separated list of per-rank token counts. "
                             "Empty (default) uses 1,2,4,8,16,32,64,128,256.")
    parser.add_argument("--num-max-tokens-per-rank", type=int, default=256 if readiness else 0)
    parser.add_argument('--per-rank-tokens-list', default='',
                        help='Semicolon-separated rank token vectors, as in the SM100 tests.')
    parser.add_argument('--num-tokens-per-rank-vector', default='',
                        help='One asymmetric count per rank, including zero; one test case.')
    parser.add_argument("--activation", choices=("swiglu",), default="swiglu")
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--fast-math", type=int, default=1)
    parser.add_argument("--cosine-threshold", type=float, default=None,
                        help="Override the raw BF16 gate. Default is 0.97 for non-SE; SE uses the processed gate.")
    parser.add_argument("--processed-cosine-threshold", type=float, default=None,
                        help="Separate threshold against the quantized-input reference; "
                             "defaults to --cosine-threshold for compatibility.")
    parser.add_argument("--gc-replays", "--steps", type=int, default=200 if readiness else 5)
    if readiness:
        parser.add_argument("--mode", choices=("eager", "graph"), default="graph")
    parser.add_argument('--shared-expert', action='store_true')
    parser.add_argument('--num-shared-experts', type=int, default=int(shared))
    parser.add_argument('--shared-weight-gran-k', type=int, choices=(32, 128))
    parser.add_argument('--shared-weight-block-n', type=int, choices=(1, 32, 128))
    parser.add_argument("--output-json", type=str, default=("results/_accuracy_results_sm120_gc_se.json" if shared
                                 else "results/_accuracy_results_sm120_gc.json"))
    if readiness:
        parser.set_defaults(output_json="results/_readiness_results_sm120_gc.json")
    args = parser.parse_args()
    if args.require_cake_exact:
        args.compare_cake = True
    if readiness and (args.gc_replays < 1 or args.shared_expert or args.num_shared_experts):
        parser.error('routed readiness requires positive --steps and zero shared experts')
    if args.num_shared_experts < int(shared):
        parser.error(f'--num-shared-experts must be at least {int(shared)}')
    args.num_shared_experts = args.num_shared_experts or int(args.shared_expert)
    args.shared_expert = bool(args.num_shared_experts)
    args.shared_weight_gran_k = args.shared_weight_gran_k or (128 if args.hidden == 6144 else 32)
    shared_n = args.shared_weight_block_n or args.shared_weight_gran_k
    if (shared_n, args.shared_weight_gran_k) not in ((1, 32), (32, 32), (128, 128)):
        parser.error('unsupported shared block N/K')
    if args.compare_cake and (readiness or args.num_shared_experts > 1
                              or not args.fast_math or not args.activation_clamp.is_integer()):
        parser.error('Cake comparison requires an accuracy run with at most one shared expert, '
                     'fast math and an integer clamp')

    sweep = ([64] if readiness else DEFAULT_TOKENS_PER_RANK)
    if args.num_tokens_per_rank_list:
        sweep = [int(t) for t in args.num_tokens_per_rank_list.split(",") if t.strip()]
    if args.per_rank_tokens_list and args.num_tokens_per_rank_vector:
        parser.error('provide only one asymmetric vector option')
    vectors = [[int(t) for t in vec.split(',') if t.strip()]
               for vec in (args.per_rank_tokens_list or args.num_tokens_per_rank_vector).split(';') if vec.strip()]
    if vectors and (args.num_tokens_per_rank_list or any(not v or min(v) < 0 for v in vectors)):
        parser.error('asymmetric vectors require nonnegative counts and no token sweep')
    num_max_tokens_per_rank = args.num_max_tokens_per_rank or max(1, max(map(max, vectors)) if vectors else max(sweep))

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    plan = vectors or [[t] * num_ranks for t in sweep]
    if any(len(v) != num_ranks or min(v) < 0 or max(v) > num_max_tokens_per_rank for v in plan):
        raise ValueError('token counts must fit the rank count and capacity')

    if rank == 0:
        print(f"[tests/test_sm120_gc_moe] world_size={num_ranks} device={device}")

    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) != (12, 0):
        if rank == 0:
            print(f"Skipping: SM120 only, got SM{major}{minor}")
        dist.destroy_process_group()
        raise RuntimeError("SM120 hardware is required; this is not a passed test")

    if args.num_experts % num_ranks != 0:
        if rank == 0:
            print(f"Skipping: num_experts={args.num_experts} not divisible by "
                  f"num_ranks={num_ranks}")
        dist.destroy_process_group()
        return

    if rank == 0:
        print(f"Config: hidden={args.hidden} intermediate={args.intermediate_hidden} "
              f"experts={args.num_experts} topk={args.num_topk} "
              f"clamp={args.activation_clamp} fast_math={bool(args.fast_math)} "
              f"num_max_tokens_per_rank={num_max_tokens_per_rank} sweep={sweep}")

    caster_source = Path(inspect.getfile(per_token_cast_to_fp4))
    Eloc = args.num_experts // num_ranks
    h, i, e = args.hidden, args.intermediate_hidden, args.num_experts
    if args.shared_expert:
        # SM100 SE generates each expert shard with seed 7 + owner.
        w1_full = torch.empty((e, 2 * i, h), dtype=torch.bfloat16, device=device)
        w2_full = torch.empty((e, h, i), dtype=torch.bfloat16, device=device)
        for owner in range(num_ranks):
            w1, w2 = build_full_weights(Eloc, i, h, device, seed=7 + owner)
            w1_full[owner * Eloc:(owner + 1) * Eloc].copy_(w1)
            w2_full[owner * Eloc:(owner + 1) * Eloc].copy_(w2)
        del w1, w2
    else:
        w1_full, w2_full = build_full_weights(e, i, h, device)
    local = slice(rank * Eloc, (rank + 1) * Eloc)
    reference_routed = tuple(cast_grouped_weights_to_fp4(w[local]) for w in (w1_full, w2_full))
    packed_routed = tuple((payload, codes.view(torch.int32).transpose(-2, -1).contiguous().transpose(-2, -1))
                          for payload, codes in reference_routed)
    candidate_weights = gc_moe.transform_weights_for_gc_moe_sm120(*packed_routed, activation=args.activation)
    del packed_routed
    shared_kwargs = {}
    shared_recipe = (1, 128 if args.shared_weight_gran_k == 128 else 1, args.shared_weight_gran_k)
    shared_reference = None
    shared_input = None
    if args.shared_expert:
        shared_raw = build_shared_weights(i * args.num_shared_experts, h, device, seed=11)
        cast_shared = per_token_cast_to_fp8 if shared_n == 1 else per_block_cast_to_fp8
        shared_quant = tuple(cast_shared(w, use_ue8m0=True, gran_k=args.shared_weight_gran_k)
                             for w in shared_raw)
        shared_reference = tuple(
            (data, (scale.contiguous().view(torch.int32) >> 23).to(torch.uint8)
                   .repeat_interleave(shared_n, dim=0)[:data.size(0)])
            for data, scale in shared_quant)
        shared_input = shared_quant
        if shared_n != shared_recipe[1]:
            shared_input = tuple((data, scale.repeat_interleave(shared_n, dim=0))
                                 for data, scale in shared_quant)
        packed_shared = tuple((data, pack_ue8m0_weight_scales(codes))
                              for data, codes in shared_reference)
        w1, w2 = gc_moe.transform_weights_for_gc_moe_sm120(
            *packed_shared, shared_recipe=shared_recipe)
        shared_kwargs = dict(shared_l1_weights=w1, shared_l2_weights=w2, shared_recipe=shared_recipe)

    cake_layer = None
    if args.compare_cake:
        from benchmarks.benchmark_sm120_gc_moe import _make_cake_layer
        from flashinfer.moe_ep import MoEEpTensors
        from flashinfer.moe_ep.kernel_src.sm120.split_cutedsl_megakernel import (
            select_graph_compile_bucket,
        )

        cake_routed = tuple(w.view(torch.float4_e2m1fn_x2) for w, _ in reference_routed) + tuple(
            sf.view(torch.float8_e8m0fnu) for _, sf in reference_routed)
        cake_layer = _make_cake_layer(
            rank, num_ranks, experts=e, capacity=num_max_tokens_per_rank,
            hidden=h, intermediate=i, topk=args.num_topk, device=device,
            swiglu_limit=int(args.activation_clamp), transport_table='all_lsa',
            routed=cake_routed, shared=shared_input, shared_gran_k=args.shared_weight_gran_k)
        del cake_routed

    candidate_buffer = gc_moe.get_symm_buffer_for_gc_moe_sm120(
        dist.group.WORLD, args.num_experts, num_max_tokens_per_rank,
        args.num_topk, args.hidden, args.intermediate_hidden,
        activation=args.activation, num_shared_experts=args.num_shared_experts,
    )
    if rank == 0:
        print(f"Symmetric buffer: GC {candidate_buffer.buffer.nbytes / 2 ** 30:.3f} GiB")

    # Check the host contract without submitting an invalid pointer to CUDA.
    from gc_moe.apis.sm120_gc_moe import validate_call
    invalid_y = torch.empty(args.hidden + 1, dtype=torch.bfloat16, device=device)[1:].view(1, args.hidden)
    assert invalid_y.is_contiguous() and invalid_y.data_ptr() % 16 != 0
    try:
        validate_call(invalid_y, *candidate_weights, candidate_buffer, None,
                      (1, 1, FP4_BLOCK), args.activation, args.activation_clamp,
                      bool(args.fast_math), 1)
    except ValueError:
        pass
    else:
        raise AssertionError('SM120 validation accepted a misaligned output')

    recv_stats = torch.zeros(Eloc, dtype=torch.int32, device=device)
    results = []
    all_pass = True
    threshold = args.cosine_threshold
    if threshold is None and not args.shared_expert:
        threshold = .97
    processed_threshold = args.processed_cosine_threshold
    if processed_threshold is None:
        processed_threshold = .99 if args.shared_expert else .97

    for vector in plan:
        T = vector[rank]
        global_bound = max(1, max(vector))
        recv_stats.zero_()
        x_bf16, topk_idx, topk_w = build_tokens(
            T, args.hidden, args.num_experts, args.num_topk, rank, device,
        )
        if args.routing_pattern == 'rank-hotspot':
            if args.num_topk > Eloc:
                raise ValueError('rank-hotspot requires top-k <= experts per rank')
            topk_idx.copy_(torch.arange(args.num_topk, device=device).expand(T, -1))
        elif args.routing_pattern == 'zero-weight':
            topk_w.zero_()
        ref = reference_moe_forward(
            x_bf16, topk_idx, topk_w, w1_full, w2_full,
            activation=args.activation, activation_clamp=args.activation_clamp,
        )
        x_fp8, x_sf = cast_tokens_to_fp8(x_bf16)
        candidate_buffer.x[:T].copy_(x_fp8)
        candidate_buffer.x_sf[:T].copy_(x_sf)
        if args.shared_expert:
            block_m = gc_moe.get_block_m_for_gc_moe(
                num_ranks, args.num_experts, candidate_buffer.num_max_tokens_per_rank, T, args.num_topk)
            make_shared_input_sf(x_sf, candidate_buffer.shared_l1_acts_sf, block_m)
        candidate_buffer.topk_idx[:T].copy_(topk_idx)
        candidate_buffer.topk_weights[:T].copy_(topk_w)
        torch.cuda.synchronize(device)
        dist.barrier()

        def independent(payload, sf, x_bf16=x_bf16, topk_idx=topk_idx, topk_w=topk_w):
            out = torch.empty_like(x_bf16)
            routed, shared = run_independent(out, reference_routed, shared_reference,
                (payload, sf), topk_idx, topk_w, vector, rank, args.activation_clamp, shared_recipe)
            return out, routed, shared

        processed_ref, routed_ref, shared_ref = independent(x_fp8, x_sf)
        if args.shared_expert:
            ref = (ref.float() + reference_shared_forward(x_bf16, shared_raw,
                args.activation_clamp).float()).to(torch.bfloat16)

        if readiness:
            guarded = torch.full((T + 2, args.hidden), 123., dtype=torch.bfloat16, device=device)
            y = guarded[1:-1]
        else:
            y = torch.empty((T, args.hidden), dtype=torch.bfloat16, device=device)

        def candidate_call(y=y):
            gc_moe.sm120_fp8_fp4_gc_moe(
                y, *candidate_weights, candidate_buffer,
                recipe=(1, 1, FP4_BLOCK), activation=args.activation,
                activation_clamp=args.activation_clamp,
                fast_math=bool(args.fast_math), num_tokens_bound=global_bound,
                cumulative_local_expert_recv_stats=recv_stats, **shared_kwargs)

        candidate_call()
        torch.cuda.synchronize(device)
        first_eager = y.clone()
        expected = torch.bincount(topk_idx.flatten(), minlength=args.num_experts).to(torch.int32)
        dist.all_reduce(expected)
        expected = expected[rank * Eloc:(rank + 1) * Eloc]
        stats_ok = torch.equal(recv_stats, expected)
        changing_cos = 1.
        changing_stable = True
        if not readiness:
            for _ in range(args.gc_replays):
                candidate_call()

        replay = candidate_call
        graph = None
        if not readiness or args.mode == "graph":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                candidate_call()
            replay = graph.replay
        if not readiness:
            for _ in range(args.gc_replays):
                replay()
            torch.cuda.synchronize(device)

        if args.shared_expert or readiness:
            alt_x, alt_sf = cast_tokens_to_fp8(-x_bf16.roll(1, 0))
            # Reverse token order for routing as well. Receive counts stay
            # the same, so each replay can check the exact stats increment.
            alt_idx = topk_idx.flip(0) if readiness else topk_idx
            alt_w = topk_w.flip(0) if readiness else topk_w
            alt_ref, _, _ = independent(alt_x, alt_sf, topk_idx=alt_idx, topk_w=alt_w)

            def stage(x, sf, ids, weights):
                candidate_buffer.x[:T].copy_(x)
                candidate_buffer.x_sf[:T].copy_(sf)
                if args.shared_expert:
                    make_shared_input_sf(sf, candidate_buffer.shared_l1_acts_sf, block_m)
                candidate_buffer.topk_idx[:T].copy_(ids)
                candidate_buffer.topk_weights[:T].copy_(weights)

            stage(alt_x, alt_sf, alt_idx, alt_w)
            replay()
            torch.cuda.synchronize(device)
            alt_output = y.clone()
            changing_cos = cosine(y, alt_ref)
            inputs = ((x_fp8, x_sf, topk_idx, topk_w, first_eager),
                      (alt_x, alt_sf, alt_idx, alt_w, alt_output))
            # Finish with the original input so the common accuracy checks
            # below still compare against the original independent reference.
            for step in range(args.gc_replays + 1):
                index = step % 2 if step < args.gc_replays else 0
                x, sf, ids, weights, expected_y = inputs[index]
                stage(x, sf, ids, weights)
                if readiness:
                    recv_stats.zero_()
                replay()
                torch.cuda.synchronize(device)
                changing_stable &= torch.equal(y.view(torch.int16), expected_y.view(torch.int16))
                if readiness:
                    stats_ok &= torch.equal(recv_stats, expected)
                    changing_stable &= bool((guarded[0] == 123.).all() and (guarded[-1] == 123.).all())
            del stage, inputs
        changing = torch.tensor([changing_cos, float(changing_stable)],
                                dtype=torch.float64, device=device)
        dist.all_reduce(changing, op=dist.ReduceOp.MIN)
        del graph, replay

        repeat_diff = diff(y, first_eager)
        cos = cosine(y, ref)
        cos_processed = cosine(y, processed_ref)
        shared_cos = cosine(candidate_buffer.shared_y[:T], shared_ref) if args.shared_expert else 1.
        shared_residual_cos = cosine(y.float() - routed_ref, shared_ref) if args.shared_expert else 1.
        shared_min = torch.tensor([shared_cos, shared_residual_cos], dtype=torch.float64, device=device)
        dist.all_reduce(shared_min, op=dist.ReduceOp.MIN)
        cos_t = torch.tensor([cos, cos_processed], dtype=torch.float64, device=device)
        cos_min_t = cos_t.clone()
        dist.all_reduce(cos_min_t, op=dist.ReduceOp.MIN)
        finite = torch.tensor([int(torch.isfinite(y).all())], device=device)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        zero_ok = torch.tensor([int(args.routing_pattern != 'zero-weight' or args.shared_expert
                                    or not bool(torch.count_nonzero(y)))], device=device)
        dist.all_reduce(zero_ok, op=dist.ReduceOp.MIN)
        stats_flag = torch.tensor([int(stats_ok)], device=device)
        dist.all_reduce(stats_flag, op=dist.ReduceOp.MIN)
        cos_min = float(cos_min_t[0].item())
        cos_processed_min = float(cos_min_t[1].item())
        ok = ((threshold is None or cos_min >= threshold) and cos_processed_min >= processed_threshold
              and float(shared_min[0]) >= .99999 and float(shared_min[1]) >= processed_threshold
              and float(changing[0]) >= processed_threshold and bool(changing[1])
              and bool(finite.item()) and bool(stats_flag.item()) and bool(zero_ok.item())
              and repeat_diff <= 1e-6)
        cake = None
        if cake_layer is not None and any(vector):
            tensors = MoEEpTensors(
                hidden_states=x_fp8, topk_ids=topk_idx, topk_weights=topk_w,
                scales=x_sf.view(torch.uint8).view(T, h // 32).view(torch.float8_e8m0fnu))
            cake_layer.stage_inputs(tensors, compile_tokens_per_rank=select_graph_compile_bucket(
                global_bound, num_max_tokens_per_rank))
            cake_y = torch.full_like(y, float('nan'))
            cake_layer.compute_staged(output=cake_y)
            torch.cuda.synchronize(device)
            local_cake = dict(
                raw_cosine=cosine(cake_y, ref),
                processed_cosine=cosine(cake_y, processed_ref),
                gc_cosine=cosine(cake_y, y),
                shared_residual_cosine=cosine(cake_y.float() - routed_ref, shared_ref)
                if args.shared_expert else 1.,
                gc_max_abs=float((cake_y.float() - y.float()).abs().max()) if T else 0.,
                finite=bool(torch.isfinite(cake_y).all()),
                protocol_error=cake_layer._kernel.protocol_error(cake_layer._workspace),
            )
            local_cake['passed'] = (
                local_cake['finite'] and local_cake['protocol_error'] == 0
                and (threshold is None or local_cake['raw_cosine'] >= threshold)
                and min(local_cake[k] for k in ('processed_cosine', 'gc_cosine',
                                               'shared_residual_cosine')) >= processed_threshold
                and (not args.require_cake_exact or local_cake['gc_max_abs'] == 0.))
            cake = [None] * num_ranks
            dist.all_gather_object(cake, local_cake)
            ok = ok and all(row['passed'] for row in cake)
            del tensors, cake_y
        all_pass = all_pass and ok
        output_sha256 = [None] * num_ranks
        digest = hashlib.sha256(y.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        dist.all_gather_object(output_sha256, digest)
        if rank == 0:
            status = "OK" if ok else "FAIL"
            print(f"  tokens_per_rank={T:4d} total_tokens={sum(vector):5d} "
                  f"cos_min={cos_min:.4f} cos_processed_min={cos_processed_min:.4f} "
                  f"stats_exact={bool(stats_flag.item())} [{status}]")
        results.append({
            "num_tokens_per_rank": T,
            "cos_min": cos_min,
            "cos_processed_min": cos_processed_min,
            "repeat_diff": repeat_diff,
            "recv_stats_exact": bool(stats_flag.item()),
            "output_sha256_per_rank": output_sha256,
            "finite": bool(finite.item()),
            "zero_weight_exact": bool(zero_ok.item()),
            "gc_replays_eager_and_graph": args.gc_replays if not readiness else 0,
            **({"readiness_mode": args.mode, "readiness_steps": args.gc_replays} if readiness else {}),
            "passed": ok,
            "vector": vector,
            "shared_cos_min": float(shared_min[0]),
            "sm100_shared_residual_cos_min": float(shared_min[1]),
            "changing_input_cos_min": float(changing[0]),
            "changing_input_bitwise_stable": bool(changing[1]),
            **({"cake_by_rank": cake} if args.compare_cake else {}),
        })
        del independent, candidate_call
        del ref, processed_ref, first_eager, y, x_bf16, x_fp8, x_sf, topk_idx, topk_w
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
                    "activation_clamp": args.activation_clamp,
                    "fast_math": bool(args.fast_math),
                    "num_max_tokens_per_rank": num_max_tokens_per_rank,
                    "shared_expert": args.shared_expert,
                    "num_shared_experts": args.num_shared_experts,
                    "shared_weight_gran_k": args.shared_weight_gran_k,
                    "shared_weight_block_n": shared_n,
                    "num_tokens_per_rank_vector": vectors[0] if len(vectors) == 1 else [],
                    "per_rank_token_plan": plan,
                    "accuracy_fixture": "sm100",
                    "case_category": args.case_category,
                    "routing_pattern": args.routing_pattern,
                    "compare_cake": args.compare_cake,
                },
                "threshold": threshold,
                "processed_threshold": processed_threshold,
                "operator": "gc_moe.sm120_fp8_fp4_gc_moe",
                "quantizer": {"source": str(caster_source),
                              "sha256": hashlib.sha256(caster_source.read_bytes()).hexdigest()},
                "module": gc_moe.__file__,
                "all_pass": all_pass,
                "sources": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                            for name in (
                                'gc_moe/impls/sm120_fp8_fp4_gc_moe.py',
                                'gc_moe/jit_kernels/impls/sm120_fp8_fp4_gc_moe.py',
                                'gc_moe/jit_kernels/heuristics/sm120_gc_moe.py',
                                'gc_moe/layout/sm120_gc_moe.py',
                                'gc_moe/runtime/sm120_gc_moe.py',
                                'gc_moe/apis/sm120_gc_moe.py',
                                'gc_moe/gc/__init__.py',
                                'gc_moe/sm120_gc/__init__.py',
                                'gc_moe/mma/sm120.py',
                                'gc_moe/common/math.py',
                                'gc_moe/ptx/tma.py',
                                'gc_moe/ptx/ld_st.py',
                                'gc_moe/ptx/utils.py',
                                'tests/test_sm120_gc_moe.py')},
                "environment": {name: value for name, value in os.environ.items()
                                if name.startswith('GC_MOE_')},
                "results": results,
            }, f, indent=2)
        print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'} "
              f"(threshold={threshold}). Wrote {args.output_json}")

    if cake_layer is not None:
        cake_layer.destroy()
    candidate_buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
