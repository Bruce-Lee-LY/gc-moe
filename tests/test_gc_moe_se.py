"""Multi-rank Mega MoE and GC MoE SE accuracy against dequantized math."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import torch
import torch.distributed as dist
import deep_gemm
import gc_moe
assert Path(gc_moe.__file__).resolve().is_relative_to(_ROOT), gc_moe.__file__
from typing import Optional, Tuple

from deep_gemm.utils import align, per_block_cast_to_fp8, per_token_cast_to_fp4
from deep_gemm.utils import per_token_cast_to_fp8
from test_gc_moe import build_tokens, cast_tokens_to_fp8


DEFAULT_TOKENS_PER_RANK = [1, 2, 4, 8, 16, 32, 64, 128, 256]

def cosine(a, b):
    if a.numel() == 0:
        return 1.0 if b.numel() == 0 else 0.0
    a, b = a.flatten().float(), b.flatten().float()
    denominator = a.norm() * b.norm()
    if denominator.item() == 0:
        return float(torch.equal(a, b))
    return float((a @ b / denominator).item())


ACTIVATION_GRAN_K = 32
ROUTED_RECIPE = (1, 1, 32)
SHARED_RECIPE = (1, 128, 128)


def cast_grouped_weights_to_fp4(
    weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    groups, n, k = weights.shape
    quantized = torch.empty(
        (groups, n, k // 2), dtype=torch.int8, device=weights.device
    )
    scales = torch.empty(
        (groups, n, k // ACTIVATION_GRAN_K), dtype=torch.float, device=weights.device
    )
    for group_idx in range(groups):
        quantized[group_idx], scales[group_idx] = per_token_cast_to_fp4(
            weights[group_idx], use_ue8m0=True, gran_k=ACTIVATION_GRAN_K
        )
    scales = deep_gemm.transform_sf_into_required_layout(
        scales, n, k, ROUTED_RECIPE[1:], groups
    )
    return quantized, scales


def cast_shared_weight_to_fp8(
    weight: torch.Tensor, weight_recipe=SHARED_RECIPE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n, k = weight.shape
    if weight_recipe == (1, 1, 32):
        quantized, scales = per_token_cast_to_fp8(weight, use_ue8m0=True, gran_k=32)
    else:
        assert weight_recipe in ((1, 32, 32), (1, 128, 128))
        quantized, scales = per_block_cast_to_fp8(
            weight, use_ue8m0=True, gran_k=weight_recipe[2]
        )
    scale_recipe = weight_recipe[1:]
    if weight_recipe == (1, 32, 32):
        block_scales = scales
        scales = block_scales.repeat_interleave(32, dim=0)[:n]
        assert torch.equal(scales[::32].contiguous().view(torch.uint8),
                           block_scales.contiguous().view(torch.uint8))
        scale_recipe = (1, 32)
    scales = deep_gemm.transform_sf_into_required_layout(
        scales, n, k, scale_recipe, num_groups=None
    )
    return quantized, scales


def build_local_weights(
    local_experts: int,
    intermediate: int,
    hidden: int,
    rank: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(7 + rank)
    l1 = torch.randn(
        (local_experts, 2 * intermediate, hidden),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ) * 0.05
    l2 = torch.randn(
        (local_experts, hidden, intermediate),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ) * 0.05
    return l1, l2


def build_shared_weights(
    intermediate: int,
    hidden: int,
    num_shared_experts: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    shared_intermediate = intermediate * num_shared_experts
    generator = torch.Generator(device=device).manual_seed(11)
    l1 = torch.randn(
        (2 * shared_intermediate, hidden),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ) * 0.05
    l2 = torch.randn(
        (hidden, shared_intermediate),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ) * 0.05
    return l1, l2


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
        token_idx // block_m * align(block_m, 128)
        + (within_block // 128) * 128
        + (within_block % 32) * 4
        + (within_block % 128) // 32
    )
    destination[transformed] = source
    return destination


def stage_inputs(
    buffer,
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    shared_sf: Optional[torch.Tensor] = None,
) -> None:
    tokens = x_fp8.size(0)
    buffer.x[:tokens].copy_(x_fp8)
    buffer.x_sf[:tokens].copy_(x_sf)
    if shared_sf is not None:
        buffer.shared_l1_acts_sf.copy_(shared_sf)
    buffer.topk_idx[:tokens].copy_(topk_idx)
    buffer.topk_weights[:tokens].copy_(topk_weights)


def dequantize_reference(pair, gran_k, fp4=False):
    """Decode the pre-layout values and UE8M0 exponents using only Torch."""
    values, packed_sf = pair
    sf = (packed_sf.contiguous().view(torch.uint8).to(torch.int32) << 23).view(torch.float32)
    if fp4:
        codes = torch.stack((values & 15, (values >> 4) & 15), dim=-1).flatten(-2)
        table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=values.device)
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
    groups = mid.reshape(mid.size(0), -1, ACTIVATION_GRAN_K)
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
    x = dequantize_reference(x_pair, ACTIVATION_GRAN_K)
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
        l1, l2 = [dequantize_reference(pair, shared_recipe[2]) for pair in shared_weights]
        shared = reference_linear_pair(x, l1, l2, activation_clamp)
        output.copy_((routed + shared).to(torch.bfloat16))
        return routed, shared
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def parse_per_rank_vectors(value):
    return [[int(t) for t in v.split(',') if t.strip()] for v in value.split(';') if v.strip()]


def build_weights(args, rank, world, device, reference=False):
    raw = build_local_weights(args.num_experts // world, args.intermediate_hidden, args.hidden, rank, device)
    quantized = tuple(cast_grouped_weights_to_fp4(w) for w in raw)
    base_routed = deep_gemm.transform_weights_for_mega_moe(*quantized, activation=args.activation)
    gc_routed = gc_moe.transform_weights_for_gc_moe(*quantized, activation=args.activation)
    del raw
    raw = build_shared_weights(args.intermediate_hidden, args.hidden, args.num_shared_experts, device)
    recipe = (1, 128, 128) if args.shared_weight_gran_k == 128 else (1, 1, 32)
    weight_recipe = (1, args.shared_weight_gran_k, args.shared_weight_gran_k)
    shared = tuple(cast_shared_weight_to_fp8(w, weight_recipe) for w in raw)
    base_shared = deep_gemm.transform_weights_for_mega_moe(*shared, activation=args.activation)
    gc_shared = gc_moe.transform_weights_for_gc_moe(*shared, activation=args.activation)
    del raw
    result = base_routed, gc_routed, shared, base_shared, gc_shared, recipe
    return (*result, quantized) if reference else result


def allocate_buffer(args, capacity, operator, shared=True, base=None):
    factory = gc_moe.get_symm_buffer_for_gc_moe if operator == 'gc' else deep_gemm.get_symm_buffer_for_mega_moe
    kwargs = dict(num_shared_experts=args.num_shared_experts if shared else 0,
                  mma_type='fp8xfp4', activation=args.activation)
    if base is not None:
        return gc_moe.GCSymmBuffer(dist.group.WORLD, args.num_experts, capacity, args.num_topk,
                                     args.hidden, args.intermediate_hidden, base=base, **kwargs)
    return factory(dist.group.WORLD, args.num_experts, capacity, args.num_topk,
                   args.hidden, args.intermediate_hidden, **kwargs)


def prepare_shared_sf(args, x_sf, buffer, tokens, world, operator):
    query = gc_moe.get_block_m_for_gc_moe if operator == 'gc' else deep_gemm.get_block_m_for_mega_moe
    bm = query(world, args.num_experts, buffer.num_max_tokens_per_rank, tokens, args.num_topk, 'fp8xfp4')
    return make_shared_input_sf(x_sf, torch.zeros_like(buffer.shared_l1_acts_sf), bm), bm

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hidden', type=int, default=6144)
    p.add_argument('--intermediate-hidden', type=int, default=2048)
    p.add_argument('--num-experts', type=int, default=256)
    p.add_argument('--num-topk', type=int, default=8)
    p.add_argument('--num-shared-experts', type=int, default=1)
    p.add_argument('--shared-weight-gran-k', type=int, choices=[32, 128], default=128)
    p.add_argument('--omit-shared-recipe', action='store_true')
    p.add_argument('--case-category', choices=['default', 'boundary', 'asymmetric'], default='default')
    p.add_argument('--num-tokens-per-rank-list', default='')
    p.add_argument('--per-rank-tokens-list', default='')
    p.add_argument('--num-max-tokens-per-rank', type=int, default=0)
    p.add_argument('--activation', choices=['swiglu'], default='swiglu')
    p.add_argument('--activation-alpha', type=float, default=1.0)
    p.add_argument('--activation-clamp', type=float, default=10.0)
    p.add_argument('--fast-math', type=int, default=1)
    p.add_argument('--cosine-threshold', type=float, default=0.99,
                   help='Uniform minimum cosine for independent math, GC-to-Mega and omitted-vs-explicit recipe checks (0 to 1).')
    p.add_argument('--gc-replays', type=int, default=5)
    p.add_argument('--all-routes-invalid', action='store_true')
    p.add_argument('--cross-operator-buffer', action='store_true')
    p.add_argument('--output-json', default='results/_accuracy_results_gc_se.json')
    args = p.parse_args()
    if not 0.0 <= args.cosine_threshold <= 1.0:
        p.error('--cosine-threshold must be between 0 and 1 for all cosine checks')
    assert args.num_shared_experts >= 1
    assert args.gc_replays >= 5
    assert not args.omit_shared_recipe or args.shared_weight_gran_k == 32
    vectors = parse_per_rank_vectors(args.per_rank_tokens_list)
    sweep = [int(t) for t in args.num_tokens_per_rank_list.split(',') if t.strip()] or DEFAULT_TOKENS_PER_RANK
    capacity = args.num_max_tokens_per_rank or max(1, max(max(v) for v in vectors) if vectors else max(sweep))
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    assert torch.cuda.get_device_capability(device) in [(10, 0), (10, 3)]
    assert args.num_experts % world == 0, 'num_experts must be divisible by world_size'
    plan = vectors or [[t] * world for t in sweep]
    assert all(len(v) == world and min(v) >= 0 and max(v) <= capacity for v in plan)
    mr, gr, original_shared, ms, gs, recipe, reference_routed = build_weights(args, rank, world, device, reference=True)
    base_buffer = allocate_buffer(args, capacity, 'base')
    gc_buffer = allocate_buffer(args, capacity, 'gc', base=base_buffer if args.cross_operator_buffer else None)
    if args.cross_operator_buffer:
        for name in gc_buffer._views:
            left, right = getattr(gc_buffer, name), getattr(base_buffer, name)
            assert (left is None and right is None) or (left is not None and right is not None
                    and left.data_ptr() == right.data_ptr() and left.shape == right.shape
                    and left.stride() == right.stride()), name
    common = dict(recipe=ROUTED_RECIPE, activation=args.activation,
                  activation_clamp=args.activation_clamp, activation_alpha=args.activation_alpha,
                  fast_math=bool(args.fast_math))
    results = []
    for index, vector in enumerate(plan):
        tokens = vector[rank]
        x, indices, routing_weights = build_tokens(tokens, args.hidden, args.num_experts, args.num_topk, rank, device)
        if args.all_routes_invalid:
            indices.fill_(-1)
        x8, xs = cast_tokens_to_fp8(x)
        base_sf, base_bm = prepare_shared_sf(args, xs, base_buffer, tokens, world, 'base')
        gc_sf, gc_bm = prepare_shared_sf(args, xs, gc_buffer, tokens, world, 'gc')
        stage_inputs(base_buffer, x8, xs, indices, routing_weights, base_sf)
        stage_inputs(gc_buffer, x8, xs, indices, routing_weights, gc_sf)
        independent = torch.empty((tokens, args.hidden), dtype=torch.bfloat16, device=device)
        dist.barrier()
        routed, standalone = run_independent(
            independent, reference_routed, original_shared, (x8, xs), indices,
            routing_weights, vector, rank, args.activation_clamp, recipe)
        outputs = {name: torch.empty_like(independent) for name in ('base', 'gc')}
        records = {name: [] for name in outputs}

        def invoke(name, explicit=False):
            if name == 'base':
                op, rw, sw, buffer, sf = deep_gemm.fp8_fp4_mega_moe, mr, ms, base_buffer, base_sf
            else:
                op, rw, sw, buffer, sf = gc_moe.fp8_fp4_gc_moe, gr, gs, gc_buffer, gc_sf
            # Supplementary same-allocation switching also tests SF restaging in
            # a captured graph. Formal independent buffers need no copy here.
            if args.cross_operator_buffer:
                buffer.shared_l1_acts_sf.copy_(sf)
            kwargs = {} if args.omit_shared_recipe and not explicit else {'shared_recipe': recipe}
            op(outputs[name], *rw, buffer, shared_l1_weights=sw[0], shared_l2_weights=sw[1], **common, **kwargs)

        for name in outputs:
            invoke(name)
        torch.cuda.synchronize(device)
        # Store a snapshot after every launch, then check every repetition.
        for _ in range(args.gc_replays):
            for name in outputs:
                invoke(name)
                records[name].append(outputs[name].clone())
        graphs = {}
        for name in outputs:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                invoke(name)
            graphs[name] = graph
        for _ in range(args.gc_replays):
            for name in outputs:
                graphs[name].replay()
                records[name].append(outputs[name].clone())
        torch.cuda.synchronize(device)
        metrics = []
        reference_finite = all(bool(torch.isfinite(t).all()) for t in [independent, routed, standalone])
        for base, candidate in zip(records['base'], records['gc']):
            metrics.append(dict(
                mega_independent=cosine(base, independent), gc_independent=cosine(candidate, independent),
                mega_shared=cosine(base.float() - routed.float(), standalone),
                gc_shared=cosine(candidate.float() - routed.float(), standalone),
                gc_mega=cosine(candidate, base),
                max_abs_gc_mega=float((candidate.float() - base.float()).abs().max()) if tokens else 0.0,
                finite=reference_finite and bool(torch.isfinite(base).all()) and bool(torch.isfinite(candidate).all())))
        local = {k: min(float(m[k]) for m in metrics) for k in
                 ['mega_independent', 'gc_independent', 'mega_shared', 'gc_shared', 'gc_mega']}
        local.update(rank=rank, num_tokens_per_rank=tokens, gc_block_m=gc_bm, base_block_m=base_bm,
                     finite=all(m['finite'] for m in metrics), checked_outputs_per_operator=len(metrics),
                     max_abs_gc_mega=max(m['max_abs_gc_mega'] for m in metrics), repeat_checks=metrics)
        if args.omit_shared_recipe:
            local['default_explicit_cos'] = {}
            for name in outputs:
                previous = outputs[name].clone()
                invoke(name, explicit=True)
                local['default_explicit_cos'][name] = cosine(previous, outputs[name])
        local['passed'] = (local['finite'] and local['gc_mega'] >= args.cosine_threshold
                           and min(local[k] for k in ['mega_independent', 'gc_independent', 'mega_shared', 'gc_shared']) >= args.cosine_threshold
                           and min(local.get('default_explicit_cos', {'unused': 1}).values()) >= args.cosine_threshold)
        rank_checks = [None] * world
        dist.all_gather_object(rank_checks, local)
        passed = all(row['passed'] for row in rank_checks)
        case = dict(case=f'asym_{index}' if vectors else f'uniform_{tokens}',
                    num_tokens_per_rank=None if vectors else tokens, tokens_per_rank_by_rank=vector,
                    num_tokens_global=sum(vector), passed=passed, rank_checks=rank_checks)
        results.append(case)
        if rank == 0:
            print(f"CASE {case['case']} tokens_per_rank={case['num_tokens_per_rank']} "
                  f"total_tokens={case['num_tokens_global']} vector={vector} gc_mega={min(r['gc_mega'] for r in rank_checks):.9f} "
                  f"independent={min(min(r['mega_independent'],r['gc_independent']) for r in rank_checks):.9f} "
                  f"shared={min(min(r['mega_shared'],r['gc_shared']) for r in rank_checks):.9f} "
                  f"finite={all(r['finite'] for r in rank_checks)} {'PASS' if passed else 'FAIL'}", flush=True)
        del graphs, graph, records, outputs, metrics, x, x8, xs, indices, routing_weights
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    all_pass = all(case['passed'] for case in results)
    if rank == 0:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(world_size=world, case_category=args.case_category, config=vars(args), requested_capacity=capacity,
                         aligned_capacity={'baseline': base_buffer.num_max_tokens_per_rank,
                                           'candidate': gc_buffer.num_max_tokens_per_rank},
                         routed_recipe=ROUTED_RECIPE,
                         shared_recipe=recipe,
                         shared_weight_recipe=(1, args.shared_weight_gran_k, args.shared_weight_gran_k),
                         activation_gran_k=ACTIVATION_GRAN_K,
                         eager_repeats=args.gc_replays, graph_repeats=args.gc_replays,
                         reference='external Torch dequant, distributed routed sum and local shared branch; no MoE reference calls',
                         operators={'baseline': 'deep_gemm.fp8_fp4_mega_moe', 'candidate': 'gc_moe.fp8_fp4_gc_moe'},
                         modules={'baseline': deep_gemm.__file__, 'candidate': gc_moe.__file__},
                         all_pass=all_pass, results=results), indent=2) + '\n')
        print('ALL PASS' if all_pass else 'SOME FAILED', str(out), flush=True)
    torch.cuda.synchronize(device)
    gc_buffer.destroy()
    base_buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()
    if not all_pass:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
