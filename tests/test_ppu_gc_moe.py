"""Check PPU MXFP4 GC MoE accuracy and workspace reuse against a staged reference."""

import argparse
import json
import os
import random
import sys
from datetime import timedelta
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist

import gc_moe
from gc_moe.jit_kernels.heuristics.ppu_gc_moe import PPURoleConfig


_MODELS = {
    # name: (hidden, intermediate, experts, topk)
    'glm52': (6144, 2048, 256, 8),
    'dsv41flash': (5120, 2304, 384, 6),
}

_BOUNDARIES = None
_CAPACITY = 256
DEFAULT_TOKENS_PER_RANK = (1, 2, 4, 8, 16, 32, 64, 128, 256)
_BOUNDARY_TOKENS = (0, 3, 5, 7, 9, 15, 17, 31, 33, 63, 65, 95, 96, 97,
                    127, 129, 191, 192, 193, 239, 240, 241, 255)
_ASYMMETRIC_TOKENS = (
    (0,) * 8, (256, 0) * 4, (1, 0) * 4, DEFAULT_TOKENS_PER_RANK[:-1],
    (256, 0, 0, 0, 0, 0, 0, 0), (256,) * 8, (256, 256, 0, 256, 256, 256, 256, 256),
    (256, 256, 256, 0, 256, 256, 256, 256), (0, 0, 0, 0, 0, 256, 0, 0),
    (0, 0, 0, 0, 0, 0, 256, 0), (256, 256, 256, 256, 256, 256, 256, 0),
    (0, 1, 2, 3, 4, 5, 7, 8), (9, 15, 16, 17, 31, 32, 33, 63),
    (64, 65, 95, 96, 97, 127, 128, 129), (191, 192, 193, 239, 240, 241, 255, 256),
)


def _ue8m0_scale(amax: torch.Tensor, headroom: float, min_exp: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pick the UE8M0 power-of-two scale covering `amax / headroom`."""
    log2 = torch.ceil(torch.log2(amax.clamp_min(2 ** -40)) - headroom)
    exponent = (log2 + 127).clamp(min_exp, 254)
    return torch.exp2(exponent - 127.0), exponent.to(torch.uint8)


def quant_mxfp4(x: torch.Tensor, group: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per (row, K32) MXFP4: e2m1 codes with UE8M0 scales.

    Returns packed (..., K/2) uint8 and UE8M0 bytes (..., K/32) uint8.
    Works for weights (..., N, K) and activations (T, K) alike.
    """
    shape, k = x.shape[:-1], x.shape[-1]
    dev = x.device
    global _BOUNDARIES
    if _BOUNDARIES is None or _BOUNDARIES.device != dev:
        _BOUNDARIES = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=dev)
    w = x.float().reshape(*shape, k // group, group)
    amax = w.abs().amax(-1)
    scale, code = _ue8m0_scale(amax, 2.0, 113)  # |q| <= 4 fits e2m1 max 6
    q = w / scale.unsqueeze(-1)
    codes = torch.bucketize(q.abs(), _BOUNDARIES)
    sign = (q < 0).to(torch.uint8)
    nibbles = (sign << 3) | codes.to(torch.uint8)
    packed = (nibbles[..., 1::2] << 4 | nibbles[..., 0::2]).reshape(*shape, k // 2)
    return packed.contiguous(), code.reshape(*shape, k // group).contiguous()


def _gather(tensor, ranks):
    local = tensor.cpu().contiguous()
    gathered = [torch.empty_like(local) for _ in range(ranks)]
    dist.all_gather(gathered, local)
    return torch.stack(gathered)


def _requantize(x):
    groups = x.reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    amax = groups.abs().amax(dim=-1)
    bits = amax.view(torch.int32)
    # Round amax / 4 up to a power of two and encode the UE8M0 exponent.
    exponent = ((bits >> 23) + ((bits & 0x7fffff) != 0) - 2).clamp(113, 254)
    inverse = ((254 - exponent) << 23).view(torch.float32)
    values = groups * inverse.unsqueeze(-1)
    magnitude = values.abs()
    boundaries = values.new_tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.])
    codes = torch.bucketize(magnitude.contiguous(), boundaries)
    # E2M1 uses round-to-nearest-even at the midpoint of adjacent codes.
    codes += ((codes & 1) != 0) & (magnitude == boundaries[codes.clamp_max(6)])
    nibbles = codes.to(torch.uint8) | ((values < 0).to(torch.uint8) << 3)
    packed = nibbles[..., ::2] | (nibbles[..., 1::2] << 4)
    return packed.flatten(-2), exponent.to(torch.uint8)


class ReferenceMoE:
    def __init__(self, deep_gemm, w1, w1_sf, w2, w2_sf, rank, ranks):
        self.gemm = deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked
        self.w1 = w1, deep_gemm.preprocess_mxfp4_scales(w1_sf)
        self.w2 = w2, deep_gemm.preprocess_mxfp4_scales(w2_sf)
        self.rank = rank
        self.ranks = ranks

    def __call__(self, packed, scales, ids, weights, token_counts=None, *, bias=None, clamp=None):
        bound, topk = ids.shape
        tokens = bound if token_counts is None else token_counts[self.rank]
        experts, hidden, half_inter = self.w2[0].shape
        if bound == 0:
            return torch.empty((0, hidden), dtype=torch.bfloat16, device='cuda')
        inter = 2 * half_inter
        routes = _gather(ids, self.ranks).reshape(self.ranks, -1)
        owners = routes // experts
        valid = (routes >= 0) & (routes < self.ranks * experts)
        local_routes = (owners == self.rank) & valid
        if token_counts is not None:
            positions = torch.arange(bound * topk, device='cpu')
            limits = torch.tensor(token_counts, device='cpu') * topk
            local_routes &= positions < limits[:, None]
        routed = local_routes.nonzero()
        local_ids = routes[routed[:, 0], routed[:, 1]].long() % experts
        counts = torch.bincount(local_ids, minlength=experts)
        order = local_ids.argsort(stable=True)
        starts = counts.cumsum(0) - counts
        offsets = torch.arange(order.numel(), device='cpu') - starts[local_ids[order]]
        rows = torch.empty_like(offsets)
        rows[order] = offsets
        # Pad the masked GEMM buffers to the actual maximum expert load.
        m = max(16, ((int(counts.max()) + 15) // 16) * 16)
        coordinates = (local_ids.cuda(), rows.cuda())
        source = routed[:, 0], routed[:, 1] // topk
        x = packed.new_zeros((experts, m, hidden // 2))
        sf = scales.new_full((experts, m, hidden // 32), 127)
        x[coordinates] = _gather(packed, self.ranks)[source].cuda()
        sf[coordinates] = _gather(scales, self.ranks)[source].cuda()
        route_weights = _gather(weights, self.ranks).reshape(self.ranks, -1)
        route_weights = route_weights[routed[:, 0], routed[:, 1]].cuda()

        if routed.numel():
            counts_cuda = counts.to(device='cuda', dtype=torch.int32)
            gate_up = torch.empty((experts, m, 2 * inter), dtype=torch.bfloat16, device='cuda')
            self.gemm((x, sf), self.w1, None, gate_up, counts_cuda, m)
            gate, up = gate_up[coordinates].float().chunk(2, dim=-1)
            if clamp is not None:
                gate = gate.clamp_max(clamp)
                up = up.clamp(-clamp, clamp)
            activation = ((gate / (1.0 + torch.exp(-gate))) * up) * route_weights.unsqueeze(-1)
            mid, mid_sf = _requantize(activation)
            grouped_mid = packed.new_zeros((experts, m, inter // 2))
            grouped_mid_sf = scales.new_full((experts, m, inter // 32), 127)
            grouped_mid[coordinates] = mid
            grouped_mid_sf[coordinates] = mid_sf
            l2 = gate_up.new_empty((experts, m, hidden))
            self.gemm((grouped_mid, grouped_mid_sf), self.w2, None, l2, counts_cuda, m)
            contributions = l2[coordinates].cpu()
        else:
            contributions = torch.empty((0, hidden), dtype=torch.bfloat16, device='cpu')

        # Exchange BF16 contributions back in source rank/slot order.
        send = contributions.contiguous().view(torch.uint8)
        valid_slots = valid[self.rank, :tokens * topk].nonzero().flatten()
        recv = torch.empty((len(valid_slots), 2 * hidden), dtype=torch.uint8, device='cpu')
        source_owners = owners[self.rank, valid_slots]
        recv_splits = torch.bincount(source_owners.long(), minlength=self.ranks).tolist()
        send_splits = torch.bincount(routed[:, 0], minlength=self.ranks).tolist()
        dist.all_to_all_single(recv, send, recv_splits, send_splits)
        slots = torch.zeros((tokens * topk, hidden), dtype=torch.bfloat16, device='cuda')
        slots[valid_slots[source_owners.argsort(stable=True)].cuda()] = recv.view(torch.bfloat16).cuda()
        result = (torch.zeros((tokens, hidden), dtype=torch.float32, device='cuda')
                  if bias is None else bias.float())
        for contribution in slots.view(tokens, topk, hidden).unbind(dim=1):
            result.add_(contribution.float())
        return result.bfloat16()


class SharedExpert:
    """Standalone MXFP4 FC1, SwiGLU and FC2 for a single local shared expert."""

    def __init__(self, deep_gemm, w1, w1_sf, w2, w2_sf, capacity):
        self.gemm = deep_gemm.gemm_fp4_fp4_bf16_nt
        self.w1 = w1, deep_gemm.preprocess_mxfp4_scales(w1_sf)
        self.w2 = w2, deep_gemm.preprocess_mxfp4_scales(w2_sf)
        self.hidden, half_inter = w2.shape
        self.inter = 2 * half_inter
        self.gate_up = torch.empty((capacity, 2 * self.inter), dtype=torch.bfloat16, device='cuda')
        self.output = torch.empty((capacity, self.hidden), dtype=torch.bfloat16, device='cuda')

    def __call__(self, tokens, packed, scales, *, clamp=None):
        if not tokens:
            return self.output[:0]
        self.gemm((packed[:tokens], scales[:tokens]), self.w1, None, self.gate_up[:tokens])
        gate, up = self.gate_up[:tokens].float().chunk(2, dim=-1)
        if clamp is not None:
            gate = gate.clamp_max(clamp)
            up = up.clamp(-clamp, clamp)
        mid, mid_sf = _requantize((gate / (1.0 + torch.exp(-gate))) * up)
        self.gemm((mid, mid_sf), self.w2, None, self.output[:tokens])
        return self.output[:tokens]


# ---------------------------------------------------------------------------
# Accuracy and workspace reuse.
# ---------------------------------------------------------------------------

def received_counts(ids, token_counts, rank, ranks, local_experts):
    gathered = _gather(ids, ranks)
    limits = torch.tensor(token_counts, device="cpu")[:, None, None]
    positions = torch.arange(ids.shape[0], device="cpu")[None, :, None]
    first = rank * local_experts
    valid = (positions < limits) & (gathered >= first) & (gathered < first + local_experts)
    return torch.bincount(gathered[valid] - first, minlength=local_experts)


def check_reuse(*, step, reference, x, packed, scales, ids, weights, out, experts, rank, ranks,
                token_sizes, rounds, shared_expert=False, readiness=False, seed=20261110):
    chooser = random.Random(seed)
    generator = torch.Generator('cpu').manual_seed(seed + rank)
    topk = ids.shape[1]
    local_experts = experts // ranks
    results = []
    if readiness:
        stats = torch.zeros(local_experts, dtype=torch.int32, device=packed.device)
        accumulated = torch.zeros_like(stats)
    for request in range(rounds):
        bound = chooser.choice(token_sizes)
        token_counts = None
        tokens = bound
        if readiness:
            token_counts = tuple(
                0 if (r + request) % 4 == 0 else bound if (r + request) % 2 else bound // 2
                for r in range(ranks))
            tokens = token_counts[rank]
        if request % 5 == 0:
            packed[:bound].random_(0, 256)
            scales[:bound].random_(113, 122)
            expected_packed = packed[:bound].clone()
            expected_scales = scales[:bound].clone()
        else:
            x[:bound].normal_(mean=0, std=.1)
            expected_packed, expected_scales = quant_mxfp4(x[:bound].float())
            packed[:bound].copy_(expected_packed)
            scales[:bound].copy_(expected_scales)
        if request % 4 == 0 and topk <= local_experts:
            # Rotate the expert owner; ranks with no local hits must publish.
            owner = (request // 4) % ranks
            selected = (torch.arange(topk, device='cpu') + owner * local_experts).expand(bound, -1)
        elif request % 4 == 1 and topk <= local_experts:
            selected = (torch.arange(topk, device='cpu') + rank * local_experts).expand(bound, -1)
        else:
            selected = torch.stack([torch.randperm(experts, generator=generator, device='cpu')[:topk]
                                    for _ in range(bound)])
        ids[:bound].copy_(selected)
        weights[:bound].uniform_()
        weights[:bound] /= weights[:bound].sum(-1, keepdim=True)
        if request % 11 == 0:
            weights[:bound].zero_()
        out.fill_(float('nan'))
        if readiness:
            step(tokens, bound, stats=stats)
        else:
            step(tokens)
        candidate = out[:tokens].clone()
        finite = torch.isfinite(candidate).all().item()
        guard = torch.isnan(out[tokens:]).all().item()
        quant_exact = (torch.equal(packed[:bound], expected_packed)
                       and torch.equal(scales[:bound], expected_scales))
        expected = reference(tokens, expected_packed, expected_scales, token_counts)
        same = torch.equal(candidate, expected)
        zero = shared_expert or request % 11 != 0 or (candidate == 0).all().item()
        status = dict(rank=rank, request=request, tokens=tokens, finite=finite,
                      guard=guard, quant_exact=quant_exact, exact=same, zero=zero)
        if readiness:
            out.fill_(float('nan'))
            step(tokens, bound, stats=stats)
            accumulated += 2 * received_counts(ids[:bound], token_counts, rank, ranks, local_experts).to(stats)
            status.update(repeat_exact=torch.equal(out[:tokens], candidate),
                          recv_stats_exact=torch.equal(stats, accumulated),
                          guard=guard and torch.isnan(out[tokens:]).all().item())
        statuses = [None] * ranks
        dist.all_gather_object(statuses, status)
        passed = all(s['finite'] and s['guard'] and s['quant_exact'] and s['exact']
                     and s['zero'] and s.get('repeat_exact', True)
                     and s.get('recv_stats_exact', True) for s in statuses)
        results.append(dict(request=request, num_tokens_per_rank=tokens, vector=token_counts,
                            passed=passed, ranks=statuses))
        assert passed, statuses
        if rank == 0 and (request % 20 == 0 or request == rounds - 1):
            print(f'REUSE_PASS request={request} T={tokens} ranks={ranks}', flush=True)
    return results


def check_contract(args, deep_gemm, group, rank, ranks, packed, scales, ids, weights,
                   out, l1_weights, l2_weights, shared_weights):
    hidden, inter, experts, topk = args.shape
    local = experts // ranks
    shared = args.shared_expert
    buffer = gc_moe.get_symm_buffer_for_gc_moe_ppu(
        group, experts, args.num_max_tokens_per_rank, topk, hidden, inter,
        num_shared_experts=int(shared),
        cache_dir=args.cache_dir)
    buffer.x.copy_(packed)
    buffer.x_sf.copy_(scales)
    buffer.topk_weights.copy_(weights)
    p1, p2 = gc_moe.transform_weights_for_gc_moe_ppu(l1_weights, l2_weights)
    legacy_p2 = p2
    if p2[0].ndim == 4:
        legacy_payload = (p2[0].view(inter // 128, local, hidden // 512, 512, 64)
                          .permute(1, 2, 0, 3, 4).contiguous())
        legacy_p2 = legacy_payload, p2[1]
    reference = ReferenceMoE(deep_gemm, *l1_weights, *l2_weights, rank, ranks)
    shared_options = {}
    if shared:
        # K128 is expanded into four identical K32 scale bytes before MMA.
        (sw1, ss1), (sw2, ss2) = shared_weights
        se1, se2 = ss1[:, ::4].contiguous(), ss2[:, ::4].contiguous()
        recipe = (1, 128, 128)
        sp1, sp2 = gc_moe.transform_weights_for_gc_moe_ppu(
            (sw1, se1), (sw2, se2), shared_recipe=recipe)
        shared_ref = SharedExpert(deep_gemm, sw1, se1.repeat_interleave(4, -1),
                                 sw2, se2.repeat_interleave(4, -1), args.num_max_tokens_per_rank)
        shared_options = dict(shared_l1_weights=sp1, shared_l2_weights=sp2, shared_recipe=recipe)
    stats = torch.zeros(local, dtype=torch.int32, device='cuda')
    accumulated = torch.zeros_like(stats)
    results = []
    for request, bound in enumerate(args.tokens):
        token_counts = tuple(bound if (r + request) % 3 else 0 for r in range(ranks))
        t = token_counts[rank]
        routes = ids.clone()
        if request % 4 == 1:
            routes.fill_(-1)
        elif request % 4 == 2:
            routes.fill_(0)  # Repeated routes exercise the full worst-case receive bound.
            token_counts = (bound,) * ranks
            t = bound
        elif request % 4 == 3:
            routes[::2, 0] = -(1 << 40)
            routes[1::2, -1] = 1 << 40
        buffer.topk_idx.copy_(routes)
        clamp = .0625 if request % 2 else None
        out.fill_(float('nan'))
        use_shared = shared and request != 4
        gc_moe.ppu_fp4_fp4_gc_moe(
            out[:t], p1, p2, buffer, stats, num_tokens_bound=bound,
            activation_clamp=clamp, fast_math=clamp is None,
            **(shared_options if use_shared else {}))
        bias = shared_ref(t, packed, scales, clamp=clamp) if use_shared else None
        expected = reference(packed[:bound], scales[:bound], routes[:bound], weights[:bound],
                             token_counts, bias=bias, clamp=clamp)
        accumulated += received_counts(routes[:bound], token_counts, rank, ranks, local).to(stats)
        assert torch.equal(out[:t], expected), (rank, request, t)
        assert torch.isnan(out[t:]).all().item()
        assert torch.equal(stats, accumulated), (rank, request, stats, accumulated)
        assert torch.equal(buffer.x, packed) and torch.equal(buffer.x_sf, scales)
        assert torch.equal(buffer.topk_idx, routes) and torch.equal(buffer.topk_weights, weights)
        if legacy_p2 is not p2:
            out.fill_(float('nan'))
            gc_moe.ppu_fp4_fp4_gc_moe(
                out[:t], p1, legacy_p2, buffer, num_tokens_bound=bound,
                activation_clamp=clamp, fast_math=clamp is None,
                **(shared_options if use_shared else {}))
            assert torch.equal(out[:t], expected), (rank, request, t, 'legacy W2 panels')
            assert torch.isnan(out[t:]).all().item()
        results.append(dict(request=request, vector=token_counts, shared_expert=use_shared,
                            stats_exact=True, output_exact=True,
                            legacy_w2_exact=True, passed=True))
    # Two layers may share a workspace and schedule while their weights,
    # output buffers, and counters change on every submission.
    buffer.topk_idx.copy_(ids)
    tokens = 1
    expected = reference(packed[:tokens], scales[:tokens], ids[:tokens], weights[:tokens])
    expected_counts = received_counts(ids[:tokens], (tokens,) * ranks, rank, ranks, local).to(stats)
    zero_w2 = torch.zeros_like(p2[0]), p2[1]
    alternate_out = torch.full_like(out, float('nan'))
    alternate_stats = torch.zeros_like(stats)
    for weights2, y, counter, expected_y in (
            (p2, out, stats, expected),
            (zero_w2, alternate_out, alternate_stats, torch.zeros_like(expected)),
            (p2, out, stats, expected)):
        y.fill_(float('nan'))
        before = counter.clone()
        gc_moe.ppu_fp4_fp4_gc_moe(y[:tokens], p1, weights2, buffer, counter,
                                 num_tokens_bound=tokens)
        assert torch.equal(y[:tokens], expected_y) and torch.isnan(y[tokens:]).all().item()
        assert torch.equal(counter, before + expected_counts)
    dist.barrier(group=group)
    buffer.destroy()
    if rank == 0:
        print(f'CONTRACT_PASS model={args.model} shared={shared} cases={len(args.tokens)} ranks={ranks}',
              flush=True)

    return results


def cosine(actual: torch.Tensor, reference: torch.Tensor) -> float:
    actual, reference = actual.flatten().float(), reference.flatten().float()
    norm_product = (actual.norm() * reference.norm()).item()
    if norm_product == 0:
        return float(torch.equal(actual, reference))
    return float((actual @ reference).item() / norm_product)


def run_accuracy(step, reference, x, packed, scales, output, cases, rank):
    results = []
    for token_counts in cases:
        bound, tokens = max(token_counts), token_counts[rank]
        output.fill_(float("nan"))
        xp, sf = quant_mxfp4(x[:bound].float())
        step(tokens, bound)
        expected = reference(tokens, xp, sf, token_counts)
        actual = output[:tokens]
        checks = torch.tensor([
            torch.equal(actual, expected),
            torch.equal(xp[:tokens], packed[:tokens]) and torch.equal(sf[:tokens], scales[:tokens]),
            bool(torch.isnan(output[tokens:]).all()),
            bool(torch.isfinite(actual).all()),
        ], dtype=torch.int32, device="cpu")
        max_abs = float((actual.float() - expected.float()).abs().max()) if tokens else 0.0
        metrics = torch.tensor([cosine(actual, expected), -max_abs], dtype=torch.float64, device="cpu")
        dist.all_reduce(checks, op=dist.ReduceOp.MIN)
        dist.all_reduce(metrics, op=dist.ReduceOp.MIN)
        output_exact, quant_exact, guard, finite = map(bool, checks.tolist())
        cos_min, neg_max_abs = metrics.tolist()
        max_abs = abs(neg_max_abs)
        passed = output_exact and quant_exact and guard and finite
        if rank == 0:
            status = "OK" if passed else "FAIL"
            print(f"  tokens_per_rank={tokens:4d} total_tokens={sum(token_counts):5d} "
                  f"cos_processed_min={cos_min:.9f} max_abs={max_abs:.6g} "
                  f"output_exact={output_exact} quant_exact={quant_exact} "
                  f"guard={guard} finite={finite} [{status}]", flush=True)
        results.append(dict(num_tokens_per_rank=tokens, vector=token_counts,
                            cos_processed_min=cos_min, max_abs=max_abs,
                            output_exact=output_exact, quant_exact=quant_exact,
                            guard=guard, finite=finite, passed=passed))
    return results


# ---------------------------------------------------------------------------
# Public GC MoE calls and workload setup.
# ---------------------------------------------------------------------------

class GCMoE:
    def __init__(self, args, group, output, routed_weights, shared_weights=(), *, shared_weight_gran_k=32):
        hidden, inter, experts, topk = args.shape
        self.workspace = gc_moe.get_symm_buffer_for_gc_moe_ppu(
            group, experts, args.num_max_tokens_per_rank, topk, hidden, inter,
            num_shared_experts=int(bool(shared_weights)), cache_dir=args.cache_dir)
        self.weights = gc_moe.transform_weights_for_gc_moe_ppu(*routed_weights)
        self.shared_options = {}
        if shared_weights:
            shared_recipe = (1, 1, 32) if shared_weight_gran_k == 32 else (1, 128, 128)
            shared_l1, shared_l2 = gc_moe.transform_weights_for_gc_moe_ppu(
                *shared_weights, shared_recipe=shared_recipe)
            self.shared_options = dict(shared_l1_weights=shared_l1, shared_l2_weights=shared_l2)
        self.output = output

    def __call__(self, tokens, bound=None, *, stats=None):
        gc_moe.ppu_fp4_fp4_gc_moe(
            self.output[:tokens], *self.weights, self.workspace,
            cumulative_local_expert_recv_stats=stats, num_tokens_bound=tokens if bound is None else bound, **self.shared_options)

    def destroy(self):
        self.workspace.destroy()


def build_workload(args, rank, ranks, *, shared_weight_gran_k=32):
    hidden, inter, experts, topk = args.shape
    local_experts = experts // ranks
    capacity = args.num_max_tokens_per_rank
    generator = torch.Generator("cpu").manual_seed(777 + 131 * rank)
    x = (torch.randn(capacity, hidden, generator=generator, device="cpu") * .1).bfloat16().cuda()
    ids = torch.stack([torch.randperm(experts, generator=generator, device="cpu")[:topk]
                       for _ in range(capacity)]).cuda()
    weights = torch.rand(capacity, topk, generator=generator, device="cpu").cuda()
    weights /= weights.sum(-1, keepdim=True)
    packed, scales = quant_mxfp4(x.float())

    torch.manual_seed(20261109 + rank)
    w1, s1 = quant_mxfp4((torch.randn(local_experts, 2 * inter, hidden, device="cuda")
                         * .02).bfloat16())
    w2, s2 = quant_mxfp4((torch.randn(local_experts, hidden, inter, device="cuda")
                         * .02).bfloat16())
    shared_weights = ()
    if args.shared_expert:
        shared_weights = (
            quant_mxfp4((torch.randn(2 * inter, hidden, device="cuda") * .02).bfloat16(),
                        group=shared_weight_gran_k),
            quant_mxfp4((torch.randn(hidden, inter, device="cuda") * .02).bfloat16(),
                        group=shared_weight_gran_k),
        )
    output = torch.empty(capacity, hidden, dtype=torch.bfloat16, device="cuda")

    return x, packed, scales, ids, weights, ((w1, s1), (w2, s2)), shared_weights, output


def run_tests(args, group, rank, ranks):
    import deep_gemm

    x, packed, scales, ids, weights, routed_weights, shared_weights, output = build_workload(args, rank, ranks)
    if args.contract:
        # Exercise the same public contract on a non-default stream.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            results = check_contract(args, deep_gemm, group, rank, ranks, packed, scales,
                                     ids, weights, output, *routed_weights, shared_weights)
        return results

    experts = args.shape[2]
    ref = ReferenceMoE(deep_gemm, *routed_weights[0], *routed_weights[1], rank, ranks)
    shared_ref = (SharedExpert(deep_gemm, *shared_weights[0], *shared_weights[1],
                               args.num_max_tokens_per_rank) if shared_weights else None)
    gc = GCMoE(args, group, output, routed_weights, shared_weights)
    workspace = gc.workspace
    try:
        workspace.x.copy_(packed)
        workspace.x_sf.copy_(scales)
        workspace.topk_idx.copy_(ids)
        workspace.topk_weights.copy_(weights)

        def reference(tokens, xp, sf, token_counts=None):
            bound = tokens if token_counts is None else max(token_counts)
            bias = shared_ref(tokens, xp, sf) if shared_ref is not None else None
            return ref(xp, sf, workspace.topk_idx[:bound], workspace.topk_weights[:bound],
                       token_counts, bias=bias)

        if args.gc_replays:
            results = check_reuse(step=gc, reference=reference, x=x, packed=workspace.x,
                                  scales=workspace.x_sf, ids=workspace.topk_idx,
                                  weights=workspace.topk_weights, out=output, experts=experts,
                                  rank=rank, ranks=ranks, token_sizes=args.tokens,
                                  rounds=args.gc_replays, shared_expert=args.shared_expert,
                                  readiness=args.readiness)
        else:
            results = run_accuracy(gc, reference, x, workspace.x, workspace.x_sf,
                                   output, args.cases, rank)
        return results
    finally:
        gc.destroy()


def add_workload_args(parser, *, shared=False, capacity=_CAPACITY):
    parser.add_argument("--model", help="glm52, dsv41flash, or H,I,E,TOPK")
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--intermediate-hidden", type=int)
    parser.add_argument("--num-experts", type=int)
    parser.add_argument("--num-topk", type=int)
    parser.add_argument("--num-max-tokens-per-rank", type=int, default=capacity)
    parser.add_argument("--num-tokens-per-rank-list", "--tokens",
                        help="Comma-separated token counts per rank.")
    parser.add_argument("--cache-dir", type=Path, help="Override the PPU JIT cache directory.")
    parser.add_argument("--ranks", type=int, help="Number of PPUs for direct Python launches.")
    parser.add_argument("--shared-expert", action="store_true")
    parser.add_argument("--num-shared-experts", type=int, choices=(0, 1), default=int(shared))
    parser.add_argument("--output-json", type=Path)


def parse_args(*, shared=False, readiness=False):
    parser = argparse.ArgumentParser(description=__doc__)
    add_workload_args(parser, shared=shared)
    parser.add_argument("--per-rank-tokens-list", help="Semicolon-separated list of per-rank token vectors.")
    parser.add_argument("--num-tokens-per-rank-vector", help="One comma-separated per-rank vector.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--gc-replays", "--stress", "--steps", type=int,
                      default=200 if readiness else 0, metavar="ROUNDS",
                      help="Check the workspace with changing inputs and routes.")
    mode.add_argument("--contract", action="store_true", help="Exercise the public packed-input API.")
    mode.add_argument("--case-category", choices=("default", "boundary", "asymmetric"),
                      default="default")
    parser.add_argument("legacy", nargs="*", help=argparse.SUPPRESS)
    if readiness:
        parser.add_argument("--mode", choices=("eager",), default="eager")
    args = parser.parse_args()
    args.readiness = readiness
    if readiness and (args.contract or args.case_category != "default"
                      or args.gc_replays < 1):
        parser.error("readiness requires positive eager replays")
    if args.legacy:
        if len(args.legacy) > 2 or args.model is not None:
            parser.error("expected at most a positional cache directory and model")
        if len(args.legacy) == 2:
            if args.cache_dir is not None:
                parser.error("provide only one cache directory")
            args.cache_dir = Path(args.legacy[0])
        args.model = args.legacy[-1]

    configure_workload(parser, args, shared=shared)
    try:
        if args.per_rank_tokens_list and args.num_tokens_per_rank_vector:
            raise ValueError("provide only one per-rank vector option")
        vectors = args.per_rank_tokens_list or args.num_tokens_per_rank_vector
        args.vectors = [tuple(map(int, vec.split(","))) for vec in vectors.split(";")] if vectors else []
        if args.vectors and (args.num_tokens_per_rank_list or args.contract or args.gc_replays):
            raise ValueError("per-rank vectors apply to accuracy checks only")
        if args.num_tokens_per_rank_list is not None and args.case_category == "asymmetric":
            raise ValueError("use per-rank vectors to override the asymmetric cases")
        args.tokens = (tuple(map(int, args.num_tokens_per_rank_list.split(",")))
                       if args.num_tokens_per_rank_list is not None else
                       _BOUNDARY_TOKENS if args.case_category == "boundary" else DEFAULT_TOKENS_PER_RANK)
        if args.gc_replays < 0:
            raise ValueError("expected a nonnegative replay count")
    except ValueError as exc:
        parser.error(str(exc))
    if args.output_json is None:
        suffix = "_se" if args.shared_expert else ""
        kind = "readiness" if readiness else "accuracy"
        args.output_json = Path(f"results/_{kind}_results_ppu_gc{suffix}.json")
    return args


def configure_workload(parser, args, *, shared):
    try:
        if args.model is None:
            shape = _MODELS["dsv41flash"]
        elif args.model in _MODELS:
            shape = _MODELS[args.model]
        else:
            shape = tuple(map(int, args.model.split(",")))
        if len(shape) != 4:
            raise ValueError("expected model geometry H,I,E,TOPK")
        overrides = args.hidden, args.intermediate_hidden, args.num_experts, args.num_topk
        args.shape = tuple(default if value is None else value for default, value in zip(shape, overrides))
        if args.model is None:
            args.model = next((model for model, geometry in _MODELS.items() if geometry == args.shape),
                              ",".join(map(str, args.shape)))
        if args.num_max_tokens_per_rank <= 0:
            raise ValueError("expected positive capacity")
    except ValueError as exc:
        parser.error(str(exc))
    args.num_shared_experts = args.num_shared_experts or int(args.shared_expert)
    args.shared_expert = bool(args.num_shared_experts)
    if shared and not args.shared_expert:
        parser.error("the SE entry requires a shared expert")


def launch_ranks(args):
    os.environ.setdefault("DEEPEP_DISABLE_NETWORK", "1")
    if "LOCAL_RANK" not in os.environ:
        ranks = 8 if args.ranks is None else args.ranks
        if not 1 <= ranks <= 8:
            raise ValueError("PPU GC MoE supports one to eight ranks")
        os.execv(sys.executable, [sys.executable, "-m", "torch.distributed.run", "--standalone",
                                  f"--nproc-per-node={ranks}", *sys.argv])

    local_rank = int(os.environ["LOCAL_RANK"])
    ranks = int(os.environ["WORLD_SIZE"])
    if not 1 <= ranks <= 8 or int(os.environ["LOCAL_WORLD_SIZE"]) != ranks:
        raise ValueError("PPU GC MoE tests require one to eight ranks on one host")
    if args.ranks is not None and args.ranks != ranks:
        raise ValueError("--ranks must match the torchrun world size")
    hidden, inter, experts, topk = args.shape
    if experts % ranks:
        raise ValueError("routed experts must divide evenly across ranks")
    PPURoleConfig(hidden, inter, experts // ranks, topk, ranks)
    return local_rank, ranks


def init_dist(local_rank):
    torch.cuda.set_device(local_rank)
    torch.set_default_device("cuda")
    # The PPU uses SAILSHMEM for dispatch and Gloo for CPU reference collectives.
    dist.init_process_group(backend="gloo", timeout=timedelta(seconds=600))


def main(*, shared=False, readiness=False):
    args = parse_args(shared=shared, readiness=readiness)
    local_rank, ranks = launch_ranks(args)
    hidden, inter, experts, topk = args.shape
    if args.vectors:
        args.cases = args.vectors
    elif args.case_category == "asymmetric":
        args.cases = _ASYMMETRIC_TOKENS
    else:
        args.cases = [(tokens,) * ranks for tokens in args.tokens]
    min_tokens = int(args.gc_replays > 0)
    if any(len(vec) != ranks or not all(min_tokens <= t <= args.num_max_tokens_per_rank for t in vec)
           for vec in args.cases):
        raise ValueError("token counts must fit the world size and workspace capacity")

    init_dist(local_rank)
    rank = dist.get_rank()
    try:
        if rank == 0:
            print(f"[tests/test_ppu_gc_moe] world_size={ranks} device=cuda:{local_rank}")
            print(f"Config: model={args.model} hidden={hidden} intermediate={inter} "
                  f"experts={experts} topk={topk} shared_expert={args.shared_expert}", flush=True)
        results = run_tests(args, dist.group.WORLD, rank, ranks)
        all_pass = all(row["passed"] for row in results)
        if rank == 0:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            with args.output_json.open("w") as f:
                json.dump({
                    "world_size": ranks,
                    "operator": "gc_moe.ppu_fp4_fp4_gc_moe",
                    "config": {
                        "model": args.model, "hidden": hidden, "intermediate_hidden": inter,
                        "num_experts": experts, "num_topk": topk,
                        "num_max_tokens_per_rank": args.num_max_tokens_per_rank,
                        "num_shared_experts": args.num_shared_experts,
                        "case_category": args.case_category, "gc_replays": args.gc_replays,
                        "contract": args.contract, "readiness": args.readiness,
                    },
                    "all_pass": all_pass,
                    "results": results,
                }, f, indent=2)
            print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}. Wrote {args.output_json}", flush=True)
        return 0 if all_pass else 1
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
