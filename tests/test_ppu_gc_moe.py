"""PPU routed expert accuracy against masked MXFP4 grouped GEMM."""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist

from gc_moe import get_symm_buffer_for_gc_moe_ppu, ppu_fp4_fp4_gc_moe, transform_weights_for_gc_moe_ppu
from gc_moe.ppu_gc import MAX_TOKENS


DEFAULT_TOKENS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
FLASH_ROUTED_READINESS_TOKENS = (
    0, 1, 2, 8, 9, 10, 11, 17, 31, 32, 33, 63,
    64, 65, 79, 80, 81, 106, 107, 108, 119, 120, 121, 128, 139, 140, 167, 168,
    170, 171, 172, 245, 246, 247, 256,
)
FLASH_SE_READINESS_TOKENS = (0, 1, 8, 9, 17, 31, 32, 33, 63, 64, 128, 139, 140, 159, 160, 161, 256)
FLASH41_READINESS_TOKENS = (
    0, 1, 2, 3, 8, 9, 17, 31, 32, 33, 39, 40, 63, 64, 65,
    71, 72, 73, 95, 96, 97, 128, 139, 140, 159, 160, 161, 191, 192, 193, 255, 256,
)
MODEL_PRESETS = {'dsv4.1-flash': (5120, 2304, 384, 6), 'dsv4-flash': (4096, 2048, 256, 6)}
_BOUNDARIES = None


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


def _unpack_down_output(tensor):
    return tensor.unflatten(-1, (tensor.shape[-1] // 64, 4, 8, 2)).transpose(-3, -2).reshape_as(tensor)


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
        self.slots = slots.view(tokens, topk, hidden)
        result = (torch.zeros((tokens, hidden), dtype=torch.float32, device='cuda')
                  if bias is None else bias.float())
        for contribution in self.slots.unbind(dim=1):
            result.add_(contribution.float())
        return result.bfloat16()


def quant_ptpc(x):
    """E4M3 with one FP32 scale per activation token or weight channel."""
    x = x.float()
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1.e-10) / 448.
    return (x * scale.reciprocal()).clamp(-448., 448.).to(torch.float8_e4m3fn), scale


class PTPCSharedExpert:
    """Independent shared expert with PPU DeepGEMM and a Torch SwiGLU reference."""

    def __init__(self, deep_gemm, w1, w2, capacity):
        self.gemm = deep_gemm.fp8_gemm_nt
        self.w1, self.w2 = w1, w2
        hidden, inter = w2[0].shape
        self.gate_up = w1[0].new_empty((capacity, 2 * inter), dtype=torch.bfloat16)
        self.output = w2[0].new_empty((capacity, hidden), dtype=torch.bfloat16)

    def __call__(self, tokens, inputs, *, clamp=None):
        output = self.output[:tokens]
        if tokens:
            self.gemm(tuple(t[:tokens] for t in inputs), self.w1, self.gate_up[:tokens])
            gate, up = self.gate_up[:tokens].chunk(2, dim=-1)
            if clamp is not None:
                gate = gate.clamp_max(clamp)
                up = up.clamp(-clamp, clamp)
            gate = gate.float()
            mid = quant_ptpc((gate * gate.sigmoid()).bfloat16() * up)
            self.gemm(mid, self.w2, output)
        return output


def init_dist():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('gloo', timeout=timedelta(minutes=15))
    return dist.group.WORLD


def workload(rank, ranks, capacity, geometry, *, shared, skew=False):
    torch.manual_seed(20261109 + rank)
    device = 'cuda'
    hidden, intermediate, num_experts, topk = geometry
    experts = num_experts // ranks
    x = (torch.randn(capacity, hidden, device=device) * .1).bfloat16()
    generator = torch.Generator().manual_seed(777 + 131 * rank)
    ids = torch.stack([torch.randperm(num_experts, generator=generator)[:topk] for _ in range(capacity)]).cuda()
    if skew:
        ids[:] = torch.arange(topk, device=device)
    weights = torch.rand(capacity, topk, generator=generator).cuda()
    weights /= weights.sum(-1, keepdim=True)
    def routed_weight(n, k):
        weight = torch.empty((experts, n, k // 2), dtype=torch.uint8, device=device)
        scales = torch.empty((experts, n, k // 32), dtype=torch.uint8, device=device)
        for start in range(0, experts, 8):
            end = min(start + 8, experts)
            src = (torch.randn((end - start, n, k), device=device) * .02).bfloat16()
            wp, sf = quant_mxfp4(src)
            weight[start:end].copy_(wp)
            scales[start:end].copy_(sf)
        return weight, scales

    shapes = ((2 * intermediate, hidden), (hidden, intermediate))
    routed = tuple(routed_weight(n, k) for n, k in shapes)
    shared_weights = (tuple(quant_ptpc((torch.randn((n, k), device=device) * .02).bfloat16())
                            for n, k in shapes) if shared else None)
    return x, ids, weights, routed, shared_weights


def make_workspace(group, capacity, geometry, *, shared):
    hidden, intermediate, num_experts, topk = geometry
    return get_symm_buffer_for_gc_moe_ppu(
        group, num_experts, capacity, topk, hidden, intermediate,
        num_shared_experts=int(shared), shared_fp8=shared, graph_safe=True)


def stage(workspace, packed, scales, ids, weights, shared_inputs):
    workspace.x.copy_(packed)
    workspace.x_sf.copy_(scales)
    workspace.topk_idx.copy_(ids)
    workspace.topk_weights.copy_(weights)
    if shared_inputs is not None:
        workspace.shared_x.copy_(shared_inputs[0])
        workspace.shared_x_sf.copy_(shared_inputs[1])


def _cosine(left, right):
    if torch.equal(left, right):
        return 1.
    left, right = (x.flatten().cpu().double() for x in (left, right))
    norm = left.norm() * right.norm()
    return float((left.dot(right) / norm).clamp(-1., 1.)) if norm else 0.


def _mega_metrics(result, actual, routed_ref, shared_ref, shared_output, hidden_states,
                  ids, weights, mega, mega_weights, clamp, group):
    import deep_gemm

    tokens = actual.size(0)
    mega_output = torch.empty_like(actual, memory_format=torch.contiguous_format)
    deep_gemm.mega_moe_pre_dispatch(hidden_states[:tokens], ids[:tokens], weights[:tokens],
                                   mega.x, mega.x_sf, mega.topk_idx, mega.topk_weights,
                                   num_tokens=tokens)
    deep_gemm.fp4_fp4_mega_moe(mega_output, *mega_weights, mega, activation_clamp=clamp)
    expected = routed_ref
    cosine = {}
    errors = {}
    if shared_ref is not None:
        mega_routed = mega_output.clone()
        mega_output.add_(shared_ref)
        expected = routed_ref + shared_ref
        cosine['mega_shared_residual_cosine'] = _cosine(mega_output.float() - mega_routed.float(), shared_ref)
        cosine['gc_shared_residual_cosine'] = _cosine(actual.float() - routed_ref.float(), shared_ref)
        errors['gc_shared_max_abs'] = float((shared_output.float() - shared_ref.float()).abs().max()) if tokens else 0.
    finite = bool(torch.isfinite(mega_output).all())
    cosine.update(mega_ref_cosine=_cosine(mega_output, expected) if finite else -1.,
                  gc_ref_cosine=_cosine(actual, expected),
                  gc_mega_cosine=_cosine(actual, mega_output) if finite else -1.)
    errors.update(mega_ref_max_abs=float((mega_output.float() - expected.float()).abs().max()) if tokens else 0.,
                  gc_mega_max_abs=float((actual.float() - mega_output.float()).abs().max()) if tokens else 0.)
    lowest = torch.tensor(list(cosine.values()), dtype=torch.float64, device='cpu')
    highest = torch.tensor(list(errors.values()), dtype=torch.float64, device='cpu')
    finite_ranks = torch.tensor(int(finite), dtype=torch.int32, device='cpu')
    dist.all_reduce(lowest, op=dist.ReduceOp.MIN, group=group)
    dist.all_reduce(highest, op=dist.ReduceOp.MAX, group=group)
    dist.all_reduce(finite_ranks, op=dist.ReduceOp.SUM, group=group)
    result.update(zip(cosine, lowest.tolist()))
    result.update(zip(errors, highest.tolist()))
    result['mega_finite_ranks'] = int(finite_ranks)


def check(args, group, *, shared, readiness=False):
    import deep_gemm

    rank, ranks = group.rank(), group.size()
    calls = args.calls_per_case if readiness else 1
    vectors = [(t,) * ranks for t in args.tokens]
    if args.asymmetric:
        vectors += [tuple((rank + 1) % (args.capacity + 1) for rank in range(ranks)),
                    (args.capacity,) + (0,) * (ranks - 1)]
        if readiness and not shared and ranks > 1:
            vectors.append((min(8, args.capacity),) + (0,) * (ranks - 1))
    cases_per_round = len(vectors)
    vectors *= args.replays
    if any(t < 0 or t > args.capacity for vector in vectors for t in vector):
        raise ValueError('every token count must fit capacity')
    geometry = MODEL_PRESETS[args.model]
    hidden, intermediate, num_experts, topk = geometry
    x, ids, weights, routed, shared_weights = workload(rank, ranks, args.capacity, geometry,
                                                     shared=shared, skew=args.skew)
    if args.mask:
        ids[::2, 0] = -1
        ids[::3, min(1, topk - 1)] = num_experts
        weights[::2, min(2, topk - 1)] = 0
    packed, scales = quant_mxfp4(x)
    shared_inputs = quant_ptpc(x) if shared else None
    ref = ReferenceMoE(deep_gemm, *routed[0], *routed[1], rank, ranks)
    shared_ref = PTPCSharedExpert(deep_gemm, *shared_weights, args.capacity) if shared else None
    mega = None
    if args.compare_mega:
        mega = deep_gemm.get_symm_buffer_for_mega_moe(group, num_experts, args.capacity, topk, hidden, intermediate)
        mega_weights = deep_gemm.transform_weights_for_mega_moe(*routed)
        # Both paths ignore invalid routes. MegaMoE uses -1 for masked slots.
        mega_ids = ids.masked_fill((ids < 0) | (ids >= num_experts), -1)
        mega_inputs = (x, mega_ids)
        mega_alternate = (x.flip(0), mega_ids.flip(0)) if args.replays > 1 else None
    routed = transform_weights_for_gc_moe_ppu(*routed)
    shared_w1, shared_w2 = transform_weights_for_gc_moe_ppu(*shared_weights) if shared else (None, None)
    workspace = make_workspace(group, args.capacity, geometry, shared=shared)
    inputs = (packed, scales, ids, weights, shared_inputs)
    alternate = None
    if args.replays > 1:
        flipped_shared = None
        if shared:
            fp8, sf = shared_inputs
            flipped_shared = fp8.view(torch.uint8).flip(0).view(torch.float8_e4m3fn), sf.flip(0)
        alternate = (packed.flip(0), scales.flip(0), ids.flip(0), weights.flip(0), flipped_shared)

    out_storage = torch.empty(x.numel() + 2, dtype=x.dtype, device=x.device)
    experts = num_experts // ranks
    stats = torch.zeros(experts, dtype=torch.int32, device=x.device)
    graphs = {}
    try:
        results = []
        for index, counts in enumerate(vectors):
            if index % cases_per_round == 0:
                use_alternate = index // cases_per_round % 2
                packed, scales, ids, weights, shared_inputs = alternate if use_alternate else inputs
                if mega is not None:
                    mega_x, mega_topk = mega_alternate if use_alternate else mega_inputs
                stage(workspace, packed, scales, ids, weights, shared_inputs)
            tokens, bound = counts[rank], max(counts)
            # Exercise both the vector store and an offset contiguous output.
            offset = tokens % 2
            out = out_storage[offset:offset + x.numel()].view_as(x)
            out_storage.fill_(float('nan'))
            before = stats.clone()
            def step():
                ppu_fp4_fp4_gc_moe(
                    out[:tokens], *routed, workspace, stats, activation_clamp=args.clamp,
                    num_tokens_bound=bound, shared_l1_weights=shared_w1, shared_l2_weights=shared_w2)

            if args.graph:
                if counts not in graphs:
                    step()
                    torch.cuda.synchronize()
                    stats.copy_(before)
                    dist.barrier(group)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(calls):
                            step()
                    graphs[counts] = graph
                graphs[counts].replay()
            else:
                if args.debug:
                    torch.cuda.synchronize()
                for _ in range(calls):
                    step()
                if args.debug:
                    torch.cuda.synchronize()

            routed_expected = ref(packed[:bound], scales[:bound], ids[:bound], weights[:bound],
                                  counts, clamp=args.clamp)
            shared_expected = shared_ref(tokens, shared_inputs, clamp=args.clamp) if shared else None
            expected = routed_expected + shared_expected if shared else routed_expected
            actual = out[:tokens]
            visible = _gather(ids[:bound], ranks)
            valid = torch.arange(bound)[None, :, None] < torch.tensor(counts)[:, None, None]
            local = visible[valid.expand_as(visible) & (visible >= rank * experts)
                            & (visible < (rank + 1) * experts)]
            expected_stats = torch.bincount(local - rank * experts, minlength=experts) * calls
            if args.debug and tokens:
                received = (stats - before).cpu()
                valid_slots = (ids[:tokens] >= 0) & (ids[:tokens] < num_experts)
                returns = _unpack_down_output(workspace._returns[:tokens])
                differences = (returns != ref.slots).any(dim=-1) & valid_slots
                shared_debug = ''
                if shared:
                    shared_output = _unpack_down_output(workspace._shared_scratch[-1][:tokens])
                    error = (shared_output.float() - shared_expected.float()).abs().max().item()
                    shared_debug = f'shared max_abs={error} '
                print(f'rank={rank} {shared_debug}'
                      f'scatter_mismatches={differences.nonzero().tolist()} '
                      f'stats_match={torch.equal(received, expected_stats)}', flush=True)
            output_guard = (bool(torch.isnan(out_storage[:offset]).all())
                            and bool(torch.isnan(out_storage[offset + actual.numel():]).all()))
            shared_inputs_match = not shared or (
                torch.equal(workspace.shared_x.view(torch.uint8), shared_inputs[0].view(torch.uint8))
                and torch.equal(workspace.shared_x_sf, shared_inputs[1])
            )
            checks = torch.tensor([
                torch.equal(actual, expected), output_guard,
                torch.equal((stats - before).cpu(), expected_stats),
                bool(torch.isfinite(actual).all()), torch.equal(workspace.x, packed),
                torch.equal(workspace.x_sf, scales), torch.equal(workspace.topk_idx, ids),
                torch.equal(workspace.topk_weights, weights), shared_inputs_match,
            ], dtype=torch.int32, device='cpu')
            error = torch.tensor(float((actual.float() - expected.float()).abs().max()) if tokens else 0.,
                                 device='cpu')
            rank_counts = torch.tensor([int(checks.all()), int(checks[3])], dtype=torch.int32, device='cpu')
            dist.all_reduce(rank_counts, op=dist.ReduceOp.SUM, group=group)
            dist.all_reduce(checks, op=dist.ReduceOp.MIN, group=group)
            dist.all_reduce(error, op=dist.ReduceOp.MAX, group=group)
            result = dict(tokens=counts, max_abs=error.item(), exact=bool(checks[0]),
                          passed=bool(checks.all()), output_guard=bool(checks[1]), stats=bool(checks[2]),
                          rank_passes=int(rank_counts[0]), finite_ranks=int(rank_counts[1]))
            if mega is not None and bound:
                shared_output = _unpack_down_output(workspace._shared_scratch[-1][:tokens]) if shared else None
                _mega_metrics(result, actual, routed_expected, shared_expected, shared_output,
                              mega_x, mega_topk, weights, mega, mega_weights, args.clamp, group)
            results.append(result)
            if rank == 0:
                print(json.dumps(result), flush=True)
        return results
    finally:
        graphs.clear()
        workspace.destroy()
        if mega is not None:
            torch.cuda.synchronize()
            mega.destroy()


def main(*, shared=False, readiness=False):
    if readiness:
        variant = 'fused SE' if shared else 'routed'
        description = f'PPU {variant} readiness with alternating inputs and workspace reuse.'
    else:
        description = 'PPU fused shared-expert accuracy against grouped GEMM and PTPC GEMM.' if shared else __doc__
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model', choices=MODEL_PRESETS, default='dsv4.1-flash')
    parser.add_argument('--capacity', type=int, default=MAX_TOKENS)
    parser.add_argument('--tokens', type=int, nargs='+', default=DEFAULT_TOKENS)
    parser.add_argument('--clamp', type=float, default=10.0)
    parser.add_argument('--asymmetric', action='store_true')
    parser.add_argument('--skew', action='store_true')
    parser.add_argument('--mask', action='store_true')
    if readiness:
        parser.add_argument('--mode', choices=('graph', 'eager'), default='graph')
        parser.add_argument('--calls-per-case', type=int, default=1,
                            help='Consecutive decode calls before checking the output and receive counts.')
    else:
        parser.add_argument('--graph', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--side-stream', action='store_true')
    replay_flags = ('--replays', '--steps') if readiness else ('--replays',)
    parser.add_argument(*replay_flags, type=int, default=1, help='Repeat the cases with alternating input payloads.')
    parser.add_argument('--compare-mega', action='store_true',
                        help='Also record DeepGEMM MegaMoE accuracy against the reference and GC MoE.')
    parser.add_argument('--output-json', type=Path)
    if readiness:
        parser.set_defaults(tokens=None, asymmetric=True, mask=True, side_stream=True, replays=20)
    args = parser.parse_args()
    if readiness:
        args.graph = args.mode == 'graph'
        if args.tokens is None:
            if args.model == 'dsv4.1-flash':
                args.tokens = FLASH41_READINESS_TOKENS
            else:
                args.tokens = FLASH_SE_READINESS_TOKENS if shared else FLASH_ROUTED_READINESS_TOKENS
        if args.calls_per_case < 1:
            parser.error('expected positive calls per case')
    if not 1 <= args.capacity <= MAX_TOKENS or args.replays < 1:
        parser.error(f'expected capacity in [1, {MAX_TOKENS}] and positive replays')
    group = init_dist()
    try:
        stream = torch.cuda.Stream() if args.side_stream else torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            results = check(args, group, shared=shared, readiness=readiness)
        all_pass = all(row['passed'] for row in results)
        if args.output_json and group.rank() == 0:
            report = (dict(model=args.model, shared_expert=shared, ranks=group.size(),
                           readiness_mode=args.mode, readiness_steps=args.replays,
                           checked_cases_per_rank=len(results), all_pass=all_pass, results=results)
                      if readiness else results)
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(report, indent=2) + '\n')
        if not all_pass:
            raise SystemExit(1)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
