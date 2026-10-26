"""SM90 fused shared experts versus GC and native Mega unfused operators.

The baseline includes both dense GEMMs, a single Triton BF16/SwiGLU/K32
quantization kernel, and the complete BF16 final addition. Preprocessing is
identical for both calls and is performed before timing/capture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import deep_gemm
import gc_moe
assert Path(gc_moe.__file__).resolve().is_relative_to(_ROOT), gc_moe.__file__
from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8
from test_sm90_gc_moe import (
    DEFAULT_TOKENS_PER_RANK, build_tokens, cast_tokens_to_fp8,
    cast_grouped_weights_to_fp4, dequantize_humming_expert, cosine, diff,
    error_metrics, package_audit,
)
from test_gc_moe_se import build_local_weights, build_shared_weights, parse_per_rank_vectors

ROUTED_RECIPE = (1, 1, 32)
SHARED_RECIPE = (1, 1, 32)
DENSE_SHARED_RECIPE = (1, 32, 32)


@triton.jit
def _shared_mid_kernel(linear, quantized, scales, I: tl.constexpr,
                       SF_STRIDE: tl.constexpr, CLAMP: tl.constexpr,
                       BLOCK: tl.constexpr):
    token = tl.program_id(0)
    group = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    col = group[:, None] * 32 + tl.arange(0, 32)[None, :]
    gate = tl.load(linear + token * (2 * I) + col, group[:, None] < I // 32, other=0).to(tl.float32)
    up = tl.load(linear + token * (2 * I) + I + col, group[:, None] < I // 32, other=0).to(tl.float32)
    gate = tl.minimum(gate, CLAMP)
    up = tl.minimum(tl.maximum(up, -CLAMP), CLAMP)
    mid = gate * (1.0 / (1.0 + tl.exp(-gate))) * up
    amax = tl.maximum(tl.max(tl.abs(mid), axis=1), 1.e-4)
    sf = tl.exp2(tl.ceil(tl.log2(amax * (1.0 / 448.0))))
    tl.store(quantized + token * I + col, (mid / sf[:, None]).to(tl.float8e4nv), group[:, None] < I // 32)
    # Directly produce DeepGEMM's MN-major, 16-byte aligned FP32 layout.
    tl.store(scales + token + group * SF_STRIDE, sf, group < I // 32)


def aligned_sf(tokens, k, device):
    return torch.empty_strided((tokens, k // 32), (1, max(4, (tokens + 3) // 4 * 4)),
                               dtype=torch.float32, device=device)


def build_weights(args, rank, world, device, reference=False):
    raw = build_local_weights(args.num_experts // world, args.intermediate_hidden,
                              args.hidden, rank, device)
    quantized = tuple(cast_grouped_weights_to_fp4(w) for w in raw)
    routed = gc_moe.transform_weights_for_gc_moe_sm90(*quantized)
    mega_routed = deep_gemm.transform_weights_for_mega_moe_sm90(*quantized, layout='tiled')
    raw_shared = build_shared_weights(args.intermediate_hidden, args.hidden,
                                     args.num_shared_experts, device)
    recipe = (1, 128, 128) if args.shared_weight_gran_k == 128 else DENSE_SHARED_RECIPE
    shared = tuple(per_block_cast_to_fp8(w, use_ue8m0=True, gran_k=recipe[2])
                   for w in raw_shared)
    # Native SM90 dense weights use N32/K32 block scales. Only scale values
    # are repeated: GLM expands 4x4, while Flash already has this recipe.
    dense_shared = tuple((w, deep_gemm.transform_sf_into_required_layout(
        sf.repeat_interleave(recipe[1] // 32, 0).repeat_interleave(recipe[2] // 32, 1).contiguous(),
        w.shape[0], w.shape[1], DENSE_SHARED_RECIPE, is_sfa=False)) for w, sf in shared)
    # The existing fused transform consumes per-row K32 scales. Broadcasting
    # the original block scales preserves every FP8 byte and dequantized value.
    fused_input = tuple((w, sf.repeat_interleave(recipe[1], 0)
                        .repeat_interleave(recipe[2] // 32, 1).contiguous()) for w, sf in shared)
    fused_shared = gc_moe.transform_weights_for_gc_moe_sm90(*fused_input, shared_recipe=SHARED_RECIPE)
    for (w, sf), (dense_w, dense_sf), (fused_w, fused_sf) in zip(shared, dense_shared, fused_input):
        assert w.data_ptr() == dense_w.data_ptr() == fused_w.data_ptr()
        assert torch.equal(dense_sf[::recipe[1] // 32, ::recipe[2] // 32], sf)
        assert torch.equal(fused_sf[::recipe[1], ::recipe[2] // 32], sf)
    result = routed, mega_routed, dense_shared, fused_shared, recipe
    return (*result, quantized) if reference else result


def allocate_buffer(args, capacity, shared, mega=False):
    if mega:
        return deep_gemm.get_symm_buffer_for_mega_moe_sm90(
            dist.group.WORLD, args.num_experts, capacity, args.num_topk,
            args.hidden, args.intermediate_hidden, activation=args.activation)
    return gc_moe.get_symm_buffer_for_gc_moe_sm90(
        dist.group.WORLD, args.num_experts, capacity, args.num_topk,
        args.hidden, args.intermediate_hidden, activation=args.activation,
        num_shared_experts=args.num_shared_experts if shared else 0)


def prepare_inputs(x):
    routed = cast_tokens_to_fp8(x)
    if x.shape[0]:
        shared = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32)
    else:
        shared = (x.to(torch.float8_e4m3fn), torch.empty((0, x.shape[1] // 32), device=x.device))
    sf = aligned_sf(x.shape[0], x.shape[1], x.device)
    sf.copy_(shared[1])
    return routed, (shared[0], sf)


def stage_inputs(buffer, routed, shared, indices, weights):
    tokens = routed[0].shape[0]
    buffer.x[:tokens].copy_(routed[0])
    buffer.x_sf[:tokens].copy_(routed[1])
    buffer.topk_idx[:tokens].copy_(indices)
    buffer.topk_weights[:tokens].copy_(weights)
    if getattr(buffer, 'num_shared_experts', 0):
        buffer.shared_l1_acts[:tokens].copy_(shared[0])
        buffer.shared_l1_acts_sf[:tokens].copy_(shared[1])


class StandaloneShared:
    def __init__(self, tokens, hidden, intermediate, device, weights, clamp):
        self.tokens, self.intermediate = tokens, intermediate
        self.weights, self.clamp = weights, clamp
        self.linear = torch.empty((tokens, intermediate * 2), dtype=torch.bfloat16, device=device)
        self.mid = torch.empty((tokens, intermediate), dtype=torch.float8_e4m3fn, device=device)
        self.sf = aligned_sf(tokens, intermediate, device)
        self.output = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device)

    def __call__(self, inputs):
        if self.tokens:
            deep_gemm.fp8_gemm_nt(inputs, self.weights[0], self.linear, recipe=DENSE_SHARED_RECIPE)
            _shared_mid_kernel[(self.tokens, triton.cdiv(self.intermediate // 32, 4))](
                self.linear, self.mid, self.sf, self.intermediate, self.sf.stride(1), self.clamp,
                BLOCK=4, num_warps=4, enable_fp_fusion=False)
            deep_gemm.fp8_gemm_nt((self.mid, self.sf), self.weights[1], self.output, recipe=DENSE_SHARED_RECIPE)
        return self.output


OPERATORS = {
    'gc_unfused': 'SM90 GC routed + standalone DeepGEMM SE + BF16 add',
    'mega_unfused': 'deep_gemm.sm90_fp8_fp4_mega_moe + standalone DeepGEMM SE + BF16 add',
    'gc_se': 'gc_moe.sm90_fp8_fp4_gc_moe fused SE',
}


def make_calls(args, tokens, bound, routed_weights, mega_weights, shared_weights, fused_weights,
               gc_buffer, mega_buffer, fused_buffer, shared_input, outputs, stats=None):
    # Both unfused paths execute this identical dense implementation and input.
    standalone = StandaloneShared(tokens, args.hidden,
        args.intermediate_hidden * args.num_shared_experts, outputs['gc_se'].device,
        shared_weights, args.activation_clamp)
    common = dict(recipe=ROUTED_RECIPE, activation=args.activation,
                  activation_clamp=args.activation_clamp, fast_math=bool(args.fast_math),
                  num_tokens_bound=bound)

    def unfused(name, operator, weights, buffer):
        operator(outputs[name], *weights, buffer,
            cumulative_local_expert_recv_stats=None if stats is None else stats[name], **common)
        shared = standalone(shared_input)
        torch.add(outputs[name], shared, out=outputs[name])

    def fused():
        gc_moe.sm90_fp8_fp4_gc_moe(outputs['gc_se'], *routed_weights, fused_buffer,
            shared_l1_weights=fused_weights[0], shared_l2_weights=fused_weights[1],
            cumulative_local_expert_recv_stats=None if stats is None else stats['gc_se'],
            shared_recipe=SHARED_RECIPE, **common)

    return {
        'gc_unfused': lambda: unfused('gc_unfused', gc_moe.sm90_fp8_fp4_gc_moe, routed_weights, gc_buffer),
        'mega_unfused': lambda: unfused('mega_unfused', deep_gemm.sm90_fp8_fp4_mega_moe, mega_weights, mega_buffer),
        'gc_se': fused,
    }, standalone


def run_independent(raw_routed, shared_weights, routed_input, shared_input,
                    indices, weights, vector, rank, clamp):
    """Distributed processed-weight FP32 math, with the baseline BF16 boundaries."""
    tokens, hidden = routed_input[0].shape
    world, capacity = len(vector), max(1, max(vector))
    def gather(tensor):
        padded = torch.zeros((capacity, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        padded[:tokens].copy_(tensor)
        peers = [torch.empty_like(padded) for _ in range(world)]
        dist.all_gather(peers, padded)
        return torch.cat([p[:t] for p, t in zip(peers, vector)])
    x = (routed_input[0].float().reshape(tokens, hidden // 128, 128)
         * routed_input[1].unsqueeze(-1)).reshape(tokens, hidden)
    all_x, all_idx, all_w = gather(x), gather(indices), gather(weights)
    routed = torch.zeros_like(all_x)
    local = raw_routed[0][0].shape[0]
    for expert in range(local):
        rows, slots = (all_idx == rank * local + expert).nonzero(as_tuple=True)
        if not rows.numel():
            continue
        w1, w2 = (dequantize_humming_expert(pair, expert) for pair in raw_routed)
        gate, up = (all_x[rows] @ w1.T).chunk(2, -1)
        gate, up = gate.clamp(max=clamp), up.clamp(-clamp, clamp)
        mid = gate * torch.sigmoid(gate) * up * all_w[rows, slots, None]
        groups = mid.reshape(-1, w2.shape[1] // 128, 128)
        sf = groups.abs().amax(-1).clamp_min(1e-10) * (1.0 / 448.0)
        q = (groups * sf.reciprocal().unsqueeze(-1)).to(torch.float8_e4m3fn)
        mid = (q.float() * sf.unsqueeze(-1)).reshape_as(mid)
        routed.index_add_(0, rows, (mid @ w2.T).to(torch.bfloat16).float())
    dist.all_reduce(routed)
    start = sum(vector[:rank])
    routed = routed[start:start + tokens].to(torch.bfloat16)
    if tokens:
        x = (shared_input[0].float().reshape(tokens, hidden // 32, 32)
             * shared_input[1].unsqueeze(-1)).reshape(tokens, hidden)
        w1, w2 = [(w.float().reshape(w.shape[0] // 32, 32, w.shape[1] // 32, 32)
                   * sf[:, None, :, None]).reshape_as(w) for w, sf in shared_weights]
        gate, up = (x @ w1.reshape(w1.shape[0], -1).T).to(torch.bfloat16).float().chunk(2, -1)
        gate, up = gate.clamp(max=clamp), up.clamp(-clamp, clamp)
        mid = gate * torch.sigmoid(gate) * up
        q, sf = per_token_cast_to_fp8(mid, use_ue8m0=True, gran_k=32)
        restored = (q.float().reshape(tokens, -1, 32) * sf.unsqueeze(-1)).reshape_as(mid)
        shared = (restored @ w2.reshape(w2.shape[0], -1).T).to(torch.bfloat16)
    else:
        shared = torch.empty_like(routed)
    return (routed.float() + shared.float()).to(torch.bfloat16), routed, shared


def dependency_audit():
    import deep_gemm._C as binary
    p = Path(binary.__file__)
    from gc_moe.runtime.sm90_gc_moe import get_context
    context = get_context()
    return dict(gc=package_audit(gc_moe), deepgemm_C=str(p),
                deepgemm_C_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                green_context=dict(compute_sms=context.compute_sms,
                                   dispatch_sms=context.dispatch_sms))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hidden', type=int, default=6144)
    p.add_argument('--intermediate-hidden', type=int, default=2048)
    p.add_argument('--num-experts', type=int, default=256)
    p.add_argument('--num-topk', type=int, default=8)
    p.add_argument('--num-shared-experts', type=int, default=1)
    p.add_argument('--shared-weight-gran-k', type=int, choices=[32, 128], default=128)
    p.add_argument('--case-category', choices=['default', 'boundary', 'asymmetric'], default='default')
    p.add_argument('--num-tokens-per-rank-list', default='')
    p.add_argument('--per-rank-tokens-list', default='')
    p.add_argument('--num-max-tokens-per-rank', type=int, default=0)
    p.add_argument('--activation', choices=['swiglu'], default='swiglu')
    p.add_argument('--activation-alpha', type=float, default=1.0)
    p.add_argument('--activation-clamp', type=float, default=10.)
    p.add_argument('--fast-math', type=int, default=1)
    p.add_argument('--cosine-threshold', type=float, default=.99)
    p.add_argument('--processed-diff-threshold', type=float, default=.002)
    p.add_argument('--gc-replays', type=int, default=5)
    p.add_argument('--all-routes-invalid', action='store_true')
    p.add_argument('--output-json', default='results/_accuracy_results_sm90_gc_se.json')
    args = p.parse_args()
    assert args.activation_alpha == 1. and args.gc_replays >= 5 and args.num_shared_experts >= 1
    vectors = parse_per_rank_vectors(args.per_rank_tokens_list)
    sweep = [int(t) for t in args.num_tokens_per_rank_list.split(',') if t.strip()] or DEFAULT_TOKENS_PER_RANK
    capacity = args.num_max_tokens_per_rank or max(1, max(max(v) for v in vectors) if vectors else max(sweep))
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group('nccl', device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    assert torch.cuda.get_device_capability(device) == (9, 0)
    assert args.num_experts % world == 0
    plan = vectors or [[t] * world for t in sweep]
    assert all(len(v) == world and min(v) >= 0 and max(v) <= capacity for v in plan)
    routed, mega_routed, shared, fused_shared, recipe, reference_routed = build_weights(args, rank, world, device, reference=True)
    base_buffer, fused_buffer = allocate_buffer(args, capacity, False), allocate_buffer(args, capacity, True)
    mega_buffer = allocate_buffer(args, capacity, False, mega=True)
    assert len({b.buffer.data_ptr() for b in (base_buffer, mega_buffer, fused_buffer)}) == 3
    # Validate malformed calls on the host; never launch a shifted kernel ABI.
    from gc_moe.apis.sm90_gc_moe import validate_call
    contract_y = torch.empty((1, args.hidden), dtype=torch.bfloat16, device=device)
    def validate_shared(y, first, second):
        return validate_call(y, *routed, fused_buffer, None, ROUTED_RECIPE,
                             args.activation, args.activation_clamp, bool(args.fast_math),
                             1, first, second, SHARED_RECIPE)
    validate_shared(contract_y, *fused_shared)
    invalid_calls = []
    for side in range(2):
        pair = fused_shared[side]
        for malformed in ((), pair[:1], (*pair, pair[1]), (pair[0], object())):
            pairs = list(fused_shared)
            pairs[side] = malformed
            invalid_calls.append((contract_y, *pairs))
    for call in invalid_calls:
        try:
            validate_shared(*call)
        except ValueError:
            pass
        else:
            raise AssertionError('SM90 validation accepted a malformed shared call')

    results = []
    for index, vector in enumerate(plan):
        tokens, bound = vector[rank], max(1, max(vector))
        x, indices, weights = build_tokens(tokens, args.hidden, args.num_experts, args.num_topk, rank, device)
        if args.all_routes_invalid:
            indices.fill_(-1)
        routed_input, shared_input = prepare_inputs(x)
        for b in (base_buffer, mega_buffer, fused_buffer):
            stage_inputs(b, routed_input, shared_input, indices, weights)
        reference, routed_ref, shared_ref = run_independent(reference_routed, shared,
            routed_input, shared_input, indices, weights, vector, rank, args.activation_clamp)
        outputs = {name: torch.empty_like(reference) for name in OPERATORS}
        stats = {name: torch.zeros(args.num_experts // world, dtype=torch.int32, device=device) for name in outputs}
        calls, standalone = make_calls(args, tokens, bound, routed, mega_routed, shared, fused_shared,
            base_buffer, mega_buffer, fused_buffer, shared_input, outputs, stats)
        records = {name: [] for name in outputs}
        for name in calls:
            calls[name]()
        torch.cuda.synchronize()
        expected = torch.bincount(indices[indices >= 0], minlength=args.num_experts).to(torch.int32)
        dist.all_reduce(expected)
        expected = expected[rank * (args.num_experts // world):(rank + 1) * (args.num_experts // world)]
        stats_ok = all(torch.equal(t, expected) for t in stats.values())
        for _ in range(args.gc_replays):
            for name in calls:
                calls[name]()
                records[name].append(outputs[name].clone())
        graphs = {}
        for name in calls:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                calls[name]()
            graphs[name] = graph
        for _ in range(args.gc_replays):
            for name in calls:
                graphs[name].replay()
                records[name].append(outputs[name].clone())
        torch.cuda.synchronize()
        repeats = []
        for i in range(len(records['gc_se'])):
            measured = {}
            for name in calls:
                output = records[name][i]
                measured[name + '_diff'] = diff(output, reference)
                measured[name + '_independent'] = cosine(output, reference)
            for name in ('gc_unfused', 'mega_unfused'):
                fused, base = records['gc_se'][i], records[name][i]
                suffix = 'gc_se_' + name
                measured[suffix + '_diff'] = diff(fused, base)
                measured[suffix + '_cosine'] = cosine(fused, base)
                measured[suffix + '_max_abs'], measured[suffix + '_relative_l2'] = error_metrics(fused, base)
            repeats.append(measured)
        cosine_keys = [k for k in repeats[0] if k.endswith(('_independent', '_cosine'))]
        diff_keys = [k for k in repeats[0] if k.endswith('_diff')]
        local = {k: min(m[k] for m in repeats) if k in cosine_keys else max(m[k] for m in repeats)
                 for k in repeats[0]}
        local.update(rank=rank, finite=all(bool(torch.isfinite(t).all()) for t in (*outputs.values(), reference)),
                     recv_stats_exact=stats_ok, repeats_exact=all(torch.equal(v, records[name][0]) for name, values in records.items() for v in values),
                     shared_diff=diff(fused_buffer.shared_y[:tokens], shared_ref),
                     standalone_shared_diff=diff(standalone.output, shared_ref),
                     shared_gc_unfused_diff=diff(fused_buffer.shared_y[:tokens], standalone.output),
                     checked_outputs_per_operator=len(records['gc_se']), repeat_checks=repeats)
        local['shared_output_abs_max'] = (float(fused_buffer.shared_y[:tokens].float().abs().max())
                                          if tokens else 0.)
        local['shared_only_output_nonzero'] = (not args.all_routes_invalid or tokens == 0
                                               or local['shared_output_abs_max'] > 0.)
        # Exercise captured changing inputs and the uint32 epoch rollover on
        # the first case of each process, including global T0 processes.
        if index == 0:
            changed_routed, changed_shared = prepare_inputs(-x * .75)
            for b in (base_buffer, mega_buffer, fused_buffer):
                stage_inputs(b, changed_routed, changed_shared, indices, weights)
            # Captured baseline reads its own dense input pair; update it too.
            shared_input[0].copy_(changed_shared[0]); shared_input[1].copy_(changed_shared[1])
            changed_ref, _, _ = run_independent(reference_routed, shared,
                changed_routed, changed_shared, indices, weights, vector, rank, args.activation_clamp)
            for seed, expected, prefix in ((0x7ffffffe, -0x7ffffffe, 'half_'),
                                            (-2, 2, '')):
                for b in (base_buffer, fused_buffer):
                    b.header[0] = seed
                for _ in range(4):
                    for name in calls:
                        graphs[name].replay()
                torch.cuda.synchronize()
                local[prefix + 'changing_input_diff'] = max(diff(t, changed_ref) for t in outputs.values())
                local[prefix + 'epoch_wrap_ok'] = all(int(b.header[0].item()) == expected
                                                      for b in (base_buffer, fused_buffer))
        del graphs, graph
        local['passed'] = (local['finite'] and local['recv_stats_exact'] and local['repeats_exact']
            and local['shared_only_output_nonzero']
            and min(local[k] for k in cosine_keys) >= args.cosine_threshold
            and max(local[k] for k in (*diff_keys, 'shared_diff', 'standalone_shared_diff', 'shared_gc_unfused_diff')) <= args.processed_diff_threshold
            and local.get('changing_input_diff', 0.) <= args.processed_diff_threshold
            and local.get('half_changing_input_diff', 0.) <= args.processed_diff_threshold
            and local.get('epoch_wrap_ok', True) and local.get('half_epoch_wrap_ok', True))
        rank_checks = [None] * world
        dist.all_gather_object(rank_checks, local)
        case = dict(case=f'asym_{index}' if vectors else f'uniform_{tokens}',
                    num_tokens_per_rank=None if vectors else tokens, tokens_per_rank_by_rank=vector,
                    num_tokens_global=sum(vector), passed=all(r['passed'] for r in rank_checks), rank_checks=rank_checks)
        results.append(case)
        if rank == 0:
            print(f"CASE {case['case']} vector={vector} "
                  f"gc_se_gc_unfused_diff={max(r['gc_se_gc_unfused_diff'] for r in rank_checks):.9g} "
                  f"gc_se_mega_unfused_diff={max(r['gc_se_mega_unfused_diff'] for r in rank_checks):.9g} "
                  f"shared_diff={max(r['shared_gc_unfused_diff'] for r in rank_checks):.9g} "
                  f"{'PASS' if case['passed'] else 'FAIL'}", flush=True)
        del calls, standalone, records, outputs, reference, routed_ref, shared_ref
        torch.cuda.synchronize(); torch.cuda.empty_cache()
    all_pass = all(c['passed'] for c in results)
    rank_audits = [None] * world
    dist.all_gather_object(rank_audits, dict(rank=rank, **dependency_audit()))
    assert len({a['gc']['source_sha256'] for a in rank_audits}) == 1
    if rank == 0:
        out = Path(args.output_json); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(world_size=world, case_category=args.case_category, config=vars(args),
            requested_capacity=capacity, routed_recipe=ROUTED_RECIPE, shared_recipe=recipe,
            shared_raw_recipe=recipe, shared_dense_recipe=DENSE_SHARED_RECIPE,
            shared_fused_recipe=SHARED_RECIPE,
            operators=OPERATORS,
            audit=rank_audits[0], rank_audits=rank_audits, all_pass=all_pass, results=results), indent=2) + '\n')
        print('ALL PASS' if all_pass else 'SOME FAILED', out, flush=True)
    torch.cuda.synchronize(); fused_buffer.destroy(); base_buffer.destroy(); mega_buffer.destroy()
    dist.barrier(); dist.destroy_process_group()
    if not all_pass:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
