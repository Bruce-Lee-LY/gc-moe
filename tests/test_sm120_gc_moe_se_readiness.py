"""Exercise multiple shared experts and replay old graphs on a reused workspace."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist

import gc_moe
from gc_moe.apis.sm120_gc_moe import validate_call
from gc_moe.layout.sm120_gc_moe import SM120GCBufferLayout
from tests.test_sm120_gc_moe import (
    build_full_weights, build_shared_weights, build_tokens, cast_grouped_weights_to_fp4,
    cast_tokens_to_fp8, dequantize_activation, make_shared_input_sf,
    reference_processed_moe_forward, reference_shared_forward,
)


def main():
    from deep_gemm.utils import per_block_cast_to_fp8, per_token_cast_to_fp8

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replays', type=int, default=1000)
    parser.add_argument('--mode', choices=('graph', 'eager'), default='graph')
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args()
    if args.replays < 1:
        parser.error('--replays must be positive')
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    dist.init_process_group('nccl', device_id=device)
    world = dist.get_world_size()
    h, i, e, capacity = 512, 256, max(8, world), 256
    assert e % world == 0
    raw = tuple(cast_grouped_weights_to_fp4(w) for w in build_full_weights(e, i, h, device))
    local = e // world
    routed = gc_moe.transform_weights_for_gc_moe_sm120(*[
        tuple(t[rank * local:(rank + 1) * local].contiguous() for t in pair) for pair in raw])
    buffer = gc_moe.get_symm_buffer_for_gc_moe_sm120(
        dist.group.WORLD, e, capacity, 1, h, i, num_shared_experts=3)
    # All routed views remain byte-for-byte at their original offsets.
    plain = SM120GCBufferLayout(world, e, capacity, 1, h, i)
    assert all(buffer.layout.views[n] == spec for n, spec in plain.views.items())
    assert buffer.shared_l2_acts.shape == (capacity, 3 * i)
    weights = {}
    for count, block_n, gran_k in ((1, 128, 128), (2, 32, 32), (3, 1, 32)):
        original = build_shared_weights(count * i, h, device)
        # 1: both packed UE8M0; 2: mixed formats; 3: both arbitrary FP32.
        cast_shared = per_token_cast_to_fp8 if block_n == 1 else per_block_cast_to_fp8
        quant = tuple(cast_shared(w, gran_k=gran_k, use_ue8m0=count == 1 or (count == 2 and j == 0))
                      for j, w in enumerate(original))
        recipe = (1, 128, 128) if gran_k == 128 else (1, 1, 32)
        shared_input = quant
        if block_n != recipe[1]:
            shared_input = tuple((data, sf.repeat_interleave(block_n, dim=0)) for data, sf in quant)
        transformed = gc_moe.transform_weights_for_gc_moe_sm120(*shared_input, shared_recipe=recipe)
        assert tuple(p[1].dtype for p in transformed) == {
            1: (torch.int32, torch.int32), 2: (torch.int32, torch.float32),
            3: (torch.float32, torch.float32)}[count]
        if count == 1:
            packed = []
            for data, sf in quant:
                codes = (sf.contiguous().view(torch.int32) >> 23).to(torch.uint8)
                codes = codes.repeat_interleave(recipe[1], dim=0)
                codes = torch.nn.functional.pad(codes, (0, -codes.size(1) % 4), value=127)
                packed.append((data, codes.view(torch.int32).t().contiguous().t()))
            control = gc_moe.transform_weights_for_gc_moe_sm120(*packed)
            for (payload, sf), (control_payload, control_sf) in zip(transformed, control):
                assert torch.equal(payload.view(torch.uint8), control_payload.view(torch.uint8))
                assert torch.equal(sf, control_sf)
            transformed = control
        weights[count] = (quant, transformed, recipe)

    # Reject malformed shared arguments before any CUDA launch.
    y_check = torch.empty((1, h), dtype=torch.bfloat16, device=device)
    valid = weights[1][1]
    negative = (
        (valid[0], None, 'paired weights'),
        (valid[0], torch.empty(1, device=device), 'missing pair'),
        (valid[0], (torch.empty((h // 64, 3, 64, 128), dtype=torch.uint8, device=device),
                    valid[1][1]), 'fractional shared expert'),
        (valid[0], (torch.empty((h // 64, 8, 64, 128), dtype=torch.uint8, device=device),
                    valid[1][1]), 'workspace exceeded'),
        (valid[0], (valid[1][0], valid[1][1][:, :0]), 'invalid scale shape'),
    )
    for w1, w2, label in negative:
        try:
            validate_call(y_check, *routed, buffer, None, (1, 1, 32), 'swiglu',
                          10., True, 1, w1, w2, (1, 128, 128))
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted {label}')

    records, captures = [], []
    cases = [(0, 0), (1, 1), (2, 17), (3, 64), (1, 256), (0, 17),
             (2, 0), (3, 17), (1, 17), (2, None)]
    for index, (count, tokens) in enumerate(cases):
        asymmetric = tokens is None
        tokens = min(capacity, 2 ** rank) if asymmetric and rank else (0 if asymmetric else tokens)
        bound = min(capacity, 2 ** (world - 1)) if asymmetric else max(1, tokens)
        x, routes, route_w = build_tokens(tokens, h, e, 1, rank, device)
        qx, sf = cast_tokens_to_fp8(x)
        buffer.x[:tokens].copy_(qx)
        buffer.x_sf[:tokens].copy_(sf)
        buffer.topk_idx[:tokens].copy_(routes)
        buffer.topk_weights[:tokens].copy_(route_w)
        ref = reference_processed_moe_forward(qx, sf, routes, route_w, *raw[0], *raw[1], i, 10.)
        kwargs = {}
        shared_ref = None
        if count:
            block_m = gc_moe.get_block_m_for_gc_moe(world, e, capacity, tokens, 1)
            make_shared_input_sf(sf, buffer.shared_l1_acts_sf, block_m)
            quant, pair, recipe = weights[count]
            kwargs = dict(shared_l1_weights=pair[0], shared_l2_weights=pair[1], shared_recipe=recipe)
            shared_ref = reference_shared_forward(dequantize_activation(qx, sf), quant, 10., recipe[2])
            ref = (ref.float() + shared_ref.float()).to(torch.bfloat16)
        if (count, tokens) == (1, 1) and not asymmetric:
            bound = None
        y = torch.empty_like(ref)
        gc_moe.sm120_fp8_fp4_gc_moe(y, *routed, buffer, activation_clamp=10.,
                             num_tokens_bound=bound, **kwargs)
        torch.cuda.synchronize()
        def cosine(a, b):
            return float(torch.nn.functional.cosine_similarity(
                a.float().flatten(), b.float().flatten(), dim=0)) if a.numel() else 1.
        cos = cosine(y, ref)
        shared_cos = cosine(buffer.shared_y[:tokens], shared_ref) if count else 1.
        metrics = torch.tensor([cos, shared_cos, float(torch.isfinite(y).all())], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.MIN)
        assert metrics[0] >= .99 and metrics[1] >= .99999 and metrics[2] == 1, metrics
        expected = y.clone()
        def launch(y=y, bound=bound, kwargs=kwargs):
            gc_moe.sm120_fp8_fp4_gc_moe(y, *routed, buffer, activation_clamp=10.,
                                 num_tokens_bound=bound, **kwargs)

        replay = launch
        if args.mode == 'graph':
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch()
            replay = graph.replay
        captures.append((replay, y, expected, qx, sf, routes, route_w))
        records.append(dict(case=index, shared_experts=count, local_tokens=tokens,
            asymmetric=asymmetric, processed_cos_min=float(metrics[0]),
            shared_cos_min=float(metrics[1])))
    # Revisit captured graphs in reverse, after other counts/formats reused
    # every shared workspace region and changed the dispatch epoch.
    for record, (replay, y, expected, qx, sf, routes, route_w) in reversed(list(zip(records, captures))):
        tokens = record['local_tokens']
        buffer.x[:tokens].copy_(qx)
        buffer.x_sf[:tokens].copy_(sf)
        if record['shared_experts']:
            block_m = gc_moe.get_block_m_for_gc_moe(world, e, capacity, tokens, 1)
            make_shared_input_sf(sf, buffer.shared_l1_acts_sf, block_m)
        buffer.topk_idx[:tokens].copy_(routes)
        buffer.topk_weights[:tokens].copy_(route_w)
        stable = True
        for _ in range(args.replays):
            replay()
            torch.cuda.synchronize()
            stable = stable and torch.equal(y, expected)
        flag = torch.tensor(int(stable), device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        assert flag == 1, record
        record.update(bitwise_stable=True, checked_replays_per_rank=args.replays, passed=True)
        if rank == 0:
            print(record, flush=True)
    if rank == 0:
        sources = ('gc_moe/gc/__init__.py', 'gc_moe/sm120_gc/__init__.py', 'gc_moe/apis/sm120_gc_moe.py',
            'gc_moe/layout/sm120_gc_moe.py', 'gc_moe/jit_kernels/impls/sm120_fp8_fp4_gc_moe.py',
            'gc_moe/impls/sm120_fp8_fp4_gc_moe.py', 'tests/test_sm120_gc_moe.py',
            'tests/test_sm120_gc_moe_se_readiness.py')
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(dict(all_pass=True, world_size=world, mode=args.mode,
            negative_cases=len(negative), results=records,
            checked_replays_per_rank=len(records) * args.replays,
            sources={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}), indent=2) + '\n')
    captures.clear()
    del replay
    if args.mode == 'graph':
        del graph
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
