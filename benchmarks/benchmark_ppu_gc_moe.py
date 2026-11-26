"""Time PPU routed decode against DeepGEMM-for-sail MegaMoE."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
from statistics import mean, median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist

from benchmarks.ppu_shared_expert import SerialPTPCSharedExpert
from gc_moe import ppu_fp4_fp4_gc_moe, transform_weights_for_gc_moe_ppu
from gc_moe.layout.ppu_gc_moe import LOCAL_RANKS
from gc_moe.ppu_gc import MAX_TOKENS
from tests.test_ppu_gc_moe import (
    DEFAULT_TOKENS, MODEL_PRESETS,
    init_dist, make_workspace, quant_mxfp4, quant_ptpc, stage, workload,
)


def measure(step, args, group):
    for _ in range(max(args.warmup, 1)):
        step()
    torch.cuda.synchronize()
    dist.barrier(group)
    graph_events = args.mode == 'graph' and group.size() > LOCAL_RANKS
    event_options = {'enable_timing': True}
    if graph_events:
        event_options['external'] = True
    start, stop = (torch.cuda.Event(**event_options) for _ in range(2))
    graph = None
    if args.mode == 'graph':
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            if graph_events:
                # A decode inside the graph rendezvous keeps host launch skew out of device timing.
                step()
                start.record()
            for _ in range(args.iters):
                step()
            if graph_events:
                stop.record()
        graph.replay()
        torch.cuda.synchronize()
    samples = []
    for _ in range(args.repeats):
        dist.barrier(group)
        if not graph_events:
            start.record()
        if graph is None:
            for _ in range(args.iters):
                step()
        else:
            graph.replay()
        if graph_events:
            torch.cuda.synchronize()
        else:
            stop.record()
            stop.synchronize()
        samples.append(1000 * start.elapsed_time(stop) / args.iters)
    slowest = torch.tensor(samples, dtype=torch.float64, device='cpu')
    dist.all_reduce(slowest, op=dist.ReduceOp.MAX, group=group)
    return {'median_us': median(slowest.tolist()), 'samples_us': slowest.tolist()}


def run(args, group, *, shared):
    rank, ranks = group.rank(), group.size()
    capacity = max(args.tokens)
    geometry = MODEL_PRESETS[args.model]
    hidden, intermediate, experts, topk = geometry
    x, ids, route_weights, routed, shared_weights = workload(rank, ranks, capacity, geometry,
                                                            shared=shared, skew=args.skew)
    shared_inputs = quant_ptpc(x) if shared else None
    output = torch.empty_like(x)
    candidate = baseline = baseline_shared = None
    baseline_name = 'mega_moe_ptpc' if shared else 'mega_moe'
    try:
        if args.arm != 'baseline':
            candidate = make_workspace(group, capacity, geometry, shared=shared)
            candidate_weights = transform_weights_for_gc_moe_ppu(*routed)
            shared_w1, shared_w2 = transform_weights_for_gc_moe_ppu(*shared_weights) if shared else (None, None)
            packed, scales = quant_mxfp4(x)
            stage(candidate, packed, scales, ids, route_weights, shared_inputs)
        if args.arm != 'gc':
            import deep_gemm

            baseline = deep_gemm.get_symm_buffer_for_mega_moe(group, experts, capacity, topk, hidden, intermediate)
            baseline_weights = deep_gemm.transform_weights_for_mega_moe(*routed)
            if shared:
                baseline_shared = SerialPTPCSharedExpert(deep_gemm, *shared_weights, capacity)

        rows = []
        for t in args.tokens:
            y, hidden_states, route_ids, weights = output[:t], x[:t], ids[:t], route_weights[:t]
            shared_step = baseline_shared.bind(t, shared_inputs, args.clamp) if baseline_shared is not None else None

            def gc_step():
                ppu_fp4_fp4_gc_moe(y, *candidate_weights, candidate, activation_clamp=args.clamp,
                                    num_tokens_bound=t, shared_l1_weights=shared_w1,
                                    shared_l2_weights=shared_w2)

            def baseline_step():
                deep_gemm.mega_moe_pre_dispatch(hidden_states, route_ids, weights, baseline.x, baseline.x_sf,
                                               baseline.topk_idx, baseline.topk_weights, num_tokens=t)
                deep_gemm.fp4_fp4_mega_moe(y, *baseline_weights, baseline, activation_clamp=args.clamp)
                if shared_step is not None:
                    y.add_(shared_step())

            row = {'tokens_per_rank': t}
            if baseline is not None:
                row[baseline_name] = measure(baseline_step, args, group)
            if candidate is not None:
                row['gc_moe'] = measure(gc_step, args, group)
            if baseline is not None and candidate is not None:
                row['speedup'] = row[baseline_name]['median_us'] / row['gc_moe']['median_us']
            rows.append(row)
            if rank == 0:
                print(json.dumps(row), flush=True)
        report = {
            'device': torch.cuda.get_device_name(), 'torch': str(torch.__version__),
            'deep_gemm': version('deep-gemm') if baseline is not None else None,
            'model': args.model, 'shared_expert': shared,
            'hidden': hidden, 'intermediate': intermediate, 'experts': experts, 'topk': topk,
            'ranks': ranks, 'mode': args.mode,
            'warmup': args.warmup, 'iterations': args.iters, 'repeats': args.repeats,
            'gc_moe_aiu': bool(candidate._aiu) if candidate is not None else None,
            'gc_moe_m_tiles': candidate._m_tiles if candidate is not None else None,
            'activation_clamp': args.clamp, 'skew': args.skew, 'results': rows,
        }
        if baseline_shared is not None:
            report['baseline_shared_swiglu'] = 'native_ptpc'
        if args.mode == 'graph' and ranks > LOCAL_RANKS:
            report['timing'] = 'graph_events_after_decode_rendezvous'
        if baseline is not None and candidate is not None:
            base_mean = mean(row[baseline_name]['median_us'] for row in rows)
            gc_mean = mean(row['gc_moe']['median_us'] for row in rows)
            report['comparison'] = {
                f'{baseline_name}_mean_us': base_mean,
                'gc_moe_mean_us': gc_mean,
                'speedup_from_mean_latency': base_mean / gc_mean,
                'mean_of_pointwise_speedups': mean(row['speedup'] for row in rows),
                'mean_latency_reduction_pct': 100 * (1 - gc_mean / base_mean),
            }
            if rank == 0:
                print(json.dumps({'comparison': report['comparison']}), flush=True)
        return report
    finally:
        if candidate is not None:
            candidate.destroy()
        if baseline is not None:
            torch.cuda.synchronize()
            baseline.destroy()


def main(*, shared=False):
    description = 'Time PPU fused SE against MegaMoE plus serial PTPC SE.' if shared else __doc__
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model', choices=MODEL_PRESETS, default='dsv4.1-flash')
    parser.add_argument('--tokens', type=int, nargs='+', default=DEFAULT_TOKENS)
    parser.add_argument('--mode', choices=('graph', 'eager'), default='graph')
    parser.add_argument('--arm', choices=('both', 'gc', 'baseline'), default='both')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--clamp', type=float, default=10.0)
    parser.add_argument('--skew', action='store_true')
    parser.add_argument('--output-json', type=Path)
    args = parser.parse_args()
    if min(args.tokens) < 1 or max(args.tokens) > MAX_TOKENS or args.warmup < 0 or min(args.iters, args.repeats) < 1:
        parser.error(f'expected 1–{MAX_TOKENS} tokens, a nonnegative warmup and positive iterations and repeats')
    group = init_dist()
    try:
        report = run(args, group, shared=shared)
        if args.output_json and group.rank() == 0:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(report, indent=2) + '\n')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
