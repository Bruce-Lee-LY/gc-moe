"""Single-node, eight-rank DSV4.1 Flash SM90 GC MoE readiness regression.

Run from the repository root:
  python -m torch.distributed.run --standalone --nproc-per-node=8 \
      tests/test_sm90_gc_moe_readiness.py --num-processes 8
Add --mode eager for the eager control, or --steps 1 for the first A-to-B change.
The graph captures one operator call and reuses fixed input and workspace addresses.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import traceback


HIDDEN = 5120
INTERMEDIATE = 2304
EXPERTS = 384
TOPK = 6
RANKS = 8
CAPACITY = 256
ACT_BLOCK = 32
ROUTED_ACT_BLOCK = 128
ROUTED_RECIPE = (1, 1, 32)
# Like the SM90 replay regression, identical inputs must reproduce exact outputs.
TOLERANCE = 0.0


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-processes", type=int, default=8,
                        help="expected torchrun world size; this test requires eight ranks")
    parser.add_argument("--num-tokens", type=int, default=64, help="tokens per rank, 1..256")
    parser.add_argument("--steps", type=int, default=200,
                        help="alternating B,A replays after the initial A replay")
    parser.add_argument("--mode", choices=("graph", "eager"), default="graph")
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--output-dir", type=Path, default=Path("sm90_gc_moe_readiness_evidence"))
    args = parser.parse_args()
    if args.num_processes != RANKS:
        parser.error("--num-processes must be 8 for this single-node regression")
    if not 1 <= args.num_tokens <= CAPACITY:
        parser.error("--num-tokens must be between 1 and 256")
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not 0 <= args.seed < 2**63 - 2000000:
        parser.error("--seed must be in [0, 2**63 - 2000000)")
    return args


def _json_line(handle, record):
    handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()


def _exchange(group, log, report):
    _json_line(log, report)
    reports = [None] * RANKS
    dist.all_gather_object(reports, report, group=group)
    failures = [item for item in reports if item["failed"]]
    if failures:
        raise RuntimeError("rank failures: " + json.dumps(failures, sort_keys=True))


def _cast_grouped_weights(shape, generator, device):
    groups, n, k = shape
    quantized = torch.empty((groups, n, k // 2), dtype=torch.int8, device=device)
    scales = torch.empty((groups, n, k // ACT_BLOCK), dtype=torch.float32, device=device)
    for expert in range(groups):
        raw = torch.randn((n, k), dtype=torch.bfloat16, device=device, generator=generator) * 0.05
        quantized[expert], scales[expert] = deep_gemm.utils.per_token_cast_to_fp4(
            raw, use_ue8m0=True, gran_k=ACT_BLOCK)
    # SM90's transform accepts natural per-row K32 scales, before any SF packing.
    return quantized, scales


def _weights(rank, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed + rank * 104729)
    local_experts = EXPERTS // RANKS
    l1 = _cast_grouped_weights((local_experts, 2 * INTERMEDIATE, HIDDEN), generator, device)
    l2 = _cast_grouped_weights((local_experts, HIDDEN, INTERMEDIATE), generator, device)
    return gc_moe.transform_weights_for_gc_moe_sm90(l1, l2)


def _variant(variant, tokens, rank, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed + variant * 1000003 + rank * 7919)
    raw = torch.randn((tokens, HIDDEN), dtype=torch.bfloat16, device=device, generator=generator) * 0.5
    raw = raw + (0.25 if variant else -0.25)
    x_fp8, x_sf = deep_gemm.utils.per_token_cast_to_fp8(
        raw, use_ue8m0=False, gran_k=ROUTED_ACT_BLOCK)
    rows = torch.arange(tokens, dtype=torch.int64, device=device)[:, None]
    slots = torch.arange(TOPK, dtype=torch.int64, device=device)[None, :]
    # All routed work lands on rank 0; every row still selects six unique experts.
    topk_idx = (rows * 17 + slots * 17 + rank * 3 + variant * 5) % (EXPERTS // RANKS)
    topk_weights = torch.rand((tokens, TOPK), dtype=torch.float32, device=device, generator=generator)
    topk_weights += 0.1 + 0.05 * variant
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    return {"x": x_fp8, "x_sf": x_sf, "topk_idx": topk_idx, "topk_weights": topk_weights}


def _stage(buffer, data, tokens, device, group, log, label):
    try:
        buffer.x[:tokens].copy_(data["x"])
        buffer.x_sf[:tokens].copy_(data["x_sf"])
        buffer.topk_idx[:tokens].copy_(data["topk_idx"])
        buffer.topk_weights[:tokens].copy_(data["topk_weights"])
        torch.cuda.synchronize(device)
        report = {"rank": dist.get_rank(), "event": "stage", "label": label, "failed": False}
    except Exception:
        report = {"rank": dist.get_rank(), "event": "stage", "label": label,
                  "failed": True, "error": traceback.format_exc()}
    # CPU coordination makes all input copies complete before any rank launches.
    # It neither clears readiness nor synchronizes layers inside GC MoE.
    _exchange(group, log, report)


def _launch(output, l1, l2, buffer):
    gc_moe.sm90_fp8_fp4_gc_moe(
        output, l1, l2, buffer, recipe=ROUTED_RECIPE, activation="swiglu",
        activation_clamp=10.0, fast_math=True, num_tokens_bound=output.shape[0])


def _addresses(output, weights, buffer):
    tensors = {"output": output, "workspace": buffer.buffer}
    tensors.update({name: getattr(buffer, name) for name in buffer.layout.views})
    for index, components in enumerate(weights):
        for component, tensor in enumerate(components):
            tensors[f"weight_{index}_{component}"] = tensor
    return {name: tensor.data_ptr() for name, tensor in tensors.items()}


def _max_abs(actual, expected):
    delta = (actual.float() - expected.float()).abs()
    return float(delta.max().item()) if bool(torch.isfinite(delta).all().item()) else None


def _execute(output, weights, buffer, device, group, log, label, graph=None, stream=None):
    _json_line(log, {"rank": dist.get_rank(), "event": "execute_begin", "label": label})
    try:
        # Detect unwritten output elements without touching workspace signals.
        output.fill_(float("nan"))
        if graph is not None:
            graph.replay()
        elif stream is not None:
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                _launch(output, *weights, buffer)
        else:
            _launch(output, *weights, buffer)
        torch.cuda.synchronize(device)
        report = {"rank": dist.get_rank(), "event": "execute_end", "label": label, "failed": False}
    except Exception:
        report = {"rank": dist.get_rank(), "event": "execute_end", "label": label,
                  "failed": True, "error": traceback.format_exc()}
    _exchange(group, log, report)


def _check_output(output, references, variant, weights, buffer, addresses,
                  group, log, event, mode, step, replay):
    finite = bool(torch.isfinite(output.float()).all().item())
    current_error = _max_abs(output, references[variant])
    previous_error = _max_abs(output, references[1 - variant])
    fixed_addresses = _addresses(output, weights, buffer) == addresses
    report = {"rank": dist.get_rank(), "event": event, "mode": mode,
              "step": step, "replay": replay, "variant": variant, "previous_variant": 1 - variant,
              "finite": finite, "current_max_abs": current_error, "previous_max_abs": previous_error,
              "addresses_unchanged": fixed_addresses,
              "failed": (not finite or not fixed_addresses or current_error is None or
                         previous_error is None or current_error > TOLERANCE or
                         previous_error <= TOLERANCE or current_error >= previous_error)}
    _exchange(group, log, report)


def main():
    args = _arguments()
    # Parse --help before loading CUDA dependencies, then test this checkout.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    global torch, dist, deep_gemm, gc_moe
    import torch
    import torch.distributed as dist
    import deep_gemm
    import deep_gemm.utils
    import gc_moe

    if not Path(gc_moe.__file__).resolve().is_relative_to(repo_root):
        raise RuntimeError("gc_moe must be imported from the checkout under test")

    if (int(os.environ.get("WORLD_SIZE", "0")) != args.num_processes or
            int(os.environ.get("LOCAL_WORLD_SIZE", "0")) != args.num_processes or
            int(os.environ.get("GROUP_RANK", "0")) != 0 or "LOCAL_RANK" not in os.environ):
        raise SystemExit("launch with torchrun: exactly eight ranks on one node are required")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError("SM90 readiness regression requires Hopper GPUs")
    dist.init_process_group(backend="nccl", device_id=device, timeout=timedelta(minutes=5))
    rank = dist.get_rank()
    if dist.get_world_size() != RANKS or rank != local_rank:
        raise RuntimeError("exactly eight single-node ranks are required")
    host_group = dist.new_group(backend="gloo", timeout=timedelta(minutes=5))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    buffer = None
    try:
        with (args.output_dir / f"routed_rank{rank}.jsonl").open("w", encoding="utf-8") as log:
            try:
                config = {"event": "config", "test": "sm90_routed_fp8_fp4", "rank": rank,
                          "world_size": RANKS, "mode": args.mode, "steps": args.steps,
                          "num_tokens_per_rank": args.num_tokens, "hidden": HIDDEN,
                          "intermediate": INTERMEDIATE, "experts": EXPERTS, "topk": TOPK,
                          "requested_capacity_per_rank": CAPACITY, "recipe": ROUTED_RECIPE,
                          "activation": "swiglu", "activation_alpha": 1.0,
                          "activation_clamp": 10.0, "fast_math": True,
                          "seed": args.seed, "tolerance": TOLERANCE,
                          "weight_seed": args.seed + rank * 104729,
                          "variant_seeds": [args.seed + i * 1000003 + rank * 7919 for i in range(2)],
                          "reference_repeats": 3, "gc_moe_calls_per_replay": 1, "model": "dsv4.1-flash",
                          "total_tokens": args.num_tokens * RANKS}
                _json_line(log, config)
                if rank == 0:
                    print(json.dumps(config, sort_keys=True), flush=True)
                weights = _weights(rank, device, args.seed)
                buffer = gc_moe.get_symm_buffer_for_gc_moe_sm90(
                    group=dist.group.WORLD, num_experts=EXPERTS,
                    num_max_tokens_per_rank=CAPACITY, num_topk=TOPK,
                    hidden=HIDDEN, intermediate_hidden=INTERMEDIATE,
                    activation="swiglu")
                variants = [_variant(i, args.num_tokens, rank, device, args.seed) for i in range(2)]
                output = torch.empty((args.num_tokens, HIDDEN), dtype=torch.bfloat16, device=device)
                addresses = _addresses(output, weights, buffer)
                _json_line(log, {"event": "addresses", "rank": rank, "tensors": addresses,
                                 "allocated_capacity_per_rank": buffer.num_max_tokens_per_rank})

                # Three eager samples per input also warm up the exact kernel on
                # a side stream. References and graph use this same workspace.
                warmup_stream = torch.cuda.Stream(device=device)
                references = []
                for variant in range(2):
                    first = None
                    for repeat in range(3):
                        label = f"reference_{variant}_{repeat}"
                        _stage(buffer, variants[variant], args.num_tokens, device, host_group, log, label)
                        _execute(output, weights, buffer, device, host_group, log, label, stream=warmup_stream)
                        snapshot = output.clone()
                        finite = bool(torch.isfinite(snapshot.float()).all().item())
                        stability = _max_abs(snapshot, snapshot if first is None else first)
                        report = {"event": "reference", "rank": rank, "variant": variant,
                                  "repeat": repeat, "finite": finite, "stability_max_abs": stability,
                                  "failed": not finite or stability is None or stability > TOLERANCE}
                        _exchange(host_group, log, report)
                        if first is None:
                            first = snapshot
                    references.append(first)
                separation = _max_abs(references[0], references[1])
                _exchange(host_group, log, {"event": "reference_separation", "rank": rank,
                          "max_abs": separation, "failed": separation is None or separation <= 4 * TOLERANCE})

                _stage(buffer, variants[0], args.num_tokens, device, host_group, log, "capture_A")
                graph = None
                if args.mode == "graph":
                    _json_line(log, {"event": "capture_begin", "rank": rank})
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        # The entire capture body contains exactly one GC MoE call.
                        _launch(output, *weights, buffer)
                    torch.cuda.synchronize(device)
                    _exchange(host_group, log, {"event": "capture_end", "rank": rank, "failed": False})

                _execute(output, weights, buffer, device, host_group, log, "initialize_A", graph=graph)
                _check_output(output, references, 0, weights, buffer, addresses,
                              host_group, log, "initialize", args.mode, -1, 1)
                # step=0/replay=2 is the first changed-input replay: B after A.
                for step in range(args.steps):
                    variant = 1 if step % 2 == 0 else 0
                    label = f"step_{step}_replay_{step + 2}"
                    _stage(buffer, variants[variant], args.num_tokens, device, host_group, log, label)
                    _execute(output, weights, buffer, device, host_group, log, label, graph=graph)
                    _check_output(output, references, variant, weights, buffer, addresses,
                                  host_group, log, "step", args.mode, step, step + 2)
                summary = {"event": "summary", "rank": rank, "passed": True,
                           "steps_completed": args.steps, "launches_checked": args.steps + 1}
                _json_line(log, summary)
                if rank == 0:
                    print(json.dumps(summary, sort_keys=True), flush=True)
            except Exception:
                _json_line(log, {"event": "fatal", "rank": rank, "error": traceback.format_exc()})
                raise
    finally:
        if buffer is not None:
            buffer.destroy()
        dist.destroy_process_group(host_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
