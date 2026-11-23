"""Distributed Kimi K3 complete-output readiness regression.

Run from the repository root:
  python -m torch.distributed.run --standalone --nproc-per-node=8 \
      tests/test_gc_moe_k3_readiness.py --num-processes 8
Multi-node runs use the world size set by torchrun.
Both controlled and continuous submissions are checked by default.
Add --mode eager for the eager control or --schedule controlled for the synchronized case.
This file is standalone and uses no other test or benchmark.
The graph captures one operator call with fixed input and final output addresses.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import traceback
import sys
import time
from math import isfinite


HIDDEN = 3584
INTERMEDIATE = 3072
EXPERTS = 896
TOPK = 16
SHARED_HIDDEN = 7168
SHARED_INTERMEDIATE = 6144
SHARED_EXPERTS = 2
CAPACITY = 256
ACT_BLOCK = 32
ROUTED_RECIPE = (1, 1, 32)
VARIANT_COUNT = 3
# Exact BF16 agreement is required after reference reproducibility is established.
READINESS_TOLERANCE = 0.0
SEPARATION_CHUNK = 128
MIN_REFERENCE_GAP = 1 / 256


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-processes", type=int,
                        help="optional expected torchrun world size")
    parser.add_argument("--num-tokens", type=int, default=64, help="tokens per rank, 1..256")
    parser.add_argument("--steps", type=int, default=200,
                        help="three-input changes per submission schedule after initialization")
    parser.add_argument("--mode", choices=("graph", "eager"), default="graph")
    parser.add_argument("--schedule", choices=("both", "controlled", "continuous"), default="both")
    parser.add_argument("--rank-delay-ms", type=float, default=5.0,
                        help="rotating-rank host delay every 16 continuous submissions")
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--output-dir", type=Path, default=Path("gc_moe_k3_readiness_evidence"))
    args = parser.parse_args()
    if args.num_processes is not None and args.num_processes < 2:
        parser.error("--num-processes must be at least 2")
    if not 1 <= args.num_tokens <= CAPACITY:
        parser.error("--num-tokens must be between 1 and 256")
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not 0 <= args.seed < 2**63 - 4000000:
        parser.error("--seed must be in [0, 2**63 - 4000000)")
    if not isfinite(args.rank_delay_ms) or args.rank_delay_ms < 0:
        parser.error("--rank-delay-ms must be finite and nonnegative")
    return args


def _json_line(handle, record):
    handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()


def _exchange(group, log, report):
    _json_line(log, report)
    reports = [None] * dist.get_world_size(group)
    dist.all_gather_object(reports, report, group=group)
    failures = [item for item in reports if item["failed"]]
    if failures:
        raise RuntimeError("rank failures: " + json.dumps(failures, sort_keys=True))


def _check_invalid_layout(rank, host_group, log):
    try:
        gc_moe.GCK3SymmBuffer(
            dist.group.WORLD, EXPERTS, CAPACITY, TOPK, HIDDEN,
            0 if rank == 0 else INTERMEDIATE, num_shared_experts=SHARED_EXPERTS,
            shared_hidden=SHARED_HIDDEN, shared_intermediate_hidden=SHARED_INTERMEDIATE,
        )
    except ValueError as exc:
        error = str(exc)
    else:
        error = ""
    _exchange(host_group, log, {
        "event": "invalid_workspace", "rank": rank, "error": error,
        "failed": error != "K3 workspace configuration failed on group ranks [0]",
    })


def _cast_grouped_weights(shape, generator, device):
    groups, n, k = shape
    # Each byte holds two FP4 E2M1 values; SF is per-row K32 UE8M0.
    quantized = torch.empty((groups, n, k // 2), dtype=torch.int8, device=device)
    scales = torch.empty((groups, n, k // ACT_BLOCK), dtype=torch.float32, device=device)
    for expert in range(groups):
        raw = torch.randn((n, k), dtype=torch.bfloat16, device=device, generator=generator) * 0.05
        quantized[expert], scales[expert] = deep_gemm.utils.per_token_cast_to_fp4(
            raw, use_ue8m0=True, gran_k=ACT_BLOCK)
    packed = deep_gemm.transform_sf_into_required_layout(scales, n, k, (1, ACT_BLOCK), groups)
    return quantized, packed


def _weights(rank, world_size, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed + rank * 104729)
    local_experts = EXPERTS // world_size
    l1 = _cast_grouped_weights((local_experts, 2 * INTERMEDIATE, HIDDEN), generator, device)
    l2 = _cast_grouped_weights((local_experts, HIDDEN, INTERMEDIATE), generator, device)
    l1, l2 = gc_moe.transform_weights_for_gc_moe_k3(l1, l2, activation="situ")
    shared_generator = torch.Generator(device=device).manual_seed(seed + 1)
    shared_l1 = torch.randn((2 * SHARED_INTERMEDIATE, SHARED_HIDDEN), dtype=torch.bfloat16,
                           device=device, generator=shared_generator) * 0.02
    shared_l2 = torch.randn((SHARED_HIDDEN, SHARED_INTERMEDIATE), dtype=torch.bfloat16,
                           device=device, generator=shared_generator) * 0.02
    shared_l1, shared_l2 = gc_moe.transform_weights_for_gc_moe_k3(
        shared_l1, shared_l2, activation="situ")
    norm = 1 + 0.1 * torch.randn(HIDDEN, dtype=torch.bfloat16, device=device,
                                  generator=shared_generator)
    projection = torch.randn((SHARED_HIDDEN, HIDDEN), dtype=torch.bfloat16, device=device,
                             generator=shared_generator) * HIDDEN ** -0.5
    front = torch.randn((EXPERTS + HIDDEN, SHARED_HIDDEN), dtype=torch.bfloat16,
                        device=device, generator=shared_generator) * SHARED_HIDDEN ** -0.5
    # Force remote readiness: the bias puts all Top16 experts on EP rank 0.
    bias = torch.full((EXPERTS,), -2.0, dtype=torch.float32, device=device)
    bias[:local_experts] = 2.0
    return l1, l2, shared_l1, shared_l2, norm, projection, front, bias


def _variant(variant, tokens, rank, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed + variant * 1000003 + rank * 7919)
    return (torch.randn((tokens, SHARED_HIDDEN), dtype=torch.bfloat16, device=device,
                        generator=generator) * 0.5 + ((variant - 1) * 0.25))


def _stage(hidden_states, data, device, group, log, label):
    try:
        hidden_states.copy_(data)
        torch.cuda.synchronize(device)
        report = {"rank": dist.get_rank(), "event": "stage", "label": label, "failed": False}
    except Exception:
        report = {"rank": dist.get_rank(), "event": "stage", "label": label,
                  "failed": True, "error": traceback.format_exc()}
    # CPU coordination makes all input copies complete before any rank launches.
    # It neither clears readiness nor synchronizes layers inside GC MoE K3.
    _exchange(group, log, report)


def _launch(output, l1, l2, shared_l1, shared_l2, norm, projection, front, bias, prefix, buffer, hidden_states):
    gc_moe.fp8_fp4_gc_moe_k3(
        output, l1, l2, buffer, hidden_states=hidden_states,
        front_weight=front, router_bias=bias,
        shared_l1_weights=shared_l1, shared_l2_weights=shared_l2,
        norm_weight=norm, up_proj_weight=projection, prefix_sum=prefix,
        recipe=ROUTED_RECIPE, activation="situ", activation_beta=4.0,
        activation_linear_beta=25.0, activation_clamp=None, fast_math=True)


def _addresses(output, weights, buffer, hidden_states):
    tensors = {"output": output, "workspace": buffer.buffer, "hidden_states": hidden_states}
    for name in ("x", "x_sf", "topk_idx", "topk_weights", "l1_acts", "l1_acts_sf",
                 "l2_acts", "l2_acts_sf", "shared_l2_acts", "routed_latent", "shared_output", "front_logits"):
        tensors[name] = getattr(buffer, name)
    for index, item in enumerate(weights):
        if isinstance(item, tuple):
            tensors[f"weight_{index}"] = item[0]
            tensors[f"weight_sf_{index}"] = item[1]
        else:
            tensors[f"weight_{index}"] = item
    return {name: tensor.data_ptr() for name, tensor in tensors.items()}


def _enqueue(output, weights, buffer, hidden_states, graph):
    # Only public outputs are poisoned; workspace readiness is never reset.
    output.fill_(float("nan"))
    if graph is not None:
        graph.replay()
    else:
        _launch(output, *weights, buffer, hidden_states)


def _execute(output, weights, buffer, hidden_states, device, group, log, label, graph=None, stream=None):
    _json_line(log, {"rank": dist.get_rank(), "event": "execute_begin", "label": label})
    try:
        if stream is not None:
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                _enqueue(output, weights, buffer, hidden_states, graph)
        else:
            _enqueue(output, weights, buffer, hidden_states, graph)
        torch.cuda.synchronize(device)
        report = {"rank": dist.get_rank(), "event": "execute_end", "label": label, "failed": False}
    except Exception:
        report = {"rank": dist.get_rank(), "event": "execute_end", "label": label,
                  "failed": True, "error": traceback.format_exc()}
    _exchange(group, log, report)


def _max_abs(actual, expected):
    delta = (actual.float() - expected.float()).abs()
    return float(delta.max().item()) if bool(torch.isfinite(delta).all().item()) else None


def _reference_separation(references, tokens, rank):
    report = {"event": "reference_separation", "rank": rank, "pairs": {}, "failed": False}
    # Each W2-width segment of every token must distinguish every input pair.
    # A large difference elsewhere must not hide an indistinguishable token.
    for left in range(VARIANT_COUNT):
        for right in range(left + 1, VARIANT_COUNT):
            delta = (references[left].float() - references[right].float()).abs()
            gap = (float(delta.reshape(tokens, -1, SEPARATION_CHUNK).amax(-1).amin().item())
                   if bool(torch.isfinite(delta).all().item()) else None)
            report["failed"] |= gap is None or gap <= MIN_REFERENCE_GAP
            report["pairs"][f"{left}-{right}"] = gap
    return report


def _context_report(context, expected_primary, requested, rank):
    actual = bool(context.primary_ctx_roles)
    return {"event": "context", "rank": rank, "primary_ctx_requested": requested,
            "primary_ctx_expected": expected_primary, "primary_ctx_roles": actual,
            "dispatch_sms": int(context.dispatch_sms), "compute_sms": int(context.compute_sms),
            "failed": actual != expected_primary}


def _check_output(output, references, variant, tokens, addresses_unchanged,
                  rank, event, mode, schedule, step, replay, generation):
    report = {"event": event, "rank": rank, "mode": mode, "schedule": schedule,
              "step": step, "replay": replay, "generation": generation, "variant": variant,
              "addresses_unchanged": addresses_unchanged, "failed": not addresses_unchanged}
    current_error = _max_abs(output, references[variant])
    other_errors = {str(old): _max_abs(output, references[old])
                    for old in range(VARIANT_COUNT) if old != variant}
    report.update(finite=current_error is not None, current_max_abs=current_error,
                  other_max_abs=other_errors)
    report["failed"] |= (
        current_error is None or current_error > READINESS_TOLERANCE
        or any(error is None or error <= READINESS_TOLERANCE for error in other_errors.values())
    )
    return report


def _run_continuous(args, output, weights, buffer, hidden_states, variants, references,
                    graph, addresses, device, group, log, start_generation):
    # Snapshot copies consume each result on the SAME caller stream before
    # the fixed output address is poisoned/reused. No collective, .item(), or
    # GPU synchronization is allowed between submissions in this loop.
    snapshots = torch.empty((args.steps + 1, *output.shape), dtype=output.dtype, device=device)
    _exchange(group, log, {"event": "batch_begin", "rank": dist.get_rank(),
                          "schedule": "continuous", "failed": False})
    delayed = []
    addresses_unchanged = True
    for replay in range(args.steps + 1):
        generation = start_generation + replay
        variant = generation % VARIANT_COUNT
        if replay % 16 == 0 and dist.get_rank() == (replay // 16) % args.num_processes:
            time.sleep(args.rank_delay_ms / 1000)
            delayed.append(replay)
        hidden_states.copy_(variants[variant])
        _enqueue(output, weights, buffer, hidden_states, graph)
        snapshots[replay].copy_(output)
        addresses_unchanged &= _addresses(output, weights, buffer, hidden_states) == addresses
    torch.cuda.synchronize(device)
    _exchange(group, log, {"event": "batch_end", "rank": dist.get_rank(),
                          "schedule": "continuous", "submissions": args.steps + 1,
                          "delayed_replays": delayed, "rank_delay_ms": args.rank_delay_ms,
                          "addresses_unchanged": addresses_unchanged,
                          "failed": not addresses_unchanged})
    # Host checks/collectives happen only after every queued result is saved.
    for replay in range(args.steps + 1):
        generation = start_generation + replay
        report = _check_output(snapshots[replay], references, generation % VARIANT_COUNT,
                               args.num_tokens, addresses_unchanged, dist.get_rank(),
                               "initialize" if replay == 0 else "step", args.mode,
                               "continuous", replay - 1, replay + 1, generation)
        _exchange(group, log, report)


def main():
    args = _arguments()
    # Keep --help usable without loading CUDA dependencies.
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
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    if world_size < 2 or "LOCAL_RANK" not in os.environ:
        raise SystemExit("launch with torchrun and at least two ranks")
    if args.num_processes is not None and args.num_processes != world_size:
        raise SystemExit("--num-processes must match torchrun WORLD_SIZE")
    args.num_processes = world_size
    if EXPERTS % world_size or EXPERTS // world_size < TOPK:
        raise SystemExit("the readiness fixture needs at least TopK experts on each rank")
    requested_primary = os.environ.get("GC_MOE_PRIMARY_CTX", "0")
    expected_primary = int(requested_primary) != 0
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3)):
        raise RuntimeError("SM100/103 hardware required")
    dist.init_process_group(backend="nccl", device_id=device, timeout=timedelta(minutes=5))
    rank = dist.get_rank()
    host_group = dist.new_group(backend="gloo", timeout=timedelta(minutes=5))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    buffer = None
    graph = None
    checks_complete = False
    schedules = ("controlled", "continuous") if args.schedule == "both" else (args.schedule,)
    with (args.output_dir / f"kimi_k3_rank{rank}.jsonl").open("w", encoding="utf-8") as log:
        try:
            from gc_moe.runtime import gc_moe_k3 as runtime
            context = runtime.get_context(num_ranks=world_size)
            _exchange(host_group, log, _context_report(context, expected_primary, requested_primary, rank))
            config = {"event": "config", "test": "kimi_k3_full_output", "rank": rank,
                      "world_size": world_size, "model": "kimi-k3", "mode": args.mode,
                      "steps": args.steps, "schedules": schedules, "variants": VARIANT_COUNT,
                      "num_tokens_per_rank": args.num_tokens, "total_tokens": args.num_tokens * world_size,
                      "hidden": HIDDEN, "intermediate": INTERMEDIATE, "experts": EXPERTS, "topk": TOPK,
                      "requested_capacity_per_rank": CAPACITY, "recipe": ROUTED_RECIPE,
                      "activation": "situ", "activation_beta": 4.0, "activation_linear_beta": 25.0,
                      "activation_clamp": None, "fast_math": True, "seed": args.seed,
                      "readiness_tolerance": READINESS_TOLERANCE,
                      "separation_chunk": SEPARATION_CHUNK, "min_reference_gap": MIN_REFERENCE_GAP,
                      "variant_seeds": [args.seed + i * 1000003 + rank * 7919 for i in range(VARIANT_COUNT)],
                      "weight_seed": args.seed + rank * 104729,
                      "reference_repeats": 3, "gc_moe_calls_per_replay": 1,
                      "rank_delay_ms": args.rank_delay_ms,
                      "compute_capability": torch.cuda.get_device_capability(device),
                      "device_name": torch.cuda.get_device_name(device),
                      "gc_module": gc_moe.__file__, "quantization_module": deep_gemm.__file__,
                      "shared_hidden": SHARED_HIDDEN, "shared_intermediate": SHARED_INTERMEDIATE,
                      "shared_experts": SHARED_EXPERTS, "shared_weight_dtype": "bfloat16"}
            _json_line(log, config)
            if rank == 0:
                print(json.dumps(config, sort_keys=True), flush=True)
            prefix_generator = torch.Generator(device=device).manual_seed(args.seed + rank)
            prefix = torch.randn((args.num_tokens, SHARED_HIDDEN), device=device,
                                 dtype=torch.bfloat16, generator=prefix_generator) * 0.125
            _check_invalid_layout(rank, host_group, log)
            weights = (*_weights(rank, world_size, device, args.seed), prefix)
            buffer = gc_moe.GCK3SymmBuffer(
                group=dist.group.WORLD, num_experts=EXPERTS, num_max_tokens_per_rank=CAPACITY,
                num_topk=TOPK, hidden=HIDDEN, intermediate_hidden=INTERMEDIATE,
                mma_type="fp8xfp4", activation="situ" , num_shared_experts=SHARED_EXPERTS,
                shared_hidden=SHARED_HIDDEN, shared_intermediate_hidden=SHARED_INTERMEDIATE)
            hidden_states = torch.empty((args.num_tokens, SHARED_HIDDEN), dtype=torch.bfloat16, device=device)
            variants = [_variant(i, args.num_tokens, rank, device, args.seed) for i in range(VARIANT_COUNT)]
            output = torch.empty((args.num_tokens, SHARED_HIDDEN), dtype=torch.bfloat16, device=device)
            addresses = _addresses(output, weights, buffer, hidden_states)
            _json_line(log, {"event": "addresses", "rank": rank, "tensors": addresses,
                             "allocated_capacity_per_rank": buffer.num_max_tokens_per_rank})

            # Freeze three eager references per input on the same workspace.
            # Fail on instability instead of widening readiness tolerance.
            warmup_stream = torch.cuda.Stream(device=device)
            references = []
            for variant in range(VARIANT_COUNT):
                first = None
                for repeat in range(3):
                    label = f"reference_{variant}_{repeat}"
                    _stage(hidden_states, variants[variant], device, host_group, log, label)
                    _execute(output, weights, buffer, hidden_states, device, host_group, log, label,
                             stream=warmup_stream)
                    snapshot = output.clone()
                    stability = _max_abs(snapshot, snapshot if first is None else first)
                    _exchange(host_group, log, {"event": "reference", "rank": rank, "variant": variant,
                              "repeat": repeat, "stability_max_abs": stability,
                              "failed": stability is None or stability > READINESS_TOLERANCE})
                    if first is None:
                        first = snapshot
                references.append(first)
            _exchange(host_group, log, _reference_separation(references, args.num_tokens, rank))

            _stage(hidden_states, variants[0], device, host_group, log, "capture_0")
            if args.mode == "graph":
                _json_line(log, {"event": "capture_begin", "rank": rank})
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    # Exactly one K3 call; staging and output poison stay outside capture.
                    _launch(output, *weights, buffer, hidden_states)
                torch.cuda.synchronize(device)
                _exchange(host_group, log, {"event": "capture_end", "rank": rank, "failed": False})

            generation = 0
            for schedule in schedules:
                if schedule == "continuous":
                    _run_continuous(args, output, weights, buffer, hidden_states, variants,
                                    references, graph, addresses, device, host_group, log, generation)
                else:
                    for replay in range(args.steps + 1):
                        current_generation = generation + replay
                        variant = current_generation % VARIANT_COUNT
                        label = f"controlled_{replay}"
                        _stage(hidden_states, variants[variant],
                               device, host_group, log, label)
                        _execute(output, weights, buffer, hidden_states, device, host_group, log, label, graph=graph)
                        fixed = _addresses(output, weights, buffer, hidden_states) == addresses
                        report = _check_output(output, references, variant, args.num_tokens,
                                               fixed, rank, "initialize" if replay == 0 else "step",
                                               args.mode, schedule, replay - 1, replay + 1, current_generation)
                        _exchange(host_group, log, report)
                generation += args.steps + 1
            checks_complete = True
            _json_line(log, {"event": "checks_complete", "rank": rank, "launches_checked": generation})
        except Exception:
            _json_line(log, {"event": "fatal", "rank": rank, "error": traceback.format_exc()})
            raise
        finally:
            cleanup = {"event": "cleanup", "rank": rank, "failed": False}
            try:
                # Captured K3 nodes retain storage until the outer graph retires.
                if graph is not None:
                    torch.cuda.synchronize(device)
                    graph.reset()
                    graph = None
                if buffer is not None:
                    buffer.destroy()
                    buffer = None
            except Exception:
                cleanup.update(failed=True, error=traceback.format_exc())
            try:
                if checks_complete:
                    _exchange(host_group, log, cleanup)
                else:
                    _json_line(log, cleanup)
                if cleanup["failed"]:
                    raise RuntimeError("K3 readiness cleanup failed: " + cleanup["error"])
            finally:
                try:
                    dist.destroy_process_group(host_group)
                finally:
                    dist.destroy_process_group()
        # Reached only after checks, graph/workspace retirement, and group teardown succeed.
        summary = {"event": "summary", "rank": rank, "passed": True,
                   "steps_completed": args.steps * len(schedules), "launches_checked": generation,
                   "schedules": {s: {"steps_completed": args.steps, "launches_checked": args.steps + 1}
                                 for s in schedules}, "cleanup_complete": True}
        _json_line(log, summary)
        if rank == 0:
            print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
