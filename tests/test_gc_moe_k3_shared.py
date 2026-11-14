"""K3 EP8 routed + BF16 shared 的最小数值与 CUDA Graph 回放检查。

默认只验证；加 --benchmark 测GC三role加外置TP2 RS，逐调用L2 cold，计时不含AG和输入stage。
先取得八卡 gpu-run 锁：
  NCCL_NVLS_ENABLE=0 python -m torch.distributed.run --standalone \
      --nproc-per-node=8 tests/test_gc_moe_k3_shared.py --steps 8

原 DeepGEMM MegaMoE SiTU 是 routed 参考；shared 使用独立 PyTorch BF16/SiTU 参考。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


TOKENS = 128
ROUTED_HIDDEN = 3584
SHARED_HIDDEN = 7168
INTERMEDIATE = 3072
EXPERTS = 896
TOPK = 16
RANKS = 8
CAPACITY = 256


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8,
                        help="交替输入 A/B 的 graph 回放次数；不包括 eager 两次检查")
    parser.add_argument("--mode", choices=("graph", "eager"), default="graph")
    parser.add_argument("--flat-shared-tokens", type=int,
                        help="直接验证本卡连续[n,7168]输入与partial，不执行shared AG/RS")
    parser.add_argument("--shared-capacity", type=int,
                        help="shared工作区行容量，至少等于实际输入行数")
    parser.add_argument("--shared-intermediate", type=int, default=INTERMEDIATE,
                        help="本卡shared权重的中间维；正的64倍数")
    parser.add_argument("--tp-pairs", default="0,1;2,3;4,5;6,7",
                        help="每个 pair 的顺序决定两端 partial 的 canonical owner 行；RS 只返回本 owner")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/k3_shared_validation"))
    parser.add_argument("--benchmark", action="store_true",
                        help="验证后测GC三role+外置TP2 RS，逐调用L2 cold；不含AllGather/输入stage")
    parser.add_argument("--warmup", type=int, default=10, help="计时前的独立冷调用次数")
    parser.add_argument("--iters", type=int, default=20, help="计时轮数")
    parser.add_argument("--inner", type=int, default=20, help="每轮独立冷调用次数；每次Graph仅一个GC+RS调用")
    args = parser.parse_args()
    if args.flat_shared_tokens is not None and args.flat_shared_tokens < 1:
        parser.error("--flat-shared-tokens 必须为正")
    shared_tokens = args.flat_shared_tokens if args.flat_shared_tokens is not None else 2 * TOKENS
    if args.shared_capacity is not None and args.shared_capacity < shared_tokens:
        parser.error("--shared-capacity 不能小于实际shared输入行数")
    if args.shared_intermediate < 64 or args.shared_intermediate % 64:
        parser.error("--shared-intermediate 必须为正的64倍数")
    if args.flat_shared_tokens is not None and args.benchmark:
        parser.error("flat shared模式用于数值验证，不复用TP2 RS的计时范围")
    if args.steps < 2:
        parser.error("--steps 至少为 2，必须覆盖 A→B→A")
    if args.warmup < 0 or args.iters < 1 or args.inner < 1:
        parser.error("--warmup 不得为负，--iters 和 --inner 必须为正")
    if args.benchmark and args.mode != "graph":
        parser.error("--benchmark 需要 --mode graph，先验证 graph 回放")
    try:
        args.tp_pairs = tuple(tuple(int(i) for i in pair.split(","))
                              for pair in args.tp_pairs.split(";"))
    except ValueError:
        parser.error("--tp-pairs 应形如 0,1;2,3;4,5;6,7")
    if (len(args.tp_pairs) != 4 or any(len(pair) != 2 for pair in args.tp_pairs)
            or sorted(i for pair in args.tp_pairs for i in pair) != list(range(RANKS))):
        parser.error("--tp-pairs 必须将 0..7 恰好划分为四对")
    return args


def _write(log, value):
    log.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    log.flush()


def _exchange(host_group, log, value):
    _write(log, value)
    reports = [None] * RANKS
    dist.all_gather_object(reports, value, group=host_group)
    bad = [report for report in reports if not report.get("passed", True)]
    if bad:
        raise AssertionError(json.dumps(bad, ensure_ascii=False, allow_nan=False))


def _source_identity(root):
    # 记录未提交修改，避免仅凭git SHA混淆不同源码。
    sources = sorted((root / "gc_moe").rglob("*.py")) + [
        Path(__file__).resolve()]
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sources}
    git_sha = subprocess.check_output(
        ["git", "-c", f"safe.directory={root}", "rev-parse", "HEAD"],
        cwd=root, text=True).strip()
    return {"git_sha": git_sha, "source_sha256": hashes,
            "source_tree_sha256": hashlib.sha256(
                json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def _latency_stats(samples):
    ordered = sorted(samples)

    def percentile(fraction):
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {"p50_us": percentile(0.5), "p90_us": percentile(0.9),
            "min_us": ordered[0], "max_us": ordered[-1]}


def _benchmark(launch, args, device, host_group, rank, log, metadata):
    # 每张图只有一次GC三role与外部RS；反序pair的输入重排也在launch内。
    # AG、输入stage、prepare、capture、驱逐及跨rank同步均在计时区之外。
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    dist.barrier(group=host_group)
    with torch.cuda.graph(graph, stream=stream):
        launch()
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)

    eviction_bytes = 506 * 1024 * 1024
    eviction = torch.ones(eviction_bytes // 4, dtype=torch.float32, device=device)
    sink = torch.empty((), dtype=torch.float32, device=device)

    def cold():
        torch.sum(eviction, dim=0, out=sink)
        torch.cuda.current_stream(device).synchronize()
        # 每个rank先完成本地驱逐，host barrier再保证八个rank都已完成。
        dist.barrier(group=host_group)

    for _ in range(args.warmup):
        cold()
        graph.replay()
        torch.cuda.current_stream(device).synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for iteration in range(args.iters):
        for inner in range(args.inner):
            cold()
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            elapsed_us = start.elapsed_time(end) * 1000.0
            samples.append(elapsed_us)
            _write(log, {"event": "benchmark_iteration", "rank": rank,
                         "iteration": iteration, "inner": inner,
                         "full_call_us": elapsed_us})

    all_samples = [None] * RANKS
    dist.all_gather_object(all_samples, samples, group=host_group)
    # 先逐调用取八卡最大值，再求分位数。
    rank_max = [max(values) for values in zip(*all_samples)]
    report = {
        "event": "benchmark", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metric": "complete_gc_plus_external_tp2_rs_l2_cold_cuda_graph_us",
        "metadata": metadata,
        "scope": "GC routed EP8 plus shared BF16 FC1/SiTU/FC2; all three roles; "
                 "external TP2 RS and owner reorder included; AG and input stage excluded",
        "warmup_cold_calls": args.warmup, "iterations": args.iters,
        "cold_calls_per_iteration": args.inner,
        "timed_graph_replays": args.iters * args.inner,
        "complete_calls_per_graph": 1,
        "cache_policy": "L2 cold before every independent call; no intra-call eviction",
        "eviction_bytes_per_rank": eviction_bytes,
        "eviction_method": "float32 read reduction into persistent scalar",
        "pre_start_rendezvous": "current-stream synchronize then EP8 Gloo barrier",
        "per_rank": [{"rank": i, "samples_us": values, **_latency_stats(values)}
                     for i, values in enumerate(all_samples)],
        "cross_rank_max": {"samples_us": rank_max, **_latency_stats(rank_max)},
    }
    _write(log, {"event": "benchmark_summary", "rank": rank,
                 **_latency_stats(samples), "cross_rank_max": _latency_stats(rank_max)})
    # NCCL Graph资源先于进程组退出；计时结果不依赖后续Graph存活。
    del graph
    return report


def _metrics(actual, expected, *, atol, rtol, relative_l2_limit,
             elementwise_tolerance=None):
    metadata_ok = actual.shape == expected.shape and actual.dtype == expected.dtype
    if not metadata_ok:
        return {"passed": False, "actual_shape": list(actual.shape),
                "expected_shape": list(expected.shape), "actual_dtype": str(actual.dtype),
                "expected_dtype": str(expected.dtype)}
    a, b = actual.float().reshape(-1), expected.float().reshape(-1)
    finite = bool((torch.isfinite(a).all() & torch.isfinite(b).all()).item())
    if not finite:
        return {"passed": False, "finite": False}
    difference = a - b
    relative_l2 = float((difference.norm() / b.norm().clamp_min(1e-30)).item())
    denom = a.norm() * b.norm()
    cosine = (float((torch.dot(a, b) / denom).item()) if float(denom.item())
              else float(torch.equal(actual, expected)))
    tolerance = (atol + rtol * b.abs() if elementwise_tolerance is None
                 else elementwise_tolerance.float().reshape(-1))
    outside = int((difference.abs() > tolerance).sum().item())
    result = {"passed": outside == 0 and relative_l2 <= relative_l2_limit,
            "finite": True, "max_abs": float(difference.abs().max().item()),
            "relative_l2": relative_l2, "cosine": cosine,
            "outside_elementwise_tolerance": outside,
            "atol": atol, "rtol": rtol, "relative_l2_limit": relative_l2_limit}
    if elementwise_tolerance is not None:
        result["elementwise_tolerance_max"] = float(tolerance.max().item())
        result["elementwise_tolerance_mean"] = float(tolerance.mean().item())
    return result


def _exact_bf16_metrics(actual, expected):
    result = _metrics(actual, expected, atol=0.0, rtol=0.0, relative_l2_limit=0.0)
    if actual.shape == expected.shape and actual.dtype == expected.dtype:
        # 数值相等不能区分 +0/-0；RS 契约要求 BF16 bits 也完全相同。
        mismatches = int((actual.view(torch.int16) != expected.view(torch.int16)).sum().item())
        result["bitwise_mismatches"] = mismatches
        result["passed"] = result["passed"] and mismatches == 0
    return result


def _flat_shared_reference(x, w13, w2):
    intermediate = w2.shape[1]
    fc1 = torch.mm(x, w13.T)  # BF16 输出，保留原模型 FC1 的舍入边界。
    gate, up = fc1[:, :intermediate].float(), fc1[:, intermediate:].float()
    # K3 SiTU：sigmoid 使用原 gate，beta 只用于 tanh softcap。
    gate = (4.0 * torch.tanh(gate / 4.0)) * torch.sigmoid(gate)
    up = 25.0 * torch.tanh(up / 25.0)
    activation = (gate * up).to(torch.bfloat16)
    partial = torch.mm(activation, w2.T)
    return {"input": x, "activation": activation, "partial": partial}


def _shared_reference(x_local, w13, w2, pair):
    # 先按实际 EP rank 收集，再按显式 TP pair 排序；不能假定 peer == rank ^ 1。
    inputs = [torch.empty_like(x_local) for _ in range(RANKS)]
    dist.all_gather(inputs, x_local)
    x = torch.cat([inputs[pair[0]], inputs[pair[1]]], dim=0)
    result = _flat_shared_reference(x, w13, w2)
    activation, partial = result["activation"], result["partial"]
    partials = [torch.empty_like(partial) for _ in range(RANKS)]
    dist.all_gather(partials, partial)
    # 两个 TP partial 各自已是 BF16；求和一次后再 BF16 舍入。
    pair_partial_abs = tuple(partials[rank].float().abs() for rank in pair)
    partial_tolerance = ((0.0625 + 0.02 * pair_partial_abs[0])
                         + (0.0625 + 0.02 * pair_partial_abs[1]))
    summed = partials[pair[0]].float() + partials[pair[1]].float()
    output = summed.to(torch.bfloat16)
    return {"input": x, "activation": activation, "partial": partial, "output": output,
            "pair_partial_abs": pair_partial_abs, "sum_partial_tolerance": partial_tolerance,
            "sum_rounding_error": (output.float() - summed).abs()}


def _stage(buffer, routed, x_fixed, x_variant, device, host_group):
    buffer.x[:TOKENS].copy_(routed[0])
    buffer.x_sf[:TOKENS].copy_(routed[1])
    buffer.topk_idx[:TOKENS].copy_(routed[2])
    buffer.topk_weights[:TOKENS].copy_(routed[3])
    x_fixed.copy_(x_variant)
    torch.cuda.synchronize(device)
    # 这里只同步下一次调用的输入准备，不清空新旧通信协议的任何 flag。
    dist.barrier(group=host_group)


def main():
    args = _arguments()
    flat_shared = args.flat_shared_tokens is not None
    shared_tokens = args.flat_shared_tokens if flat_shared else 2 * TOKENS
    shared_intermediate = args.shared_intermediate
    # exact-RS检查沿用两项BF16 partial求和再舍入的契约，使用软件NCCL规约。
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    if os.environ["NCCL_NVLS_ENABLE"] != "0":
        raise RuntimeError("Exact TP2 RS validation requires NCCL_NVLS_ENABLE=0")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "benchmarks"))
    global torch, dist
    import torch
    import torch.distributed as dist
    import gc_moe
    import deep_gemm
    from benchmark_gc_moe import (
        build_local_weights, build_tokens, cast_grouped_weights_to_fp4,
        per_token_cast_to_fp8,
    )
    from gc_moe import K3SharedWorkspace, prepare_k3_shared_gc
    from gc_moe.apis.k3_gc_moe import K3_SITU_CONTRACT
    from gc_moe.runtime.gc_moe import get_context

    if not Path(gc_moe.__file__).resolve().is_relative_to(root):
        raise RuntimeError("必须测试当前 checkout，不能导入环境中的旧 gc_moe")
    if int(os.environ.get("WORLD_SIZE", "0")) != RANKS:
        raise RuntimeError("需要 torchrun 的八个单节点 rank")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # 禁止参考 GEMM 在 K 归约中提前做 BF16 截断。
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=10))
    rank = dist.get_rank()
    if rank != local_rank or dist.get_world_size() != RANKS:
        raise RuntimeError("此验证只支持单节点 EP8")
    host_group = dist.new_group(backend="gloo", timeout=timedelta(minutes=10))
    pair = next(pair for pair in args.tp_pairs if rank in pair)
    owner = pair.index(rank)
    # 所有rank以相同顺序创建TP组；collective的rank顺序与显式owner顺序分别处理。
    tp_group = None
    for ranks in args.tp_pairs:
        group = dist.new_group(ranks=sorted(ranks), backend="nccl", timeout=timedelta(minutes=10))
        if rank in ranks:
            tp_group = group
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / f"rank{rank}.jsonl").open("w", encoding="utf-8") as log:
        try:
            context = get_context(num_ranks=RANKS)
            metadata = {"event": "config", "rank": rank,
                      "tokens_per_rank": TOKENS, "total_tokens": TOKENS * RANKS,
                      "shared_input_tokens": shared_tokens,
                      "shared_workspace_capacity": args.shared_capacity or shared_tokens,
                      "shared_mode": "flat" if flat_shared else "external_tp2",
                      "tp_pair": None if flat_shared else pair,
                      "shared_owner": None if flat_shared else owner,
                      "shared_output_shape": [shared_tokens if flat_shared else TOKENS, SHARED_HIDDEN],
                      "shared_collective": ("none" if flat_shared else
                          "external_all_gather_then_gc_then_external_tp2_reduce_scatter"),
                      "nccl_nvls_enable": os.environ["NCCL_NVLS_ENABLE"],
                      "benchmark_includes_all_gather": False,
                      "benchmark_includes_input_stage": False,
                      "benchmark_includes_reduce_scatter": not flat_shared,
                      "tp_pairs": args.tp_pairs, "ep_size": RANKS,
                      "dispatch_sms": context.dispatch_sms, "compute_sms": context.compute_sms,
                      "mode": args.mode, "steps": args.steps,
                      "routed_shape": [ROUTED_HIDDEN, INTERMEDIATE, EXPERTS, TOPK],
                      "shared_shape": [SHARED_HIDDEN, shared_intermediate],
                      "routed_precision": "FP8 activation / FP4 weight / BF16 output",
                      "shared_precision": "BF16 activation / weight / output",
                      "activation_contract": K3_SITU_CONTRACT,
                      "routed_reference": "original DeepGEMM MegaMoE SiTU",
                      "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                      "gpu_name": torch.cuda.get_device_name(device),
                      "gc_moe_env": {key: value for key, value in sorted(os.environ.items())
                                     if key.startswith("GC_MOE_")},
                      "benchmark_settings": {"enabled": args.benchmark,
                                             "warmup": args.warmup,
                                             "iters": args.iters, "inner": args.inner},
                      "identity": _source_identity(root),
                      "validation_only": not args.benchmark, "passed": True}
            _exchange(host_group, log, metadata)

            # 与既有 benchmark 相同的权重生成、FP4 量化和 GC payload/SF 重排。
            raw_w1, raw_w2 = build_local_weights(
                EXPERTS // RANKS, INTERMEDIATE, ROUTED_HIDDEN, rank, device)
            quant_w1 = cast_grouped_weights_to_fp4(raw_w1)
            quant_w2 = cast_grouped_weights_to_fp4(raw_w2)
            weights = gc_moe.transform_weights_for_gc_moe(quant_w1, quant_w2)
            # DeepGEMM's transform/allocation API calls the gated layout swiglu;
            # its execution API below explicitly selects the original SiTU math.
            reference_weights = deep_gemm.transform_weights_for_mega_moe(
                quant_w1, quant_w2, activation="swiglu")
            del raw_w1, raw_w2, quant_w1, quant_w2
            base_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
                dist.group.WORLD, EXPERTS, CAPACITY, TOPK,
                ROUTED_HIDDEN, INTERMEDIATE, mma_type="fp8xfp4", activation="swiglu")
            candidate_buffer = gc_moe.GCSymmBuffer(
                dist.group.WORLD, EXPERTS, CAPACITY, TOPK,
                ROUTED_HIDDEN, INTERMEDIATE)
            y_baseline = torch.empty((TOKENS, ROUTED_HIDDEN), dtype=torch.bfloat16, device=device)
            y_candidate = torch.empty_like(y_baseline)
            x_fixed = torch.empty(
                (shared_tokens if flat_shared else TOKENS, SHARED_HIDDEN),
                dtype=torch.bfloat16, device=device)
            gathered_fixed = (x_fixed if flat_shared else torch.empty(
                (shared_tokens, SHARED_HIDDEN), dtype=torch.bfloat16, device=device))
            # 外置AG按group rank收集；GC只接收排列完毕的连续矩阵。
            gathered_parts = None if flat_shared else [
                gathered_fixed[pair.index(r) * TOKENS:(pair.index(r) + 1) * TOKENS]
                for r in sorted(pair)]

            def gather_shared_input():
                if not flat_shared:
                    dist.all_gather(gathered_parts, x_fixed, group=tp_group)
                    # 此维护测试在capture外准备AG输入，不把AG算入GC+RS benchmark。
                    torch.cuda.synchronize(device)

            generator = torch.Generator(device=device).manual_seed(20261107 + rank * 101)
            w13 = torch.randn((2 * shared_intermediate, SHARED_HIDDEN), dtype=torch.bfloat16,
                              device=device, generator=generator) * 0.03
            w2 = torch.randn((SHARED_HIDDEN, shared_intermediate), dtype=torch.bfloat16,
                             device=device, generator=generator) * 0.03

            variants = []
            references = []
            for variant in range(2):
                raw_x, indices, weights_topk = build_tokens(
                    TOKENS, ROUTED_HIDDEN, EXPERTS, TOPK, rank + variant * RANKS, device)
                fp8, scales = per_token_cast_to_fp8(
                    raw_x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
                # B 使用更大输入，覆盖 gate/up softcap；只测小值可能漏掉省略 tanh 的错误。
                x_shared = torch.randn(x_fixed.shape, dtype=torch.bfloat16,
                                       device=device, generator=generator) * (0.5 if variant == 0 else 2.0)
                x_shared += (rank + 1) * (0.03125 if variant == 0 else -0.03125)
                routed = (fp8, scales, indices, weights_topk)
                variants.append((routed, x_shared))
                shared_reference = (_flat_shared_reference(x_shared, w13, w2) if flat_shared
                                    else _shared_reference(x_shared, w13, w2, pair))
                _stage(base_buffer, routed, x_fixed, x_shared, device, host_group)
                deep_gemm.fp8_fp4_mega_moe(
                    y_baseline, *reference_weights, base_buffer,
                    recipe=(1, 1, 32), activation="situ", fast_math=True)
                torch.cuda.synchronize(device)
                shared_reference["routed"] = y_baseline.clone()
                references.append(shared_reference)

            workspace = K3SharedWorkspace(
                candidate_buffer, tokens=args.shared_capacity or shared_tokens,
                intermediate=shared_intermediate,
                compute_sms=context.compute_sms)
            # 工作区可大于本次输入；尾部sentinel检查FC1/FC2是否错误写过真实行界。
            workspace.activations[shared_tokens:].fill_(7)
            workspace.partial[shared_tokens:].fill_(7)
            prepared = prepare_k3_shared_gc(y_candidate, *weights, candidate_buffer,
                                           gathered_fixed, w13, w2, workspace, context=context)
            shared_partial = prepared.shared_partial
            shared_output = (shared_partial if flat_shared else torch.empty(
                (TOKENS, SHARED_HIDDEN), dtype=torch.bfloat16, device=device))
            # NCCL按sorted(pair)分发；显式pair可以反序，需将两个owner块映射到group rank。
            rs_input = (shared_partial if flat_shared or pair == tuple(sorted(pair))
                        else torch.empty_like(shared_partial))

            def launch():
                prepared.launch()
                if flat_shared:
                    return
                if pair != tuple(sorted(pair)):
                    for group_owner, ep_rank in enumerate(sorted(pair)):
                        canonical_owner = pair.index(ep_rank)
                        rs_input[group_owner * TOKENS:(group_owner + 1) * TOKENS].copy_(
                            shared_partial[canonical_owner * TOKENS:(canonical_owner + 1) * TOKENS])
                dist.reduce_scatter_tensor(shared_output, rs_input, group=tp_group)

            fixed_addresses = (x_fixed.data_ptr(), gathered_fixed.data_ptr(), workspace.buffer.data_ptr(),
                               workspace.partial.data_ptr(), shared_partial.data_ptr(),
                               rs_input.data_ptr(), shared_output.data_ptr(), y_candidate.data_ptr())

            def check(variant, phase, step):
                reference = references[variant]
                if flat_shared:
                    checks = {
                        "workspace_tail": {"passed": bool(
                            (workspace.activations[shared_tokens:] == 7).all().item()
                            and (workspace.partial[shared_tokens:] == 7).all().item())},
                        "shared_activation": _metrics(
                            workspace.activations[:shared_tokens], reference["activation"],
                            atol=0.03125, rtol=0.015, relative_l2_limit=0.0075),
                        "shared_partial": _metrics(shared_partial, reference["partial"],
                            atol=0.0625, rtol=0.02, relative_l2_limit=0.01),
                        "routed": _metrics(y_candidate, reference["routed"],
                            atol=0.125, rtol=0.005, relative_l2_limit=0.005),
                        "flat_output_contract": {"passed":
                            shared_partial.shape == (shared_tokens, SHARED_HIDDEN)
                            and shared_partial.is_contiguous()
                            and shared_partial.data_ptr() == workspace.partial.data_ptr()
                            == prepared.shared_partial.data_ptr()},
                    }
                    current_addresses = (
                        x_fixed.data_ptr(), gathered_fixed.data_ptr(), workspace.buffer.data_ptr(),
                        workspace.partial.data_ptr(), shared_partial.data_ptr(),
                        rs_input.data_ptr(), shared_output.data_ptr(), y_candidate.data_ptr())
                    unchanged = current_addresses == fixed_addresses
                    _exchange(host_group, log, {"event": "check", "rank": rank,
                        "phase": phase, "step": step, "variant": variant, "checks": checks,
                        "fixed_addresses": unchanged,
                        "passed": unchanged and all(c["passed"] for c in checks.values())})
                    return
                # 两端 partial 都按 pair[0]、pair[1] 的 token 顺序存放。
                # peer 必须取与当前 rank 相同的 canonical owner 行，不能取 peer 自己的 owner 行。
                actual_partials = [torch.empty_like(shared_partial) for _ in range(RANKS)]
                dist.all_gather(actual_partials, shared_partial)
                local_rows = slice(owner * TOKENS, (owner + 1) * TOKENS)
                other_rows = slice((1 - owner) * TOKENS, (2 - owner) * TOKENS)
                peer_rank = pair[1 - owner]
                local_partial = actual_partials[rank][local_rows]
                peer_partial = actual_partials[peer_rank][local_rows]
                actual_sum = local_partial.float() + peer_partial.float()
                exact_output = actual_sum.to(torch.bfloat16)
                # 随机输入按 rank 区分；显式确认测试能识别“误取 peer 自己的行”这种错位。
                wrong_owner_output = (local_partial.float()
                                      + actual_partials[peer_rank][other_rows].float()).to(torch.bfloat16)
                wrong_owner_mismatches = int((exact_output.view(torch.int16)
                    != wrong_owner_output.view(torch.int16)).sum().item())
                # 相反符号 partial 抵消时，按 sum 本身的相对误差会错误拒绝合格的 GEMM。
                # 传播双方原有 partial 容差，并计入实际/参考两次 BF16 sum 的舍入量。
                sum_tolerance = (reference["sum_partial_tolerance"][local_rows]
                                 + reference["sum_rounding_error"][local_rows]
                                 + (exact_output.float() - actual_sum).abs())
                output_contract = (
                    shared_partial.shape == (2 * TOKENS, SHARED_HIDDEN)
                    and shared_output.shape == (TOKENS, SHARED_HIDDEN)
                    and shared_partial.dtype == shared_output.dtype == torch.bfloat16
                    and shared_partial.is_contiguous() and shared_output.is_contiguous()
                    and shared_partial.data_ptr() == workspace.partial.data_ptr()
                    == prepared.shared_partial.data_ptr()
                    and shared_output.data_ptr() != workspace.partial.data_ptr()
                    and dist.get_rank(tp_group) == sorted(pair).index(rank))
                checks = {
                    "workspace_tail": {"passed": bool(
                        (workspace.activations[shared_tokens:] == 7).all().item()
                        and (workspace.partial[shared_tokens:] == 7).all().item())},
                    "external_gathered_input": _metrics(gathered_fixed,
                        reference["input"],
                        atol=0.0, rtol=0.0, relative_l2_limit=0.0),
                    "shared_activation": _metrics(workspace.activations[:shared_tokens], reference["activation"],
                        atol=0.03125, rtol=0.015, relative_l2_limit=0.0075),
                    "shared_partial": _metrics(shared_partial, reference["partial"],
                        atol=0.0625, rtol=0.02, relative_l2_limit=0.01),
                    "routed": _metrics(y_candidate, reference["routed"], atol=0.125,
                                       rtol=0.005, relative_l2_limit=0.005),
                    "owner_rs_contract": {"passed": output_contract and wrong_owner_mismatches > 0,
                        "local_owner": owner, "peer_rank": peer_rank,
                        "canonical_rows": [local_rows.start, local_rows.stop],
                        "wrong_peer_owner_bitwise_mismatches": wrong_owner_mismatches},
                    "shared_local_exact_rs": _exact_bf16_metrics(shared_output, exact_output),
                    "shared_local": _metrics(shared_output,
                        reference["output"][local_rows], atol=0.0, rtol=0.0, relative_l2_limit=0.01,
                        elementwise_tolerance=sum_tolerance),
                }
                # 本 owner 的 ideal 指标保留为诊断；正式门槛是 exact RS + partial 误差传播 + 原 relative L2。
                diagnostics = {
                    "shared_local_ideal": _metrics(shared_output,
                        reference["output"][local_rows],
                        atol=0.0625, rtol=0.02, relative_l2_limit=0.01),
                    "sum_tolerance_rule": "sum(0.0625 + 0.02 * abs(reference_partial)) "
                                          "+ abs(BF16(reference_sum) - reference_sum) "
                                          "+ abs(BF16(actual_sum) - actual_sum)",
                    "pair_partial_abs_max": [float(value.max().item())
                                             for value in reference["pair_partial_abs"]],
                }
                current_addresses = (x_fixed.data_ptr(), gathered_fixed.data_ptr(), workspace.buffer.data_ptr(),
                                     workspace.partial.data_ptr(), shared_partial.data_ptr(),
                                     rs_input.data_ptr(), shared_output.data_ptr(), y_candidate.data_ptr())
                unchanged = current_addresses == fixed_addresses
                _exchange(host_group, log, {"event": "check", "rank": rank,
                          "phase": phase, "step": step, "variant": variant,
                          "checks": checks, "diagnostics": diagnostics,
                          "fixed_addresses": unchanged,
                          "passed": unchanged and all(c["passed"] for c in checks.values())})

            # eager 两套输入检查同时预热编译与 child graph；不把 capture 当执行。
            for variant in range(2):
                routed, x_variant = variants[variant]
                _stage(candidate_buffer, routed, x_fixed, x_variant, device, host_group)
                gather_shared_input()
                y_candidate.fill_(float("nan"))
                shared_output.fill_(float("nan"))
                launch()
                torch.cuda.synchronize(device)
                check(variant, "eager", variant)

            graph = None
            if args.mode == "graph":
                graph = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                dist.barrier(group=host_group)
                with torch.cuda.graph(graph, stream=stream):
                    launch()
                torch.cuda.current_stream(device).wait_stream(stream)
                torch.cuda.synchronize(device)

            for step in range(args.steps):
                variant = step % 2
                routed, x_variant = variants[variant]
                _stage(candidate_buffer, routed, x_fixed, x_variant, device, host_group)
                gather_shared_input()
                y_candidate.fill_(float("nan"))
                shared_output.fill_(float("nan"))
                if graph is None:
                    launch()
                else:
                    graph.replay()
                torch.cuda.synchronize(device)
                check(variant, args.mode, step)
            if args.benchmark:
                report = _benchmark(launch, args, device, host_group, rank, log, metadata)
                # 密集回放后的结果也必须通过原容差；失败时不生成正式性能结果文件。
                check(variant, "benchmark_postcheck", args.warmup + args.iters * args.inner)
                if rank == 0:
                    report["passed"] = True
                    (args.output_dir / "benchmark.json").write_text(
                        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                        encoding="utf-8")
                    stats = report["cross_rank_max"]
                    print(f"BENCH GC+external RS (L2 cold; AG/stage excluded), rank-max: "
                          f"p50={stats['p50_us']:.3f} us, p90={stats['p90_us']:.3f} us",
                          flush=True)
            _exchange(host_group, log, {"event": "summary", "rank": rank, "passed": True,
                      "eager_checked": 2, "replays_checked": args.steps,
                      "benchmark": args.benchmark,
                      "scope": ("original MegaMoE SiTU reference; flat shared BF16/SiTU partial"
                                if flat_shared else
                                "original MegaMoE SiTU reference; external TP2 AG; shared BF16/SiTU; external TP2 RS")})
            if rank == 0:
                print(f"PASS K3 shared: eager=2, {args.mode}={args.steps}, ranks=8", flush=True)
        except Exception:
            _write(log, {"event": "fatal", "rank": rank, "passed": False,
                         "traceback": traceback.format_exc()})
            raise
    # Graph可能持有NCCL资源，须先释放再销毁TP进程组。
    del graph
    dist.destroy_process_group(tp_group)
    dist.destroy_process_group(host_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
