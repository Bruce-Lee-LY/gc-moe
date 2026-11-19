"""离线编译 K3 shared role2；不创建 CUDA context，不分配或执行 GPU tensor。

在带 CuTe DSL/ptxas 的 CPU-only 容器中运行：
  python scripts/compile_k3_shared_gc.py --output-dir runs/k3_shared_compile
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/k3_shared_compile"))
    parser.add_argument("--shared-rows", type=int, default=256,
                        help="完整shared输入矩阵的总行数，包含Graph padding")
    parser.add_argument("--shared-intermediate", type=int, default=3072,
                        help="本卡shared权重的中间维；正的64倍数")
    parser.add_argument("--compute-sms", type=int, default=136,
                        help="离线编译的 compute CTA 数；默认对应 B300 原生分区")
    parser.add_argument("--dispatch-sms", type=int, default=12,
                        help="离线编译的 dispatch CTA 数")
    args = parser.parse_args()
    if args.dispatch_sms < 2 or args.compute_sms < 32 or args.compute_sms % 2:
        parser.error("需要 dispatch >= 2、偶数 compute >= 32")
    if args.shared_rows < 1:
        parser.error("--shared-rows 必须为正")
    if args.shared_intermediate < 64 or args.shared_intermediate % 64:
        parser.error("--shared-intermediate 必须为正的64倍数")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_tensor
    import torch
    from gc_moe.apis.gc_moe import ValidatedCall
    from gc_moe.apis.k3_gc_moe import K3_SITU_CONTRACT
    from gc_moe.layout.k3_shared import K3SharedLayout
    from gc_moe.jit_kernels.impls.sm100_fp8_fp4_gc_moe_k3_shared_external_rs import (
        _assemble_k3_role2, make_k3_shared_role2,
    )
    from gc_moe.jit_kernels.heuristics.gc_moe import get_ring_tokens
    from gc_moe.layout.gc_moe import GCBufferLayout
    from gc_moe.runtime.jit import compiler_identity

    # 安全边界：脚本本身不申请 GPU；即使宿主有卡也不能调用 torch.empty(cuda)。
    if torch.cuda.is_initialized():
        raise RuntimeError("离线编译入口不应已有 CUDA context")
    target = "sm_103f"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir.resolve()
    config = ValidatedCall(128, 32, 3584, 3072, 112, 0, 32, math.inf, 1.0, True)
    ring_tokens, sf_ring_tokens = get_ring_tokens(8, 896, 256, 16, 3584, 3072, args.dispatch_sms + args.compute_sms)
    routed_layout = GCBufferLayout(8, 896, 256, 16, 3584, 3072,
                                   ring_tokens, sf_ring_tokens, 0)
    shared_layout = K3SharedLayout(
        compute_sms=args.compute_sms, tokens=args.shared_rows, intermediate=args.shared_intermediate)
    context = SimpleNamespace(dispatch_sms=args.dispatch_sms, compute_sms=args.compute_sms)
    sym_buffer = SimpleNamespace(
        layout=routed_layout, group=SimpleNamespace(size=lambda: 8),
        num_max_tokens_per_rank=256, num_topk=16, num_experts=896,
        num_shared_experts=0, hidden=3584, intermediate_hidden=3072)
    kernel = make_k3_shared_role2(
        config, sym_buffer, context, shared_layout, shared_rows=args.shared_rows)

    def fake(dtype, shape, stride, alignment=16):
        return make_fake_tensor(dtype, shape, stride, assumed_align=alignment)

    signal = fake(cutlass.Int32, (routed_layout.nbytes // 4,), (1,))
    peers = fake(cutlass.Int64, (74,), (1,))
    maps = tuple(fake(cutlass.Uint8, (128,), (1,), 128) for _ in range(18))
    shared_state = fake(cutlass.Uint8, (shared_layout.nbytes,), (1,))
    shared_maps = fake(cutlass.Uint8, (4, 128), (128, 1), 128)
    compile_args = (
        signal, peers, cutlass.Uint32(0), cutlass.Uint32(128),
        maps[0], None, None, cutlass.Float32(math.inf), cutlass.Float32(1.0), *maps[1:],
        shared_state, shared_maps,
    )
    identity = compiler_identity(target, type(kernel).__call__, root / "gc_moe",
                                 f"gc_moe_k3_shared_flat_m128_s2_n{args.shared_rows}_i{args.shared_intermediate}_role2_v1")
    source_path = root / "gc_moe/impls/sm100_fp8_fp4_gc_moe_k3_shared.py"
    manifest = {
        "passed": False, "stage": "frontend_compile_started",
        "validation": "CuTe frontend + launch ABI + ptxas + static resource audit only",
        "gpu_executed": False, "target": target,
        "activation_contract": K3_SITU_CONTRACT,
        "command": [sys.executable, *sys.argv],
        "source_path": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "threads": 640, "grid": args.compute_sms, "cluster": 2,
        "dispatch_sms": args.dispatch_sms,
        "dynamic_smem": kernel.dynamic_smem,
        "routed_stages": kernel.math_stages,
        "shared_workspace_bytes": shared_layout.nbytes,
        "shared_rows": args.shared_rows,
        "shared_intermediate": args.shared_intermediate,
        "shared_input_contract": "contiguous input [n,7168], output preserves input row order",
        "register_budget": kernel.cta_register_budget,
        "register_targets": {"routed_control": 64, "routed_epilogue": 160,
                             "shared_epilogue": 48, "shared_producer_and_epoch": 48},
    }
    manifest_path = output / "manifest.json"
    # 新一次编译先写未通过状态，不能在失败后留下旧的 passed=true。
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "compiler-identity.json").write_text(
        json.dumps(dataclasses.asdict(identity), indent=2, default=str) + "\n")
    print("Compiling K3 shared role2 from fake CUDA tensor descriptors; GPU execution disabled.",
          flush=True)
    compiled = cute.compile(kernel, *compile_args,
                            options=identity.compile_options +
                            f" --keep-ptx --dump-dir {output}")
    raw_ptx = compiled.artifacts.PTX
    (output / "role2-frontend.ptx").write_text(raw_ptx.rstrip("\x00"))
    patched_ptx, cubin, assembler_version = _assemble_k3_role2(
        raw_ptx, str(output), target=target, ptxas_options=identity.ptxas_options)
    (output / "role2-launch.ptx").write_text(patched_ptx)
    cubin_path = output / "role2-launch.cubin"
    cubin_path.write_bytes(cubin)
    manifest.update(stage="resource_audit", cubin_bytes=len(cubin),
                    assembler_version=assembler_version,
                    cuda_context_initialized=torch.cuda.is_initialized())
    # 复用已生成 PTX 追加一次诊断汇编，不重新跑 CuTe frontend，不启动 GPU。
    diagnostic_command = [identity.tool_realpath, f"-arch={target}",
                          *identity.ptxas_options, "-v", "--warn-on-spills",
                          "-o", str(output / "role2-diagnostic.cubin"),
                          str(output / "role2-launch.ptx")]
    diagnostic = subprocess.run(diagnostic_command, text=True, capture_output=True, check=False)
    (output / "ptxas-diagnostics.txt").write_text(diagnostic.stdout + diagnostic.stderr)
    manifest["diagnostic_command"] = diagnostic_command
    manifest["diagnostic_exit_code"] = diagnostic.returncode
    # cuobjdump 仅读取已生成的 cubin，可在无 GPU 的容器中检查资源。
    cuobjdump = shutil.which("cuobjdump")
    if not cuobjdump and os.environ.get("CUDA_HOME"):
        candidate = Path(os.environ["CUDA_HOME"]) / "bin/cuobjdump"
        if candidate.is_file():
            cuobjdump = str(candidate)
    failures = []
    if not cuobjdump:
        failures.append("cuobjdump is required to verify the initial CTA register pool")
    else:
        resource_command = [cuobjdump, "--dump-resource-usage", str(cubin_path)]
        result = subprocess.run(resource_command, text=True, capture_output=True, check=False)
        resource_text = result.stdout + result.stderr
        (output / "resource-usage.txt").write_text(resource_text)
        manifest["resource_dump_command"] = resource_command
        manifest["resource_dump_exit_code"] = result.returncode
        resources = re.findall(r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)", resource_text)
        if result.returncode or len(resources) != 1:
            failures.append("expected exactly one resource record for the role2 launch cubin")
        else:
            registers, stack, static_smem = map(int, resources[0])
            budget = kernel.cta_register_budget
            minimum_registers = math.ceil(budget / 640)
            manifest.update(initial_registers_per_thread=registers,
                            initial_cta_register_pool=registers * 640,
                            required_initial_registers_per_thread=minimum_registers,
                            stack_bytes_per_thread=stack, static_smem=static_smem,
                            total_smem=static_smem + kernel.dynamic_smem,
                            smem_limit=232448)
            # setmaxnreg 只能重分配本 CTA 启动时已获得的池，不能借 SM 剩余寄存器。
            if not minimum_registers <= registers <= 102 or registers % 8:
                failures.append(f"initial register count {registers} does not fund budget {budget} "
                                "within the 640-thread launch limit and 8-register allocation granularity")
            if static_smem + kernel.dynamic_smem > 232448:
                failures.append("static plus dynamic SMEM exceeds the 227 KiB per-CTA limit")
    observed = [(action, int(count)) for action, count in re.findall(
        r"setmaxnreg\.(inc|dec)\.sync\.aligned\.u32\s+(\d+)", patched_ptx)]
    expected = [("dec", 64), ("inc", 160), ("dec", 48), ("dec", 48)]
    manifest["ptx_register_reallocation"] = observed
    if observed != expected:
        failures.append(f"unexpected register reallocation sequence: {observed}")
    if kernel.cta_register_budget != 64 * 128 + 160 * 256 + 48 * 256:
        failures.append("declared register budget disagrees with warpgroup targets")
    if diagnostic.returncode:
        failures.append("diagnostic ptxas assembly failed")
    if manifest["cuda_context_initialized"]:
        failures.append("offline compilation unexpectedly initialized a CUDA context")
    manifest.update(passed=not failures, stage="complete" if not failures else "resource_audit_failed",
                    failures=failures)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    if failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    main()
