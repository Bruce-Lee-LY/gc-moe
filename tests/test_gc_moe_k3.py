# SPDX-License-Identifier: Apache-2.0
# Portions of the TGV kernel are copyright (c) 2026 NVIDIA CORPORATION,
# FlashInfer team and SGLang Team.
"""Kimi K3 full-output accuracy against the selected Mega MoE K3 baseline.

The default `--baseline sglang` was tested with PyTorch 2.13 and SGLang DeepGEMM.
The repository default requirements are separate from the baseline environments.
For each run the DeepGEMM build must match the PyTorch ABI used by both
operators. Install SGLang dependencies separately from standard DeepGEMM,
which provides the same ``deep_gemm`` import:

    python -m pip install --target /path/to/k3-deps sgl-deep-gemm==0.2.8 flashinfer-python==0.7.4
    export PYTHONPATH="/path/to/k3-deps${PYTHONPATH:+:$PYTHONPATH}"

If the wheel's PyTorch ABI differs, compile sgl-project/DeepGEMM tag v0.2.8
with the active PyTorch using ``build_sgl_deep_gemm.sh`` and install the resulting
wheel in that directory. FlashInfer supplies GEMM and RMSNorm for the SGLang baseline.

For `--baseline vllm`, use vllm-project/DeepGEMM at
44cba7d28066182852005c59052f035ebc18e211. Check out that revision and its
submodules. Build the wheel with PyTorch 2.14 (tested with CUDA 13.0),
a C++20 compiler with std::format and elfutils headers. Use
`DG_FORCE_BUILD=1 python -m pip wheel --no-build-isolation --no-deps .`,
then install the wheel in a separate `--target` directory. If built from a
source archive, write the full revision to `vllm_deep_gemm_revision` beside
the installed `deep_gemm` package. Put the selected dependency directory
first on PYTHONPATH. Both distributions use the same module name, so run
the two baselines in separate processes.

For vLLM's small-batch Gate fast path, set VLLM_K3_CUTEDSL_DIR to the
vllm/model_executor/kernels/linear/cute_dsl directory from vLLM 89d9b4d187.
Only its two standalone BF16 Gate kernels are loaded. Without that directory
the Gate uses cuBLAS at all sizes, which is recorded in the test evidence.
The baseline reproduces the stock BF16 schedule with Triton Top16, SiTU and
RMSNorm replacements. RMSNorm follows vLLM's default compiled IR semantics.

"""

from __future__ import annotations
import argparse
import gc
import time
import weakref
from datetime import timedelta
import json
from math import isfinite
import os
from pathlib import Path
import sys
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import gc_moe
from gc_moe.layout.gc_moe_k3 import ceil_div

assert Path(gc_moe.__file__).resolve().is_relative_to(ROOT), gc_moe.__file__
import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp4
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, distribution
from importlib.util import module_from_spec, spec_from_file_location
from typing import List, NamedTuple, Optional, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import triton
import triton.language as tl
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor, make_fake_stream


# SGLang Mega MoE K3 baseline (da395ae091).
# A merged BF16 front and Triton pre-dispatch prepare FP8 inputs for Mega MoE.
# Shared BF16 GEMMs and SiTU run on a side
# stream, which joins after the routed RMSNorm and up projection. The Triton
# Add3 and SiTU and the CuTeDSL TGV implementation below are benchmark code.

def _check_deep_gemm():
    try:
        package = distribution("sgl-deep-gemm")
    except PackageNotFoundError as exc:
        raise RuntimeError("K3 tests require sgl-deep-gemm==0.2.8") from exc
    imported = Path(deep_gemm.__file__).resolve()
    installed = Path(package.locate_file("deep_gemm/__init__.py")).resolve()
    if package.version != "0.2.8" or imported != installed:
        raise RuntimeError(
            f"K3 tests require sgl-deep-gemm==0.2.8 on PYTHONPATH; imported {imported}"
        )


@contextmanager
def _configure_sglang_mega_moe_k3(limit, *, pdl=None):
    previous = deep_gemm.get_num_sms()
    target = min(previous, limit)
    target -= target % 2
    previous_pdl = deep_gemm.get_pdl() if pdl is not None else None
    change_pdl = pdl is not None and pdl != previous_pdl
    if target != previous:
        deep_gemm.set_num_sms(target)
    try:
        if change_pdl:
            deep_gemm.set_pdl(pdl)
        yield
    finally:
        if change_pdl:
            deep_gemm.set_pdl(previous_pdl)
        if target != previous:
            deep_gemm.set_num_sms(previous)


def _use_tgv(m, n, k):
    # SGLang's K3 TP8 shared projections and the generic B300 TGV policy.
    if m <= 8 and (n, k) in ((1536, 7168), (7168, 768), (7168, 3584)):
        return True
    if k % 8 or n < 1024 or not 2048 <= k <= 6144:
        return False
    ragged = m % 16 != 0
    if n <= 1024:
        return m <= 512 and (m >= 48 or m % 16 >= 9)
    if n <= 2624:
        return m <= 128 and k >= 4096
    if n <= 4096:
        return m <= 64
    if k < 4096:
        return k >= 3072 and 25 <= m <= 48 and ragged
    if n <= 6144:
        if m <= 64:
            return m <= 32 or k >= 6144 or ragged
        return m <= 72 and k >= 6144 and ragged
    if n <= 8192:
        return m <= 48 or (m <= 63 and ragged)
    return n <= 12288 and k >= 6144 and (m <= 32 or (m <= 48 and ragged))


class _SharedMLP:
    def __init__(self, a, weights):
        from flashinfer import autotune, mm_bf16

        self.a = a
        self.w1, self.w2 = weights
        self.clamp = (
            float(torch.tensor(a.activation_clamp, dtype=torch.bfloat16))
            if a.activation_clamp is not None
            else None
        )
        self.side = torch.cuda.Stream(device=a.device)
        self.autotune = autotune
        self.mm_bf16 = mm_bf16
        self.tuned_shapes = set()

    @staticmethod
    def _low_m(x, weight):
        return 0 < x.shape[0] <= 16 and weight.shape[1] % 128 == 0

    def _gemm(self, x, weight, *, graph_path, out=None):
        if out is None:
            out = x.new_empty((x.shape[0], weight.shape[0]))
        if graph_path and self._low_m(x, weight):
            return self.mm_bf16(x, weight.T, None, pdl=True, out=out, backend="cute-dsl")
        if _use_tgv(x.shape[0], *weight.shape):
            return tgv_bf16_gemm(x, weight, out)
        return torch.mm(x, weight.T, out=out)

    def forward(self, x, out, graph_path):
        a = self.a
        gate_up = self._gemm(x, self.w1, graph_path=graph_path)
        mid = x.new_empty((x.shape[0], self.w2.shape[1]))
        situ_and_mul(
            gate_up,
            mid,
            beta=a.activation_beta,
            linear_beta=a.activation_linear_beta,
            clamp=self.clamp,
        )
        self._gemm(mid, self.w2, graph_path=graph_path, out=out)

    def prepare_graph(self, x, out):
        m = x.shape[0]
        if not 0 < m <= 16 or m in self.tuned_shapes:
            return
        # SGLang autotunes the eligible GEMMs before recording a CUDA graph.
        with self.autotune(True):
            self.forward(x, out, graph_path=True)
        self.tuned_shapes.add(m)


@triton.jit
def _sglang_k3_front_epilogue(
    Merged, Bias, Latent, Ids, Weights, E: tl.constexpr, H: tl.constexpr,
    TOPK: tl.constexpr, EB: tl.constexpr, HB: tl.constexpr,
):
    row = tl.program_id(0)
    eid = tl.arange(0, EB)
    tl.extra.cuda.gdc_wait()
    logits = tl.load(Merged + row * (E + H) + eid, eid < E, other=0)
    raw = tl.div_rn(1.0, 1.0 + tl.extra.cuda.libdevice.exp(-logits))
    score = raw + tl.load(Bias + eid, eid < E, other=0)
    bits = tl.where(score == 0.0, 0.0, tl.where(score == score, score, -1e30)).to(tl.uint32, bitcast=True)
    key = tl.where((bits & 0x80000000) != 0, ~bits, bits | 0x80000000)
    packed = (key.to(tl.uint64) << 32) | (0xffffffff - eid.to(tl.uint32)).to(tl.uint64)
    selected = tl.topk(tl.where(eid < E, packed, 0), TOPK)
    slots = tl.arange(0, TOPK)
    selected_ids = tl.sort((0xffffffff - selected.to(tl.uint32)).to(tl.int32), descending=False)
    winner = tl.load(Merged + row * (E + H) + selected_ids)
    weights = tl.div_rn(1.0, 1.0 + tl.extra.cuda.libdevice.exp(-winner))
    # Ordered FP32 summation matches the EP front's ascending expert-ID order.
    total = tl.full((), 0.0, tl.float32)
    for i in tl.static_range(TOPK):
        total += tl.sum(tl.where(slots == i, weights, 0.0), 0)
    tl.store(Ids + row * TOPK + slots, selected_ids)
    tl.store(Weights + row * TOPK + slots, tl.div_rn(weights, tl.where(total > 0, total, 1.0)))
    ch = tl.arange(0, HB)
    latent = tl.load(Merged + row * (E + H) + E + ch, ch < H, other=0)
    tl.store(Latent + row * H + ch, latent.to(tl.bfloat16), ch < H)
    tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _sglang_k3_pre_dispatch(
    Latent, Ids, Weights, Valid, X, SF, I, W, H: tl.constexpr, K: tl.constexpr,
    HG: tl.constexpr, SF_STRIDE: tl.constexpr, MASK_ROWS: tl.constexpr,
):
    row = tl.program_id(0)
    groups = tl.arange(0, HG)
    lanes = tl.arange(0, 32)
    ch = groups[:, None] * 32 + lanes[None, :]
    slot = tl.arange(0, K)
    live = True
    if MASK_ROWS:
        live = row < tl.load(Valid)
    tl.extra.cuda.gdc_wait()
    val = tl.load(Latent + row * H + ch, (ch < H) & live, other=0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(val), 1), 1e-10)
    scale = amax * (1.0 / 448.0)
    exp = (scale.to(tl.uint32, bitcast=True) + 0x7fffff) >> 23
    inv = ((254 - exp) << 23).to(tl.float32, bitcast=True)
    payload = (val * inv[:, None]).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    tl.store(X + row * H + ch, payload, ch < H)
    tl.store(SF + row * SF_STRIDE + groups, exp.to(tl.uint8), groups < H // 32)
    tl.store(I + row * K + slot, tl.load(Ids + row * K + slot, live, other=-1))
    tl.store(W + row * K + slot, tl.load(Weights + row * K + slot, live, other=0))
    tl.extra.cuda.gdc_launch_dependents()


def _sglang_k3_front(states, front_weight, bias, hidden, topk):
    t, experts = states.shape[0], bias.numel()
    # Small EP decode uses merged_fp32 on B300. The cast/TopK epilogue is
    # local Triton so the baseline needs no source/JIT extension imports.
    if not (t <= 1024 and experts == 896 and topk == 16):
        logits = torch.mm(states, front_weight[:experts].T, out_dtype=torch.float32)
        latent = torch.mm(states, front_weight[experts:].T)
        ids, weights = _reference_routes(logits, bias, topk)
        return latent, ids, weights
    merged = torch.mm(states, front_weight.T, out_dtype=torch.float32)
    latent = states.new_empty((t, hidden))
    ids = torch.empty((t, topk), device=states.device, dtype=torch.int32)
    weights = torch.empty((t, topk), device=states.device, dtype=torch.float32)
    _sglang_k3_front_epilogue[(t,)](merged, bias, latent, ids, weights, experts, hidden,
        topk, triton.next_power_of_2(experts), triton.next_power_of_2(hidden),
        num_warps=8, launch_pdl=True, enable_fp_fusion=False)
    return latent, ids, weights


def _reference_routes(logits, bias, topk):
    raw = logits.sigmoid()
    ranked = (raw + bias).nan_to_num(nan=-1e30)
    ids = ranked.argsort(dim=-1, descending=True, stable=True)[:, :topk].sort(dim=-1).values
    weights = raw.gather(1, ids)
    total = weights.sum(dim=-1, keepdim=True)
    return ids, weights / torch.where(total > 0, total, 1.)


def _reference_quantization(latent):
    grouped = latent.float().reshape(latent.shape[0], latent.shape[1] // 32, 32)
    scale = (grouped.abs().amax(-1).clamp_min(1e-10) * (1.0 / 448.0)).contiguous()
    exp = (scale.view(torch.int32) + 0x7fffff) >> 23
    inv = ((254 - exp) << 23).view(torch.float32)
    payload = (grouped * inv.unsqueeze(-1)).reshape_as(latent).to(torch.float8_e4m3fn)
    return payload, exp.to(torch.uint8).contiguous().view(torch.int32)


class SGLangMegaMoEK3:
    def __init__(self, a, pairs, shared_weights, front_weight, router_bias):
        _check_deep_gemm()
        if (
            a.activation_beta != 4.0
            or a.activation_linear_beta != 25.0
            or a.activation_clamp is not None
        ):
            raise ValueError("SGLang Mega MoE K3 SiTU requires beta=4, linear_beta=25 and no clamp")
        self.a = a
        n_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
        reserved = max(0, int(os.environ.get("SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS", "2")))
        self.max_sms = max(2, n_sms - reserved)
        pdl_env = os.environ.get("SGLANG_DEEPGEMM_PDL", "true").strip().lower()
        self.pdl = pdl_env in ("true", "1", "yes", "on")
        self.weights = deep_gemm.transform_weights_for_mega_moe(
            *pairs, mma_type="fp8xfp4", activation=a.activation
        )
        with _configure_sglang_mega_moe_k3(self.max_sms):
            self.buffer = deep_gemm.get_symm_buffer_for_mega_moe(
                dist.group.WORLD,
                a.num_experts,
                a.num_max_tokens_per_rank,
                a.num_topk,
                a.hidden,
                a.intermediate_hidden,
                mma_type="fp8xfp4",
            )
        self.empty_output = torch.empty((1, a.hidden), device=a.device, dtype=torch.bfloat16)
        self.shared = _SharedMLP(a, shared_weights)
        self.front_weight, self.router_bias = front_weight, router_bias
        self.global_num_tokens = None

    def call(self, out, states, graph_path=False, *, projection, valid_token_count=None):
        a = self.a
        t = out[0].shape[0]
        current = torch.cuda.current_stream(a.device)
        capturing = torch.cuda.is_current_stream_capturing()
        graph_path = graph_path or capturing
        b = self.buffer
        if t:
            latent, ids, weights = _sglang_k3_front(states, self.front_weight, self.router_bias,
                                                     a.hidden, a.num_topk)
            self.shared.side.wait_stream(current)
            with torch.cuda.stream(self.shared.side):
                self.shared.forward(states, out[1][:t], graph_path)
            _sglang_k3_pre_dispatch[(t,)](
                latent, ids, weights, valid_token_count, b.x.view(torch.uint8),
                b.x_sf.view(torch.uint8), b.topk_idx, b.topk_weights, a.hidden, a.num_topk,
                triton.next_power_of_2(a.hidden // 32), b.x_sf.stride(0) * b.x_sf.element_size(),
                valid_token_count is not None, num_warps=8, launch_pdl=True)
        else:
            b.topk_idx[0].fill_(-1)
            b.topk_weights[0].zero_()

        with _configure_sglang_mega_moe_k3(self.max_sms, pdl=self.pdl):
            deep_gemm.fp8_fp4_mega_moe(
                out[0] if t else self.empty_output,
                *self.weights,
                self.buffer,
                recipe=(1, 1, 32),
                activation=a.activation,
                activation_clamp=a.activation_clamp,
                fast_math=bool(a.fast_math),
                global_num_tokens=None if graph_path else self.global_num_tokens,
            )
        if t:
            norm, up_weight, prefix, eps = projection
            if a.routed_scaling_factor != 1.0:
                out[0].mul_(a.routed_scaling_factor)
            up = self.project_routed(out[0], norm, up_weight, eps, graph_path)
            current.wait_stream(self.shared.side)
            add_k3_outputs(up, out[1][:t], prefix, out[2])

    def project_routed(self, latent, norm_weight, up_weight, eps, graph_path):
        from flashinfer import rmsnorm

        normalized = rmsnorm(latent, norm_weight, eps=eps, enable_pdl=True)
        return self.shared._gemm(normalized, up_weight, graph_path=graph_path)


# vLLM Mega MoE K3 baseline (89d9b4d187, unquantized BF16 projections).
# The routed kernel comes from the DeepGEMM build pinned by that checkout.
# Triton supplies the standalone routing, quantization and normalization.
VLLM_DEEPGEMM_REVISION = "44cba7d28066182852005c59052f035ebc18e211"
BASELINE_NAMES = {"sglang": "SGLang Mega MoE K3", "vllm": "vLLM Mega MoE K3"}


def _check_vllm_deep_gemm():
    imported = Path(deep_gemm.__file__).resolve()
    try:
        package = distribution("deep-gemm")
    except PackageNotFoundError as exc:
        raise RuntimeError("Install the pinned vLLM DeepGEMM in a separate directory") from exc
    installed = Path(package.locate_file("deep_gemm/__init__.py")).resolve()
    revision = imported.parent.parent / "vllm_deep_gemm_revision"
    pinned = package.version.partition("+")[2] == VLLM_DEEPGEMM_REVISION[:7]
    if revision.is_file():
        pinned = revision.read_text().strip() == VLLM_DEEPGEMM_REVISION
    if imported != installed or not pinned:
        raise RuntimeError(
            f"vLLM Mega MoE K3 expects DeepGEMM {VLLM_DEEPGEMM_REVISION} "
            f"(wheel local version or vllm_deep_gemm_revision marker); imported {imported}"
        )


@triton.jit
def _vllm_k3_topk(Logits, Bias, Ids, Weights, E: tl.constexpr, K: tl.constexpr,
                  EB: tl.constexpr, KB: tl.constexpr, PDL: tl.constexpr,
                  ROUTED_SCALE: tl.constexpr):
    row = tl.program_id(0)
    if PDL:
        tl.extra.cuda.gdc_wait()
    expert = tl.arange(0, EB)
    logit = tl.load(Logits + row * E + expert, expert < E, other=0)
    raw = tl.div_rn(1.0, 1.0 + tl.extra.cuda.libdevice.exp(-logit))
    score = raw + tl.load(Bias + expert, expert < E, other=0)
    bits = tl.where(score == 0.0, 0.0, tl.where(score == score, score, -1e30)).to(tl.uint32, bitcast=True)
    key = tl.where((bits & 0x80000000) != 0, ~bits, bits | 0x80000000)
    ranked = (key.to(tl.uint64) << 32) | (0xffffffff - expert.to(tl.uint32)).to(tl.uint64)
    ids = (0xffffffff - tl.topk(tl.where(expert < E, ranked, 0), KB).to(tl.uint32)).to(tl.int32)
    slot = tl.arange(0, KB)
    winner = tl.load(Logits + row * E + ids, slot < K, other=0)
    weight = tl.div_rn(1.0, 1.0 + tl.extra.cuda.libdevice.exp(-winner))
    total = tl.sum(tl.where(slot < K, weight, 0.0), 0)
    weight = tl.div_rn(weight, tl.where(total > 0, total, 1.0)) * ROUTED_SCALE
    tl.store(Ids + row * K + slot, ids, slot < K)
    tl.store(Weights + row * K + slot, weight, slot < K)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _vllm_k3_pre_dispatch(
    Latent, Ids, Weights, X, SF, I, W,
    T, H: tl.constexpr, K: tl.constexpr, SF_M: tl.constexpr, SF_K: tl.constexpr,
    BM: tl.constexpr, KB: tl.constexpr,
):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    k_tile = tl.program_id(1)
    ch = k_tile * 128 + tl.arange(0, 128)
    values = tl.load(Latent + row[:, None] * H + ch[None, :], row[:, None] < T, 0).to(tl.float32)
    groups = tl.reshape(values, (BM, 4, 32))
    amax = tl.maximum(tl.max(tl.abs(groups), 2), 1e-4)
    scale = tl.div_rn(amax, 448.)
    bits = scale.to(tl.uint32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7fffff) != 0).to(tl.uint32)
    exponent = tl.minimum(tl.maximum(exponent, 1), 254)
    factor = (exponent << 23).to(tl.float32, bitcast=True)
    quant = tl.reshape(groups * tl.div_rn(1., factor)[:, :, None], (BM, 128))
    tl.store(X + row[:, None] * H + ch[None, :], quant.to(tl.float8e4nv), row[:, None] < T)
    packed = tl.sum(exponent << (8 * tl.arange(0, 4)[None, :]), 1)
    tl.store(SF + row * SF_M + k_tile * SF_K, packed, row < T)
    if k_tile == 0:
        slot = tl.arange(0, KB)
        offset = row[:, None] * K + slot[None, :]
        mask = (row[:, None] < T) & (slot[None, :] < K)
        tl.store(I + offset, tl.load(Ids + offset, mask, other=-1), mask)
        tl.store(W + offset, tl.load(Weights + offset, mask, other=0), mask)


@triton.jit
def _vllm_k3_rmsnorm(X, W, Y, H: tl.constexpr, B: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    value = tl.load(X + row * H + col, col < H, 0).to(tl.float32)
    norm = tl.rsqrt(tl.sum(value * value, 0) / H + EPS)
    # The compiled vLLM native IR casts to the weight dtype before multiplying.
    scaled = (value * norm).to(tl.bfloat16).to(tl.float32)
    weight = tl.load(W + col, col < H, 0).to(tl.float32)
    tl.store(Y + row * H + col, scaled * weight, col < H)


def _load_vllm_gate_kernel(directory, name):
    module_name = f"_k3_vllm{name}"
    path = directory / f"{name}.py"
    spec = spec_from_file_location(module_name, path)
    module = module_from_spec(spec)
    # CuTeDSL inspects source in the defining module at compile time.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class VLLMMegaMoEK3:
    def __init__(self, a, pairs, shared_weights, front_weight, router_bias):
        _check_vllm_deep_gemm()
        if a.activation != "situ" or a.activation_clamp is not None:
            raise ValueError("vLLM Mega MoE K3 requires SiTU without clamp")
        self.a = a
        self.gate = front_weight[:a.num_experts]
        source_dir = os.environ.get("VLLM_K3_CUTEDSL_DIR")
        self.gate_sources = None
        self.gate_compiled = {}
        if source_dir:
            directory = Path(source_dir).expanduser().resolve()
            dot = _load_vllm_gate_kernel(directory, "_ll_bf16_dotprod")
            split = _load_vllm_gate_kernel(directory, "_ll_bf16_splitk")
            self.gate_sources = dot.LLBf16Dotprod, split.LLBf16SplitK
        self.down = front_weight[a.num_experts:]
        self.router_bias = router_bias
        self.shared_w1, self.shared_w2 = shared_weights
        self.side = torch.cuda.Stream(device=a.device)
        self.weights = deep_gemm.transform_weights_for_mega_moe(*pairs, activation=a.activation)
        self.buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            dist.group.WORLD, a.num_experts, a.num_max_tokens_per_rank, a.num_topk,
            a.hidden, a.intermediate_hidden, mma_type="fp8xfp4", activation=a.activation,
        )
        if hasattr(deep_gemm, "destroy_localizer"):
            deep_gemm.destroy_localizer()
        torch.cuda.synchronize(a.device)
        dist.barrier()

    def _gate_logits(self, states):
        m, k = states.shape
        n = self.gate.shape[0]
        if self.gate_sources is None or m > 16 or states.dtype != torch.bfloat16:
            return torch.mm(states, self.gate.T, out_dtype=torch.float32)

        logits = torch.empty((m, n), device=states.device, dtype=torch.float32)
        stream = cuda.CUstream(torch.cuda.current_stream(states.device).cuda_stream)
        dotprod = m <= 4 or k < 2048
        args = (m, k, n, stream) if dotprod else (stream, 1.0)
        if m not in self.gate_compiled:
            dot, split = self.gate_sources
            # Kimi K3's (K, N) = (7168, 896) uses upstream's default tactics.
            tensors = tuple(
                make_fake_compact_tensor(dtype, shape, stride_order=(1, 0), assumed_align=16)
                for dtype, shape in (
                    (cutlass.BFloat16, (m, k)), (cutlass.BFloat16, (n, k)),
                    (cutlass.Float32, (m, n)),
                )
            )
            if dotprod:
                kernel = dot(k=k, bs=128, use_pdl=True, prefetch_pdl_weights=m == 1)
                options = "--enable-tvm-ffi --ptxas-options -maxrregcount=64"
            else:
                kernel = split(split_k=6, num_stages=4, use_pdl=True)
                options = "--enable-tvm-ffi"
            self.gate_compiled[m] = cute.compile(kernel, *tensors, *args, options=options)
        self.gate_compiled[m](states, self.gate, logits, *(args[2:] if dotprod else args))
        return logits

    def call(self, out, states, graph_path=False, *, projection):
        a = self.a
        t = states.shape[0]
        b = self.buffer
        if t:
            current = torch.cuda.current_stream(a.device)
            ids = torch.empty((t, a.num_topk), device=a.device, dtype=torch.int32)
            weights = torch.empty((t, a.num_topk), device=a.device, dtype=torch.float32)
            # Fork before launching either producer. This mirrors the stock
            # router/latent overlap, with an explicit join before staging.
            if t <= 256:
                self.side.wait_stream(current)
            logits = self._gate_logits(states)
            pdl = t <= 16
            _vllm_k3_topk[(t,)](logits, self.router_bias, ids, weights, a.num_experts,
                a.num_topk, triton.next_power_of_2(a.num_experts),
                triton.next_power_of_2(a.num_topk), pdl, a.routed_scaling_factor,
                num_warps=8, launch_pdl=pdl, enable_fp_fusion=False)
            if t <= 256:
                with torch.cuda.stream(self.side):
                    latent = torch.mm(states, self.down.T)
                current.wait_stream(self.side)
                latent.record_stream(current)
            else:
                latent = torch.mm(states, self.down.T)
            block_m = 32 if t >= 1024 else 8 if t >= 64 else 1
            _vllm_k3_pre_dispatch[(triton.cdiv(t, block_m), a.hidden // 128)](
                latent, ids, weights, b.x, b.x_sf, b.topk_idx,
                b.topk_weights, t, a.hidden, a.num_topk, *b.x_sf.stride(),
                block_m, triton.next_power_of_2(a.num_topk),
                num_warps=4, enable_fp_fusion=False)

        deep_gemm.fp8_fp4_mega_moe(
            out[0], *self.weights, b, activation=a.activation,
            activation_alpha=a.activation_beta, activation_beta=a.activation_linear_beta,
            fast_math=bool(a.fast_math),
        )
        if t:
            gate_up = torch.mm(states, self.shared_w1.T)
            mid = states.new_empty((t, self.shared_w2.shape[1]))
            situ_and_mul(gate_up, mid, beta=a.activation_beta,
                         linear_beta=a.activation_linear_beta, clamp=None, pdl=False)
            torch.mm(mid, self.shared_w2.T, out=out[2])
            norm, up_weight, prefix, eps = projection
            normalized = torch.empty_like(out[0])
            _vllm_k3_rmsnorm[(t,)](out[0], norm, normalized, a.hidden,
                                   triton.next_power_of_2(a.hidden), eps, num_warps=4)
            # Reuse the shared output as the beta=1 input of the stock BF16
            # projection. Prefix is a separate step to match the GC output.
            out[2].addmm_(normalized, up_weight.T)
            if prefix is not None:
                out[2].add_(prefix)

def add_k3_outputs(up, shared, prefix, out):
    if prefix is None:
        torch.add(up, shared, out=out)
        return
    numel = out.numel()
    if numel == 0:
        return
    inputs = (up, shared, prefix, out)
    if numel % 16 or any(t.dtype != torch.bfloat16 or not t.is_contiguous()
                         or t.shape != out.shape or t.data_ptr() % 32 for t in inputs):
        torch.add(up + shared, prefix, out=out)
        return
    vectors = numel // 16
    threads = next((n for n in (128, 256) if vectors <= 256 * n), 512)
    _k3_add3[(triton.cdiv(vectors, threads),)](
        up, shared, prefix, out, vectors, threads, num_warps=threads // 32, launch_pdl=True)


@triton.jit
def _k3_add3(A, B, C, Y, vectors, threads: tl.constexpr):
    vid = tl.program_id(0) * threads + tl.arange(0, threads)
    offset = vid * 16
    # The MegaMoE tail takes the non-prefetching Add3 path in SGLang.
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    tl.inline_asm_elementwise(
        """{
        .reg .pred p;
        .reg .b32 a<8>, b<8>, c<8>;
        setp.ne.u32 p, $5, 0;
        @p ld.global.v8.b32 {a0,a1,a2,a3,a4,a5,a6,a7}, [$1];
        @p ld.global.v8.b32 {b0,b1,b2,b3,b4,b5,b6,b7}, [$2];
        @p ld.global.v8.b32 {c0,c1,c2,c3,c4,c5,c6,c7}, [$3];
        @p add.rn.bf16x2 a0, a0, b0;
        @p add.rn.bf16x2 a1, a1, b1;
        @p add.rn.bf16x2 a2, a2, b2;
        @p add.rn.bf16x2 a3, a3, b3;
        @p add.rn.bf16x2 a4, a4, b4;
        @p add.rn.bf16x2 a5, a5, b5;
        @p add.rn.bf16x2 a6, a6, b6;
        @p add.rn.bf16x2 a7, a7, b7;
        @p add.rn.bf16x2 a0, a0, c0;
        @p add.rn.bf16x2 a1, a1, c1;
        @p add.rn.bf16x2 a2, a2, c2;
        @p add.rn.bf16x2 a3, a3, c3;
        @p add.rn.bf16x2 a4, a4, c4;
        @p add.rn.bf16x2 a5, a5, c5;
        @p add.rn.bf16x2 a6, a6, c6;
        @p add.rn.bf16x2 a7, a7, c7;
        @p st.global.v8.b32 [$4], {a0,a1,a2,a3,a4,a5,a6,a7};
        mov.b32 $0, 0;
        }""",
        "=r,l,l,l,l,r",
        [A + offset, B + offset, C + offset, Y + offset, (vid < vectors).to(tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


# BF16 shared SiTU with the same PDL handoff as SGLang.
@triton.jit
def _tanh(x):
    return tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _situ_kernel(
    gate_up,
    out,
    elements,
    width: tl.constexpr,
    beta: tl.constexpr,
    linear_beta: tl.constexpr,
    clamp: tl.constexpr,
    block_size: tl.constexpr,
    PDL: tl.constexpr,
):
    pid = tl.program_id(0)
    if width % 16 == 0:
        # Put a 32-byte group on the contiguous axis for vector loads/stores.
        vectors: tl.constexpr = block_size // 16
        row_vectors: tl.constexpr = width // 16
        vec = pid * vectors + tl.arange(0, vectors)
        lane = tl.arange(0, 16)
        idx = vec[:, None] * 16 + lane[None, :]
        in_idx = (vec + vec // row_vectors * row_vectors)[:, None] * 16 + lane[None, :]
        valid = vec[:, None] < elements // 16
    else:
        idx = pid * block_size + tl.arange(0, block_size)
        in_idx = idx + idx // width * width
        valid = idx < elements
    if PDL:
        tl.extra.cuda.gdc_wait()
    gate = tl.load(gate_up + in_idx, valid, other=0)
    up = tl.load(gate_up + in_idx + width, valid, other=0)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()

    g = gate.to(tl.float32)
    u = up.to(tl.float32)
    if clamp is not None:
        g = tl.minimum(g, clamp)
        u = tl.clamp(u, -clamp, clamp)
    sigmoid = 1.0 / (1.0 + tl.exp(-g))
    g = beta * _tanh(g * (1.0 / beta)) * sigmoid
    if linear_beta > 0:
        u = linear_beta * _tanh(u * (1.0 / linear_beta))
    tl.store(out + idx, g * u, valid)


def situ_and_mul(gate_up, out, *, beta, linear_beta, clamp, pdl=True):
    elements = out.numel()
    if elements == 0:
        return
    block_size = 4096
    _situ_kernel[(triton.cdiv(elements, block_size),)](
        gate_up,
        out,
        elements,
        out.shape[1],
        beta,
        linear_beta or 0.0,
        clamp,
        block_size,
        pdl,
        num_warps=8,
        enable_fp_fusion=False,
        launch_pdl=pdl,
    )


# SM10x TGV kernel from SGLang da395ae091.
# Tuple format: (cta_m, cta_n, num_ab_stage, use_2cta); cta_k is fixed at
# 128. Sorted within each use_2cta block by (cta_m, cta_n, num_ab_stage).
# Tactic 1 (1-CTA 64×8 stages=8) is the default.
_TGV_CUTE_EXT_CTA_K: int = 128
_TGV_CUTE_EXT_DEFAULT_TACTIC: int = 1
_TGV_CUTE_EXT_TACTIC_CONFIGS: List[Tuple[int, int, int, bool]] = [
    # 1-CTA configs
    (64, 8, 6, False),  # 0
    (64, 8, 8, False),  # 1 (default)
    (64, 8, 10, False),  # 2
    (64, 8, 12, False),  # 3
    (64, 16, 6, False),  # 4
    (64, 16, 8, False),  # 5
    (64, 16, 11, False),  # 6
    (64, 32, 6, False),  # 7
    (64, 32, 9, False),  # 8
    (64, 64, 7, False),  # 9
    (64, 128, 4, False),  # 10
    (128, 8, 6, False),  # 11
    (128, 16, 6, False),  # 12
    (128, 32, 5, False),  # 13
    (128, 64, 4, False),  # 14
    (128, 128, 3, False),  # 15
    # 2-CTA configs
    (64, 16, 6, True),  # 16
    (64, 16, 8, True),  # 17
    (64, 16, 12, True),  # 18
    (64, 32, 6, True),  # 19
    (64, 32, 8, True),  # 20
    (64, 32, 11, True),  # 21
    (64, 64, 6, True),  # 22
    (64, 64, 9, True),  # 23
    (64, 128, 7, True),  # 24
    (128, 16, 6, True),  # 25
    (128, 32, 6, True),  # 26
    (128, 64, 5, True),  # 27
    (128, 128, 4, True),  # 28
]


class WorkTileInfo(NamedTuple):
    """Which output tile this CTA processes. Mirrors the original WorkTileInfo."""

    M_idx: cutlass.Int32
    N_idx: cutlass.Int32
    L_idx: cutlass.Int32
    K_idx_start: cutlass.Int32
    K_idx_end: cutlass.Int32


def _tcgen05_fence(*, before: bool) -> None:
    """``tcgen05.fence::{before,after}_thread_sync`` around an mbarrier hand-off of
    tcgen05 work (PTX ISA canonical tcgen05 synchronization patterns)."""
    llvm.inline_asm(
        None,
        [],
        f"tcgen05.fence::{'before' if before else 'after'}_thread_sync;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class TgvGemmCuteExtKernel:
    """
    Low-latency Blackwell GEMM kernel rewritten with cute_ext primitives,
    keeping the raw-mbarrier 7-warp specialization from the C++ kernel
    in ``include/flashinfer/gemm/tgv_gemm.cuh``.
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        cta_m: int = 64,
        cta_n: int = 8,
        cta_k: int = _TGV_CUTE_EXT_CTA_K,
        num_ab_stage: int = 8,
        use_2cta: bool = False,
        use_pdl: bool = False,
        pdl_launch: Optional[bool] = None,
        pdl_count: int = -1,
        out_dtype: Type[cutlass.Numeric] = cutlass.BFloat16,
    ):
        self.acc_dtype = acc_dtype
        self.out_dtype = out_dtype
        self.cta_m = cta_m
        self.cta_n = cta_n
        self.cta_k = cta_k
        self.num_ab_stage = num_ab_stage
        self.use_2cta = use_2cta
        self.use_pdl = use_pdl
        self.pdl_launch = pdl_launch if pdl_launch is not None else use_pdl
        self.pdl_count = pdl_count

        # 1-CTA: cta_n ∈ [8, 256] step 8 (bf16 tcgen05.mma atom limit).
        # 2-CTA: cta_n ∈ [16, 256] step 16 (bf16 K-major cluster mma).
        min_n, step_n = (16, 16) if use_2cta else (8, 8)
        if cta_n < min_n or cta_n > 256 or cta_n % step_n != 0:
            raise ValueError(
                f"cta_n={cta_n} invalid for use_2cta={use_2cta}: "
                f"bf16 K-major mma requires N ∈ [{min_n}, 256] step {step_n}"
            )

        # Fixed configuration matching the C++ / DSL kernels.
        self.threads_per_cta = 256  # 8 warps (warp 3 unused)
        if use_2cta:
            # 2-CTA cluster along M; joint MMA tile = (cta_m*2, cta_n).
            self.cluster_shape = (2, 1, 1)
            self.mma_tiler_mn = (cta_m * 2, cta_n)
            self.cta_group = tcgen05.CtaGroup.TWO
            self.tma_op = cute_ext.OperationTypeEnum.SM100_TMA_LOAD_2SM
        else:
            # 1 SM mode, 1x1 cluster, no multicast.
            self.cluster_shape = (1, 1, 1)
            self.mma_tiler_mn = (cta_m, cta_n)
            self.cta_group = tcgen05.CtaGroup.ONE
            self.tma_op = cute_ext.OperationTypeEnum.SM90_TMA_LOAD

    def __repr__(self) -> str:
        return (
            f"TgvGemmCuteExtKernel_cta{self.cta_m}x{self.cta_n}x{self.cta_k}"
            f"_2cta{int(self.use_2cta)}_pdl{int(self.use_pdl)}"
            f"_out{self.out_dtype.__name__.lower()}"
        )

    @cute.experimental.jit
    def __call__(
        self,
        a: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L), K-major
        b: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L), K-major
        c: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L), M-major
        stream: cuda.CUstream,
    ):
        # Each CTA processes one (CTA_M, CTA_N) output tile, no persistence.
        # Round the (M, N) tile counts up to a full cluster so 2-CTA launches
        # don't leave an odd M-CTA without its peer (e.g. M=2880, cta_m=64 →
        # 45 M-tiles, invalid for cluster=(2,1,1)). cute_ext OOB-protects the
        # extra CTA's GMEM stores.
        grid = cute.round_up(
            (
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n),
                c.layout.shape[2],
            ),
            self.cluster_shape,
        )
        self.kernel(a, b, c).launch(
            grid=grid,
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(utils.get_smem_capacity_in_bytes("sm_100")),
            stream=stream,
            use_pdl=self.pdl_launch,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        mA: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L), K-major
        mB: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L), K-major
        mC: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L), M-major
    ):
        """
        Device-side dispatcher: build MMA descriptor, allocate SMEM/TMEM/
        mbarriers, init barriers, dispatch into dma_a/dma_b/mma/epilog warps.
        """
        DMA_Stage = self.num_ab_stage

        # ---- Tiled MMA ----
        # make_trivial_tiled_mma picks the largest tcgen05.mma atom that fits
        # mma_tiler_mn. 1-CTA bf16 (64,8): Mma_M=(16,4)=64, Mma_N=8, Mma_K=16.
        # 2-CTA bf16 K-major (128,N): Mma_M=128 split across cluster, Mma_K=16.
        a_major = utils.LayoutEnum.from_tensor(mA).mma_major_mode()
        b_major = utils.LayoutEnum.from_tensor(mB).mma_major_mode()
        ab_dtype = mA.element_type  # bf16
        c_dtype = mC.element_type  # bf16
        d_layout = utils.LayoutEnum.from_tensor(mC)

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype,
            ab_dtype,
            a_major,
            b_major,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_mn,
        )
        num_mma_ctas = cute.size(tiled_mma.thr_id.shape)  # 1 (1-CTA) or 2 (2-CTA)

        # NumMma_K = CTA_K/Mma_K — inner-K trip count for the MMA warp.
        # bf16 default: Mma_K=16, CTA_K=128 → mma_inst_tile_k = 8.
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = self.cta_k // mma_inst_shape_k

        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        a_tiler_mk = (self.cta_m, self.cta_k)  # CTA tile of A
        # 2-CTA splits B across cluster M (each CTA loads Mma_N/2 cols).
        b_tiler_nk = (self.cta_n // num_mma_ctas, self.cta_k)
        c_tiler_mn = (self.cta_m, self.cta_n)  # CTA tile of C

        # ---- WorkTileInfo (static 1-tile-per-CTA scheduler) ----
        # blockIdx maps directly to (M_tile, N_tile, batch). K_idx_start/end
        # carry over from C++ for future split-K; here every CTA reduces the
        # full K range (= ceil(K/CTA_K) tiles, e.g. 12 for K=1536).
        bidx, bidy, bidz = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        # In 2-CTA mode: identify which CTA within the (2,1,1) cluster owns
        # the leader role (issues MMAs / multicasts commits). In 1-CTA mode
        # every CTA is its own leader.
        leader_rank = cutlass.Int32(0)
        if cutlass.const_expr(self.use_2cta):
            cta_rank_in_cluster = cute.arch.make_warp_uniform(
                cute.arch.block_idx_in_cluster()
            )
            is_leader = cta_rank_in_cluster == 0
        else:
            cta_rank_in_cluster = leader_rank
            is_leader = cutlass.Boolean(True)

        # ---- SMEM A/B (DMA_Stage-staged, swizzled ComposedLayout) ----
        # alignment=1024 covers TMA's natural alignment requirements. 2-CTA
        # halves the M extent of sA and the N extent of sB internally (the SS
        # atom splits A across cluster M and B across cluster M).
        a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mnk_tiler,
            ab_dtype,
            DMA_Stage,
        )  # ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage) — Sw<3,4,3>
        b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mnk_tiler,
            ab_dtype,
            DMA_Stage,
        )  # ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage) — Sw<3,4,3>

        # sA per-CTA M extent = cta_m for both modes; sB per-CTA N extent =
        # cta_n (1-CTA) or cta_n/2 (2-CTA, SS atom splits B across cluster M).
        sA = cute_ext.allocate(  # ((Mma_M_per_cta, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
            ab_dtype,
            cute.AddressSpace.smem,
            a_smem_layout_staged,
            alignment=1024,
        )
        sB = cute_ext.allocate(  # ((Mma_N_per_cta, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
            ab_dtype,
            cute.AddressSpace.smem,
            b_smem_layout_staged,
            alignment=1024,
        )

        # ---- TMEM accumulator layout (manual alloc; see MMA warp) ----
        # cute_ext.make_tmem_layout_acc accounts for cta_group when handed
        # the cluster MMA tile, so the resulting layout is already per-CTA.
        # acc_layout per-CTA: ((Mma_M_per_cta, Mma_N), NumMma_M, NumMma_N, AccStage)
        # = (((16,4), cta_n), 1, 1, 1) for both 1-CTA and 2-CTA modes.
        # We still allocate TMEM manually (via cute.arch.alloc_tmem below) for better performance.
        acc_layout = cute_ext.make_tmem_layout_acc(
            tiled_mma,
            self.mma_tiler_mn,
            acc_stage=1,
        )

        # ---- Raw mbarriers (Int64 SMEM arrays) + tmem_base_ptr Int32 slot ----
        # Two flavors of barriers exist on Hopper/Blackwell:
        #   1. Named barriers (bar.arv/bar.sync) — Ampere-era, 16 hardware
        #      barriers per SM, only handle intra-CTA thread sync.
        #   2. Mbarriers (mbarrier.* PTX) — 64-bit in SMEM per barrier,
        #      software-programmable. Used here for (a) thread sync within a
        #      CTA, (b) TMA transaction-count tracking (TMA→SM signaling),
        #      and (c) 1-arrival "wake the consumer warp" patterns.
        # Allocated as 1D tensors; .iterator gives a Pointer[Int64] supporting
        # `bar + stage` arithmetic. Arrival counts are in the module docstring.
        bar_full_arr = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(DMA_Stage),
            alignment=8,
        )
        bar_empty_arr = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(DMA_Stage),
            alignment=8,
        )
        bar_mma_epilog_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )
        bar_tmem_alloc_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )
        # 2-CTA only: rendezvous of the two MMA warps before the collective
        # TMEM dealloc. 1-CTA passes None and never touches it.
        bar_tmem_dealloc = None
        if cutlass.const_expr(self.use_2cta):
            bar_tmem_dealloc_arr = cute_ext.allocate(
                cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
            )
            bar_tmem_dealloc = bar_tmem_dealloc_arr.iterator  # Pointer[Int64]
        # alloc_tmem writes the TMEM base address to this slot; both MMA and
        # EPILOG read it back via retrieve_tmem_ptr.
        tmem_base_arr = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        )

        bar_full = bar_full_arr.iterator  # Pointer[Int64], DMA_Stage
        bar_empty = bar_empty_arr.iterator  # Pointer[Int64], DMA_Stage
        bar_mma_epilog = bar_mma_epilog_arr.iterator  # Pointer[Int64]
        bar_tmem_alloc = bar_tmem_alloc_arr.iterator  # Pointer[Int64]
        tmem_base_ptr = tmem_base_arr.iterator  # Pointer[Int32]

        # ---- Barrier init (1 thread, PTX mbarrier.init is single-thread) ----
        # bar_full arrival = 2 * num_mma_ctas:
        #   1-CTA: 2 (own DMA_A + DMA_B)
        #   2-CTA: 4 (own + peer DMA_A/DMA_B, routed via SM100_TMA_LOAD_2SM
        #          onto the leader's bar). In 2-CTA the peer's copy is unused.
        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(DMA_Stage):
                    cute.arch.mbarrier_init(bar_full + i, 2 * num_mma_ctas)
                for i in range(DMA_Stage):
                    cute.arch.mbarrier_init(bar_empty + i, 1)  # MMA tcgen05.commit
                cute.arch.mbarrier_init(bar_mma_epilog, 1)  # MMA tcgen05.commit
                cute.arch.mbarrier_init(
                    bar_tmem_alloc, 32 + 128
                )  # MMA + 4 EPILOG warps
                if cutlass.const_expr(self.use_2cta):
                    cute.arch.mbarrier_init(bar_tmem_dealloc, 32)  # peer CTA's MMA warp

        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.use_2cta):
            # Cluster-wide sync: ensure peer's barriers are visible before any
            # cross-CTA arrives. Splitting arrive/wait lets the per-CTA tile
            # setup below overlap with the cluster barrier. (Implicit
            # intra-CTA sync — cluster.arrive requires all threads to reach.)
            cute.arch.cluster_arrive_relaxed()
        else:
            cute.arch.barrier()

        work_tile_info = WorkTileInfo(
            M_idx=bidx,
            N_idx=bidy,
            L_idx=bidz,
            K_idx_start=cutlass.Int32(0),
            K_idx_end=cute.ceil_div(cute.size(mA, mode=[1]), self.cta_k),
        )
        k_tile_count = work_tile_info.K_idx_end - work_tile_info.K_idx_start

        # ---- CTA-to-Value maps for TMA (constexpr) ----
        # Encodes "which cluster CTA is responsible for which slice", i.e.
        # the CTA-coord → logical-multicast-id layout. For our 1×1 cluster
        # (no multicast) it's an identity layout; cute_ext.tma_load still
        # needs this explicit layout to construct the TMA descriptor.
        a_cta_v_map = cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A")
        b_cta_v_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")

        # ---- Per-CTA tile views ----
        # local_tile(t, tiler, coord) = zipped_divide(t, tiler)[coord]; modes
        # passed `None` stay free. Per-CTA M/N extents:
        #   gA M extent = cta_m   (both modes; SS atom splits cluster M)
        #   gB N extent = cta_n   (1-CTA) or cta_n//2 (2-CTA)
        #   gD M×N      = cta_m × cta_n (cluster N is identical for both CTAs)
        gA_tile = cute.local_tile(  # (cta_m, cta_k, Tiles_K)
            mA,
            a_tiler_mk,
            (work_tile_info.M_idx, None, work_tile_info.L_idx),
        )
        # gB n-index: 1-CTA just bidy; 2-CTA each cluster N_idx covers
        # num_mma_ctas tiles of b_tiler, each CTA loads one half.
        if cutlass.const_expr(self.use_2cta):
            gB_n_idx = work_tile_info.N_idx * num_mma_ctas + cta_rank_in_cluster
        else:
            gB_n_idx = work_tile_info.N_idx
        gB_tile = cute.local_tile(  # (cta_n//num_mma_ctas, cta_k, Tiles_K)
            mB,
            b_tiler_nk,
            (gB_n_idx, None, work_tile_info.L_idx),
        )
        gD_tile = cute.local_tile(  # (cta_m, cta_n)
            mC,
            c_tiler_mn,
            (work_tile_info.M_idx, work_tile_info.N_idx, work_tile_info.L_idx),
        )

        if cutlass.const_expr(self.use_2cta):
            # Cluster sync deferred to here — tile-view setup above runs in
            # parallel with the cluster barrier latency.
            cute.arch.cluster_wait()

        # ---- Warp dispatch (warp 3 idle, kept for 256-thread parity) ----
        if warp_idx == 0:
            self.dma_a_warp(
                bar_full,
                bar_empty,
                leader_rank,
                gA_tile,
                sA,
                a_cta_v_map,
                k_tile_count,
            )
        elif warp_idx == 1:
            self.dma_b_warp(
                bar_full,
                bar_empty,
                leader_rank,
                gB_tile,
                sB,
                b_cta_v_map,
                k_tile_count,
            )
        elif warp_idx == 2:
            self.mma_warp(
                is_leader,
                bar_full,
                bar_empty,
                bar_mma_epilog,
                bar_tmem_alloc,
                bar_tmem_dealloc,
                cta_rank_in_cluster,
                tiled_mma,
                sA,
                sB,
                tmem_base_ptr,
                acc_layout,
                mma_inst_tile_k,
                k_tile_count,
            )
        elif warp_idx >= 4:
            # Epilog tid is 128..255 in the CTA; offset to 0..127 for partition.
            epi_tid = tidx - 128
            self.epilog_warp(
                bar_mma_epilog,
                bar_tmem_alloc,
                tmem_base_ptr,
                acc_layout,
                gD_tile,
                epi_tid,
                c_dtype,
                d_layout,
            )

        if cutlass.const_expr(self.use_2cta):
            # Cluster-wide exit barrier (sgl-project/sglang#32907): cluster
            # sync previously existed only at kernel entry, so one CTA could
            # retire while its peer still had in-flight peer-redirected
            # mbarrier arrives / multicast tcgen05.commit / TMEM dealloc
            # targeting it -> CUDBG_EXCEPTION_CLUSTER_BLOCK_NOT_PRESENT
            # (Xid 13 "CTA Not Present"). All 8 warps (256 threads) of both
            # cluster CTAs arrive here after their dispatch branch returns,
            # so no CTA can exit until the peer's tail work has landed.
            # relaxed arrive (matches the entry barrier at L378): this fence is
            # only about CTA lifetime, not inter-CTA smem visibility -- every
            # cross-CTA datum above already flows through mbarrier phases.
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

    # ====================================================================
    # DMA_A WARP — TMA-loads A tiles into sA[..., stage], one per K-iter.
    # 1-CTA: SM90_TMA_LOAD into local sA, arrives on local bar_full.
    # 2-CTA: SM100_TMA_LOAD_2SM (cluster-aware); peer's complete_tx routes
    #        onto the LEADER's bar_full. arrive_and_expect_tx is also
    #        peer-redirected to leader_rank.
    # ====================================================================
    @cute.experimental.jit
    def dma_a_warp(
        self,
        bar_full,  # Pointer[Int64], DMA_Stage entries
        bar_empty,  # Pointer[Int64], DMA_Stage entries
        leader_rank: cutlass.Int32,  # used for the 2-CTA arrive-peer redirect
        gA_tile: cute.Tensor,  # (CTA_M, CTA_K, Tiles_K) — this CTA's A strip
        sA: cute.Tensor,  # ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
        a_cta_v_map: cute.Layout,
        k_tile_count: cutlass.Int32,
    ):
        DMA_Stage = self.num_ab_stage

        # Phase-bit walkthrough for `bar_empty` (DMA_Stage=2 example).
        # Init phase = 0; phase flips on each arrival-count round. We wait
        # on the *old* phase about to flip away.
        #   k_tile  stage  old empty_phase  → wait for flip
        #     0       0          1                0→1 (init=0≠1, passes)
        #     1       1          1                0→1
        #     2       0          0                1→0 (slot reused)
        #     3       1          0                1→0
        #     4       0          1                0→1
        # Flip empty_phase once per DMA_Stage iters.
        empty_phase = cutlass.Int32(1)
        pdl_count = self.pdl_count

        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % DMA_Stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)

            # cute_ext.tma_load only manages TX bytes, NEVER the arrival
            # count (in the pipeline path producer_commit handles that).
            # bar_full[stage] needs 2 (1-CTA) or 4 (2-CTA) arrivals; we
            # arrive ourselves via the fused mbarrier.arrive.expect_tx PTX,
            # paired with update_expect_tx=False so TX bytes aren't
            # double-counted. peer_cta_rank_in_cluster=leader_rank in 2-CTA
            # routes the arrive onto the leader's bar; None (1-CTA) lands
            # on the local CTA's bar.
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar_full + stage,
                    # 1 stage of A SMEM = sizeof(bf16) × CTA_M × CTA_K
                    cute.size_in_bytes(
                        sA.element_type,
                        cute.slice_(sA.layout, (None, None, None, 0)),
                    ),
                    peer_cta_rank_in_cluster=leader_rank if self.use_2cta else None,
                )
            cute_ext.tma_load(
                gA_tile[None, None, k_tile],  # (CTA_M, CTA_K) GMEM slice
                sA[None, None, None, stage],  # ((Mma_M,Mma_K),NumMma_M,NumMma_K)
                (bar_full + stage).value,  # Pointer→ir.Value bridge
                cta_v_map=a_cta_v_map,
                tma_operation_type=self.tma_op,
                update_expect_tx=False,
            )

            if stage == (DMA_Stage - 1):
                empty_phase = empty_phase ^ 1

            # PDL: launch dependent grid at pdl_count (default -1 = end).
            if cutlass.const_expr(self.use_pdl):
                if k_tile == pdl_count:
                    cute.arch.griddepcontrol_launch_dependents()

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        # producer tail: drain the remaining bar_empty arrivals from MMA so the
        # CTA stays alive until MMA's last tcgen05.commit lands — otherwise the
        # arrive can hit freed SMEM (illegal memory access).
        for k_tile in cutlass.range(DMA_Stage, unroll=1):
            stage = (k_tile + k_tile_count) % DMA_Stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if stage == (DMA_Stage - 1):
                empty_phase = empty_phase ^ 1

    # ====================================================================
    # DMA_B WARP — same as DMA_A but for B; also drives the PDL
    # griddepcontrol.wait and signals the epilog when all activation loads
    # are issued.
    # 2-CTA: SM100_TMA_LOAD_2SM + arrive redirected to leader_rank.
    # ====================================================================
    @cute.experimental.jit
    def dma_b_warp(
        self,
        bar_full,  # Pointer[Int64], DMA_Stage entries
        bar_empty,  # Pointer[Int64], DMA_Stage entries
        leader_rank: cutlass.Int32,
        gB_tile: cute.Tensor,  # (CTA_N, CTA_K, Tiles_K) — this CTA's B strip
        sB: cute.Tensor,  # ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
        b_cta_v_map: cute.Layout,
        k_tile_count: cutlass.Int32,
    ):
        DMA_Stage = self.num_ab_stage

        # PDL griddepcontrol.wait: B is the activation, produced by an
        # upstream kernel. Block until that kernel signals completion. A
        # is weights (already resident), so DMA_A doesn't need this.
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        # See dma_a_warp for empty_phase + fused arrive-and-expect-tx logic.
        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % DMA_Stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)

            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar_full + stage,
                    # 1 stage of B SMEM = sizeof(bf16) × CTA_N × CTA_K
                    cute.size_in_bytes(
                        sB.element_type,
                        cute.slice_(sB.layout, (None, None, None, 0)),
                    ),
                    peer_cta_rank_in_cluster=leader_rank if self.use_2cta else None,
                )
            cute_ext.tma_load(
                gB_tile[None, None, k_tile],  # (CTA_N, CTA_K) GMEM slice
                sB[None, None, None, stage],  # ((Mma_N,Mma_K),NumMma_N,NumMma_K)
                (bar_full + stage).value,
                cta_v_map=b_cta_v_map,
                tma_operation_type=self.tma_op,
                update_expect_tx=False,
            )

            if stage == (DMA_Stage - 1):
                empty_phase = empty_phase ^ 1

        # producer tail: drain the remaining bar_empty arrivals from MMA so the
        # CTA stays alive until MMA's last tcgen05.commit lands — otherwise the
        # arrive can hit freed SMEM (illegal memory access).
        for k_tile in cutlass.range(DMA_Stage, unroll=1):
            stage = (k_tile + k_tile_count) % DMA_Stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if stage == (DMA_Stage - 1):
                empty_phase = empty_phase ^ 1

    # ====================================================================
    # MMA WARP — owns the TMEM accumulator and issues every tcgen05.mma.
    # 1-CTA: every CTA's MMA warp runs the K-loop on its own TMEM.
    # 2-CTA: BOTH CTAs alloc/dealloc TMEM (cluster-coherent via
    #        is_two_cta=True); only the leader runs the K-loop; commits
    #        multicast to both CTAs (mask=0b11) so each one's bar_empty /
    #        bar_mma_epilog still gets signaled.
    # ====================================================================
    @cute.experimental.jit
    def mma_warp(
        self,
        is_leader: cutlass.Boolean,
        bar_full,  # Pointer[Int64], DMA_Stage entries
        bar_empty,  # Pointer[Int64], DMA_Stage entries
        bar_mma_epilog,  # Pointer[Int64], 1 entry
        bar_tmem_alloc,  # Pointer[Int64], 1 entry, 160-arrival
        bar_tmem_dealloc,  # Pointer[Int64], 1 entry, 32-arrival (2-CTA only, else None)
        cta_rank_in_cluster: cutlass.Int32,
        tiled_mma: cute.TiledMma,
        sA: cute.Tensor,  # ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
        sB: cute.Tensor,  # ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
        tmem_base_ptr,  # Pointer[Int32] — SMEM slot for TMEM addr
        acc_layout: cutlass.Constexpr,  # TMEM accumulator layout
        mma_inst_tile_k: cutlass.Constexpr,  # NumMma_K — inner loop count
        k_tile_count: cutlass.Int32,  # Tiles_K — outer loop count
    ):
        DMA_Stage = self.num_ab_stage

        # ---- TMEM allocation (manual) ----
        # We allocate TMEM explicitly here rather than via the convenience
        # helper ``cute_ext.allocate(tmem)`` so that we can *hide the TMEM
        # allocation latency*: the alloc must happen here in MMA's prolog
        # (we need the TMEM ptr before the first tcgen05.mma), but it
        # serializes against the next CTA's own alloc on the back-to-back
        # PDL launch path. By manually emitting
        # ``relinquish_tmem_alloc_permit`` right after our alloc completes,
        # the next CTA can start ITS allocation in parallel with this
        # CTA's MMA work rather than stalling at its own prolog waiting
        # for us. cute_ext.allocate(tmem) does not expose this early-
        # relinquish control — worth ~3% on the PDL hot path.
        #
        # TMEM on SM100 has 128 lanes × 512 columns × 4B = 256KB total. We
        # allocate half (256 of 512 cols) so the next CTA can prefetch its
        # alloc on the other half. For CTA_M=64 the accumulator only uses
        # 64 lanes (16 lanes × 4 subpartitions), well within half-of-TMEM.
        # 2-CTA: alloc/relinquish/dealloc are cluster-coherent (is_two_cta=True).
        num_tmem_cols = 256
        cute.arch.alloc_tmem(num_tmem_cols, tmem_base_ptr, is_two_cta=self.use_2cta)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)  # phase 0: 32 of 160
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=self.use_2cta)

        # Bind acc_layout to the just-allocated TMEM ptr. Drop the trailing
        # AccStage=1 mode so the MMA atom sees (MMA, Mma_M, Mma_N).
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        tAcc = cute.make_tensor(
            tmem_ptr, acc_layout
        )  # ((Mma_M_per_cta, Mma_N), NumMma_M, NumMma_N, AccStage)
        acc_view = tAcc[
            None, None, None, 0
        ]  # ((Mma_M_per_cta, Mma_N), NumMma_M, NumMma_N)

        if is_leader:
            # MMA atom: each cute_ext.dot call = 1 tcgen05.mma. ACCUMULATE:
            #   False → C  = A*B  (first inst of first k-tile only)
            #   True  → C += A*B
            mma_atom = cute.make_mma_atom(tiled_mma.op)
            # 2-CTA: mask=0b11 multicasts the commit to both CTAs' bar_empty.
            commit_mask = 0b11 if self.use_2cta else None

            full_phase = cutlass.Int32(0)
            for k_tile in cutlass.range(k_tile_count, unroll=1):
                stage = k_tile % DMA_Stage
                cute.arch.mbarrier_wait(bar_full + stage, full_phase)

                # Inner-K loop: NumMma_K straight-line tcgen05.mma instructions.
                # cute_ext.dot expects rank-3 operands in the MMA fragment
                # profile (MMA_atom, REST_M, REST_K). Slicing
                # `sA[None,None,k_block,stage]` leaves rank 2
                # ((Mma_M, Mma_K), NumMma_M), so pad explicitly via
                # cute.append_ones.
                for k_block in range(mma_inst_tile_k):
                    if k_block == 0:
                        # First inst of the very first k_tile clears TMEM.
                        mma_atom.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                    else:
                        mma_atom.set(tcgen05.Field.ACCUMULATE, True)
                    a_frag = cute.append_ones(
                        sA[None, None, k_block, stage],
                        up_to_rank=3,
                    )
                    b_frag = cute.append_ones(
                        sB[None, None, k_block, stage],
                        up_to_rank=3,
                    )
                    cute_ext.dot(mma_atom, a_frag, b_frag, acc_view)

                # CRITICAL: tcgen05.commit MUST be inside elect_one(). Unlike
                # the C++ ``cutlass::arch::umma_arrive`` helper (which has an
                # internal elect_one_sync), the DSL's tcgen05.commit has no
                # guard — without this, all 32 lanes commit (32× redundant
                # arrivals on bar_empty). This was a major perf bug encountered
                # during development.
                with cute.arch.elect_one():
                    tcgen05.commit(bar_empty + stage, commit_mask, self.cta_group)

                if (k_tile % DMA_Stage) == (DMA_Stage - 1):
                    full_phase = full_phase ^ 1

            # Wake the epilog (1-thread arrive on the 1-arrival barrier).
            with cute.arch.elect_one():
                tcgen05.commit(bar_mma_epilog, commit_mask, self.cta_group)

        # ---- TMEM dealloc: wait for own EPILOG's tcgen05.ld to retire, free.
        # The release arrival must land in phase 1. Made while phase 0 is still
        # open (an EPILOG warp has not arrived yet), it counts toward phase 0 and
        # the parity-1 wait passes before the EPILOG's tcgen05.ld. Phase-0 wait
        # and tcgen05 fences as in flashinfer PR 7443 for the split-K kernel.
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)  # phase 1: 32 of 160
        cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
        if cutlass.const_expr(self.use_2cta):
            # tcgen05.dealloc.cta_group::2 is collective: the PTX ISA requires the
            # issuing warp and the peer CTA's warp to synchronize their TMEM
            # accesses first. Rendezvous with the peer MMA warp, as
            # cutlass.utils.TmemAllocator.free does.
            cute.arch.mbarrier_arrive(bar_tmem_dealloc, cta_rank_in_cluster ^ 1)
            cute.arch.mbarrier_wait(bar_tmem_dealloc, 0)
        # tcgen05.fence::after_thread_sync: order the dealloc after the tcgen05
        # work handed off through the waits above.
        _tcgen05_fence(before=False)
        cute.arch.dealloc_tmem(tmem_ptr, num_tmem_cols, is_two_cta=self.use_2cta)

    # ====================================================================
    # EPILOG WARPS — TMEM → RMEM → bf16 cast → direct st.global to GMEM.
    # ====================================================================
    @cute.experimental.jit
    def epilog_warp(
        self,
        bar_mma_epilog,  # Pointer[Int64], 1 entry
        bar_tmem_alloc,  # Pointer[Int64], 1 entry, 160-arrival
        tmem_base_ptr,  # Pointer[Int32] — SMEM slot from MMA
        acc_layout: cutlass.Constexpr,
        gD_tile: cute.Tensor,  # (CTA_M, CTA_N) — this CTA's output tile
        epi_tid: cutlass.Int32,  # 0..127 within the 4 EPILOG warps
        c_dtype: cutlass.Constexpr,
        d_layout: cutlass.Constexpr,
    ):
        # Sync with MMA's alloc_tmem (phase 0 of bar_tmem_alloc): 128 arrivals
        # from this warp + 32 from MMA = 160. tmem_base_ptr is uninitialized
        # before the wait clears.
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)

        # Bind acc_layout to the TMEM ptr written by alloc_tmem.
        # acc_layout shape: ((Mma_M, Mma_N), NumMma_M, NumMma_N, AccStage)
        #                 = (((16,4), Mma_N), 1, 1, 1).
        # acc_view drops the 1×1×1 outer modes → (Mma_M, Mma_N).
        #
        # Concrete TMEM layout for 1-CTA bf16 M64 N8:
        #   tmem_[32b](0x0...) o (((16,4),8),1,1,1):(((65536,2097152),1),0,0,0)
        #   TMEM addr = [31:16=dp_lane, 15:0=column]:
        #     stride between dp lanes = 1<<16 = 65536           (Mma_M_per_subp)
        #     stride between subparts = 65536 × 32 = 2097152    (NumSubp=4)
        #     stride between cols     = 1                       (N is contiguous)
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        tCtAcc = cute.make_tensor(
            tmem_ptr, acc_layout
        )  # ((Mma_M_per_cta, Mma_N), NumMma_M, NumMma_N, AccStage)
        acc_view = tCtAcc[((None, None), 0, 0, 0)]  # (Mma_M_per_cta, Mma_N)

        # ---- t2r tiled-copy + per-thread RMEM layout ----
        # ``sm100_utils.get_tmem_load_op`` picks the best tcgen05.ld atom
        # for the (M, N, K) tile and output dtype. For CTA_M=64, M/N-major
        # output, the chosen atom is SM100_TMEM_LOAD_16dp256b1x — the 16dp
        # variant is required because each MMA subpartition only uses 16
        # of the 32 lanes per subpartition (CTA_M=64 split across 4 subps =
        # 16 lanes each). Other tcgen05.ld atoms would also be functional;
        # 16dp is just optimal here.
        # epi_tile = full CTA tile: N=8 fits in one tcgen05.ld instance
        # (16dp × 8col → 4 regs per thread × 128 threads). Larger N would
        # need a sub-tile loop here.
        epi_tile = (self.cta_m, self.cta_n)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            (self.cta_m, self.cta_n, self.cta_k),
            d_layout,
            c_dtype,
            self.acc_dtype,
            epi_tile,
            self.use_2cta,
        )
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(copy_atom_t2r, acc_view)

        # flat_divide → (CTA_M, CTA_N, Rest_M=1, Rest_N=1); the rank-4 shape
        # is required because make_t2r_rmem_layout internally does [...,0,0].
        gD_epi = cute.flat_divide(gD_tile, epi_tile)

        # rmem_layout matches one t2r copy's destination. The TMEM-side
        # partition (CpyS) per-subpartition view is shape ((CpyS_N=8,
        # CpyS_M=16), NumCpy_M=1, NumCpy_N=Mma_N/CpyS_N) — i.e. each
        # tcgen05.ld.16dp256bit.x1 instance moves 16dp×8col elements per
        # subpartition. Per the PTX page for tcgen05.ld.16dp256bit, thread
        # 0 of an SM100_TMEM_LOAD_16dp256b1x copy receives 4 registers at
        # output coordinates (0,0), (0,1), (8,0), (8,1) — i.e. a (2,2)
        # per-thread value-tile, hence CpyD = (2,2). The dst rmem layout
        # mirrors this CpyD shape but with stride (1, 2) so the registers
        # are stored contiguously.
        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
        rAcc = cute_ext.allocate(  # fp32, per-thread
            self.acc_dtype,
            cute.AddressSpace.rmem,
            rmem_layout,
            alignment=32,
        )
        rD = cute_ext.allocate(  # bf16, per-thread
            c_dtype,
            cute.AddressSpace.rmem,
            rmem_layout,
            alignment=32,
        )
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)  # accumulator ready

        # TMEM → RMEM (one tcgen05.ld for the 64×8 tile).
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)

        # tcgen05.ld is async — fence makes the result visible to (a) the
        # rAcc.load() below and (b) MMA's dealloc_tmem after we arrive
        # on bar_tmem_alloc phase 1.
        cute.arch.fence_view_async_tmem_load()
        # tcgen05.fence::before_thread_sync: publish the retired tcgen05.ld to
        # MMA's dealloc across the mbarrier hand-off.
        _tcgen05_fence(before=True)

        # Phase 1 of bar_tmem_alloc: 128 from this warp + 32 from MMA = 160.
        # MMA's mbarrier_wait(bar_tmem_alloc, 1) clears and it dealloc's.
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        # fp32 → bf16 cast in RMEM.
        rD.store(rAcc.load().to(c_dtype))

        # RMEM → GMEM: cute_ext.partition_and_copy does two things —
        #   (a) PARTITION: splits non-RMEM operands by thr_copy's thread-value
        #       layout (RMEM operands skipped — already per-thread). Here only
        #       gD_epi (the dst) gets partitioned; rD stays as-is.
        #   (b) INSTRUCTION: picked from (src, dst) memspace pair —
        #         TMEM→RMEM = tcgen05.ld     (atom-driven)
        #         RMEM→GMEM = st.global     (simt_auto_vec_copy)
        #         GMEM→SMEM = cp.async       (async_op=True)
        # We REUSE thr_t2r so each lane writes to the same (m, n) GMEM coords
        # where it read from TMEM. A different copy atom would scatter
        # registers to the wrong addresses.
        #
        # OOB STORES are handled automatically by the cute_ext lowering —
        # simt_auto_vec_copy infers predBounds from the destination tensor's
        # MemRef shape (propagated from mC through local_tile / flat_divide).
        # Lanes whose (m, n) coord is past mC's shape simply don't issue an
        # st.global. Verified with a non-aligned M=63 test: writes to in-bounds rows
        # 0..62 land correctly, the M=63 row + the rest of the (64,8) tile
        # stay untouched. So the C++ / vanilla-DSL kernels' explicit predicate
        # dance (make_identity_tensor + elem_less + basic_copy_if) isn't
        # needed; the explicit `predicated_tensor_origin` marker isn't either.
        cute_ext.partition_and_copy(thr_t2r, rD, gD_epi[None, None, 0, 0])


# =====================================================================
# BMM wrapper — feed (L, M, K) / (L, K, N) / (L, M, N) PyTorch-shaped
# cute tensors and select-permute them into the kernel's (M, K, L) /
# (N, K, L) / (M, N, L) view.
# =====================================================================
@cute.experimental.jit
def _tgv_bmm(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,  # (L, M, K) from PyTorch
    b: cute.Tensor,  # (L, K, N) from PyTorch (permuted)
    c: cute.Tensor,  # (L, M, N) from PyTorch (permuted)
    stream: cuda.CUstream,
):
    a = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0]))
    b = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0]))
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    gemm_op(a, b, c, stream)


# =====================================================================
# Compile + cache helpers (FlashInfer-side wrappers around the kernel).
# Caches by *config* only — the kernel uses ``mark_layout_dynamic`` so
# the same compiled binary handles any shape. Representative tensors used
# at compile time pin the dtype, rank, and leading-dim pattern; their
# concrete extents are erased.
# =====================================================================
_TORCH_TO_CUTLASS_DTYPE = {
    torch.bfloat16: cutlass.BFloat16,
}

_TORCH_TO_CUTLASS_OUT_DTYPE = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}

# Per-process cache mapping (dtype, config…) → compiled cute_ext callable.
# We need to construct concrete cute.Tensors once and reuse the resulting
# compiled function across all live calls; a fresh build per call would
# defeat the cache.
_TGV_CUTE_EXT_COMPILE_CACHE: dict = {}


def _detect_leading_dim(t: torch.Tensor) -> int:
    """Return the index of the stride-1 dim. Used by ``_to_cute_swap`` so the
    integration layer auto-adapts to whichever of K-major / MN-major the
    caller hands us. Falls back to the innermost dim if no stride-1 dim
    exists (shouldn't happen for well-formed tensors)."""
    for i, s in enumerate(t.stride()):
        if s == 1:
            return i
    return t.dim() - 1


def _make_layout_tensor(
    shape: Tuple[int, ...], dtype: torch.dtype, leading_dim: int
) -> torch.Tensor:
    """Allocate a torch tensor of ``shape`` where ``leading_dim`` is the
    contiguous (stride-1) dim. Used to synthesize representative tensors at
    compile time for arbitrary layout patterns."""
    ndim = len(shape)
    # Permute leading_dim to last, allocate contig, then permute back.
    perm = [i for i in range(ndim) if i != leading_dim] + [leading_dim]
    permuted_shape = tuple(shape[p] for p in perm)
    t = torch.empty(permuted_shape, dtype=dtype, device="cuda")
    inv_perm = [perm.index(i) for i in range(ndim)]
    return t.permute(inv_perm)


def _make_compile_repr_tensors(
    dtype: torch.dtype,
    c_dtype: torch.dtype,
    a_leading: int,
    b_leading: int,
    c_leading: int,
):
    """Build representative tensors with strides matching the requested
    leading-dim pattern. After the A↔B swap, the cute_ext kernel sees:
        A_ce: shape (L, N_pt, K)   <- (L=1, dim 1, dim 2)
        B_ce: shape (L, K, M_pt)   <- (L=1, dim 1, dim 2)
        C_ce: shape (L, N_pt, M_pt) <- (L=1, dim 1, dim 2)
    Each of {a, b, c}_leading ∈ {1, 2} selects which dim is the stride-1
    one (K vs kernel-M for A, K vs kernel-N for B, kernel-M vs kernel-N
    for C). The compiled binary is specialized per ``(a, b, c)_leading``
    tuple.
    """
    # Pick aligned-but-small extents that satisfy the kernel's tile shape
    # while leaving room for either dim to be the contig one.
    M, N, K, L = 64, 8, 128, 1  # M_pt=M, N_pt=N
    A_t = _make_layout_tensor(
        (L, N, K), dtype, a_leading
    )  # kernel A shape (L, N_pt, K)
    B_t = _make_layout_tensor(
        (L, K, M), dtype, b_leading
    )  # kernel B shape (L, K, M_pt)
    C_t = _make_layout_tensor(
        (L, N, M), c_dtype, c_leading
    )  # kernel C shape (L, N_pt, M_pt)

    a_ = from_dlpack(A_t, assumed_align=32).mark_layout_dynamic(leading_dim=a_leading)
    b_ = from_dlpack(B_t, assumed_align=32).mark_layout_dynamic(leading_dim=b_leading)
    c_ = from_dlpack(C_t, assumed_align=32).mark_layout_dynamic(leading_dim=c_leading)

    return a_, b_, c_


def _get_compiled_cute_ext_kernel(
    dtype: torch.dtype,
    c_dtype: torch.dtype,
    cta_m: int,
    cta_n: int,
    cta_k: int,
    num_ab_stage: int,
    use_2cta: bool,
    use_pdl: bool,
    a_leading: int,
    b_leading: int,
    c_leading: int,
):
    """Compile (or fetch from cache) a TgvGemmCuteExtKernel for the given
    config + layout pattern. Returns a callable with signature
        compiled(a_, b_, c_, stream)
    accepting cute.Tensors. The same compiled binary handles any
    (M, N, K, L) consistent with the (a, b, c)_leading layout pattern
    thanks to ``mark_layout_dynamic``.
    """
    key = (
        dtype,
        c_dtype,
        cta_m,
        cta_n,
        cta_k,
        num_ab_stage,
        bool(use_2cta),
        bool(use_pdl),
        a_leading,
        b_leading,
        c_leading,
    )
    cached = _TGV_CUTE_EXT_COMPILE_CACHE.get(key)
    if cached is not None:
        return cached

    if dtype not in _TORCH_TO_CUTLASS_DTYPE:
        raise ValueError(
            f"TGV cute_ext backend supports {list(_TORCH_TO_CUTLASS_DTYPE)}; got {dtype}."
        )
    if c_dtype not in _TORCH_TO_CUTLASS_OUT_DTYPE:
        raise ValueError(
            f"TGV cute_ext output supports {list(_TORCH_TO_CUTLASS_OUT_DTYPE)}; "
            f"got {c_dtype}."
        )

    gemm = TgvGemmCuteExtKernel(
        acc_dtype=cutlass.Float32,
        cta_m=cta_m,
        cta_n=cta_n,
        cta_k=cta_k,
        num_ab_stage=num_ab_stage,
        use_2cta=use_2cta,
        use_pdl=use_pdl,
        out_dtype=_TORCH_TO_CUTLASS_OUT_DTYPE[c_dtype],
    )

    a_, b_, c_ = _make_compile_repr_tensors(
        dtype,
        c_dtype,
        a_leading,
        b_leading,
        c_leading,
    )
    fake_stream = make_fake_stream()

    compiled = cute_ext.compile(_tgv_bmm, gemm, a_, b_, c_, fake_stream)

    _TGV_CUTE_EXT_COMPILE_CACHE[key] = compiled
    return compiled


# =====================================================================
# Tensor adaptation: PyTorch (M, N) -> cute view via the A↔B swap. Now
# auto-detects each tensor's stride-1 dim so callers can pass either
# K-major or MN-major a/b (and either M-contig or N-contig out).
# =====================================================================
def _to_cute_swap(
    a_pt: torch.Tensor,  # (..., M, K)
    b_pt: torch.Tensor,  # (..., K, N)
    out_pt: torch.Tensor,  # (..., M, N)
):
    """Build the cute.Tensors fed to the TGV kernel and
    report the detected (a_leading, b_leading, c_leading) tuple so the
    caller can fetch the matching compiled binary.

    Internally swaps A↔B so the kernel writes its (kernel-M)-contiguous
    output into the N-axis of a row-major PyTorch tensor — same trick as
    the C++ runner's ``gemm_fn(b.t(), a.t(), …)``. After swap, the
    kernel's M = PyTorch N and the kernel's N = PyTorch M.

    The kernel itself (`make_trivial_tiled_mma` + `mma_major_mode()`)
    adapts to whichever axis is the stride-1 one in each operand, so we
    just need to make sure ``mark_layout_dynamic`` pins the right dim.
    """
    # Lift to 3D batched shape with a leading L=1 axis when called from mm.
    if a_pt.dim() == 2:
        a_pt = a_pt.unsqueeze(0)
        b_pt = b_pt.unsqueeze(0)
        out_pt = out_pt.unsqueeze(0)

    # Swap: feed b.transpose(-2,-1) as A_ce (shape (L, N_pt, K)),
    # a.transpose(-2,-1) as B_ce (shape (L, K, M_pt)), and
    # out.transpose(-2,-1) as C_ce (shape (L, N_pt, M_pt)).
    a_swap = b_pt.transpose(-2, -1)
    b_swap = a_pt.transpose(-2, -1)
    c_swap = out_pt.transpose(-2, -1)

    a_leading = _detect_leading_dim(a_swap)
    b_leading = _detect_leading_dim(b_swap)
    c_leading = _detect_leading_dim(c_swap)

    a_ = from_dlpack(a_swap, assumed_align=32).mark_layout_dynamic(
        leading_dim=a_leading
    )
    b_ = from_dlpack(b_swap, assumed_align=32).mark_layout_dynamic(
        leading_dim=b_leading
    )
    c_ = from_dlpack(c_swap, assumed_align=32).mark_layout_dynamic(
        leading_dim=c_leading
    )

    layout = (a_leading, b_leading, c_leading)
    return a_, b_, c_, layout


def _resolve_tactic(tactic: int) -> Tuple[int, int, int, bool]:
    """Return (cta_m, cta_n, num_ab_stage, use_2cta) for the given tactic id."""
    if tactic < 0:
        tactic = _TGV_CUTE_EXT_DEFAULT_TACTIC
    if tactic >= len(_TGV_CUTE_EXT_TACTIC_CONFIGS):
        raise ValueError(
            f"TGV cute_ext tactic {tactic} out of range [0, {len(_TGV_CUTE_EXT_TACTIC_CONFIGS)})."
        )
    return _TGV_CUTE_EXT_TACTIC_CONFIGS[tactic]


def _run_tgv(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    pdl: bool,
    tactic: int,
) -> torch.Tensor:
    """Dispatch one TGV GEMM: a (M, K) @ b (K, N) -> out (M, N)."""
    cta_m, cta_n, num_ab_stage, use_2cta = _resolve_tactic(tactic)

    a_, b_, c_, (a_leading, b_leading, c_leading) = _to_cute_swap(
        a,
        b,
        out,
    )

    compiled = _get_compiled_cute_ext_kernel(
        dtype=a.dtype,
        c_dtype=out.dtype,
        cta_m=cta_m,
        cta_n=cta_n,
        cta_k=_TGV_CUTE_EXT_CTA_K,
        num_ab_stage=num_ab_stage,
        use_2cta=use_2cta,
        use_pdl=bool(pdl),
        a_leading=a_leading,
        b_leading=b_leading,
        c_leading=c_leading,
    )

    stream = cuda.CUstream(torch.cuda.current_stream(a.device).cuda_stream)

    compiled(a_, b_, c_, stream)
    return out


# (tactic, cta_m, cta_n, num_mma_ctas), smallest cta_n first.
_TGV_TACTIC_LADDER = (
    (18, 64, 16, 2),
    (21, 64, 32, 2),
    (23, 64, 64, 2),
    (24, 64, 128, 2),
    (12, 128, 16, 1),
    (26, 128, 32, 2),
    (27, 128, 64, 2),
)


def _grid_ctas(m: int, n: int, cta_m: int, cta_n: int, num_mma_ctas: int) -> int:
    # Kernel axes are swapped: kernel-M = pytorch N, kernel-N = pytorch M.
    cluster_m = cta_m * num_mma_ctas
    return num_mma_ctas * -(n // -cluster_m) * -(m // -cta_n)


def tgv_bf16_gemm(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Pick the first SGLang tactic fitting one wave on this GPU."""
    m, n = x.shape[0], weight.shape[0]
    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    best, best_ctas = None, None
    for tactic, cta_m, cta_n, num_mma_ctas in _TGV_TACTIC_LADDER:
        ctas = _grid_ctas(m, n, cta_m, cta_n, num_mma_ctas)
        if ctas <= num_sms:
            best = tactic
            break
        if best_ctas is None or ctas < best_ctas:
            best, best_ctas = tactic, ctas
    return _run_tgv(x, weight.T, out, pdl=True, tactic=best)


# GC MoE K3 test inputs, independent references, and accuracy checks.

DEFAULT_TOKENS_PER_RANK = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def arguments():
    p = argparse.ArgumentParser(
        description="Kimi K3 accuracy and workspace replay checks"
    )
    for name, default in [
        ("hidden", 3584),
        ("intermediate-hidden", 3072),
        ("num-experts", 896),
        ("num-topk", 16),
        ("shared-hidden", 7168),
        ("shared-intermediate-hidden", 6144),
        ("num-shared-experts", 2),
        ("num-max-tokens-per-rank", 256),
        ("fast-math", 1),
        ("gc-replays", 5),
    ]:
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--baseline", choices=tuple(BASELINE_NAMES), default="sglang")
    p.add_argument("--activation", choices=("situ",), default="situ")
    p.add_argument("--activation-beta", type=float, default=4.0)
    p.add_argument("--activation-linear-beta", type=float, default=25.0)
    p.add_argument("--activation-clamp", type=float, default=None)
    p.add_argument("--routed-scaling-factor", type=float, default=1.0)
    p.add_argument(
        "--case-category",
        choices=["default", "boundary", "asymmetric"],
        default="default",
    )
    p.add_argument(
        "--num-tokens-per-rank-list",
        default=",".join(map(str, DEFAULT_TOKENS_PER_RANK)),
    )
    p.add_argument("--per-rank-tokens-list", default="")
    p.add_argument("--output-json", default="results/_k3_results.json")
    p.add_argument("--cosine-threshold", type=float, default=0.99)
    p.add_argument("--cuda-graph-max-tokens-per-rank", type=int, default=256)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--inner-iters", type=int, default=20)
    p.add_argument("--inter-iter-sleep-s", type=float, default=0.01)
    a = p.parse_args()
    a.baseline_key = f"{a.baseline}_mega_moe_k3"
    if not 0.99 <= a.cosine_threshold <= 1:
        p.error("K3 cosine threshold must be >= .99")
    return a


def initialize(a):
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    d = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=d, timeout=timedelta(minutes=20))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        raise RuntimeError("SM100/103 hardware required")
    if a.num_experts % dist.get_world_size():
        raise ValueError("experts must divide world size")
    a.rank = dist.get_rank()
    a.world = dist.get_world_size()
    a.device = d
    a.vectors = (
        [list(map(int, v.split(","))) for v in a.per_rank_tokens_list.split(";") if v]
        if a.per_rank_tokens_list
        else [[t] * a.world for t in map(int, a.num_tokens_per_rank_list.split(","))]
    )
    if any(
        len(v) != a.world or min(v) < 0 or max(v) > a.num_max_tokens_per_rank
        for v in a.vectors
    ):
        raise ValueError("invalid per-rank token matrix/capacity")
    return d


def build_weights(a):
    e = a.num_experts // a.world
    h = a.hidden
    i = a.intermediate_hidden
    d = a.device
    q1 = torch.empty((e, 2 * i, h // 2), dtype=torch.int8, device=d)
    q2 = torch.empty((e, h, i // 2), dtype=torch.int8, device=d)
    s1 = torch.empty((e, 2 * i, h // 32), dtype=torch.float32, device=d)
    s2 = torch.empty((e, h, i // 32), dtype=torch.float32, device=d)
    for le in range(e):
        g = torch.Generator(device=d).manual_seed(7 + (a.rank * e + le) * 104729)
        w1 = torch.randn((2 * i, h), dtype=torch.bfloat16, device=d, generator=g) * 0.05
        w2 = torch.randn((h, i), dtype=torch.bfloat16, device=d, generator=g) * 0.05
        q1[le], s1[le] = per_token_cast_to_fp4(w1, use_ue8m0=True, gran_k=32)
        q2[le], s2[le] = per_token_cast_to_fp4(w2, use_ue8m0=True, gran_k=32)
    # Independent reference retains the original payload and unpermuted float scales.
    ref = (q1, q2, s1, s2)
    pairs = (
        (q1, deep_gemm.transform_sf_into_required_layout(s1, 2 * i, h, (1, 32), e)),
        (q2, deep_gemm.transform_sf_into_required_layout(s2, h, i, (1, 32), e)),
    )
    g = torch.Generator(device=d).manual_seed(11)
    sw = (
        torch.randn(
            (2 * a.shared_intermediate_hidden, a.shared_hidden),
            dtype=torch.bfloat16,
            device=d,
            generator=g,
        )
        * 0.05,
        torch.randn(
            (a.shared_hidden, a.shared_intermediate_hidden),
            dtype=torch.bfloat16,
            device=d,
            generator=g,
        )
        * 0.05,
    )
    return pairs, ref, sw


def build_tokens(a, t, variant=0):
    g = torch.Generator(device=a.device).manual_seed(42 + a.rank * 7919 + variant * 12347)
    return torch.randn((t, a.shared_hidden), dtype=torch.bfloat16, device=a.device, generator=g) * 0.5


def decode_fp8(q, sf):
    if not q.shape[0]:
        return q.float()
    scales = (sf.contiguous().view(torch.uint8).to(torch.int32) << 23).view(
        torch.float32
    )
    return q.float() * scales.repeat_interleave(32, dim=-1)


def decode_fp4(q, s):
    b = q.to(torch.int16) & 255
    codes = torch.stack((b & 15, b >> 4), dim=-1).flatten(-2)
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=q.device)
    v = table[(codes & 7).long()]
    v = torch.where((codes & 8) != 0, -v, v)
    return v * s.repeat_interleave(32, dim=-1)


def activation(a, z):
    gate, up = z.to(torch.bfloat16).float().chunk(2, dim=-1)
    if a.activation_clamp is not None:
        bound = torch.tensor(
            a.activation_clamp, device=z.device, dtype=torch.bfloat16
        ).float()
        gate = gate.clamp(max=bound)
        up = up.clamp(-bound, bound)
    sigmoid = 1.0 / (1.0 + torch.exp(-gate))
    gate = sigmoid * (torch.tanh(gate / a.activation_beta) * a.activation_beta)
    if a.activation_linear_beta is not None:
        up = torch.tanh(up / a.activation_linear_beta) * a.activation_linear_beta
    return gate * up


def quantize_mid_reference(mid, *, sglang=False):
    """Independent UE8M0 scale and RN E4M3 conversion for 32-value groups."""
    if not mid.shape[0]:
        return mid
    shape = mid.shape
    tiles = mid.reshape(shape[0], -1, 32)
    amax = tiles.abs().amax(-1).contiguous()
    if sglang:
        # SGLang's W2 input uses ceil(log2(amax / 448)).
        scaled = amax * (1.0 / 448.0)
        exp = ((scaled.view(torch.int32) + 0x7FFFFF) >> 23).clamp_min(1)
        scale = (exp << 23).view(torch.float32)
        q = (tiles / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    else:
        exp = (
            (
                (amax.view(torch.int32).to(torch.int64) + 0x7FFFFF - 0x600000) >> 23
            ).clamp_min(113)
            - 8
        ).to(torch.int32)
        scale = (exp << 23).view(torch.float32)
        inv = ((254 - exp) << 23).view(torch.float32)
        q = (tiles * inv.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return (q.float() * scale.unsqueeze(-1)).reshape(shape)


def ordered_fma(slots, weights):
    # BF16 x FP32 product is exact in FP64. Adding bounded FP32 acc in FP64
    # and casting once realizes FP32 fmaf for these finite test inputs.
    out = torch.zeros(
        slots.shape[0:1] + slots.shape[2:], dtype=torch.float32, device=slots.device
    )
    for j in range(slots.shape[1]):
        out = (
            slots[:, j].double() * weights[:, j, None].double() + out.double()
        ).float()
    return out.to(torch.bfloat16)


def reference(a, data, ref_weights, sw, vector):
    q, sf, idx, w, sx = data
    t = q.shape[0]
    m = max(1, max(vector))
    xpad = torch.zeros((m, a.hidden), device=a.device, dtype=torch.float32)
    xpad[:t] = decode_fp8(q, sf)
    ip = torch.full((m, a.num_topk), -1, device=a.device, dtype=torch.int64)
    ip[:t] = idx
    wpad = torch.zeros((m, a.num_topk), device=a.device, dtype=torch.float32)
    wpad[:t] = w
    xs = [torch.empty_like(xpad) for _ in range(a.world)]
    ids = [torch.empty_like(ip) for _ in range(a.world)]
    weights = [torch.empty_like(wpad) for _ in range(a.world)]
    dist.all_gather(xs, xpad)
    dist.all_gather(ids, ip)
    dist.all_gather(weights, wpad)
    x = torch.stack(xs).flatten(0, 1)
    routes = torch.stack(ids).flatten(0, 1)
    route_weights = torch.stack(weights).flatten(0, 1)
    slots = torch.zeros(
        (2, a.world * m, a.num_topk, a.hidden), device=a.device, dtype=torch.bfloat16
    )
    q1, q2, s1, s2 = ref_weights
    e = a.num_experts // a.world
    for le in range(e):
        rows, js = (routes == (a.rank * e + le)).nonzero(as_tuple=True)
        if not rows.numel():
            continue
        w1 = decode_fp4(q1[le], s1[le])
        w2 = decode_fp4(q2[le], s2[le])
        mid = activation(a, x[rows] @ w1.T)
        slots[0, rows, js] = (quantize_mid_reference(mid) @ w2.T).to(torch.bfloat16)
        # MegaMoE applies the top-k weight before quantizing the W2 input.
        current_weights = route_weights[rows, js, None]
        if a.baseline == "vllm":
            current_weights = current_weights * a.routed_scaling_factor
        weighted = quantize_mid_reference(mid * current_weights, sglang=True)
        slots[1, rows, js] = (weighted @ w2.T).to(torch.bfloat16)
    dist.all_reduce(slots)
    local_slots = slots[:, a.rank * m : a.rank * m + t]
    y = ordered_fma(local_slots[0], w)
    mega_y = ordered_fma(local_slots[1], torch.ones_like(w))
    if t:
        sy = (
            activation(a, sx[:t].float() @ sw[0].float().T)
            .to(torch.bfloat16)
            .float()
            @ sw[1].float().T
        ).to(torch.bfloat16)
    else:
        sy = torch.empty(
            (0, a.shared_hidden), device=a.device, dtype=torch.bfloat16
        )
    return (y, sy), mega_y


def cosine(x, y):
    if x.numel() == 0:
        return float(y.numel() == 0)
    xf, yf = x.float().flatten(), y.float().flatten()
    den = xf.norm() * yf.norm()
    return float((xf @ yf / den).item()) if den.item() else float(torch.equal(x, y))


def relative_l2(x, y):
    if x.numel() == 0:
        return 0.0 if y.numel() == 0 else float("inf")
    xf, yf = x.double(), y.double()
    den = yf.norm().item()
    err = (xf - yf).norm().item()
    return err / den if den else (0.0 if err == 0 else float("inf"))


def max_abs(x, y):
    return float((x.float() - y.float()).abs().max().item()) if x.numel() else 0.0


class Operators:
    def __init__(self, a, pairs, sw):
        self.a = a
        g = torch.Generator(device=a.device).manual_seed(1727)
        self.norm_weight = (1 + 0.1 * torch.randn(
            a.hidden, device=a.device, generator=g)).to(torch.bfloat16)
        self.up_weight = (torch.randn(
            (a.shared_hidden, a.hidden), device=a.device, generator=g) *
            a.hidden ** -0.5).to(torch.bfloat16)
        self.norm_eps = 1e-6
        self.front_weight = (torch.randn((a.num_experts + a.hidden, a.shared_hidden),
            dtype=torch.bfloat16, device=a.device, generator=g) * a.shared_hidden ** -0.5)
        self.router_bias = torch.randn(a.num_experts, device=a.device, generator=g) * 0.05
        self.moe_options = dict(
            recipe=(1, 1, 32), activation=a.activation,
            activation_clamp=a.activation_clamp, activation_beta=a.activation_beta,
            routed_scaling_factor=a.routed_scaling_factor,
            activation_linear_beta=a.activation_linear_beta, fast_math=bool(a.fast_math),
        )
        baseline_cls = SGLangMegaMoEK3 if a.baseline == "sglang" else VLLMMegaMoEK3
        self.baseline = baseline_cls(a, pairs, sw, self.front_weight, self.router_bias)
        self.weights = gc_moe.transform_weights_for_gc_moe_k3(*pairs, activation=a.activation)
        self.shared_weights = gc_moe.transform_weights_for_gc_moe_k3(*sw, activation=a.activation)
        self.buffers = [
            self.baseline.buffer,
            gc_moe.get_symm_buffer_for_gc_moe_k3(
                dist.group.WORLD,
                a.num_experts,
                a.num_max_tokens_per_rank,
                a.num_topk,
                a.hidden,
                a.intermediate_hidden,
                num_shared_experts=a.num_shared_experts,
                shared_hidden=a.shared_hidden,
                shared_intermediate_hidden=a.shared_intermediate_hidden,
                activation=a.activation,
            ),
        ]
        self.alias = gc_moe.GCK3SymmBuffer(
            dist.group.WORLD,
            a.num_experts,
            a.num_max_tokens_per_rank,
            a.num_topk,
            a.hidden,
            a.intermediate_hidden,
            num_shared_experts=a.num_shared_experts,
            shared_hidden=a.shared_hidden,
            shared_intermediate_hidden=a.shared_intermediate_hidden,
            activation=a.activation,
            base=self.buffers[1],
        )
        self.alias_calls = 0
        if a.rank == 0:
            print(
                "Symmetric buffers requested=",
                a.num_max_tokens_per_rank,
                "aligned=",
                [b.num_max_tokens_per_rank for b in self.buffers],
                flush=True,
            )

    def output(self, j, t):
        a = self.a
        final = torch.empty((t, a.shared_hidden), device=a.device, dtype=torch.bfloat16)
        if j == 1:
            return final
        routed = final.new_empty((t, a.hidden))
        shared = final.new_empty((max(1, t), a.shared_hidden)) if a.baseline == "sglang" else final
        return routed, shared, final

    def reference_input(self, states):
        a = self.a
        # Independent split gate/projection and explicit BF16 latent rounding.
        logits = states.float() @ self.front_weight[:a.num_experts].float().T
        latent = (states.float() @ self.front_weight[a.num_experts:].float().T).to(torch.bfloat16)
        ids, weights = _reference_routes(logits, self.router_bias, a.num_topk)
        q, sf = _reference_quantization(latent)
        return q, sf, ids, weights, states

    def prefix(self, t):
        g = torch.Generator(device=self.a.device).manual_seed(1973 + self.a.rank)
        return (torch.randn(
            (t, self.a.shared_hidden), device=self.a.device, generator=g
        ) * 0.125).to(torch.bfloat16)

    def call(self, j, out, data, prefix):
        if j == 0:
            self.baseline.call(
                out, data, projection=(self.norm_weight, self.up_weight, prefix, self.norm_eps))
            return
        self.call_gc(out, data, prefix, self.norm_weight, self.up_weight)

    def call_gc(self, out, states, prefix, norm_weight, up_proj_weight, *,
                front_weight=None, router_bias=None, valid_token_count=None):
        self.alias_calls += 1
        buffer = self.alias if self.alias_calls % 2 else self.buffers[1]
        gc_moe.fp8_fp4_gc_moe_k3(
            out, *self.weights, buffer, hidden_states=states,
            front_weight=self.front_weight if front_weight is None else front_weight,
            router_bias=self.router_bias if router_bias is None else router_bias,
            valid_token_count=valid_token_count,
            shared_l1_weights=self.shared_weights[0], shared_l2_weights=self.shared_weights[1],
            norm_weight=norm_weight, up_proj_weight=up_proj_weight, norm_eps=self.norm_eps,
            prefix_sum=prefix, **self.moe_options,
        )

    def warm_graph(self, j, out, data, prefix):
        if j == 0:
            t = out[0].shape[0]
            if t and self.a.baseline == "sglang":
                self.baseline.shared.prepare_graph(data, out[1][:t])
                self.baseline.project_routed(
                    out[0], self.norm_weight, self.up_weight, self.norm_eps, True)
            self.baseline.call(
                out, data, graph_path=True,
                projection=(self.norm_weight, self.up_weight, prefix, self.norm_eps))
        else:
            self.call(j, out, data, prefix)

    def destroy(self):
        self.alias.destroy()
        for b in self.buffers:
            b.destroy()


def evidence(a, ops):
    routed = "deep_gemm.fp8_fp4_mega_moe"
    if a.baseline == "sglang":
        baseline = dict(
            distribution="sgl-deep-gemm==0.2.8",
            routed=routed,
            shared="bf16_gate_up_situ_down_on_side_stream",
            max_sms=ops.baseline.max_sms,
            pdl=ops.baseline.pdl,
            front="merged FP32 cuBLAS gate/down projection, Triton TopK and cast",
            pre_dispatch="Triton K32 UE8M0/FP8 and routing stores",
        )
        tail = dict(
            sglang_normalization="FlashInfer BF16 RMSNorm",
            sglang_add3="Triton, 32-byte vectors, packed BF16 adds and PDL",
            sglang_join="after_latent_rmsnorm_and_up_projection",
        )
    else:
        baseline = dict(
            distribution="vllm-project/DeepGEMM",
            revision=VLLM_DEEPGEMM_REVISION,
            routed=routed,
            shared="BF16 gate/up GEMM, Triton SiTU, BF16 down GEMM on current stream",
            gate=("vLLM ll_bf16_gemm CuTeDSL for M<=16, cuBLAS otherwise"
                  if ops.baseline.gate_sources is not None else "cuBLAS at all sizes"),
            front="FP32 gate and Triton Top16, BF16 torch.mm down on auxiliary stream",
            pre_dispatch="Triton K32 UE8M0/FP8, strided packed scales and routing stores",
        )
        tail = dict(
            vllm_normalization="Triton RMSNorm with vLLM compiled native IR rounding",
            vllm_projection="BF16 addmm_ with shared as beta=1 input, separate prefix add",
        )
    return dict(
        model="kimi-k3",
        world_size=a.world,
        case_category=a.case_category,
        requested_capacity=a.num_max_tokens_per_rank,
        aligned_capacity=[b.num_max_tokens_per_rank for b in ops.buffers],
        operators=[BASELINE_NAMES[a.baseline], "GC MoE K3"],
        modules={a.baseline_key: deep_gemm.__file__, "gc_moe_k3": gc_moe.__file__},
        output=dict(norm_eps=ops.norm_eps, up_weight_shape=tuple(ops.up_weight.shape), **tail),
        **{a.baseline_key: baseline},
        config={
            k: getattr(a, k)
            for k in (
                "hidden",
                "intermediate_hidden",
                "num_experts",
                "num_topk",
                "shared_hidden",
                "shared_intermediate_hidden",
                "num_shared_experts",
                "activation",
                "activation_beta",
                "activation_linear_beta",
                "fast_math",
            "routed_scaling_factor",
            )
        },
        run_id=os.environ.get("GC_K3_RUN_ID", "interactive"),
        source_manifest_sha256=os.environ.get(
            "GC_K3_SOURCE_MANIFEST_SHA256", "development"
        ),
    )


def write_result(a, obj):
    if a.rank == 0:
        p = Path(a.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2) + "\n")


def check_expert_weight_aliases(ops, states, prefix):
    """Output must not overwrite a weight, even in a later strided SF column."""
    from unittest.mock import patch
    from gc_moe.apis.gc_moe_k3 import _sf_overlaps
    from gc_moe.jit_kernels.impls import sm100_fp8_fp4_gc_moe_k3 as prepared

    width = states.shape[1]
    shared_w2 = ops.shared_weights[1].view(-1)
    scales = ops.weights[0][1]
    experts, rows, cols = scales.shape
    pitch = rows + width
    backing = torch.empty(experts * cols * pitch, dtype=torch.int32, device=states.device)
    padded = backing.as_strided(scales.shape, (cols * pitch, 1, pitch))
    padding = backing[rows:rows + (width + 1) // 2].view(torch.bfloat16)[:width]
    # The final column starts beyond the dense logical nbytes when enough
    # padding is present. Only its row data belongs to the scale tensor.
    last = (experts * cols - 1) * pitch
    sf_out = backing[last:].view(torch.bfloat16)[:width]
    if _sf_overlaps(padded, padding) or not _sf_overlaps(padded, sf_out):
        raise AssertionError("incorrect overlap detection for padded expert scales")
    with patch.object(prepared, "launch_gc_moe_k3") as launch:
        original = ops.weights
        try:
            ops.weights = ((original[0][0], padded), original[1])
            for target in (shared_w2[:width], sf_out):
                try:
                    ops.call_gc(target.view(1, width), states[:1], prefix[:1],
                                ops.norm_weight, ops.up_weight)
                except ValueError as error:
                    if "expert weights" not in str(error):
                        raise
                else:
                    raise AssertionError("final output aliases a live expert weight")
            launch.assert_not_called()
            ops.call_gc(padding.view(1, width), states[:1], prefix[:1],
                        ops.norm_weight, ops.up_weight)
            launch.assert_called_once()
        finally:
            ops.weights = original


def check_gc_storage_lifetime(a, ops, data, prefix, expected):
    """Evict live graph bindings, clone/replay, then release temporary inputs."""
    from gc_moe.runtime import gc_moe_k3 as runtime
    from gc_moe.jit_kernels.impls import sm100_fp8_fp4_gc_moe_k3 as prepared

    context = runtime.get_context(num_ranks=a.world)
    start_bytes = torch.cuda.memory_allocated(a.device)
    t = data.shape[0]
    # Warm binaries and fill the bounded cache. The new pointer set below
    # deliberately creates/evicts a child graph during stream capture.
    for _ in range(runtime._GRAPH_CACHE_CAPACITY):
        packet = data.clone()
        out = ops.output(1, t)
        ops.call(1, out, packet, prefix)
    del packet, out
    torch.cuda.synchronize()
    packet = data.clone()
    out = ops.output(1, t).fill_(float("nan"))
    captured_prefix = prefix.clone()
    captured_norm = ops.norm_weight.clone()
    captured_up = ops.up_weight.clone()
    captured_front = ops.front_weight.clone()
    captured_bias = ops.router_bias.clone()
    captured_live_tokens = torch.tensor(t, dtype=torch.int32, device=data.device)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        ops.call_gc(out, packet, captured_prefix, captured_norm, captured_up,
                    front_weight=captured_front, router_bias=captured_bias,
                    valid_token_count=captured_live_tokens)
    graph.instantiate()
    # Exercise CUDA's child -> parent -> clone -> executable ownership.
    clone = runtime._ok(runtime.cu.cuGraphClone(graph.raw_cuda_graph()))
    executable = runtime._ok(runtime.cu.cuGraphInstantiate(clone, 0))
    captured = {
        "output": weakref.ref(out),
        "hidden_states": weakref.ref(packet),
        "front_weight": weakref.ref(captured_front),
        "router_bias": weakref.ref(captured_bias),
        "valid_token_count": weakref.ref(captured_live_tokens),
        "prefix": weakref.ref(captured_prefix),
        "rmsnorm_weight": weakref.ref(captured_norm),
        "up_proj_weight": weakref.ref(captured_up),
    }
    graph.reset()
    del graph, out, packet, captured_prefix, captured_norm, captured_up
    del captured_front, captured_bias, captured_live_tokens
    context._clear_graph_cache()
    gc.collect()
    retained = {name: ref() is not None for name, ref in captured.items()}
    clone_retained_inputs = all(retained.values())
    # Every rank must agree before launching a graph with peer accesses.
    retained_by_rank = [None] * a.world
    dist.all_gather_object(retained_by_rank, clone_retained_inputs)

    temporary = []
    calls = runtime._GRAPH_CACHE_CAPACITY * 4
    for _ in range(calls):
        packet = data.clone()
        out = ops.output(1, t)
        temporary.extend([weakref.ref(out), weakref.ref(packet)])
        ops.call(1, out, packet, prefix)
    del out, packet
    torch.cuda.synchronize()
    bounded = (
        len(context.graphs) <= runtime._GRAPH_CACHE_CAPACITY
        and len(prepared._CACHE) <= prepared._PREPARED_CACHE_CAPACITY
    )
    context._clear_graph_cache()

    def wait_released(references):
        deadline = time.monotonic() + 5
        while True:
            runtime._collect_retired_storage()
            gc.collect()
            live = sum(ref() is not None for ref in references)
            if live == 0 or time.monotonic() >= deadline:
                return live
            time.sleep(0.005)

    temporary_live = wait_released(temporary)
    # The original child was evicted; only the independently cloned graph
    # now keeps these allocations alive. Destroy host handles immediately
    # after launch to exercise the pending-execution reference as well.
    replay = all(retained_by_rank)
    if replay:
        runtime._ok(runtime.cu.cuGraphLaunch(
            executable, torch.cuda.current_stream(a.device).cuda_stream
        ))
    runtime._ok(runtime.cu.cuGraphExecDestroy(executable))
    runtime._ok(runtime.cu.cuGraphDestroy(clone))
    torch.cuda.synchronize()
    final = captured["output"]()
    precision = (
        replay and final is not None and cosine(final, expected) >= a.cosine_threshold
        and relative_l2(final, expected) <= 0.02
    )
    del final
    captured_live = wait_released(captured.values())
    end_bytes = torch.cuda.memory_allocated(a.device)
    return dict(
        passed=bool(
            clone_retained_inputs
            and precision
            and bounded
            and not temporary_live
            and not captured_live
            and end_bytes <= start_bytes + 1024 * 1024
        ),
        primary_ctx_roles=bool(context.primary_ctx_roles),
        dispatch_sms=int(context.dispatch_sms),
        compute_sms=int(context.compute_sms),
        eager_calls=calls,
        clone_retained_inputs=clone_retained_inputs,
        clone_retained_tensors=retained,
        clone_replay_precision=bool(precision),
        caches_bounded=bounded,
        temporary_tensors_remaining=temporary_live,
        captured_tensors_remaining=captured_live,
        allocated_bytes_before=start_bytes,
        allocated_bytes_after=end_bytes,
    )


def check_front_workspace(ops, states, expected, prefix, expected_output):
    """Check the front stages independently of the shared-heavy final output."""
    a = ops.a
    t = states.shape[0]
    q, sf, ids, weights, _ = expected
    buffer = ops.buffers[1]
    quant_error = relative_l2(decode_fp8(buffer.x[:t], buffer.x_sf[:t]), decode_fp8(q, sf))
    matches = bool((buffer.topk_idx[:t] == ids).all())
    weight_error = max_abs(buffer.topk_weights[:t], weights)

    # Change the live count, inputs and bias at fixed addresses on one graph.
    # The full replay must populate the same tail that the partial one clears.
    live = t // 2 + 1
    staged_states = states.clone()
    staged_bias = ops.router_bias.clone()
    count = torch.tensor(t, device=a.device, dtype=torch.int32)
    out = ops.output(1, t)

    def call():
        ops.call_gc(out, staged_states, prefix, ops.norm_weight, ops.up_weight,
                    router_bias=staged_bias, valid_token_count=count)

    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    snapshots = []
    for partial in (False, True, False):
        if partial:
            staged_states.zero_()
            staged_bias.zero_()
        else:
            staged_states.copy_(states)
            staged_bias.copy_(ops.router_bias)
        count.fill_(live if partial else t)
        out.fill_(float("nan"))
        graph.replay()
        snapshots.append((out.clone(), buffer.topk_idx[:t].clone(), buffer.topk_weights[:t].clone(),
                          buffer.x[:t].view(torch.uint8).clone()))
    torch.cuda.synchronize()
    graph.reset()

    full_before, partial, full_after = snapshots
    partial_out, partial_ids, partial_weights, partial_fp8 = partial
    expected_ids = torch.arange(a.num_topk, device=a.device)
    exact_ties = bool((partial_ids[:live] == expected_ids).all()
                      and (partial_weights[:live] == 1.0 / a.num_topk).all())
    masked_tail = bool((partial_ids[live:] == -1).all()
                       and (partial_weights[live:] == 0).all())
    zero_fp8 = bool((partial_fp8[:live] == 0).all())
    partial_live_output = torch.equal(partial_out[:live], prefix[:live])
    full_outputs = all(
        torch.equal(full_ids, ids)
        and max_abs(full_weights, weights) < 1e-5
        and cosine(full_out, expected_output) >= a.cosine_threshold
        and relative_l2(full_out, expected_output) <= 0.02
        for full_out, full_ids, full_weights, _ in (full_before, full_after)
    )
    return dict(rank=a.rank, quantization_relative_l2=quant_error,
                routes_equal=matches, max_abs_weight_error=weight_error,
                exact_ties=exact_ties, masked_tail=masked_tail, zero_fp8=zero_fp8,
                graph_live_counts=[t, live, t], partial_live_output=partial_live_output,
                full_outputs=full_outputs,
                passed=quant_error < .005 and matches and weight_error < 1e-5
                and exact_ties and masked_tail and zero_fp8
                and partial_live_output and full_outputs)


def check_routed_warmup_replay(ops, states, expected_routes, prefix, expected_output,
                               block_m, midpoint_min_tokens):
    """Replay both in-ring routed warmup policies at fixed tensor addresses."""
    a = ops.a
    t = states.shape[0]
    live = min(t // 2 + 1, midpoint_min_tokens - 1)
    live_counts = (t, live, t)
    buffer = ops.buffers[1]
    ring_blocks = buffer.layout.ring_tokens // block_m

    # Keep routing dispersed in both replays. Count incoming routes across
    # all source ranks to verify that both live counts have in-ring work.
    received = torch.stack([
        torch.bincount(expected_routes[:n].flatten().long(), minlength=a.num_experts)
        for n in (t, live)
    ])
    dist.all_reduce(received)
    m_blocks = ((received + block_m - 1) // block_m).view(2, a.world, -1).sum(dim=-1)
    full_blocks, partial_blocks = m_blocks[:, a.rank].tolist()
    in_ring = 0 < full_blocks <= ring_blocks and 0 < partial_blocks <= ring_blocks

    count = torch.tensor(t, dtype=torch.int32, device=a.device)
    out = ops.output(1, t)

    def call():
        ops.call_gc(out, states, prefix, ops.norm_weight, ops.up_weight,
                    valid_token_count=count)

    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()

    routes_ok = []
    outputs_ok = []
    for n in live_counts:
        count.fill_(n)
        out.fill_(float("nan"))
        graph.replay()
        routes_ok.append(
            torch.equal(buffer.topk_idx[:n], expected_routes[:n])
            and bool((buffer.topk_idx[n:t] == -1).all())
            and bool((buffer.topk_weights[n:t] == 0).all())
        )
        outputs_ok.append(
            cosine(out[:n], expected_output[:n]) >= a.cosine_threshold
            and relative_l2(out[:n], expected_output[:n]) <= 0.02
        )
    torch.cuda.synchronize()
    graph.reset()
    return dict(rank=a.rank, graph_live_counts=live_counts,
                midpoint_min_tokens=midpoint_min_tokens, ring_blocks=ring_blocks,
                routed_m_blocks=[full_blocks, partial_blocks, full_blocks],
                routes_ok=routes_ok, outputs_ok=outputs_ok,
                passed=in_ring and all(routes_ok) and all(outputs_ok))


def final_reference(ops, routed, shared, prefix, *, fused_beta, routed_scale=None,
                    round_norm_before_weight=False):
    scale = ops.a.routed_scaling_factor if routed_scale is None else routed_scale
    x = (routed * scale).float()
    inv = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + ops.norm_eps)
    normalized = x * inv
    if round_norm_before_weight:
        normalized = normalized.to(ops.norm_weight.dtype).float()
    normalized = (normalized * ops.norm_weight.float()).to(torch.bfloat16)
    projected = normalized.float() @ ops.up_weight.float().T
    if fused_beta:
        out = (projected + shared.float()).to(torch.bfloat16)
    else:
        out = projected.to(torch.bfloat16) + shared
    return out if prefix is None else out + prefix


def check_outputs(ops, data, prefix, expected):
    t = data[0].shape[0]
    samples = [[], []]
    for j in (0, 1):
        out = ops.output(j, t)
        final = out[2] if j == 0 else out
        graphs = []
        for k in (0, 1):
            p = prefix if k else None
            ops.warm_graph(j, out, data[k], p)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                ops.call(j, out, data[k], p)
            graphs.append(graph)
        for replay in range(max(2, ops.a.gc_replays)):
            k = replay % 2
            final.fill_(float("nan"))
            graphs[k].replay()
            samples[j].append((k, final.clone()))
        for k in (0, 1, 0, 1):
            final.fill_(float("nan"))
            ops.call(j, out, data[k], prefix if k else None)
            samples[j].append((k, final.clone()))
        graphs.clear()
    torch.cuda.synchronize()

    metrics = []
    for j, series in enumerate(samples):
        finite = all(bool(torch.isfinite(out).all()) for _, out in series)
        metrics.extend((
            min(cosine(out, expected[j][k]) for k, out in series) if finite else -1.0,
            max(relative_l2(out, expected[j][k]) for k, out in series) if finite else -1.0,
            max(max_abs(out, expected[j][k]) for k, out in series) if finite else -1.0,
        ))
    cross = min(cosine(base, gc) for (_, base), (_, gc) in zip(*samples))
    return (*metrics, cross)


def run():
    a = arguments()
    initialize(a)
    pairs, weights, shared_weights = build_weights(a)
    ops = Operators(a, pairs, shared_weights)
    results = []
    lifetime = []
    front_checks = []
    warmup_replay_checks = []
    invalid_count = torch.zeros((), device=a.device, dtype=torch.int32)
    for vector in a.vectors:
        t = vector[a.rank]
        if a.baseline == "sglang":
            ops.baseline.global_num_tokens = sum(vector)
        data = [build_tokens(a, t, k) for k in (0, 1)]
        staged = [ops.reference_input(x) for x in data]
        r = [reference(a, x, weights, shared_weights, vector) for x in staged]
        prefix = ops.prefix(t)
        expected = [
            [final_reference(ops, mega, shared, prefix if k else None,
                             fused_beta=a.baseline == "vllm",
                             routed_scale=1.0 if a.baseline == "vllm" else None,
                             round_norm_before_weight=a.baseline == "vllm")
             for k, ((_, shared), mega) in enumerate(r)],
            [final_reference(ops, routed, shared, prefix if k else None, fused_beta=k == 0)
             for k, ((routed, shared), _) in enumerate(r)],
        ]
        values = check_outputs(ops, data, prefix, expected)

        symmetric_default = a.case_category == "default" and len(set(vector)) == 1
        if symmetric_default and not lifetime and t >= 32:
            lifetime = [None] * a.world
            front_checks = [None] * a.world
            check_expert_weight_aliases(ops, data[1], prefix)
            front_check = check_front_workspace(ops, data[1], staged[1], prefix, expected[1][1])
            dist.all_gather_object(front_checks, front_check)
            local = check_gc_storage_lifetime(a, ops, data[1], prefix, expected[1][1])
            dist.all_gather_object(lifetime, local)

        if symmetric_default and not warmup_replay_checks and t:
            buffer = ops.buffers[1]
            block_m = gc_moe.get_block_m_for_gc_moe_k3(
                a.world, a.num_experts, buffer.num_max_tokens_per_rank, t, a.num_topk)
            midpoint_min_tokens = ceil_div((a.num_experts // a.world) * (block_m // 2), a.num_topk)
            if 1 < midpoint_min_tokens <= t:
                local = check_routed_warmup_replay(
                    ops, data[1], staged[1][2], prefix, expected[1][1], block_m, midpoint_min_tokens)
                warmup_replay_checks = [None] * a.world
                dist.all_gather_object(warmup_replay_checks, local)

        # A fully padded local bucket still invalidates every old route.
        out = ops.output(1, t)
        ops.call_gc(out, data[0], prefix, ops.norm_weight, ops.up_weight,
                    valid_token_count=invalid_count)
        invalid_ok = (
            bool((ops.buffers[1].topk_idx[:t] == -1).all())
            and bool((ops.buffers[1].topk_weights[:t] == 0).all())
        )
        metrics = torch.tensor((*values, int(invalid_ok)), device=a.device, dtype=torch.float64)
        ranks = [torch.empty_like(metrics) for _ in range(a.world)]
        dist.all_gather(ranks, metrics)
        per = [row.cpu().tolist() for row in ranks]
        ok = all(
            all(isfinite(v) for v in x) and min(x[0], x[3], x[6]) >= a.cosine_threshold
            and 0 <= x[1] <= 0.02 and 0 <= x[4] <= 0.02 and x[7]
            for x in per
        )
        entry = dict(
            num_tokens_per_rank=t, num_tokens_global=sum(vector), asym_vector=vector, passed=ok,
            **{f"{a.baseline_key}_cos_min": min(x[0] for x in per),
               f"{a.baseline_key}_relative_l2_max": max(x[1] for x in per),
               f"{a.baseline_key}_reference_max_abs": max(x[2] for x in per),
               f"gc_vs_{a.baseline_key}_cos_min": min(x[6] for x in per)},
            gc_moe_k3_cos_min=min(x[3] for x in per),
            gc_moe_k3_relative_l2_max=max(x[4] for x in per),
            gc_moe_k3_reference_max_abs=max(x[5] for x in per),
            invalid_routes_passed=all(x[7] for x in per),
            per_rank_checks=per,
        )
        results.append(entry)
        if a.rank == 0:
            status = "OK" if ok else "FAIL"
            print(f'tokens_per_rank={t:4d} total_tokens={sum(vector):5d} '
                  f'GC MoE K3 cos={entry["gc_moe_k3_cos_min"]:.6f} '
                  f'{BASELINE_NAMES[a.baseline]} cos={entry[f"{a.baseline_key}_cos_min"]:.6f} '
                  f'[{status}]', flush=True)

    passed = all(x["passed"]
                 for checks in (results, lifetime, front_checks, warmup_replay_checks)
                 for x in checks)
    report = evidence(a, ops)
    report.update(all_pass=passed, threshold=a.cosine_threshold, relative_l2_threshold=0.02,
                  lifecycle_checks=lifetime, front_checks=front_checks,
                  warmup_replay_checks=warmup_replay_checks, case_count=len(results),
                  rank_case_count=len(results) * a.world, results=results)
    write_result(a, report)
    if a.rank == 0:
        status = "ALL PASS" if passed else "SOME FAILED"
        print(f"{status} ({len(results)} cases, {len(results) * a.world} rank-cases)", flush=True)
    torch.cuda.synchronize()
    ops.destroy()
    dist.destroy_process_group()
    if not passed:
        raise AssertionError("K3 full-output accuracy or replay check failed")


if __name__ == "__main__":
    run()
