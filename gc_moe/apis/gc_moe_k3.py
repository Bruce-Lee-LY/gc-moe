"""Checked host contract for the complete K3 worker graph."""
from __future__ import annotations

from ctypes import c_float
from dataclasses import dataclass
from math import inf, isfinite
from typing import TYPE_CHECKING
import torch
from ..jit_kernels.heuristics.gc_moe_k3 import get_block_m_for_gc_moe_k3

if TYPE_CHECKING:
    from ..gc_k3 import GCK3SymmBuffer


@dataclass(frozen=True)
class ValidatedCall:
    tokens: int
    block_m: int
    hidden: int
    intermediate: int
    local_experts: int
    shared_experts: int
    shared_hidden: int
    shared_intermediate: int
    clamp: float
    beta: float
    linear_beta: float | None
    fast_math: bool


def _check_weight(w, shape, dtype, name):
    if (not isinstance(w, torch.Tensor) or w.dtype != dtype or not w.is_cuda or
            tuple(w.shape) != shape or not w.is_contiguous() or w.data_ptr() % 16):
        raise ValueError(f'{name} requires aligned contiguous CUDA {dtype} {shape}')


def _overlaps(a, b):
    return (a.numel() and b.numel() and
            a.data_ptr() < b.data_ptr() + b.nbytes and
            b.data_ptr() < a.data_ptr() + a.nbytes)


def _sf_overlaps(sf, dst):
    """Intersect MN-major scale rows with a contiguous destination, skipping column padding."""
    if not dst.numel():
        return False
    start = dst.data_ptr() - sf.data_ptr()
    stop = start + dst.nbytes
    pitch = sf.stride(-1) * sf.element_size()
    row_bytes = sf.size(-2) * sf.element_size()
    columns = sf.size(0) * sf.size(-1)
    # First column whose last byte is at or after the destination start.
    column = max(0, (start - row_bytes) // pitch + 1)
    return column < columns and column * pitch < stop


def _check_sf(sf, shape, name):
    if (not isinstance(sf, torch.Tensor) or sf.dtype != torch.int32 or
            tuple(sf.shape) != shape or not sf.is_cuda or sf.stride(-2) != 1 or
            sf.stride(-1) < shape[-2] or sf.stride(-1) % 4 or sf.data_ptr() % 16 or
            sf.stride(0) != sf.stride(-1)*shape[-1]):
        raise ValueError(f'{name} requires aligned MN-major packed int32 scales {shape}')


def _validate_call(out, t, l1_weights, l2_weights, sym_buffer, shared_x, shared_output,
                   shared_l1_weights, shared_l2_weights, recipe,
                   activation_clamp, activation_beta, activation_linear_beta, fast_math):
    if recipe != (1, 1, 32):
        raise ValueError('K3 requires recipe (1, 1, 32)')
    clamp = c_float(inf if activation_clamp is None else float(activation_clamp)).value
    beta = c_float(float(activation_beta)).value
    linear = None if activation_linear_beta is None else c_float(float(activation_linear_beta)).value
    if not (clamp >= 0 and isfinite(beta) and beta > 0 and
            (linear is None or (isfinite(linear) and linear > 0))):
        raise ValueError('clamp must be nonnegative; supplied betas finite and positive')
    h, i = sym_buffer.hidden, sym_buffer.intermediate_hidden
    e = sym_buffer.num_experts // sym_buffer.group.size()
    for pair in (l1_weights, l2_weights):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError('routed weights must be transformed (FP4 payload, SF) pairs')
    _check_weight(l1_weights[0], (e, i // 128, h // 128, 256, 64),
                  torch.int8, 'routed W1 slice-major payload')
    _check_weight(l2_weights[0], (e, h // 256, i // 128, 256, 64),
                  torch.int8, 'routed W2 slice-major payload')
    _check_sf(l1_weights[1], (e,2*i,h//128), 'routed W1 SF')
    _check_sf(l2_weights[1], (e,h,i//128), 'routed W2 SF')
    hs, si = sym_buffer.shared_hidden, sym_buffer.shared_intermediate_hidden
    if not isinstance(shared_x, torch.Tensor) or shared_x.ndim != 2 or shared_x.shape[0] < max(1, t):
        raise ValueError('shared_x requires physical rows >= max(1,T)')
    _check_weight(shared_x, (shared_x.shape[0], hs), torch.bfloat16, 'shared input')
    _check_weight(shared_l1_weights, (si // 128, hs // 64, 256, 64),
                  torch.bfloat16, 'shared W1 slice-major payload')
    _check_weight(shared_l2_weights, (hs // 256, si // 64, 256, 64),
                  torch.bfloat16, 'shared W2 slice-major payload')
    tensors = (shared_x, *l1_weights, *l2_weights, shared_l1_weights, shared_l2_weights)
    device = sym_buffer.buffer.device
    if any(x.device != device for x in tensors) or torch.cuda.current_device() != device.index:
        raise ValueError('all tensors must use the workspace and current CUDA device')
    for dst in (out, sym_buffer.buffer):
        if any(_overlaps(w, dst) for w in (l1_weights[0], l2_weights[0],
                                            shared_l1_weights, shared_l2_weights)) or any(
                _sf_overlaps(sf, dst) for sf in (l1_weights[1], l2_weights[1])):
            raise ValueError('expert weights must be disjoint from output and workspace')
    if _overlaps(shared_x, shared_output):
        raise ValueError('shared input and workspace output must not overlap')
    bm = get_block_m_for_gc_moe_k3(sym_buffer.group.size(), sym_buffer.num_experts,
                                  sym_buffer.num_max_tokens_per_rank, t, sym_buffer.num_topk)
    return ValidatedCall(t, bm, h, i, e, sym_buffer.num_shared_experts, hs, si,
                         clamp, beta, linear, bool(fast_math))


def fp8_fp4_gc_moe_k3(
    out, l1_weights, l2_weights, sym_buffer: GCK3SymmBuffer, *,
    hidden_states, front_weight, router_bias,
    shared_l1_weights, shared_l2_weights, norm_weight, up_proj_weight,
    prefix_sum=None, norm_eps=1e-6, routed_scaling_factor=1.0,
    route_renormalize=True, valid_token_count=None,
    recipe=(1, 1, 32), activation: str = 'situ', activation_clamp=None,
    activation_beta=4.0, activation_linear_beta=25.0, fast_math=True,
):
    """Write the complete MoE and shared-expert sum to BF16 ``out[T, H_shared]``.

    The number of routed experts must be divisible by 32. The routed
    intermediate width must be a multiple of 128 and at most 4096.
    At least one shared expert is required.

    ``hidden_states`` is the BF16 post-attention normalization output.
    ``front_weight`` concatenates the BF16 gate and latent down-projection
    weights along the output axis. ``router_bias`` is FP32. Use transformed
    slice-major expert weights and a BF16 up-projection with shape
    ``[H_shared, H]``. Optional CUDA int32 ``valid_token_count`` identifies
    the live prefix of a static bucket. Only its live rows define outputs.
    Expert weights must be disjoint from the output and workspace.
    With fast_math=True, SiTU uses approximate exponential, reciprocal and tanh.
    Serialize calls and graph replays that share a workspace allocation.
    With a prefix, round the projection and both additions to BF16 in order.
    Without a prefix, add the shared expert to the FP32 projection before rounding.
    """
    if activation != 'situ':
        raise ValueError("K3 requires activation='situ'")
    if sym_buffer.buffer is None or sym_buffer.handle is None:
        raise ValueError('K3 symmetric buffer has been destroyed')
    if not isinstance(out, torch.Tensor) or out.ndim != 2:
        raise ValueError('final output requires BF16 [T,H_shared]')
    t = out.shape[0]
    h, hs = sym_buffer.hidden, sym_buffer.shared_hidden
    if t > sym_buffer.num_max_tokens_per_rank:
        raise ValueError('tokens exceed symmetric capacity')
    _check_weight(out, (t, hs), torch.bfloat16, 'final output')
    experts = sym_buffer.num_experts
    if experts % 32:
        raise ValueError('the K3 gate width must align to the K32 epilogue groups')
    _check_weight(hidden_states, (t, hs), torch.bfloat16, 'hidden states')
    _check_weight(front_weight, (experts + h, hs), torch.bfloat16, 'gate and down-projection')
    _check_weight(router_bias, (experts,), torch.float32, 'router correction bias')
    if valid_token_count is not None:
        _check_weight(valid_token_count, (), torch.int32, 'valid token count')
    routed_scale = c_float(float(routed_scaling_factor)).value
    if not isfinite(routed_scale) or routed_scale <= 0:
        raise ValueError('routed scaling factor must be positive and finite')
    _check_weight(norm_weight, (h,), torch.bfloat16, 'routed RMSNorm weight')
    _check_weight(up_proj_weight, (hs, h), torch.bfloat16, 'routed up projection')
    if prefix_sum is not None:
        _check_weight(prefix_sum, (t, hs), torch.bfloat16, 'prefix sum')
    workspace = sym_buffer.buffer
    if out.device != workspace.device or _overlaps(out, workspace):
        raise ValueError('final output must be on the workspace device and disjoint from it')
    for tensor in (hidden_states, front_weight, router_bias, valid_token_count,
                   norm_weight, up_proj_weight, prefix_sum):
        if tensor is not None and (tensor.device != out.device or
                                   _overlaps(tensor, out) or _overlaps(tensor, workspace)):
            raise ValueError('K3 inputs must be on the workspace device and disjoint from writes')
    norm_eps = c_float(norm_eps).value
    if not isfinite(norm_eps) or norm_eps <= 0:
        raise ValueError('norm_eps must be positive and finite')

    routed = sym_buffer.routed_latent[:t]
    shared = sym_buffer.shared_output[:t]
    # The zero-row compute node still needs a nonempty map argument in its
    # ABI. No local SE task uses that descriptor on a zero-row rank.
    shared_x = hidden_states if t else sym_buffer.shared_output[:1]
    config = _validate_call(
        out, t, l1_weights, l2_weights, sym_buffer, shared_x, shared,
        shared_l1_weights, shared_l2_weights,
        recipe, activation_clamp, activation_beta, activation_linear_beta, fast_math,
    )
    from ..jit_kernels.impls.sm100_fp8_fp4_gc_moe_k3 import launch_gc_moe_k3

    launch_gc_moe_k3(
        config, routed, l1_weights, l2_weights, sym_buffer, shared_l1_weights,
        shared_l2_weights, shared_x=shared_x,
        shared_y=shared, routed_norm=norm_weight if t else None, norm_eps=norm_eps,
        routed_scale=routed_scale,
        projection=(up_proj_weight, prefix_sum, out) if t else None,
        front=(front_weight, router_bias, valid_token_count, bool(route_renormalize)) if t else None,
    )
