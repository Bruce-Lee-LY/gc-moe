"""Validate caller tensors and launch Sail GC MoE decode."""

from dataclasses import dataclass
from math import inf, isfinite
from operator import index

import torch

from ..jit_kernels.impls.ppu_gc_moe import launch_gc_moe
from ..layout.ppu_gc_moe import FP4_FP4_PACKET_BYTES, FP8_FP4_PACKET_BYTES, WEIGHT_PACKET_BYTES, check_tensor, pack_weight
from ..runtime.ppu_gc_moe import PPUGCSymmBuffer


WeightPair = tuple[torch.Tensor, torch.Tensor]
PackedFP8FP8PTPCWeights = tuple[WeightPair, WeightPair]


@dataclass(frozen=True)
class ValidatedCall:
    tokens: int
    bound: int
    clamp: float
    routed_weights: tuple[torch.Tensor, torch.Tensor]
    shared_fp8_fp8_ptpc_weights: PackedFP8FP8PTPCWeights | None
    routed_inverse_scales: tuple[float, float]


def get_symm_buffer_for_gc_moe_ppu(*args, **kwargs):
    """Allocate a Sail GC MoE workspace for the supplied device geometry."""
    return PPUGCSymmBuffer(*args, **kwargs)


def _weight_payload(pair) -> torch.Tensor:
    if not isinstance(pair, (tuple, list)) or len(pair) != 2 or not isinstance(pair[0], torch.Tensor):
        raise ValueError('expected weight and scale tensor pairs')
    return pair[0]


def _routed_weight(pair, experts, n, k, device, routed_precision):
    payload = _weight_payload(pair)
    if routed_precision == 'fp8_fp4':
        inverse = float(pair[1])
        if not isfinite(inverse) or inverse <= 0:
            raise ValueError('expected a finite positive W4A8 inverse weight scale')
        shape = (experts, n // 128, k // 64, FP8_FP4_PACKET_BYTES)
    else:
        inverse = 1.0
        shape = (experts, n // 128, k // 128, FP4_FP4_PACKET_BYTES)
    return check_tensor(payload, shape, torch.uint8, device), inverse


def _fp8_fp8_ptpc_weight(pair, n, k, device, *, first):
    payload = _weight_payload(pair)
    if payload.dtype == torch.float8_e4m3fn:
        pair = pack_weight(pair, n, k, precision='fp8_fp8_ptpc', first=first)
    shape = (n // 128, k // 64, WEIGHT_PACKET_BYTES)
    return (check_tensor(pair[0], shape, torch.uint8, device),
            check_tensor(pair[1], (n, 1), torch.float32, device))


def validate_call(y, l1_weights, l2_weights, sym_buffer, stats, recipe, activation,
                  activation_clamp, num_tokens_bound, shared_l1_weights, shared_l2_weights,
                  shared_recipe, *, routed_precision) -> ValidatedCall:
    if not isinstance(sym_buffer, PPUGCSymmBuffer) or sym_buffer.buffer is None:
        raise ValueError('expected a live PPU workspace')
    if sym_buffer.routed_precision != routed_precision:
        raise ValueError('workspace routed precision differs from the requested operator')

    device = sym_buffer._device
    if (tuple(recipe) != (1, 1, 32) or activation != 'swiglu' or shared_recipe is not None
            or torch.cuda.current_device() != device.index):
        raise ValueError('invalid quantization recipe, activation or current device')
    clamp = inf if activation_clamp is None else float(activation_clamp)
    if not clamp >= 0:
        raise ValueError('expected a nonnegative activation clamp')

    if not isinstance(y, torch.Tensor) or y.ndim != 2:
        raise ValueError('y must be a contiguous BF16 [tokens, hidden] tensor')
    tokens = y.size(0)
    capacity = sym_buffer.num_max_tokens_per_rank
    bound = capacity if num_tokens_bound is None else index(num_tokens_bound)
    if not 0 <= tokens <= bound <= capacity:
        raise ValueError('invalid token count or bound')

    hidden, intermediate = sym_buffer.hidden, sym_buffer.intermediate_hidden
    experts = sym_buffer.layout.local_experts
    check_tensor(y, (tokens, hidden), torch.bfloat16, device)
    if stats is not None:
        check_tensor(stats, (experts,), torch.int32, device)
    routed_w1, inv1 = _routed_weight(l1_weights, experts, 2 * intermediate, hidden, device, routed_precision)
    routed_w2, inv2 = _routed_weight(l2_weights, experts, hidden, intermediate, device, routed_precision)

    shared = shared_l1_weights is not None
    if shared != (shared_l2_weights is not None) or (shared and sym_buffer.shared_x is None):
        raise ValueError('supply both shared weight pairs and a PTPC workspace')
    shared_fp8_fp8_ptpc_weights = None
    if shared:
        shared_fp8_fp8_ptpc_weights = (
            _fp8_fp8_ptpc_weight(shared_l1_weights, 2 * intermediate, hidden, device, first=True),
            _fp8_fp8_ptpc_weight(shared_l2_weights, hidden, intermediate, device, first=False),
        )
    return ValidatedCall(tokens, bound, clamp, (routed_w1, routed_w2), shared_fp8_fp8_ptpc_weights, (inv1, inv2))


def ppu_fp4_fp4_gc_moe(y, l1_weights, l2_weights, sym_buffer,
                      cumulative_local_expert_recv_stats=None, recipe=(1, 1, 32),
                      activation='swiglu', activation_clamp=None, fast_math=True,
                      num_tokens_bound=None, shared_l1_weights=None, shared_l2_weights=None,
                      shared_recipe=None) -> None:
    """Run packed W4A4 decode with an optional W8A8 PTPC shared expert.

    `fast_math` is accepted for API compatibility. The Sail backend uses the
    same arithmetic for both values.
    """
    config = validate_call(y, l1_weights, l2_weights, sym_buffer,
                           cumulative_local_expert_recv_stats, recipe, activation,
                           activation_clamp, num_tokens_bound, shared_l1_weights,
                           shared_l2_weights, shared_recipe, routed_precision='fp4_fp4')
    launch_gc_moe(config, y, sym_buffer, cumulative_local_expert_recv_stats)


def ppu_fp8_fp4_gc_moe(y, l1_weights, l2_weights, sym_buffer,
                      cumulative_local_expert_recv_stats=None, recipe=(1, 1, 32),
                      activation='swiglu', activation_clamp=None, fast_math=True,
                      num_tokens_bound=None, shared_l1_weights=None, shared_l2_weights=None,
                      shared_recipe=None) -> None:
    """Run W4A8 routed decode with an optional W8A8 PTPC shared expert."""
    config = validate_call(y, l1_weights, l2_weights, sym_buffer,
                           cumulative_local_expert_recv_stats, recipe, activation,
                           activation_clamp, num_tokens_bound, shared_l1_weights,
                           shared_l2_weights, shared_recipe, routed_precision='fp8_fp4')
    launch_gc_moe(config, y, sym_buffer, cumulative_local_expert_recv_stats)
