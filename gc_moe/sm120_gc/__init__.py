"""Public allocation, weight preparation and launch for SM120 GC MoE."""
from __future__ import annotations

import types

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from ..gc import _gc_interleave_rows, _gc_transpose_sf, transform_weights_for_gc_moe
from ..layout.sm120_gc_moe import SM120GCBufferLayout, align


class SM120GCSymmBuffer:
    """Own symmetric SM120 workspace. Serialize calls and retire graphs before destroy."""
    def __init__(self, group, num_experts, num_max_tokens_per_rank, num_topk,
                 hidden, intermediate_hidden, activation='swiglu', num_shared_experts=0):
        ranks = group.size()
        if torch.cuda.get_device_capability() != (12, 0):
            raise ValueError('SM120 GC requires compute capability 12.0')
        if not (1 <= ranks <= 32 and num_experts > 0 and num_experts % ranks == 0
                and 1 <= num_topk <= min(32, num_experts)
                and 512 <= hidden <= 8192 and hidden % 128 == 0
                and 256 <= intermediate_hidden <= 4096 and intermediate_hidden % 128 == 0
                and num_max_tokens_per_rank > 0 and activation == 'swiglu'):
            raise ValueError('invalid SM120 GC geometry or activation')
        if not isinstance(num_shared_experts, int) or num_shared_experts < 0:
            raise ValueError('num_shared_experts must be a nonnegative integer')
        self.group, self.rank_idx = group, dist.get_rank(group)
        self.num_experts, self.num_topk = num_experts, num_topk
        self.num_max_tokens_per_rank = align(num_max_tokens_per_rank)
        self.hidden, self.intermediate_hidden = hidden, intermediate_hidden
        self.num_shared_experts = num_shared_experts
        self.mma_type, self.activation = 'fp8xfp4', activation
        self.layout = SM120GCBufferLayout(ranks, num_experts, self.num_max_tokens_per_rank,
                                         num_topk, hidden, intermediate_hidden, num_shared_experts)
        allocator = torch if ranks == 1 else symm_mem
        self.buffer = allocator.empty(self.layout.nbytes, dtype=torch.int8, device='cuda')
        self.handle = (types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()]) if ranks == 1
                       else symm_mem.rendezvous(self.buffer, group=group))
        self.peer_addresses = tuple(int(ptr) for ptr in self.handle.buffer_ptrs)
        if (len(self.peer_addresses) != ranks
                or any(ptr <= 0 or ptr % 16 for ptr in self.peer_addresses)
                or self.peer_addresses[self.rank_idx] != self.buffer.data_ptr()):
            raise RuntimeError('symmetric rendezvous returned invalid peer addresses')
        self.buffer.zero_()
        for name, spec in self.layout.views.items():
            setattr(self, name, self.buffer.narrow(0, spec.offset, spec.nbytes)
                    .view(getattr(torch, spec.dtype)).view(spec.shape))
        self.shared_l1_acts = self.x
        # Shared FC1 reads the SM100 staging layout. It is rank-local, so
        # keep it outside the symmetric allocation.
        self.shared_l1_acts_sf = None
        if num_shared_experts:
            rows = self.num_max_tokens_per_rank // 8 * 128
            self.shared_l1_acts_sf = torch.empty(
                (hidden // 128, rows), dtype=torch.int32, device=self.buffer.device).zero_().t()
        self.peer_table.copy_(torch.tensor(self.peer_addresses, dtype=torch.int64, device=self.buffer.device))
        self.header[0] = 1
        group.barrier().wait()
        torch.cuda.synchronize()

    def destroy(self):
        if self.buffer is not None:
            from ..jit_kernels.impls.sm120_fp8_fp4_gc_moe import release_workspace
            release_workspace(self)
        for name in self.layout.views:
            setattr(self, name, None)
        self.shared_l1_acts = self.shared_l1_acts_sf = None
        self.peer_addresses = self.handle = self.buffer = self.group = None


def get_symm_buffer_for_gc_moe_sm120(group, num_experts, num_max_tokens_per_rank,
                                     num_topk, hidden, intermediate_hidden,
                                     activation='swiglu', num_shared_experts=0):
    return SM120GCSymmBuffer(group, num_experts, num_max_tokens_per_rank, num_topk,
                             hidden, intermediate_hidden, activation, num_shared_experts)


def _scale_codes(scales):
    if scales.dtype in (torch.uint8, torch.float8_e8m0fnu):
        return scales.view(torch.uint8)
    if scales.dtype != torch.float32:
        raise ValueError('scales must be UE8M0 bytes or FP32 powers of two')
    bits = scales.contiguous().view(torch.int32)
    return ((bits >> 23) & 255).to(torch.uint8)


def _mn_major(sf: torch.Tensor) -> torch.Tensor:
    columns, rows = sf.shape[-1], sf.shape[-2]
    mn_major = torch.empty((*sf.shape[:-2], columns, rows), dtype=sf.dtype, device=sf.device)
    mn_major = mn_major.transpose(-2, -1).copy_(sf)
    return mn_major


def _normalize_routed_weights(pair):
    """Normalize routed inputs to the common packed FP4 weight contract."""

    packed, scales = pair
    if packed.ndim != 3 or packed.dtype not in (torch.int8, torch.uint8):
        raise ValueError('expected [E, N, K/2] packed MXFP4')
    e, n, half_k = packed.shape
    scale_cols = half_k // (64 if scales.dtype == torch.int32 else 16)
    if n % 128 or half_k % 64 or tuple(scales.shape) != (e, n, scale_cols):
        raise ValueError('invalid SM120 weight or scale shape')
    words = scales if scales.dtype == torch.int32 else _scale_codes(scales).contiguous().view(torch.int32)
    return packed.view(torch.int8).contiguous(), _mn_major(words)


def transform_weights_for_gc_moe_sm120(l1_weights, l2_weights, activation='swiglu',
                                      *, shared_recipe=None):
    """Convert checkpoint weights on the current stream.

    Inputs are two (payload, scale) pairs, with payloads [E, N, K/2] packed
    MXFP4 and scales [E, N, K/128] packed int32, or [E, N, K/32] UE8M0
    bytes (or FP32 powers of two). Routed W1/W2 use the SM100
    [E, K/128, N, 64] packed payload and MN-major int32 scales.
    Call outside CUDA Graph capture with ordered source writes.

    Two-dimensional FP8 shared weights are also accepted with shared_recipe
    (1,128,128) or (1,1,32). Shared W1/W2 have shapes
    [2*S*I,H] and [H,S*I]; all gate rows precede all up rows.
    Row-expanded packed int32 and exact power-of-two scales use the SM100
    [K/128, N, 128] payload and MN-major int32 layout. Arbitrary FP32
    scales retain the SM120 layout.
    """
    if activation != 'swiglu':
        raise ValueError('GC MoE supports swiglu')
    pairs = (l1_weights, l2_weights)
    if any(not isinstance(pair, (tuple, list)) or len(pair) != 2
           or any(not isinstance(t, torch.Tensor) for t in pair) for pair in pairs):
        raise ValueError('transform needs two original quantized (payload, SF) pairs')
    if any(not t.is_cuda or t.device.index != torch.cuda.current_device()
           for pair in pairs for t in pair):
        raise ValueError('transform inputs must use the current CUDA device')
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('transform weights outside CUDA Graph capture')
    stream = torch.cuda.current_stream()
    for pair in pairs:
        for tensor in pair:
            tensor.record_stream(stream)

    if l1_weights[0].dtype == torch.float8_e4m3fn:
        if (l1_weights[0].ndim != 2 or l2_weights[0].ndim != 2
                or l1_weights[0].shape[0] != 2 * l2_weights[0].shape[1]
                or l2_weights[0].shape[0] != l1_weights[0].shape[1]):
            raise ValueError('shared W1/W2 geometry mismatch')
        recipe = None if shared_recipe is None else tuple(shared_recipe)
        if recipe is not None and recipe not in ((1, 1, 32), (1, 128, 128)):
            raise ValueError('unsupported shared recipe')
        return (_transform_shared(l1_weights, recipe, True),
                _transform_shared(l2_weights, recipe, False))
    if shared_recipe is not None:
        raise ValueError('shared_recipe applies only to FP8 shared weights')
    if (l1_weights[0].ndim != 3 or l2_weights[0].ndim != 3
            or l1_weights[0].shape[0] != l2_weights[0].shape[0]
            or l1_weights[0].shape[1] != 4 * l2_weights[0].shape[2]
            or l2_weights[0].shape[1] != 2 * l1_weights[0].shape[2]):
        raise ValueError('W1/W2 geometry mismatch')
    return transform_weights_for_gc_moe(_normalize_routed_weights(l1_weights),
                                        _normalize_routed_weights(l2_weights), activation)


def _transform_shared(pair, recipe, first):
    from ..jit_kernels.impls.runtime_utils import repack_fp8_weights

    data, sf = pair
    n, k = data.shape
    if n <= 0 or k <= 0 or n % 64 or k % 128 or data.dtype != torch.float8_e4m3fn:
        raise ValueError('invalid SM120 FP8 shared weight geometry, scale or recipe')
    packed = sf.dtype == torch.int32
    if packed:
        # SM120 launchable shared weights have K >= 256, so K32 and K128
        # packed column counts are distinct.
        if recipe is None:
            gran_k = 32 if sf.shape == (n, k // 128) else 128
        else:
            gran_k = recipe[2]
        if sf.shape != (n, (k + gran_k * 4 - 1) // (gran_k * 4)):
            raise ValueError('shared packed UE8M0 scales must have one scale row per weight row')
        scales = sf.contiguous().view(torch.uint8)[:, :k // gran_k]
    else:
        recipe = (1, 1, 32) if recipe is None else recipe
        if sf.dtype != torch.float32 or sf.shape != (n // recipe[1], k // recipe[2]):
            raise ValueError('invalid SM120 FP8 shared weight geometry, scale or recipe')
        scales = sf.repeat_interleave(recipe[1], 0)
        gran_k = recipe[2]
    if not packed:
        bits = scales.contiguous().view(torch.int32)
        exponent = (bits >> 23) & 255
        packed = bool(((bits & -2139095041) == 0).logical_and(
            exponent > 0).logical_and(exponent < 255).all())
        if packed:
            scales = exponent.to(torch.uint8)
    if packed:
        scales = torch.nn.functional.pad(scales, (0, -scales.size(-1) % 4), value=127)
        words = scales.contiguous().view(torch.int32)
        if first:
            data = _gc_interleave_rows(data.view(torch.uint8)).view(data.dtype)
            words = _gc_interleave_rows(words)
        return repack_fp8_weights(data.contiguous()), _gc_transpose_sf(_mn_major(words))

    scales = scales.repeat_interleave(gran_k // 32, 1)
    values = data.view(torch.uint8)
    if first:
        values = _gc_interleave_rows(values)
        scales = _gc_interleave_rows(scales)
    tiles = values.reshape(n // 64, 64, k // 128, 8, 16).permute(0, 2, 1, 3, 4)
    sectors = (torch.arange(8, device=data.device)[None, :]
               ^ (torch.arange(64, device=data.device)[:, None] & 7))
    payload = tiles.gather(3, sectors[None, None, :, :, None].expand(
        n // 64, k // 128, 64, 8, 16)).reshape(n // 64, k // 128, 64, 128)
    tiled_sf = scales.reshape(n // 64, 64, k // 128, 4).permute(0, 2, 1, 3).contiguous()
    return payload, tiled_sf


def sm120_fp8_fp4_gc_moe(y, l1_weights, l2_weights, sym_buffer,
                         cumulative_local_expert_recv_stats=None, recipe=(1, 1, 32),
                         activation='swiglu', activation_clamp=None, fast_math=True,
                         num_tokens_bound=None, shared_l1_weights=None,
                         shared_l2_weights=None, shared_recipe=None):
    from ..apis.sm120_gc_moe import validate_call

    config = validate_call(
        y, l1_weights, l2_weights, sym_buffer, cumulative_local_expert_recv_stats,
        recipe, activation, activation_clamp, fast_math, num_tokens_bound,
        shared_l1_weights, shared_l2_weights, shared_recipe)
    from ..jit_kernels.impls.sm120_fp8_fp4_gc_moe import launch_sm120_gc_moe
    launch_sm120_gc_moe(
        config, y, l1_weights, l2_weights, sym_buffer, cumulative_local_expert_recv_stats,
        shared_l1_weights, shared_l2_weights)
