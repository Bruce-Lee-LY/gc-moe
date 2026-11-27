"""Sail GC MoE symmetric memory and Tensor Cell weight layouts."""

from dataclasses import dataclass
from functools import cached_property
from math import prod

import torch

from .gc_moe import align


LOCAL_RANKS, MAX_RANKS, MAX_TOKENS = 8, 64, 256
WEIGHT_PACKET_BYTES = 8192
FP4_FP4_SCALE_PACKET_BYTES = 512
HEADER_BYTES = 256
FP4_FP4_PACKET_BYTES = WEIGHT_PACKET_BYTES + FP4_FP4_SCALE_PACKET_BYTES
FP8_FP4_PACKET_BYTES = WEIGHT_PACKET_BYTES // 2 + FP4_FP4_SCALE_PACKET_BYTES // 2


def check_tensor(tensor, shape, dtype, device):
    if (not isinstance(tensor, torch.Tensor) or tensor.shape != shape or tensor.dtype != dtype
            or tensor.device != device or not tensor.is_contiguous()):
        raise ValueError(f'expected contiguous {dtype} {shape} on {device}')
    return tensor


def check_weight_dimensions(hidden: int, intermediate: int) -> None:
    if not (1024 <= hidden <= 8192 and hidden % 512 == 0
            and intermediate >= 128 and intermediate % 128 == 0):
        raise ValueError('expected hidden in [1024, 8192] aligned to 512 and a positive intermediate aligned to 128')


@dataclass(frozen=True)
class BufferView:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    offset: int
    nbytes: int


@dataclass(frozen=True)
class PPUBufferLayout:
    ranks: int
    experts: int
    capacity: int
    topk: int
    hidden: int
    routed_precision: str = 'fp4_fp4'

    @property
    def local_experts(self) -> int:
        return self.experts // self.ranks

    @property
    def max_routes(self) -> int:
        return self.capacity * self.ranks * self.topk

    @property
    def header_bytes(self) -> int:
        return align((5 * max(LOCAL_RANKS, self.ranks) + 6) * 4, HEADER_BYTES)

    @cached_property
    def views(self) -> tuple[BufferView, ...]:
        ranks, capacity, topk, hidden = self.ranks, self.capacity, self.topk, self.hidden
        if self.routed_precision == 'fp8_fp4':
            activation = ('x', (capacity, hidden), torch.float8_e4m3fn, 1)
            scale = ('x_sf', (capacity, 1), torch.float32, 4)
        else:
            activation = ('x', (capacity, hidden // 2), torch.uint8, 1)
            scale = ('x_sf', (capacity, hidden // 32), torch.uint8, 1)
        inbox_row_bytes = align(activation[1][-1] + scale[1][-1] * scale[3] + topk * 4, HEADER_BYTES)
        specs = (
            activation,
            scale,
            ('topk_idx', (capacity, topk), torch.int64, 8),
            ('topk_weights', (capacity, topk), torch.float32, 4),
            ('_returns', (capacity, topk, hidden), torch.bfloat16, 2),
            ('_inbox', (ranks, capacity, inbox_row_bytes), torch.uint8, 1),
            ('_inbox_ids', (ranks, capacity, topk), torch.int32, 4),
        )
        offset = self.header_bytes
        views = []
        for name, shape, dtype, itemsize in specs:
            offset = align(offset, HEADER_BYTES)
            nbytes = prod(shape) * itemsize
            views.append(BufferView(name, shape, dtype, offset, nbytes))
            offset += nbytes
        return tuple(views)

    @property
    def nbytes(self) -> int:
        last = self.views[-1]
        return last.offset + last.nbytes


def pack_weight(pair, n, k, *, precision, first):
    weight, scale = pair
    leading = weight.shape[:-2]
    device = weight.device
    mxfp4_weights = precision in ('fp4_fp4', 'fp8_fp4')
    fp8_fp4 = precision == 'fp8_fp4'
    dtype = torch.uint8 if mxfp4_weights else torch.float8_e4m3fn
    check_tensor(weight, (*leading, n, k // 2 if mxfp4_weights else k), dtype, device)
    check_tensor(scale, (*leading, n, k // 32 if mxfp4_weights else 1),
                 torch.uint8 if mxfp4_weights else torch.float32, device)
    k_tiles = k // (64 if precision == 'fp4_fp4' else 32)
    fragment_bytes = 256 if fp8_fp4 else 512
    inverse_scale = 1.0
    if fp8_fp4:
        low, high = (int(v.item()) for v in torch.aminmax(scale))
        if low == 0 or high == 255 or high - low > 26:
            raise ValueError('W4A8 requires finite E8M0 exponents with a spread of at most 26')
        shift = 114 - low if low < 114 else min(0, 140 - high)
        inverse_scale = 2.0 ** -shift
        scale = ((scale.to(torch.int16) + shift - 114) * 4).to(torch.uint8)
    # K tile, warp lane and the four packed B registers in instruction order.
    packed = weight.view(torch.uint8).reshape(-1, n // 16, 2, 8, k_tiles, 2, 4, 2 if fp8_fp4 else 4)
    packed = packed.permute(0, 1, 4, 3, 6, 2, 5, 7).contiguous()
    # All B fragments used by one warp are contiguous within each K step.
    # FC1 keeps the gate panels followed by their corresponding up panels.
    if first:
        packed = packed.view(-1, 2, n // 64, 2, k_tiles, fragment_bytes).permute(0, 2, 4, 1, 3, 5)
    else:
        packed = packed.view(-1, n // 64, 4, k_tiles, fragment_bytes).permute(0, 1, 3, 2, 4)
    packed = packed.contiguous().view(*leading, n // 16, k_tiles, fragment_bytes)
    if fp8_fp4:
        # Keep each magnitude selector in place. Interleave the two groups of
        # four sign bits in the high and low nibbles of each byte, respectively.
        # Bound temporary storage independently of the number of local experts.
        for words in packed.view(torch.int32).flatten().split(8 * 1024 * 1024):
            rearranged = words & 0x77777777
            for byte in range(4):
                low_k_sign = (words >> (3 + 4 * byte)) & 1
                high_k_sign = (words >> (19 + 4 * byte)) & 1
                rearranged |= (low_k_sign << (7 + 8 * byte)) | (high_k_sign << (3 + 8 * byte))
            words.copy_(rearranged)
    if mxfp4_weights:
        scale_bytes = 1 if fp8_fp4 else 2
        # One vector per lane group carries the exponents of all four panels.
        if first:
            scale = scale.reshape(-1, 2, n // 64, 2, 2, 8, k_tiles, scale_bytes)
            scale = scale.permute(0, 2, 6, 5, 1, 3, 4, 7)
        else:
            scale = scale.reshape(-1, n // 64, 4, 2, 8, k_tiles, scale_bytes)
            scale = scale.permute(0, 1, 5, 4, 2, 3, 6)
        scale = scale.contiguous().view(*leading, n // 16, k_tiles, 16 * scale_bytes)
    pairs = n // 128

    def packet(tensor, slice_bytes):
        grouped = tensor.view(-1, pairs, 2, k_tiles // 2, 2, slice_bytes)
        return grouped.permute(0, 1, 3, 2, 4, 5).reshape(*leading, pairs, k_tiles // 2, 4 * slice_bytes)

    # Each AIU copy spans two N64 groups and two K slices. MXFP4 exponents
    # follow the B fragments in the same packet, keeping their original bytes.
    packed = packet(packed, 4 * fragment_bytes)
    if fp8_fp4:
        return torch.cat((packed, packet(scale, 64)), dim=-1), inverse_scale
    if mxfp4_weights:
        packed = torch.cat((packed, packet(scale, 128)), dim=-1)
        scale = packed[..., WEIGHT_PACKET_BYTES:]
    return packed, scale


def transform_weights_for_gc_moe_ppu(l1_weights, l2_weights, activation='swiglu',
                                    *, routed_precision=None, routed_fp8=None):
    """Pack routed MXFP4 or shared E4M3 weights for PPU decode.

    Pass the workspace's routed precision for MXFP4 weights. Without a routed
    precision, MXFP4 weights use FP4 × FP4 and E4M3 weights use FP8 PTPC.
    """
    if routed_precision is None:
        routed_precision = 'fp8_fp4' if routed_fp8 else None
    elif routed_precision not in ('fp4_fp4', 'fp8_fp4'):
        raise ValueError('routed_precision must be fp4_fp4 or fp8_fp4')
    elif routed_fp8 is not None and (routed_precision == 'fp8_fp4') != routed_fp8:
        raise ValueError('conflicting routed precision settings')

    if activation != 'swiglu':
        raise ValueError('only SwiGLU is supported')
    for pair in (l1_weights, l2_weights):
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                or any(not isinstance(tensor, torch.Tensor) for tensor in pair)):
            raise ValueError('expected weight and scale tensor pairs')
    precision = routed_precision
    if precision is None:
        precision = 'fp4_fp4' if l1_weights[0].dtype == torch.uint8 else 'fp8_fp8_ptpc'
    mxfp4_weights = precision in ('fp4_fp4', 'fp8_fp4')
    if (l1_weights[0].ndim != (3 if mxfp4_weights else 2) or l1_weights[0].ndim != l2_weights[0].ndim
            or l1_weights[0].shape[:-2] != l2_weights[0].shape[:-2]):
        raise ValueError(f'invalid weight shapes for {precision.upper()}')
    hidden, width = l2_weights[0].shape[-2:]
    intermediate = width * 2 if mxfp4_weights else width
    check_weight_dimensions(hidden, intermediate)
    return (pack_weight(l1_weights, 2 * intermediate, hidden, precision=precision, first=True),
            pack_weight(l2_weights, hidden, intermediate, precision=precision, first=False))
