"""Sail GC MoE symmetric memory and Tensor Cell weight layouts."""

from dataclasses import dataclass
from functools import cached_property
from math import prod

import torch

from .gc_moe import align


MAX_RANKS, MAX_TOKENS = 8, 256
WEIGHT_PACKET_BYTES, SCALE_PACKET_BYTES = 8192, 512
HEADER_BYTES = 256


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

    @property
    def local_experts(self) -> int:
        return self.experts // self.ranks

    @property
    def max_routes(self) -> int:
        return self.capacity * self.ranks * self.topk

    @cached_property
    def views(self) -> tuple[BufferView, ...]:
        ranks, capacity, topk, hidden = self.ranks, self.capacity, self.topk, self.hidden
        inbox_row_bytes = align(hidden // 2 + hidden // 32 + topk * 4, HEADER_BYTES)
        specs = (
            ('x', (capacity, hidden // 2), torch.uint8, 1),
            ('x_sf', (capacity, hidden // 32), torch.uint8, 1),
            ('topk_idx', (capacity, topk), torch.int64, 8),
            ('topk_weights', (capacity, topk), torch.float32, 4),
            ('_returns', (capacity, topk, hidden), torch.bfloat16, 2),
            ('_inbox', (ranks, capacity, inbox_row_bytes), torch.uint8, 1),
            ('_inbox_ids', (ranks, capacity, topk), torch.int32, 4),
        )
        offset = HEADER_BYTES
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


def pack_weight(pair, n, k, *, fp4, first):
    weight, scale = pair
    leading = weight.shape[:-2]
    device = weight.device
    dtype = torch.uint8 if fp4 else torch.float8_e4m3fn
    check_tensor(weight, (*leading, n, k // 2 if fp4 else k), dtype, device)
    check_tensor(scale, (*leading, n, k // 32 if fp4 else 1),
            torch.uint8 if fp4 else torch.float32, device)
    k_tiles = k // (64 if fp4 else 32)
    # K tile, warp lane and the four packed B registers in instruction order.
    packed = weight.view(torch.uint8).reshape(-1, n // 16, 2, 8, k_tiles, 2, 4, 4)
    packed = packed.permute(0, 1, 4, 3, 6, 2, 5, 7).contiguous()
    # All B fragments used by one warp are contiguous within each K step.
    # FC1 keeps the gate panels followed by their corresponding up panels.
    if first:
        packed = packed.view(-1, 2, n // 64, 2, k_tiles, 512).permute(0, 2, 4, 1, 3, 5)
    else:
        packed = packed.view(-1, n // 64, 4, k_tiles, 512).permute(0, 1, 3, 2, 4)
    packed = packed.contiguous().view(*leading, n // 16, k_tiles, 512)
    if fp4:
        # One vector per lane group carries the exponents of all four panels.
        if first:
            scale = scale.reshape(-1, 2, n // 64, 2, 2, 8, k_tiles, 2)
            scale = scale.permute(0, 2, 6, 5, 1, 3, 4, 7)
        else:
            scale = scale.reshape(-1, n // 64, 4, 2, 8, k_tiles, 2)
            scale = scale.permute(0, 1, 5, 4, 2, 3, 6)
        scale = scale.contiguous().view(*leading, n // 16, k_tiles, 32)
    pairs = n // 128

    def packet(tensor, slice_bytes):
        grouped = tensor.view(-1, pairs, 2, k_tiles // 2, 2, slice_bytes)
        return grouped.permute(0, 1, 3, 2, 4, 5).reshape(*leading, pairs, k_tiles // 2, 4 * slice_bytes)

    # Each AIU copy spans two N64 groups and two K slices. MXFP4 exponents
    # follow the B fragments in the same packet, keeping their original bytes.
    packed = packet(packed, 2048)
    if fp4:
        packed = torch.cat((packed, packet(scale, 128)), dim=-1)
        scale = packed[..., WEIGHT_PACKET_BYTES:]
    return packed, scale


def transform_weights_for_gc_moe_ppu(l1_weights, l2_weights, activation='swiglu'):
    """Pack routed MXFP4 or shared E4M3 weights for the PPU Tensor Cell."""
    if activation != 'swiglu':
        raise ValueError('only SwiGLU is supported')
    for pair in (l1_weights, l2_weights):
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                or any(not isinstance(tensor, torch.Tensor) for tensor in pair)):
            raise ValueError('expected weight and scale tensor pairs')
    fp4 = l1_weights[0].dtype == torch.uint8
    if (l1_weights[0].ndim != (3 if fp4 else 2) or l1_weights[0].ndim != l2_weights[0].ndim
            or l1_weights[0].shape[:-2] != l2_weights[0].shape[:-2]):
        raise ValueError('expected grouped MXFP4 or a single PTPC shared expert')
    hidden, width = l2_weights[0].shape[-2:]
    intermediate = width * 2 if fp4 else width
    check_weight_dimensions(hidden, intermediate)
    return (pack_weight(l1_weights, 2 * intermediate, hidden, fp4=fp4, first=True),
            pack_weight(l2_weights, hidden, intermediate, fp4=fp4, first=False))
