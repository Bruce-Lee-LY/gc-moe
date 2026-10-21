"""Host tensor-map specifications for the four-phase GC math worker.

The role2 kernel receives CUtensorMap values for tiled TMA loads. These
specifications define physical dimensions, strides and the FP4 unpack
format for routed and shared inputs, weights and intermediate outputs.
CUDA encoding is deferred until a real call with live device pointers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from cuda.bindings import driver as cu



_DTYPE = {
    torch.int32: cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_INT32,
    torch.float32: cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
    torch.bfloat16: cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    torch.float8_e4m3fn: cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.int8: cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
}
_SWIZZLE = {
    0: cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
    32: cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_32B,
    64: cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B,
    128: cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
}


@dataclass(frozen=True)
class TmaDescriptorSpec:
    address: int
    dtype: torch.dtype
    dims: tuple[int, ...]
    strides_bytes: tuple[int, ...]
    box: tuple[int, ...]
    swizzle_bytes: int

    def __post_init__(self) -> None:
        rank = len(self.dims)
        if self.dtype not in _DTYPE or rank not in (2, 3, 4, 5):
            raise ValueError('unsupported GC TMA descriptor type/rank')
        if len(self.strides_bytes) != rank - 1 or len(self.box) != rank:
            raise ValueError('invalid GC TMA rank')
        if self.swizzle_bytes not in _SWIZZLE or self.address % 16:
            raise ValueError('unaligned GC TMA base or swizzle')
        if any(x <= 0 for x in self.dims + self.box):
            raise ValueError('GC TMA dimensions must be positive')
        if any(x % 16 for x in self.strides_bytes):
            raise ValueError('GC TMA strides must be 16-byte aligned')

    def encode(self) -> cu.CUtensorMap:
        """Encode only at launch preparation, with a CUDA context current."""
        result = cu.cuTensorMapEncodeTiled(
            _DTYPE[self.dtype], len(self.dims), self.address,
            tuple(cu.cuuint64_t(x) for x in self.dims),
            tuple(cu.cuuint64_t(x) for x in self.strides_bytes),
            tuple(cu.cuuint32_t(x) for x in self.box),
            tuple(cu.cuuint32_t(1) for _ in self.dims),
            cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            _SWIZZLE[self.swizzle_bytes],
            cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
        if result[0] != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuTensorMapEncodeTiled failed: {result[0]}")
        return result[1]


def _base(tensor: torch.Tensor, dims: Sequence[int], strides_elements: Sequence[int],
          box: Sequence[int], swizzle: int) -> TmaDescriptorSpec:
    element_bytes = tensor.element_size()
    box = tuple(box)
    if swizzle:
        box = (swizzle // element_bytes, *box[1:])
    return TmaDescriptorSpec(
        tensor.data_ptr(), tensor.dtype, tuple(dims),
        tuple(int(s) * element_bytes for s in strides_elements), box, swizzle)


def matrix_2d(tensor: torch.Tensor, inner: int, outer: int,
              block_inner: int, block_outer: int,
              outer_stride: int, swizzle: int) -> TmaDescriptorSpec:
    """Reference make_tma_2d_desc, including FP4 unpack to 8b SMEM."""
    if tensor.dtype == torch.int8 and inner % 128:
        raise ValueError('FP4 unpack requires a 128-element aligned K axis')
    return _base(tensor, (inner, outer), (outer_stride,),
                 (block_inner, block_outer), swizzle)


def routed_weight_4d(tensor: torch.Tensor, block_k: int, load_n: int,
                     num_domains: int = 2, swizzle: int = 128) -> TmaDescriptorSpec:
    """Reference dense 3D weight → (K,N/domain,domain,expert) TMA."""
    if tensor.ndim != 3 or tensor.dtype != torch.int8:
        raise ValueError('routed FP4 weights must be packed int8 3D')
    if not tensor.is_contiguous():
        raise ValueError('routed FP4 weight dense storage must be contiguous')
    experts = tensor.shape[0]
    n_per_domain = tensor.shape[1] // num_domains
    k = tensor.shape[-1] * 2
    if tensor.shape[1] % num_domains:
        raise ValueError('N must divide descriptor slices')
    slice_stride = n_per_domain * tensor.stride(-2)
    batch_stride = tensor.stride(0)
    return _base(tensor, (k, n_per_domain, num_domains, experts),
                 (tensor.stride(-2), slice_stride, batch_stride),
                 (block_k, load_n, 1, 1), swizzle)


def shared_weight_4d(tensor: torch.Tensor, block_k: int, load_n: int,
                     num_domains: int = 2, swizzle: int = 128) -> TmaDescriptorSpec:
    """Reference dense 2D shared FP8 weight → (K,N/domain,domain,1)."""
    if tensor.ndim != 2 or tensor.dtype != torch.float8_e4m3fn:
        raise ValueError('shared FP8 weights must be 2D')
    if not tensor.is_contiguous():
        raise ValueError('shared FP8 weight dense storage must be contiguous')
    n_per_domain = tensor.shape[0] // num_domains
    if tensor.shape[0] % num_domains:
        raise ValueError('shared N must divide descriptor slices')
    k = tensor.shape[-1]
    slice_stride = n_per_domain * tensor.stride(-2)
    return _base(tensor, (k, n_per_domain, num_domains, 1),
                 (tensor.stride(-2), slice_stride, slice_stride * num_domains),
                 (block_k, load_n, 1, 1), swizzle)


def sf_2d(tensor: torch.Tensor, mn: int, k: int, block_mn: int,
          gran_k: int, groups: int, smem_outer: int,
          sf_k_stride: int = 0) -> TmaDescriptorSpec:
    """Reference K-major packed-int32 UE8M0 descriptor, MN axis first."""
    if tensor.dtype != torch.int32 or gran_k not in (32, 128):
        raise ValueError('GC SF is packed int32 at K32 or K128')
    aligned_mn = (mn + 3) // 4 * 4
    outer_stride = sf_k_stride or aligned_mn
    if outer_stride < aligned_mn:
        raise ValueError('SF K stride shorter than aligned MN axis')
    k_packs = (k + gran_k * 4 - 1) // (gran_k * 4)
    return matrix_2d(tensor, aligned_mn, k_packs * groups,
                     block_mn, smem_outer, outer_stride, 0)


@dataclass(frozen=True)
class GCTmaPlan:
    """The 18 tensor maps passed to all three reference role specializations."""

    l1_acts: TmaDescriptorSpec
    l1_acts_sf: TmaDescriptorSpec
    l1_weights: TmaDescriptorSpec
    l1_weights_sf: TmaDescriptorSpec
    l1_output: TmaDescriptorSpec
    l2_acts: TmaDescriptorSpec
    l2_acts_sf: TmaDescriptorSpec
    l2_weights: TmaDescriptorSpec
    l2_weights_sf: TmaDescriptorSpec
    shared_l1_acts: TmaDescriptorSpec
    shared_l1_acts_sf: TmaDescriptorSpec
    shared_l1_weights: TmaDescriptorSpec
    shared_l1_weights_sf: TmaDescriptorSpec
    shared_l1_output: TmaDescriptorSpec
    shared_l2_acts: TmaDescriptorSpec
    shared_l2_acts_sf: TmaDescriptorSpec
    shared_l2_weights: TmaDescriptorSpec
    shared_l2_weights_sf: TmaDescriptorSpec

    def ordered(self) -> tuple[TmaDescriptorSpec, ...]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__)


def build_gc_tma_plan(config, sym_buffer, l1_weights, l2_weights,
                      shared_l1_weights=None, shared_l2_weights=None,
                      shared_weight_gran_k: int = 32) -> GCTmaPlan:
    """Mirror the descriptor call sequence in C++ `sm100_fp8_fp4_gc_moe.hpp`.

    This is host metadata only. Encoding requires the live CUDA context and
    happens when the unified role kernel is prepared for launch.
    """
    bm, bk, bn = config.block_m, 128, 128
    h, i = sym_buffer.hidden, sym_buffer.intermediate_hidden
    sf_m = (bm + 127) // 128 * 128
    ring, sf_ring = sym_buffer.layout.ring_tokens, sym_buffer.layout.sf_ring_tokens
    shared_count = sym_buffer.num_shared_experts
    shared_i = shared_count * i
    if shared_weight_gran_k not in (32, 128):
        raise ValueError('shared UE8M0 K granularity must be 32 or 128')
    if bool(shared_l1_weights) != bool(shared_l2_weights) or \
            bool(shared_l1_weights) != bool(shared_count):
        raise ValueError('shared tensors and expert count disagree')
    def acts(t, inner, outer, block_inner, block_outer, swizzle):
        return matrix_2d(t, inner, outer, block_inner, block_outer,
                         t.stride(-2), swizzle)

    def scales(t, mn, k, block_mn, gran, groups):
        return sf_2d(t, mn, k, block_mn, gran, groups,
                     (bk + gran * 4 - 1) // (gran * 4), t.stride(-1))

    routed = (
        acts(sym_buffer.l1_acts, h, ring, bk, bm // 2, 128),
        scales(sym_buffer.l1_acts_sf, sf_ring, h, sf_m, 32, 1),
        routed_weight_4d(l1_weights[0], bk, bn),
        scales(l1_weights[1], 2 * i, h, bn, 32,
               sym_buffer.num_experts // sym_buffer.group.size()),
        acts(sym_buffer.l2_acts, i, ring, bn // 2,
             config.store_m_l1, 64),
        acts(sym_buffer.l2_acts, i, ring, bk, bm // 2, 128),
        scales(sym_buffer.l2_acts_sf, sf_ring, i, sf_m, 32, 1),
        routed_weight_4d(l2_weights[0], bk, bn),
        scales(l2_weights[1], h, i, bn, 32,
               sym_buffer.num_experts // sym_buffer.group.size()),
    )
    if shared_count:
        shared_sf_tokens = sym_buffer.shared_l1_acts_sf.shape[0]
        shared = (
            acts(sym_buffer.shared_l1_acts, h,
                 sym_buffer.num_max_tokens_per_rank, bk, bm // 2, 128),
            scales(sym_buffer.shared_l1_acts_sf, shared_sf_tokens,
                   h, sf_m, 32, 1),
            shared_weight_4d(shared_l1_weights[0], bk, bn),
            scales(shared_l1_weights[1], 2 * shared_i, h, bn,
                   shared_weight_gran_k, 1),
            acts(sym_buffer.shared_l2_acts, shared_i,
                 sym_buffer.num_max_tokens_per_rank, bn // 2,
                 config.store_m_l1, 64),
            acts(sym_buffer.shared_l2_acts, shared_i,
                 sym_buffer.num_max_tokens_per_rank, bk, bm // 2, 128),
            scales(sym_buffer.shared_l2_acts_sf,
                   sym_buffer.shared_l2_acts_sf.shape[0],
                   shared_i, sf_m, 32, 1),
            shared_weight_4d(shared_l2_weights[0], bk, bn),
            scales(shared_l2_weights[1], h, shared_i, bn,
                   shared_weight_gran_k, 1),
        )
    else:
        # The reference supplies the routed maps in unused shared positions.
        shared = routed
    return GCTmaPlan(*routed, *shared)
