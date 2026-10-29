"""Host TensorMap construction and CUDA encoding utilities."""

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


def make_tma_2d_desc(tensor: torch.Tensor, inner: int, outer: int,
              block_inner: int, block_outer: int,
              outer_stride: int, swizzle: int) -> TmaDescriptorSpec:
    """Construct a 2D map, including FP4 unpack to 8-bit shared memory."""
    if tensor.dtype == torch.int8 and inner % 128:
        raise ValueError('FP4 unpack requires a 128-element aligned K axis')
    return _base(tensor, (inner, outer), (outer_stride,),
                 (block_inner, block_outer), swizzle)


def make_tma_weights_3d_desc(tensor: torch.Tensor, block_k: int, load_n: int,
                     num_domains: int = 2, swizzle: int = 128) -> TmaDescriptorSpec:
    """Map a dense 3D weight tensor to a 4D (K,N/domain,domain,expert) map."""
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


def make_tma_weights_2d_desc(tensor: torch.Tensor, block_k: int, load_n: int,
                     num_domains: int = 2, swizzle: int = 128) -> TmaDescriptorSpec:
    """Map a dense 2D FP8 weight tensor to a 4D (K,N/domain,domain,1) map."""
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


def make_tma_sf_desc(tensor: torch.Tensor, mn: int, k: int, block_mn: int,
          gran_k: int, groups: int, smem_outer: int,
          sf_k_stride: int = 0) -> TmaDescriptorSpec:
    """Construct a K-major packed-int32 UE8M0 map with the MN axis first."""
    if tensor.dtype != torch.int32 or gran_k not in (32, 128):
        raise ValueError('GC SF is packed int32 at K32 or K128')
    aligned_mn = (mn + 3) // 4 * 4
    outer_stride = sf_k_stride or aligned_mn
    if outer_stride < aligned_mn:
        raise ValueError('SF K stride shorter than aligned MN axis')
    k_packs = (k + gran_k * 4 - 1) // (gran_k * 4)
    return make_tma_2d_desc(tensor, aligned_mn, k_packs * groups,
                     block_mn, smem_outer, outer_stride, 0)
