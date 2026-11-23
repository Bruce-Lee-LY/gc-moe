"""Public GC MoE allocation and weight-layout entrypoints."""

from __future__ import annotations

from dataclasses import astuple, fields
from typing import Tuple, overload
import types
import weakref

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from ..jit_kernels.heuristics.gc_moe_k3 import get_block_m_for_gc_moe_k3, get_ring_tokens
from ..layout.gc_moe_k3 import GCK3BufferLayout, TOKEN_ALIGNMENT, align


_LIVE_BUFFERS = weakref.WeakValueDictionary()


class GCK3SymmBuffer:
    """Independent K3 symmetric workspace; serialize calls and retire captured graphs.

    New allocations report local configuration errors collectively and
    require matching dispatch SMs and workspace layouts.
    GC_MOE_DISPATCH_SMS is read when allocating the workspace.
    """

    _views = ('x', 'x_sf', 'topk_idx', 'topk_weights', 'shared_l2_acts',
              'l1_acts', 'l1_acts_sf', 'l2_acts', 'l2_acts_sf',
              'routed_latent', 'shared_output', 'front_logits')

    def __init__(self, group: dist.ProcessGroup, num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int, hidden: int,
                 intermediate_hidden: int, mma_type: str = 'fp8xfp4',
                 activation: str = 'situ', base=None, *, num_shared_experts: int,
                 shared_hidden: int, shared_intermediate_hidden: int):
        self.group = group
        self.rank_idx = dist.get_rank(group=group)
        num_ranks = group.size()
        self.num_experts = num_experts
        self.num_topk = num_topk
        self.hidden, self.intermediate_hidden = hidden, intermediate_hidden
        self.num_shared_experts = num_shared_experts
        self.shared_hidden = shared_hidden
        self.shared_intermediate_hidden = shared_intermediate_hidden
        self.mma_type, self.activation = mma_type, activation
        config_error = None
        status = -2
        try:
            if mma_type != 'fp8xfp4':
                raise ValueError('K3 requires fp8xfp4')
            if activation != 'situ':
                raise ValueError("K3 requires activation='situ'")
            if base is not None and not isinstance(base, GCK3SymmBuffer):
                raise ValueError('base must be a GC-compatible GCK3SymmBuffer')
            if num_max_tokens_per_rank <= 0 or hidden <= 0 or intermediate_hidden <= 0:
                raise ValueError('K3 capacity and routed dimensions must be positive')
            self.num_max_tokens_per_rank = align(num_max_tokens_per_rank, TOKEN_ALIGNMENT)
            num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
            ring_tokens, sf_ring_tokens = get_ring_tokens(
                num_ranks, num_experts, self.num_max_tokens_per_rank, num_topk,
                hidden, intermediate_hidden, num_sms)
            self.layout = GCK3BufferLayout(
                num_ranks, num_experts, self.num_max_tokens_per_rank, num_topk,
                hidden, intermediate_hidden, ring_tokens, sf_ring_tokens,
                num_shared_experts, shared_hidden, shared_intermediate_hidden)
            if base is None:
                from ..runtime.gc_moe_k3 import get_dispatch_sm_config

                status = -1
                self._requested_dispatch_sms, self._dispatch_sms = get_dispatch_sm_config(
                    num_ranks=num_ranks)
                status = -2
                if num_ranks > 1:
                    local_config = torch.tensor(
                        (self._dispatch_sms, *astuple(self.layout)),
                        dtype=torch.int64, device='cuda')
        except (ValueError, RuntimeError, TypeError, ArithmeticError) as exc:
            if base is not None or num_ranks == 1:
                raise
            config_error = exc

        if base is None:
            if num_ranks > 1:
                if config_error is not None:
                    # Use an identical shape even when no local layout could be constructed.
                    values = (status,) + (0,) * len(fields(GCK3BufferLayout))
                    local_config = torch.tensor(values, dtype=torch.int64, device='cuda')
                group_configs = torch.empty(
                    (num_ranks, local_config.numel()), dtype=local_config.dtype,
                    device=local_config.device)
                dist.all_gather(list(group_configs.unbind()), local_config, group=group)
                configs = group_configs.tolist()
                invalid_layouts = [rank for rank, config in enumerate(configs) if config[0] == -2]
                invalid_dispatch = [rank for rank, config in enumerate(configs) if config[0] == -1]
                if invalid_layouts:
                    message = f'K3 workspace configuration failed on group ranks {invalid_layouts}'
                    if invalid_dispatch:
                        message += f'; dispatch SM configuration failed on group ranks {invalid_dispatch}'
                    raise ValueError(message) from config_error
                if invalid_dispatch:
                    raise ValueError(
                        f'K3 dispatch SM configuration failed on group ranks {invalid_dispatch}'
                    ) from config_error
                counts = [config[0] for config in configs]
                if any(count != counts[0] for count in counts):
                    raise ValueError(f'K3 dispatch SM count must match on all ranks; got {counts}')
                mismatched_ranks = [rank for rank, config in enumerate(configs)
                                    if config[1:] != configs[0][1:]]
                if mismatched_ranks:
                    raise ValueError(
                        f'K3 workspace layout differs from group rank 0 on ranks {mismatched_ranks}')
            allocator = torch if num_ranks == 1 else symm_mem
            self.buffer = allocator.empty(self.layout.nbytes, dtype=torch.int8, device='cuda')
            self.handle = (types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
                           if group.size() == 1 else symm_mem.rendezvous(self.buffer, group=group))
            peer_addresses = tuple(int(ptr) for ptr in self.handle.buffer_ptrs)
            if (len(peer_addresses) != group.size() or
                    any(ptr <= 0 or ptr % 16 for ptr in peer_addresses) or
                    peer_addresses[self.rank_idx] != self.buffer.data_ptr()):
                raise RuntimeError('symmetric rendezvous returned invalid peer addresses')
            self.peer_addresses = peer_addresses
            self.buffer.zero_()
            group.barrier().wait()
            torch.cuda.synchronize()
        else:
            if base.buffer is None or base.handle is None or base.group is not group:
                raise ValueError('base must be a live allocation in the same group')
            for name in ('num_experts', 'num_max_tokens_per_rank', 'num_topk', 'hidden',
                         'intermediate_hidden', 'num_shared_experts', 'shared_hidden',
                         'shared_intermediate_hidden', 'mma_type', 'activation'):
                if getattr(base, name) != getattr(self, name):
                    raise ValueError(f"incompatible base: {name}")
            if self.layout != base.layout:
                raise ValueError('incompatible base: GC workspace layout')
            if self.layout.nbytes > base.buffer.nbytes:
                raise ValueError('base allocation too small')
            self._requested_dispatch_sms = base._requested_dispatch_sms
            self._dispatch_sms = base._dispatch_sms
            self.buffer, self.handle = base.buffer, base.handle
            peer_addresses = tuple(int(ptr) for ptr in self.handle.buffer_ptrs)
            if (len(peer_addresses) != group.size() or
                    any(ptr <= 0 or ptr % 16 for ptr in peer_addresses) or
                    peer_addresses[self.rank_idx] != self.buffer.data_ptr()):
                raise RuntimeError('base rendezvous returned invalid peer addresses')
            # A compatible base may own the allocation. The 592-byte host
            # SymBuffer value is assembled per call from these rendezvous UVA
            # addresses and retained by each child Graph argument vector.
            self.peer_addresses = peer_addresses
        for name in self._views:
            spec = self.layout.views[name]
            if spec.nbytes == 0:
                view = None
            else:
                dtype = getattr(torch, spec.dtype)
                elem_size = torch.empty((), dtype=dtype).element_size()
                raw = self.buffer.narrow(0, spec.offset, spec.nbytes).view(dtype)
                view = raw.as_strided(spec.shape, spec.stride) if spec.stride else raw.view(spec.shape)
                if spec.nbytes % elem_size:
                    raise AssertionError(f"unaligned view {name}")
            setattr(self, name, view)
        _LIVE_BUFFERS[id(self)] = self

    def destroy(self):
        if self.buffer is not None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('destroy K3 workspaces outside stream capture')
            pointer = self.buffer.data_ptr()
            shared = any(other is not self and other.buffer is not None and
                         other.buffer.data_ptr() == pointer
                         for other in tuple(_LIVE_BUFFERS.values()))
            if not shared:
                from ..runtime.gc_moe_k3 import _release_workspace_storage
                _release_workspace_storage(pointer)
        _LIVE_BUFFERS.pop(id(self), None)
        for name in self._views:
            setattr(self, name, None)
        self.peer_addresses = self.handle = self.buffer = self.group = None


def get_symm_buffer_for_gc_moe_k3(group: dist.ProcessGroup, num_experts: int,
                                 num_max_tokens_per_rank: int, num_topk: int,
                                 hidden: int, intermediate_hidden: int,
                                 mma_type: str = 'fp8xfp4', activation: str = 'situ', *,
                                 num_shared_experts: int, shared_hidden: int,
                                 shared_intermediate_hidden: int) -> GCK3SymmBuffer:
    return GCK3SymmBuffer(
        group, num_experts, num_max_tokens_per_rank, num_topk, hidden,
        intermediate_hidden, mma_type, activation,
        num_shared_experts=num_shared_experts,
        shared_hidden=shared_hidden, shared_intermediate_hidden=shared_intermediate_hidden,
    )


def _gc_interleave_rows(t: torch.Tensor) -> torch.Tensor:
    if t.ndim not in (2, 3) or t.shape[-2] % 16:
        raise ValueError('W1 rows must be a multiple of 16')
    grouped = t if t.ndim == 3 else t.unsqueeze(0)
    groups, rows = grouped.shape[:2]
    half = rows // 2
    gate = grouped[:, :half].reshape(groups, half // 8, 8, *grouped.shape[2:])
    up = grouped[:, half:].reshape(groups, half // 8, 8, *grouped.shape[2:])
    result = torch.empty_like(grouped).copy_(torch.stack((gate, up), dim=2).reshape(grouped.shape))
    return result if t.ndim == 3 else result.squeeze(0)


def _gc_transpose_sf(sf: torch.Tensor) -> torch.Tensor:
    if sf.dtype != torch.int32 or sf.ndim not in (2, 3) or sf.shape[-2] % 128:
        raise ValueError('SF must be packed int32 with 128-aligned rows')
    grouped = sf if sf.ndim == 3 else sf.unsqueeze(0)
    groups, rows, packs = grouped.shape
    result = torch.empty_like(grouped).copy_(
        grouped.reshape(groups, -1, 4, 32, packs).transpose(2, 3).reshape(grouped.shape))
    return result if sf.ndim == 3 else result.squeeze(0)


def _k3_slice_major(weight):
    """Pack an N256 slice's K segments consecutively, keeping its axis explicit."""
    rows, cols = weight.shape[-2:]
    if not weight.is_contiguous() or rows % 256 or cols % 64:
        raise ValueError('slice-major weights require contiguous whole N256/K segments')
    return (weight.view(*weight.shape[:-2], rows // 256, 256, cols // 64, 64)
                  .transpose(-3, -2).contiguous())


@overload
def transform_weights_for_gc_moe_k3(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor], activation: str = 'situ',
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]: ...


@overload
def transform_weights_for_gc_moe_k3(
    l1_weights: torch.Tensor, l2_weights: torch.Tensor, activation: str = 'situ',
) -> Tuple[torch.Tensor, torch.Tensor]: ...


def transform_weights_for_gc_moe_k3(l1_weights, l2_weights, activation: str = 'situ'):
    """Pack original routed FP4 or shared BF16 weights into owned slice-major tensors.

    Routed inputs are FP4 payload/packed K32 SF pairs. SF must already be
    dense MN-major int32 [E,N,K/128], with strides (N*(K/128),1,N),
    four UE8M0 K32 scales per word and N divisible by 128. As in the
    SM100 transform, row interleave/permutation preserves the supported
    layout; it does not normalize row-major or arbitrary padded inputs.
    Shared inputs are dense BF16 W1/W2 without scale factors. W1 rows
    are interleaved before packing. Each payload is
    [..., N/256, K/tile_k, 256, 64], where tile_k is 128 for packed
    FP4 and 64 for BF16. SF layout is unchanged.
    Call outside capture with ordered source writes.
    Copies run asynchronously on the current CUDA stream; raw inputs may die
    on return.
    """
    if activation != 'situ':
        raise ValueError("K3 requires activation='situ'")
    if isinstance(l1_weights, torch.Tensor) and isinstance(l2_weights, torch.Tensor):
        if (l1_weights.dtype != torch.bfloat16 or l2_weights.dtype != torch.bfloat16 or
                l1_weights.ndim != 2 or l2_weights.ndim != 2 or
                l1_weights.shape != (2*l2_weights.shape[1], l2_weights.shape[0]) or
                not l1_weights.is_cuda or l1_weights.device != l2_weights.device or
                l1_weights.device.index != torch.cuda.current_device()):
            raise ValueError('shared transform requires BF16 W1[2Is,Hs], W2[Hs,Is]')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('transform weights outside capture')
        for tensor in (l1_weights,l2_weights):
            tensor.record_stream(torch.cuda.current_stream())
        source_storage = {l1_weights.untyped_storage().data_ptr(),
                          l2_weights.untyped_storage().data_ptr()}
        packed = [_k3_slice_major(w) for w in (_gc_interleave_rows(l1_weights), l2_weights)]
        # A degenerate K64 permutation may return a view of its input.
        packed = [t.clone() if t.untyped_storage().data_ptr() in source_storage else t
                  for t in packed]
        return packed[0], packed[1]
    pairs = (l1_weights, l2_weights)
    if any(not isinstance(pair, (tuple, list)) or len(pair) != 2 or
           any(not isinstance(t, torch.Tensor) for t in pair) for pair in pairs):
        raise ValueError('transform needs two original quantized (payload, SF) pairs')
    if any(payload.ndim != 3 or payload.dtype != torch.int8 for payload, _ in pairs):
        raise ValueError('routed transform requires 3D packed int8 FP4 payloads')
    if any(not t.is_cuda or t.device.index != torch.cuda.current_device()
           for pair in pairs for t in pair):
        raise ValueError('transform inputs must use the current CUDA device')
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('transform weights outside CUDA Graph capture')
    stream = torch.cuda.current_stream()
    for pair in pairs:
        for tensor in pair:
            tensor.record_stream(stream)
    source_storage = {pair[0].untyped_storage().data_ptr() for pair in pairs}
    raw1, raw2 = _gc_interleave_rows(l1_weights[0]), l2_weights[0]
    packed = [_k3_slice_major(w) for w in (raw1, raw2)]
    # A degenerate K128 permutation may return a view of its input.
    packed = [t.clone() if t.untyped_storage().data_ptr() in source_storage else t
              for t in packed]
    sf1 = _gc_transpose_sf(_gc_interleave_rows(l1_weights[1]))
    sf2 = _gc_transpose_sf(l2_weights[1])
    return (packed[0], sf1), (packed[1], sf2)


from ..apis.gc_moe_k3 import fp8_fp4_gc_moe_k3
