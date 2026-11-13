"""Public allocation, weight preparation and launch for PPU GC MoE."""

from __future__ import annotations

import math
import operator
from contextlib import nullcontext

import torch
import torch.distributed as dist

from ..jit_kernels.heuristics.ppu_gc_moe import PPURoleConfig
from ..jit_kernels.impls.ppu_fp4_fp4_gc_moe import build_ppu_libraries
from ..layout.ppu_gc_moe import repack_sfw1, repack_sfw2, repack_weight_panels
from ..runtime.ppu_gc_moe import CommState, PipelineState, Role12State


try:
    from torch.overrides import _get_current_function_mode, _pop_mode_temporarily
    from torch.utils._device import DeviceContext
except ImportError:
    DeviceContext = None


_get_current_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)
_DEFAULT_RECIPE = (1, 1, 32)
_get_current_device = getattr(torch._C, "_cuda_getDevice", torch.cuda.current_device)
_is_stream_capturing = getattr(
    torch._C, "_cuda_isCurrentStreamCapturing", torch.cuda.is_current_stream_capturing)


def _tensor_metadata_context():
    # DeviceContext supplies a default to tensor factories. Metadata
    # reads do not allocate tensors.
    if DeviceContext is not None and type(_get_current_function_mode()) is DeviceContext:
        return _pop_mode_temporarily()
    return nullcontext()


class PPUGCSymmBuffer:
    """Own the PPU symmetric workspace and its caller-written MXFP4 inputs.

    Serialize launches and writes on the stream used to create the workspace. All ranks
    use the same token bound. The local token count is the output row count.
    """

    def __init__(self, group, num_experts, num_max_tokens_per_rank, num_topk,
                 hidden, intermediate_hidden, activation='swiglu',
                 num_shared_experts=0, *, cache_dir=None):
        ranks = group.size()
        if (num_max_tokens_per_rank <= 0 or num_experts % ranks
                or num_shared_experts not in (0, 1) or activation != 'swiglu'):
            raise ValueError('invalid PPU GC workspace geometry or activation')
        self.config = PPURoleConfig(hidden, intermediate_hidden, num_experts // ranks, num_topk, ranks)
        self.group, self.rank_idx = group, dist.get_rank(group)
        self.num_experts, self.num_topk = num_experts, num_topk
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.hidden, self.intermediate_hidden = hidden, intermediate_hidden
        self.num_shared_experts = num_shared_experts
        self.cache_dir = cache_dir
        self._stream = torch.cuda.current_stream()
        comm, self._pipeline_path, _ = build_ppu_libraries(self.config, (), cache_dir=cache_dir)
        self._comm = CommState(str(comm), group, self.rank_idx, ranks, hidden,
                               num_max_tokens_per_rank, num_topk, self.config.local_experts,
                               precleared_dispatch=True, shared_expert=bool(num_shared_experts),
                               symmetric_inputs=True)
        self.buffer = self._comm.buf
        self.layout = self._comm.L
        for name in ('x', 'x_sf', 'topk_idx', 'topk_weights'):
            setattr(self, name, getattr(self._comm, name))
        device = self._device = self.buffer.device
        self.topk_idx.fill_(-1)
        self.topk_weights.zero_()
        self._starts = torch.arange(self.config.local_experts, dtype=torch.int32, device=device)
        self._starts.mul_(self._comm.cap_e)
        rows = self._comm.cap_e + (num_max_tokens_per_rank if num_shared_experts else 0)
        self._act = torch.empty((rows, intermediate_hidden // 2), dtype=torch.uint8, device=device)
        self._act_sf = torch.empty((rows, intermediate_hidden // 32), dtype=torch.uint8, device=device)
        self._counts = torch.empty(0, dtype=torch.int32, device=device)
        self._groups = torch.empty(rows + 1, dtype=torch.int32, device=device)
        self._empty_output = torch.empty((1, hidden), dtype=torch.bfloat16, device=device)
        self._weight_shapes = tuple(
            (((experts, intermediate_hidden // 256, hidden // 128, 2, 256, 64),
              (experts, hidden // 128, 2 * intermediate_hidden, 4)),
             ((experts, hidden // 512, intermediate_hidden // 128, 512, 64),
              (experts, intermediate_hidden // 128, hidden, 4)))
            for experts in (self.config.local_experts, 1)
        )
        self._k_major_w2_shape = (intermediate_hidden // 128, self.config.local_experts,
                                  hidden // 512, 512 * 64)
        self._stream_ptr = self._stream.cuda_stream
        self._libraries = {}
        self._pipelines = {}

    def destroy(self):
        if self.buffer is None:
            return
        torch.cuda.synchronize(self.buffer.device)
        self._pipelines.clear()
        self._libraries.clear()
        self._comm = self.buffer = self.group = None
        self.x = self.x_sf = self.topk_idx = self.topk_weights = None
        self._starts = self._act = self._act_sf = self._counts = self._groups = self._empty_output = None


def get_symm_buffer_for_gc_moe_ppu(group, num_experts, num_max_tokens_per_rank,
                                  num_topk, hidden, intermediate_hidden,
                                  activation='swiglu', num_shared_experts=0, *, cache_dir=None):
    return PPUGCSymmBuffer(group, num_experts, num_max_tokens_per_rank, num_topk,
                           hidden, intermediate_hidden, activation, num_shared_experts,
                           cache_dir=cache_dir)


def transform_weights_for_gc_moe_ppu(l1_weights, l2_weights, activation='swiglu',
                                     *, shared_recipe=None):
    """Prepare K32 MXFP4 panels, or expand K128 shared UE8M0 scales exactly."""
    if activation != 'swiglu':
        raise ValueError('PPU GC MoE supports swiglu')
    granularity = 32
    if shared_recipe is not None:
        recipe = tuple(shared_recipe)
        if recipe not in ((1, 1, 32), (1, 128, 128)):
            raise ValueError('expected K32 or K128 shared weight recipe')
        granularity = recipe[2]

    def unpack(pair):
        payload, scales = pair
        if (payload.dtype not in (torch.int8, torch.uint8) or payload.ndim not in (2, 3)
                or not payload.is_cuda or payload.device != scales.device
                or scales.dtype not in (torch.uint8, torch.int32)):
            raise ValueError('expected MXFP4 weight payload and UE8M0 scales on one device')
        payload = payload.contiguous().view(torch.uint8)
        scales = scales.contiguous().view(torch.uint8)
        k = 2 * payload.shape[-1]
        if k % granularity or scales.shape != (*payload.shape[:-1], k // granularity):
            raise ValueError('UE8M0 scale shape does not match the weight recipe')
        if granularity == 128:
            scales = scales.repeat_interleave(4, dim=-1)
        if payload.ndim == 2:
            payload, scales = payload.unsqueeze(0), scales.unsqueeze(0)
        return payload, scales

    w1, sfw1 = unpack(l1_weights)
    w2, sfw2 = unpack(l2_weights)
    if w1.device != w2.device or (shared_recipe is not None and len(w1) != 1):
        raise ValueError('weight pairs must use one device and cover one shared expert')
    p1, p2 = repack_weight_panels(w1, w2, k_major_w2=len(w1) > 1)
    return (p1, repack_sfw1(sfw1)), (p2, repack_sfw2(sfw2))


def _weight_addresses(pair, shapes, device, alternate_shape=None):
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        raise ValueError('call transform_weights_for_gc_moe_ppu before launching')
    payload, scales = pair
    if (not isinstance(payload, torch.Tensor) or not isinstance(scales, torch.Tensor)
            or payload.dtype != torch.uint8 or scales.dtype != torch.uint8
            or payload.device != device or scales.device != device
            or not payload.is_contiguous() or not scales.is_contiguous()
            or scales.shape != shapes[1]):
        raise ValueError('call transform_weights_for_gc_moe_ppu before launching')
    shape = payload.shape
    alternate = shape != shapes[0]
    if alternate and shape != alternate_shape:
        raise ValueError('call transform_weights_for_gc_moe_ppu before launching')
    payload_address, scale_address = payload.data_ptr(), scales.data_ptr()
    if (payload_address | scale_address) & 15:
        raise ValueError('call transform_weights_for_gc_moe_ppu before launching')
    return payload_address, scale_address, alternate


def ppu_fp4_fp4_gc_moe(y, l1_weights, l2_weights, sym_buffer,
                        cumulative_local_expert_recv_stats=None, recipe=_DEFAULT_RECIPE,
                        activation='swiglu', activation_clamp=None, fast_math=True,
                        num_tokens_bound=None, shared_l1_weights=None,
                        shared_l2_weights=None, shared_recipe=None):
    """Compute GC MoE from the packed activation and route views in sym_buffer."""
    b = sym_buffer
    if not isinstance(b, PPUGCSymmBuffer) or b.buffer is None:
        raise ValueError('expected a live PPUGCSymmBuffer')
    device = b._device
    bound = 0 if num_tokens_bound is None else operator.index(num_tokens_bound)
    if bound == 0:
        bound = b.num_max_tokens_per_rank
    if (recipe is not _DEFAULT_RECIPE and tuple(recipe) != _DEFAULT_RECIPE) or activation != 'swiglu':
        raise ValueError('PPU GC MoE requires recipe (1, 1, 32) and swiglu')
    clamp = math.inf if activation_clamp is None else float(activation_clamp)
    if math.isnan(clamp) or clamp < 0:
        raise ValueError('activation_clamp must be nonnegative')
    shared = shared_l1_weights is not None
    if (shared != (shared_l2_weights is not None) or shared > b.num_shared_experts
            or shared_recipe is not None and (not shared or tuple(shared_recipe) not in
                                              ((1, 1, 32), (1, 128, 128)))):
        raise ValueError('expected both shared weight pairs and shared workspace capacity')
    with _tensor_metadata_context():
        shape = y.shape
        output_address = y.data_ptr()
        if (len(shape) != 2 or shape[1] != b.hidden or y.device != device or y.dtype != torch.bfloat16
                or not y.is_contiguous() or output_address % 512
                or _get_current_device() != device.index):
            raise ValueError('expected a 512-byte aligned contiguous BF16 output on the workspace device')
        tokens = shape[0]
        if not 0 <= tokens <= bound <= b.num_max_tokens_per_rank:
            raise ValueError('num_tokens_bound must cover every rank and fit the workspace')
        w1_shape, w2_shape = b._weight_shapes[0]
        w1_address, s1_address, _ = _weight_addresses(l1_weights, w1_shape, device)
        w2_address, s2_address, k_major_w2 = _weight_addresses(
            l2_weights, w2_shape, device, b._k_major_w2_shape)
        addresses = w1_address, s1_address, w2_address, s2_address
        shared_weights = (shared_l1_weights, shared_l2_weights) if shared else None
        if shared:
            sw1_shape, sw2_shape = b._weight_shapes[1]
            sw1_address, ss1_address, _ = _weight_addresses(shared_l1_weights, sw1_shape, device)
            sw2_address, ss2_address, _ = _weight_addresses(shared_l2_weights, sw2_shape, device)
            addresses += sw1_address, ss1_address, sw2_address, ss2_address
        stats = cumulative_local_expert_recv_stats
        if stats is not None and (stats.shape != (b.config.local_experts,) or stats.dtype != torch.int32
                                  or stats.device != device or not stats.is_contiguous()):
            raise ValueError('stats must be contiguous int32 [local_experts] on the workspace device')
        stats_address = stats.data_ptr() if stats is not None else 0

    stream = (_get_current_raw_stream(device.index) if _get_current_raw_stream is not None
              else torch.cuda.current_stream(device).cuda_stream)
    if stream != b._stream_ptr or _is_stream_capturing():
        raise ValueError('use the workspace creation stream outside graph capture')

    key = b.config.schedule(bound, shared_expert=shared), shared, math.isfinite(clamp), bool(fast_math), k_major_w2
    output = y if tokens else b._empty_output
    if not tokens:
        output_address = output.data_ptr()
    signature = (*addresses, stats_address, output_address), clamp
    cached = b._pipelines.get(key)
    if cached is None:
        path = b._libraries.get(key)
        if path is None:
            _, _, roles = build_ppu_libraries(b.config, (bound,), cache_dir=b.cache_dir,
                                             shared_expert=shared, activation_clamp=key[2], fast_math=key[3],
                                             k_major_w2=k_major_w2)
            path = b._libraries[key] = str(roles[bound])
        w1, s1 = l1_weights
        w2, s2 = l2_weights
        args = (b.x, b.x_sf, b.topk_idx, b.topk_weights, addresses[0], s1, addresses[2], s2,
                b._starts, b._act, b._act_sf, b._counts, b._counts, b._groups, b.intermediate_hidden)
        role = Role12State(path, b._comm, *args, workers=1, weight_panels=(w1, w2),
                           stats=stats, shared_weights=shared_weights, activation_clamp=clamp)
        pipeline = PipelineState(str(b._pipeline_path), b._comm, None, *args, output, role12=role)
    else:
        pipeline = cached[1]
    if cached is None or cached[0] != signature:
        tensors = (*l1_weights, *l2_weights, stats, output)
        if shared:
            tensors += (*shared_l1_weights, *shared_l2_weights)
        for tensor in tensors:
            if tensor is not None:
                tensor.record_stream(b._stream)
        pipeline.bind_request(addresses, stats_address, output_address, clamp, tensors)
        b._pipelines[key] = (signature, pipeline)
    pipeline.output_rows = tokens or 1
    pipeline.launch(tokens, stream=b._stream_ptr)
