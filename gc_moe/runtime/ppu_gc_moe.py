"""Workspace ownership and stream resources for Sail GC MoE."""

import os
from pathlib import Path

import torch
import torch.distributed as dist

from ..jit_kernels.impls.ppu_gc_moe import load
from ..layout.ppu_gc_moe import LOCAL_RANKS, MAX_RANKS, MAX_TOKENS, PPUBufferLayout, check_weight_dimensions


def _check_peer_status(group, error):
    errors = [None] * group.size()
    dist.all_gather_object(errors, error, group=group)
    for rank, message in enumerate(errors):
        if message is not None:
            raise RuntimeError(f'PPU GC MoE requires directly mapped ICN peers (rank {rank}): {message}')


class PPUGCSymmBuffer:
    """Own caller-staged inputs, peer mailboxes and local decode scratch.

    All ranks issue calls in the same order with the same token bound. Serialize
    every use of this workspace, including input staging, on ordered streams.
    Warm up before graph capture and keep all captured buffers and weights alive.
    """

    def __init__(self, group, num_experts, num_max_tokens_per_rank, num_topk,
                 hidden, intermediate_hidden, activation='swiglu', num_shared_experts=0,
                 *, cache_dir=None, graph_safe=False, routed_precision=None, shared_fp8_fp8_ptpc=None,
                 routed_fp8=None, shared_fp8=None):
        if routed_precision is None:
            routed_precision = 'fp8_fp4' if routed_fp8 else 'fp4_fp4'
        elif routed_fp8 is not None and (routed_precision == 'fp8_fp4') != routed_fp8:
            raise ValueError('conflicting routed precision settings')
        if routed_precision not in ('fp4_fp4', 'fp8_fp4'):
            raise ValueError('routed_precision must be fp4_fp4 or fp8_fp4')
        if shared_fp8_fp8_ptpc is None:
            shared_fp8_fp8_ptpc = False if shared_fp8 is None else shared_fp8
        elif shared_fp8 is not None and shared_fp8_fp8_ptpc != shared_fp8:
            raise ValueError('conflicting shared precision settings')
        check_weight_dimensions(hidden, intermediate_hidden)
        capacity = num_max_tokens_per_rank
        ranks = group.size()
        if (not 1 <= capacity <= MAX_TOKENS or not 1 <= ranks <= MAX_RANKS
                or not ranks <= num_experts <= 65535 or num_experts % ranks
                or not 1 <= num_topk <= min(16, num_experts)
                or activation != 'swiglu' or num_shared_experts not in (0, 1)
                or bool(num_shared_experts) != shared_fp8_fp8_ptpc):
            raise ValueError('invalid decode capacity, EP size, activation or shared-expert configuration')
        if torch.cuda.is_current_stream_capturing():
            raise ValueError('allocate the workspace outside graph capture')
        self._aiu = int(os.environ.get('GC_MOE_PPU_AIU', '1'))
        if self._aiu not in (0, 1):
            raise ValueError('GC_MOE_PPU_AIU must be 0 or 1')
        self._m_tiles = int(os.environ.get('GC_MOE_PPU_M_TILES', '0'))
        if self._m_tiles not in (0, 1, 2, 3):
            raise ValueError('GC_MOE_PPU_M_TILES must be 0 (automatic), 1, 2 or 3')
        self.group, self.rank_idx = group, group.rank()
        self.num_experts, self.num_topk = num_experts, num_topk
        self.hidden, self.intermediate_hidden = hidden, intermediate_hidden
        self.num_max_tokens_per_rank = capacity
        self.num_shared_experts = num_shared_experts
        self.routed_precision = routed_precision
        routed_fp8_fp4 = routed_precision == 'fp8_fp4'
        self._device = device = torch.device('cuda', torch.cuda.current_device())
        self._library = load(Path(cache_dir or Path.home() / '.cache/gc_moe/ppu').expanduser().resolve(),
                             hidden, intermediate_hidden, num_experts, num_topk, max(LOCAL_RANKS, ranks),
                             routed_precision)

        self.layout = layout = PPUBufferLayout(ranks, num_experts, capacity, num_topk, hidden, routed_precision)
        if ranks > 1:
            import torch.distributed._symmetric_memory as symm_mem

            self.buffer = symm_mem.empty(layout.nbytes, dtype=torch.uint8, device=device)
            try:
                self._handle = symm_mem.rendezvous(self.buffer, group=group)
                peers = self._handle.buffer_ptrs
                error = None
                if ranks > LOCAL_RANKS and (len(set(peers)) != ranks or len(peers) != ranks
                                            or any(not ptr or ptr % 16 for ptr in peers)
                                            or peers[self.rank_idx] != self.buffer.data_ptr()):
                    error = 'symmetric memory rendezvous did not provide all peer mappings'
            except (RuntimeError, AttributeError) as exc:
                if ranks <= LOCAL_RANKS:
                    raise
                error = str(exc)
            if ranks > LOCAL_RANKS:
                _check_peer_status(group, error)
        else:
            self.buffer = torch.empty(layout.nbytes, dtype=torch.uint8, device=device)
            self._handle = None
            peers = [self.buffer.data_ptr()]
        self.buffer[:layout.header_bytes].zero_()
        for view in layout.views:
            tensor = self.buffer.narrow(0, view.offset, view.nbytes).view(view.dtype).view(view.shape)
            setattr(self, view.name, tensor)
        self._offsets = tuple(view.offset for view in layout.views)
        self.topk_idx.fill_(-1)
        self.topk_weights.zero_()
        self._peers = torch.tensor(peers, dtype=torch.int64, device=device)
        if ranks > LOCAL_RANKS:
            self._probe_icn_peers()

        max_routes = layout.max_routes
        local_experts = layout.local_experts
        max_tiles = (max_routes + 15) // 16 + local_experts
        # Every expert can own a partially filled tile in addition to the live rows.
        can_use_large_pages = ranks > LOCAL_RANKS and (not routed_fp8_fp4 or hidden <= 2 * intermediate_hidden)
        max_page_panels = 8 if can_use_large_pages else 6
        rows = (max_tiles + local_experts * ((self._m_tiles or max_page_panels) - 1)) * 16
        activation_width = hidden if routed_fp8_fp4 else hidden // 2
        scale_width = 1 if routed_fp8_fp4 else hidden // 32
        scale_dtype = torch.float32 if routed_fp8_fp4 else torch.uint8
        mid_width = 2 * intermediate_hidden if routed_fp8_fp4 else intermediate_hidden // 2
        mid_scale_width = 1 if routed_fp8_fp4 else intermediate_hidden // 32
        self._scratch = tuple(torch.empty(shape, device=device, dtype=dtype) for shape, dtype in (
            ((rows, activation_width), torch.uint8), ((rows, scale_width), scale_dtype),
            ((rows,), torch.int64), ((max_tiles, 3), torch.int32), ((1,), torch.int32),
            ((rows, mid_width), torch.uint8), ((rows, mid_scale_width), scale_dtype),
        ))
        self.shared_x = self.shared_x_sf = None
        self._shared_scratch = ()
        if shared_fp8_fp8_ptpc:
            self.shared_x = torch.empty((capacity, hidden), dtype=torch.float8_e4m3fn, device=device)
            self.shared_x_sf = torch.empty((capacity, 1), dtype=torch.float32, device=device)
            self._shared_scratch = tuple(torch.empty(shape, device=device, dtype=dtype) for shape, dtype in (
                ((capacity, intermediate_hidden), torch.bfloat16), ((capacity + 1, intermediate_hidden), torch.uint8),
                ((capacity, 1), torch.float32), ((capacity, hidden), torch.bfloat16),
            ))
            # The extra row holds the five counters used by the small SE queue.
            self._shared_scratch[1][-1, :20].zero_()
        self._units = torch.cuda.get_device_properties(device).multi_processor_count
        # W4A8 reads routing weights from the source rank during combine.
        self._grouped_weights = None if routed_fp8_fp4 else torch.empty(rows, dtype=torch.float32, device=device)
        pull_routes = ranks >= LOCAL_RANKS and self._aiu and local_experts <= 64
        # The push dispatcher indexes larger batches to keep its per-CTA
        # route scan and dynamic shared-memory allocation bounded.
        route_index = (torch.empty((local_experts, max_routes), dtype=torch.int32, device=device)
                       if not pull_routes and max_routes >= 3072 else None)
        if pull_routes:
            # Reserve for worst-case skew at the smallest page size.
            overflow_pages = (max_routes + 15) // 16
            route_counts = torch.zeros(((2 + overflow_pages) * local_experts,), dtype=torch.int32, device=device)
        elif route_index is not None:
            route_counts = torch.empty((local_experts,), dtype=torch.int32, device=device)
        else:
            route_counts = None
        self._route_index = (route_index, route_counts)
        self._finished_first, self._down_tiles = (
            torch.empty(max_tiles, dtype=torch.int32, device=device) for _ in range(2)
        )
        self._silu_table = torch.empty((2, 65536), dtype=torch.float32, device=device)
        stream = torch.cuda.current_stream(device)
        status = self._library.gc_moe_init_silu(self._silu_table.data_ptr(), stream.cuda_stream)
        if status:
            raise RuntimeError(f'PPU SiLU table initialization failed with HGGC status {status}')
        self._se_pipeline = None
        if shared_fp8_fp8_ptpc and self._aiu:
            side = torch.cuda.Stream(device=device)
            first, last = torch.cuda.Event(), torch.cuda.Event()
            first.record()
            last.record()
            self._se_pipeline = (side, first, last)
        stream.synchronize()
        dist.barrier(group)

    @property
    def routed_fp8(self):
        """Compatibility view of the routed activation mode."""
        return self.routed_precision == 'fp8_fp4'

    def _probe_icn_peers(self):
        stream = torch.cuda.current_stream(self._device)
        stream.synchronize()
        dist.barrier(self.group)
        status = self._library.gc_moe_probe_peers(self._peers.data_ptr(), self.rank_idx, stream.cuda_stream)
        error = f'peer store probe failed with HGGC status {status}' if status else None
        try:
            stream.synchronize()
        except RuntimeError as exc:
            error = str(exc)
        _check_peer_status(self.group, error)
        ranks = self.group.size()
        written = self.buffer[:ranks * 16].view(torch.int32).view(ranks, 4).cpu()
        values = torch.arange(1, ranks + 1, dtype=torch.int32, device='cpu')[:, None].expand_as(written)
        _check_peer_status(self.group, None if torch.equal(written, values) else 'peer store contents differ')
        self.buffer[:self.layout.header_bytes].zero_()

    def destroy(self):
        if self.buffer is None:
            return
        torch.cuda.synchronize(self._device)
        self.buffer = self._handle = self.group = self._peers = None
        self._scratch = self._shared_scratch = ()
        self.x = self.x_sf = self.topk_idx = self.topk_weights = self._returns = self._inbox = None
        self._inbox_ids = self._grouped_weights = None
        self.shared_x = self.shared_x_sf = None
        self._silu_table = None
        self._route_index = ()
        self._finished_first = self._down_tiles = None
        self._se_pipeline = None
