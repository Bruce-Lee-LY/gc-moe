"""Host bindings and workspace state for the native PPU GC MoE pipeline."""

from __future__ import annotations

import ctypes
import struct

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from ..layout.ppu_gc_moe import comm_layout


_I = ctypes.c_int
_P = ctypes.c_void_p
_LL = ctypes.c_longlong


class CommState:
    """Per-rank symmetric buffer for one shape and dispatch protocol.

    All ranks use the same protocol. Precleared requests reuse counters
    prepared by the preceding request and run on one ordered stream.
    """

    def __init__(self, lib_path: str, group, rank: int, ranks: int,
                 H: int, T: int, topk: int, e_local: int, *,
                 precleared_dispatch: bool = False, shared_expert: bool = False,
                 symmetric_inputs: bool = False):
        self.lib = ctypes.CDLL(lib_path)
        lib = self.lib
        # argtypes mirror the exact C order (pointers interleaved with ints)
        lib.gc_comm_dispatch_push.argtypes = (
            [_P, ctypes.POINTER(_LL)] + [_I] * 2 + [_P] * 4
            + [_I] * 10 + [_P] * 3 + [_I] * 1 + [_P] * 7)
        lib.gc_comm_combine_scatter.argtypes = (
            [_P, ctypes.POINTER(_LL)] + [_I] * 2 + [_P] * 5
            + [_I] * 7 + [_P] * 3 + [_I] * 3)
        # sym, po, ranks, my, gen, barr_off, cursors, done, e_counts,
        # e_local, dcnt, dgen, ccnt, cgen
        lib.gc_comm_reset_barrier.argtypes = (
            [_P, ctypes.POINTER(_LL)] + [_I] * 4 + [_P] * 3 + [_I] * 5)
        # e_counts, e_local, cap_e, rows_g, out_src, out_t, out_slot,
        # starts, out_R, rec_src_c, rec_t_c, rec_slot_c
        lib.gc_comm_finalize.argtypes = [_P, _I, _I, _I] + [_P] * 8
        self._has_finalize_reset = hasattr(lib, 'gc_comm_finalize_reset')
        if self._has_finalize_reset:
            lib.gc_comm_finalize_reset.argtypes = lib.gc_comm_finalize.argtypes + [_P, _I]
            lib.gc_comm_finalize_reset.restype = None
        self._has_finalize_groups = hasattr(lib, 'gc_comm_finalize_groups')
        if self._has_finalize_groups:
            lib.gc_comm_finalize_groups.argtypes = lib.gc_comm_finalize_reset.argtypes + [_P, _I]
            lib.gc_comm_finalize_groups.restype = None

        self.group, self.rank, self.ranks = group, rank, ranks
        self.H, self.T, self.topk, self.e_local = H, T, topk, e_local
        self.grouped_remote = True
        self.precleared_dispatch = precleared_dispatch
        self.shared_expert = shared_expert
        if shared_expert and not precleared_dispatch:
            raise ValueError('shared expert requires precleared dispatch')
        self.L = comm_layout(H, T, topk, ranks, grouped_experts=e_local,
                             shared_expert=shared_expert, precleared=precleared_dispatch,
                             symmetric_inputs=symmetric_inputs)
        # per-local-expert output partition bound (unconditional worst case:
        # every row to one expert; rows-per-expert <= total rows)
        self.cap_e = ranks * T * topk

        # symmetric allocation + rendezvous (SAILSHMEM) — done ONCE
        self.buf = symm_mem.empty(self.L['size'], dtype=torch.uint8, device='cuda')
        self.handle = symm_mem.rendezvous(self.buf, group=group)
        ptrs = [int(p) for p in self.handle.buffer_ptrs]
        my = ptrs[rank]
        # The role12 host launcher copies a fixed eight-entry peer table.
        self.peer_offs = (ctypes.c_longlong * 8)(
            *[ptrs[r] - my for r in range(ranks)])
        self.buf32 = self.buf.view(torch.int32)
        if symmetric_inputs:
            for name, shape, dtype, itemsize in (
                ('x', (T, H // 2), torch.uint8, 1),
                ('x_sf', (T, H // 32), torch.uint8, 1),
                ('topk_idx', (T, topk), torch.int64, 8),
                ('topk_weights', (T, topk), torch.float32, 4),
            ):
                rows, cols = shape
                setattr(self, name, self.buf.narrow(0, self.L[name], rows * cols * itemsize)
                        .view(dtype).view(shape))

        # local (non-symmetric) scratch + outputs.  dispatch outputs are
        # GROUPED BY LOCAL EXPERT: row index = e*cap_e + i, so
        # row_starts[e] = e*cap_e and row_counts = out_counts (kernel-written
        # per-expert atomic cursors) -- no python argsort/gather/scan.
        rows_g = self.cap_e * e_local
        dev = self.buf.device
        self.cursors = torch.zeros(ranks, dtype=torch.int32, device=dev)
        self.done = torch.zeros(2, dtype=torch.int32, device=dev)
        start = self.L['rsvp_counts'] // 4
        self.out_counts = self.buf32[start:start + e_local]
        legacy_rows = 0 if precleared_dispatch else rows_g
        self.out_pool = torch.empty(legacy_rows, H // 2, dtype=torch.uint8, device=dev)
        self.out_sf = torch.empty(legacy_rows, H // 32, dtype=torch.uint8, device=dev)
        self.out_t = torch.empty(legacy_rows, dtype=torch.int32, device=dev)
        self.out_slot = torch.empty(legacy_rows, dtype=torch.int32, device=dev)
        self.out_src = torch.empty(legacy_rows, dtype=torch.int32, device=dev)
        self.out_w = torch.empty(legacy_rows, dtype=torch.float32, device=dev)
        self.out_e = torch.empty(T * topk if precleared_dispatch else rows_g,
                                 dtype=torch.int32, device=dev)
        self.out_rows = torch.zeros(1, dtype=torch.int32, device=dev)
        # device-barrier generation (monotonic; barr words zeroed ONCE here
        # -- symm_mem.empty is uninitialized and reset() deliberately skips
        # the barr region, so garbage words would never satisfy >= want)
        self.barr_gen = 0
        self.buf32[self.L['barr'] // 4:self.L['barr'] // 4 + 16].zero_()
        # grouped-finalize outputs (compact rec arrays + starts + [R, ngrp])
        self.starts = torch.zeros(e_local + int(shared_expert), dtype=torch.int32, device=dev)
        if precleared_dispatch:
            # FC1 group descriptors are uint4. One group per received row
            # is a safe bound even when all routes are concentrated.
            groups = self.cap_e + (T if shared_expert else 0) + 1
            self.compact_src = torch.zeros(groups, dtype=torch.int32, device=dev)
            self.compact_t = torch.zeros(4 * groups, dtype=torch.int32, device=dev)
            self.compact_slot = torch.zeros_like(self.starts)
        else:
            self.compact_src = torch.zeros(rows_g + 1, dtype=torch.int32, device=dev)
            self.compact_t = torch.zeros_like(self.compact_src)
            self.compact_slot = torch.zeros_like(self.compact_src)
        self.compact_rn = torch.zeros(3, dtype=torch.int32, device=dev)
        self.reset()
        torch.cuda.synchronize()
        dist.barrier(group=group)

    # ---- per-call reset of every ctl word + scratch -----------------------
    def reset(self):
        b, L, n = self.buf32, self.L, self.ranks
        for k in ('dcnt', 'dgen', 'ccnt', 'cgen'):
            o = L[k] // 4
            count = n
            if self.precleared_dispatch:
                if k == 'dcnt':
                    count = self.e_local
                elif k == 'dgen':
                    count = 2 * n
            b[o:o + count].zero_()
        self.cursors.zero_()
        self.done.zero_()
        self.out_counts.zero_()


class Role12State:
    """Role12 with caller-provided MXFP4 activations.

    `x_packed` and `x_sf` are contiguous uint8 tensors with shapes
    [rows, H/2] and [rows, H/32]. Both routed and shared experts consume
    these buffers as input. Append shared weights as the last weight group
    and reserve one additional FC1 activation row per local token of capacity.
    """

    def __init__(self, path, comm, x_packed, x_sf, topk_idx, topk_w,
                 w1_ptr, sfw1, w2_ptr, sfw2, x_starts, act, act_sf,
                 out_rows, w1_cnt, unused, inter, ncta=312, workers=8,
                 *, weight_panels=None, stats=None, shared_weights=None,
                 activation_clamp=float('inf')):
        rows = x_packed.shape[0] if x_packed.ndim == 2 else 0
        if (x_packed.shape != (rows, comm.H // 2)
                or x_sf.shape != (rows, comm.H // 32)
                or x_packed.dtype != torch.uint8 or x_sf.dtype != torch.uint8
                or not x_packed.is_cuda or x_packed.device != x_sf.device
                or x_packed.device != comm.buf.device
                or not x_packed.is_contiguous() or not x_sf.is_contiguous()
                or x_packed.data_ptr() % 512 or x_sf.data_ptr() % 16):
            raise ValueError('role12 requires aligned packed MXFP4 [rows,H/2] and UE8M0 [rows,H/32] input')
        for routes, dtypes in ((topk_idx, (torch.int32, torch.int64)), (topk_w, (torch.float32,))):
            if (routes.ndim != 2 or routes.shape[1] != comm.topk or routes.dtype not in dtypes
                    or routes.device != x_packed.device or not routes.is_contiguous()):
                raise ValueError('role12 requires contiguous int32/int64 IDs and float32 weights [rows,topk]')
        if stats is not None and (stats.shape != (comm.e_local,) or stats.dtype != torch.int32
                                  or stats.device != x_packed.device or not stats.is_contiguous()):
            raise ValueError('stats must be contiguous int32 [local_experts] on the input device')
        self.input_rows = min(rows, topk_idx.shape[0], topk_w.shape[0])
        self.device = x_packed.device
        self.lib = ctypes.CDLL(path)
        has_shared_expert = hasattr(self.lib, 'gc_moe_role12_shared_expert')
        self.has_shared_expert = has_shared_expert
        if (has_shared_expert and not comm.shared_expert
                or shared_weights is not None and not has_shared_expert):
            raise ValueError('role12 library and shared expert workspace differ')
        experts = comm.e_local + int(has_shared_expert)
        weight_experts = comm.e_local if shared_weights is not None else experts
        fc2_scale_bytes = weight_experts * comm.H * inter // 32
        for scales, size in ((sfw1, 2 * fc2_scale_bytes), (sfw2, fc2_scale_bytes)):
            if scales.dtype != torch.uint8 or scales.numel() != size or not scales.is_contiguous():
                raise ValueError('role12 requires contiguous UE8M0 weight scales for every expert')
        panel_layout = getattr(self.lib, 'gc_moe_role12_weight_panel', None)
        if panel_layout:
            panel_layout.argtypes = []
            panel_layout.restype = ctypes.c_int
        panel_format = panel_layout() if panel_layout else 0
        if bool(panel_format) != (weight_panels is not None):
            raise ValueError('role12 library and weight layout differ')
        if panel_format:
            w1_panel, w2_panel = weight_panels
            w1_ptr, w2_ptr = w1_panel.data_ptr(), w2_panel.data_ptr()
            w2_experts = w2_panel.shape[1] if panel_format == 2 else len(w2_panel)
            if len(w1_panel) != weight_experts or w2_experts != weight_experts:
                raise ValueError('weight panels do not match the expert count')
        if has_shared_expert and (x_starts.numel() != comm.e_local
                                  or min(act.shape[0], act_sf.shape[0]) < comm.cap_e + comm.T):
            raise ValueError('shared expert requires routed input starts and extra FC1 rows')
        self.entry = self.lib.gc_moe_role12
        self.entry.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                               ctypes.POINTER(ctypes.c_int),
                               ctypes.POINTER(ctypes.c_longlong)]
        self.entry.restype = None
        self.entry_start_reset = getattr(self.lib, 'gc_moe_role12_with_start_reset', None)
        if self.entry_start_reset is not None:
            self.entry_start_reset.argtypes = self.entry.argtypes + [
                ctypes.POINTER(ctypes.c_int), ctypes.c_int]
            self.entry_start_reset.restype = None
        if comm.precleared_dispatch and (
                self.entry_start_reset is None
                or not hasattr(self.lib, 'gc_moe_role12_precleared_dispatch')):
            raise ValueError('precleared dispatch requires a matching MXFP4 role12 library')
        self.lib.gc_moe_role12_occ.restype = ctypes.c_int
        self.lib.gc_moe_role12_occ.argtypes = []
        try:
            grouped = self.lib.gc_moe_role12_grouped_remote
            grouped.argtypes = []
            grouped.restype = ctypes.c_int
            self.grouped_remote = bool(grouped())
        except AttributeError:
            self.grouped_remote = False
        if self.grouped_remote and not comm.grouped_remote:
            raise ValueError('grouped-remote library requires GC_GROUPED_REMOTE=1 before CommState')
        self.w1_storage = w1_cnt
        source_words = 2 * comm.e_local * comm.ranks
        if self.grouped_remote and w1_cnt.numel() < source_words:
            self.w1_storage = torch.empty(source_words, dtype=torch.int32, device='cuda')
        self.group_storage = unused
        try:
            group_tile = self.lib.gc_moe_role12_group_tile
            group_tile.restype = ctypes.c_int
            group_tile.argtypes = []
            group_capacity = comm.cap_e + (comm.T if comm.shared_expert else 0) + 1
            tile = group_tile()
            if tile:
                if unused.numel() < group_capacity:
                    self.group_storage = torch.empty(group_capacity, dtype=torch.int32, device='cuda')
                counter_capacity = experts * ((comm.cap_e + tile - 1) // tile)
                if self.w1_storage.numel() < counter_capacity:
                    self.w1_storage = torch.empty(
                        max(source_words, counter_capacity), dtype=torch.int32, device='cuda')
        except AttributeError:
            pass
        self.scratch = torch.zeros(max(8, comm.e_local + 4), dtype=torch.int32, device='cuda')
        sym = comm.buf.data_ptr()
        # Precleared dispatch writes activations directly to the symmetric
        # pool. The legacy pool pointer can carry per-expert receive counts.
        pool = sym + comm.L['dcnt'] if comm.precleared_dispatch else comm.out_pool.data_ptr()
        # Keep this order in sync with role12::Ptr.
        addresses = [
            x_packed.data_ptr(), x_sf.data_ptr(), topk_idx.data_ptr(), topk_w.data_ptr(),
            sym, comm.cursors.data_ptr(), comm.done.data_ptr(),
            comm.out_counts.data_ptr(), pool, comm.out_sf.data_ptr(),
            comm.out_t.data_ptr(), comm.out_slot.data_ptr(), comm.out_src.data_ptr(),
            comm.out_w.data_ptr(), comm.out_e.data_ptr(), comm.starts.data_ptr(),
            comm.compact_rn.data_ptr(), comm.compact_t.data_ptr(),
            comm.compact_slot.data_ptr(), comm.compact_src.data_ptr(),
            w1_ptr, sfw1.data_ptr(), w2_ptr, sfw2.data_ptr(), x_starts.data_ptr(),
            act.data_ptr(), act_sf.data_ptr(), out_rows.data_ptr(), self.w1_storage.data_ptr(),
            self.scratch.data_ptr(), self.group_storage.data_ptr(),
            stats.data_ptr() if stats is not None else None,
        ]
        if shared_weights is None:
            addresses.extend([None] * 4)
        else:
            addresses.extend(t.data_ptr() for pair in shared_weights for t in pair)
        addresses.append(None)  # STREAM is populated at submission.
        self.ptrs = (ctypes.c_void_p * 37)(*addresses)
        layout = comm.L
        # Matches role12::Int exactly. GENERATION index15 counts only this
        # unified launch; CommState's peer barrier generation is independent.
        self.ints = (ctypes.c_int * 20)(
            comm.ranks, comm.rank, comm.T, comm.topk, comm.H, inter, comm.e_local,
            layout['cap'], comm.cap_e, layout['pool'], layout['sf'], layout['rec'],
            layout['rsvp_counts'] if self.grouped_remote else layout['dcnt'],
            layout['dgen'], self.w1_storage.numel(), 0, ncta, workers,
            int(topk_idx.dtype == torch.int64),
            struct.unpack('=i', struct.pack('=f', activation_clamp))[0])
        self.peer_offs = comm.peer_offs
        # Retain every Python owner of the pointers passed to native code.
        self.owners = (comm, x_packed, x_sf, topk_idx, topk_w,
                       x_starts, act, act_sf, out_rows, w1_cnt, self.w1_storage,
                       unused, self.group_storage)
        self.request_tensors = (sfw1, sfw2, weight_panels, stats, shared_weights)


_initialized_tables = set()


class PipelineState:
    def __init__(self, path, comm, math_entry, x_packed, x_sf, topk_idx, topk_w,
                 w1_ptr, sfw1, w2_ptr, sfw2, x_starts, act, act_sf,
                 out_rows, w1_cnt, unused, inter, output, ncta=312, role12=None):
        self.lib = ctypes.CDLL(path)
        if comm.precleared_dispatch and (
                role12 is None or not hasattr(self.lib, 'gc_moe_pipeline_mxfp4_input')):
            raise ValueError('precleared dispatch requires its packed MXFP4 pipeline')
        try:
            local_reset = self.lib.gc_moe_pipeline_local_combine_reset
            local_reset.restype = ctypes.c_int
            local_reset.argtypes = []
            needs_local_reset = int(local_reset())
        except AttributeError:
            needs_local_reset = False
        if needs_local_reset:
            try:
                supported = comm.lib.gc_comm_local_combine_reset
                supported.restype = ctypes.c_int
                supported.argtypes = []
                supported = int(supported())
            except AttributeError:
                supported = False
            if supported < needs_local_reset:
                raise ValueError('local combine-reset host requires matching opt-in comm library')
            if needs_local_reset >= 2 and comm.done.numel() < 2:
                raise ValueError('fused combine-reset requires a two-word done allocation')
        self.entry = (self.lib.gc_moe_pipeline_role12 if role12
                      else self.lib.gc_moe_pipeline)
        output_address = output.data_ptr()
        role_launcher = role12 and (role12.entry_start_reset or role12.entry)
        combine_entry = None
        if role12 and role12.entry_start_reset is not None:
            self.entry = getattr(self.lib, 'gc_moe_pipeline_role12_start_reset', None)
            if self.entry is None or needs_local_reset != 2:
                raise ValueError('role12 start reset requires the matching mode2 host library')
            combine_entry = getattr(self.lib, 'gc_moe_pipeline_role12_combine', None)
            if (combine_entry is not None and comm.precleared_dispatch
                    and output_address and output_address % 512 == 0):
                combine_role = getattr(role12.lib, 'gc_moe_role12_with_combine', None)
                if combine_role is not None:
                    combine_role.argtypes = role_launcher.argtypes + [ctypes.c_void_p]
                    combine_role.restype = None
                    self.entry = combine_entry
                    role_launcher = combine_role
        self.entry.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_void_p]
        self.entry.restype = None
        if role12 and role12.has_shared_expert and self.entry != combine_entry:
            raise ValueError('shared expert requires role3 combine and a 512-byte aligned output')
        addresses = [
            x_packed.data_ptr(), x_sf.data_ptr(), topk_idx.data_ptr(), topk_w.data_ptr(),
            comm.buf.data_ptr(), comm.cursors.data_ptr(), comm.done.data_ptr(),
            comm.out_counts.data_ptr(), comm.out_pool.data_ptr(), comm.out_sf.data_ptr(),
            comm.out_t.data_ptr(), comm.out_slot.data_ptr(), comm.out_src.data_ptr(),
            comm.out_w.data_ptr(), comm.out_e.data_ptr(), comm.starts.data_ptr(),
            comm.compact_rn.data_ptr(), comm.compact_t.data_ptr(),
            comm.compact_slot.data_ptr(), comm.compact_src.data_ptr(),
            w1_ptr, sfw1.data_ptr(), w2_ptr, sfw2.data_ptr(), x_starts.data_ptr(),
            act.data_ptr(), act_sf.data_ptr(), out_rows.data_ptr(), w1_cnt.data_ptr(),
            None, unused.data_ptr()]
        self.ptrs = (ctypes.c_void_p * 31)(*addresses)
        layout = comm.L
        self.ints = (ctypes.c_int * 18)(
            comm.ranks, comm.rank, comm.T, comm.topk, comm.H, inter, comm.e_local,
            layout['cap'], comm.cap_e, layout['pool'], layout['sf'], layout['rec'],
            layout['dcnt'], layout['dgen'], w1_cnt.numel(), 0, ncta,
            getattr(math_entry, 'group_tile', 0))
        self.comm_ints = (ctypes.c_int * 5)(
            layout['barr'], layout['ccnt'], layout['cgen'], layout['cbuf'], layout['crec'])
        finalize = (comm.lib.gc_comm_finalize_groups if getattr(math_entry, 'group_tile', 0)
                    else comm.lib.gc_comm_finalize_reset)
        self.functions = (ctypes.c_void_p * 6)(*[
            ctypes.cast(f, ctypes.c_void_p).value for f in
            (comm.lib.gc_comm_reset_barrier, comm.lib.gc_comm_dispatch_push,
             finalize, math_entry, comm.lib.gc_comm_combine_scatter)])
        if output.dtype != torch.bfloat16:
            raise ValueError('PPU GC MoE requires a BF16 output buffer')
        if role12 and (output.ndim != 2 or output.shape[1] != comm.H
                       or not output.is_contiguous() or output.device != role12.device):
            raise ValueError('role12 requires contiguous BF16 output [rows,H] on the input device')
        self.output = ctypes.c_void_p(output_address)
        self.output_rows = output.shape[0] if role12 else 0
        self.role12 = role12
        if role12:
            self.ptrs, self.ints = role12.ptrs, role12.ints
            self.functions[3] = ctypes.cast(role_launcher, ctypes.c_void_p).value
        if role12:
            initialize = getattr(role12.lib, 'gc_moe_role12_init_gate_table', None)
            key = (role12.lib._handle, torch.cuda.current_device())
            if initialize is not None and key not in _initialized_tables:
                initialize.argtypes = []
                initialize.restype = ctypes.c_int
                status = initialize()
                if status:
                    raise RuntimeError(f'gate table initialization failed: {status}')
                _initialized_tables.add(key)
        if role12:
            if comm.precleared_dispatch:
                self.functions[5] = ctypes.cast(comm.lib.gc_comm_combine_precleared, ctypes.c_void_p).value
            else:
                self.functions[4] = ctypes.cast(comm.lib.gc_comm_combine_direct, ctypes.c_void_p).value
        self.comm = comm
        self.owners = (comm, math_entry, x_packed, x_sf, topk_idx, topk_w,
                       x_starts, act, act_sf, out_rows, w1_cnt, unused, role12)
        self.request_tensors = (sfw1, sfw2, output)

    def bind_request(self, weights, stats, output, clamp, tensors):
        """Update the request addresses while retaining the workspace's launch state."""
        self.ptrs[20:24] = weights[:4]  # W1, SFW1, W2, SFW2
        self.ptrs[31] = stats  # STATS
        if self.role12.has_shared_expert:
            self.ptrs[32:36] = weights[4:]  # SE_W1, SE_SFW1, SE_W2, SE_SFW2
        self.output.value = output
        self.ints[19] = struct.unpack('=i', struct.pack('=f', clamp))[0]  # CLAMP_BITS
        self.role12.request_tensors = None
        self.request_tensors = tensors

    def launch(self, tokens, *, stream=None):
        if self.role12 and not 0 <= tokens <= min(self.comm.T, self.role12.input_rows, self.output_rows):
            raise ValueError('token count exceeds input, output or workspace capacity')
        self.ints[2] = tokens
        if self.role12:
            self.ints[15] += 1
            self.ptrs[-1] = stream
        first_generation = self.comm.barr_gen + 1
        self.comm.barr_gen += 2
        self.entry(self.ptrs, self.ints, self.comm.peer_offs, self.comm_ints,
                   self.functions, first_generation, self.output)
