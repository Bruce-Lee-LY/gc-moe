"""Symmetric-buffer and MXFP4 weight layouts for PPU GC MoE."""

import torch


RANKS = 8


def comm_layout(H: int, T: int, topk: int, ranks: int = RANKS,
                *, grouped_experts: int, shared_expert: bool = False,
                precleared: bool = False, symmetric_inputs: bool = False) -> dict:
    """Byte offsets of every region inside one rank's symmetric buffer.

    Senders reserve rows in each receiver expert's partition. Precleared
    dispatch packs each expert's first four tiles into a common region
    and keeps a separate overflow partition per expert. The total capacity
    remains local_experts * ranks * cap rows. Scale rows are padded to
    the 256- or 512-byte transport width.
    Records contain {token, slot|source<<8, weight, expert}.
    Combine arena: token-major BF16 receive rows [T*topk, H]. SE stores its
    local BF16 output after the routed rows. The legacy pipeline allocates
    additional sender-partition capacity and records. The per-(owner, sender) partition bound is cap = T*topk.
    The total allocation fits in the int32 kernel offsets.
    """
    cap = T * topk
    hb, sb = H // 2, H // 32
    sfs = ((sb + 255) & ~255) if precleared else 512
    off = 0

    def nxt(sz):
        nonlocal off
        off = (off + sz + 255) & ~255

    d = {}
    if symmetric_inputs:
        d['x'] = off; nxt(T * hb)
        d['x_sf'] = off; nxt(T * sb)
        d['topk_idx'] = off; nxt(cap * 8)
        d['topk_weights'] = off; nxt(cap * 4)
    rows = ranks * cap * grouped_experts
    d['pool'] = off; nxt(rows * hb)
    d['sf'] = off;   nxt(rows * sfs)
    d['rec'] = off;  nxt(rows * 16)
    # Precleared dispatch uses this aligned region for per-expert readiness.
    d['dcnt'] = off; nxt((grouped_experts if precleared else ranks) * 4)
    # Independent per-source payload and metadata generations.
    d['dgen'] = off; nxt((2 if precleared else 1) * ranks * 4)
    d['rsvp_counts'] = off; nxt(grouped_experts * 4)
    if precleared:
        combine_rows = cap + (T if shared_expert else 0)
    else:
        combine_rows = max(ranks * cap, cap + T) if shared_expert else ranks * cap
    d['cbuf'] = off; nxt(combine_rows * H * 2)
    if shared_expert:
        d['shared_y'] = d['cbuf'] + cap * H * 2
    d['crec'] = off; nxt(0 if precleared else ranks * cap * 16)
    d['ccnt'] = off; nxt(ranks * 4)
    d['cgen'] = off; nxt(ranks * 4)
    nxt((16 << 20) + 64)  # Preserve the measured barrier offset.
    d['barr'] = off; nxt(64)          # device-barrier gen words (never reset)
    d['size'] = off
    d['cap'] = cap
    assert off < 2 ** 31, 'symmetric buffer must stay below 2GB (int32 offsets)'
    return d


def repack_weight_panels(w1, w2, *, w1_width=256, w2_width=512, k_major_w2=False):
    """Convert row-major uint8 MXFP4 weights to the role12 panel layout."""
    experts, hidden, inter_bytes = w2.shape
    inter = 2 * inter_bytes
    if (w1_width not in (128, 256) or w2_width not in (256, 512)
            or hidden % 128 or inter % 128 or hidden % w2_width or inter % w1_width
            or w1.shape != (experts, 2 * inter, hidden // 2)
            or w1.dtype != torch.uint8 or w2.dtype != torch.uint8):
        raise ValueError('role12 panels require matching MXFP4 weights and aligned H/I')
    w1_panel = (w1.view(experts, 2, inter // w1_width, w1_width, hidden // 128, 64)
                .permute(0, 2, 4, 1, 3, 5).contiguous())
    w2_tiles = w2.view(experts, hidden // w2_width, w2_width, inter // 128, 64)
    if k_major_w2:
        # The four-dimensional shape distinguishes the K-major layout
        # from legacy expert-major panels.
        w2_panel = w2_tiles.permute(3, 0, 1, 2, 4).contiguous().flatten(3)
    else:
        w2_panel = w2_tiles.permute(0, 1, 3, 2, 4).contiguous()
    return w1_panel, w2_panel


def repack_sfw1(w1_s):
    """[E, 2I, H/32] u8 -> [E, nt, 2I, 4B] tile-major (nt = H/128).

    Each (e, tile) block is one contiguous 2I*4B slab in MMA lane order.
    """
    E, twoI, ns = w1_s.shape
    nt = ns // 4
    return _pair_scales(w1_s.view(E, twoI, nt, 4).permute(0, 2, 1, 3).contiguous())


def repack_sfw2(w2_s):
    """[E, H, I/32] u8 -> [E, nt, H, 4B] tile-major in MMA lane order."""
    E, H, ns = w2_s.shape
    nt = ns // 4
    return _pair_scales(w2_s.view(E, H, nt, 4).permute(0, 2, 1, 3).contiguous())


def _pair_scales(scales):
    # Interleave the halves of each 16-row tile without changing UE8M0 bytes.
    words = scales.view(torch.int32).reshape(*scales.shape[:-2], -1, 16)
    lower, upper = words[..., :8], words[..., 8:]
    first = (lower & 0xffff) | ((upper & 0xffff) << 16)
    second = ((lower >> 16) & 0xffff) | (upper & -65536)
    return torch.cat((first, second), dim=-1).view(torch.uint8).reshape_as(scales)
