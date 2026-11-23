"""GC MoE block, ring and role-local shared-memory sizing rules."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import ceil, sqrt
from struct import pack, unpack

from ...layout.gc_moe_k3 import (
    CANDIDATE_BLOCK_M,
    ROUTED_BLOCK_M,
    RING_TOKEN_ALIGNMENT,
    COMBINE_WARPS,
    COMPUTE_ROLE,
    DISPATCH_ROLE,
    FINALIZE_ROLE,
    WORKER_ROLES,
    FINALIZE_THREADS,
    SharedStorageLayout,
    align,
    ceil_div,
    get_role_shared_layout,
)


SMEM_CAPACITY = 232448
BLOCK_N = 128
BLOCK_K = 128


def _f32(value):
    return unpack('f', pack('f', value))[0]


def get_block_m_for_gc_moe_k3(num_ranks: int, num_experts: int,
                           num_max_tokens_per_rank: int, num_tokens: int,
                           num_topk: int, mma_type: str = 'fp8xfp4') -> int:
    if (num_ranks <= 0 or num_experts <= 0 or num_experts % num_ranks or
            num_tokens < 0 or num_tokens > num_max_tokens_per_rank or
            num_topk <= 0 or num_topk > min(32,num_experts) or mma_type != 'fp8xfp4'):
        raise ValueError('invalid GC MoE block query')
    # Round every load-estimate operation to float32 for stable tile thresholds.
    expected = _f32(_f32(_f32(_f32(num_tokens) * _f32(num_ranks))
                                * _f32(num_topk)) / _f32(num_experts))
    covered = _f32(expected + _f32(sqrt(expected)))
    block_m = 16 if expected <= 10 else 32
    if expected > 24:
        num_blocks = ceil(_f32(covered / 240))
        if num_blocks == 1 and covered > 192 and num_experts // num_ranks < 14:
            num_blocks = 2
        for candidate in CANDIDATE_BLOCK_M:
            if candidate < 64:
                continue
            block_m = candidate
            if _f32(num_blocks * candidate) >= covered:
                break
        # The legacy count above is derived from the 240-row tile, so a
        # mid-sized estimate can leave the last block mostly padding.  Trade
        # at most one extra m-block of weight traffic for at least a full
        # tile of padded MMA rows, and only for tiles with the wider
        # Dispatch loader (12 or 24 warps, depending on the local expert count).
        legacy_rows = _f32(num_blocks * block_m)
        for candidate in (96, 128, 192):
            if candidate >= block_m:
                break
            candidate_blocks = ceil(_f32(covered / candidate))
            if candidate_blocks - num_blocks > 1:
                continue
            rows = _f32(candidate_blocks * candidate)
            if _f32(rows + candidate) <= legacy_rows:
                block_m = candidate
                break
    atom = CANDIDATE_BLOCK_M[0]
    if num_tokens > atom:
        shared_m = get_shared_block_m(block_m, num_tokens)
        # Avoid a separate, narrow routed ring when both task types can
        # use a small common tile. Cover skew with two times the estimated
        # load, rounded to a supported MMA tile.
        covered_peak = 2 * covered
        peak = next((m for m in CANDIDATE_BLOCK_M if m >= covered_peak),
                    CANDIDATE_BLOCK_M[-1])
        if shared_m > 4 * atom:
            # Keep the normal routed sizing if the skew estimate would
            # already fill the shared tile. Padding a wider tile in that
            # regime consumes TMA stages without keeping the rings aligned.
            if covered_peak <= shared_m:
                block_m = max(block_m, shared_m // 2, peak)
        elif atom // 2 <= covered <= atom:
            # With roughly one routed atom per expert, the larger tile
            # also gives Dispatch room to widen its reservation traversal.
            block_m = max(block_m, min(2 * shared_m, 4 * atom))
        else:
            block_m = max(block_m, min(shared_m, max(2 * atom, peak)))
    return max(ROUTED_BLOCK_M[0], block_m)


def get_shared_block_m(block_m: int, tokens: int) -> int:
    shared_tiles = (16, 32, 64, 96, 128)
    widest = shared_tiles[-1]
    # Group the rows into at most two adjacent shared M tiles for weight reuse.
    if block_m <= widest and ceil_div(tokens, widest) <= 2:
        return next((m for m in shared_tiles if m >= tokens), widest)
    return block_m


def num_l1_warmup_waves(total_m_blocks: int, num_clusters: int,
                        l1_n_clusters: int, l2_n_clusters: int) -> int:
    first_l2_m = ceil_div(num_clusters, l2_n_clusters)
    first_l2_waves = ceil_div(first_l2_m * l1_n_clusters, num_clusters)
    diff = max(0, l1_n_clusters - l2_n_clusters)
    interleave = ceil_div(l1_n_clusters + (total_m_blocks - 1) * diff, num_clusters) + 1
    return max(first_l2_waves, interleave)


def num_max_live_pool_blocks(total_m_blocks: int, num_sms: int,
                             hidden: int, intermediate: int) -> int:
    if num_sms < 2 or num_sms % 2 or hidden % 256 or intermediate * 2 % 256:
        raise ValueError('invalid GC MoE cluster/ring geometry')
    clusters = num_sms // 2
    l1_n = intermediate * 2 // 256
    l2_n = hidden // 256
    l1_waves = ceil_div(total_m_blocks * l1_n, clusters)
    warmup = min(num_l1_warmup_waves(total_m_blocks, clusters, l1_n, l2_n), l1_waves)
    live_after_warmup = ceil_div(min(warmup * clusters, total_m_blocks * l1_n), l1_n)
    frontier = ceil_div(total_m_blocks * (l2_n - l1_n), l2_n) if l2_n > l1_n else 0
    wave_margin = ceil_div(clusters, min(l1_n, l2_n))
    return min(total_m_blocks, live_after_warmup + frontier + wave_margin)


def get_ring_tokens(num_ranks: int, num_experts: int, capacity: int, topk: int,
                    hidden: int, intermediate: int, num_sms: int) -> tuple[int, int]:
    local_experts = num_experts // num_ranks
    max_routed = capacity * num_ranks * min(topk, local_experts)
    largest = 0
    for bm in CANDIDATE_BLOCK_M:
        pool_blocks = ceil_div(max_routed, bm) + local_experts
        largest = max(largest, num_max_live_pool_blocks(pool_blocks, num_sms, hidden, intermediate) * bm)
    ring_tokens = align(largest, RING_TOKEN_ALIGNMENT)
    sf_tokens = max((ring_tokens // bm) * align(bm, 128) for bm in ROUTED_BLOCK_M)
    return ring_tokens, sf_tokens


@dataclass(frozen=True)
class RoleConfig:
    role: int
    block_m: int
    block_n: int
    block_k: int
    store_m_l1: int
    store_m_l2: int
    dispatch_threads: int
    threads: int
    num_stages: int
    dynamic_smem: int
    load_m: int
    sf_block_m: int
    swizzle_acts: int
    swizzle_weights: int
    swizzle_output: int
    routing_warps: int
    pull_bytes: int
    shared_layout: SharedStorageLayout


def _compact_dual_ring(shared_m: int, routed_m: int, num_experts: int,
                       pull_bytes: int, shared_depth: int, routed_depth: int,
                       epilogue_geometry: dict[str, int]) -> tuple[int, SharedStorageLayout]:
    # Preserve both initial depths, prioritizing additional shared stages.
    for shared_stages in range(32 - routed_depth, shared_depth - 1, -1):
        for routed_stages in range(32 - shared_stages, routed_depth - 1, -1):
            layout = get_role_shared_layout(
                COMPUTE_ROLE, shared_m, num_experts, pull_bytes, shared_stages,
                routed_ring=(routed_m, routed_stages), omit_shared_sf=True,
                **epilogue_geometry)
            if layout.field_end <= SMEM_CAPACITY:
                return shared_stages, layout
    raise ValueError('compacted dual rings cannot fit their initial depths')


@lru_cache(maxsize=256)
def get_role_config(role: int, block_m: int, num_experts: int,
                    hidden: int, *, local_experts: int,
                    routed_block_m: int | None = None,
                    route_entries: int = 0) -> RoleConfig:
    if role not in WORKER_ROLES or block_m not in CANDIDATE_BLOCK_M:
        raise ValueError('invalid GC role/block M')
    pull_bytes = hidden
    while pull_bytes > 8192:
        if pull_bytes % 2:
            raise ValueError('raw-address pull must divide hidden bytes')
        pull_bytes //= 2
    store1 = 8 if block_m == 48 else 16 if block_m <= 96 else 32 if block_m <= 192 else 24
    store2 = 8 if block_m == 240 else store1
    # Large local expert tables need 96 registers per dispatch thread.
    # Keep that path within the per-SM register budget.
    routing_warps = (24 if local_experts <= 64 else 12) if block_m >= 96 else 8
    if role == DISPATCH_ROLE and 64 <= block_m < 96 and route_entries >= 16 * 32:
        # Widen only if one CTA can afford both its registers and TMA pull rows.
        dispatch_registers = 48 if local_experts <= 64 else 96
        expanded_layout = get_role_shared_layout(
            role, block_m, num_experts, pull_bytes, 2,
            store_m_l1=store1, store_m_l2=store2, dispatch_warps=16,
            block_n=BLOCK_N, block_k=BLOCK_K, sf_block_m=align(block_m, 128))
        if 16 * 32 * dispatch_registers <= 64512 and expanded_layout.field_end <= SMEM_CAPACITY:
            routing_warps = 16
    dispatch_threads = routing_warps * 32 if role == DISPATCH_ROLE else 128
    dispatch_warps = dispatch_threads // 32
    smem_cd = align(max(2 * store1 * 128 * 2, 2 * store2 * 128 * 2), 1024)
    smem_amax = store1 * 8 * 4
    stage = block_m // 2 * 128 + 128 * 128 + align(block_m, 128) * 4 + 128 * 4 + 16
    # The launch budget uses four base dispatch warps. The actual Dispatch
    # layout below uses the selected warp count and two stages.
    budget_dispatch = (1024 if role == COMPUTE_ROLE else
                       align(num_experts * 4, 1024) + align(4 * pull_bytes, 1024))
    budget_barriers = (4 + 4 + 4) * 8
    budget_fixed = budget_dispatch + smem_cd + smem_amax + budget_barriers + 2 * 32 + 4
    budget_stages = (SMEM_CAPACITY - budget_fixed) // stage
    if budget_stages < 2:
        raise ValueError('insufficient shared memory for two TMA stages')
    dynamic_smem = budget_fixed + budget_stages * stage
    stages = 2 if role == DISPATCH_ROLE else budget_stages
    # Actual SharedStorage uses the widened dispatch pull region. Keep its byte
    # count distinct from the launch allocation retained by the reference.
    actual_dispatch = (1024 if role == COMPUTE_ROLE else align(num_experts * 4, 1024) +
                       align(dispatch_warps * pull_bytes, 1024))
    actual_barriers = (dispatch_warps + 4 + 4) * 8
    fixed = actual_dispatch + smem_cd + smem_amax + actual_barriers + 2 * 32 + 4
    if role != DISPATCH_ROLE and fixed + stages * stage > dynamic_smem:
        raise ValueError('role shared-storage footprint exceeds launch allocation')
    threads = (dispatch_threads if role == DISPATCH_ROLE else
               FINALIZE_THREADS if role == FINALIZE_ROLE else 384)
    layout_geometry = dict(
        store_m_l1=store1, store_m_l2=store2, dispatch_warps=dispatch_warps,
        block_n=BLOCK_N, block_k=BLOCK_K, sf_block_m=align(block_m, 128))
    shared_layout = get_role_shared_layout(
        role, block_m, num_experts, pull_bytes, stages, **layout_geometry)
    if routed_block_m is not None:
        if not (role == COMPUTE_ROLE and 16 <= routed_block_m < block_m <= 128 and
                routed_block_m in CANDIDATE_BLOCK_M):
            raise ValueError('invalid fused compute routed/shared M widths')
        for routed_stages in range(32 - stages, stages, -1):
            candidate = get_role_shared_layout(
                role, block_m, num_experts, pull_bytes, stages,
                routed_ring=(routed_block_m, routed_stages), **layout_geometry)
            if candidate.field_end <= SMEM_CAPACITY:
                shared_layout = candidate
                dynamic_smem = candidate.field_end
                break
        if shared_layout.routed_payload is not None:
            epilogue = get_role_config(role, routed_block_m, num_experts, hidden,
                                       local_experts=local_experts)
            stages, shared_layout = _compact_dual_ring(
                block_m, routed_block_m, num_experts, pull_bytes, stages,
                shared_layout.routed_payload.stages,
                dict(layout_geometry, store_m_l1=epilogue.store_m_l1,
                     store_m_l2=epilogue.store_m_l2))
            dynamic_smem = shared_layout.field_end
    if role == DISPATCH_ROLE:
        # The dispatch layout has no MMA planes; use its actual footprint.
        dynamic_smem = shared_layout.field_end
        if dynamic_smem > SMEM_CAPACITY:
            raise ValueError('dispatch pull region exceeds shared memory')
    if shared_layout.field_end > dynamic_smem:
        raise ValueError('role shared storage exceeds its launch allocation')
    return RoleConfig(role, block_m, BLOCK_N, BLOCK_K, store1, store2,
                      dispatch_threads, threads, stages, dynamic_smem,
                      block_m // 2, align(block_m, 128),
                      128, 128, 64, routing_warps, pull_bytes, shared_layout)


def get_combine_geometry(hidden: int) -> tuple[int, int, int]:
    """Split the row into aligned warp chunks for register-resident reduction."""
    if hidden <= 0 or hidden % 256:
        raise ValueError('finalize requires whole 256-column vector groups')
    # Each lane owns eight BF16 values per 256-column group. Prefer at
    # most two groups per lane to limit the number of FP32 accumulators
    # without recruiting more warps than needed. For wider rows use the
    # largest exact split available within the combine warp group.
    groups = hidden // 256
    candidates = [chunks for chunks in range(1, COMBINE_WARPS + 1) if groups % chunks == 0]
    chunks = min(candidates, key=lambda c: (max(0, groups // c - 2), c))
    vectors_per_lane = groups // chunks
    return chunks, vectors_per_lane * 512, vectors_per_lane
