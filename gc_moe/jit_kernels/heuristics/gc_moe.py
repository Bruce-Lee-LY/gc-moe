"""GC MoE block, ring and role-local shared-memory sizing rules."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt
from struct import pack, unpack

from ...layout.gc_moe import CANDIDATE_BLOCK_M, align, ceil_div


SMEM_CAPACITY = 232448
BLOCK_N = 128
BLOCK_K = 128


def _f32(value):
    return unpack('f', pack('f', value))[0]


def get_block_m_for_gc_moe(num_ranks: int, num_experts: int,
                           num_max_tokens_per_rank: int, num_tokens: int,
                           num_topk: int, mma_type: str = 'fp8xfp4') -> int:
    if (num_ranks <= 0 or num_experts <= 0 or num_experts % num_ranks or
            num_tokens < 0 or num_tokens > num_max_tokens_per_rank or
            num_topk <= 0 or num_topk > 32 or mma_type not in ('fp8xfp4', 'fp8xfp8', 'bf16xbf16')):
        raise ValueError('invalid GC MoE block query')
    # The C++ selector rounds each arithmetic operation to float32.
    expected = _f32(_f32(_f32(_f32(num_tokens) * _f32(num_ranks))
                                * _f32(num_topk)) / _f32(num_experts))
    covered = _f32(expected + _f32(sqrt(expected)))
    block_m = 16 if expected <= 10 else 32
    if expected > 24:
        num_blocks = ceil(_f32(covered / 240))
        if num_blocks == 1 and covered > 192 and num_experts // num_ranks < 14:
            num_blocks = 2
        for candidate in (64, 96, 128, 192, 240):
            block_m = candidate
            if _f32(num_blocks * candidate) >= covered:
                break
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
    ring_tokens = align(largest, 1920)
    sf_tokens = max((ring_tokens // bm) * align(bm, 128) for bm in CANDIDATE_BLOCK_M)
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
    stage_bytes: int
    fixed_bytes: int


def get_role_config(role: int, block_m: int, num_experts: int,
                    hidden: int, pull_bytes: int) -> RoleConfig:
    if role not in (1, 2, 3) or block_m not in CANDIDATE_BLOCK_M or block_m % 16:
        raise ValueError('invalid GC role/block M')
    store1 = 8 if block_m <= 16 else 16 if block_m <= 96 else 32 if block_m <= 192 else 24
    store2 = 8 if block_m == 240 else store1
    dispatch_threads = (384 if block_m >= 96 else 256) if role == 1 else 128
    dispatch_warps = dispatch_threads // 32
    smem_cd = align(max(2 * store1 * 64 * 2, 2 * store2 * 128 * 2), 1024)
    smem_amax = store1 * 8 * 4
    stage = block_m // 2 * 128 + 128 * 128 + align(block_m, 128) * 4 + 128 * 4 + 16
    # get_gc_moe_config budgets dispatch/finalize SMEM with the base four
    # dispatch warps. Its JIT separately widens role1 and fixes its stages=2.
    budget_dispatch = (1024 if role == 2 else
                       align(num_experts * 4, 1024) + align(4 * pull_bytes, 1024))
    budget_barriers = (4 + 4 + 16 + 4) * 8
    budget_fixed = budget_dispatch + smem_cd + smem_amax + budget_barriers + 2 * 32 + 4
    budget_stages = (SMEM_CAPACITY - budget_fixed) // stage
    if budget_stages < 2:
        raise ValueError('insufficient shared memory for two TMA stages')
    dynamic_smem = budget_fixed + budget_stages * stage
    stages = 2 if role == 1 else budget_stages
    # Actual SharedStorage uses the widened role1 pull region. Keep its byte
    # count distinct from the launch allocation retained by the reference.
    actual_dispatch = (1024 if role == 2 else align(num_experts * 4, 1024) +
                       align(dispatch_warps * pull_bytes, 1024))
    actual_barriers = (dispatch_warps + 4 + 16 + 4) * 8
    fixed = actual_dispatch + smem_cd + smem_amax + actual_barriers + 2 * 32 + 4
    if fixed + stages * stage > dynamic_smem:
        raise ValueError('role shared-storage footprint exceeds launch allocation')
    threads = dispatch_threads if role == 1 else 384
    return RoleConfig(role, block_m, 128, 128, store1, store2,
                      dispatch_threads, threads, stages, dynamic_smem,
                      stage, fixed)
