"""Byte-exact symmetric workspace layout for the SM100 GC MoE K3 implementation.

Signals, routing workspace and tensor buffers share one symmetric allocation.
Ring slots retain generation counters until the finalize role clears them.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lcm


FRONT_ROLE = 1
ROUTER_ROLE = 2
DISPATCH_ROLE = 3
COMPUTE_ROLE = 4
FINALIZE_ROLE = 5
PROJECTION_REDUCE_ROLE = 6
ROUTED_ASSIST_ROLE = 7
WORKER_ROLES = (DISPATCH_ROLE, COMPUTE_ROLE, FINALIZE_ROLE)
K3_ROLES = (FRONT_ROLE, ROUTER_ROLE, *WORKER_ROLES)
ASSIST_WORKER_ROLES = (DISPATCH_ROLE, COMPUTE_ROLE, ROUTED_ASSIST_ROLE, FINALIZE_ROLE)
ASSIST_ROLES = (FRONT_ROLE, ROUTER_ROLE, *ASSIST_WORKER_ROLES)
VALID_ROLE_ORDERS = (WORKER_ROLES, K3_ROLES, (*K3_ROLES, PROJECTION_REDUCE_ROLE),
                     ASSIST_ROLES, (*ASSIST_ROLES, PROJECTION_REDUCE_ROLE))


CANDIDATE_BLOCK_M = (16, 32, 48, 64, 96, 128, 192, 240)
# Shared BF16 tasks may use M16/M32. Only routed tiles index the FP8 scale ring.
ROUTED_BLOCK_M = tuple(m for m in CANDIDATE_BLOCK_M if m >= 48)
# Ring slots must be divisible by every supported block M.
RING_TOKEN_ALIGNMENT = lcm(*CANDIDATE_BLOCK_M)
# Token-row padding is separate from the routed ring geometry.
TOKEN_ALIGNMENT = 16
MAX_RANKS = 72
MAX_EXPERTS = 2048
MAX_EXPERTS_PER_RANK = 512
# Keep host launch sizing and device register tiers on the same threshold.
SMALL_LOCAL_EXPERT_TABLE_MAX = 64
MAX_RING_BLOCKS = 1 << 20
MAX_SHARED_L2_BLOCKS = 1 << 15
# Accumulator barriers reserved here; the compute role indexes them by
# tile.epilogue_stages, so the two depths must agree.
ACCUM_BARRIER_STAGES = 2
# finalize uses one dispatch warpgroup and two combine warpgroups.
COMBINE_WARPS = 8
FINALIZE_THREADS = (4 + COMBINE_WARPS) * 32
MAX_PROJECTION_ROWS = MAX_SHARED_L2_BLOCKS * 8
PROJECTION_K_TILE = 64
PROJECTION_CHANNELS_PER_CTA = 128
PROJECTION_ROW_TILES = (16, 32, 64, 128)


def align(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass(frozen=True)
class MmaPayloadLayout:
    stages: int
    activation_pitch: int
    data_a: int
    data_b: int
    data_sfa: int
    data_sfb: int
    end: int


def _mma_payload_layout(start: int, stages: int, block_m: int,
                        block_n: int, block_k: int, sf_block_m: int, *,
                        with_scales: bool = True) -> MmaPayloadLayout:
    activation_pitch = (block_m // 2) * block_k
    data_a = align(start, 1024)
    data_b = align(data_a + stages * activation_pitch, 1024)
    ab_end = data_b + stages * block_n * block_k
    if not with_scales:
        # The BF16 path never accesses the SF pointers. Keep them inside the slab.
        return MmaPayloadLayout(stages, activation_pitch, data_a, data_b,
                                data_a, data_a, ab_end)
    data_sfa = align(ab_end, 4)
    data_sfb = align(data_sfa + stages * sf_block_m * 4, 4)
    return MmaPayloadLayout(stages, activation_pitch, data_a, data_b, data_sfa,
                            data_sfb, data_sfb + stages * block_n * 4)


@dataclass(frozen=True)
class SharedStorageLayout:
    expert_counts: int
    pull: int
    cd: int
    data_a: int
    data_b: int
    data_sfa: int
    data_sfb: int
    amax: int
    tasks: int
    dispatch_bars: int
    data_full: int
    data_empty: int
    accum_full: int
    accum_empty: int
    task_full: int
    task_empty: int
    tmem_ptr: int
    reusable_bytes: int
    field_end: int
    routed_payload: MmaPayloadLayout | None = None


def get_role_shared_layout(role: int, block_m: int, num_experts: int,
                           pull_bytes: int, stages: int, *,
                           store_m_l1: int, store_m_l2: int, dispatch_warps: int,
                           block_n: int, block_k: int,
                           sf_block_m: int,
                           routed_ring: tuple[int, int] | None = None,
                           omit_shared_sf: bool = False) -> SharedStorageLayout:
    """Lay out one role's shared data and control fields in byte order."""
    if role not in WORKER_ROLES or block_m % 16 or block_m < 16 or stages < 2:
        raise ValueError('invalid shared-storage role geometry')
    if routed_ring is not None and (role != COMPUTE_ROLE or not 16 <= routed_ring[0] < block_m or
                                    routed_ring[0] % 16 or not 2 <= routed_ring[1] <= 32 - stages):
        raise ValueError('invalid overlapped compute routed ring')
    if omit_shared_sf and routed_ring is None:
        raise ValueError('only the separate BF16 ring can omit scale factors')
    values = {}
    cursor = 0
    routed_payload = None

    def reserve(name: str, size: int, alignment: int) -> int:
        nonlocal cursor
        cursor = align(cursor, alignment)
        values[name] = cursor
        cursor += size
        return values[name]

    if role == COMPUTE_ROLE:
        values['expert_counts'] = values['pull'] = 0
        cursor = 1
    else:
        reserve('expert_counts', num_experts * 4, 1024)
        reserve('pull', dispatch_warps * pull_bytes, 1024)
    reserve('cd', max(2 * 2 * store_m_l1 * block_n,
                      2 * store_m_l2 * block_n * 2), 1024)
    if role == DISPATCH_ROLE:
        # Dispatch only pulls remote tokens into the ring; MMA stage planes
        # belong to the compute role.
        values['data_a'] = values['data_b'] = 0
        values['data_sfa'] = values['data_sfb'] = 0
    else:
        payload = _mma_payload_layout(cursor, stages, block_m, block_n,
                                      block_k, sf_block_m, with_scales=not omit_shared_sf)
        if routed_ring is not None:
            routed_m, routed_stages = routed_ring
            routed_payload = _mma_payload_layout(cursor, routed_stages, routed_m,
                                                 block_n, block_k, sf_block_m)
        values.update(data_a=payload.data_a, data_b=payload.data_b,
                      data_sfa=payload.data_sfa, data_sfb=payload.data_sfb)
        cursor = max(payload.end, routed_payload.end) if routed_payload else payload.end
    reserve('amax', 8 * store_m_l1 * 4, 8)
    reserve('tasks', 2 * 32, 16)
    reserve('dispatch_bars', dispatch_warps * 8, 8)
    barrier_stages = stages + (routed_payload.stages if routed_payload else 0)
    reserve('data_full', barrier_stages * 8, 8)
    reserve('data_empty', barrier_stages * 8, 8)
    reserve('accum_full', ACCUM_BARRIER_STAGES * 8, 8)
    reserve('accum_empty', ACCUM_BARRIER_STAGES * 8, 8)
    reserve('task_full', 2 * 8, 8)
    reserve('task_empty', 2 * 8, 8)
    reserve('tmem_ptr', 4, 4)
    # Finalize keeps the combine row slot at tasks while projection uses the slab.
    return SharedStorageLayout(**values, reusable_bytes=values['tasks'],
                               field_end=cursor, routed_payload=routed_payload)


def _signal_offsets() -> dict[str, int]:
    fields = (
        ('grid_sync_count', 4, 4),
        ('l1_task_count', 2, 4),
        ('l2_task_count', 2, 4),
        ('shared_l1_task_count', 2, 4),
        ('shared_l2_task_count', 2, 4),
        # Persistent receive-generation parity: replay N reads/writes
        # generation parity & 1 of expert_recv_count/expert_recv_count_sum
        # (slot range [gen * count, (gen + 1) * count) inside each field's
        # dual-generation region); finalize flips this word once per replay.
        # Lives in the header padding.  dispatch publishes the finalize epoch
        # at its entry, so finalize reads it instead of a grid barrier.
        ('recv_parity', 1, 4),
        ('finalize_epoch', 1, 4),
        ('routed_pool_blocks', 1, 4),
        ('combine_ready_grid_idx', MAX_RANKS, 8),
        ('expert_send_count', MAX_EXPERTS, 8),
        ('expert_recv_count', 2 * MAX_EXPERTS, 8),
        ('expert_recv_count_sum', 2 * MAX_EXPERTS_PER_RANK, 8),
        ('l1_full_count', MAX_RING_BLOCKS, 4),
        ('l1_empty_count', MAX_RING_BLOCKS, 4),
        ('l2_full_mask', MAX_RING_BLOCKS, 8),
        ('l2_empty_count', MAX_RING_BLOCKS, 4),
        ('shared_l2_full_count', MAX_SHARED_L2_BLOCKS, 4),
        ('combine_peer_publisher', 1, 4),
        ('combine_next_row', 1, 4),
        ('projection_row_ready', MAX_PROJECTION_ROWS, 8),
        ('dispatch_source_ready', 2 * MAX_RANKS, 8),
    )
    offset = 0
    result = {}
    for name, count, element_bytes in fields:
        field_alignment = 128 if name in (
            'combine_ready_grid_idx', 'expert_send_count'
        ) else element_bytes
        offset = align(offset, field_alignment)
        result[name] = offset
        offset += count * element_bytes
    result['size'] = align(offset, 128)
    return result


SIGNAL_OFFSETS = _signal_offsets()


@dataclass(frozen=True)
class BufferView:
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: str
    stride: tuple[int, ...] | None = None


@dataclass(frozen=True)
class GCK3BufferLayout:
    num_ranks: int
    num_experts: int
    capacity: int
    topk: int
    hidden: int
    intermediate: int
    ring_tokens: int
    sf_ring_tokens: int
    num_shared_experts: int
    shared_hidden: int
    shared_intermediate: int

    def __post_init__(self) -> None:
        ranks, experts, capacity = self.num_ranks, self.num_experts, self.capacity
        if not (0 < ranks <= MAX_RANKS and 0 < experts <= MAX_EXPERTS and
                experts % ranks == 0 and experts // ranks <= MAX_EXPERTS_PER_RANK):
            raise ValueError('invalid rank/expert configuration')
        if not (capacity > 0 and self.topk > 0 and
                self.topk <= min(32, experts)):
            raise ValueError('invalid capacity/topk/shared configuration')
        if not (self.hidden > 0 and self.intermediate > 0 and
                self.hidden % 256 == 0 and self.intermediate % 128 == 0):
            raise ValueError('hidden must be a positive multiple of 256 and intermediate a positive multiple of 128')
        if self.ring_tokens <= 0 or self.ring_tokens % RING_TOKEN_ALIGNMENT != 0:
            raise ValueError(f'ring token capacity must be positive and {RING_TOKEN_ALIGNMENT}-aligned')
        required_sf_ring_tokens = max(
            (self.ring_tokens // block_m) * align(block_m, 128)
            for block_m in ROUTED_BLOCK_M)
        if (self.sf_ring_tokens < required_sf_ring_tokens or
                self.sf_ring_tokens % 4 != 0):
            raise ValueError('invalid scale-factor ring capacity')
        if (self.num_shared_experts <= 0 or self.shared_hidden <= 0 or
                self.shared_hidden % 256 or self.shared_intermediate <= 0 or
                self.shared_intermediate % 128):
            raise ValueError('K3 requires shared experts with hidden aligned to 256 and intermediate to 128')
        if self.ring_tokens > MAX_RING_BLOCKS * min(CANDIDATE_BLOCK_M):
            raise ValueError('ring signal capacity exceeded')
        if ceil_div(capacity, 8) > MAX_SHARED_L2_BLOCKS:
            raise ValueError('shared L2 signal capacity exceeded')

    @property
    def num_max_pool_tokens(self) -> int:
        local_experts = self.num_experts // self.num_ranks
        return align(self.num_ranks * self.capacity * min(self.topk, local_experts)
                     + local_experts * (max(CANDIDATE_BLOCK_M) - 1), RING_TOKEN_ALIGNMENT)

    @property
    def workspace_offsets(self) -> dict[str, int]:
        source = SIGNAL_OFFSETS['size']
        metadata = source + self.num_experts * self.capacity * 4
        end = align(metadata + self.num_max_pool_tokens * 12, 16)
        return {'signals': 0, 'src_token_topk_idx': source,
                'token_src_metadata': metadata, 'end': end}

    @property
    def views(self) -> dict[str, BufferView]:
        c, h, i = self.capacity, self.hidden, self.intermediate
        r, sr = self.ring_tokens, self.sf_ring_tokens
        offset = self.workspace_offsets['end']
        result: dict[str, BufferView] = {}

        def add(name: str, nbytes: int, shape: tuple[int, ...], dtype: str,
                stride: tuple[int, ...] | None = None) -> None:
            nonlocal offset
            result[name] = BufferView(offset, nbytes, shape, dtype, stride)
            offset += nbytes

        add('x', c * h, (c, h), 'float8_e4m3fn')
        add('x_sf', c * h // 32, (c, h // 128), 'int32')
        add('topk_idx', c * self.topk * 8, (c, self.topk), 'int64')
        add('topk_weights', c * self.topk * 4, (c, self.topk), 'float32')
        add('shared_l2_acts', c * self.shared_intermediate * 2,
            (c, self.shared_intermediate), 'bfloat16')
        add('l1_acts', r * h, (r, h), 'float8_e4m3fn')
        add('l1_acts_sf', sr * h // 32, (sr, h // 128), 'int32', (1, sr))
        add('l2_acts', r * i, (r, i), 'float8_e4m3fn')
        add('l2_acts_sf', sr * i // 32, (sr, i // 128), 'int32', (1, sr))
        add('combine_tokens', self.topk * c * h * 2,
            (self.topk, c, h), 'bfloat16')
        add('routed_latent', c * h * 2, (c, h), 'bfloat16')
        add('shared_output', c * self.shared_hidden * 2,
            (c, self.shared_hidden), 'bfloat16')
        add('front_logits', c * self.num_experts * 4, (c, self.num_experts), 'float32')
        # Reserve split-K storage when each half can feed an eight-stage
        # TMA ring. The launch policy also checks output geometry and SMs.
        if h >= 2 * 8 * PROJECTION_K_TILE:
            rows = min(c, 2 * max(PROJECTION_ROW_TILES))
            width = self.shared_hidden
            add('projection_partials', 2 * rows * width * 4, (rows, 2, width), 'float32')
        return result

    @property
    def nbytes(self) -> int:
        views = self.views
        v = views.get('projection_partials', views['front_logits'])
        return v.offset + v.nbytes
