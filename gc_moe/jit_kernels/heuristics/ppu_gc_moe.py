"""Select the PPU expert tile and dispatch schedule from the launch geometry."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PPURoleConfig:
    hidden: int
    intermediate: int
    local_experts: int
    topk: int
    ranks: int

    def __post_init__(self):
        # Remote dispatch copies whole 512-byte FP4 tiles.
        # Weight panels require I % 256 == 0.
        if not (0 < self.hidden <= 16384 and self.hidden % 1024 == 0
                and self.intermediate > 0 and self.intermediate % 256 == 0
                and 1 <= self.ranks <= 8 and 1 <= self.local_experts <= 64
                and 1 <= self.topk <= min(32, self.ranks * self.local_experts)):
            raise ValueError(f'unsupported PPU GC MoE geometry: {self}')

    def schedule(self, tokens_per_rank: int, *, shared_expert: bool = False) -> str:
        """Select a schedule using the same token bound on every rank.

        The route bound selects the tile, as on SM100 and SM120.
        Kernels size the work from actual receive counts.
        """
        if tokens_per_rank < 0:
            raise ValueError('the token bound must be nonnegative')
        routes = tokens_per_rank * self.topk
        experts = self.local_experts

        # Prefer the two-stage full-width tile for short FP4 rows or fewer
        # than eight FC1 panels. Narrow tiles can hit hgcc's scalar register
        # alignment issue for these smaller operand geometries.
        if self.hidden <= 4 * 1024 or self.intermediate < 8 * 256:
            if tokens_per_rank <= 64 and routes <= 16 * experts:
                return 'm32_direct_two_stage'
            return 'm32_grouped' if routes <= 16 * experts else 'm64_grouped'

        chunked = ('m48_direct_chunked' if tokens_per_rank <= 64 and not shared_expert
                   else 'm64_grouped_chunked')
        # Eight routes let each warp of a narrow tile share one activation
        # load across two routes. Use it for small SE requests and as an
        # alternative to the wide routed tile near its lower threshold.
        if self.topk == 8 and 1 < tokens_per_rank <= 8:
            if shared_expert and (routes < experts or routes == 2 * experts):
                return 'm16_small_batch'
            if not shared_expert and experts <= routes < 2 * experts:
                return 'm16_small_batch'
            if routes <= 2 * experts:
                return 'm16_short_decode'
        if shared_expert and self.topk > 4 and 1 < tokens_per_rank <= 32:
            # The shared expert fits in one 48-row tile. Use direct
            # dispatch when the route bound warrants a full-width tile.
            threshold = (2 if self.topk < 8 else 4) * experts
            if threshold < routes <= 4 * threshold:
                return 'm48_direct_chunked'
            if experts <= routes <= threshold:
                return 'm16_deep_ring'
        if self.topk < 8:
            # The extra shared expert work narrows the range for the
            # chunked ring. The routed direct tile can start earlier.
            chunked_min = 3 if chunked == 'm48_direct_chunked' else 6
            chunked_limit = 24 if shared_expert else 32
            if self.topk > 4 and chunked_min * experts <= routes <= chunked_limit * experts:
                return chunked
            if tokens_per_rank > 64 or routes > 8 * experts:
                return 'm32_grouped' if routes <= 16 * experts else 'm64_grouped'
            # Up to four routes fit in the four-warp tile. Keep the
            # two-stage receiver for that case and the smallest SE request.
            if self.topk <= 4 or (shared_expert and tokens_per_rank <= 1):
                return 'm16_two_stage_receiver'
            if routes <= 2 * experts:
                return 'm16_token_dispatch' if shared_expert else 'm16_deep_ring'
            if routes <= 4 * experts:
                return 'm16_direct_split'
            return 'm32_direct_token_dispatch'

        # Keep the low-route path on the deeper ring. Routed CTAs can
        # reuse each activation load across two destination slots.
        if tokens_per_rank <= 1 or (not shared_expert and routes <= experts):
            return 'm16_deep_ring'
        # A full-width FC1 tile reduces the number of panels once the
        # launch has at least two route slots per local expert.
        if chunked == 'm48_direct_chunked' and 2 * experts <= routes <= 32 * experts:
            return chunked
        if tokens_per_rank > 64 or routes >= 8 * experts:
            if routes <= 32 * experts:
                return chunked
            return 'm80_grouped'
        if routes <= 2 * experts:
            return 'm16_direct' if routes <= experts else 'm16_two_stage'
        return 'm16_deep_ring' if routes <= 4 * experts else 'm32_direct_narrow'
