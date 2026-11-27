#pragma once

#include "workspace.cuh"

#include <cmath>
#include <cstring>

namespace gc_moe {

struct DecodeSchedule {
    int routes;
    bool routed_only, paged_dispatch, fused_dispatch, large_routed;
    bool tiny_shared, overlap, large_queue;
    bool small_pages, medium_pages, tall_pages, paired_scales, wide_pages, page_dispatch;
    bool pipeline, queue_shared_m64, clamp_in_table, prefetch_returns;
};

inline DecodeSchedule select_schedule(Launch& p, bool side) {
    const bool routed_only = !p.shared_x;
    const int routes = p.bound * p.ranks * TopK;
    const bool full_group_aiu = p.ranks == MaxRanks && p.aiu;
    const bool paged_dispatch = CanPage && full_group_aiu && (routed_only || side);
    const bool fused_dispatch = paged_dispatch && p.shared_x;
    const bool large_routed = paged_dispatch && routed_only && p.bound >= 64;
    if (!p.m_tiles) {
        // Expected rows per expert guide the tile size. Actual receive counts
        // determine page overflow and how many M panels W1 and W2 need.
        if (RoutedFP8_FP4 && paged_dispatch) {
            // Reserve room above the mean and shrink the actual MMA for
            // partially filled pages. This avoids repeating B on short tails.
            const double mean = double(routes) / Experts;
            const double target = mean + 2 * std::sqrt(mean);
            if (target <= 16)
                p.m_tiles = 1;
            else if (target <= 32)
                p.m_tiles = 2;
            else if (target <= 48)
                p.m_tiles = 3;
            else if (target <= 64)
                p.m_tiles = 4;
            else if (target <= 96 || !CanUseM128)
                p.m_tiles = 6;
            else
                p.m_tiles = 8;
        } else if (paged_dispatch) {
            // Leave room for the tail of the expert load before an M64 page
            // fills. Wider down projections repay the larger page sooner.
            const int tall_threshold = BalancedProjections ? 56 : 58;
            // With long W1 reductions, a second M16 page is more expensive
            // than padding the first page to M32.
            const int medium_threshold = large_routed && LongGate ? 9 : 12;
            // Leave headroom for variation in expert load before a second M96
            // page repeats the weight loads.
            if (CanUseM128 && routes >= 67 * Experts)
                p.m_tiles = 8;
            else if (routes >= tall_threshold * Experts)
                p.m_tiles = 6;
            else if (routes >= 20 * Experts)
                p.m_tiles = 4;
            else
                p.m_tiles = routes >= medium_threshold * Experts ? 2 : 1;
        } else {
            p.m_tiles = routes >= 29 * Experts ? 3 : (routes >= 16 * Experts ? 2 : 1);
        }
    }

    const bool has_shared_stream = side && p.shared_x && p.tokens;
    const bool tiny_shared = fused_dispatch && p.bound > 0 && p.bound <= TileM;
    const bool overlap = has_shared_stream &&
                         (full_group_aiu || routes <= 4 * Experts || routes >= 8 * Experts || p.m_tiles >= 3);
    const bool sparse_queue = routes >= Experts / 4 && routes <= Experts / 2;
    const bool large_queue = (routed_only || overlap || fused_dispatch) && p.ranks == MaxRanks && p.m_tiles >= 3;
    const bool small_pages = paged_dispatch && p.m_tiles == 1 && p.bound > 0;
    const bool medium_pages = paged_dispatch && p.m_tiles == 2 && p.bound > 0;
    const bool tall_pages = paged_dispatch && (p.m_tiles == 6 || (CanUseM128 && p.m_tiles == 8)) && p.bound > 0;
    const bool paired_scales = paged_dispatch && p.m_tiles == 4 && p.bound > 0;
    // For near-full M64 tiles with a shorter W1 reduction, the indexed
    // layout keeps the independent column-group pipelines busy.
    const bool wide_pages = paired_scales &&
                            (!routed_only || LongGate || (large_routed && routes < 46 * Experts));
    const bool page48 = RoutedFP8_FP4 && paged_dispatch && p.m_tiles == 3 && p.bound > 0;
    const bool page_dispatch = small_pages || medium_pages || page48 || wide_pages || tall_pages;
    const bool pipeline = RoutedFP8_FP4 ||
                          (p.aiu && (overlap || routed_only || fused_dispatch) &&
                           (page_dispatch || large_queue ||
                            (p.m_tiles < 3 && (sparse_queue || routes >= 2 * Experts))));
    const bool queue_shared_m64 = BalancedProjections && fused_dispatch && medium_pages &&
                                  p.bound <= 64 && p.tokens > 32;

    uint32_t clamp_bits;
    std::memcpy(&clamp_bits, &p.clamp, sizeof(clamp_bits));
    // The rounded gate and a BF16 clamp both have SiLU table entries.
    const bool clamp_in_table = pipeline && paged_dispatch && (clamp_bits & 0xffffu) == 0;
    const bool prefetch_returns = clamp_in_table && small_pages && p.bound > TileM && p.bound <= 2 * TileM;
    return {routes, routed_only, paged_dispatch, fused_dispatch, large_routed, tiny_shared, overlap, large_queue,
            small_pages, medium_pages, tall_pages, paired_scales, wide_pages, page_dispatch,
            pipeline, queue_shared_m64, clamp_in_table, prefetch_returns};
}

}  // namespace gc_moe
