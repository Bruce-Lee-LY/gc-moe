#include "combine.cuh"
#include "dispatch.cuh"
#include "projection.cuh"

extern "C" int gc_moe_init_silu(float* table, hggcStream_t stream) {
    gc_moe::make_silu_table<<<256, 256, 0, stream>>>(table);
    return hggcGetLastError();
}

extern "C" int gc_moe_decode(const gc_moe::Launch* request, hggcStream_t stream,
                             hggcStream_t side, hggcEvent_t start, hggcEvent_t done) {
    using namespace gc_moe;
    auto p = *request;
    const auto schedule = select_schedule(p, side != nullptr);

    if (schedule.overlap) {
        if (const auto status = hggcEventRecord(start, stream); status != hggcSuccess)
            return status;
        if (const auto status = hggcStreamWaitEvent(side, start, 0); status != hggcSuccess)
            return status;
        if (schedule.tiny_shared) {
            if (!BalancedProjections && p.bound <= 4)
                launch_tiny_shared<true>(p, side);
            else
                launch_tiny_shared<false>(p, side);
        } else if (schedule.queue_shared_m64) {
            launch_tiny_shared<false, 64>(p, side);
        } else {
            launch_shared_projection<true>(p, side);
            decode_ptpc_quantize<<<p.tokens, Threads, 0, side>>>(p);
            launch_shared_projection<false>(p, side);
        }
        if (const auto status = hggcEventRecord(done, side); status != hggcSuccess)
            return status;
    }
    launch_dispatch(p, schedule, stream);
    launch_routed(p, schedule, stream);
    return launch_finalize(p, schedule, stream, done);
}
