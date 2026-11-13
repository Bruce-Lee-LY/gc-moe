#include "common.cuh"

// Host submission order for the PPU communication and expert kernels.
extern "C" {
void gc_comm_reset_barrier(void*,const long long*,int,int,int,int,
                          int*,int*,int*,int,int,int,int,int);
void gc_comm_dispatch_push(void*,const long long*,int,int,
                          const void*,const void*,const int*,const float*,
                          int,int,int,int,int,int,int,int,int,int,int*,int*,
                          int*,int,void*,void*,int*,int*,int*,float*,int*,int*);
void gc_comm_finalize_reset(const int*,int,int,int,const int*,const int*,
                           const int*,int*,int*,int*,int*,int*,int*,int);
void gc_comm_finalize_groups(const int*,int,int,int,const int*,const int*,
                            const int*,int*,int*,int*,int*,int*,int*,int,int*,int);
void gc_moe_fused(void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
                  void*,void*,void*,void*,void*,int,int,int,int,void*,void*,int);
void gc_comm_combine_scatter(void*,const long long*,int,int,const float*,
                            const int*,const int*,const int*,const int*,int,int,
                            int,int,int,int,int,int*,int*,float*,int,int,int);
void gc_moe_role12(void**,const int*,const long long*);
#ifdef GC_ROLE12_START_RESET
#ifndef GC_FUSED_COMBINE_RESET
#error GC_ROLE12_START_RESET requires mode2 combine reset
#endif
void gc_moe_role12_with_start_reset(void**,const int*,const long long*,const int*,int);
#endif
// c = {barr_off, ccnt_off, cgen_off, cbuf_off, crec_off}; f stores six launchers.
static void pipeline_impl(void** p,const int* i,const long long* peer,const int* c,
                          void** f,int first_barrier_generation,float* output,
                          bool unified, bool start_reset = false) {
    using namespace role12;
    auto ip = [&](Ptr k) { return static_cast<int*>(p[k]); };
    auto fp = [&](Ptr k) { return static_cast<float*>(p[k]); };
    auto reset = reinterpret_cast<decltype(&gc_comm_reset_barrier)>(f[0]);
    auto dispatch = reinterpret_cast<decltype(&gc_comm_dispatch_push)>(f[1]);
    auto finalize = reinterpret_cast<decltype(&gc_comm_finalize_reset)>(f[2]);
    auto math = reinterpret_cast<decltype(&gc_moe_fused)>(f[3]);
    auto combine = reinterpret_cast<decltype(&gc_comm_combine_scatter)>(f[4]);
    auto barrier = [&](int gen, bool combine_only = false) {
        int reset_experts = i[E];
#ifdef GC_LOCAL_COMBINE_RESET
        if (combine_only) reset_experts = -reset_experts;
#endif
        reset(p[SYM],peer,i[RANKS],i[MY],gen,c[0],ip(CURSORS),ip(DONE),
              ip(COUNTS),reset_experts,i[CNT_OFF],i[GEN_OFF],c[1],c[2]);
    };
#ifdef GC_FUSED_COMBINE_RESET
    if (!start_reset) barrier(-(first_barrier_generation+1));
#else
    barrier(first_barrier_generation);
#endif
    if (unified) {
#ifdef GC_ROLE12_START_RESET
        if (start_reset) {
            auto role12 = reinterpret_cast<decltype(&gc_moe_role12_with_start_reset)>(f[3]);
            role12(p,i,peer,c,first_barrier_generation+1);
        } else
#endif
        {
        auto role12 = reinterpret_cast<decltype(&gc_moe_role12)>(f[3]);
        role12(p,i,peer);
        }
    } else {
    dispatch(p[SYM],peer,i[RANKS],i[MY],p[XP],p[SF],ip(TOPK_IDX),fp(TOPK_W),
             i[T],i[TOPK],i[H],i[E],i[POOL_OFF],i[SF_OFF],i[REC_OFF],
             i[CNT_OFF],i[GEN_OFF],i[CAP],ip(CURSORS),ip(DONE),ip(COUNTS),
             i[CAP_E],p[POOL],p[POOL_SF],ip(REC_T),ip(REC_SLOT),ip(REC_SRC),
             fp(REC_W),ip(BUCKET),ip(UNUSED));
    if (i[ROLE1_CTAS] > 0) { // separate-mode slot17 carries the optional group tile
        auto groups = reinterpret_cast<decltype(&gc_comm_finalize_groups)>(f[2]);
        groups(ip(COUNTS),i[E],i[CAP_E],i[CAP_E]*i[E],
               ip(REC_SRC),ip(REC_T),ip(REC_SLOT),ip(STARTS),ip(RN),
               ip(COMPACT_SRC),ip(COMPACT_T),ip(COMPACT_SLOT),
               ip(W1_CNT),i[NCOUNTERS],ip(UNUSED),i[ROLE1_CTAS]);
    } else {
    finalize(ip(COUNTS),i[E],i[CAP_E],i[CAP_E]*i[E],
             ip(REC_SRC),ip(REC_T),ip(REC_SLOT),ip(STARTS),ip(RN),
             ip(COMPACT_SRC),ip(COMPACT_T),ip(COMPACT_SLOT),
             ip(W1_CNT),i[NCOUNTERS]);
    }
    math(p[POOL],p[POOL_SF],p[W1],p[SFW1],p[W2],p[SFW2],p[STARTS],p[X_STARTS],
         p[COUNTS],p[UNUSED],p[UNUSED],p[REC_W],p[ACT],p[ACT_SF],p[OUT],
         i[E],i[I],i[H],i[TOPK],ip(RN)+1,p[W1_CNT],i[NCTA]);
    }
#ifndef GC_FUSED_COMBINE_RESET
    barrier(first_barrier_generation+1,true);
#endif
    int combine_record_offset = c[4];
#ifdef GC_FUSED_COMBINE_RESET
    combine_record_offset = -(first_barrier_generation+1);
#endif
    auto finish = [&](auto launcher, auto... extra) {
        launcher(p[SYM], peer, i[RANKS], i[MY], fp(OUT), ip(COMPACT_SRC), ip(COMPACT_T),
                 ip(COMPACT_SLOT), ip(RN), i[CAP_E], i[H], c[3], combine_record_offset,
                 c[1], c[2], i[CAP], ip(CURSORS), ip(DONE), output, i[T], i[TOPK], 1,
                 extra...);
    };
    if (start_reset && f[5]) {
        using Combine = void (*)(void*, const long long*, int, int, const float*,
                                 const int*, const int*, const int*, const int*, int, int,
                                 int, int, int, int, int, int*, int*, float*, int, int, int, void*);
        finish(reinterpret_cast<Combine>(f[5]), p[STREAM]);
    } else {
        finish(combine);
    }
}
void gc_moe_pipeline(void** p,const int* i,const long long* peer,const int* c,
                     void** f,int gen,float* output) {
    pipeline_impl(p,i,peer,c,f,gen,output,false);
}
void gc_moe_pipeline_role12(void** p,const int* i,const long long* peer,const int* c,
                            void** f,int gen,float* output) {
    pipeline_impl(p,i,peer,c,f,gen,output,true);
}
#ifdef GC_ROLE12_START_RESET
void gc_moe_pipeline_role12_start_reset(void** p,const int* i,const long long* peer,
                                        const int* c,void** f,int gen,float* output) {
    pipeline_impl(p,i,peer,c,f,gen,output,true,true);
}
#endif
int gc_moe_pipeline_local_combine_reset() {
#ifdef GC_FUSED_COMBINE_RESET
    return 2;
#elif defined(GC_LOCAL_COMBINE_RESET)
    return 1;
#else
    return 0;
#endif
}
}

extern "C" void gc_moe_pipeline_role12_combine(
    void** p, const int* i, const long long* peer, const int* c,
    void** f, int gen, void* output) {
    using Launcher = void (*)(void**, const int*, const long long*, const int*, int, void*);
    reinterpret_cast<Launcher>(f[3])(p, i, peer, c, gen + 1, output);
}

extern "C" int gc_moe_pipeline_mxfp4_input() { return 1; }
