// PPU 8-rank cross-rank communication layer — role1 dispatch push + role3
// combine scatter. Standalone library.
//
// Communication protocol:
//   * One symmetric memory buffer per rank (torch symm_mem / SAILSHMEM, set up
//     ONCE by python).  Peer address = local base + host-side offset table,
//     passed BY VALUE into the kernel (SymBuffer contract).
//   * Publish: every warp leader __ppu_threadfence_system_uncache() (PPU fence
//     drains only the issuing warp), then the LAST CTA (local ticket counter)
//     fires ppu.atom.add.release.sys atomics on the OWNER's count/gen words.
//   * Subscribe: ppu.ld.acquire.sys spin + __threadfence() subscriber fence,
//     then consume with L2 (L1-bypassing) loads — L2/LLC is the coherence
//     point for peer stores.
//   * No shared cross-rank cursor: each sender advances its OWN local cursor
//     array. Dispatch writes its OWN sender partition; combine writes the
//     uniquely owned (token, topk-slot) coordinates. Cross-rank
//     atomics are used exclusively for count/tag publication (bounded values),
//     never for cursors or data accumulation.
//
// DATA-PATH FACTS (probe-validated on 8x M890P, tests/probe/ppu_remote_rw_probe[34].cu):
//   * The `bulk b32x4` intrinsics are WARP-TILE ops, NOT per-lane 16B ops:
//     executed warp-wide with a warp-uniform base, reg j of lane k moves the
//     4B word at byte offset 4*(32*j + k) of a 512B tile (lane k's uint4 =
//     bytes {4k, 128+4k, 256+4k, 384+4k}).  Single-lane / divergent / uniform-
//     address misuse silently drops or replicates words.  Correct idiom:
//       uint4 v = __ppu_global_ldcg_bulk_b32x4(src);      // tile read (own VA)
//       __ppu_remote_store_bulk_volatile_b32x4(dst, v);   // tile write (peer VA)
//     = byte-exact 512B memcpy in 2 warp instructions (probe4).
//   * PTX st.global.cg.v4.b32 / ld.global.cg.v4.b32 are ordinary PER-LANE 16B
//     ops and work on peer-mapped addresses (probe3 R5/R6, probe4).  Used for
//     16B records and sub-512B payloads.
//
// Arena layout consequences (layout/ppu_gc_moe.py::comm_layout):
//   * pool row stride hb (=H/2) and BF16 cbuf row stride 2H are multiples of
//     512B for both models (3072/2560, 12288/10240).
//   * sf rows are padded to a 512B stride (sb = H/32 = 192/160B is not a tile
//     multiple); senders tile-write the full 512B, receivers consume the
//     first sb bytes.
//
// Call protocol (host side, per call): python zeroes the ctl words (cnt/gen,
// ticket/pingpong flags) and the scratch (cursors, done), synchronizes, and
// dist-barriers before launching.  This keeps the kernels generation-free and
// dead-lock free (push kernels never spin; finalize kernels only spin on
// remote progress, so any grid size is safe).
//
// hgcc codegen workarounds: all STORE index math in int32 (64-bit size_t store
// addresses silently miscompile on hgcc); remote-store destination computed
// as base + (int64)int32_offset (DeepGEMM-proven form).

#include <cassert>
#include "common.cuh"

#ifndef GC_DISPATCH_CTAS
#define GC_DISPATCH_CTAS 156
#endif

#ifndef GC_COMBINE_CTAS
#define GC_COMBINE_CTAS 156
#endif
static_assert(GC_COMBINE_CHUNK_ELEMS >= 256 && GC_COMBINE_CHUNK_ELEMS % 256 == 0,
              "combine chunk must contain whole256-element wire tiles");
static_assert(GC_DISPATCH_CTAS > 0 && GC_COMBINE_CTAS > 0, "nonempty comm grids");

// ---------------------------------------------------------------------------
// Address translation table (passed by value; offsets[peer] = peer base - mine)
// ---------------------------------------------------------------------------
struct PeerOffs {
    int64_t o[8];
};

__device__ __forceinline__ int ld_acquire_sys(volatile int* p)
{
    int v;
    asm volatile("ppu.ld.acquire.sys.global.s32 %0, [%1];"
                 : "=r"(v) : "l"((void*)p) : "memory");
    return v;
}

// Fire-and-forget release add (returns nothing).
__device__ __forceinline__ void atom_add_release_sys(volatile int* p, int v)
{
    asm volatile("ppu.atom.add.release.sys.global.s32 _, [%0], %1;"
                 :: "l"((void*)p), "r"(v) : "memory");
}

__device__ __forceinline__ unsigned f2u(float f)
{
    union { float f; unsigned u; } c; c.f = f; return c.u;
}
__device__ __forceinline__ float u2f(unsigned u)
{
    union { float f; unsigned u; } c; c.u = u; return c.f;
}

// ---- bf16 payload codec for the combine fabric --------------------------------
// The deepep baseline moves combine rows in bf16; doing the same halves the
// PCIe bytes of both the push and the finalize read (T=256: 50MB -> 25MB per
// direction per rank).  RNE conversion via bit ops (no cvt instructions --
// hgcc nibble/merge traps don't apply to shifts).
__device__ __forceinline__ unsigned f32_to_bf16(unsigned u)
{
    // round-to-nearest-even on the low 16 mantissa bits
    const unsigned r = ((u >> 16) & 1u) + 0x7FFFu;
    return (u + r) >> 16;
}
__device__ __forceinline__ uint32_t sm100_bf16_pair(float low, float high)
{
    uint32_t packed;
    asm("ppu.cvt.rtte.bf16x2.f32 %0, %1, %2;"
        : "=r"(packed) : "f"(high), "f"(low));
    return packed;
}

__device__ __forceinline__ unsigned pack_bf16(unsigned w_lo, unsigned w_hi)
{
    return sm100_bf16_pair(__uint_as_float(w_lo), __uint_as_float(w_hi));
}
__device__ __forceinline__ float bf16_to_f32(unsigned w)
{
    // __uint_as_float compiles to a register move; a union bit-cast goes
    // through local memory on hgcc (measured: finalize 4x slower).
    return __uint_as_float(w << 16);
}

// ---- per-lane 16B ops (correct for single lanes AND peer-mapped addresses) --
__device__ __forceinline__ void vst16(void* dst, uint4 v)
{
    asm volatile("st.global.cg.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "l"(dst), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}

__device__ __forceinline__ uint4 vld16(const void* src)
{
    uint4 v;
    asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(src) : "memory");
    return v;
}

// ---- warp-tile ops: 512B per instruction, ALL 32 LANES must execute with a
// WARP-UNIFORM base.  reg j of lane k <-> tile byte offset 4*(32*j + k). ------
__device__ __forceinline__ uint4 tile_ld(const void* src)
{
    return __ppu_global_ldcg_bulk_b32x4(src);
}

__device__ __forceinline__ void tile_st_rmt(void* dst, uint4 v)
{
    __ppu_remote_store_bulk_volatile_b32x4(
        reinterpret_cast<volatile void*>(dst), v);
}

// End-of-kernel publication: all CTAs of this grid have finished their remote
// stores; every warp leader fences (PPU fence = issuing warp only, so every
// warp must drain itself), then the LAST arriving CTA publishes
// (rows-from-me, generation=1) into every rank's ctl words.  The publication
// atoms fire from threads 0..ranks-1 in PARALLEL -- a serial loop burns
// ~2 ranks PCIe round trips per atom pair.
__device__ __forceinline__ void publish_counts(
    void* sym, const PeerOffs& po, int ranks, int my,
    int cnt_off, int gen_off, int* cursors, int* done, bool gen_only = false)
{
    __shared__ int s_pub;
    __syncwarp();
    if ((threadIdx.x & 31u) == 0)
        __ppu_threadfence_system_uncache();   // per-warp drain (PPU fence = issuing warp only)
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned last = gridDim.x - 1;
        if (gen_only) {
            // All CTAs arrive exactly once. The last CTA restores zero before
            // publishing. The next push on this stream starts after it exits.
            s_pub = atomicInc(reinterpret_cast<unsigned*>(done + 1), last) == last;
        } else {
            s_pub = atomicAdd(done, 1) == last;
        }
    }
    __syncthreads();
    if (s_pub && threadIdx.x < ranks) {
        const int d = threadIdx.x;
        volatile int* genp =
            (volatile int*)((char*)sym + (int64_t)(gen_off + 4 * my) + po.o[d]);
        if (!gen_only) {
            const int n = *(volatile int*)&cursors[d];
            volatile int* cntp =
                (volatile int*)((char*)sym + (int64_t)(cnt_off + 4 * my) + po.o[d]);
            atom_add_release_sys(cntp, n);
        }
        atom_add_release_sys(genp, 1);   // published for this call
    }
}

// Subscriber: wait until every sender published this call's generation, then
// hand out per-sender row counts + the running base offsets. Independent
// sender words are polled by separate lanes, then prefix-summed locally.
__device__ __forceinline__ void wait_all_senders(
    void* sym, int ranks, int cnt_off, int gen_off, int* s_base)
{
    if (threadIdx.x < ranks) {
        const int s = threadIdx.x;
        volatile int* genp = (volatile int*)((char*)sym + (int64_t)(gen_off + 4 * s));
        while (ld_acquire_sys(genp) < 1)
            __nanosleep(128);
        __threadfence();   // subscriber fence AFTER observing this sender
        volatile int* cntp = (volatile int*)((char*)sym + (int64_t)(cnt_off + 4 * s));
        s_base[s + 1] = ld_acquire_sys(cntp);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        s_base[0] = 0;
        for (int s = 1; s <= ranks; ++s)
            s_base[s] += s_base[s - 1];
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Dispatch push (role1).  Warp per (token, slot).  Expert e is owned by rank
// e / e_local.  Remote rows land in the OWNER's arena partition [my] (only
// this rank ever writes that partition, via its OWN local cursor -> no shared
// cross-rank cursor).  Local rows go straight into the local partition.
// Record per row: {t, slot|src<<8, gate_w, e_local} (16B, one v4 store).
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256)
gc_comm_dispatch_push_kernel(const uint8_t* x_packed, const uint8_t* x_sf,
                             const int* topk_idx, const float* topk_w,
                             void* sym, PeerOffs po, int ranks, int my,
                             int T, int topk, int hidden, int e_local,
                             int pool_off, int sf_off, int rec_off,
                             int cnt_off, int gen_off, int cap,
                             int* cursors, int* done)
{
    const int hb = hidden >> 1;        // packed bytes / row
    const int sb = hidden >> 5;        // scale bytes / row
    const int sbw = sb >> 2;           // scale words / row
    const int lane = threadIdx.x & 31;
    const int wpg = gridDim.x * (blockDim.x >> 5);
    const int total = T * topk;

    for (int i = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
         i < total; i += wpg) {
        const int t = i / topk;
        const int k = i - t * topk;
        const int e = topk_idx[i];
        const int owner = e / e_local;
        int slot = 0;
        if (lane == 0) slot = atomicAdd(&cursors[owner], 1);
        slot = __shfl_sync(0xFFFFFFFFu, slot, 0);
        const int prow = my * cap + slot;   // partition [SENDER] in owner arena (own cursor only)
        const bool local = (owner == my);

        // ---- packed row (hb bytes, multiple of 512) ----
        const char* srcrow = (const char*)x_packed + (int64_t)t * hb;
        if (local) {
            const int n16 = hb >> 4;
            char* dst = (char*)sym + (int64_t)pool_off + (int64_t)(prow * hb);
            for (int c = lane; c < n16; c += 32)
                vst16(dst + (int64_t)(c << 4), vld16(srcrow + (int64_t)(c << 4)));
        } else {
            char* dst = (char*)sym + (int64_t)pool_off + (int64_t)(prow * hb) + po.o[owner];
            for (int off = 0; off < hb; off += 512)   // warp-uniform
                tile_st_rmt(dst + (int64_t)off, tile_ld(srcrow + (int64_t)off));
        }

        // ---- scale row (sb bytes; arena row stride is padded to 512) ----
        const char* srow = (const char*)x_sf + (int64_t)t * sb;
        if (local) {
            const int n16 = sb >> 4;
            char* dst = (char*)sym + (int64_t)sf_off + (int64_t)(prow * 512);
            for (int c = lane; c < n16; c += 32)
                vst16(dst + (int64_t)(c << 4), vld16(srow + (int64_t)(c << 4)));
        } else {
            char* dst = (char*)sym + (int64_t)sf_off + (int64_t)(prow * 512) + po.o[owner];
            // build the 512B tile from 4 predicated scalar loads (src row is
            // only sb bytes; padding lanes contribute 0)
            uint4 v;
            v.x = (lane < sbw) ? *(const unsigned*)(srow + 4 * lane) : 0u;
            v.y = (32 + lane < sbw) ? *(const unsigned*)(srow + 128 + 4 * lane) : 0u;
            v.z = (64 + lane < sbw) ? *(const unsigned*)(srow + 256 + 4 * lane) : 0u;
            v.w = (96 + lane < sbw) ? *(const unsigned*)(srow + 384 + 4 * lane) : 0u;
            tile_st_rmt(dst, v);
        }

        // ---- record {t, slot|src<<8, w, e_local} (16B, one v4 store) ----
        // e_local = LOCAL expert id within the owner's shard (e - owner*
        // e_local), so the receiver can group rows per local expert without
        // the sender's routing table.  slot fits 8 bits (topk <= 256), src
        // fits 8 bits (ranks <= 256).
        if (lane == 0) {
            uint4 rec;
            rec.x = (unsigned)t;
            rec.y = (unsigned)k | ((unsigned)my << 8);
            rec.z = f2u(topk_w[i]);
            rec.w = (unsigned)(e - owner * e_local);
            char* dst = (char*)sym + (int64_t)rec_off + (int64_t)(prow * 16);
            if (local)
                vst16(dst, rec);
            else
                vst16(dst + po.o[owner], rec);
        }
    }
    publish_counts(sym, po, ranks, my, cnt_off, gen_off, cursors, done);
}

// Dispatch receive: wait all senders, then re-emit the arena rows GROUPED BY
// LOCAL EXPERT (warp per row).  Output row index = e*cap_e + c where c comes
// from a per-expert atomic cursor; when the kernel completes the cursor IS
// the row count for that expert, so the compute side gets row_starts[e] =
// e*cap_e and row_counts[e] = e_counts[e] with no python argsort/gather/scan.
// cap_e = ranks*T*topk is the unconditional worst case (every row to one
// expert); with the usual distinct-topk-per-token routing invariant, ranks*T
// also bounds any single expert.
extern "C" __global__ void __launch_bounds__(256)
gc_comm_dispatch_finalize_kernel(void* sym, PeerOffs po, int ranks, int hidden,
                                 int pool_off, int sf_off, int rec_off,
                                 int cnt_off, int gen_off, int cap,
                                 int e_local, int cap_e, int* e_counts,
                                 uint8_t* out_pool, uint8_t* out_sf,
                                 int* out_t, int* out_slot, int* out_src,
                                 float* out_w, int* out_e, int* out_rows)
{
    __shared__ int s_base[9];
    wait_all_senders(sym, ranks, cnt_off, gen_off, s_base);
    const int total = s_base[ranks];
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *out_rows = total;

    const int hb = hidden >> 1;
    const int sb = hidden >> 5;
    const int sbw = sb >> 2;
    const int lane = threadIdx.x & 31;
    const int wpg = gridDim.x * (blockDim.x >> 5);

    for (int g = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
         g < total; g += wpg) {
        int s = 0;
        while (g >= s_base[s + 1]) ++s;
        const int j = g - s_base[s];
        const int prow = s * cap + j;

        // record first: {t, slot|src<<8, w, e_local} decides the output slot
        int t = 0, kslot = 0, src = 0, e = 0, orow = 0;
        float wv = 0.f;
        if (lane == 0) {
            const uint4 rec = vld16((const void*)((char*)sym + (int64_t)rec_off + (int64_t)(prow * 16)));
            t = (int)rec.x;
            kslot = (int)(rec.y & 0xFFu);
            src = (int)((rec.y >> 8) & 0xFFu);
            wv = u2f(rec.z);
            e = (int)rec.w;
            const int c = atomicAdd(&e_counts[e], 1);
            orow = e * cap_e + c;          // int32: e*cap_e*hb < 2^31 (layout)
        }
        orow = __shfl_sync(0xFFFFFFFFu, orow, 0);

        // packed row: one 512B tile per iteration; per lane 4 coalesced words
        const char* srcp = (const char*)sym + (int64_t)pool_off + (int64_t)(prow * hb);
        unsigned* dstp = (unsigned*)(out_pool + (int64_t)orow * hb);
        for (int off = 0; off < hb; off += 512) {   // warp-uniform
            const uint4 v = tile_ld(srcp + (int64_t)off);
            unsigned* o = dstp + (off >> 2);
            o[lane] = v.x;
            o[32 + lane] = v.y;
            o[64 + lane] = v.z;
            o[96 + lane] = v.w;
        }

        // scale row: one tile; consume first sbw words
        const uint4 v = tile_ld((const void*)((char*)sym + (int64_t)sf_off + (int64_t)(prow * 512)));
        unsigned* dsts = (unsigned*)(out_sf + (int64_t)orow * sb);
        dsts[lane] = v.x;                          // lane < 32 <= sbw always
        if (32 + lane < sbw)
            dsts[32 + lane] = v.y;

        // record fields -> grouped outputs (lane 0 holds them)
        if (lane == 0) {
            out_t[orow] = t;
            out_slot[orow] = kslot;
            out_src[orow] = src;
            out_w[orow] = wv;
            out_e[orow] = e;
        }
    }
}

// ---------------------------------------------------------------------------
// Combine scatter (role3).  Each computed row has exactly one producer (the
// expert owner), keyed (token-owner rank, t, slot) -> no data write collisions
// on the fabric. Warp per row, or (row,256-hidden chunk) with
// GC_COMBINE_CHUNKS; destination row is TOKEN-MAJOR
// (t*topk + slot, coordinate-keyed: no cursor allocation, no
// arrival-order dependence) so the receive side sums topk consecutive rows
// per token with plain loads and one store per element -- the old
// sender-partition form forced a per-element float atomicAdd on receive
// (topk-way RMW conflicts on every out element; 12.6M atomics at glm T=256).
// ---------------------------------------------------------------------------
// r_count comes from DEVICE memory (compact_rn[0], written by the grouped
// finalize kernel) so python never syncs for it: the grid is sized from the
// static bound and surplus CTAs exit immediately.
template<bool RowWire>
__global__ void __launch_bounds__(256)
gc_comm_combine_push_kernel(const float* rows, const int* rec_owner,
                            const int* rec_t, const int* rec_slot,
                            void* sym, PeerOffs po, int ranks, int my,
                            const int* r_count_dev, int r_bound, int hidden,
                            int cbuf_off, int crec_off, int cnt_off, int gen_off,
                            int cap, int* cursors, int* done,
                            int topk)
{
    const bool gen_only = crec_off < 0;
    __shared__ int s_r;
    if (threadIdx.x == 0)
        s_r = *r_count_dev;
    __syncthreads();
    const int r_count = s_r < r_bound ? s_r : r_bound;

    const int rowb = hidden << 2;     // fp32 row bytes (multiple of 1024)
    const int rowb2 = hidden << 1;    // bf16 row bytes (multiple of 512)
    const int lane = threadIdx.x & 31;
    const int wpg = gridDim.x * (blockDim.x >> 5);

    const int chunks = (hidden + GC_COMBINE_CHUNK_ELEMS - 1) / GC_COMBINE_CHUNK_ELEMS;
    for (int task = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
         task < r_count * chunks; task += wpg) {
        const int r = task / chunks;
        const int begin = (task - r * chunks) * (4 * GC_COMBINE_CHUNK_ELEMS);
        const int limit = begin + 4 * GC_COMBINE_CHUNK_ELEMS;
        const int end = limit < rowb ? limit : rowb;
        const int owner = rec_owner[r];
        // token-major destination (see header comment); the cursor is now an
        // arrival COUNT only (feeds publish_counts; slots no longer allocated)
        const int t = rec_t[r];
        const int slot = rec_slot[r];
        const int prow = t * topk + slot;
        if (!gen_only && lane == 0 && begin == 0) atomicAdd(&cursors[owner], 1);
        const char* srcrow = (const char*)rows + (int64_t)r * rowb;
        const bool local = (owner == my);
        char* dst = (char*)sym + (int64_t)cbuf_off + (int64_t)(prow * rowb2);
        if constexpr (RowWire) {
            const char* wire = reinterpret_cast<const char*>(rows) + int64_t(r) * rowb2;
            if (local) {
                auto load = [](const char* address) {
                    uint64_t words;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                                 : "=l"(words) : "l"(address) : "memory");
                    return words;
                };
                for (int off = begin >> 1; off < (end >> 1); off += 512) {
                    const char* half = wire + off + 8 * lane;
                    const uint64_t a = load(half);
                    const uint64_t b = load(half + 256);
                    const uint4 packed{
                        uint32_t(a & 0xffffu) | uint32_t((b & 0xffffu) << 16),
                        uint32_t((a >> 16) & 0xffffu) | uint32_t(b & 0xffff0000u),
                        uint32_t((a >> 32) & 0xffffu) | uint32_t((b >> 16) & 0xffff0000u),
                        uint32_t(a >> 48) | uint32_t((b >> 32) & 0xffff0000u),
                    };
                    vst16(dst + off + 16 * lane, packed);
                }
            } else {
                dst += po.o[owner];
                auto pack_pair = [](const uint16_t* base, int element) {
                    return uint32_t(base[element]) | (uint32_t(base[element + 128]) << 16);
                };
                for (int off = begin >> 1; off < (end >> 1); off += 512) {
                    const auto* base = reinterpret_cast<const uint16_t*>(wire + off);
                    const uint4 packed{pack_pair(base, lane), pack_pair(base, lane + 32),
                                       pack_pair(base, lane + 64), pack_pair(base, lane + 96)};
                    tile_st_rmt(dst + off, packed);
                }
            }
        } else if (local) {
            // pack 1024B of f32 -> 512B of bf16 per warp iteration.  Pairing
            // is (element q, element q+128): a lane's two unpacked atomics
            // then land 512B apart and each warp instruction touches 32
            // CONTIGUOUS output elements (the f32 kernel's coalesced atomic
            // pattern -- adjacent-element pairing serializes the RMWs).
            for (int base = begin; base < end; base += 1024) {
                const uint4 a = vld16(srcrow + base + (int64_t)(16 * lane));
                const uint4 b = vld16(srcrow + base + (int64_t)(512 + 16 * lane));
                uint4 o;
                o.x = pack_bf16(a.x, b.x);
                o.y = pack_bf16(a.y, b.y);
                o.z = pack_bf16(a.z, b.z);
                o.w = pack_bf16(a.w, b.w);
                vst16(dst + (int64_t)(base >> 1) + (int64_t)(16 * lane), o);
            }
        } else {
            dst += po.o[owner];
            // packed word 32*j+k = elements 32*j+k and 32*j+k+128 = input
            // words 32*j+k (tile A reg j lane k) and 128+32*j+k (tile B reg
            // j lane k): same lane, same reg -> no shuffles, both bulk ops.
            for (int base = begin; base < end; base += 1024) {   // warp-uniform
                const uint4 a = tile_ld(srcrow + base);
                const uint4 b = tile_ld(srcrow + base + 512);
                uint4 o;
                o.x = pack_bf16(a.x, b.x);
                o.y = pack_bf16(a.y, b.y);
                o.z = pack_bf16(a.z, b.z);
                o.w = pack_bf16(a.w, b.w);
                tile_st_rmt(dst + (int64_t)(base >> 1), o);
            }
        }
        // (no crec write: the token-major position IS the record -- the
        // receive side derives (t, slot) from the row index)
    }
    publish_counts(sym, po, ranks, my, cnt_off, gen_off, cursors, done, gen_only);
}

__device__ __forceinline__ void acquire_combine_peers(void* sym, int ranks, int gen_off, int generation)
{
    if (threadIdx.x < ranks) {
        auto* ready = reinterpret_cast<volatile int*>(static_cast<char*>(sym) + gen_off)
            + threadIdx.x;
        while (ld_acquire_sys(ready) < generation) __nanosleep(128);
        __threadfence();
    }
    __syncthreads();
}

// Combine receive: owner waits all producers, then reduces.  TOKEN-MAJOR
// arena (push writes row t*topk+slot): token t's topk contributions are
// CONSECUTIVE rows. Each (token, 256-element chunk) is an independent warp
// task that accumulates all topk rows in registers and stores once (the old
// arrival-order form did one float atomicAdd per element with topk-way
// conflicts).  atomic_add=1 (pipeline mode): out is [out_T, hidden] and NO
// separate topk-reduce pass is needed.  atomic_add=0 (one-producer-per-key
// test mode): out[t, slot, :], one plain copy per row.
template<int LoadAhead, bool RowWire = false, bool BulkOutput = false, int FixedTopk = 0>
__global__ void __launch_bounds__(256)
gc_comm_combine_finalize_kernel(void* sym, PeerOffs po, int ranks, int hidden,
                                int cbuf_off, int crec_off, int cnt_off, int gen_off,
                                int cap, float* out, int out_T, int runtime_topk,
                                int atomic_add, int generation)
{
    const int topk = FixedTopk ? FixedTopk : runtime_topk;
    __shared__ int s_base[9];
    // The fused-reset pipeline places each payload directly at (token, slot).
    // All peers still publish and acquire completion, including empty senders.
    if (crec_off < 0)
        acquire_combine_peers(sym, ranks, gen_off, generation);
    else
        wait_all_senders(sym, ranks, cnt_off, gen_off, s_base);
    const int rowb2 = hidden << 1;    // bf16 row bytes (multiple of 512)
    const int lane = threadIdx.x & 31;
    const int wpg = gridDim.x * (blockDim.x >> 5);

    if (atomic_add) {
        // Independent (token, 256-element chunk) tasks. A warp-per-token
        // loop serialized 20/24 chunks and left most SMs idle at small T.
        const int chunks = hidden >> 8;
        for (int task = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
             task < out_T * chunks; task += wpg) {
            const int t = task / chunks;
            const int off = (task - t * chunks) << 9;
            const char* src0 = (const char*)sym + (int64_t)cbuf_off
                + (int64_t)t * topk * rowb2;
            // packed tile: word 32*j+k of lane k = elements 32*j+k and
            // 32*j+k+128 (the strided pairing set at pack time)
            {   // one warp-uniform packed chunk
                float acc[8];
                sum_combine_slots<LoadAhead>(src0 + off, rowb2, topk, acc);
                auto* o = reinterpret_cast<uint16_t*>(out) + int64_t(t) * hidden + (off >> 1);
                if constexpr (BulkOutput) {
                    const uint4 words{sm100_bf16_pair(acc[0], acc[4]),
                                      sm100_bf16_pair(acc[1], acc[5]),
                                      sm100_bf16_pair(acc[2], acc[6]),
                                      sm100_bf16_pair(acc[3], acc[7])};
                    __ppu_global_stcg_bulk_b32x4(o, words);
                } else {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const uint32_t pair = sm100_bf16_pair(acc[j], acc[j + 4]);
                        if constexpr (RowWire) {
                            reinterpret_cast<uint32_t*>(o)[32 * j + lane] = pair;
                        } else {
                            o[32 * j + lane] = uint16_t(pair);
                            o[128 + 32 * j + lane] = uint16_t(pair >> 16);
                        }
                    }
                }
            }
        }
    } else {
        // one row per (t, slot): plain copy, out is [out_T, topk, hidden]
        for (int g = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
             g < out_T * topk; g += wpg) {
            const char* src = (const char*)sym + (int64_t)cbuf_off
                + (int64_t)g * rowb2;
            float* o = out + (int64_t)g * hidden;
            for (int off = 0; off < rowb2; off += 512) {   // warp-uniform
                const uint4 v = tile_ld(src + (int64_t)off);
                o[(off >> 1) + lane] = bf16_to_f32(v.x & 0xFFFFu);
                o[(off >> 1) + 32 + lane] = bf16_to_f32(v.y & 0xFFFFu);
                o[(off >> 1) + 64 + lane] = bf16_to_f32(v.z & 0xFFFFu);
                o[(off >> 1) + 96 + lane] = bf16_to_f32(v.w & 0xFFFFu);
                o[(off >> 1) + 128 + lane] = bf16_to_f32(v.x >> 16);
                o[(off >> 1) + 160 + lane] = bf16_to_f32(v.y >> 16);
                o[(off >> 1) + 192 + lane] = bf16_to_f32(v.z >> 16);
                o[(off >> 1) + 224 + lane] = bf16_to_f32(v.w >> 16);
            }
        }
    }
}

template<int Hidden, int Topk>
__global__ void __launch_bounds__(128)
gc_comm_combine_fixed_kernel(void* sym, int ranks, int cbuf_off, int gen_off,
                             uint16_t* out, int tokens, int generation) {
    acquire_combine_peers(sym, ranks, gen_off, generation);
    constexpr int chunks = Hidden / 256;
    constexpr int row_bytes = Hidden * 2;
    constexpr int warps = 4;
    const auto* buffer = static_cast<const char*>(sym) + cbuf_off;
    for (int task = blockIdx.x * warps + (threadIdx.x >> 5);
         task < tokens * chunks; task += gridDim.x * warps) {
        const int token = task / chunks;
        const int chunk = task % chunks;
        const char* src = buffer + size_t(token) * Topk * row_bytes + chunk * 512;
        float acc[8];
        sum_combine_slots<Topk == 6 ? 6 : 4>(src, row_bytes, Topk, acc);
        const uint4 words{sm100_bf16_pair(acc[0], acc[4]), sm100_bf16_pair(acc[1], acc[5]),
                          sm100_bf16_pair(acc[2], acc[6]), sm100_bf16_pair(acc[3], acc[7])};
        __ppu_global_stcg_bulk_b32x4(out + size_t(task) * 256, words);
    }
}

// ---------------------------------------------------------------------------
// Device-side cross-rank barrier.  A gloo dist.barrier costs ~110us on this
// box; this kernel costs one launch + fabric RTT.  Monotonic GENERATION
// words: every call each rank atomically adds 1 to the barrier word of every
// rank (barr_off + 4*my, peer-translated), then spins until every word
// barr_off + 4*s >= ranks * gen.  Words are NEVER reset (reset() skips the
// barr region), so gen = per-rank python call counter.  Same-stream ordering
// makes the following launch wait for the barrier -- no host sync needed.
// ---------------------------------------------------------------------------
// Device-side all-rank barrier.  Rank my adds 1 to word[my] on EVERY rank's
// barr region (monotonic, never reset), so word[s] seen by an observer
// counts rank s's arrivals: wait until every word >= gen.  probe-validated
// (tests/probe_peer_atom.cu): the peer release-atomics and acquire loads all
// work; an early `want = ranks*gen` version deadlocked purely by logic.
// Reset+barrier fused into ONE launch (replaces ~8 python-side ops with
// per-op launch overhead).  Safety argument for zeroing BEFORE the arrival
// atom (no second phase): every word zeroed here -- dcnt/dgen (ccnt/cgen)
// consumed by MY dispatch (combine) finalize kernel, cursors/done/e_counts
// touched only by MY own kernels -- was already read by a kernel on MY
// stream that completed before this one started, and that consumer's
// acquire-spins observed every peer atom.  So the zero cannot clobber any
// in-flight peer write, and the arrival release orders the zeroed state for
// whoever acquires the barr word.  Threads 0..ranks-1 fire the peer RMWs in
// PARALLEL (a serial loop costs ~24us of PCIe round trips) and then each
// watches one word.
extern "C" __global__ void __launch_bounds__(32)
gc_comm_reset_barrier_kernel(void* sym, PeerOffs po, int ranks, int my,
                             int gen, int barr_off, int* cursors, int* done,
                             int* e_counts, int e_local, int dcnt_off,
                             int dgen_off, int ccnt_off, int cgen_off)
{
    // First reset already zeroed combine controls and acquired all peers.
    // Negative e_local is an opt-in host signal for local scratch reset.
    const bool local_combine = e_local < 0;
    int arrival_count = 1, wanted_generation = gen;
    if (gen < 0) {
        arrival_count = 2;
        wanted_generation = -gen;
    }
    if (threadIdx.x == 0) {
        int* b = (int*)sym;
        if (!local_combine) {
        for (int i = 0; i < ranks; ++i) {
            *(volatile int*)&b[(dcnt_off >> 2) + i] = 0;
            *(volatile int*)&b[(dgen_off >> 2) + i] = 0;
            *(volatile int*)&b[(ccnt_off >> 2) + i] = 0;
            *(volatile int*)&b[(cgen_off >> 2) + i] = 0;
        }
        }
        for (int i = 0; i < ranks; ++i) cursors[i] = 0;
        *done = 0;
        for (int i = 0; i < e_local; ++i) e_counts[i] = 0;
        __ppu_threadfence_system_uncache();   // zero stores sys-visible
    }
    __syncthreads();
    // parallel arrival: thread d publishes to peer d's word[my]
    if (threadIdx.x < ranks) {
        const int d = threadIdx.x;
        volatile int* p =
            (volatile int*)((char*)sym + (int64_t)(barr_off + 4 * my) + po.o[d]);
        atom_add_release_sys(p, arrival_count);
        // then watch word[d]: thread d returns when rank d has arrived
        volatile int* w =
            (volatile int*)((char*)sym + (int64_t)(barr_off + 4 * d));
        if (!local_combine) {
            while (ld_acquire_sys(w) < wanted_generation) __nanosleep(128);
        }
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Grouped-finalize compaction (replaces a ~7-op python torch chain whose
// per-op launch overhead dominates at ~25us each).  Inputs are the dispatch
// outputs (sparse, grouped: row = e*cap_e + i).  Outputs:
//   starts[e]  (int32, COMPACT prefix sum of counts)
//   out_rn[2]  = { R (dense total), ngrp = ceil(max count / 64) } -- python
//                reads BOTH with one .tolist() sync
//   compact rec_src/rec_t/rec_slot at cidx = starts[e] + i (dump slot
//   rows_g absorbs unused sparse slots, exactly like the python scatter)
// Phase 1 (CTA 0, thread 0) emits starts/R; every CTA builds its own shared
// prefix so phase 2 has no cross-CTA dependency.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256)
gc_comm_finalize_kernel(const int* e_counts, int e_local, int cap_e,
                        int rows_g, const int* out_src, const int* out_t,
                        const int* out_slot, int* starts, int* out_rn,
                        int* rec_src_c, int* rec_t_c, int* rec_slot_c,
                        int* compute_counters, int counter_count,
                        int* workgroups, int tile_rows)
{
    // Same-stream compute starts only after this kernel completes. Reset
    // the per-invocation L1 publication counters without another launch.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < counter_count;
         i += gridDim.x * blockDim.x)
        compute_counters[i] = 0;
    __shared__ int s_pref[49];      // s_pref[e] = compact base of expert e
    if (threadIdx.x == 0) {
        int acc = 0, mx = 0;
        for (int e = 0; e < e_local; ++e) {
            s_pref[e] = acc;
            if (blockIdx.x == 0) starts[e] = acc;
            acc += e_counts[e];
            if (e_counts[e] > mx) mx = e_counts[e];
        }
        s_pref[e_local] = acc;
        if (blockIdx.x == 0) {
            out_rn[0] = acc;
            out_rn[1] = (mx + 63) >> 6;     // ngrp (64-row tiles, >=0 ok:
                                            // kernel treats 0 as skip via
                                            // task count; python clamps to 1)
            if (workgroups) {
                const int stride = (mx + tile_rows - 1) / tile_rows;
                int n = 0;
                for (int e = 0; e < e_local; ++e) {
                    const int groups = (e_counts[e] + tile_rows - 1) / tile_rows;
                    for (int g = 0; g < groups; ++g)
                        workgroups[n++] = e * stride + g;
                }
                out_rn[2] = n;
            }
        }
    }
    __syncthreads();

    // O(R) compaction: iterate COMPACT indices and binary-search the shared
    // prefix for the owning expert (the old form scanned the whole
    // cap_e*e_local sparse space -- sized for T_MAX -- costing ~95us at
    // every T; R is T-proportional).
    const int total = s_pref[e_local];
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < total;
         c += gridDim.x * blockDim.x) {
        int lo = 0, hi = e_local - 1;
        while (lo < hi) {
            const int mid = (lo + hi) >> 1;
            if (s_pref[mid + 1] <= c) lo = mid + 1; else hi = mid;
        }
        const int r = lo * cap_e + (c - s_pref[lo]);
        rec_src_c[c] = out_src[r];
        rec_t_c[c] = out_t[r];
        rec_slot_c[c] = out_slot[r];
    }
}

// ---------------------------------------------------------------------------
// Host launchers (extern "C", flat args).  Python owns the symmetric buffer
// and passes byte offsets computed by tests/ppu_comm_common.py.
// ---------------------------------------------------------------------------

static void fill_offs(PeerOffs& po, const long long* peer_offs, int ranks)
{
    for (int i = 0; i < 8; ++i)
        po.o[i] = (i < ranks && peer_offs) ? peer_offs[i] : 0;
}

extern "C" void gc_comm_dispatch_push(
    void* sym, const long long* peer_offs, int ranks, int my_rank,
    const void* x_packed, const void* x_sf, const int* topk_idx,
    const float* topk_w, int T, int topk, int hidden, int e_local,
    int pool_off, int sf_off, int rec_off, int cnt_off, int gen_off, int cap,
    int* cursors, int* done, int* e_counts, int cap_e,
    void* out_pool, void* out_sf, int* out_t, int* out_slot, int* out_src,
    float* out_w, int* out_e, int* out_rows)
{
    PeerOffs po;
    fill_offs(po, peer_offs, ranks);
    const int warps = (T * topk + 7) & ~7;      // >= 1 CTA
    const dim3 grid((unsigned)((warps + 7) >> 3));
    gc_comm_dispatch_push_kernel<<<grid, 256>>>(
        (const uint8_t*)x_packed, (const uint8_t*)x_sf, topk_idx, topk_w,
        sym, po, ranks, my_rank, T, topk, hidden, e_local,
        pool_off, sf_off, rec_off, cnt_off, gen_off, cap, cursors, done);
    // The worst-case row bound is ranks times the average receive count.
    // A bounded strided grid avoids thousands of redundant polling CTAs;
    // the kernel's row loop still covers arbitrarily skewed routing.
    const int receive_ctas = (ranks * T * topk + 7) >> 3;
    const dim3 grid2((unsigned)(receive_ctas < GC_DISPATCH_CTAS
                               ? receive_ctas : GC_DISPATCH_CTAS));
    gc_comm_dispatch_finalize_kernel<<<grid2, 256>>>(
        sym, po, ranks, hidden, pool_off, sf_off, rec_off, cnt_off, gen_off,
        cap, e_local, cap_e, e_counts, (uint8_t*)out_pool, (uint8_t*)out_sf,
        out_t, out_slot, out_src, out_w, out_e, out_rows);
}

template<bool RowWire>
static void combine_scatter(
    void* sym, const long long* peer_offs, int ranks, int my_rank,
    const float* rows, const int* rec_owner, const int* rec_t,
    const int* rec_slot, const int* r_count_dev, int r_bound, int hidden,
    int cbuf_off, int crec_off, int cnt_off, int gen_off, int cap,
    int* cursors, int* done, float* out, int out_T, int topk, int atomic_add)
{
    PeerOffs po;
    fill_offs(po, peer_offs, ranks);
    // bf16 packing works a 1024B input / 512B output tile pair per iteration
    assert(((hidden << 2) & 1023) == 0 && "hidden*4 must be a multiple of 1024");
    // r_bound is the all-rank worst case (typically 8x the actual local R).
    // Avoid draining/fencing thousands of empty CTAs before publication.
    // Empty senders must still publish a completion tag.
    // Use average routed rows only for grid sizing. The device-side strided
    // loop still covers the complete worst-case r_bound under arbitrary skew.
    const int chunks = (hidden + GC_COMBINE_CHUNK_ELEMS - 1) / GC_COMBINE_CHUNK_ELEMS;
    const int push_tasks = out_T * topk * chunks;
    const int push_ctas = push_tasks > 0 ? ((push_tasks + 7) >> 3) : 1;
    int selected_grid = push_ctas < GC_COMBINE_CTAS ? push_ctas : GC_COMBINE_CTAS;
    if (crec_off < 0) {
        if (atomic_add != 1) __builtin_trap();
        static const int capacity = [] {
            int device = 0, blocks = 0;
            hggcDeviceProp properties{};
            hggcGetDevice(&device);
            hggcGetDeviceProperties(&properties, device);
            hggcOccupancyMaxActiveBlocksPerMultiprocessor(
                &blocks, gc_comm_combine_push_kernel<RowWire>, 256, 0);
            return blocks * properties.multiProcessorCount;
        }();
        if (capacity < 1) __builtin_trap();
        if (selected_grid > capacity) selected_grid = capacity;
    }
    const dim3 grid((unsigned)selected_grid);
    // The receive reduction overwrites every output element in either mode.
    gc_comm_combine_push_kernel<RowWire><<<grid, 256>>>(
        rows, rec_owner, rec_t, rec_slot, sym, po, ranks, my_rank, r_count_dev,
        r_bound, hidden, cbuf_off, crec_off, cnt_off, gen_off, cap, cursors,
        done, topk);
    // Bound the polling fanout while exposing hidden chunks in parallel.
    const int tasks = out_T * (atomic_add ? (hidden >> 8) : topk);
    const int receive_ctas = (tasks + 7) >> 3;
    const dim3 grid2((unsigned)(receive_ctas < 156 ? receive_ctas : 156));
    if (crec_off < 0 && topk == 8) {
        gc_comm_combine_finalize_kernel<4><<<grid2, 256>>>(
            sym, po, ranks, hidden, cbuf_off, crec_off, cnt_off, gen_off, cap,
            out, out_T, topk, atomic_add, 1);
        return;
    }
    gc_comm_combine_finalize_kernel<1><<<grid2, 256>>>(
        sym, po, ranks, hidden, cbuf_off, crec_off, cnt_off, gen_off, cap,
        out, out_T, topk, atomic_add, 1);
}

extern "C" void gc_comm_combine_scatter(
    void* sym, const long long* peer_offs, int ranks, int my_rank,
    const float* rows, const int* rec_owner, const int* rec_t,
    const int* rec_slot, const int* r_count_dev, int r_bound, int hidden,
    int cbuf_off, int crec_off, int cnt_off, int gen_off, int cap,
    int* cursors, int* done, float* out, int out_T, int topk, int atomic_add)
{
    combine_scatter<false>(sym, peer_offs, ranks, my_rank,
        rows, rec_owner, rec_t, rec_slot, r_count_dev, r_bound, hidden,
        cbuf_off, crec_off, cnt_off, gen_off, cap, cursors, done, out, out_T, topk, atomic_add);
}

extern "C" void gc_comm_reset_barrier(
    void* sym, const long long* peer_offs, int ranks, int my_rank,
    int gen, int barr_off, int* cursors, int* done, int* e_counts,
    int e_local, int dcnt_off, int dgen_off, int ccnt_off, int cgen_off)
{
    PeerOffs po;
    fill_offs(po, peer_offs, ranks);
    gc_comm_reset_barrier_kernel<<<1, 32>>>(
        sym, po, ranks, my_rank, gen, barr_off, cursors, done, e_counts,
        e_local, dcnt_off, dgen_off, ccnt_off, cgen_off);
}
extern "C" int gc_comm_local_combine_reset() {
    return 2;
}

extern "C" void gc_comm_finalize(
    const int* e_counts, int e_local, int cap_e, int rows_g,
    const int* out_src, const int* out_t, const int* out_slot,
    int* starts, int* out_rn, int* rec_src_c, int* rec_t_c, int* rec_slot_c)
{
    const int grid = rows_g < 2048 ? 1 : 8;
    gc_comm_finalize_kernel<<<grid, 256>>>(
        e_counts, e_local, cap_e, rows_g, out_src, out_t, out_slot,
        starts, out_rn, rec_src_c, rec_t_c, rec_slot_c, nullptr, 0, nullptr, 0);
}

// Optional extended entry; the original ABI remains available to old callers.
extern "C" void gc_comm_finalize_reset(
    const int* e_counts, int e_local, int cap_e, int rows_g,
    const int* out_src, const int* out_t, const int* out_slot,
    int* starts, int* out_rn, int* rec_src_c, int* rec_t_c, int* rec_slot_c,
    int* compute_counters, int counter_count)
{
    const int grid = rows_g < 2048 ? 1 : 8;
    gc_comm_finalize_kernel<<<grid, 256>>>(
        e_counts, e_local, cap_e, rows_g, out_src, out_t, out_slot,
        starts, out_rn, rec_src_c, rec_t_c, rec_slot_c,
        compute_counters, counter_count, nullptr, 0);
}

// Optional dense workgroup schedule. Only this API requires out_rn[3].
extern "C" void gc_comm_finalize_groups(
    const int* e_counts, int e_local, int cap_e, int rows_g,
    const int* out_src, const int* out_t, const int* out_slot,
    int* starts, int* out_rn, int* rec_src_c, int* rec_t_c, int* rec_slot_c,
    int* compute_counters, int counter_count, int* workgroups, int tile_rows)
{
    assert(workgroups && tile_rows > 0);
    const int grid = rows_g < 2048 ? 1 : 8;
    gc_comm_finalize_kernel<<<grid, 256>>>(
        e_counts, e_local, cap_e, rows_g, out_src, out_t, out_slot,
        starts, out_rn, rec_src_c, rec_t_c, rec_slot_c,
        compute_counters, counter_count, workgroups, tile_rows);
}

// FC2 has already stored and published every contribution. The ABI matches
// the ordinary combine launcher, so the host submission order is unchanged.
static void combine_direct(
    void* sym, const long long* peer_offs, int ranks, int my_rank,
    const float* rows, const int* rec_owner, const int* rec_t,
    const int* rec_slot, const int* r_count_dev, int r_bound, int hidden,
    int cbuf_off, int crec_off, int cnt_off, int gen_off, int cap,
    int* cursors, int* done, float* out, int out_T, int topk, int atomic_add,
    int generation, hggcStream_t stream)
{
    if (crec_off >= 0 || atomic_add != 1) __builtin_trap();
    if (out_T == 0) return;
    PeerOffs po;
    fill_offs(po, peer_offs, ranks);
    const int tasks = out_T * (hidden >> 8);
    const int count = (tasks + 7) >> 3;
    const int grid = count < 156 ? count : 156;
    const bool bulk_output = (reinterpret_cast<uintptr_t>(out) & 511) == 0 && hidden % 256 == 0;
    if (bulk_output && out_T >= 128) {
        const int blocks = (tasks + 3) >> 2;
        const int fixed_grid = blocks < 156 ? blocks : 156;
        auto* result = reinterpret_cast<uint16_t*>(out);
        if (hidden == 6144 && topk == 8) {
            gc_comm_combine_fixed_kernel<6144, 8><<<fixed_grid, 128, 0, stream>>>(
                sym, ranks, cbuf_off, gen_off, result, out_T, generation);
            return;
        }
        if (hidden == 5120 && topk == 6) {
            gc_comm_combine_fixed_kernel<5120, 6><<<fixed_grid, 128, 0, stream>>>(
                sym, ranks, cbuf_off, gen_off, result, out_T, generation);
            return;
        }
    }
    // Load all six slots before accumulating them in the original FP32 order.
    if (bulk_output && topk == 6) {
        gc_comm_combine_finalize_kernel<6, true, true, 6><<<grid, 256, 0, stream>>>(
            sym, po, ranks, hidden, cbuf_off, crec_off, cnt_off, gen_off, cap,
            out, out_T, topk, atomic_add, generation);
    } else if (bulk_output) {
        gc_comm_combine_finalize_kernel<4, true, true><<<grid, 256, 0, stream>>>(
            sym, po, ranks, hidden, cbuf_off, crec_off, cnt_off, gen_off, cap,
            out, out_T, topk, atomic_add, generation);
    } else {
        gc_comm_combine_finalize_kernel<4, true><<<grid, 256, 0, stream>>>(
            sym, po, ranks, hidden, cbuf_off, crec_off, cnt_off, gen_off, cap,
            out, out_T, topk, atomic_add, generation);
    }
}

extern "C" void gc_comm_combine_direct(
    void* sym, const long long* peer_offs, int ranks, int my_rank,
    const float* rows, const int* rec_owner, const int* rec_t,
    const int* rec_slot, const int* r_count_dev, int r_bound, int hidden,
    int cbuf_off, int crec_off, int cnt_off, int gen_off, int cap,
    int* cursors, int* done, float* out, int out_T, int topk, int atomic_add)
{
    combine_direct(sym, peer_offs, ranks, my_rank, rows, rec_owner, rec_t, rec_slot,
                   r_count_dev, r_bound, hidden, cbuf_off, crec_off, cnt_off, gen_off,
                   cap, cursors, done, out, out_T, topk, atomic_add, 1, nullptr);
}

__global__ void gc_comm_empty_precleared_combine_kernel(void* sym, int ranks, int gen_off, int generation)
{
    acquire_combine_peers(sym, ranks, gen_off, generation);
}

extern "C" void gc_comm_combine_precleared(
    void* sym, const long long* peer_offs, int ranks, int my_rank,
    const float* rows, const int* rec_owner, const int* rec_t,
    const int* rec_slot, const int* r_count_dev, int r_bound, int hidden,
    int cbuf_off, int crec_off, int cnt_off, int gen_off, int cap,
    int* cursors, int* done, float* out, int out_T, int topk, int atomic_add, void* stream_ptr)
{
    const auto stream = reinterpret_cast<hggcStream_t>(stream_ptr);
    if (crec_off >= 0 || atomic_add != 1) __builtin_trap();
    // This ABI uses the negated barrier generation in place of a record offset.
    const int generation = -(crec_off / 2);
    if (out_T == 0) {
        // The next request may send before peers enter it. Even an empty
        // output rank must first acquire the current combine publications.
        gc_comm_empty_precleared_combine_kernel<<<1, 32, 0, stream>>>(sym, ranks, gen_off, generation);
        return;
    }
    combine_direct(sym, peer_offs, ranks, my_rank, rows, rec_owner, rec_t,
                           rec_slot, r_count_dev, r_bound, hidden, cbuf_off,
                   crec_off, cnt_off, gen_off, cap, cursors, done, out,
                   out_T, topk, atomic_add, generation, stream);
}
