// PPU GC MoE role12 kernel. Dispatch publishes expert rows to the FC1 and
// FC2 tasks. Role3 reduces the routed slots and any shared expert output.
#define FROLE12
#ifdef FROLE12_DIRECT_METADATA
#if !defined(FROLE12_PRECLEARED_DISPATCH) || !defined(FROLE12_DIRECT_ROUTES)
#error Early direct metadata requires precleared direct dispatch
#endif
#define FROLE12_EARLY_METADATA
#endif
#if !defined(FROLE12_PRECLEARED_DISPATCH)
#undef FROLE12_EARLY_METADATA
#undef FROLE12_W1_GROUP_RECORDS
#undef FROLE12_COMBINE
#endif
#if defined(FROLE12_EARLY_METADATA) && !defined(FROLE12_DIRECT_METADATA) && (defined(FROLE12_DIRECT_ROUTES) || !defined(FROLE12_READY_FC2))
#error Early grouped metadata requires grouped reservations and the FC2 ready queue
#endif
#if defined(FROLE12_W1_GROUP_RECORDS) && !defined(FROLE12_EARLY_METADATA)
#error FC1 group records require early metadata
#endif
// PPU MXFP4 expert math. FC1 publishes ready row groups to FC2, which
// writes each weighted BF16 contribution in the original combine layout.

#if defined(FROLE12_EXPERT_PAYLOAD) && (!defined(FROLE12_EARLY_METADATA) || !defined(FROLE12_PRECLEARED_DISPATCH))
#error Per-expert payload counts require early metadata and precleared dispatch
#endif
#include "common.cuh"
#include <type_traits>

#ifdef FROLE12_PRECLEARED_DISPATCH
constexpr int routed_scale_stride = ((FCONST_H / 32) + 255) & ~255;
#else
constexpr int routed_scale_stride = 512;
#endif

#if defined(FL1_FULL_RING) && (defined(FVBOTTOM) || defined(FISSUE_POST))
#error FL1_FULL_RING requires the top-wait path
#endif
#if (defined(FL1_REFILL_LAST_TWO) || defined(FL1_REFILL_ALL_FRAGMENTS)) && !defined(FL1_FULL_RING)
#error Early refill requires the full FC1 ring
#endif
#if defined(FL1_REFILL_LAST_TWO) && defined(FL1_REFILL_ALL_FRAGMENTS)
#error Select one FC1 refill point
#endif

#if defined(FROLE12_GROUPED_REMOTE) && (!defined(FROLE12) || !defined(FROLE12_META_READY_INPUT) || defined(FROLE12_DIRECT_INPUT) || defined(FROLE12_BUCKET_INPUT))
#error FROLE12_GROUPED_REMOTE requires its metadata-ready symmetric grouped binding
#endif

#if defined(FROLE12_WEIGHT_PANEL) && (!defined(FROLE12) || !defined(FL2B_SINGLE))
#error Weight panels require role12 with single-descriptor L2
#endif

#ifndef NBUF
#define NBUF 3
#endif

#ifndef FROLE12_POLL_NS
#define FROLE12_POLL_NS 128
#endif

#ifndef FTILE_ROWS
#define FTILE_ROWS 64
#endif
#ifdef FHALF_N
#define FTHREADS 128
#define FBCOLS 128
#else
#define FTHREADS 256
#define FBCOLS 256
#endif

static_assert(FTILE_ROWS <= FTHREADS, "scale staging threads");
static_assert(FTILE_ROWS > 0 && FTILE_ROWS % 16 == 0 && FTILE_ROWS <= 256, "m16 tile multiple");

// Default k-loop pipeline mode: FVBOTTOM (wait at the bottom of the tile,
// v11 op order).  It won every measurement vs the TOP-barrier form (~+70us)
// and FISSUE_POST (~+22us).  Other modes stay reachable for ablation:
// -DFISSUE_POST, or -DFNOVBOTTOM for the TOP-barrier form.
#if !defined(FVBOTTOM) && !defined(FISSUE_POST) && !defined(FNOVBOTTOM)
#define FVBOTTOM
#endif

namespace {

__device__ __forceinline__ int routed_input_row(int expert, int slot, int capacity)
{
#ifdef FROLE12_PRECLEARED_DISPATCH
    // Keep the first four tiles of each expert together. The remaining
    // rows retain the full per-expert capacity without a compaction pass.
    const int head = min(capacity, 4 * FTILE_ROWS);
    if (slot < head) return expert * head + slot;
    return head * FCONST_E + expert * (capacity - head) + (slot - head);
#else
    return expert * capacity + slot;
#endif
}

#ifdef FROLE12_SHARED_EXPERT
__device__ __forceinline__ int early_shared_fc1_tasks(int groups, int panels, int grid)
{
    if (groups * panels >= grid) return 0;
#ifdef FROLE12_CAP_SHARED_FC1
    // Reserve complete FC1 groups so each group can publish directly to FC2.
    groups = min(groups, max(1, grid / (3 * panels)));
#endif
    return groups * panels;
}
#endif

template <int ChunkBytes>
__device__ __forceinline__ size_t activation_offset(int row0, int row, int rows,
                                                   int row_bytes, int column)
{
#ifdef FROLE12_K_MAJOR_ACT
    // Each workgroup occupies rows * row_bytes bytes. Within that range,
    // pack one K chunk across all rows so FC2 can fetch it contiguously.
    return size_t(row0) * row_bytes +
           (size_t(column / ChunkBytes) * rows + row) * ChunkBytes + column % ChunkBytes;
#else
    return size_t(row0 + row) * row_bytes + column;
#endif
}

__device__ __forceinline__ uint32_t sm100_bf16_pair(float low, float high)
{
    uint32_t packed;
    asm("ppu.cvt.rtte.bf16x2.f32 %0, %1, %2;"
        : "=r"(packed) : "f"(high), "f"(low));
    return packed;
}

static __device__ float sm100_gate_table[65536];

__global__ void sm100_fill_gate_table()
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < 65536; i += gridDim.x * blockDim.x) {
        const float gate = __uint_as_float(uint32_t(i) << 16);
        sm100_gate_table[i] = gate / (1.f + expf(-gate));
    }
}

__device__ __forceinline__ void mma_m16n16k64_f4(
    float* d, const uint32_t* a, const uint32_t* b,
    uint32_t meta_a, uint32_t meta_b, uint32_t a_part = 0, uint32_t b_part = 0)
{
    asm volatile(
        "ppu.tc02.mma.mx.sync.aligned.m16n16k64.row.col.f32.f4.f4.f32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, "
        "{%12, %13, %14, %15}, {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%16, %17}, %18, %19, 0;"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "r"(meta_a), "r"(meta_b), "r"(a_part), "r"(b_part));
}

__device__ __forceinline__ uint32_t smem_addr32(const void* p)
{
    uint32_t a;
    asm("{.reg .u64 u; ppu.cvta.to.shared.u64 u, %1; ppu.cvt.u32.u64 %0, u; }"
        : "=r"(a) : "l"(p));
    return a;
}

__device__ __forceinline__ uint32_t smem_addr32(uint32_t address)
{
    return address;
}

__device__ __forceinline__ int activation_copy_rows(int rows)
{
#ifdef FROLE12_ACTIVE_A_ROWS
    // Keep full m16 blocks for the live rows. Unused blocks produce no output.
    return (rows + 15) & -16;
#else
    return FTILE_ROWS;
#endif
}

template <typename TsmAddress>
__device__ __forceinline__ void aiu_copy_b8_swzl(
    TsmAddress smem_dst, const void* gmem_base,
    int dim_w, int dim_h, int cube_w, int cube_h,
    uint64_t stride_w, int coord_w, int coord_h, int swzl_mode)
{
    const uint64_t tsn = stride_w * (uint64_t)dim_h;
    asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.LLC::128B.padz.swzl.b8"
        " [%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, {%10, %11, %12}, %13;\n"
        :: "r"(smem_addr32(smem_dst)), "l"(gmem_base),
           "r"(dim_w), "r"(dim_h), "r"(1),
           "r"(cube_w), "r"(cube_h), "r"(1),
           "l"(stride_w), "l"(tsn),
           "r"(coord_w), "r"(coord_h), "r"(0),
           "r"(swzl_mode));
}

template <typename TsmAddress>
__device__ __forceinline__ void aiu_copy_weight_b8(
    TsmAddress smem_dst, const void* gmem_base,
    int dim_w, int dim_h, int cube_w, int cube_h,
    uint64_t stride_w, int coord_w, int coord_h, int swzl_mode)
{
    const uint64_t tsn = stride_w * (uint64_t)dim_h;
    asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.rt.LLC::128B.2d.tile.padz.swzl.b8"
        " [%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, {%10, %11, %12}, %13;\n"
        :: "r"(smem_addr32(smem_dst)), "l"(gmem_base),
           "r"(dim_w), "r"(dim_h), "r"(1),
           "r"(cube_w), "r"(cube_h), "r"(1),
           "l"(stride_w), "l"(tsn),
           "r"(coord_w), "r"(coord_h), "r"(0),
           "r"(swzl_mode));
}

__device__ __forceinline__ uint32_t pack_fc1_activation_scale(
    uint32_t low_rows, uint32_t high_rows, int half)
{
#ifdef FROLE12_FC1_PACK_ASF
    return __byte_perm(low_rows, high_rows, half == 0 ? 0x5410 : 0x7632);
#else
    return ((low_rows >> (16 * half)) & 0xFFFFu)
        | (((high_rows >> (16 * half)) & 0xFFFFu) << 16);
#endif
}

template <typename TsmAddress>
__device__ __forceinline__ void aiu_copy_linear_b32(
    TsmAddress dst, const void* src, int width, int height, int tile_w,
    int tile_h, uint64_t pitch, int x, int y)
{
    const uint64_t stride_n = pitch * height;
    asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.rt.2d.tile.padz.linear.b32"
        " [%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, {%10, %11, %12}, %13;\n"
        :: "r"(smem_addr32(dst)), "l"(src),
           "r"(width), "r"(height), "r"(1),
           "r"(tile_w), "r"(tile_h), "r"(1), "l"(pitch), "l"(stride_n),
           "r"(x), "r"(y), "r"(0), "r"(0));
}

#ifdef FROLE12_W1_SCALE_PAIR
template <typename TsmAddress>
__device__ __forceinline__ void aiu_copy_w1_scale_pair(
    TsmAddress dst, const uint8_t* src, int inter, int n_base)
{
    const int row_words = inter;
    const int tile_words = FBCOLS;
    const int offset = n_base;
    const uint64_t pitch = uint64_t(row_words) * 4;
    // The K tile stores gate and up as two adjacent slices. Copy the same
    // columns from each slice into consecutive scale buffers.
    asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.rt.2d.tile.padz.linear.b32"
        " [%0], [%1], {%2, %3, %4}, {%5, %3, %4}, {%6, %6}, {%7, %8, %8}, %8;"
        :: "r"(smem_addr32(dst)), "l"(src), "r"(row_words), "r"(1), "r"(2),
           "r"(tile_words), "l"(pitch), "r"(offset), "r"(0));
}
#endif

#ifdef FL1_GATE_PAIR
// The descriptor's N slices select gate and up in the original weight
// allocation and write them to the consecutive shared-memory cubes.
__device__ __forceinline__ void aiu_copy_gate_up(
    const void* dst, const void* base, int kbytes, int inter,
    int cols, int k, int n)
{
    const uint64_t slice = uint64_t(kbytes) * inter;
    asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.rt.LLC::128B.2d.tile.padz.swzl.b8"
        " [%0], [%1], {%2, %3, %9}, {%10, %4, %9}, {%5, %6}, {%7, %8, %11}, %12;\n"
        :: "r"(smem_addr32(dst)), "l"(base), "r"(kbytes), "r"(inter), "r"(cols),
           "l"(uint64_t(kbytes)), "l"(slice), "r"(k), "r"(n),
           "r"(2), "r"(64), "r"(0), "r"(1));
}
#endif

__device__ __forceinline__ void ldmatrix_b_frag(
    uint32_t* vreg, const void* smem_logical, int lbo, int swzl_mode)
{
#ifdef FROLE12_LDM_B32
    const uint32_t tsm = smem_addr32(smem_logical) >> 4;
#define LDM_B_ADDR_CONSTRAINT "r"
#else
    const uint64_t tsm = (uint64_t)(uintptr_t)smem_logical >> 4;
#define LDM_B_ADDR_CONSTRAINT "l"
#endif
    asm volatile(
        "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 {%0, %1, %2, %3}, [%4], %5, 1, %6;"
        : "=r"(vreg[0]), "=r"(vreg[1]), "=r"(vreg[2]), "=r"(vreg[3])
        : LDM_B_ADDR_CONSTRAINT(tsm), "r"(lbo), "r"(swzl_mode));
#undef LDM_B_ADDR_CONSTRAINT
}

// A-operand fragment from an AIU-staged swizzled cube (probe-verified:
// cube [64 rows][64B], AIU swzl mode 1; lbo=1 / sbo=32 / swzl=1 yields the
// registers in exactly the mma's expected order -- the v11 hand-gather
// ac[mb][h][i] = A_tile[16mb + lane>>2 + 8*(i&1)]
//                [4*(lane&3) + 16*(i>>1) + 32*h]).
__device__ __forceinline__ void ldmatrix_a_frag(
    uint32_t* vreg, const void* smem_logical)
{
    const uint64_t tsm = (uint64_t)(uintptr_t)smem_logical >> 4;
    asm volatile(
        "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 {%0, %1, %2, %3},"
        " [%4], 1, 32, 1;"
        : "=r"(vreg[0]), "=r"(vreg[1]), "=r"(vreg[2]), "=r"(vreg[3])
        : "l"(tsm));
}


#ifdef FROLE12_A_TSM_OFFSET
// The LDM operand is a TSM address in 16-byte units.
__device__ __forceinline__ void ldmatrix_a_at(uint32_t* v, uint32_t address) {
    asm volatile(
        "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 {%0, %1, %2, %3}, [%4], 1, 32, 1;"
        : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]) : "r"(address));
}

#endif

#ifdef FROLE12_B_TSM_OFFSET
__device__ __forceinline__ void ldmatrix_b_at(uint32_t* v, uint32_t address) {
    asm volatile(
        "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 {%0, %1, %2, %3}, [%4], 32, 1, 1;"
        : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]) : "r"(address));
}

#endif

__device__ __forceinline__ void cp_async_fence()
{
    asm volatile("ppu.cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_all()
{
    asm volatile("ppu.cp.async.wait_group 0;\n");
}

#if defined(FL1_FULL_RING) || defined(FROLE12_W2_FULL_RING)
__device__ __forceinline__ void cp_async_wait_full_ring(int tile, int tiles)
{
    if (tile + NBUF - 1 < tiles)
        asm volatile("ppu.cp.async.wait_group %0;" :: "n"(NBUF - 1));
#ifdef FL1_EXACT_TAIL_WAIT
    else if (NBUF >= 4 && tile + 2 < tiles)
        asm volatile("ppu.cp.async.wait_group 2;");
    else if (NBUF >= 3 && tile + 1 < tiles)
        asm volatile("ppu.cp.async.wait_group 1;");
#endif
    else
        cp_async_wait_all();
}
#endif

// Wait until only the NBUF-2 MOST RECENT groups remain outstanding -- i.e.
// the oldest group (the tile being consumed) is complete.  wait_group N is
// "wait until <= N groups pending", so it is a NO-OP when fewer than N+1
// groups are in flight; the k-loop tail therefore falls back to wait-all.
// Pass the tile consumed AFTER this wait: top-wait uses ks, bottom-wait ks+1.
__device__ __forceinline__ void cp_async_wait_pipe(int ks, int ntiles)
{
#if NBUF > 2 && !defined(FWAITALL)
    if (ks + NBUF - 2 < ntiles)
        asm volatile("ppu.cp.async.wait_group %0;\n" :: "n"(NBUF - 2));
    else
        cp_async_wait_all();
#else
    (void)ks; (void)ntiles;
    cp_async_wait_all();
#endif
}

__device__ __forceinline__ uint8_t cvt_e2m1x2(float hi, float lo)
{
    uint16_t d;
    asm("ppu.cvt.rtte.satfinite.e2m1x2.f32 %0, %1, %2;"
        : "=h"(d) : "f"(hi), "f"(lo));
    return (uint8_t)d;
}

__device__ __forceinline__ int ceil_log2(float a)
{
    uint32_t u = __float_as_uint(a);
    int e = (int)((u >> 23) & 0xFF) - 127;
    return e + ((u & 0x7FFFFF) != 0 ? 1 : 0);
}

// v12 epilogue numerics (engine-proven, bit-exact on both models).
__device__ __forceinline__ float __ppu_bits_u2f(uint32_t u)
{
    union { uint32_t u; float f; } t;
    t.u = u;
    return t.f;
}

__device__ __forceinline__ float swiglu_val(float g, float u)
{
    const float denominator = 1.f + expf(-g);
    const float reciprocal = __fdividef(1.f, denominator);
    const float quotient = g * reciprocal;
    return fmaf(fmaf(-quotient, denominator, g), reciprocal, quotient) * u;
}

__device__ __forceinline__ void cnt_release(int* p, int v)
{
    asm volatile("ppu.atom.add.release.gpu.global.s32 _, [%0], %1;"
                 :: "l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ int cnt_acquire(const int* p)
{
    int v;
    asm volatile("ppu.ld.acquire.gpu.global.s32 %0, [%1];"
                 : "=r"(v) : "l"(p) : "memory");
    return v;
}

#ifdef FROLE12_READY_FC2
#ifdef FROLE12_SHARED_EXPERT
constexpr int group_expert_bits = FCONST_E == 64 ? 7 : 6;
#else
constexpr int group_expert_bits = 6;
#endif
constexpr int group_expert_mask = (1 << group_expert_bits) - 1;

__device__ __forceinline__ int finish_fc1_slab(int* count)
{
    int old_count;
    asm volatile("ppu.atom.add.release.gpu.global.s32 %0, [%1], %2;"
                 : "=r"(old_count) : "l"(count), "r"(1) : "memory");
    return old_count;
}
#endif

}  // namespace

// ---------------------------------------------------------------------------
// L1: fused expert FFN (gate+up) + SwiGLU + per-32-col requant to mxfp4.
// CTA tile: 64 pool rows x 256 gate cols (+ 256 up cols), 256 threads.
// Per warp: 32 gate + 32 up B-cols (2 n16 per side) x 4 m16 row blocks ->
// 32 mma per 64B k-tile, 128 f32 accumulators per thread.
// L2: 64 pool rows x 512 hidden cols, same warp split as v11.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void
gc_moe_fused_v12_kernel(const uint8_t* __restrict__ x_pool,     // [R, H/2]
                         const uint8_t* __restrict__ sfx_pool,   // [R, H/32]
                         const uint8_t* __restrict__ w1_packed,  // [E*2I, H/2]
                         const uint8_t* __restrict__ sfw1,       // [E, nt, 2I, 4B]
                         const uint8_t* __restrict__ w2_packed,  // [E*H, I/2]
                         const uint8_t* __restrict__ sfw2,       // [E, nt, H, 4B]
                         const int* __restrict__ row_starts,
                         const int* __restrict__ x_starts,
                         const int* __restrict__ row_counts,
                         const int* __restrict__ expert_rows,
                         const int* __restrict__ row_slot,
                         const float* __restrict__ row_w,
                         uint8_t* __restrict__ act_packed,       // [R, I/2]
                         uint8_t* __restrict__ act_sf,           // [R, I/32]
                         float* __restrict__ out_rows,            // [R, H] weighted
                         int e_local_arg, int inter_arg, int hidden_arg, int topk,
                         const int* __restrict__ ngrp_dev,        // group stride and count
                         int* __restrict__ w1_cnt                 // group count
                         , const int* __restrict__ input_ready
                         , int active_tokens, int route_capacity
#ifdef FROLE12_SHARED_EXPERT
                         , const uint8_t* request_x, const uint8_t* request_sf
                         , char* shared_y
                         , const uint8_t* se_w1, const uint8_t* se_sfw1
                         , const uint8_t* se_w2, const uint8_t* se_sfw2
#endif
                         , float clamp
                         , const int* metadata_ready, int metadata_generation
                         , const uint4* __restrict__ grouped_records
#ifdef FROLE12_READY_FC2
                         , int* fc2_ready_groups, int* fc2_group_tail, int* fc2_task_head
#endif
#ifdef FROLE12_W1_GROUP_RECORDS
                         , const uint4* __restrict__ w1_records
#endif
                         , char* combine_sym, const int64_t* combine_peers, int combine_offset
#ifdef FROLE12_EXPERT_PAYLOAD
                         , const volatile int* expert_payload
#endif
                         )
{
    constexpr int e_local = FCONST_E
#ifdef FROLE12_SHARED_EXPERT
        + 1
#endif
        , inter = FCONST_I, hidden = FCONST_H;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tid = threadIdx.x;
    const int ncta = gridDim.x;
#if defined(FROLE12_OVERLAP_FC2) || defined(FROLE12_READY_FC2)
    constexpr bool grid_phase = false;
#else
    const bool grid_phase = active_tokens >= 32;
#endif
    const int nb1 = inter / FBCOLS;        // L1 slabs per (e, group)
    const int hb2 = (hidden + 2 * FBCOLS - 1) / (2 * FBCOLS);     // L2 slabs per (e, group)
    constexpr int m_blocks_per_tile = FTILE_ROWS / 16;
    constexpr int fc1_nb = FBCOLS / (16 * (FTHREADS / 32));
    constexpr int fc2_nb = 2 * fc1_nb;
    constexpr int asf_issuer = 3;
    static_assert(fc1_nb == 2);
    // ONE smem set (NBUF-deep), used sequentially by both passes.
#if defined(FROLE12_PAD_WEIGHT_RING)
    // Keep each gate/up pair contiguous and stagger consecutive ring slots.
    struct WeightSlot {
        uint8_t tiles[2][FBCOLS][64];
        uint8_t padding[128];
    };
    __shared__ __align__(128) WeightSlot weight_ring[NBUF];
#define B_TILE(s, side) weight_ring[s].tiles[side]
#else
    __shared__ __align__(128) uint8_t cube_pair[NBUF][2][FBCOLS][64];
#define B_TILE(s, side) cube_pair[s][side]
#endif
#define CUBE_G(s) B_TILE(s, 0)
#define CUBE_U(s) B_TILE(s, 1)
    __shared__ __align__(128) uint32_t Bsf[NBUF][2 * FBCOLS];
    // A tiles are AIU-staged as swizzled cubes (64 rows x 64B) and gathered
    // by ldmatrix (4x fewer smem ops than the v11 hand-gather).
    __shared__ __align__(128) uint8_t A_cube[NBUF][FTILE_ROWS][64];
#ifdef FROLE12_PACK_W1_TSM_SCALES
    constexpr int w1_pairs = FTILE_ROWS / 2;
    constexpr int w1_scale_tiles = FCONST_H / 128;
#endif
#ifdef FROLE12_ASF_CHUNK
    static_assert(FROLE12_ASF_CHUNK >= NBUF);
    // FC1 and FC2 reuse the same TSM. Pad the inner dimension to spread
    // scale reads across more banks.
    __shared__ __align__(128) union {
        uint32_t l1[2][FTILE_ROWS][FROLE12_ASF_CHUNK + 4];
#ifdef FROLE12_PACK_W1_TSM_SCALES
        uint32_t l1_packed[w1_scale_tiles][2][w1_pairs];
#endif
#ifdef FROLE12_K_MAJOR_ACT
        uint32_t l2[FCONST_I / 128][FTILE_ROWS + 4];
#else
        uint32_t l2[FTILE_ROWS][FCONST_I / 128 + 4];
#endif
    } staged_asf;
#ifdef FROLE12_PACK_W1_TSM_SCALES
    auto* packed_w1_asf = &staged_asf.l1_packed[0][0][0];
#endif
#else
#if defined(FROLE12_PACK_W1_TSM_SCALES) && !defined(FROLE12_PRELOAD_ASF)
    __shared__ __align__(128) union {
        uint32_t raw[NBUF][FTILE_ROWS];
        uint32_t packed[w1_scale_tiles][2][w1_pairs];
    } scales;
    auto& Asf4 = scales.raw;
    auto* packed_w1_asf = &scales.packed[0][0][0];
#else
    __shared__ __align__(16) uint32_t Asf4[NBUF][FTILE_ROWS];   // 4 kgroups/row/tile
#endif
#endif

#ifdef FROLE12_PRELOAD_ASF
#ifdef FROLE12_ASF_TRANSPOSE
    // FC1 reads contiguous rows for each K tile. Padding spreads the
    // transpose stores for different K tiles across the TSM banks.
    constexpr int asf_rows = FCONST_H / 128;
    constexpr int asf_pitch = FTILE_ROWS + 1;
#else
    // Pad the row stride to spread adjacent rows over more TSM banks.
    constexpr int asf_rows = FTILE_ROWS;
    constexpr int asf_pitch = FCONST_H / 128 + 4;
#endif
#ifdef FROLE12_FC2_STAGE_ASF
    constexpr int fc2_scale_tiles = FCONST_I / 128;
    constexpr int fc2_scale_rows = FTILE_ROWS / 2;
    // FC1 is done using its scale tile when the CTA enters FC2.
    __shared__ __align__(128) union {
        uint32_t fc1[asf_rows][asf_pitch];
        uint32_t fc2[fc2_scale_tiles][2][fc2_scale_rows];
    } asf_tile;
    auto& AsfTile = asf_tile.fc1;
#else
    __shared__ __align__(128) uint32_t AsfTile[asf_rows][asf_pitch];
#endif
#if defined(FROLE12_PACK_W1_TSM_SCALES) && !defined(FROLE12_ASF_CHUNK)
    static_assert(asf_rows * asf_pitch >= w1_scale_tiles * 2 * w1_pairs);
    auto* packed_w1_asf = &AsfTile[0][0];
#endif
#endif
    const int w1_kbytes = hidden >> 1;
    const int w1_tiles = w1_kbytes >> 6;
#ifdef FROLE12_W1_OFFSET32
    // One rank's packed gate/up weights are larger than their scale array.
    static_assert(uint64_t(FCONST_E) * FCONST_I * FCONST_H < (1ull << 32));
    using w1_offset = uint32_t;
#else
    using w1_offset = size_t;
#endif
    const w1_offset w1_sf_pitch = w1_offset(2 * inter) * 4u;
    // Issue tile kt's B slab from two warps or one paired descriptor.
    // warp 2 issues the A tile (64 rows x 64B) from the pool.  rowh/dh:
    // the A tensor is the group's row window [rowh, rowh + dh) of x_pool;
    // rows beyond dh pad to zero (padz) -- their outputs are never stored.
    // NOTE: buffer indices are passed EXPLICITLY (rolling wrap counters in
    // the k loop) -- `kt % NBUF` costs a full integer-div sequence on this
    // ISA (~15 SASS ops x 6 sites/iter = ~+160us at T=256 with NBUF=3).
#ifdef FROLE12_SHARED_EXPERT
    auto input_rows = [&](int expert, int row) {
        return (expert == FCONST_E ? request_x : x_pool) + size_t(row) * w1_kbytes;
    };
    auto input_scale_stride = [&](int expert) { return expert == FCONST_E ? hidden / 32 : routed_scale_stride; };
    auto input_scales = [&](int expert, int row) {
        return (expert == FCONST_E ? request_sf : sfx_pool) + size_t(row) * input_scale_stride(expert);
    };
#else
    auto input_rows = [&](int, int row) { return x_pool + size_t(row) * w1_kbytes; };
    auto input_scale_stride = [](int) { return routed_scale_stride; };
    auto input_scales = [&](int, int row) { return sfx_pool + size_t(row) * routed_scale_stride; };
#endif
    const auto w1_base = [&](int expert) {
#ifdef FROLE12_SHARED_EXPERT
        if (expert == FCONST_E && se_w1) return se_w1;
#endif
        return w1_packed + w1_offset(expert) * w1_offset(inter) * w1_offset(hidden);
    };
    const auto w1_scale_base = [&](int expert) {
#ifdef FROLE12_SHARED_EXPERT
        if (expert == FCONST_E && se_sfw1) return se_sfw1;
#endif
        return sfw1 + w1_offset(expert) * w1_tiles * w1_sf_pitch;
    };
    const auto w2_base = [&](int expert) {
#ifdef FROLE12_SHARED_EXPERT
        if (expert == FCONST_E && se_w2) return se_w2;
#endif
        return w2_packed + size_t(expert) * hidden * (inter / 2);
    };
    constexpr size_t w2_panel_bytes = 512 * 64;
    struct W2Panel {
        const uint8_t* data;
        size_t k_stride;
    };
    const auto w2_panel = [&](int expert, size_t column_panel) {
#ifdef FROLE12_K_MAJOR_W2
        if (expert < FCONST_E) {
            constexpr int panels = FCONST_H / 512;
            return W2Panel{w2_packed + (size_t(expert) * panels + column_panel) * w2_panel_bytes,
                           size_t(FCONST_E) * panels * w2_panel_bytes};
        }
#endif
        return W2Panel{w2_base(expert) + column_panel * (inter / 128) * w2_panel_bytes,
                       w2_panel_bytes};
    };
    const auto w2_scale_base = [&](int expert) {
#ifdef FROLE12_SHARED_EXPERT
        if (expert == FCONST_E && se_sfw2) return se_sfw2;
#endif
        return sfw2 + size_t(expert) * hidden * (inter / 32);
    };
#ifdef FROLE12_ACTIVATION_CLAMP
    const float bound = __uint_as_float(sm100_bf16_pair(clamp, 0.f) << 16);
#endif
    auto input_start = [&](int expert, int group) {
#ifdef FROLE12_SHARED_EXPERT
        if (expert == FCONST_E) return group * FTILE_ROWS;
#endif
#ifdef FROLE12_PRECLEARED_DISPATCH
        return routed_input_row(expert, group * FTILE_ROWS, route_capacity);
#else
        return x_starts[expert] + group * FTILE_ROWS;
#endif
    };
#ifdef FROLE12_W1_TASK_BASES
    struct W1Sources {
        const uint8_t* weights;
        const uint8_t* scales;
        const uint8_t* activation;
        const uint8_t* activation_scales;
        int scale_stride;
    };
    auto w1_sources = [&](int expert, int column, int row) {
        constexpr int panel = 256;
        const w1_offset index = column / panel;
        return W1Sources{
            w1_base(expert) + index * w1_tiles * (2 * panel * 64),
            w1_scale_base(expert),
            input_rows(expert, row),
            input_scales(expert, row),
            input_scale_stride(expert),
        };
    };
#else
    using W1Sources = int;
    auto w1_sources = [](int expert, int, int) { return expert; };
#endif
    auto issue_w1 = [&](W1Sources source, int n_base, int kt, int s, int rowh, int dh,
                        bool copy_a = true, bool copy_b = true) {
        constexpr int panel = 256;
        static_assert(FCONST_I % panel == 0 && panel % FBCOLS == 0);
#ifdef FROLE12_W1_TASK_BASES
        const uint8_t* w1_e = source.weights + kt * (2 * panel * 64);
        const uint8_t* activation = source.activation;
        const uint8_t* activation_scales = source.activation_scales;
        const int scale_stride = source.scale_stride;
#else
        const int e = source;
        const uint8_t* activation = input_rows(e, rowh);
        const uint8_t* activation_scales = input_scales(e, rowh);
        const int scale_stride = input_scale_stride(e);
        const w1_offset panel_index = n_base / panel;
        const uint8_t* w1_e = w1_base(e) + (panel_index * w1_tiles + kt) * (2 * panel * 64);
#endif
        const int b_pitch = 64, b_rows = panel, b_col = n_base % panel, b_k = 0;
#ifdef FROLE12_SPLIT_W1_COPY
#ifdef FROLE12_INTERLEAVE_W1_ISSUERS
        const bool issues_weight = (warp & 2) == 0;
        const int side = warp >> 2;
#else
        const bool issues_weight = warp < 4;
        const int side = warp >> 1;
#endif
        if (copy_b && issues_weight) {
            constexpr int height = FBCOLS / 2;
            const int row = (warp & 1) * height;
            aiu_copy_weight_b8(&B_TILE(s, side)[row][0], w1_e, b_pitch, 2 * b_rows,
                             64, height, b_pitch, b_k, side * b_rows + b_col + row, 1);
        }
        if (copy_a && warp == 2) {
            aiu_copy_b8_swzl(&A_cube[s][0][0],
                             activation, w1_kbytes, dh,
                             64, activation_copy_rows(dh), w1_kbytes, kt << 6, 0, 1);
        }
#else
#ifdef FL1_GATE_PAIR
        if (copy_b && warp == 0) {
            aiu_copy_gate_up(&CUBE_G(s)[0][0], w1_e, b_pitch, b_rows,
                             FBCOLS, b_k, b_col);
        }
#else
        if (copy_b && warp == 0) {
            aiu_copy_weight_b8(&CUBE_G(s)[0][0], w1_e, b_pitch, 2 * b_rows,
                             64, FBCOLS, b_pitch, b_k, b_col, 1);
        } else if (copy_b && warp == 1) {
            aiu_copy_weight_b8(&CUBE_U(s)[0][0], w1_e, b_pitch, 2 * b_rows,
                             64, FBCOLS, b_pitch, b_k, b_rows + b_col, 1);
        }
#endif
        else if (copy_a && warp == 2) {
            aiu_copy_b8_swzl(&A_cube[s][0][0],
                             activation, w1_kbytes, dh,
                             64, activation_copy_rows(dh), w1_kbytes, kt << 6, 0, 1);
        }
#endif
#if defined(FROLE12_ASF_CHUNK) && !defined(FROLE12_PACK_W1_TSM_SCALES)
        if (copy_a && warp == asf_issuer && kt % FROLE12_ASF_CHUNK == 0) {
            const int chunk_slot = (kt / FROLE12_ASF_CHUNK) & 1;
            aiu_copy_linear_b32(&staged_asf.l1[chunk_slot][0][0],
                                activation_scales, w1_tiles, dh,
                                FROLE12_ASF_CHUNK + 4, FTILE_ROWS, scale_stride, kt, 0);
        }
#endif
        if (copy_b && warp == 3) {
#ifdef FROLE12_W1_SCALE_PAIR
            const w1_offset tile_bytes = w1_sf_pitch;
#ifdef FROLE12_W1_TASK_BASES
            const uint8_t* sfbase = source.scales + kt * tile_bytes;
#else
            const uint8_t* sfbase = w1_scale_base(e) + kt * tile_bytes;
#endif
            aiu_copy_w1_scale_pair(Bsf[s], sfbase, inter, n_base);
#else
#ifdef FROLE12_W1_TASK_BASES
            const uint8_t* sfbase = source.scales;
#else
            const uint8_t* sfbase = w1_scale_base(e);
#endif
            aiu_copy_linear_b32(Bsf[s], sfbase, 2 * inter, w1_tiles,
                                FBCOLS, 1, w1_sf_pitch, n_base, kt);
            aiu_copy_linear_b32(Bsf[s] + FBCOLS, sfbase, 2 * inter, w1_tiles,
                                FBCOLS, 1, w1_sf_pitch, inter + n_base, kt);
#endif
        }
    };
    // Issue immutable weights before dispatch publishes the routed rows.
#if defined(FROLE12_EARLY_METADATA) && defined(FL1_FULL_RING) && defined(FROLE12_PREFILL_W1_WEIGHTS)
    constexpr int speculative_tiles = NBUF;
#else
    constexpr int speculative_tiles = NBUF - 1;
#endif
    const int first_block = blockIdx.x;
    int first_fc1 = first_block;
#ifdef FROLE12_SHARED_EXPERT
    const int shared_groups = (active_tokens + FTILE_ROWS - 1) / FTILE_ROWS;
    const int shared_tasks = shared_groups * nb1;
    const int early_tasks = early_shared_fc1_tasks(shared_groups, nb1, ncta);
    // Compute shared FC1 from the caller-prepared input.
    // Assign those tiles to the last CTAs, which consume the tail
    // of the dispatch queue, and leave CTA0 free to finalize routed metadata.
    if (early_tasks) {
        first_fc1 += early_tasks;
        if (first_fc1 >= ncta) first_fc1 -= ncta;
    }
    const bool early_shared = first_fc1 < early_tasks;
    const bool shared_first = first_fc1 < shared_tasks;
#endif
    auto prefetch_w1 = [&](int expert) {
        const int column = (first_fc1 % nb1) * FBCOLS;
#pragma unroll
        for (int p = 0; p < speculative_tiles; ++p) {
            issue_w1(w1_sources(expert, column, 0), column, p, p, 0, 0, false);
            cp_async_fence();
        }
    };
#ifndef FROLE12_EARLY_METADATA
#ifdef FROLE12_SHARED_EXPERT
    const int guessed_expert = shared_first ? FCONST_E : (first_fc1 - shared_tasks) / nb1;
    const bool speculate = !early_shared &&
        (shared_first || (active_tokens >= 16 && guessed_expert < FCONST_E));
#else
    const int guessed_expert = first_fc1 / nb1;
    const bool speculate = active_tokens >= 16 && guessed_expert < e_local;
#endif
    if (speculate) prefetch_w1(guessed_expert);
#elif defined(FROLE12_SHARED_EXPERT)
    if (shared_first && !early_shared) prefetch_w1(FCONST_E);
#endif

    int ngrp = 0, workgroup_count = 0;
    bool reuse_first = false;
#ifdef FROLE12_EXPERT_PAYLOAD
    bool input_acquired = false;
    auto wait_input = [&] {
        if (input_acquired) return;
        if (tid == 0)
            while (!cnt_acquire(input_ready)) __nanosleep(FROLE12_POLL_NS);
        __syncthreads();
        __threadfence();
        input_acquired = true;
    };
    __shared__ int all_payload_ready;
    auto wait_expert = [&](int expert) {
        if (input_acquired || expert == FCONST_E) return;
        if (tid == 0) {
            all_payload_ready = cnt_acquire(input_ready) != 0;
            if (!all_payload_ready) {
                const int expected = row_counts[expert];
                int ready;
                do {
                    asm volatile("ppu.ld.acquire.sys.global.s32 %0, [%1];"
                                 : "=r"(ready) : "l"((const void*)(expert_payload + expert)) : "memory");
                    if (ready < expected) __nanosleep(FROLE12_POLL_NS);
                } while (ready < expected);
            }
        }
        __syncthreads();
        __threadfence();
        input_acquired = all_payload_ready != 0;
    };
#endif
    auto acquire_metadata = [&] {
        if (tid == 0)
            while (cnt_acquire(metadata_ready) < metadata_generation) __nanosleep(FROLE12_POLL_NS);
        __syncthreads();
        __threadfence();
#if defined(FROLE12_READY_FC2) || defined(FROLE12_EARLY_METADATA)
        ngrp = *ngrp_dev;
#else
        __shared__ int s_ngrp;
        if (FTILE_ROWS != 64) {
            if (warp == 0) {
                int mx = 0;
                for (int e = lane; e < e_local; e += 32) mx = max(mx, row_counts[e]);
#pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, off));
                if (lane == 0) s_ngrp = (mx + FTILE_ROWS - 1) / FTILE_ROWS;
            }
            __syncthreads();
            ngrp = s_ngrp;
        } else {
            ngrp = max(1, *ngrp_dev);
        }
#endif
        workgroup_count = ngrp_dev[1];
#ifdef FROLE12_EARLY_METADATA
        reuse_first = first_fc1 < workgroup_count * nb1;
        if (reuse_first
#ifdef FROLE12_SHARED_EXPERT
            && !shared_first
#endif
        ) {
#ifdef FROLE12_W1_GROUP_RECORDS
            const int expert = w1_records[first_fc1 / nb1].x & group_expert_mask;
#elif defined(FROLE12_READY_FC2)
            const int expert = expert_rows[first_fc1 / nb1] & group_expert_mask;
#else
            const int expert = expert_rows[first_fc1 / nb1] / ngrp;
#endif
            prefetch_w1(expert);
        }
#ifndef FROLE12_EXPERT_PAYLOAD
        if (tid == 0)
            while (cnt_acquire(input_ready) == 0) __nanosleep(FROLE12_POLL_NS);
        __syncthreads();
        __threadfence();
#endif
#else
        reuse_first = speculate && first_fc1 < workgroup_count * nb1 &&
#ifdef FROLE12_READY_FC2
                      (expert_rows[first_fc1 / nb1] & group_expert_mask) == guessed_expert;
#else
                      expert_rows[first_fc1 / nb1] / ngrp == guessed_expert;
#endif
        if (speculate && !reuse_first) {
            cp_async_wait_all();
            __syncthreads();
        }
#endif
    };
#ifdef FROLE12_SHARED_EXPERT
    if (!early_shared) acquire_metadata();
#else
    acquire_metadata();
#endif

    auto issue_w2_weights = [&](int s, const uint8_t* weights, int col) {
        constexpr int panel = 512;
#if defined(FROLE12_W2_SPLIT_2)
        constexpr int issuers = 2;
        const bool issues_weight = warp < issuers;
        const int issuer = warp;
#elif FROLE12_W2_COPY_WARPS == 4
        constexpr int issuers = 4;
        const bool issues_weight = (warp & 2) == 0;
        const int issuer = (warp & 1) | ((warp >> 1) & 2);
#elif defined(FROLE12_W2_COPY_WARPS)
        constexpr int issuers = FROLE12_W2_COPY_WARPS;
        const bool issues_weight = warp < issuers;
        const int issuer = warp;
#else
        constexpr int issuers = 1;
        const bool issues_weight = warp == 0;
        constexpr int issuer = 0;
#endif
        if (issues_weight) {
            constexpr int height = 2 * FBCOLS / issuers;
            const int row = issuer * height;
            aiu_copy_weight_b8(&B_TILE(s, row / FBCOLS)[row % FBCOLS][0],
                               weights, 64, panel, 64, height, 64, 0, col + row, 1);
        }
    };

    auto issue_w2_stage = [&](int slot, W2Panel weights, const uint8_t* scales, int column, int tile) {
        constexpr int tiles = FCONST_I / 128;
        issue_w2_weights(slot, weights.data + tile * weights.k_stride, column % 512);
        if (warp == 3)
            aiu_copy_linear_b32(Bsf[slot], scales, FCONST_H, tiles,
                                2 * FBCOLS, 1, FCONST_H * 4, column, tile);
    };

#ifdef FROLE12_PREFETCH_FC2_FROM_FC1
    bool fc2_preissued = false;
#ifdef FROLE12_READY_FC2
    __shared__ int fc2_group, fc2_task;
    bool fc2_claimed = false;
#elif defined(FROLE12_OVERLAP_FC2)
#error FC2 prefetch requires the ready queue or the block-index task assignment
#endif
    auto preissue_fc2 = [&] {
#ifdef FROLE12_READY_FC2
        if (tid == 0) {
            fc2_task = atomicAdd(fc2_task_head, 1);
            fc2_group = fc2_task < workgroup_count * hb2
                ? cnt_acquire(fc2_ready_groups + fc2_task / hb2) : 0;
        }
        __syncthreads();
        fc2_claimed = true;
        // FC1 may still owe the group's last slab. Check once and let
        // the normal FC2 gate wait after the epilogue.
        if (!fc2_group) return;
        const int task = fc2_task;
        const int expert = (fc2_group - 1) & group_expert_mask;
#else
        const int task = blockIdx.x;
        const int expert = expert_rows[task / hb2] / ngrp;
#endif
        const int column = (task % hb2) * (2 * FBCOLS);
        const auto w2 = w2_panel(expert, column / 512);
        const auto* scales = w2_scale_base(expert);
        constexpr int tiles = FCONST_I / 128;
#pragma unroll
        for (int tile = 0; tile < NBUF - 1; ++tile) {
            if (tile < tiles) issue_w2_stage(tile + 1, w2, scales, column, tile);
            cp_async_fence();
        }
        fc2_preissued = true;
    };
#endif

    // ===================== pass 1: L1 tiles (e, g, nb) =====================
    {
        const int kb_half = hidden >> 1;
        const int n_scale = hidden >> 5;
        const int ntiles = kb_half >> 6;          // 64B K tiles
        const int m0 = lane >> 2;
#ifdef FROLE12_CACHE_ROUTE_WEIGHT
        __shared__ float route_weights[FTILE_ROWS];
#endif
#ifdef FL1_FULL_RING
        constexpr int prefill = NBUF;
#else
        constexpr int prefill = NBUF - 1;
#endif

#ifdef FROLE12_PREFETCH_W1_TAIL
    int ring_start = 0;
#endif
    for (int t = first_fc1; ; t += ncta) {
#ifdef FROLE12_SHARED_EXPERT
    const bool early = early_shared && t == first_fc1;
    if (!early && t >= workgroup_count * nb1) break;
    const bool first_task = t == first_fc1 || (early_shared && t == first_fc1 + ncta);
    const bool prefetch_next = active_tokens > 0 && !early;
#else
    if (t >= workgroup_count * nb1) break;
    const bool first_task = t == first_fc1;
    const bool prefetch_next = active_tokens > 0;
#endif
    const int n_base = (t % nb1) * FBCOLS;
#ifdef FROLE12_W1_GROUP_RECORDS
    const uint4 record = w1_records[t / nb1];
    const int group_code = record.x;
#else
#if defined(FROLE12_SHARED_EXPERT) && defined(FROLE12_READY_FC2)
    const int group_code = early ? (t / nb1 << group_expert_bits) | FCONST_E : expert_rows[t / nb1];
#elif defined(FROLE12_SHARED_EXPERT)
    const int group_code = early ? 0 : expert_rows[t / nb1];
#else
    const int group_code = expert_rows[t / nb1];
#endif
#endif
#ifdef FROLE12_READY_FC2
    const int e = group_code & group_expert_mask;
    const int g = group_code >> group_expert_bits;
#else
#ifdef FROLE12_SHARED_EXPERT
    const int e = early ? FCONST_E : group_code / ngrp;
    const int g = early ? t / nb1 : group_code - e * ngrp;
#else
    const int e = group_code / ngrp;
    const int g = group_code - e * ngrp;
#endif
#endif
#ifdef FROLE12_W1_GROUP_RECORDS
    const int row0 = record.y;
    const int row0x = record.z;
    int rows = record.w;
#else
    const int row0 = row_starts[e] + g * FTILE_ROWS;
    int rows = row_counts[e] - g * FTILE_ROWS;
    const int row0x = input_start(e, g);
#endif
    if (rows <= 0) continue;
    if (rows > FTILE_ROWS) rows = FTILE_ROWS;
#ifdef FROLE12_EXPERT_PAYLOAD
    wait_expert(e);
#endif
    auto route_weight = [&](int row) {
#ifdef FROLE12_SHARED_EXPERT
        if (e == FCONST_E) return 1.f;
#endif
        return __uint_as_float(grouped_records[row0x + row].z);
    };
    // Bsf values are register-prefetched at the top of the producing tile
    // and stored after the mma chain (v11: -121us L2 / -24us L1).
    auto bsf_load = [&](int ks, uint32_t& b0, uint32_t& b1) {
    };
    auto bsf_store = [&](int sb, uint32_t b0, uint32_t b1) {
    };

    // A data now streams via AIU into A_cube; only the tiny per-row scale
    // (1 u32 per row per tile) still stages through registers.
#if defined(FROLE12_PRELOAD_ASF) || defined(FROLE12_ASF_CHUNK) || defined(FROLE12_PACK_W1_TSM_SCALES)
    auto asf_load = [&](int, auto, uint32_t&) {};
    auto asf_store = [&](int, uint32_t) {};
#else
    auto asf_load = [&](int ks, auto prow, uint32_t& asfv) {
        if (tid < FTILE_ROWS)
            asfv = *reinterpret_cast<const uint32_t*>(
                input_scales(e, prow(tid)) + ((size_t)ks << 2));
    };
    auto asf_store = [&](int sb, uint32_t asfv) {
        if (tid < FTILE_ROWS) Asf4[sb][tid] = asfv;
    };
#endif

    auto compute_rows = [&](auto m_blocks) {
        constexpr int MB = decltype(m_blocks)::value;
        constexpr int rb0 = 0;
        const int nm = (rows - rb0 + 15) >> 4;      // 1..4 m16 blocks
        const int rows_here = rows - rb0;
        auto prow_g = [&](int r) {
            return row0x + rb0 + ((r < rows_here) ? r : rows_here - 1);
        };

#ifdef FROLE12_A_TSM_OFFSET
        const uint32_t a_units = smem_addr32(&A_cube[0][0][0]) >> 4;
#endif
#ifdef FROLE12_B_TSM_OFFSET
        const uint32_t b_units = smem_addr32(&CUBE_G(0)[0][0]) >> 4;
#endif
        float dreg[2][MB][fc1_nb][8];      // [side][m16][n16][j]
#pragma unroll
        for (int sd = 0; sd < 2; ++sd)
#pragma unroll
            for (int mb = 0; mb < MB; ++mb)
#pragma unroll
                for (int nb = 0; nb < fc1_nb; ++nb)
#pragma unroll
                    for (int j = 0; j < 8; ++j) dreg[sd][mb][nb][j] = 0.f;

        // prologue: fill the pipeline with tiles 0..NBUF-2 (one AIU group
        // per tile -- group granularity == tile granularity), and stage
        // tile 0's A/Bsf into smem (exposed once per task).
        uint32_t pf_b0 = 0, pf_b1 = 0;   // regs for the NEXT tile
        uint32_t pf_asf = 0;
        const int arowh = row0x + rb0;      // A window base (pool rows)
        const int adh = rows - rb0;
        const auto source = w1_sources(e, n_base, arowh);         // A window height (padz beyond)
#if defined(FROLE12_PRELOAD_ASF) && !defined(FROLE12_ASF_TRANSPOSE) && !defined(FROLE12_PACK_W1_TSM_SCALES)
        if (warp == 3) {
            aiu_copy_linear_b32(AsfTile, input_scales(e, arowh),
                                ntiles, adh, asf_pitch, FTILE_ROWS, input_scale_stride(e), 0, 0);
            cp_async_fence();
            cp_async_wait_all();
        }
#endif
        if (!prefetch_next || first_task) {
#pragma unroll
            for (int p = 0; p < prefill; ++p) {
                if (p < ntiles) {
                    const bool copy_b = t != first_fc1 || !reuse_first || p >= speculative_tiles;
                    issue_w1(source, n_base, p, p, arowh, adh, true, copy_b);
                }
                cp_async_fence();
            }
        }
#ifdef FROLE12_PACK_W1_TSM_SCALES
        // Pack scale operands once for both K halves. Subsequent MMAs
        // read one word instead of loading two rows and permuting bytes.
        const auto* raw = reinterpret_cast<const uint32_t*>(input_scales(e, arowh));
        const int pitch = input_scale_stride(e) / 4;
        for (int i = tid; i < w1_scale_tiles * w1_pairs; i += FTHREADS) {
            const int pair = i / w1_scale_tiles, k = i % w1_scale_tiles;
            const int r = 16 * (pair / 8) + pair % 8;
            const uint32_t lo = r < adh ? raw[r * pitch + k] : 0;
            const uint32_t hi = r + 8 < adh ? raw[(r + 8) * pitch + k] : 0;
            packed_w1_asf[2 * k * w1_pairs + pair] = __byte_perm(lo, hi, 0x5410);
            packed_w1_asf[(2 * k + 1) * w1_pairs + pair] = __byte_perm(lo, hi, 0x7632);
        }
#elif defined(FROLE12_ASF_TRANSPOSE)
        // Overlap the coalesced scale reads with the operand prefill.
        const auto* scale_base = reinterpret_cast<const uint32_t*>(input_scales(e, arowh));
        constexpr int width = FCONST_H / 128;
        static_assert(width % 4 == 0);
        for (int index = tid; index < FTILE_ROWS * (width / 4); index += FTHREADS) {
            const int row = index / (width / 4);
            const int k = 4 * (index % (width / 4));
            const uint4 words = row < adh
                ? *reinterpret_cast<const uint4*>(scale_base + row * (input_scale_stride(e) / 4) + k)
                : uint4{0, 0, 0, 0};
            AsfTile[k][row] = words.x;
            AsfTile[k + 1][row] = words.y;
            AsfTile[k + 2][row] = words.z;
            AsfTile[k + 3][row] = words.w;
        }
#endif
        if (ntiles > 0) {
            asf_load(0, prow_g, pf_asf);
            asf_store(0, pf_asf);
            bsf_load(0, pf_b0, pf_b1);
            bsf_store(0, pf_b0, pf_b1);
        }
#ifdef FROLE12_CACHE_ROUTE_WEIGHT
        // The first K tile's CTA barrier makes these records visible.
        if (tid < rows) route_weights[tid] = route_weight(tid);
#endif
        // The full ring acquires tile zero at the first iteration's top
        // wait and CTA join, so the remaining prefills can stay in flight.
#if !defined(FL1_FULL_RING) && (defined(FVBOTTOM) || defined(FISSUE_POST))
        // v11 op order (barrier+wait at the BOTTOM): the pf registers then
        // have v11's live ranges (no barrier inside), which keeps hgcc's
        // spill budget at v11 levels -- the top-barrier form cost +40
        // stack words and ~+1us/iteration.
        cp_async_wait_all();
        __syncthreads();
#endif

#ifdef FROLE12_W1_REFILL_ADDRESS
        // Form per-warp descriptor bases outside the K loop.
        static_assert(NBUF == 3);
#if defined(FROLE12_PAD_WEIGHT_RING)
#error Prepared FC1 refill requires contiguous weight slots
#endif
        static_assert(FTHREADS == 256 && FBCOLS == 256);
        constexpr int copy_rows = FBCOLS / 2;
        const bool refill_b = (warp & 2) == 0;
        const int refill_side = warp >> 2;
        const int refill_row = (warp & 1) * copy_rows;
        const int refill_column = refill_side * FBCOLS + refill_row;
        const int copy_a_rows = activation_copy_rows(adh);
        const uint32_t refill_dst = refill_b
            ? smem_addr32(&B_TILE(0, refill_side)[refill_row][0])
            : warp == 2 ? smem_addr32(&A_cube[0][0][0]) : smem_addr32(Bsf[0]);
        const int refill_pitch = refill_b
            ? 2 * FBCOLS * 64 : warp == 2 ? FTILE_ROWS * 64 : 2 * FBCOLS * 4;
        const auto* refill_base = refill_b
            ? source.weights : warp == 2 ? source.activation : source.scales;
#endif
#ifdef FROLE12_PREFETCH_W1_TAIL
        static_assert(NBUF == 3);
        constexpr int advance = (FCONST_H / 128) % NBUF;
        const int unwrapped_start = ring_start + advance;
        const int next_start = advance == 0 ? 0 :
            (unwrapped_start < NBUF ? unwrapped_start : unwrapped_start - NBUF);
        const int next_task = t + ncta;
        const bool next_valid = prefetch_next && next_task < workgroup_count * nb1;
        const int next_column = (next_task % nb1) * FBCOLS;
        int sroll = ring_start;
#else
        int sroll = 0;
#endif
        for (int ks = 0; ks < ntiles; ++ks) {
            // rolling buffer indices for this iteration
            const int s1r = sroll + 1;
            const int sb1 = (s1r == NBUF) ? 0 : s1r;          // tile ks+1
            const int sbm = sroll ? sroll - 1 : NBUF - 1;     // tile ks+NBUF-1
            // gmem prefetch for tile ks+1 (register-deferred stores below)
            if (ks + 1 < ntiles) {
                asf_load(ks + 1, prow_g, pf_asf);
                bsf_load(ks + 1, pf_b0, pf_b1);
            }
#ifdef FL1_FULL_RING
#ifdef FROLE12_PREFETCH_W1_TAIL
            // The drain refills each consumed slot with the next panel, so
            // two groups can still be outstanding after this wait.
            if (next_valid && ks >= ntiles - NBUF)
                asm volatile("ppu.cp.async.wait_group 2;");
            else
#endif
                cp_async_wait_full_ring(ks, ntiles);
            __syncthreads();
#elif defined(FVBOTTOM) && !defined(FISSUE_POST)
            // refill 2 tiles ahead: buffer (ks+NBUF-1)%NBUF was last read
            // by iteration ks-1's gather, before the bottom barrier.
            if (ks + NBUF - 1 < ntiles) issue_w1(source, n_base, ks + NBUF - 1, sbm, arowh, adh);
            cp_async_fence();
#else
            // wait for tile ks's AIU group (NBUF-2 newest stay in flight),
            // then join all warps: the barrier doubles as the visibility
            // fence for the previous iteration's deferred A/Bsf stores.
            cp_async_wait_pipe(ks, ntiles);
            __syncthreads();
            // refill the buffer freed by tile ks-1 with tile ks+NBUF-1
            // (its buffer index is (ks-1)%NBUF -- read for the last time
            // by the previous iteration's gather, before the barrier).
            if (ks + NBUF - 1 < ntiles) issue_w1(source, n_base, ks + NBUF - 1, sbm, arowh, adh);
            cp_async_fence();
#endif

#if defined(FROLE12_PREFETCH_NEXT_W1)
            // This two-slot ring reads its final K tile from slot one.
            // Fill slot zero's weights while the last MMA is still in flight.
            static_assert(NBUF == 2 && (FCONST_H / 128) % 2 == 0);
            const int next = t + ncta;
            if (ks + 1 == ntiles && prefetch_next && next < workgroup_count * nb1) {
                const int expert = expert_rows[next / nb1] & group_expert_mask;
                const int column = (next % nb1) * FBCOLS;
                issue_w1(w1_sources(expert, column, 0), column, 0, 0, 0, 0, false);
                cp_async_fence();
            }
#endif

            // Consume one 64-element K half at a time, shortening fragment
            // live ranges while retaining the same accumulator order.
            const int s = sroll;
#ifdef FROLE12_W1_REFILL_ADDRESS
            const int refill_k = ks + NBUF;
            uint32_t refill_address = refill_dst + s * refill_pitch;
            const auto* refill_src = refill_base;
            if (refill_b) refill_src += refill_k * (2 * FBCOLS * 64);
#if defined(FROLE12_W1_SCALE_PAIR)
            else if (warp == 3) refill_src += refill_k * w1_sf_pitch;
#endif
            // Prepare the refill addresses before the last operand gather.
            if (refill_k < ntiles && (refill_b || warp == 2 || warp == 3))
                asm volatile("" : "+r"(refill_address), "+l"(refill_src));
#endif
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                uint32_t ac[MB][4], mac[MB];
#pragma unroll
                for (int group = 0; group < (MB + 3) / 4; ++group) {
                    const int mb_part = min(4 * group + (lane & 3), MB - 1);
                    const int arow = 16 * mb_part + (lane >> 2);
#ifdef FROLE12_PACK_W1_TSM_SCALES
                    mac[group] = packed_w1_asf[(2 * ks + h) * w1_pairs + 8 * mb_part + (lane >> 2)];
#elif defined(FROLE12_ASF_CHUNK)
                    const int chunk_slot = (ks / FROLE12_ASF_CHUNK) & 1;
                    const int chunk_col = ks % FROLE12_ASF_CHUNK;
                    mac[group] = pack_fc1_activation_scale(staged_asf.l1[chunk_slot][arow][chunk_col], staged_asf.l1[chunk_slot][arow + 8][chunk_col], h);
#elif defined(FROLE12_ASF_TRANSPOSE)
                    mac[group] = pack_fc1_activation_scale(AsfTile[ks][arow], AsfTile[ks][arow + 8], h);
#elif defined(FROLE12_PRELOAD_ASF)
                    mac[group] = pack_fc1_activation_scale(AsfTile[arow][ks], AsfTile[arow + 8][ks], h);
#else
                    mac[group] = pack_fc1_activation_scale(Asf4[s][arow], Asf4[s][arow + 8], h);
#endif
                }
#ifdef FROLE12_W1_EARLY_BSCALE
                const int npart = lane & 3;
                const int packed_n = (npart >> 1) * FBCOLS
                    + 16 * (fc1_nb * warp + (npart & (fc1_nb - 1))) + (lane >> 2);
                const uint32_t packed_mb = Bsf[s][packed_n + 8 * h];
#endif
                auto load_a_fragments = [&] {
#pragma unroll
                    for (int mb = 0; mb < MB; ++mb) {
#ifdef FROLE12_A_TSM_OFFSET
                        const uint32_t address = a_units + 4 * (s * FTILE_ROWS + 16 * mb) + 2 * h;
                        ldmatrix_a_at(ac[mb], address);
#else
                        ldmatrix_a_frag(ac[mb], &A_cube[s][16 * mb][32 * h]);
#endif
                    }
                };
#ifndef FROLE12_W1_B_FIRST
                load_a_fragments();
#endif
#ifdef FL1_REFILL_ALL_FRAGMENTS
                constexpr int fragment_slots = 2 * fc1_nb;
#else
                constexpr int fragment_slots = 2;
#endif
                uint32_t bfr[fragment_slots][4];
#ifndef FROLE12_W1_EARLY_BSCALE
                const int npart = lane & 3;
                const int packed_n = (npart >> 1) * FBCOLS
                    + 16 * (fc1_nb * warp + (npart & (fc1_nb - 1))) + (lane >> 2);
                const uint32_t packed_mb = Bsf[s][packed_n + 8 * h];
#endif
                auto load_b = [&](int part) {
#ifdef FROLE12_B_TSM_OFFSET
                    const int side = part >= fc1_nb;
                    const int row = 16 * (fc1_nb * warp + part % fc1_nb);
                    const uint32_t address = b_units + 4 * ((2 * s + side) * FBCOLS + row) + 2 * h;
                    ldmatrix_b_at(bfr[part & (fragment_slots - 1)], address);
#else
                    const uint8_t (*cbe)[64] = part < fc1_nb ? CUBE_G(s) : CUBE_U(s);
                    ldmatrix_b_frag(bfr[part & (fragment_slots - 1)],
                        &cbe[16 * (fc1_nb * warp + part % fc1_nb)][32 * h], 32, 1);
#endif
                };
#if defined(FL1_REFILL_LAST_TWO) || defined(FL1_REFILL_ALL_FRAGMENTS)
                // Every warp has loaded the last A and B fragments before
                // the join. The final MMA chain no longer reads this slot.
                auto refill = [&] {
                    if (ks + NBUF < ntiles) {
                        __syncthreads();
#ifdef FROLE12_W1_REFILL_ADDRESS
                        if (refill_b) {
                            aiu_copy_weight_b8(refill_address, refill_src, 64, 2 * FBCOLS,
                                                64, copy_rows, 64, 0, refill_column, 1);
                        } else if (warp == 2) {
                            aiu_copy_b8_swzl(refill_address, refill_src, w1_kbytes, adh,
                                               64, copy_a_rows, w1_kbytes, refill_k << 6, 0, 1);
                        } else if (warp == 3) {
                            if (refill_k % FROLE12_ASF_CHUNK == 0) {
                                const int chunk = (refill_k / FROLE12_ASF_CHUNK) & 1;
                                aiu_copy_linear_b32(&staged_asf.l1[chunk][0][0],
                                    source.activation_scales, w1_tiles, adh,
                                    FROLE12_ASF_CHUNK + 4, FTILE_ROWS, source.scale_stride, refill_k, 0);
                            }
#ifdef FROLE12_W1_SCALE_PAIR
                            aiu_copy_w1_scale_pair(refill_address, refill_src, inter, n_base);
#else
                            aiu_copy_linear_b32(refill_address, refill_src, 2 * inter, w1_tiles,
                                                 FBCOLS, 1, w1_sf_pitch, n_base, refill_k);
                            aiu_copy_linear_b32(refill_address + FBCOLS * 4, refill_src, 2 * inter,
                                w1_tiles, FBCOLS, 1, w1_sf_pitch, inter + n_base, refill_k);
#endif
                        }
#else
                        issue_w1(source, n_base, ks + NBUF, sroll, arowh, adh);
#endif
                        cp_async_fence();
                    }
#ifdef FROLE12_PREFETCH_W1_TAIL
                    else if (next_valid) {
                        // This slot has retired from the current task. Fill its
                        // immutable next-task operands before the last MMAs.
                        __syncthreads();
                        const int next_expert = w1_records[next_task / nb1].x & group_expert_mask;
                        const auto early_source = w1_sources(next_expert, next_column, 0);
                        issue_w1(early_source, next_column, ks + NBUF - ntiles, s, 0, 0, false);
                        cp_async_fence();
                    }
#endif
                };
#endif
                load_b(0);
#ifdef FROLE12_W1_B_FIRST
                load_a_fragments();
#endif
#ifdef FL1_REFILL_ALL_FRAGMENTS
                if (h == 1) {
#pragma unroll
                    for (int part = 1; part < 2 * fc1_nb; ++part) load_b(part);
                    refill();
                }
#endif
#pragma unroll
                for (int part = 0; part < 2 * fc1_nb; ++part) {
#ifdef FL1_REFILL_ALL_FRAGMENTS
                    if (h == 0 && part + 1 < 2 * fc1_nb) load_b(part + 1);
#else
                    if (part + 1 < 2 * fc1_nb) load_b(part + 1);
#endif
#ifdef FL1_REFILL_LAST_TWO
                    if (h == 1 && part == 2 * fc1_nb - 2) refill();
#endif
#pragma unroll
                    for (int mb = 0; mb < MB; ++mb)
                        mma_m16n16k64_f4(dreg[part / fc1_nb][mb][part % fc1_nb], ac[mb],
                                        bfr[part & (fragment_slots - 1)], mac[mb >> 2], packed_mb, mb & 3,
                                        2 * (part / fc1_nb) + part % fc1_nb);
                }
            }
#if   defined(FVBOTTOM) && !defined(FL1_FULL_RING)
            // bottom wait (v11 position): leaves NBUF-2 newest tiles in
            // flight across the barrier -- tile ks+1 is drained here (the
            // next gather reads it right after the barrier), tile ks+2
            // stays in flight across it.
            cp_async_wait_pipe(ks + 1, ntiles);
#endif
            // deferred smem stores for tile ks+1: the prefetch loads have
            // had the gather + mma chain to land; their visibility fence
            // is the NEXT iteration's top barrier (FVBOTTOM: the barrier
            // right below).
            if (ks + 1 < ntiles) {
                bsf_store(sb1, pf_b0, pf_b1);
                asf_store(sb1, pf_asf);
            }
#ifdef FL1_FULL_RING
#if !defined(FL1_REFILL_LAST_TWO) && !defined(FL1_REFILL_ALL_FRAGMENTS)
            // Refill the consumed slot only after every warp finishes its reads.
            if (ks + NBUF < ntiles) {
                __syncthreads();
                issue_w1(source, n_base, ks + NBUF, sroll, arowh, adh);
                cp_async_fence();
            }
#endif
#elif defined(FVBOTTOM) || defined(FISSUE_POST)
            __syncthreads();
#endif
            sroll = sb1;   // advance the rolling buffer index
        }
        // The next task's prologue can reuse slots still read by the tail.
        __syncthreads();

        // The completed task has released its ring slots. Dense groups
        // have one nonempty row tile each, and all payload is already acquired.
        const int next = t + ncta;
        if (prefetch_next && next < workgroup_count * nb1) {
#ifdef FROLE12_W1_GROUP_RECORDS
            const uint4 record = w1_records[next / nb1];
            const int expert = record.x & group_expert_mask;
            const int row = record.z;
            const int height = record.w;
#else
            const int code = expert_rows[next / nb1];
#ifdef FROLE12_READY_FC2
            const int expert = code & group_expert_mask;
            const int group = code >> group_expert_bits;
#else
            const int expert = code / ngrp;
            const int group = code - expert * ngrp;
#endif
            const int row = input_start(expert, group);
            const int height = min(row_counts[expert] - group * FTILE_ROWS, FTILE_ROWS);
#endif
            const int column = (next % nb1) * FBCOLS;
#ifdef FROLE12_EXPERT_PAYLOAD
            wait_expert(expert);
#endif
            const auto next_source = w1_sources(expert, column, row);
#pragma unroll
            for (int p = 0; p < prefill; ++p) {
#if defined(FROLE12_PREFETCH_NEXT_W1)
                if (p < ntiles)
                    issue_w1(next_source, column, p, p, row, height, true, p != 0);
#else
                if (p < ntiles) {
#ifdef FROLE12_PREFETCH_W1_TAIL
                    const int slot = next_start + p < NBUF ? next_start + p : next_start + p - NBUF;
                    issue_w1(next_source, column, p, slot, row, height, true, false);
#else
                    issue_w1(next_source, column, p, p, row, height);
#endif
                }
#endif
                cp_async_fence();
            }
#ifdef FROLE12_PREFETCH_W1_TAIL
            ring_start = next_start;
#endif
        }

#ifdef FROLE12_PREFETCH_FC2_FROM_FC1
        // Reuse the drained weight ring on the last FC1 task while the
        // registers are still in the SwiGLU and requantization epilogue.
        if (next >= workgroup_count * nb1
#ifndef FROLE12_READY_FC2
            && blockIdx.x < workgroup_count * hb2
#endif
#ifdef FROLE12_SHARED_EXPERT
            && !early
#endif
        )
            preissue_fc2();
#endif
        // ---- SwiGLU epilogue + requant (register accumulators)
#pragma unroll
        for (int mb = 0; mb < MB; ++mb) {
            if (mb < nm) {
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    // Skip an empty row half without splitting a scale group
                    // or changing the active lanes of its warp collectives.
                    if constexpr (FTILE_ROWS == 16) {
                        if (rb0 + 16 * mb + 8 * r >= rows) continue;
                    }
                    const int row = rb0 + 16 * mb + m0 + 8 * r;
                    const bool row_ok = row < rows;   // gate stores only:
                    // warp-collective shfl below must stay uniform under
                    // the task loop (divergent continue crashes it).
#ifdef FROLE12_CACHE_ROUTE_WEIGHT
                    const float weight = row_ok ? route_weights[row] : 0.f;
#else
                    const float weight = row_ok ? route_weight(row) : 0.f;
#endif
                    float vals[2][4];
#pragma unroll
                    for (int nb = 0; nb < 2; ++nb)
#pragma unroll
                        for (int jj = 0; jj < 4; ++jj) {
                            const int j = 2 * r + (jj & 1) + 4 * (jj >> 1);
                            const uint32_t gu = sm100_bf16_pair(dreg[0][mb][nb][j], dreg[1][mb][nb][j]);
                            float up = __uint_as_float(gu & 0xffff0000u);
                            uint32_t gate_bits = gu & 0xffffu;
#ifdef FROLE12_ACTIVATION_CLAMP
                            const float unclamped_gate = __uint_as_float(gate_bits << 16);
                            if (unclamped_gate > bound) gate_bits = __float_as_uint(bound) >> 16;
                            up = fminf(fmaxf(up, -bound), bound);
#endif
#ifdef FROLE12_PRECISE_MATH
                            const float gate = __uint_as_float(gate_bits << 16);
                            const float silu = gate / (1.f + expf(-gate));
#else
                            const float silu = sm100_gate_table[gate_bits];
#endif
                            vals[nb][jj] = (silu * up) * weight;
                        }
                    float amax = 0.f;
#pragma unroll
                    for (int nb = 0; nb < 2; ++nb)
#pragma unroll
                        for (int jj = 0; jj < 4; ++jj)
                            amax = fmaxf(amax, fabsf(vals[nb][jj]));
                    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 1));
                    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 2));

                    const int scale_exp = min(max(ceil_log2(amax) - 2 + 127, 113), 254);
                    const float inv_scale =
                        __ppu_bits_u2f((uint32_t)(254 - scale_exp) << 23);
                    if (row_ok && (lane & 3) == 0)
                        act_sf[activation_offset<4>(row0, row, rows, inter >> 5,
                                                     (n_base + warp * 32) >> 5)] =
                            (uint8_t)scale_exp;

#if defined(FROLE12_PACK_ACTIVATION_STORES)
                    uint32_t packed = 0;
#pragma unroll
                    for (int nb = 0; nb < 2; ++nb) {
#pragma unroll
                        for (int b2 = 0; b2 < 2; ++b2) {
                            const float v0 = vals[nb][2 * b2 + 0] * inv_scale;
                            const float v1 = vals[nb][2 * b2 + 1] * inv_scale;
                            packed |= uint32_t(cvt_e2m1x2(v1, v0)) << (8 * (2 * nb + b2));
                        }
                    }
                    // Transpose four lanes' four bytes. The exact same
                    // quantized bytes then use one aligned store per lane.
                    uint32_t neighbor = __shfl_xor_sync(0xffffffffu, packed, 1);
                    packed = __byte_perm(packed, neighbor, (lane & 1) ? 0x3715 : 0x6240);
                    neighbor = __shfl_xor_sync(0xffffffffu, packed, 2);
                    packed = __byte_perm(packed, neighbor, (lane & 2) ? 0x3276 : 0x5410);
                    if (row_ok) {
                        const int byte_col = (n_base >> 1) + warp * 16 + (lane & 3) * 4;
                        *reinterpret_cast<uint32_t*>(act_packed +
                            activation_offset<64>(row0, row, rows, inter >> 1, byte_col)) = packed;
                    }
#else
    #pragma unroll
                    for (int nb = 0; nb < 2; ++nb) {
    #pragma unroll
                        for (int b2 = 0; b2 < 2; ++b2) {
                            const float v0 = vals[nb][2 * b2 + 0] * inv_scale;
                            const float v1 = vals[nb][2 * b2 + 1] * inv_scale;
                            const uint8_t byte = cvt_e2m1x2(v1, v0);
                            const int col = warp * 32 + nb * 16 + 2 * (lane & 3) + 8 * b2;
                            if (row_ok)
                                act_packed[activation_offset<64>(row0, row, rows, inter >> 1,
                                                                  (n_base + col) >> 1)] = byte;
                        }
                    }
#endif
                }
            }
        }
    };
#if defined(FROLE12_OVERLAP_FC2) || defined(FROLE12_ADAPT_M)
    // Select the shorter MMA chain once per tile using the received row count.
    static_assert(m_blocks_per_tile >= 2);
#ifdef FROLE12_ADAPT_M16
    if (rows <= 16)
        compute_rows(std::integral_constant<int, 1>{});
    else
#endif
#ifdef FROLE12_ADAPT_M_ALL
    static_assert(FTILE_ROWS == 80);
    if (rows <= 16)
        compute_rows(std::integral_constant<int, 1>{});
    else if (rows <= 32)
        compute_rows(std::integral_constant<int, 2>{});
    else if (rows <= 48)
        compute_rows(std::integral_constant<int, 3>{});
    else
#endif
    if (m_blocks_per_tile >= 4 && rows <= 32)
        compute_rows(std::integral_constant<int, 2>{});
    else if (rows <= FTILE_ROWS - 16)
        compute_rows(std::integral_constant<int, m_blocks_per_tile - 1>{});
    else
#endif
        compute_rows(std::integral_constant<int, m_blocks_per_tile>{});

#ifdef FROLE12_SHARED_EXPERT
        // FC1 can fill the private activation rows before routed metadata.
        // Publish only after CTA0 has reset the shared FC2 queue and counters.
        if (early) acquire_metadata();
#endif
#ifdef FROLE12_EXPERT_PAYLOAD
        // FC1 can start as soon as this expert has all its rows. The W1
        // counters and source histograms still share storage until finalize.
        wait_input();
#endif
        // publish: per-warp fence (PPU fences drain only the issuing warp),
        // then release-count the finished slab.
#ifndef NO_PUBLISH
        if (!grid_phase) {
            if ((tid & 31) == 0) __threadfence();
            __syncthreads();
#ifdef FROLE12_READY_FC2
            if (tid == 0) {
                int* count = &w1_cnt[e * ngrp + g];
                if (finish_fc1_slab(count) == nb1 - 1) {
                    // Import every slab's release before publishing the group.
                    (void)cnt_acquire(count);
                    const int ticket = atomicAdd(fc2_group_tail, 1);
                    cnt_release(fc2_ready_groups + ticket, group_code + 1);
                }
            }
#else
            if (tid == 0) cnt_release(&w1_cnt[e * ngrp + g], 1);
#endif
        }
#endif
    }   // close task loop
    }

    // Grouped receive already acquired the input payload. Its unused
    // input_ready storage starts at SCRATCH[4], cleared before dispatch.
    int* grid_gate = const_cast<int*>(input_ready);
    bool first_l2 = true;
    if (grid_phase) {
#ifdef FROLE12_EXPERT_PAYLOAD
        // The FC2 grid gate shares the receive-ready word. Even CTAs
        // without FC1 work must acquire the receiver's first ticket.
        wait_input();
#endif
        __syncwarp();
        if ((tid & 31) == 0) __threadfence();
        __syncthreads();
        if (tid == 0) cnt_release(grid_gate, 1);
    }

    // ===================== pass 2: L2 tiles (e, g, hb) =====================
    // gated on all nb1 L1 slabs of (e, g).  The first NBUF-1 W2 tiles are
    // PRE-GATE (weights only), so the B stream overlaps the gate spin.
    {
        __shared__ uint2 task_records[FTILE_ROWS];
        const int kb_half = inter >> 1;
        const int n_scale = inter >> 5;
        const int ntiles = kb_half >> 6;
        const int m0 = lane >> 2;

#ifdef FROLE12_READY_FC2
    // Wait on the FC2 queue after publishing this CTA's FC1 slabs.
#ifdef FROLE12_PREFETCH_FC2_FROM_FC1
    bool prefetched = fc2_preissued;
    if (!fc2_claimed) {
        if (tid == 0) fc2_task = atomicAdd(fc2_task_head, 1);
        __syncthreads();
    }
#else
    __shared__ int fc2_group, fc2_task;
    bool prefetched = false;
    if (tid == 0) fc2_task = atomicAdd(fc2_task_head, 1);
    __syncthreads();
#endif
    int next_task = fc2_task;
    for (int t = next_task; t < workgroup_count * hb2; t = next_task) {
        const int n_base = (t % hb2) * (2 * FBCOLS);
        if (!prefetched) {
            if (tid == 0) {
                int ready;
                while ((ready = cnt_acquire(fc2_ready_groups + t / hb2)) == 0)
                    __nanosleep(FROLE12_POLL_NS);
                fc2_group = ready;
            }
            __syncthreads();
        }
        __threadfence();
        const int group_code = fc2_group - 1;
        const int e = group_code & group_expert_mask;
        const int g = group_code >> group_expert_bits;
#else
#ifdef FROLE12_OVERLAP_FC2
    // Give CTAs with an extra FC1 tile the last FC2 residue classes.
    // FC2 still acquires the completion count of the group it consumes.
    const int extra_l1 = workgroup_count * nb1 % ncta;
    const int first_l2_task = blockIdx.x >= extra_l1 ? blockIdx.x - extra_l1
                                                     : blockIdx.x + ncta - extra_l1;
#else
    const int first_l2_task = blockIdx.x;
#endif
    for (int t = first_l2_task; t < workgroup_count * hb2; t += ncta) {
        const int n_base = (t % hb2) * (2 * FBCOLS);      // this CTA's hidden-col slab
        const int group_code = expert_rows[t / hb2];
        const int e = group_code / ngrp;
        const int g = group_code - e * ngrp;
#endif
        const int row0 = row_starts[e] + g * FTILE_ROWS;
        int rows = row_counts[e] - g * FTILE_ROWS;
        if (rows <= 0) continue;
        if (rows > FTILE_ROWS) rows = FTILE_ROWS;

    constexpr int panel = 512;
    static_assert(FCONST_H % panel == 0 && panel % (2 * FBCOLS) == 0);
    const size_t panel_index = n_base / panel;
    const auto w2 = w2_panel(e, panel_index);
    auto issue = [&](int kt, int sb) {
        issue_w2_stage(sb, w2, w2_scale_base(e), n_base, kt);
    };
    // A stream (act rows, COMPACT layout): separate from issue() so the
    // PRE-GATE weight prefetch can run without the act dependency.
    auto issueA = [&](int kt, int sb, int rowh, int dh) {
#ifdef FROLE12_ASF_CHUNK
        if (kt == 0 && warp == asf_issuer) {
#ifdef FROLE12_K_MAJOR_ACT
            aiu_copy_linear_b32(&staged_asf.l2[0][0], act_sf + size_t(rowh) * n_scale,
                                dh, ntiles, FTILE_ROWS + 4, ntiles, uint64_t(dh) * 4, 0, 0);
#else
            aiu_copy_linear_b32(&staged_asf.l2[0][0], act_sf + size_t(rowh) * n_scale,
                                ntiles, dh, FCONST_I / 128 + 4, FTILE_ROWS, n_scale, 0, 0);
#endif
        }
#endif
#ifdef FROLE12_K_MAJOR_ACT
        if (warp == 2)
            aiu_copy_b8_swzl(&A_cube[sb][0][0],
                             act_packed + activation_offset<64>(rowh, 0, dh, kb_half, kt << 6),
                             64, dh, 64, activation_copy_rows(dh), 64, 0, 0, 1);
#else
        if (warp == 2)
            aiu_copy_b8_swzl(&A_cube[sb][0][0],
                             act_packed + (size_t)rowh * kb_half, kb_half, dh,
                             64, activation_copy_rows(dh), kb_half, kt << 6, 0, 1);
#endif
    };
    auto bsf_load = [&](int ks, uint32_t& b0, uint32_t& b1) {
    };
    auto bsf_store = [&](int sb, uint32_t b0, uint32_t b1) {
    };

#if defined(FROLE12_ASF_CHUNK) || defined(FROLE12_FC2_STAGE_ASF)
    auto asf_load = [&](int, auto, uint32_t&) {};
    auto asf_store = [&](int, uint32_t) {};
#else
    auto asf_load = [&](int ks, auto prow, uint32_t& asfv) {
        if (tid < FTILE_ROWS)
            asfv = *reinterpret_cast<const uint32_t*>(
                &act_sf[activation_offset<4>(row0, prow(tid) - row0, rows, n_scale, ks << 2)]);
    };
    auto asf_store = [&](int sb, uint32_t asfv) {
        if (tid < FTILE_ROWS) Asf4[sb][tid] = asfv;
    };
#endif

    // pre-gate: launch tiles 0..NBUF-2 of the W2 slab + tile-0 Bsf staging
    // (weights and scales only -- nothing depends on the pass-1 outputs).
    // The AIU groups land while this CTA spins on the w1_cnt gate.
    uint32_t pf_b0 = 0, pf_b1 = 0;
    uint32_t pf_asf = 0;
#ifdef FROLE12_READY_FC2
    if (!prefetched) {
#else
    if (t == first_l2_task
#ifdef FROLE12_PREFETCH_FC2_FROM_FC1
        && !fc2_preissued
#endif
    ) {
#endif
        for (int p = 0; p < NBUF - 1; ++p) {
            if (p < ntiles) issue(p, p + 1);
            cp_async_fence();
        }
    }
    if (ntiles > 0) {
        bsf_load(0, pf_b0, pf_b1);
        bsf_store(1, pf_b0, pf_b1);
    }

#ifndef FROLE12_READY_FC2
#ifdef FROLE12_EXPERT_PAYLOAD
    // A CTA with only FC2 work can prefetch W2, but the receiver must
    // clear the shared W1 counters before it can test them.
    wait_input();
#endif
    if (grid_phase) {
        if (first_l2) {
#ifdef FROLE12_EARLY_METADATA
            constexpr int payload_ready = 1;
#else
            constexpr int payload_ready = 0;
#endif
            if (tid == 0)
                while (cnt_acquire(grid_gate) < ncta + payload_ready) __nanosleep(FROLE12_POLL_NS);
            __syncthreads();
            __threadfence();
            first_l2 = false;
        }
        __syncthreads();
    } else
    {
        if (tid == 0) {
            const int* p = &w1_cnt[e * ngrp + g];
            // exponential backoff: each acquire load is MMIO-class (~0.75us).
            int ns = 512;
            while (cnt_acquire(p) < nb1) {
                __nanosleep(ns);
                if (ns < 4096) ns <<= 1;
            }
        }
        __syncthreads();
        __threadfence();   // subscriber fence after observing the count
    }
#endif

    auto compute_rows = [&](auto m_blocks) {
        constexpr int MB = decltype(m_blocks)::value;
        constexpr int rb0 = 0;
        const int nm = (rows - rb0 + 15) >> 4;
        const int rows_here = rows - rb0;
        auto prow_g = [&](int r) {
            return row0 + rb0 + ((r < rows_here) ? r : rows_here - 1);
        };
#ifdef FROLE12_A_TSM_OFFSET
        const uint32_t a_units = smem_addr32(&A_cube[0][0][0]) >> 4;
#endif
#ifdef FROLE12_B_TSM_OFFSET
        const uint32_t b_units = smem_addr32(&CUBE_G(0)[0][0]) >> 4;
#endif
        float dreg[MB][fc2_nb][8];        // [m16][n16][j]
#pragma unroll
        for (int mb = 0; mb < MB; ++mb)
#pragma unroll
            for (int nb = 0; nb < fc2_nb; ++nb)
#pragma unroll
                for (int j = 0; j < 8; ++j) dreg[mb][nb][j] = 0.f;

        const int arowh = row0 + rb0;        // A window base (compact rows)
        const int adh = rows - rb0;          // A window height (padz beyond)
        // prologue (B tiles pre-gated above unless FNOPREGATE): stage A.
        if (rb0 == 0) {
            // A copies for the pre-gated tiles must wait for the gate (act
            // data); they join fresh AIU groups drained by the wait below.
#ifdef FROLE12_W2_FULL_RING
            // The previous task reserved slot zero for its output. Load
            // the last prefetched weight tile now that the wire is free.
            if (NBUF - 1 < ntiles) issue(NBUF - 1, 0);
            cp_async_fence();
            constexpr int prefill = NBUF;
#else
            constexpr int prefill = NBUF - 1;
#endif
#pragma unroll
            for (int p = 0; p < prefill; ++p) {
                if (p < ntiles) issueA(p, p + 1 == NBUF ? 0 : p + 1, arowh, adh);
                cp_async_fence();
            }
            if (ntiles > 0) {
                asf_load(0, prow_g, pf_asf);
                asf_store(1, pf_asf);
            }
        } else {
            // later rb0 groups re-run the full prologue
#pragma unroll
            for (int p = 0; p < NBUF - 1; ++p) {
                if (p < ntiles) { issue(p, p + 1); issueA(p, p + 1, arowh, adh); }
                cp_async_fence();
            }
            if (ntiles > 0) {
                asf_load(0, prow_g, pf_asf);
                asf_store(1, pf_asf);
                bsf_load(0, pf_b0, pf_b1);
                bsf_store(1, pf_b0, pf_b1);
            }
        }
#ifdef FROLE12_FC2_STAGE_ASF
        // Build the MMA operands once while the first AIU copies are in flight.
        for (int index = tid; index < fc2_scale_tiles * fc2_scale_rows; index += FTHREADS) {
            const int k = index / fc2_scale_rows;
            const int sr = index % fc2_scale_rows;
            const int row = 16 * (sr / 8) + sr % 8;
            uint32_t low = 0, high = 0;
            if (row < adh)
                low = *reinterpret_cast<const uint32_t*>(
                    act_sf + activation_offset<4>(arowh, row, adh, n_scale, 4 * k));
            if (row + 8 < adh)
                high = *reinterpret_cast<const uint32_t*>(
                    act_sf + activation_offset<4>(arowh, row + 8, adh, n_scale, 4 * k));
            asf_tile.fc2[k][0][sr] = __byte_perm(low, high, 0x5410);
            asf_tile.fc2[k][1][sr] = __byte_perm(low, high, 0x7632);
        }
#endif
        // Dispatch published the original routing records with the input.
        // Stage just the destination keys while the operands are in flight.
        if (tid < rows
#ifdef FROLE12_SHARED_EXPERT
            && e != FCONST_E
#endif
        ) {
            const uint4 rec = grouped_records[input_start(e, g) + tid];
            task_records[tid] = uint2{rec.x, rec.y};
        }
#if !defined(FROLE12_W2_FULL_RING) && (defined(FVBOTTOM) || defined(FISSUE_POST))
        cp_async_wait_all();
        __syncthreads();
#endif

        int sroll = 1;   // buffer of tile ks (rolling wrap, NO division)
        for (int ks = 0; ks < ntiles; ++ks) {
            const int s1r = sroll + 1;
            const int sb1 = (s1r == NBUF) ? 0 : s1r;          // tile ks+1
            const int sbm = sroll ? sroll - 1 : NBUF - 1;     // tile ks+NBUF-1
            if (ks + 1 < ntiles) {
                asf_load(ks + 1, prow_g, pf_asf);
                bsf_load(ks + 1, pf_b0, pf_b1);
            }
#if defined(FROLE12_W2_FULL_RING)
            cp_async_wait_full_ring(ks, ntiles);
            __syncthreads();
#elif defined(FVBOTTOM) && !defined(FISSUE_POST)
            if (ks + NBUF - 1 < ntiles) {
                issue(ks + NBUF - 1, sbm);
                issueA(ks + NBUF - 1, sbm, arowh, adh);
            }
            cp_async_fence();
#else
            cp_async_wait_pipe(ks, ntiles);
            __syncthreads();
            if (ks + NBUF - 1 < ntiles) {
                issue(ks + NBUF - 1, sbm);
                issueA(ks + NBUF - 1, sbm, arowh, adh);
            }
            cp_async_fence();
#endif

            const int s = sroll;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                uint32_t ac[MB][4], mac[MB];
#pragma unroll
                for (int group = 0; group < (MB + 3) / 4; ++group) {
                    const int mb_part = min(4 * group + (lane & 3), MB - 1);
#ifdef FROLE12_FC2_STAGE_ASF
                    mac[group] = asf_tile.fc2[ks][h][8 * mb_part + (lane >> 2)];
#else
                    const int arow = 16 * mb_part + (lane >> 2);
#ifdef FROLE12_ASF_CHUNK
#ifdef FROLE12_K_MAJOR_ACT
#ifdef FROLE12_FC2_PACK_ASF
                    mac[group] = __byte_perm(staged_asf.l2[ks][arow], staged_asf.l2[ks][arow + 8], h == 0 ? 0x5410 : 0x7632);
#else
                    mac[group] = ((staged_asf.l2[ks][arow] >> (16 * h)) & 0xFFFFu)
                        | (((staged_asf.l2[ks][arow + 8] >> (16 * h)) & 0xFFFFu) << 16);
#endif
#else
#ifdef FROLE12_FC2_PACK_ASF
                    mac[group] = __byte_perm(staged_asf.l2[arow][ks], staged_asf.l2[arow + 8][ks], h == 0 ? 0x5410 : 0x7632);
#else
                    mac[group] = ((staged_asf.l2[arow][ks] >> (16 * h)) & 0xFFFFu)
                        | (((staged_asf.l2[arow + 8][ks] >> (16 * h)) & 0xFFFFu) << 16);
#endif
#endif
#else
#ifdef FROLE12_FC2_PACK_ASF
                    mac[group] = __byte_perm(Asf4[s][arow], Asf4[s][arow + 8], h == 0 ? 0x5410 : 0x7632);
#else
                    mac[group] = ((Asf4[s][arow] >> (16 * h)) & 0xFFFFu)
                        | (((Asf4[s][arow + 8] >> (16 * h)) & 0xFFFFu) << 16);
#endif
#endif
#endif
                }
#ifdef FROLE12_W2_EARLY_BSCALE
                const int packed_n = 16 * (fc2_nb * warp + (lane & (fc2_nb - 1))) + (lane >> 2);
                const uint32_t packed_mb = Bsf[s][packed_n + 8 * h];
#endif
#pragma unroll
                for (int mb = 0; mb < MB; ++mb) {
#ifdef FROLE12_A_TSM_OFFSET
                    const uint32_t address = a_units + 4 * (s * FTILE_ROWS + 16 * mb) + 2 * h;
                    ldmatrix_a_at(ac[mb], address);
#else
                    ldmatrix_a_frag(ac[mb], &A_cube[s][16 * mb][32 * h]);
#endif
                }
                uint32_t bfr[2][4];
#ifndef FROLE12_W2_EARLY_BSCALE
                const int packed_n = 16 * (fc2_nb * warp + (lane & (fc2_nb - 1))) + (lane >> 2);
                const uint32_t packed_mb = Bsf[s][packed_n + 8 * h];
#endif
                auto load_b = [&](int part) {
#ifdef FROLE12_B_TSM_OFFSET
                    const int side = warp >= FTHREADS / 64;
                    const int row = 16 * (fc2_nb * (warp & (FTHREADS / 64 - 1)) + part);
                    const uint32_t address = b_units + 4 * ((2 * s + side) * FBCOLS + row) + 2 * h;
                    ldmatrix_b_at(bfr[part & 1], address);
#else
                    const uint8_t (*cbe)[64] = warp < FTHREADS / 64 ? CUBE_G(s) : CUBE_U(s);
                    ldmatrix_b_frag(bfr[part & 1],
                        &cbe[16 * (fc2_nb * (warp & (FTHREADS / 64 - 1)) + part)][32 * h], 32, 1);
#endif
                };
                load_b(0);
#pragma unroll
                for (int part = 0; part < fc2_nb; ++part) {
                    if (part + 1 < fc2_nb) load_b(part + 1);
#pragma unroll
                    for (int mb = 0; mb < MB; ++mb)
                        mma_m16n16k64_f4(dreg[mb][part], ac[mb],
                                        bfr[part & 1], mac[mb >> 2], packed_mb, mb & 3, part);
                }
            }
#if defined(FROLE12_W2_FULL_RING)
            if (ks + NBUF < ntiles) {
                __syncthreads();
                issue(ks + NBUF, sroll);
                issueA(ks + NBUF, sroll, arowh, adh);
                cp_async_fence();
            }
#elif defined(FVBOTTOM)
            cp_async_wait_pipe(ks + 1, ntiles);
#endif
            if (ks + 1 < ntiles) {
                bsf_store(sb1, pf_b0, pf_b1);
                asf_store(sb1, pf_asf);
            }
#if !defined(FROLE12_W2_FULL_RING) && (defined(FVBOTTOM) || defined(FISSUE_POST))
            __syncthreads();
#endif

            // epilogue inside the k loop, on the last tile (v11 legacy
            // placement: FL2EPIOUT regresses ~+450us on the hgcc RA cliff).
            if (ks + 1 == ntiles) {
        // The final MMA has consumed all B tiles. Reuse their storage
        // for BF16 output without increasing the TSM allocation.
        // The bottom-wait path already joined all warps above.
#if defined(FROLE12_W2_FULL_RING) || (!defined(FVBOTTOM) && !defined(FISSUE_POST))
        __syncthreads();
#endif

#ifdef FROLE12_BATCH_FC2_ROWS
        constexpr int blocks_per_batch = FROLE12_BATCH_FC2_ROWS;
#else
        constexpr int blocks_per_batch = 1;
#endif
#ifdef FROLE12_PAD_WEIGHT_RING
        static_assert(blocks_per_batch <= 2);
#else
        static_assert(blocks_per_batch <= 2 * NBUF);
#endif
        auto prefetch_w2 = [&] {
#ifdef FROLE12_READY_FC2
        if (tid == 0) {
            const int claim = atomicAdd(fc2_task_head, 1);
            fc2_task = claim;
            fc2_group = claim < workgroup_count * hb2 ?
                cnt_acquire(fc2_ready_groups + claim / hb2) : 0;
        }
        __syncthreads();
        next_task = fc2_task;
        prefetched = fc2_group != 0;
        if (prefetched) {
            const int next_column = (next_task % hb2) * (2 * FBCOLS);
            const int next_e = (fc2_group - 1) & group_expert_mask;
#else
        const int next_task = t + ncta;
        if (next_task < workgroup_count * hb2) {
            const int next_column = (next_task % hb2) * (2 * FBCOLS);
            const int next_e = expert_rows[next_task / hb2] / ngrp;
#endif
    constexpr int panel = 512;
    static_assert(FCONST_H % panel == 0 && panel % (2 * FBCOLS) == 0);
    const size_t panel_index = next_column / panel;
    const auto w2 = w2_panel(next_e, panel_index);
    auto next_issue = [&](int kt, int sb) {
        issue_w2_stage(sb, w2, w2_scale_base(next_e), next_column, kt);
    };

#pragma unroll
            for (int p = 0; p < NBUF - 1; ++p) {
                if (p < ntiles) next_issue(p, p + 1);
                cp_async_fence();
            }
        }
        };
        // Larger return batches reuse the next B slot. Refill it only
        // after the BF16 rows have been read by every warp.
        if constexpr (blocks_per_batch <= 2) prefetch_w2();
        auto* wire = reinterpret_cast<uint32_t*>(&CUBE_G(0)[0][0]);
#pragma unroll
        for (int base_mb = 0; base_mb < MB; base_mb += blocks_per_batch) {
            if (base_mb < nm) {
#pragma unroll
                for (int part = 0; part < blocks_per_batch; ++part) {
                    const int mb = base_mb + part;
                    if (mb < MB && mb < nm) {
#pragma unroll
                        for (int r = 0; r < 2; ++r) {
                            const int row = rb0 + 16 * mb + m0 + 8 * r;
                            if (row < rows) {
#pragma unroll
                                for (int nb = 0; nb < fc2_nb; ++nb) {
#if defined(FROLE12_PACK_FC2_STORES)
                                    const int word = 8 * (fc2_nb * warp + nb) + 2 * (lane & 3);
                                    const int swizzled = word ^ ((m0 & 3) << 3);
                                    const int j = 2 * r;
                                    const uint2 values{
                                        sm100_bf16_pair(dreg[mb][nb][j], dreg[mb][nb][j + 1]),
                                        sm100_bf16_pair(dreg[mb][nb][j + 4], dreg[mb][nb][j + 5])};
                                    auto* row_words = wire + (16 * part + m0 + 8 * r) * FBCOLS;
                                    *reinterpret_cast<uint2*>(row_words + swizzled) = values;
#else
#pragma unroll
                                    for (int pair = 0; pair < 2; ++pair) {
                                        const int j = 2 * r + 4 * pair;
                                        const int col = 16 * (fc2_nb * warp + nb) + 2 * (lane & 3) + 8 * pair;
                                        wire[(16 * part + m0 + 8 * r) * FBCOLS + col / 2] =
                                            sm100_bf16_pair(dreg[mb][nb][j], dreg[mb][nb][j + 1]);
                                    }
#endif
                                }
                            }
                        }
                    }
                }
                __syncthreads();
                for (int local_row = warp; local_row < 16 * blocks_per_batch; local_row += FTHREADS / 32) {
                    const int row = rb0 + 16 * base_mb + local_row;
                    if (row < rows) {
                        char* destination;
                        bool local;
#ifdef FROLE12_SHARED_EXPERT
                        if (e == FCONST_E) {
                            destination = shared_y + ((g * FTILE_ROWS + row) * hidden + n_base) * 2;
                            local = true;
                        } else
#endif
                        {
                            const uint2 rec = task_records[row];
                            const int owner = rec.y >> 8;
                            const int slot = rec.y & 255;
                            const int token = rec.x;
                            destination = combine_sym + combine_peers[owner] + combine_offset;
                            destination += ((token * topk + slot) * hidden + n_base) * 2;
                            local = combine_peers[owner] == 0;
                        }
                        for (int off = 0; off < FBCOLS; off += 128) {
#ifdef FROLE12_PACK_FC2_STORES
                            // Undo the pair packing and row rotation for the bulk store.
                            const int packed_lane = (lane & ~7) | ((lane & 3) << 1) | ((lane >> 2) & 1);
                            const int wire_lane = packed_lane ^ ((local_row & 3) << 3);
#else
                            const int wire_lane = lane;
#endif
                            const auto* words = wire + local_row * FBCOLS + off + wire_lane;
                            const uint4 payload{words[0], words[32], words[64], words[96]};
                            if (local) {
#ifdef FROLE12_SCALAR_LOCAL_WIRE
                                auto* dst = reinterpret_cast<volatile uint32_t*>(destination) + off + lane;
                                dst[0] = payload.x; dst[32] = payload.y;
                                dst[64] = payload.z; dst[96] = payload.w;
#else
                                __ppu_global_stcg_bulk_b32x4(destination + 4 * off, payload);
#endif
                            } else {
                                __ppu_remote_store_bulk_volatile_b32x4(
                                    reinterpret_cast<volatile void*>(destination + 4 * off), payload);
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }
        if constexpr (blocks_per_batch > 2) prefetch_w2();
            }   // close if (ks + 1 == ntiles)
            sroll = sb1;   // advance the rolling buffer index
        }       // close ks loop
#ifndef FROLE12_READY_FC2
        __syncthreads();   // buffer-reuse guard for static FC2 tasks
#endif
    };
#if defined(FROLE12_OVERLAP_FC2) || defined(FROLE12_ADAPT_M)
#ifdef FROLE12_ADAPT_M16
    if (rows <= 16)
        compute_rows(std::integral_constant<int, 1>{});
    else
#endif
#ifdef FROLE12_ADAPT_M_ALL
    static_assert(FTILE_ROWS == 80);
    if (rows <= 16)
        compute_rows(std::integral_constant<int, 1>{});
    else if (rows <= 32)
        compute_rows(std::integral_constant<int, 2>{});
    else if (rows <= 48)
        compute_rows(std::integral_constant<int, 3>{});
    else
#endif
    if (m_blocks_per_tile >= 4 && rows <= 32)
        compute_rows(std::integral_constant<int, 2>{});
    else if (rows <= FTILE_ROWS - 16)
        compute_rows(std::integral_constant<int, m_blocks_per_tile - 1>{});
    else
#endif
        compute_rows(std::integral_constant<int, m_blocks_per_tile>{});
        }   // close pass-2 task loop
    }
}

namespace role12 {
struct Params {
    void* p[NPTR];
    int i[NINT];
    int64_t peer[8];
    int start_generation, barr_off, ccnt_off, cgen_off, cbuf_off;
#ifdef FROLE12_COMBINE
    uint16_t* combined_output;
#endif
};
#ifdef FROLE12_PRECLEARED_DISPATCH
constexpr bool precleared_dispatch = true;
#else
constexpr bool precleared_dispatch = false;
#endif
template<class V> __device__ __forceinline__ V* ptr(const Params& p, Ptr k) {
    return reinterpret_cast<V*>(p.p[k]);
}
__device__ __forceinline__ int acquire_sys(volatile int* p) {
    int v;
    asm volatile("ppu.ld.acquire.sys.global.s32 %0, [%1];"
                 : "=r"(v) : "l"((void*)p) : "memory");
    return v;
}
__device__ __forceinline__ void add_sys(volatile int* p, int v) {
    asm volatile("ppu.atom.add.release.sys.global.s32 _, [%0], %1;"
                 :: "l"((void*)p), "r"(v) : "memory");
}
__device__ __forceinline__ void store_sys(volatile int* p, int v) {
    asm volatile("ppu.st.release.sys.global.s32 [%0], %1;"
                 :: "l"((const void*)p), "r"(v) : "memory");
}
__device__ __forceinline__ void publish_dispatch(const Params& p, int destination,
                                                bool metadata) {
    char* peer = ptr<char>(p, SYM) + p.peer[destination];
    auto* flags = reinterpret_cast<volatile int*>(peer + p.i[GEN_OFF]);
    if constexpr (precleared_dispatch) {
        // Each source rank owns one word in each phase.
        const int phase = metadata ? FCONST_RANKS : 0;
        store_sys(flags + phase + p.i[MY], p.start_generation / 2);
    } else {
        add_sys(flags + p.i[MY], 1);
    }
}
__device__ __forceinline__ uint4 load16(const void* p) {
    uint4 v;
    asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void store16(void* p, uint4 v) {
    asm volatile("st.global.cg.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ uint4 tile_load(const void* p) {
    return __ppu_global_ldcg_bulk_b32x4(p);
}
__device__ __forceinline__ void tile_store(void* p, uint4 v) {
    __ppu_remote_store_bulk_volatile_b32x4(reinterpret_cast<volatile void*>(p), v);
}
__device__ __forceinline__ void drain_gpu() {
    if ((threadIdx.x & 31) == 0) __threadfence();
    __syncthreads();
}
// The legacy protocol resets its communication counters before each request.
__device__ __forceinline__ void publish_start(const Params& p) {
    char* sym = ptr<char>(p, SYM);
    const int ranks = FCONST_RANKS;
    const int lane = threadIdx.x;
    if (lane < 32) {
        if (lane < ranks) {
            reinterpret_cast<volatile int*>(sym + p.i[CNT_OFF])[lane] = 0;
            reinterpret_cast<volatile int*>(sym + p.i[GEN_OFF])[lane] = 0;
            reinterpret_cast<volatile int*>(sym + p.ccnt_off)[lane] = 0;
            reinterpret_cast<volatile int*>(sym + p.cgen_off)[lane] = 0;
            ptr<int>(p, CURSORS)[lane] = 0;
        }
        if (lane == 0) *ptr<int>(p, DONE) = 0;
        for (int e = lane; e < FCONST_E; e += 32) ptr<int>(p, COUNTS)[e] = 0;
        // One warp owns the reset, and its system fence drains all its writes.
        __syncwarp();
        if (lane == 0) __ppu_threadfence_system_uncache();
    }
    __syncthreads();
    if (threadIdx.x < ranks) {
        char* peer = sym + p.peer[threadIdx.x] + p.barr_off;
        add_sys(reinterpret_cast<volatile int*>(peer) + p.i[MY], 2);
    }
}
__device__ __forceinline__ bool direct_dispatch(const Params& p) {
#ifdef FROLE12_DIRECT_ROUTES
#ifdef FROLE12_DIRECT_ONE_ROUTE_PER_WARP
    // CTA0 receives metadata. The other resident CTAs can send one route per warp.
    const int senders = gridDim.x - int(gridDim.x > 1);
    if (p.i[T] * FCONST_TOPK > senders * (FTHREADS / 32)) return false;
#endif
    return p.start_generation > 0 && p.i[T] <= 64;
#else
    return false;
#endif
}
__device__ __forceinline__ void initialize(const Params& p) {
    int* scratch = ptr<int>(p, SCRATCH);
    if (blockIdx.x == 0) {
        if (threadIdx.x < 5) scratch[2 + threadIdx.x] = 0;
        // Grouped dispatch overwrites the base of each nonempty expert.
        // Direct dispatch reserves its rows without the source histogram.
        if (!direct_dispatch(p)) {
            for (int j = threadIdx.x; j < FCONST_E * FCONST_RANKS; j += blockDim.x)
                ptr<int>(p, W1_CNT)[j] = 0;
        }
#ifdef FROLE12_SHARED_EXPERT
        // Shared row bounds are local. Publish them with initialization so
        // FC1 can read the same metadata before and after routed dispatch.
        if (threadIdx.x == 0) {
            ptr<int>(p, STARTS)[FCONST_E] = p.i[CAP_E];
            ptr<int>(p, COMPACT_SLOT)[FCONST_E] = p.i[T];
        }
#ifdef FROLE12_W1_GROUP_RECORDS
        for (int group = threadIdx.x, row = group * FTILE_ROWS;
             row < p.i[T]; group += blockDim.x, row += blockDim.x * FTILE_ROWS) {
            ptr<uint4>(p, COMPACT_T)[group] = uint4{
                unsigned((group << group_expert_bits) | FCONST_E),
                unsigned(p.i[CAP_E] + row),
                unsigned(row),
                unsigned(min(p.i[T] - row, FTILE_ROWS)),
            };
        }
#endif
#endif
        if constexpr (precleared_dispatch) {
            // Receive finalization cleared dispatch counters. Combine counts
            // persist across requests and only the local send ticket resets.
            if (threadIdx.x == 0) *ptr<int>(p, DONE) = 0;
        } else if (p.start_generation > 0) {
            publish_start(p);
        }
        drain_gpu();
        if (threadIdx.x == 0) cnt_release(scratch, 1);
    }
    if (threadIdx.x == 0)
        while (cnt_acquire(scratch) < p.i[GENERATION]) __nanosleep(FROLE12_POLL_NS);
    __syncthreads();
    __threadfence();
}

// Grouped dispatch and receive finalization. Grid barriers require a resident grid.
__device__ __forceinline__ void grouped_grid_barrier(const Params& p, int slot, int dispatch_ctas) {
    drain_gpu();
    int* flag = ptr<int>(p, SCRATCH) + slot;
    if (threadIdx.x == 0) {
        cnt_release(flag, 1);
        while (cnt_acquire(flag) < dispatch_ctas) __nanosleep(FROLE12_POLL_NS);
    }
    __syncthreads(); __threadfence();
}
__device__ __forceinline__ int grouped_reserve(int* counter, int n) {
    int old;
    asm volatile("ppu.atom.add.release.sys.global.s32 %0, [%1], %2;"
                 : "=r"(old) : "l"(counter), "r"(n) : "memory");
    return old;
}
#ifdef FROLE12_DIRECT_ROUTES
__device__ __forceinline__ void wait_peer_reset(const Params& p, int peer) {
    auto* arrived = reinterpret_cast<volatile int*>(ptr<char>(p, SYM) + p.barr_off);
    while (acquire_sys(arrived + peer) < p.start_generation) __nanosleep(FROLE12_POLL_NS);
}
#endif

__device__ __forceinline__ int route_id(const Params& p, int slot) {
    const int64_t expert = p.i[ROUTE_I64] ? ptr<const int64_t>(p, TOPK_IDX)[slot]
                                           : ptr<const int>(p, TOPK_IDX)[slot];
    return expert >= 0 && expert < FCONST_RANKS * FCONST_E ? int(expert) : -1;
}

__device__ __forceinline__ uint4 dispatch_scale_words(const Params& p, int token) {
    constexpr int sbw = FCONST_H / 128;
    const int lane = threadIdx.x & 31;
    const auto* sf = ptr<const uint32_t>(p, SF) + size_t(token) * sbw;
    return {lane < sbw ? sf[lane] : 0, 32 + lane < sbw ? sf[32 + lane] : 0,
            64 + lane < sbw ? sf[64 + lane] : 0, 96 + lane < sbw ? sf[96 + lane] : 0};
}

__device__ __forceinline__ void store_dispatch_record(
        const Params& p, char* peer, int owner, int local, int row, int token, int slot, uint4 scale) {
    constexpr int sbw = FCONST_H / 128;
    const int lane = threadIdx.x & 31;
    char* dst = peer + p.i[SF_OFF] + size_t(row) * routed_scale_stride;
    if (owner == p.i[MY]) {
        auto* out = reinterpret_cast<uint32_t*>(dst);
        out[lane] = scale.x;
        out[32 + lane] = scale.y;
        if constexpr (sbw > 64) {
            out[64 + lane] = scale.z;
            out[96 + lane] = scale.w;
        }
    } else if constexpr (sbw <= 64) {
        __ppu_remote_store_bulk_volatile_b32x2(reinterpret_cast<volatile void*>(dst),
                                              uint2{scale.x, scale.y});
    } else {
        tile_store(dst, scale);
    }
    if (lane == 0) {
        const uint4 record{unsigned(token), unsigned(slot | (p.i[MY] << 8)),
            __float_as_uint(ptr<const float>(p, TOPK_W)[token * FCONST_TOPK + slot]),
            unsigned(local)};
        store16(peer + p.i[REC_OFF] + size_t(row) * 16, record);
    }
}

__device__ __forceinline__ void zero_dispatch_slot(const Params& p, int index) {
    char* dst = ptr<char>(p, SYM) + p.cbuf_off + size_t(index) * FCONST_H * 2;
    for (int off = 0; off < FCONST_H * 2; off += 512)
        __ppu_global_stcg_bulk_b32x4(dst + off, uint4{});
}

#ifdef FROLE12_DIRECT_METADATA
__device__ __forceinline__ void reserve_dispatch_rows(
        const Params& p, int first, int count, int routes, int send_ctas, int* reserved_rows) {
    const int lane = threadIdx.x & 31;
    auto* sym = ptr<char>(p, SYM);
    if (lane < count) {
        const int route = first + lane;
        const int expert = route < routes ? route_id(p, route) : -1;
        int row = 0;
        if (expert >= 0) {
            const int owner = expert / FCONST_E, local = expert % FCONST_E;
            auto* counts = reinterpret_cast<int*>(sym + p.peer[owner] + p.i[CNT_OFF]);
            row = grouped_reserve(counts + local, 1);
            if (row < 0 || row >= p.i[CAP_E]) __builtin_trap();
        }
        reserved_rows[lane] = row;
    }
    __syncwarp();
    int last = 0;
    if (lane == 0) {
        __ppu_threadfence_system_uncache();
        int* scratch = ptr<int>(p, SCRATCH);
        while (cnt_acquire(scratch) < p.i[GENERATION]) __nanosleep(FROLE12_POLL_NS);
        last = atomicAdd(scratch + 3, 1) == send_ctas - 1;
    }
    if (__shfl_sync(0xffffffffu, last, 0) && lane < FCONST_RANKS)
        publish_dispatch(p, lane, true);
}
#endif

#if defined(FROLE12_TOKEN_FANOUT) && defined(FROLE12_DIRECT_METADATA)
// Each warp reuses the token loads for up to two independently reserved routes.
__device__ __forceinline__ void direct_token_fanout(const Params& p, int token, int send_ctas) {
    constexpr int warps = FTHREADS / 32;
    static_assert(warps < FCONST_TOPK && FCONST_TOPK <= 2 * warps);
    constexpr int hb = FCONST_H / 2;
    constexpr int prefetch = FROLE12_TOKEN_FANOUT < hb / 512 ? FROLE12_TOKEN_FANOUT : hb / 512;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int my = p.i[MY];
    char* sym = ptr<char>(p, SYM);
    const int first = token * FCONST_TOPK;
    __shared__ int reserved[FCONST_TOPK];
    if (warp == 0)
        reserve_dispatch_rows(p, first, FCONST_TOPK, first + FCONST_TOPK, send_ctas, reserved);

    struct Route {
        char* peer;
        char* payload;
        int row;
        int owner;
        int local;
        int slot;
    };
    auto locate = [&](int slot) -> Route {
        if (slot >= FCONST_TOPK) return {nullptr, nullptr, 0, -1, -1, slot};
        const int e = route_id(p, first + slot);
        if (e < 0) return {nullptr, nullptr, 0, -1, -1, slot};
        const int owner = e / FCONST_E, local = e % FCONST_E;
        return {sym + p.peer[owner], nullptr, 0, owner, local, slot};
    };
    auto a = locate(warp);
    auto b = locate(warp + warps);
    const auto* src = ptr<const char>(p, XP) + size_t(token) * hb;
    uint4 cached[prefetch];
    const bool has_payload = a.owner >= 0 || b.owner >= 0;
    if (has_payload) {
#pragma unroll
        for (int i = 0; i < prefetch; ++i) cached[i] = tile_load(src + i * 512);
    }
    __syncthreads();
    auto finish_route = [&](Route& route) {
        if (route.owner < 0) return;
        route.row = routed_input_row(route.local, reserved[route.slot], p.i[CAP_E]);
        route.payload = route.peer + p.i[POOL_OFF] + size_t(route.row) * hb;
    };
    finish_route(a);
    finish_route(b);

    auto store = [&](const Route& route, int offset, uint4 value) {
        if (route.owner == my)
            __ppu_global_stcg_bulk_b32x4(route.payload + offset, value);
        else if (route.owner >= 0)
            tile_store(route.payload + offset, value);
    };
    if (has_payload) {
#pragma unroll
        for (int i = 0; i < hb / 512; ++i) {
            const auto value = i < prefetch ? cached[i] : tile_load(src + i * 512);
            store(a, i * 512, value);
            store(b, i * 512, value);
        }
    }
    const uint4 scales = dispatch_scale_words(p, token);
    auto auxiliary = [&](const Route& route) {
        if (route.slot >= FCONST_TOPK) return;
        if (route.owner < 0)
            zero_dispatch_slot(p, first + route.slot);
        else
            store_dispatch_record(p, route.peer, route.owner, route.local, route.row, token, route.slot, scales);
    };
    auxiliary(a);
    auxiliary(b);
    __syncwarp();
    if (lane == 0) {
        // Make both records and payloads visible before publishing either route.
        __ppu_threadfence_system_uncache();
#ifdef FROLE12_EXPERT_PAYLOAD
        auto ready = [&](const Route& route) {
            if (route.owner >= 0) {
                auto* received = reinterpret_cast<volatile int*>(ptr<char>(p, POOL) + p.peer[route.owner]);
                add_sys(received + route.local, 1);
            }
        };
        ready(a);
        ready(b);
#endif
    }
    __syncthreads();
    __shared__ int publish;
    if (tid == 0)
        publish = atomicAdd(ptr<int>(p, DONE), 1) == send_ctas - 1;
    __syncthreads();
    if (publish && tid < FCONST_RANKS) publish_dispatch(p, tid, false);
}
#endif

__device__ __forceinline__ void grouped_dispatch(const Params& p, int dispatch_ctas) {
    const int tid=threadIdx.x, lane=tid&31, warp=tid>>5, warps=blockDim.x>>5;
    const int experts=FCONST_E*FCONST_RANKS, routes=p.i[T]*FCONST_TOPK;
    const bool direct = direct_dispatch(p);
    // A dedicated receiver can begin waiting for peer dispatches while
    // the remaining CTAs push the local routes.
#ifdef FROLE12_EARLY_METADATA
    // CTA0 joins the reservation barriers, then finalizes the receive schedule.
    const int first_sender = int(gridDim.x > 1);
#elif defined(FROLE12_DEDICATED_DISPATCH_RECEIVER)
    const int first_sender = direct && gridDim.x > 1;
#else
    constexpr int first_sender = 0;
#endif
    const int sender = int(blockIdx.x) - first_sender;
#if defined(FROLE12_TOKEN_FANOUT) && defined(FROLE12_DIRECT_METADATA)
    if (direct && p.i[T] > 0 && p.i[T] <= dispatch_ctas - first_sender) {
        if (sender >= 0 && sender < p.i[T]) direct_token_fanout(p, sender, p.i[T]);
        return;
    }
#endif
#ifdef FROLE12_DISPATCH_BY_TOKEN
    // Keep a token's routes in one CTA when there is one warp per route.
    const bool by_token = direct && FCONST_TOPK <= warps;
#else
    constexpr bool by_token = false;
#endif
    const int routes_per_cta = by_token ? FCONST_TOPK : warps;
    const int send_tasks = by_token ? p.i[T] : (routes + warps - 1) / warps;
    const int send_ctas = direct ? max(1, min(dispatch_ctas - first_sender, send_tasks))
                                 : dispatch_ctas - first_sender;
    int* bases = ptr<int>(p, W1_CNT) + experts;
    if (!direct) {
        int* counts=ptr<int>(p,W1_CNT);
        for (int j=blockIdx.x*blockDim.x+tid; j<routes; j+=dispatch_ctas*blockDim.x) {
            const int e = route_id(p, j);
            if (e >= 0) ptr<int>(p,BUCKET)[j]=atomicAdd(counts+e,1);
        }
        // Legacy dispatch acquires the peer resets at the histogram barrier.
        if (!precleared_dispatch && p.start_generation > 0 && blockIdx.x == 0 && tid < FCONST_RANKS) {
            auto* arrived = reinterpret_cast<volatile int*>(ptr<char>(p, SYM) + p.barr_off);
            while (acquire_sys(arrived + tid) < p.start_generation) __nanosleep(FROLE12_POLL_NS);
        }
        grouped_grid_barrier(p,2,dispatch_ctas);
        for (int e=blockIdx.x*blockDim.x+tid; e<experts; e+=dispatch_ctas*blockDim.x) {
            const int n=counts[e];
            if (n>0) {
                const int owner=e/FCONST_E, local=e-owner*FCONST_E;
                char* peer=ptr<char>(p,SYM)+p.peer[owner];
                const int base=grouped_reserve(reinterpret_cast<int*>(peer+p.i[CNT_OFF])+local,n);
                if (base<0 || base+n>p.i[CAP_E]) __builtin_trap();
                bases[e]=base;
            }
        }
        grouped_grid_barrier(p,3,dispatch_ctas);
#ifdef FROLE12_EARLY_METADATA
        if (blockIdx.x == 0 && tid < FCONST_RANKS)
            publish_dispatch(p, tid, true);
#endif
    }
    if (sender < 0 || sender >= send_ctas) return;
    constexpr int hb = FCONST_H / 2;
    char* sym=ptr<char>(p,SYM);
#ifdef FROLE12_DIRECT_METADATA
    int first_slot = 0;
    const bool one_slot = send_ctas == send_tasks;
    const bool vector_reserve = direct && one_slot;
    __shared__ int reserved_rows[FTHREADS / 32];
    if (vector_reserve && warp == 0)
        reserve_dispatch_rows(p, sender * routes_per_cta, routes_per_cta, routes, send_ctas, reserved_rows);
#ifdef FROLE12_DIRECT_METADATA_PREFETCH
#ifdef FROLE12_DISPATCH_PREFETCH
    constexpr int first_ahead = FROLE12_DISPATCH_PREFETCH;
#else
    constexpr int first_ahead = 5;
#endif
    int first_expert = -1;
    uint4 first_payload[first_ahead > 0 ? first_ahead : 1];
    const int first_route = sender * routes_per_cta + warp;
    if (direct && one_slot && (!by_token || warp < routes_per_cta) && first_route < routes) {
        first_expert = route_id(p, first_route);
        if (first_expert >= 0) {
            const char* src = ptr<const char>(p, XP) + int64_t(first_route / FCONST_TOPK) * hb;
#pragma unroll
            for (int tile = 0; tile < first_ahead; ++tile)
                first_payload[tile] = tile_load(src + tile * 512);
        }
    }
#endif
    if (vector_reserve) {
        __syncthreads();
        if (warp < routes_per_cta) first_slot = reserved_rows[warp];
    } else if (direct) {
        // Publish metadata as soon as all local reservations are complete.
        // Individual CTAs may begin moving payloads before this publication.
        if (lane == 0) {
            for (int j = sender * routes_per_cta + warp;
                 (!by_token || warp < routes_per_cta) && j < routes;
                 j += send_ctas * routes_per_cta) {
#ifdef FROLE12_DIRECT_METADATA_PREFETCH
                const int expert = one_slot ? first_expert : route_id(p, j);
#else
                const int expert = route_id(p, j);
#endif
                if (expert < 0) continue;
                const int owner = expert / FCONST_E, local = expert % FCONST_E;
                auto* counts = reinterpret_cast<int*>(sym + p.peer[owner] + p.i[CNT_OFF]);
                const int row = grouped_reserve(counts + local, 1);
                if (row < 0 || row >= p.i[CAP_E]) __builtin_trap();
                if (one_slot) first_slot = row;
                else ptr<int>(p, BUCKET)[j] = row;
            }
        }
        __syncwarp();
        if (lane == 0) __ppu_threadfence_system_uncache();
        __shared__ int publish_metadata;
        __syncthreads();
        if (tid == 0) {
            int* scratch = ptr<int>(p, SCRATCH);
            while (cnt_acquire(scratch) < p.i[GENERATION]) __nanosleep(FROLE12_POLL_NS);
            publish_metadata = atomicAdd(scratch + 3, 1) == send_ctas - 1;
        }
        __syncthreads();
        if (publish_metadata && tid < FCONST_RANKS)
            publish_dispatch(p, tid, true);
    }
#endif
    for (int j = sender * routes_per_cta + warp;
         (!by_token || warp < routes_per_cta) && j < routes;
         j += send_ctas * routes_per_cta) {
        const int token=j/FCONST_TOPK, slot=j-token*FCONST_TOPK;
#ifdef FROLE12_DIRECT_METADATA_PREFETCH
        const int expert = direct && one_slot ? first_expert : route_id(p, j);
#else
        const int expert = route_id(p, j);
#endif
        if (expert < 0) {
            // Materialize the missing slot as zero for both Role3 launchers.
            zero_dispatch_slot(p, j);
#ifdef FROLE12_EXPERT_PAYLOAD
            __syncwarp();
            if (lane == 0) __ppu_threadfence_system_uncache();
#endif
            continue;
        }
        const int owner=expert/FCONST_E;
        const int local=expert-owner*FCONST_E;
        const int64_t delta=p.peer[owner];
        const char* src=ptr<const char>(p,XP)+(int64_t)token*hb;
#ifdef FROLE12_DISPATCH_PREFETCH
        constexpr int ahead = FROLE12_DISPATCH_PREFETCH;
#else
        constexpr int ahead = 5;
#endif
        static_assert(ahead >= 0 && ahead * 512 <= hb);
        uint4 payload[ahead > 0 ? ahead : 1];
        if (direct) {
            // The local and peer tile formats have the same wire bytes.
            #pragma unroll
            for (int tile = 0; tile < ahead; ++tile)
#ifdef FROLE12_DIRECT_METADATA_PREFETCH
                payload[tile] = one_slot ? first_payload[tile] : tile_load(src + tile * 512);
#else
                payload[tile] = tile_load(src + tile * 512);
#endif
        }
        int slot_in_expert;
#ifdef FROLE12_DIRECT_ROUTES
        if (direct) {
            int reserved = 0;
            if (lane == 0) {
#ifdef FROLE12_DIRECT_METADATA
                reserved = one_slot ? first_slot : ptr<const int>(p, BUCKET)[j];
#else
                if constexpr (!precleared_dispatch) wait_peer_reset(p, owner);
                auto* peer_counts = reinterpret_cast<int*>(sym + delta + p.i[CNT_OFF]);
                reserved = grouped_reserve(peer_counts + local, 1);
                if (reserved < 0 || reserved >= p.i[CAP_E]) __builtin_trap();
#endif
            }
            slot_in_expert = __shfl_sync(0xffffffffu, reserved, 0);
        } else
#endif
        {
            slot_in_expert = bases[expert] + ptr<const int>(p, BUCKET)[j];
        }
        const int row = routed_input_row(local, slot_in_expert, p.i[CAP_E]);
        char* dst=sym+delta+p.i[POOL_OFF]+(int64_t)row*hb;
        if (direct) {
            if (owner == p.i[MY]) {
                #pragma unroll
                for (int tile = 0; tile < ahead; ++tile)
                    __ppu_global_stcg_bulk_b32x4(dst + tile * 512, payload[tile]);
                for (int c = ahead * 32 + lane; c < hb / 16; c += 32)
                    store16(dst + (int64_t)c * 16, load16(src + (int64_t)c * 16));
            } else {
                #pragma unroll
                for (int tile = 0; tile < ahead; ++tile)
                    tile_store(dst + tile * 512, payload[tile]);
                for (int off = ahead * 512; off < hb; off += 512)
                    tile_store(dst + off, tile_load(src + off));
            }
        } else if (owner==p.i[MY]) {
#ifdef FROLE12_GROUPED_PUSH_PAIR
#pragma unroll
            for (int off = 0; off < hb; off += 512)
                __ppu_global_stcg_bulk_b32x4(dst + off, tile_load(src + off));
#else
            for (int c=lane; c<hb/16; c+=32)
                store16(dst+(int64_t)c*16,load16(src+(int64_t)c*16));
#endif
        } else {
#ifdef FROLE12_GROUPED_PUSH_PAIR
            // Overlap two independent local loads before issuing peer stores.
#pragma unroll
            for (int off = 0; off + 512 < hb; off += 1024) {
                const uint4 first = tile_load(src + off);
                const uint4 second = tile_load(src + off + 512);
                tile_store(dst + off, first);
                tile_store(dst + off + 512, second);
            }
            if constexpr (hb % 1024)
                tile_store(dst + hb - 512, tile_load(src + hb - 512));
#else
            for (int off=0; off<hb; off+=512) tile_store(dst+off,tile_load(src+off));
#endif
        }
        store_dispatch_record(p, sym + delta, owner, local, row, token, slot, dispatch_scale_words(p, token));
#ifdef FROLE12_EXPERT_PAYLOAD
        __syncwarp();
        if (lane == 0) {
            __ppu_threadfence_system_uncache();
            auto* received = reinterpret_cast<volatile int*>(ptr<char>(p, POOL) + delta);
            add_sys(received + local, 1);
        }
#endif
    }
    __shared__ int publish;
#ifndef FROLE12_EXPERT_PAYLOAD
    __syncwarp();
    if (lane==0) __ppu_threadfence_system_uncache();
#endif
    __syncthreads();
    if (tid == 0) {
        // Direct senders can fill precleared receive rows before CTA0
        // initializes this request. The local completion ticket must wait.
        if (precleared_dispatch && direct && blockIdx.x != 0) {
            while (cnt_acquire(ptr<int>(p, SCRATCH)) < p.i[GENERATION])
                __nanosleep(FROLE12_POLL_NS);
        }
        publish = atomicAdd(ptr<int>(p, DONE), 1) == send_ctas - 1;
    }
    __syncthreads();
    if (publish && tid<FCONST_RANKS) {
#ifdef FROLE12_DIRECT_ROUTES
        // Empty destinations still receive completion after their reset.
        if (!precleared_dispatch && direct) wait_peer_reset(p, tid);
#endif
        publish_dispatch(p, tid, false);
    }
}
__device__ __forceinline__ void grouped_finalize(const Params& p) {
    if (blockIdx.x!=0) return;
    const int tid=threadIdx.x;
    const int generation = precleared_dispatch ? p.start_generation / 2 : 1;
    auto* dispatch_flags = reinterpret_cast<volatile int*>(ptr<char>(p, SYM) + p.i[GEN_OFF]);
#ifdef FROLE12_EARLY_METADATA
    constexpr int first_phase = FCONST_RANKS;
#else
    constexpr int first_phase = 0;
#endif
    if (tid<FCONST_RANKS) {
        while (acquire_sys(dispatch_flags + first_phase + tid) < generation)
            __nanosleep(FROLE12_POLL_NS);
        __threadfence();
    }
    __syncthreads();
    // Preserve expert order in both prefix arrays. The final scan warp
    // contributes zeros in lanes past the last local expert.
    constexpr int scan_warps = (FCONST_E + 31) / 32;
    static_assert(scan_warps == 1 || scan_warps == 2, "CTA0 supports up to 64 experts");
    const int warp = tid >> 5, lane = tid & 31;
    __shared__ int row_total[scan_warps], group_total[scan_warps];
    __shared__ int row_max[scan_warps];
    const int count = tid < FCONST_E ? ptr<const int>(p, COUNTS)[tid] : 0;
    if (tid < FCONST_E && p.p[STATS] && count)
        atomicAdd(ptr<int>(p, STATS) + tid, count);
    const int groups = (count + FTILE_ROWS - 1) / FTILE_ROWS;
    int rows = count, tiles = groups;
    if (warp < scan_warps) {
        const int maximum = __reduce_max_sync(0xffffffffu, count);
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int earlier_rows = __shfl_up_sync(0xffffffffu, rows, offset);
            const int earlier_tiles = __shfl_up_sync(0xffffffffu, tiles, offset);
            if (lane >= offset) {
                rows += earlier_rows;
                tiles += earlier_tiles;
            }
        }
        if (lane == 31) {
            row_total[warp] = rows;
            group_total[warp] = tiles;
            row_max[warp] = maximum;
        }
    }
    __syncthreads();
    int maximum = row_max[0], total = row_total[0], length = group_total[0];
    if constexpr (scan_warps == 2) {
        maximum = max(maximum, row_max[1]);
        total += row_total[1];
        length += group_total[1];
    }
#ifdef FROLE12_SHARED_EXPERT
    const int shared_rows = p.i[T];
    const int shared_groups = (shared_rows + FTILE_ROWS - 1) / FTILE_ROWS;
    maximum = max(maximum, shared_rows);
#else
    constexpr int shared_groups = 0;
#endif
    const int stride = (maximum + FTILE_ROWS - 1) / FTILE_ROWS;
    if (tid < FCONST_E) {
        const int row_base = rows - count + (warp ? row_total[0] : 0);
        const int group_base = shared_groups + tiles - groups + (warp ? group_total[0] : 0);
        ptr<int>(p, STARTS)[tid] = row_base;
        for (int g = 0; g < groups; ++g) {
#ifdef FROLE12_READY_FC2
            static_assert(FCONST_E <= 1 << group_expert_bits);
            ptr<int>(p, UNUSED)[group_base + g] = (g << group_expert_bits) | tid;
#ifdef FROLE12_W1_GROUP_RECORDS
            // Grouped dispatch leaves COMPACT_T free. Its FCONST_E * CAP_E
            // entries hold four words per group since each group has a row.
            static_assert(FCONST_E >= 4);
            const int row = g * FTILE_ROWS;
            ptr<uint4>(p, COMPACT_T)[group_base + g] = uint4{
                unsigned((g << group_expert_bits) | tid),
                unsigned(row_base + row),
                unsigned(routed_input_row(tid, row, p.i[CAP_E])),
                unsigned(min(count - row, FTILE_ROWS)),
            };
#endif
            // FC2 uses compact source storage for the ready queue. Zero is
            // reserved for entries that have yet to be published this request.
            ptr<int>(p, COMPACT_SRC)[group_base + g] = 0;
#else
            ptr<int>(p, UNUSED)[group_base + g] = tid * stride + g;
#endif
        }
    }
#ifdef FROLE12_SHARED_EXPERT
    for (int g = tid; g < shared_groups; g += blockDim.x) {
#ifdef FROLE12_READY_FC2
        const int code = (g << group_expert_bits) | FCONST_E;
        ptr<int>(p, COMPACT_SRC)[g] = 0;
#else
        const int code = FCONST_E * stride + g;
#endif
        ptr<int>(p, UNUSED)[g] = code;
    }
#endif
    if (tid == 0) {
        ptr<int>(p, RN)[0] = total
#ifdef FROLE12_SHARED_EXPERT
            + shared_rows
#endif
            ;
#if defined(FROLE12_READY_FC2) || defined(FROLE12_EARLY_METADATA)
        ptr<int>(p, RN)[1] = max(1, stride);
#else
        ptr<int>(p, RN)[1] = max(1, (maximum + 63) / 64);
#endif
        ptr<int>(p, RN)[2] = length + shared_groups;
    }
    if constexpr (precleared_dispatch)
        if (tid < FCONST_E) ptr<int>(p, COMPACT_SLOT)[tid] = count;
#ifdef FROLE12_EARLY_METADATA
    // Reservations are complete. B prefetch can use exact row and group
    // metadata while senders finish writing the activations and records.
    drain_gpu();
    if (tid == 0) cnt_release(ptr<int>(p, SCRATCH) + 1, 1);
    if (tid < FCONST_RANKS) {
        while (acquire_sys(dispatch_flags + tid) < generation)
            __nanosleep(FROLE12_POLL_NS);
        __threadfence();
    }
    __syncthreads();
#endif
    if constexpr (precleared_dispatch) {
        // The next request can send here after acquiring this request's
        // combine publication, which follows CTA0's exit fence.
        if (tid < FCONST_E) ptr<volatile int>(p, COUNTS)[tid] = 0;
    }
    const int reset_span = (FCONST_E
#ifdef FROLE12_SHARED_EXPERT
        + 1
#endif
        ) * stride;
    // Self arrival proves all local senders stopped using W1 source scratch.
    for (int j=tid;j<reset_span;j+=blockDim.x) ptr<int>(p,W1_CNT)[j]=0;
    drain_gpu();
#ifdef FROLE12_EARLY_METADATA
    if (tid == 0) cnt_release(ptr<int>(p, SCRATCH) + 4, 1);
#else
    if (tid==0) cnt_release(ptr<int>(p,SCRATCH)+1,1);
#endif
}


#ifdef FROLE12_COMBINE
template <int Threads = FTHREADS>
__device__ __forceinline__ void finish_combined_output(const Params& p) {
    constexpr int chunks = FCONST_H / 256, row_bytes = FCONST_H * 2;
    constexpr int warps = Threads / 32;
    const int tasks = p.i[T] * chunks;
    // CTA0 acquires all publications even on ranks with no output rows.
    if (blockIdx.x * warps >= tasks && blockIdx.x != 0) return;
    char* sym = ptr<char>(p, SYM);
    auto* ready = reinterpret_cast<volatile int*>(sym + p.cgen_off);
    // The workspace advances the barrier generation by two per request,
    // including when a request selects a different tile or replaces a pipeline.
    const int generation = precleared_dispatch ? p.start_generation / 2 : 1;
    const int topk = FCONST_TOPK;
    auto reduce = [&](int task) {
        const int token = task / chunks;
        const int offset = (task % chunks) * 512;
        const char* src = sym + p.cbuf_off + size_t(token) * topk * row_bytes + offset;
        float acc[8];
#ifdef FROLE12_SHARED_EXPERT
        const char* bias = sym + p.cbuf_off
            + size_t(p.i[CAP] + token) * row_bytes + offset;
        sum_combine_slots<FROLE12_COMBINE, true>(src, row_bytes, topk, acc, bias);
#else
        sum_combine_slots<FROLE12_COMBINE>(src, row_bytes, topk, acc);
#endif
        const uint4 words{sm100_bf16_pair(acc[0], acc[4]), sm100_bf16_pair(acc[1], acc[5]),
                          sm100_bf16_pair(acc[2], acc[6]), sm100_bf16_pair(acc[3], acc[7])};
        auto* dst = p.combined_output + size_t(token) * FCONST_H + offset / 2;
        __ppu_global_stcg_bulk_b32x4(dst, words);
    };
    const int stride = gridDim.x * warps;
    int first = blockIdx.x * warps + threadIdx.x / 32;
#ifdef FROLE12_INLINE_COMBINE
    const int lane = threadIdx.x & 31;
    const unsigned local = 1u << p.i[MY];
    bool arrived = false;
    int prev_token = -1;
    for (int task = first; task < tasks; task += stride) {
        const int token = task / chunks;
        if (token != prev_token) {
            unsigned needed = 0;
            if (lane < topk) {
                const int expert = route_id(p, token * topk + lane);
                // Invalid routes have a zero slot written by the source.
                needed = expert < 0 ? local : 1u << (expert / FCONST_E);
            }
#pragma unroll
            for (int delta = 1; delta < FCONST_TOPK; delta <<= 1)
                needed |= __shfl_xor_sync(0xffffffffu, needed, delta);
            needed = __shfl_sync(0xffffffffu, needed, 0);
#ifdef FROLE12_SHARED_EXPERT
            needed |= local;
#endif
            if (!arrived && lane < FCONST_RANKS && (needed & (1u << lane))) {
                while (acquire_sys(ready + lane) < generation)
                    __nanosleep(FROLE12_POLL_NS);
                __threadfence();
                arrived = true;
            }
            __syncwarp();
            prev_token = token;
        }
        reduce(task);
    }
    // Preserve the cross-request rendezvous on one warp even for empty ranks.
    if (blockIdx.x == 0 && threadIdx.x < FCONST_RANKS && !arrived) {
        while (acquire_sys(ready + lane) < generation)
            __nanosleep(FROLE12_POLL_NS);
        __threadfence();
    }
#else
    const unsigned all_ranks = (1u << FCONST_RANKS) - 1u;
    unsigned acquired = 0;
#ifdef FROLE12_COMBINE_EARLY
    __shared__ unsigned ready_ranks;
    const int lane = threadIdx.x & 31;
    if (threadIdx.x < 32) {
        bool arrived = false;
        if (lane < FCONST_RANKS) {
            arrived = acquire_sys(ready + lane) >= generation;
            if (arrived) __threadfence();
        }
        const unsigned mask = __ballot_sync(0xffffffffu, arrived);
        if (lane == 0) ready_ranks = mask;
    }
    __syncthreads();
    acquired = ready_ranks;
    // If this chunk's producers have all published, finish it while other
    // ranks or CTAs are still producing unrelated contributions.
    const bool local_ready =
#ifdef FROLE12_SHARED_EXPERT
        acquired & (1u << p.i[MY]);
#else
        true;
#endif
    if (local_ready && acquired && acquired != all_ranks && topk <= 32 && first < tasks) {
        const int token = first / chunks;
        const int expert = lane < topk ? route_id(p, token * topk + lane) : -1;
        const bool missing = expert >= 0 && !(acquired & (1u << (expert / FCONST_E)));
        if (__ballot_sync(0xffffffffu, missing) == 0) {
            reduce(first);
            first += stride;
        }
    }
#endif
    if (acquired != all_ranks) {
        if (threadIdx.x < FCONST_RANKS && !(acquired & (1u << threadIdx.x))) {
            while (acquire_sys(ready + threadIdx.x) < generation) __nanosleep(FROLE12_POLL_NS);
            __threadfence();
        }
        __syncthreads();
    }
    for (int task = first; task < tasks; task += stride)
        reduce(task);
#endif
}
#endif
} // namespace role12

extern "C" __global__ void __launch_bounds__(FTHREADS)
gc_moe_role12_kernel(role12::Params p) {
    using namespace role12;
    int dispatch_ctas = gridDim.x;
#if defined(FROLE12_SHARED_EXPERT) && defined(FROLE12_READY_FC2)
    // Leave the last CTAs free to run the local FC1 while the other CTAs
    // reserve and fill routed slots. They still join the same FC2 queue.
    const int shared_groups = (p.i[T] + FTILE_ROWS - 1) / FTILE_ROWS;
    const int shared_fc1_ctas = early_shared_fc1_tasks(shared_groups, FCONST_I / FBCOLS, gridDim.x);
    if (!direct_dispatch(p) && p.i[T] > FTILE_ROWS &&
        dispatch_ctas - shared_fc1_ctas > p.i[ROLE1_CTAS])
        dispatch_ctas -= shared_fc1_ctas;
#endif
    const bool dispatch_first = precleared_dispatch && direct_dispatch(p) && blockIdx.x != 0;
    if (!dispatch_first) initialize(p);
    if (blockIdx.x < dispatch_ctas) grouped_dispatch(p, dispatch_ctas);
    if (dispatch_first) initialize(p);
    if (blockIdx.x < p.i[ROLE1_CTAS]) grouped_finalize(p);
    // Role1 workers join math after their expert-copy queue is exhausted.
    // The prefetch path acquires metadata after issuing its first W1 copies.
    gc_moe_fused_v12_kernel(
        ptr<const uint8_t>(p, SYM) + p.i[POOL_OFF],
        ptr<const uint8_t>(p, SYM) + p.i[SF_OFF],
        ptr<const uint8_t>(p, W1), ptr<const uint8_t>(p, SFW1),
        ptr<const uint8_t>(p, W2), ptr<const uint8_t>(p, SFW2),
        ptr<const int>(p, STARTS), ptr<const int>(p, X_STARTS),
        ptr<const int>(p, precleared_dispatch ? COMPACT_SLOT : COUNTS), ptr<const int>(p, UNUSED),
        ptr<const int>(p, BUCKET), ptr<const float>(p, REC_W),
        ptr<uint8_t>(p, ACT), ptr<uint8_t>(p, ACT_SF), ptr<float>(p, OUT),
        FCONST_E, FCONST_I, FCONST_H, FCONST_TOPK,
        ptr<const int>(p, RN) + 1, ptr<int>(p, W1_CNT), ptr<const int>(p, SCRATCH) + 4
        , p.i[T], p.i[CAP_E]
#ifdef FROLE12_SHARED_EXPERT
        , ptr<const uint8_t>(p, XP), ptr<const uint8_t>(p, SF)
        , ptr<char>(p, SYM) + p.cbuf_off + size_t(p.i[CAP]) * FCONST_H * 2
        , ptr<const uint8_t>(p, SE_W1), ptr<const uint8_t>(p, SE_SFW1)
        , ptr<const uint8_t>(p, SE_W2), ptr<const uint8_t>(p, SE_SFW2)
#endif
        , __int_as_float(p.i[CLAMP_BITS])
        , ptr<int>(p, SCRATCH) + 1, p.i[GENERATION]
        , reinterpret_cast<const uint4*>(ptr<const char>(p,SYM) + p.i[REC_OFF])
#ifdef FROLE12_READY_FC2
        // Reuse compact-source storage and two reset scratch words for FC2.
        , ptr<int>(p, COMPACT_SRC), ptr<int>(p, SCRATCH) + 5, ptr<int>(p, SCRATCH) + 6
#endif
#ifdef FROLE12_W1_GROUP_RECORDS
        , ptr<const uint4>(p, COMPACT_T)
#endif
        , ptr<char>(p,SYM), p.peer, p.cbuf_off
#ifdef FROLE12_EXPERT_PAYLOAD
        , ptr<const volatile int>(p, POOL)
#endif
        );
    __syncwarp();
    if ((threadIdx.x & 31) == 0) __ppu_threadfence_system_uncache();
    __syncthreads();
    // Warp 0 owns the grid ticket and the peer publications. The other
    // warps have already drained their stores and can enter combine.
    if (threadIdx.x < 32) {
        int last_writer = 0;
        if (threadIdx.x == 0)
            last_writer = atomicInc(reinterpret_cast<unsigned*>(ptr<int>(p, DONE) + 1),
                                    unsigned(gridDim.x - 1)) == unsigned(gridDim.x - 1);
        if (__shfl_sync(0xffffffffu, last_writer, 0)) {
#ifdef FROLE12_EXPERT_PAYLOAD
            auto* received = ptr<volatile int>(p, POOL);
            for (int e = threadIdx.x; e < FCONST_E; e += 32) received[e] = 0;
            __syncwarp();
            if (threadIdx.x == 0) __ppu_threadfence_system_uncache();
            __syncwarp();
#endif
            if (threadIdx.x < FCONST_RANKS) {
                char* peer = ptr<char>(p, SYM) + p.peer[threadIdx.x] + p.cgen_off;
                auto* ready = reinterpret_cast<volatile int*>(peer) + p.i[MY];
                if constexpr (precleared_dispatch)
                    store_sys(ready, p.start_generation / 2);
                else
                    add_sys(ready, 1);
            }
        }
    }
#if defined(FROLE12_COMBINE) && defined(FROLE12_INLINE_COMBINE)
    if (p.combined_output) finish_combined_output(p);
#endif
}
#ifdef FROLE12_COMBINE
#ifdef FROLE12_SHARED_EXPERT
constexpr int role3_threads = 128;
#else
constexpr int role3_threads = FTHREADS;
#endif
extern "C" __global__ void __launch_bounds__(role3_threads)
gc_moe_role3_kernel(role12::Params p) {
    role12::finish_combined_output<role3_threads>(p);
}
#endif
extern "C" int gc_moe_role12_weight_panel() {
#ifdef FROLE12_K_MAJOR_W2
    return 2;
#else
    return 1;
#endif
}
#ifdef FROLE12_SHARED_EXPERT
extern "C" int gc_moe_role12_shared_expert() { return 1; }
#endif
extern "C" int gc_moe_role12_grouped_remote() {
    return 1;
}
extern "C" int gc_moe_role12_occ() {
    int blocks = 0;
    hggcOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, gc_moe_role12_kernel, FTHREADS, 0);
    return blocks;
}
extern "C" int gc_moe_role12_group_tile() { return FTILE_ROWS; }
static void launch_role12(void** ptrs, const int* ints, const long long* offsets,
                          const int* comm, int start_generation,
                          uint16_t* combined_output = nullptr) {
    using namespace role12;
    Params p{};
    for (int j = 0; j < NPTR; ++j) p.p[j] = ptrs[j];
    for (int j = 0; j < NINT; ++j) p.i[j] = ints[j];
    for (int j = 0; j < 8; ++j) p.peer[j] = offsets[j];
    if (comm) {
        if (start_generation < 1) __builtin_trap();
        p.start_generation = start_generation;
        p.barr_off = comm[0];
        p.ccnt_off = comm[1];
        p.cgen_off = comm[2];
        p.cbuf_off = comm[3];
    }
#ifdef FROLE12_COMBINE
    p.combined_output = combined_output;
#endif
    if (p.i[H] != FCONST_H || p.i[I] != FCONST_I || p.i[E] != FCONST_E ||
        p.i[RANKS] != FCONST_RANKS || p.i[GENERATION] < 1 ||
        (precleared_dispatch && !comm) || p.i[TOPK] != FCONST_TOPK ||
        p.i[T] < 0 || p.i[T] > p.i[CAP] / FCONST_TOPK)
        __builtin_trap();
    static const int multiprocessors = [] {
        int dev = 0;
        hggcDeviceProp prop{};
        hggcGetDevice(&dev);
        hggcGetDeviceProperties(&prop, dev);
        return prop.multiProcessorCount;
    }();
    static const int capacity = gc_moe_role12_occ() * multiprocessors;
    const int grid = p.i[NCTA] < capacity ? p.i[NCTA] : capacity;
    if (grid < 1 || p.i[ROLE1_CTAS] < 1 || p.i[ROLE1_CTAS] > grid) __builtin_trap();
    if (p.i[NCOUNTERS] < 2 * FCONST_E * p.i[RANKS]) __builtin_trap();
#ifdef FROLE12_SHARED_EXPERT
    if (!combined_output) __builtin_trap();
#endif
    const auto stream = reinterpret_cast<hggcStream_t>(ptrs[STREAM]);
    gc_moe_role12_kernel<<<grid, FTHREADS, 0, stream>>>(p);
#if defined(FROLE12_COMBINE) && !defined(FROLE12_INLINE_COMBINE)
    if (p.combined_output) {
        constexpr int warps = role3_threads / 32;
        // Role3 uses fewer registers and no GEMM staging buffers. Bound its
        // polling fanout independently of the number of resident Role12 CTAs.
        static const int combine_capacity = [] {
            int blocks = 0;
            hggcOccupancyMaxActiveBlocksPerMultiprocessor(
                &blocks, gc_moe_role3_kernel, role3_threads, 0);
            return min(blocks, 8) * multiprocessors;
        }();
        const int chunks = p.i[T] * (FCONST_H / 256);
        // An empty rank still needs one CTA to acquire the peer completions.
        const int combine_grid = min(combine_capacity, max(1, (chunks + warps - 1) / warps));
        gc_moe_role3_kernel<<<combine_grid, role3_threads, 0, stream>>>(p);
    }
#endif
}
extern "C" void gc_moe_role12(void** ptrs, const int* ints, const long long* offsets) {
    launch_role12(ptrs, ints, offsets
                  , nullptr, 0
                  );
}
extern "C" void gc_moe_role12_with_start_reset(void** ptrs, const int* ints,
                                                const long long* offsets,
                                                const int* comm, int generation) {
    launch_role12(ptrs, ints, offsets, comm, generation);
}

#ifdef FROLE12_PRECLEARED_DISPATCH
extern "C" int gc_moe_role12_precleared_dispatch() { return 1; }
#ifdef FROLE12_COMBINE
extern "C" void gc_moe_role12_with_combine(
    void** ptrs, const int* ints, const long long* offsets, const int* comm, int generation,
    void* output) {
    if (!output || (reinterpret_cast<uintptr_t>(output) & 511)) __builtin_trap();
    launch_role12(ptrs, ints, offsets, comm, generation, static_cast<uint16_t*>(output));
}
#endif
#endif

extern "C" int gc_moe_role12_row_wire() { return 1; }

extern "C" int gc_moe_role12_init_gate_table()
{
    sm100_fill_gate_table<<<156, 256>>>();
    const auto status = hggcGetLastError();
    return status ? int(status) : int(hggcDeviceSynchronize());
}

extern "C" int gc_moe_role12_direct_combine() { return 1; }
