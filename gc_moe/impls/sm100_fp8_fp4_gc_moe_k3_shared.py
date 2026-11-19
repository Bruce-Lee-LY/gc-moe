"""K3 routed SiTU与BF16 shared expert的CuTe设备计算。

两条流水线共用CTA和TMEM分配，分别维护输入及accumulator缓冲。
"""

from __future__ import annotations

from cutlass import Float32, Int32, Uint16, Uint32, Uint64, Uint8
import cutlass
import cutlass.cute as cute

from ..common.math import (
    abs_f32,
    bits_float,
    f32_to_bf16_bits,
    f32x2_to_bf16x2_bits,
    fast_exp_neg_f32,
    fast_rcp,
    fmax_f32,
    pack_bf16_pair,
    ue8m0_fp8_inv,
    ue8m0_fp8_scale_exp,
    warp_group4_max,
)
from ..common.tma_copy import load_mixed_k128_tma
from ..layout.gc_moe import SIGNAL_OFFSETS
from ..layout.sym_buffer import mapped_peer_base
from ..mma.sm100 import (
    blockscaled_instr_desc_hi,
    issue_mixed_k128,
    kmajor_smem_desc,
    kmajor_smem_desc_from_start,
)
from ..ptx.ld_st import (
    EVICT_FIRST,
    EVICT_NORMAL,
    ld_acquire_gpu_u32,
    ld_acquire_gpu_u64,
    ld_volatile_u32,
    red_gpu_add_u32,
    red_release_gpu_xor_u64,
    st_global_u32x4,
    st_global_u64,
    stmatrix_fp8x4_trans_late,
)
from ..ptx.tcgen05 import (
    tcgen05_after_thread_sync,
    tcgen05_before_thread_sync,
    tcgen05_commit_2cta,
    tmem_free_2cta,
    tmem_load_16dp256b4,
)
from ..ptx.tma import (
    generic_tensormap_ptr,
    prefetch_tensormap,
    tile_g2s_2cta_2d,
    tile_s2g_2d,
    tma_store_arrive,
    tma_store_fence,
    tma_store_wait_0,
    tma_store_wait_1,
)
from ..ptx.k3 import (
    _mma_bf16_shared_2cta,
    st_async_release_gpu_u64,
    fence_proxy_async_global,
    tensormap_acquire_gpu,
    tanh_approx_f32,
    transpose_w2_bf16x8,
)
from ..ptx.utils import (
    assume_u32_lt,
    device_trap,
    elect_one_sync,
    read_clock64,
    sync_aligned,
    sync_unaligned,
    warpgroup_reg_alloc,
    warpgroup_reg_dealloc,
)
from ..scheduler.gc_moe import GCBlockPhase, release_task_info, schedule_tasks
from .sm100_fp8_fp4_gc_moe import (
    GCMoERoleKernel,
    _load_w1_atom,
    _load_w2_atom,
    _load_w2_source_metadata,
    _release_w1_tmem,
    _release_w2_tmem,
    _w1_pack_fp8x4,
    sync_epilogue_warpgroup,
)


@cute.jit
def _k3_situ(gate_acc: Float32, up_acc: Float32) -> Float32:
    # FC1 accumulator先按BF16 RN舍入，再计算SiTU。
    gate = bits_float(Uint32(f32_to_bf16_bits(gate_acc)) << 16)
    up = bits_float(Uint32(f32_to_bf16_bits(up_acc)) << 16)
    sigmoid = fast_rcp(Float32(1) + fast_exp_neg_f32(gate))
    gate_out = Float32(4) * tanh_approx_f32(gate * Float32(0.25)) * sigmoid
    up_out = Float32(25) * tanh_approx_f32(up * Float32(0.04))
    return gate_out * up_out


@cute.jit
def _k3_routed_situ_pair(gate0: Float32, gate1: Float32,
                        up0: Float32, up1: Float32,
                        weight0: Float32, weight1: Float32):
    # FC1先做BF16 RN；SiTU结果与路由权重相乘，返回两个FP32值。
    value0 = _k3_situ(gate0, up0)
    value1 = _k3_situ(gate1, up1)
    return cute.arch.mul_packed_f32x2(
        (value0, value1), (weight0, weight1), rnd='rn', ftz=False)


class K3GCMoERoleKernel(GCMoERoleKernel):
    """在同一CTA内并行执行routed和BF16 shared流水线，统一分配SMEM与TMEM。"""

    def __init__(self, *args, k3_layout, shared_rows=256, **kwargs):
        from dataclasses import replace
        from ..layout.gc_moe import get_role_shared_layout
        cfg = kwargs['role_config']
        if cfg.role != 2 or cfg.block_m != 32:
            raise ValueError('K3 shared requires compute role and routed M32')
        if not 1 <= shared_rows <= k3_layout.tokens:
            raise ValueError('shared input rows exceed workspace capacity')
        token_tile_rows = min(256, (int(shared_rows) + 15) // 16 * 16)
        cta_rows = token_tile_rows // 2
        # 两CTA各搬运半个token tile；每CTA不超过48行时可保留routed十级缓冲。
        routed_stages = 10 if cta_rows <= 48 else 9
        route_layout = get_role_shared_layout(
            2, 32, 896, cfg.pull_bytes, routed_stages,
            store_m_l1=cfg.store_m_l1, store_m_l2=8,
            dispatch_warps=4, block_n=128, block_k=128, sf_block_m=128)
        kwargs['role_config'] = replace(
            cfg, num_stages=routed_stages, store_m_l2=8, shared_layout=route_layout,
            dynamic_smem=route_layout.field_end, threads=640)
        super().__init__(*args, **kwargs)
        if (self.role, self.hidden, self.intermediate, self.num_tokens,
            self.num_ranks, self.local_experts, self.topk, self.shared_experts) != (
                2, 3584, 3072, 128, 8, 112, 16, 0):
            raise ValueError('K3 routed/shared requires routed EP8/T128/H3584/I3072/E896/top16 without legacy shared slots')
        self.k3_layout = k3_layout
        self.shared_rows = int(shared_rows)
        self.shared_mma_m = 128
        self.shared_mma_n = token_tile_rows
        self.shared_cta_rows = cta_rows
        self.shared_token_tiles = (self.shared_rows + token_tile_rows - 1) // token_tile_rows
        # M128的两个token半区映射到DP0..63/64..127，每个accumulator占半tile列数。
        self.shared_accum_cols = cta_rows
        # CTA初始分配96×640=61440个32位寄存器，各warpgroup在该预算内重分配。
        # routed warp 0～3：TMA、MMA 和调度；
        self.routed_non_epi_regs_per_thread = 64
        # routed warp 4～11：epilogue。
        self.routed_epi_regs_per_thread = 160
        # shared warp 12～15：epilogue。
        self.shared_epi_regs_per_thread = 48
        # shared warp16/17负责TMA，warp18负责MMA；warp19只参加最终汇合。
        self.shared_non_epi_regs_per_thread = 48
        # 整个 CTA 的总预算，单位为 32 位寄存器个数。
        self.cta_register_budget = 61440
        self._setup_shared_math((route_layout.field_end + 1023) // 1024 * 1024)

        self.shared_layout = replace(route_layout, field_end=self.dynamic_smem)
        # routed用TMEM 0..127列；shared从128列开始，每槽宽度由shared_accum_cols决定。
        # 每槽最多128列；加上routed后最多用到第383列。
        # TMEM按2的幂分配，共申请512列；全部使用者退出后才能释放。
        self.tmem_cols = 512

    @cute.kernel
    def kernel(self, signals: cute.Tensor, peer_ptrs: cute.Tensor,
               rank_idx: Uint32, num_tokens: Uint32,
               tensor_maps=None, y_out=None, stats=None,
               activation_clamp=None, activation_alpha=None,
               tensor_map_1=None, tensor_map_2=None, tensor_map_3=None,
               tensor_map_4=None, tensor_map_5=None, tensor_map_6=None,
               tensor_map_7=None, tensor_map_8=None, tensor_map_9=None,
               tensor_map_10=None, tensor_map_11=None, tensor_map_12=None,
               tensor_map_13=None, tensor_map_14=None, tensor_map_15=None,
               tensor_map_16=None, tensor_map_17=None,
               shared_state=None, shared_maps=None):
        # 每CTA20个warp，routed/shared的资源和同步在此统一管理：
        # 0/1 routed A/B TMA；2 leader MMA；3 leader scheduler；4..11 routed epilogue。
        # 12..15 shared epilogue；16激活TMA、17权重TMA；18 leader shared MMA；19空闲。
        # 全部640线程及两CTA汇合后释放TMEM。
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        # 架构索引在入口统一为无符号 32 位，后续 worker 直接传递。
        block = Uint32(block)
        physical_warp = Uint32(cute.arch.make_warp_uniform(cute.arch.warp_idx()))
        lane = Uint32(tid % 32)
        if physical_warp == 0:
            # 18个TensorMap分别作为128B对齐的值参数传入。
            prefetch_tensormap(tensor_maps.iterator)
            prefetch_tensormap(tensor_map_1.iterator)
            prefetch_tensormap(tensor_map_2.iterator)
            prefetch_tensormap(tensor_map_3.iterator)
            prefetch_tensormap(tensor_map_4.iterator)
            prefetch_tensormap(tensor_map_5.iterator)
            prefetch_tensormap(tensor_map_6.iterator)
            prefetch_tensormap(tensor_map_7.iterator)
            prefetch_tensormap(tensor_map_8.iterator)
            prefetch_tensormap(tensor_map_9.iterator)
            prefetch_tensormap(tensor_map_10.iterator)
            prefetch_tensormap(tensor_map_11.iterator)
            prefetch_tensormap(tensor_map_12.iterator)
            prefetch_tensormap(tensor_map_13.iterator)
            prefetch_tensormap(tensor_map_14.iterator)
            prefetch_tensormap(tensor_map_15.iterator)
            prefetch_tensormap(tensor_map_16.iterator)
            prefetch_tensormap(tensor_map_17.iterator)
        smem = cutlass.utils.SmemAllocator()
        slab = cute.recast_ptr(
            smem.allocate(self.shared_layout.field_end, 1024), dtype=Uint8)
        full = cute.recast_ptr(slab + self.shared_layout.task_full, dtype=Uint64)
        empty = cute.recast_ptr(slab + self.shared_layout.task_empty, dtype=Uint64)
        task = cute.recast_ptr(slab + self.shared_layout.tasks, dtype=Uint32)
        # CD是W1的双缓冲输出暂存区。
        cd_smem = slab + self.shared_layout.cd
        data_a = slab + self.shared_layout.data_a
        data_b = slab + self.shared_layout.data_b
        data_sfa = cute.recast_ptr(slab + self.shared_layout.data_sfa, dtype=Uint32)
        data_sfb = cute.recast_ptr(slab + self.shared_layout.data_sfb, dtype=Uint32)
        epi_amax = cute.recast_ptr(slab + self.shared_layout.amax, dtype=Float32)
        data_full = cute.recast_ptr(slab + self.shared_layout.data_full, dtype=Uint64)
        data_empty = cute.recast_ptr(slab + self.shared_layout.data_empty, dtype=Uint64)
        accum_full = cute.recast_ptr(slab + self.shared_layout.accum_full, dtype=Uint64)
        accum_empty = cute.recast_ptr(slab + self.shared_layout.accum_empty, dtype=Uint64)
        tptr = cute.recast_ptr(slab + self.shared_layout.tmem_ptr, dtype=Uint32)
        cta_rank, _, _ = cute.arch.block_in_cluster_idx()
        self._initialize_shared_math(slab, physical_warp, lane)
        self._initialize_shared_storage(slab, physical_warp, lane)
        # setmaxnreg按warpgroup一致执行，为routed epilogue分配较多寄存器。
        if physical_warp < 4:
            warpgroup_reg_dealloc(self.routed_non_epi_regs_per_thread)
        elif physical_warp < 12:
            warpgroup_reg_alloc(self.routed_epi_regs_per_thread)
        elif physical_warp < 16:
            warpgroup_reg_dealloc(self.shared_epi_regs_per_thread)
        else:
            warpgroup_reg_dealloc(self.shared_non_epi_regs_per_thread)
        iteration = Uint32(0)
        # recv_parity选择本次调用的接收计数槽，两代计数交替使用。
        gen = ld_volatile_u32(
            signals.iterator + SIGNAL_OFFSETS['recv_parity'] // 4) & 1
        recv_ptr = signals.iterator + SIGNAL_OFFSETS['expert_recv_count_sum'] // 4 + \
            gen * (self.local_experts * 2)
        l1_count_ptr = signals.iterator + SIGNAL_OFFSETS['l1_task_count'] // 4
        l2_count_ptr = signals.iterator + SIGNAL_OFFSETS['l2_task_count'] // 4
        shared_l1_count_ptr = signals.iterator + SIGNAL_OFFSETS['shared_l1_task_count'] // 4
        shared_l2_count_ptr = signals.iterator + SIGNAL_OFFSETS['shared_l2_task_count'] // 4
        expert_counts = cute.make_rmem_tensor(
            ((self.local_experts + 31) // 32,), Uint32)
        if cta_rank == 0 and physical_warp == 3:
            schedule_tasks(
                task, full, empty, recv_ptr, l1_count_ptr, l2_count_ptr,
                shared_l1_count_ptr, shared_l2_count_ptr, expert_counts,
                iteration, Int32(lane), Int32(block), num_tokens,
                self.block_m,
                self.compute_sms,
                self.dispatch_sms,
                self.hidden,
                self.intermediate,
                self.l1_clusters,
                self.l2_clusters,
                self.local_experts,
                self.num_ranks,
                self.shared_experts,
            )
        # 两个TMA warp和leader MMA warp按相同顺序消费双缓冲任务槽。
        if physical_warp == 0:
            # 任务字段在读取边界显式转 Uint32，保证后续除法、移位保持无符号。
            task_words = cute.make_tensor(task, cute.make_layout((16,)))
            task_iter = Uint32(0)
            data_stage = Uint32(0)
            data_phase = Uint32(0)
            running = True
            while running:
                task_stage = task_iter % 2
                cute.arch.mbarrier_wait(full + task_stage,
                                        (task_iter // 2) % 2)
                task_phase = task_words[task_stage * 8]
                running = task_phase != GCBlockPhase.None_
                if running:
                    pool_block = Uint32(task_words[task_stage * 8 + 4])
                    valid_m = Uint32(task_words[task_stage * 8 + 5])
                    shape_k = Uint32(task_words[task_stage * 8 + 7])
                    n_cluster = Uint32(task_words[task_stage * 8 + 3])
                    local_expert = Uint32(task_words[task_stage * 8 + 1])
                    block_idx = pool_block % self.ring_blocks
                    activation_map = tensor_maps.iterator
                    activation_sf_map = tensor_map_1.iterator
                    weight_map = tensor_map_2.iterator
                    weight_sf_map = tensor_map_3.iterator
                    if task_phase == GCBlockPhase.Linear2:
                        activation_map = tensor_map_5.iterator
                        activation_sf_map = tensor_map_6.iterator
                        weight_map = tensor_map_7.iterator
                        weight_sf_map = tensor_map_8.iterator
                    # 整个warp先把选中的TensorMap参数转成generic地址，K循环内复用。
                    activation_map = generic_tensormap_ptr(activation_map)
                    activation_sf_map = generic_tensormap_ptr(activation_sf_map)
                    if task_phase == GCBlockPhase.Linear1:
                        l1_full_ptr = signals.iterator + \
                            SIGNAL_OFFSETS['l1_full_count'] // 4 + block_idx
                        required = self.block_m * \
                            (pool_block // self.ring_blocks + 1)
                        while ld_acquire_gpu_u32(l1_full_ptr) != required:
                            pass
                    # pending跨K128步缓存本任务尚未就绪的W1通道位。
                    # XOR按当前generation解释翻转标记，新任务重新初始化。
                    mask_ptr = signals.iterator + \
                        SIGNAL_OFFSETS['l2_full_mask'] // 4 + \
                        (pool_block % self.ring_blocks) * 2
                    generation = pool_block // self.ring_blocks
                    expected = Uint64(0) if (generation & 1) != 0 \
                        else Uint64(self.l1_full_mask)
                    pending = Uint64(self.l1_full_mask)
                    k_block = Uint32(0)
                    while k_block < shape_k // 128:
                        if task_phase == GCBlockPhase.Linear2:
                            required_mask = Uint64(3) << (k_block * 2)
                            while (pending & required_mask) != 0:
                                pending = ld_acquire_gpu_u64(mask_ptr) ^ expected
                        load_mixed_k128_tma(
                            activation_map, activation_sf_map,
                            weight_map, weight_sf_map,
                            data_a + data_stage * (self.block_m // 2 * 128),
                            data_sfa + data_stage * self.sf_block_m,
                            data_b + data_stage * (128 * 128),
                            data_sfb + data_stage * 128,
                            data_full + data_stage, data_empty + data_stage,
                            k_block, data_phase, block_idx,
                            n_cluster * 2 + cta_rank,
                            local_expert, valid_m, Uint32(self.block_m // 2), shape_k,
                            Uint32(0), cta_rank, 0,
                            self.block_m, self.sf_block_m,
                            self.shared_weight_gran_k,
                            False,
                            self.shared_weight_cache_hint,
                            self.routed_weight_linger)
                        data_stage = Uint32(0) if data_stage == \
                            (self.math_stages - 1) else data_stage + 1
                        data_phase ^= Uint32(data_stage == 0)
                        k_block += 1
                task_iter += 1
        elif physical_warp == 1:
            task_words = cute.make_tensor(task, cute.make_layout((16,)))
            task_iter = Uint32(0)
            data_stage = Uint32(0)
            data_phase = Uint32(0)
            running = True
            while running:
                task_stage = task_iter % 2
                cute.arch.mbarrier_wait(full + task_stage,
                                        (task_iter // 2) % 2)
                task_phase = task_words[task_stage * 8]
                running = task_phase != GCBlockPhase.None_
                if running:
                    pool_block = Uint32(task_words[task_stage * 8 + 4])
                    valid_m = Uint32(task_words[task_stage * 8 + 5])
                    shape_k = Uint32(task_words[task_stage * 8 + 7])
                    n_cluster = Uint32(task_words[task_stage * 8 + 3])
                    local_expert = Uint32(task_words[task_stage * 8 + 1])
                    block_idx = pool_block % self.ring_blocks
                    activation_map = tensor_maps.iterator
                    activation_sf_map = tensor_map_1.iterator
                    weight_map = tensor_map_2.iterator
                    weight_sf_map = tensor_map_3.iterator
                    if task_phase == GCBlockPhase.Linear2:
                        activation_map = tensor_map_5.iterator
                        activation_sf_map = tensor_map_6.iterator
                        weight_map = tensor_map_7.iterator
                        weight_sf_map = tensor_map_8.iterator
                    weight_map = generic_tensormap_ptr(weight_map)
                    weight_sf_map = generic_tensormap_ptr(weight_sf_map)
                    k_block = Uint32(0)
                    while k_block < shape_k // 128:
                        load_mixed_k128_tma(
                            activation_map, activation_sf_map,
                            weight_map, weight_sf_map,
                            data_a + data_stage * (self.block_m // 2 * 128),
                            data_sfa + data_stage * self.sf_block_m,
                            data_b + data_stage * (128 * 128),
                            data_sfb + data_stage * 128,
                            data_full + data_stage, data_empty + data_stage,
                            k_block, data_phase, block_idx,
                            n_cluster * 2 + cta_rank,
                            local_expert, valid_m, Uint32(0), shape_k,
                            Uint32(0), cta_rank, 1,
                            self.block_m, self.sf_block_m,
                            self.shared_weight_gran_k,
                            False,
                            self.shared_weight_cache_hint,
                            self.routed_weight_linger)
                        data_stage = Uint32(0) if data_stage == \
                            (self.math_stages - 1) else data_stage + 1
                        data_phase ^= Uint32(data_stage == 0)
                        k_block += 1
                task_iter += 1
        if cta_rank == 0 and physical_warp == 2:
            task_words = cute.make_tensor(task, cute.make_layout((16,)))
            # leader MMA warp在任务循环前建立weight/activation的K-major描述符基址。
            weight_desc_base = kmajor_smem_desc(
                data_b, Uint32(1024), True, Uint32(0))
            activation_desc_base = kmajor_smem_desc(
                data_a, Uint32(1024), True, Uint32(0))
            data_desc_hi = (weight_desc_base >> 32) << 32
            sf_desc_hi = kmajor_smem_desc_from_start(
                Uint32(0), Uint32(128), False)
            task_iter = Uint32(0)
            data_stage = Uint32(0)
            data_phase = Uint32(0)
            running = True
            while running:
                task_stage = task_iter % 2
                cute.arch.mbarrier_wait(full + task_stage,
                                        (task_iter // 2) % 2)
                task_phase = task_words[task_stage * 8]
                running = task_phase != GCBlockPhase.None_
                if running:
                    valid_m = Uint32(task_words[task_stage * 8 + 5])
                    shape_k = Uint32(task_words[task_stage * 8 + 7])
                    num_k_blocks = shape_k // 128
                    weight_format = Uint32(5)
                    aligned_n = (valid_m + 15) & ~15
                    instr_desc_base = blockscaled_instr_desc_hi(
                        aligned_n, weight_format, Uint32(0), Uint32(0))
                    accum_stage = task_iter % self.accum_stages
                    accum_phase = (task_iter // self.accum_stages) % 2
                    cute.arch.mbarrier_wait(accum_empty + accum_stage,
                                            accum_phase ^ 1)
                    tcgen05_after_thread_sync()
                    for k_iter in cutlass.range(
                            Int32(0), Int32(num_k_blocks),
                            Int32(1), unroll=4):
                        k_block = Uint32(k_iter)
                        cute.arch.mbarrier_wait(data_full + data_stage,
                                                data_phase)
                        # 描述符布局位固定，只推进以16B为单位的SMEM地址字段。
                        weight_desc_start = (Uint32(weight_desc_base) + data_stage * 1024)
                        activation_desc_start = (Uint32(activation_desc_base) + data_stage *
                            ((self.block_m // 2 * 128) // 16))
                        issue_mixed_k128(
                            weight_desc_start, activation_desc_start,
                            data_sfb, data_sfa, data_stage,
                            data_desc_hi, sf_desc_hi,
                            accum_stage * self.block_m,
                            Uint32(self.accum_stages * self.block_m + self.sfa_tmem_cols),
                            Uint32(self.accum_stages * self.block_m),
                            data_empty + data_stage, accum_full + accum_stage,
                            instr_desc_base, k_block,
                            Uint32(k_block == num_k_blocks - 1),
                            Uint32(0), self.sf_block_m,
                            self.shared_weight_gran_k,
                            False, self.use_small_tile_path)
                        data_stage = Uint32(0) if data_stage == \
                            (self.math_stages - 1) else data_stage + 1
                        data_phase ^= Uint32(data_stage == 0)
                task_iter += 1
            # None终止任务也增加task_iter；退出前等待最后一个有效accumulator槽被消费完。
            if task_iter > 1:
                last_valid_iter = task_iter - 2
                last_stage = last_valid_iter % self.accum_stages
                last_phase = (last_valid_iter // self.accum_stages) % 2
                cute.arch.mbarrier_wait(accum_empty + last_stage,
                                        last_phase)
        if physical_warp >= 4 and physical_warp < 12:
            if tptr[0] != 0:
                device_trap()
            task_words = cute.make_tensor(task, cute.make_layout((16,)))
            iteration = Uint32(0)
            running = True
            while running:
                stage = iteration % 2
                generation_phase = (iteration // 2) % 2
                cute.arch.mbarrier_wait(full + stage, generation_phase)
                task_phase = task_words[stage * 8]
                n_cluster = Uint32(task_words[stage * 8 + 3])
                pool_block = Uint32(task_words[stage * 8 + 4])
                valid_m = Uint32(task_words[stage * 8 + 5])
                if task_phase != GCBlockPhase.None_:
                    accum_stage = iteration % self.accum_stages
                    accum_phase = (iteration // self.accum_stages) % 2
                    cute.arch.mbarrier_wait(accum_full + accum_stage,
                                            accum_phase)
                    tcgen05_after_thread_sync()
                    release_task_info(empty, Int32(stage))
                    # 从lane0广播valid_m，使同一warp一致选择epilogue路径。
                    valid_m = cute.arch.shuffle_sync(valid_m, 0)
                    if physical_warp == 4 and \
                            task_phase == GCBlockPhase.Linear2:
                        if elect_one_sync():
                            # acc_full表明本任务的MMA输入已读完；后续从TMEM写回，可归还输入ring槽。
                            ring_block = pool_block % self.ring_blocks
                            red_gpu_add_u32(
                                signals.iterator +
                                SIGNAL_OFFSETS['l2_empty_count'] // 4 + ring_block,
                                Uint32(1))
                    if task_phase == GCBlockPhase.Linear2:
                        # 各warp内汇合elected-lane路径后继续写回。
                        cute.arch.sync_warp()
                    if task_phase == GCBlockPhase.Linear1:
                        # W1按16dp布局读回，先BF16舍入和SiTU，再乘路由权重并生成MXFP8及scale。
                        # valid_m限制有效行，量化结果经CD暂存和TMA写入ring。
                        ring_block = pool_block % self.ring_blocks
                        generation = pool_block // self.ring_blocks
                        l2_empty_ptr = signals.iterator + \
                            SIGNAL_OFFSETS['l2_empty_count'] // 4 + ring_block
                        while ld_acquire_gpu_u32(l2_empty_ptr) != \
                                generation * (self.l2_clusters * 2):
                            pass
                        route_weights = cute.recast_ptr(
                            signals.iterator + self.l1_weight_words,
                            dtype=Float32)
                        epi_warp = physical_warp - 4
                        warpgroup = epi_warp // 4
                        warp_in_wg = epi_warp % 4
                        wg_start_m = warpgroup * (self.block_m // 2)
                        valid_wg_rows = Uint32(0)
                        if valid_m > wg_start_m:
                            valid_wg_rows = valid_m - wg_start_m
                            if valid_wg_rows > (self.block_m // 2):
                                valid_wg_rows = Uint32(self.block_m // 2)
                        valid_stores = (valid_wg_rows + (self.store_m_l1 - 1)) // \
                            self.store_m_l1
                        n_block = n_cluster * 2 + cta_rank
                        cd_wg = cd_smem + warpgroup * (2 * self.store_m_l1 * 64)
                        raw_values = cute.make_rmem_tensor(
                            (1, self.store_m_l1 // 8, 8), Float32)
                        stored_cached_weight = Float32(1)
                        if valid_stores == 0:
                            _release_w1_tmem(accum_empty, accum_stage)
                        for store in cutlass.range_constexpr(
                                (self.block_m // 2) // self.store_m_l1):
                            if store < valid_stores:
                                for atom in cutlass.range_constexpr(
                                        self.store_m_l1 // 8):
                                    base = accum_stage * self.block_m + \
                                        wg_start_m + (store * self.store_m_l1 + atom * 8)
                                    _load_w1_atom(raw_values, 0, atom, base)
                                    if atom == self.store_m_l1 // 8 - 1 and \
                                            (store + 1) == valid_stores:
                                        cute.arch.fence_view_async_tmem_load()
                                        _release_w1_tmem(accum_empty, accum_stage)
                                cute.arch.fence_view_async_tmem_load()
                                activation_values = cute.make_rmem_tensor(
                                    (self.store_m_l1 // 8, 4), Float32)
                                reduced_amax = cute.make_rmem_tensor(
                                    (self.store_m_l1 // 8, 2), Float32)
                                for atom in cutlass.range_constexpr(
                                        self.store_m_l1 // 8):
                                    j = store * (self.store_m_l1 // 8) + atom
                                    if j % 4 == 0:
                                        if (j * 8 + lane) < (self.block_m // 2):
                                            weight_token = ring_block * \
                                                self.block_m + wg_start_m + \
                                                (j * 8 + lane)
                                            stored_cached_weight = \
                                                route_weights[weight_token]
                                    source_lane = ((j * 8) % 32) + \
                                        (lane % 4) * 2
                                    weight0 = cute.arch.shuffle_sync(
                                        stored_cached_weight, source_lane)
                                    weight1 = cute.arch.shuffle_sync(
                                        stored_cached_weight, source_lane + 1)
                                    value0, value1 = _k3_routed_situ_pair(
                                        raw_values[0, atom, 0],
                                        raw_values[0, atom, 1],
                                        raw_values[0, atom, 2],
                                        raw_values[0, atom, 3],
                                        weight0,
                                        weight1)
                                    value2, value3 = _k3_routed_situ_pair(
                                        raw_values[0, atom, 4],
                                        raw_values[0, atom, 5],
                                        raw_values[0, atom, 6],
                                        raw_values[0, atom, 7],
                                        weight0,
                                        weight1)
                                    activation_values[atom, 0] = value0
                                    activation_values[atom, 1] = value1
                                    activation_values[atom, 2] = value2
                                    activation_values[atom, 3] = value3
                                    amax0 = warp_group4_max(
                                        fmax_f32(
                                            fmax_f32(Float32(0), abs_f32(value0)),
                                            abs_f32(value2)))
                                    amax1 = warp_group4_max(
                                        fmax_f32(
                                            fmax_f32(Float32(0), abs_f32(value1)),
                                            abs_f32(value3)))
                                    reduced_amax[atom, 0] = amax0
                                    reduced_amax[atom, 1] = amax1
                                    if lane < 4:
                                        amax_offset = epi_warp * \
                                            self.store_m_l1 + \
                                            (atom * 8 + lane * 2)
                                        epi_amax[amax_offset] = amax0
                                        epi_amax[amax_offset + 1] = amax1
                                    cute.arch.sync_warp()
                                tma_store_wait_1()
                                sync_epilogue_warpgroup(warpgroup)
                                tma_stage = store % 2
                                cd_stage = cd_wg + (tma_stage * self.store_m_l1 * 64)
                                for atom in cutlass.range_constexpr(
                                        self.store_m_l1 // 8):
                                    partner_offset = (epi_warp ^ 1) * \
                                        self.store_m_l1 + \
                                        (atom * 8) + \
                                        (lane % 4) * 2
                                    sf_amax0 = fmax_f32(
                                        reduced_amax[atom, 0],
                                        epi_amax[partner_offset])
                                    sf_amax1 = fmax_f32(
                                        reduced_amax[atom, 1],
                                        epi_amax[partner_offset + 1])
                                    sf_exp0 = ue8m0_fp8_scale_exp(sf_amax0)
                                    sf_exp1 = ue8m0_fp8_scale_exp(sf_amax1)
                                    inv0 = ue8m0_fp8_inv(sf_exp0)
                                    inv1 = ue8m0_fp8_inv(sf_exp1)
                                    upper0, upper1 = cute.arch.mul_packed_f32x2(
                                        (activation_values[atom, 0],
                                         activation_values[atom, 1]), (inv0, inv1), rnd='rn', ftz=False)
                                    lower0, lower1 = cute.arch.mul_packed_f32x2(
                                        (activation_values[atom, 2],
                                         activation_values[atom, 3]), (inv0, inv1), rnd='rn', ftz=False)
                                    packed = _w1_pack_fp8x4(
                                        upper0, upper1, lower0, lower1)
                                    row = lane
                                    smem_offset = (atom * 8 * 64) + \
                                        row * 64 + \
                                        (warp_in_wg ^ (row // 2)) * 16
                                    stmatrix_fp8x4_trans_late(
                                        cd_stage, smem_offset, packed)
                                    if warp_in_wg % 2 == 0 and lane < 4:
                                        output_sf_words = Uint64(self.l2_sf_words)
                                        sf_stride = Uint32(self.sf_ring_tokens)
                                        l2_sf_bytes = cute.recast_ptr(
                                            signals.iterator, dtype=Uint8) + \
                                            output_sf_words * 4
                                        k_sf = n_block * 2 + \
                                            warp_in_wg // 2
                                        token_base = wg_start_m + (store * self.store_m_l1 + atom * 8)
                                        assume_u32_lt(
                                            token_base, Uint32(self.block_m))
                                        sf_token = ring_block * self.sf_block_m + \
                                            (token_base & ~127) + \
                                            (token_base & 31) * 4 + \
                                            ((token_base >> 5) & 3) + \
                                            (lane * 8)
                                        if cutlass.const_expr(
                                                self.wide_routed_w1_sf):
                                            sf_addr = Uint64(k_sf // 4) * \
                                                (Uint64(sf_stride) * 4) + \
                                                Uint64(sf_token) * 4 + \
                                                (k_sf % 4)
                                            l2_sf_bytes[sf_addr] = Uint8(sf_exp0)
                                            l2_sf_bytes[sf_addr + 16] = \
                                                Uint8(sf_exp1)
                                        else:
                                            sf_addr = (k_sf // 4) * \
                                                (sf_stride * 4) + \
                                                sf_token * 4 + k_sf % 4
                                            l2_sf_bytes[sf_addr] = Uint8(sf_exp0)
                                            l2_sf_bytes[sf_addr + 16] = \
                                                Uint8(sf_exp1)
                                    cute.arch.sync_warp()
                                sync_epilogue_warpgroup(warpgroup)
                                if warp_in_wg == 0:
                                    with cute.arch.elect_one():
                                        output_map = tensor_map_4.iterator
                                        tma_store_fence()
                                        tile_s2g_2d(
                                            output_map, cd_stage,
                                            n_block * 64,
                                            ring_block * self.block_m +
                                            wg_start_m + (store * self.store_m_l1))
                                        tma_store_arrive()
                                cute.arch.sync_warp()
                        tma_store_wait_0()
                    else:
                        # W2按BM32、每组8行读回TMEM，转置为按token连续的BF16通道。
                        epi_warp = physical_warp - 4
                        warpgroup = epi_warp // 4
                        warp_in_wg = epi_warp % 4
                        wg_start_m = warpgroup * 16
                        valid_wg_rows = Uint32(0)
                        if valid_m > wg_start_m:
                            valid_wg_rows = valid_m - wg_start_m
                            if valid_wg_rows > 16:
                                valid_wg_rows = Uint32(16)
                        valid_stores = (valid_wg_rows + 7) // 8
                        n_block = n_cluster * 2 + cta_rank
                        raw_values = cute.make_rmem_tensor((1, 1, 8), Float32)
                        if valid_stores == 0:
                            _release_w2_tmem(accum_empty, accum_stage)
                        for store in cutlass.range_constexpr(2):
                            if store < valid_stores:
                                base = accum_stage * self.block_m + \
                                    wg_start_m + (store * 8)
                                _load_w2_atom(raw_values, 0, 0, base)
                                if (store + 1) == valid_stores:
                                    cute.arch.fence_view_async_tmem_load()
                                    _release_w2_tmem(accum_empty, accum_stage)
                                cute.arch.fence_view_async_tmem_load()
                                # 每warp输出8行×32通道；每lane写连续16B。
                                # shuffle必须所有lane参与，valid_m判断放在重排之后。
                                words = transpose_w2_bf16x8(
                                    pack_bf16_pair(raw_values[0, 0, 0], raw_values[0, 0, 1]),
                                    pack_bf16_pair(raw_values[0, 0, 2], raw_values[0, 0, 3]),
                                    pack_bf16_pair(raw_values[0, 0, 4], raw_values[0, 0, 5]),
                                    pack_bf16_pair(raw_values[0, 0, 6], raw_values[0, 0, 7]),
                                    lane)
                                row_in_atom = 2 * (lane % 4) + \
                                    (lane // 4) % 2
                                token_in_block = wg_start_m + (store * 8) + row_in_atom
                                pool_token = pool_block * self.block_m + token_in_block
                                record = _load_w2_source_metadata(
                                    signals.iterator + self.metadata_words,
                                    token_in_block, valid_m, pool_token,
                                    rank_idx, Uint32(0), self.topk, False)
                                if token_in_block < valid_m:
                                    token_bytes = Uint64(self.hidden * 2)
                                    slot_bytes = self.capacity * token_bytes
                                    channel_bytes = n_block * 256 + \
                                        warp_in_wg * 64 + \
                                        (lane // 8) * 16
                                    dst_bytes = (record[2] * slot_bytes +
                                                 record[1] * token_bytes +
                                                 channel_bytes)
                                    dst = cute.make_ptr(
                                        Uint32, mapped_peer_base(peer_ptrs, record[0], self.num_ranks) +
                                        (self.combine_words * 4) + dst_bytes,
                                        cute.AddressSpace.gmem, assumed_align=16)
                                    st_global_u32x4(dst, words[0], words[1], words[2], words[3])
                    sync_aligned(256, 2)
                if task_phase == GCBlockPhase.Linear1:
                    if physical_warp == 4:
                        if elect_one_sync():
                            ring_block = pool_block % self.ring_blocks
                            n_block = n_cluster * 2 + cta_rank
                            red_release_gpu_xor_u64(
                                signals.iterator + SIGNAL_OFFSETS['l2_full_mask'] // 4 +
                                ring_block * 2,
                                Uint64(1) << n_block)
                            red_gpu_add_u32(
                                signals.iterator + SIGNAL_OFFSETS['l1_empty_count'] // 4 +
                                ring_block, Uint32(1))
                    cute.arch.sync_warp()
                running = task_phase != GCBlockPhase.None_
                iteration = iteration + 1
        # 本次kernel执行期间recv_parity不变，各warp使用相同epoch。
        epoch = Uint64(ld_volatile_u32(
            signals.iterator + SIGNAL_OFFSETS['recv_parity'] // 4)) + 1
        cluster = block // 2
        shared_cta_rank = Uint32(cta_rank)
        if physical_warp == 16:
            self._shared_activation_loader(
                slab, shared_state, shared_maps, cluster, shared_cta_rank, epoch)
        if physical_warp == 17:
            self._shared_weight_loader(slab, shared_maps, cluster, shared_cta_rank)
        if shared_cta_rank == 0 and physical_warp == 18:
            self._shared_mma_worker(slab, cluster)
        if physical_warp >= 12 and physical_warp < 16:
            self._shared_swapped_epilogue(
                slab, shared_state, cluster, shared_cta_rank,
                physical_warp - 12, lane, epoch)
        sync_unaligned(640, 1)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        if physical_warp == 4:
            tmem_free_2cta(tptr[0], Uint32(self.tmem_cols))

    @cute.jit
    def __call__(self, signals: cute.Tensor, peer_ptrs: cute.Tensor,
                 rank_idx: Uint32, num_tokens: Uint32,
                 tensor_maps=None, y_out=None, stats=None,
                 activation_clamp=None, activation_alpha=None,
                 tensor_map_1=None, tensor_map_2=None, tensor_map_3=None,
                 tensor_map_4=None, tensor_map_5=None, tensor_map_6=None,
                 tensor_map_7=None, tensor_map_8=None, tensor_map_9=None,
                 tensor_map_10=None, tensor_map_11=None, tensor_map_12=None,
                 tensor_map_13=None, tensor_map_14=None, tensor_map_15=None,
                 tensor_map_16=None, tensor_map_17=None,
                 shared_state=None, shared_maps=None):
        self.kernel(signals, peer_ptrs, rank_idx, num_tokens,
                    tensor_maps, None, None, activation_clamp, activation_alpha,
                    tensor_map_1, tensor_map_2, tensor_map_3,
                    tensor_map_4, tensor_map_5, tensor_map_6,
                    tensor_map_7, tensor_map_8, tensor_map_9,
                    tensor_map_10, tensor_map_11, tensor_map_12,
                    tensor_map_13, tensor_map_14, tensor_map_15,
                    tensor_map_16, tensor_map_17,
                    shared_state, shared_maps).launch(
            grid=(self.compute_sms, 1, 1), block=(640, 1, 1),
            max_number_threads=(640, 1, 1), cluster=(2, 1, 1),
            min_blocks_per_mp=1)

    def _setup_shared_math(self, routed_smem_end: int) -> None:
        # shared采用M128/K64、两级输入缓冲和两个accumulator槽。
        self.shared_data_stages = 2
        self.shared_accum_stages = 2
        self.shared_a_stage_bytes = 64 * self.k3_layout.block_k * 2
        self.shared_b_stage_bytes = self.shared_cta_rows * self.k3_layout.block_k * 2
        self.k3_math_base = (routed_smem_end + 1023) // 1024 * 1024
        self.k3_a_smem = self.k3_math_base
        self.k3_b_smem = self.k3_a_smem + self.shared_data_stages * self.shared_a_stage_bytes
        self.k3_data_full = self.k3_b_smem + self.shared_data_stages * self.shared_b_stage_bytes
        self.k3_data_empty = self.k3_data_full + self.shared_data_stages * 8
        self.k3_accum_full = self.k3_data_empty + self.shared_data_stages * 8
        self.k3_accum_empty = self.k3_accum_full + self.shared_accum_stages * 8
        self.k3_smem_end = (self.k3_accum_empty + self.shared_accum_stages * 8 + 1023) // 1024 * 1024
        self.dynamic_smem = self.k3_smem_end
        self.k3_tmem_base = 128
        self.tmem_cols = 512
        # 双stage总数据为16KiB + shared_cta_rows*256B，另计控制区与1024B对齐。
        # 上限检查还需包含编译器静态SMEM；不能只统计动态slab。
        if self.k3_smem_end + 1024 > 232448:
            raise ValueError('K3 shared math plus static SMEM exceeds the B300 limit')
        if self.compute_sms % 2:
            raise ValueError('K3 shared uses two-CTA clusters')

    @cute.jit
    def _initialize_shared_math(self, slab: cute.Pointer,
                                physical_warp: Uint32, lane: Uint32) -> None:
        if physical_warp == 18:
            cta_rank, _, _ = cute.arch.block_in_cluster_idx()
            full = cute.recast_ptr(slab + self.k3_data_full, dtype=Uint64)
            empty = cute.recast_ptr(slab + self.k3_data_empty, dtype=Uint64)
            acc_full = cute.recast_ptr(slab + self.k3_accum_full, dtype=Uint64)
            acc_empty = cute.recast_ptr(slab + self.k3_accum_empty, dtype=Uint64)
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.shared_data_stages):
                    if cta_rank == 0:
                        cute.arch.mbarrier_init(full + stage, 4)
                    cute.arch.mbarrier_init(empty + stage, 1)
                for stage in cutlass.range_constexpr(self.shared_accum_stages):
                    if cta_rank == 0:
                        cute.arch.mbarrier_init(acc_empty + stage, 8)
                    cute.arch.mbarrier_init(acc_full + stage, 1)
            cute.arch.mbarrier_init_fence()

    @cute.jit
    def _wait_shared_flags(self, ptr: cute.Pointer, count: cutlass.Constexpr,
                       epoch: Uint64):
        tid, _, _ = cute.arch.thread_idx()
        lane = Uint32(tid % 32)
        for index in cutlass.range(lane, count, 32):
            start = read_clock64()
            ready = Uint64(0)
            while ready != epoch:
                ready = ld_acquire_gpu_u64(ptr + index)
                if read_clock64() - start >= self.nvlink_timeout_secs * 2000000000:
                    # 超时直接 trap，避免 printf 在正常轮询路径引入参数 spill。
                    device_trap()
        # 每lane acquire一部分标记，warp同步后才由elected lane发TMA。
        cute.arch.sync_warp()


    @cute.jit
    def _wait_shared_fc1(self, shared_state: cute.Tensor, token_tile: Uint32,
                         epoch: Uint64):
        # 每个标记覆盖本token tile的32个ACT通道，两CTA分别发布自己的通道。
        flags = cute.recast_ptr(shared_state.iterator + self.k3_layout.fc1_done, dtype=Uint64)
        count = self.k3_layout.fc1_flags_per_tile
        self._wait_shared_flags(flags + token_tile * count, count, epoch)
        fence_proxy_async_global()

    @cute.jit
    def _publish_shared_fc1(self, shared_state: cute.Tensor, token_tile: Uint32,
                            channel: Uint32, epoch: Uint64):
        flags = cute.recast_ptr(shared_state.iterator + self.k3_layout.fc1_done, dtype=Uint64)
        st_async_release_gpu_u64(
            flags + token_tile * self.k3_layout.fc1_flags_per_tile + channel, epoch)

    @cute.jit
    def _shared_activation_loader(self, slab: cute.Pointer, shared_state: cute.Tensor,
                                   shared_maps: cute.Tensor, cluster: Uint32,
                                   cta_rank: Uint32, epoch: Uint64) -> None:
        full = cute.recast_ptr(slab + self.k3_data_full, dtype=Uint64)
        empty = cute.recast_ptr(slab + self.k3_data_empty, dtype=Uint64)
        data_iter = Uint32(0)
        for layer in cutlass.range_constexpr(2):
            channel_tiles = self.k3_layout.fc1_tiles if layer == 0 else self.k3_layout.fc2_tiles
            k_steps = (self.k3_layout.hidden if layer == 0 else self.k3_layout.intermediate) // self.k3_layout.block_k
            descriptor = shared_maps.iterator + (0 if layer == 0 else 2 * 128)
            tensormap_acquire_gpu(descriptor)
            task = cluster
            while task < self.shared_token_tiles * channel_tiles:
                token_tile = task // channel_tiles
                outer = token_tile * self.shared_mma_n + cta_rank * self.shared_cta_rows
                if cutlass.const_expr(layer == 1):
                    self._wait_shared_fc1(shared_state, token_tile, epoch)
                for k_step in cutlass.range(Int32(k_steps), unroll=1):
                    stage = data_iter % self.shared_data_stages
                    phase = (data_iter // self.shared_data_stages) % 2
                    cute.arch.mbarrier_wait(empty + stage, phase ^ 1)
                    with cute.arch.elect_one():
                        tile_g2s_2cta_2d(
                            slab + self.k3_b_smem + stage * self.shared_b_stage_bytes,
                            descriptor, full + stage, Uint32(k_step) * 64, outer,
                            Uint64(EVICT_NORMAL))
                        if cta_rank == 0:
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                full + stage, 2 * self.shared_b_stage_bytes)
                        else:
                            cute.arch.mbarrier_arrive(full + stage, peer_cta_rank_in_cluster=0)
                    cute.arch.sync_warp()
                    data_iter += 1
                task += self.compute_sms // 2

    @cute.jit
    def _shared_weight_loader(self, slab: cute.Pointer, shared_maps: cute.Tensor,
                               cluster: Uint32, cta_rank: Uint32) -> None:
        full = cute.recast_ptr(slab + self.k3_data_full, dtype=Uint64)
        empty = cute.recast_ptr(slab + self.k3_data_empty, dtype=Uint64)
        data_iter = Uint32(0)
        for layer in cutlass.range_constexpr(2):
            channel_tiles = self.k3_layout.fc1_tiles if layer == 0 else self.k3_layout.fc2_tiles
            k_steps = (self.k3_layout.hidden if layer == 0 else self.k3_layout.intermediate) // self.k3_layout.block_k
            descriptor = shared_maps.iterator + (128 if layer == 0 else 3 * 128)
            tensormap_acquire_gpu(descriptor)
            task = cluster
            while task < self.shared_token_tiles * channel_tiles:
                channel_tile = task % channel_tiles
                # swapAB的A为权重；每CTA供应M128中的64个输出通道。
                outer = channel_tile * self.shared_mma_m + cta_rank * 64
                for k_step in cutlass.range(Int32(k_steps), unroll=1):
                    stage = data_iter % self.shared_data_stages
                    phase = (data_iter // self.shared_data_stages) % 2
                    cute.arch.mbarrier_wait(empty + stage, phase ^ 1)
                    with cute.arch.elect_one():
                        tile_g2s_2cta_2d(
                            slab + self.k3_a_smem + stage * self.shared_a_stage_bytes,
                            descriptor, full + stage, Uint32(k_step) * 64, outer,
                            Uint64(EVICT_FIRST))
                        if cta_rank == 0:
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                full + stage, 2 * self.shared_a_stage_bytes)
                        else:
                            cute.arch.mbarrier_arrive(full + stage, peer_cta_rank_in_cluster=0)
                    cute.arch.sync_warp()
                    data_iter += 1
                task += self.compute_sms // 2

    @cute.jit
    def _shared_mma_worker(self, slab: cute.Pointer, cluster: Uint32) -> None:
        clusters = self.compute_sms // 2
        full = cute.recast_ptr(slab + self.k3_data_full, dtype=Uint64)
        empty = cute.recast_ptr(slab + self.k3_data_empty, dtype=Uint64)
        acc_full = cute.recast_ptr(slab + self.k3_accum_full, dtype=Uint64)
        acc_empty = cute.recast_ptr(slab + self.k3_accum_empty, dtype=Uint64)
        data_iter = Uint32(0)
        task_iter = Uint32(0)
        a_base = kmajor_smem_desc(slab + self.k3_a_smem, Uint32(1024), True, Uint32(0))
        b_base = kmajor_smem_desc(slab + self.k3_b_smem, Uint32(1024), True, Uint32(0))
        for layer in cutlass.range_constexpr(2):
            n_tiles = self.k3_layout.fc1_tiles if layer == 0 else self.k3_layout.fc2_tiles
            k_steps = (self.k3_layout.hidden if layer == 0 else self.k3_layout.intermediate) // self.k3_layout.block_k
            task = cluster
            while task < self.shared_token_tiles * n_tiles:
                acc_stage = task_iter % self.shared_accum_stages
                acc_phase = (task_iter // self.shared_accum_stages) % 2
                cute.arch.mbarrier_wait(acc_empty + acc_stage, acc_phase ^ 1)
                tcgen05_after_thread_sync()
                for k_step in cutlass.range(Int32(k_steps), unroll=1):
                    stage = data_iter % self.shared_data_stages
                    phase = (data_iter // self.shared_data_stages) % 2
                    cute.arch.mbarrier_wait(full + stage, phase)
                    tcgen05_after_thread_sync()
                    with cute.arch.elect_one():
                        for kk in cutlass.range_constexpr(4):
                            offset_a = stage * (self.shared_a_stage_bytes // 16) + 2 * kk
                            offset_b = stage * (self.shared_b_stage_bytes // 16) + 2 * kk
                            accumulate = Uint32(1) if kk != 0 else Uint32(k_step != 0)
                            _mma_bf16_shared_2cta(
                                a_base + offset_a, b_base + offset_b,
                                self.k3_tmem_base + acc_stage * self.shared_accum_cols,
                                accumulate, self.shared_mma_n, self.shared_mma_m)
                        tcgen05_commit_2cta(empty + stage, Uint16(3))
                        if k_step == k_steps - 1:
                            tcgen05_commit_2cta(acc_full + acc_stage, Uint16(3))
                    cute.arch.sync_warp()
                    data_iter += 1
                task_iter += 1
                task += clusters
        if task_iter != 0:
            last = task_iter - 1
            cute.arch.mbarrier_wait(
                acc_empty + last % self.shared_accum_stages,
                (last // self.shared_accum_stages) % 2)

    @cute.jit
    def _shared_swapped_epilogue(self, slab: cute.Pointer, shared_state: cute.Tensor,
                                 cluster: Uint32, cta_rank: Uint32,
                                 epi_warp: Uint32, lane: Uint32, epoch: Uint64) -> None:
        # M128：每CTA负责64通道，warp0/1与warp2/3分别读取token tile的两个半区。
        # 16dp atom 每lane取两token×两通道；两次读凑成两token×四通道。
        acc_full = cute.recast_ptr(slab + self.k3_accum_full, dtype=Uint64)
        acc_empty = cute.recast_ptr(slab + self.k3_accum_empty, dtype=Uint64)
        clusters = self.compute_sms // 2
        task_iter = Uint32(0)
        for layer in cutlass.range_constexpr(2):
            n_tiles = self.k3_layout.fc1_tiles if layer == 0 else self.k3_layout.fc2_tiles
            task = cluster
            while task < self.shared_token_tiles * n_tiles:
                token_tile = task // n_tiles
                channel_tile = task % n_tiles
                acc_stage = task_iter % self.shared_accum_stages
                acc_phase = (task_iter // self.shared_accum_stages) % 2
                cute.arch.mbarrier_wait(acc_full + acc_stage, acc_phase)
                tcgen05_after_thread_sync()
                for stripe in cutlass.range_constexpr(self.shared_accum_cols // 8):
                    address = (epi_warp * (32 << 16) + self.k3_tmem_base
                               + acc_stage * self.shared_accum_cols + stripe * 8)
                    packed = cute.make_rmem_tensor((4,), Uint32)
                    for chunk in cutlass.range_constexpr(2):
                        values = tmem_load_16dp256b4(address + chunk * (16 << 16))
                        cute.arch.fence_view_async_tmem_load()
                        for pair in cutlass.range_constexpr(2):
                            if cutlass.const_expr(layer == 0):
                                # gate/up为相邻通道，分处lane bit2两侧；配对后先BF16舍入，再计算SiTU。
                                first = values[pair * 2]
                                second = values[pair * 2 + 1]
                                peer_first = cute.arch.shuffle_sync_bfly(first, 4)
                                peer_second = cute.arch.shuffle_sync_bfly(second, 4)
                                gate_first = first
                                up_first = peer_first
                                gate_second = second
                                up_second = peer_second
                                if lane & 4 != 0:
                                    gate_first = peer_first
                                    up_first = first
                                    gate_second = peer_second
                                    up_second = second
                                first = _k3_situ(gate_first, up_first)
                                second = _k3_situ(gate_second, up_second)
                                packed[chunk * 2 + pair] = f32x2_to_bf16x2_bits(first, second)
                            else:
                                packed[chunk * 2 + pair] = f32x2_to_bf16x2_bits(    values[pair * 2], values[pair * 2 + 1])
                    p0, p1, p2, p3 = transpose_w2_bf16x8(packed[0], packed[1], packed[2], packed[3], lane)
                    # 每lane重排为同一token的连续通道；行号只由token tile和半区决定。
                    token = (token_tile * self.shared_mma_n
                             + (epi_warp // 2) * self.shared_cta_rows
                             + stripe * 8 + 2 * (lane & 3) + ((lane >> 2) & 1))
                    channel_warp = epi_warp % 2
                    if cutlass.const_expr(layer == 0):
                        # 转置后每个SiTU结果重复两份，取各packed低半字，4 ACT连续写8B。
                        pair01 = (p0 & 0xffff) | (p1 << 16)
                        pair23 = (p2 & 0xffff) | (p3 << 16)
                        channel = (channel_tile * (self.shared_mma_m // 2)
                                   + cta_rank * (self.shared_mma_m // 4)
                                   + channel_warp * 16 + (lane >> 3) * 4)
                        destination = cute.recast_ptr(
                            shared_state.iterator + self.k3_layout.act, dtype=Uint64)
                        destination += (token * self.k3_layout.intermediate + channel) // 4
                        if cutlass.const_expr(self.shared_rows % self.shared_mma_n == 0):
                            st_global_u64(destination, Uint64(pair01) | (Uint64(pair23) << 32))
                        else:
                            if token < self.shared_rows:
                                st_global_u64(destination, Uint64(pair01) | (Uint64(pair23) << 32))
                    else:
                        channel = (channel_tile * self.shared_mma_m
                                   + cta_rank * (self.shared_mma_m // 2)
                                   + channel_warp * 32 + (lane >> 3) * 8)
                        destination = cute.recast_ptr(
                            shared_state.iterator + self.k3_layout.partial, dtype=Uint32)
                        destination += (token * self.k3_layout.hidden + channel) // 2
                        # 输出保持输入行序；完整warp参与TMEM读取与同步，只屏蔽越界store。
                        if cutlass.const_expr(self.shared_rows % self.shared_mma_n == 0):
                            st_global_u32x4(destination, p0, p1, p2, p3)
                        else:
                            if token < self.shared_rows:
                                st_global_u32x4(destination, p0, p1, p2, p3)
                sync_aligned(128, 10)
                if epi_warp == 0 and lane == 0:
                    if cutlass.const_expr(layer == 0):
                        flag_channel = channel_tile * 2 + cta_rank
                        self._publish_shared_fc1(shared_state, token_tile, flag_channel, epoch)
                tcgen05_before_thread_sync()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(acc_empty + acc_stage, peer_cta_rank_in_cluster=0)
                cute.arch.sync_warp()
                task_iter += 1
                task += clusters
