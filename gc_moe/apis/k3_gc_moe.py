"""校验K3 routed/shared输入，并创建可重复启动的prepared call。"""

from math import inf

import torch

from .gc_moe import validate_call


# routed先BF16舍入，再SiTU和FP32路由权重；shared另保留BF16 activation边界。
K3_SITU_CONTRACT = ('situ', 4.0, 25.0)
K3_SHARED_OUTPUT_CONTRACT = 'flat_partial'


def prepare_k3_shared_gc(y, l1_weights, l2_weights, sym_buffer,
                         gathered_x, shared_w13, shared_w2, workspace, *,
                         stats=None, context=None, shared_w13_interleaved=False):
    """在CUDA Graph capture之前准备固定K3 SiTU(4,25)计算。

    routed权重使用recipe (1,1,32)。每次launch读取sym_buffer中的输入和TopK，
    以及连续的gathered_x [n,7168]；这些输入必须先就绪，包括Graph内部的
    stream/event依赖。n包含Graph padding，不需要按来源划分输入行。

    shared_w13为[2*I,7168]，shared_w2为[7168,I]，I须为64的倍数。
    shared_w13_interleaved=True时，W13的gate/up行已交错，直接复用；
    False时接受前半gate、后半up并重排行。

    shared_partial是工作区的BF16 [n,7168]视图，顺序与输入行一致，保存
    本卡权重的计算结果。所有输出消费者完成后才能复用该工作区。
    同一工作区的launch须串行；workspace和prepared call须存活到相关Graph退役。
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('prepare K3 shared weights/maps/binaries before CUDA graph capture')
    # 复用上游gated权重和缓冲校验；ValidatedCall不记录激活函数。
    # 此处的swiglu只选择相同的gate/up存储契约，role2固定执行K3 SiTU。
    config = validate_call(
        y, l1_weights, l2_weights, sym_buffer, None, None, stats,
        recipe=(1, 1, 32), activation='swiglu', activation_clamp=inf,
        fast_math=True, shared_recipe=(1, 1, 32), activation_alpha=1.0)
    from ..runtime.gc_moe import get_context

    context = get_context(num_ranks=sym_buffer.group.size()) if context is None else context
    if (config.tokens, config.hidden, config.intermediate, config.local_experts,
            config.shared_experts, sym_buffer.group.size(), sym_buffer.num_topk) != (128, 3584, 3072, 112, 0, 8, 16):
        raise ValueError('K3 SiTU requires EP8 routed T128/H3584/I3072/E896/top16')
    if (context.compute_sms < 32 or context.compute_sms % 2
            or context.dispatch_sms <= 1
            or workspace.layout.compute_sms != context.compute_sms):
        raise ValueError('K3 requires the runtime compute partition and a matching workspace')
    if workspace.routed_buffer is not sym_buffer or workspace.group is not sym_buffer.group:
        raise ValueError('workspace must be bound to this exact routed allocation and group')
    device = sym_buffer.buffer.device
    if gathered_x.ndim != 2 or not 1 <= gathered_x.shape[0] <= workspace.layout.tokens:
        raise ValueError('gathered_x must be a nonempty [n,7168] tensor within workspace token capacity')
    shared_rows = int(gathered_x.shape[0])
    if shared_w2.ndim != 2:
        raise ValueError('shared W2 must have shape [7168,I]')
    intermediate = int(shared_w2.shape[1])
    if intermediate < 64 or intermediate % 64 or intermediate != workspace.layout.intermediate:
        raise ValueError('shared I must be divisible by 64 and match workspace intermediate')
    for name, tensor, shape in (
        ('gathered shared input', gathered_x, (shared_rows, 7168)),
        ('shared W13', shared_w13, (2 * intermediate, 7168)),
        ('shared W2', shared_w2, (7168, intermediate)),
    ):
        if (tensor.device != device or tensor.dtype != torch.bfloat16
                or tuple(tensor.shape) != shape or not tensor.is_contiguous()
                or tensor.data_ptr() % 16):
            raise ValueError(f'{name} must be contiguous CUDA BF16 {shape} on the routed device')
    from ..jit_kernels.impls.sm100_fp8_fp4_gc_moe_k3_shared_external_rs import (
        prepare_k3_shared_binaries,
    )
    return prepare_k3_shared_binaries(
        config, y, l1_weights, l2_weights, sym_buffer, gathered_x,
        shared_w13, shared_w2, workspace, stats=stats, context=context,
        shared_w13_interleaved=shared_w13_interleaved,
        shared_rows=shared_rows)
