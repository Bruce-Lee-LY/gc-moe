"""K3 shared expert本地工作区分配与prepared对象的引用持有。"""

import torch

from ..layout.k3_shared import K3SharedLayout


class K3SharedWorkspace:
    """持有FC1标记、activation、partial及引用它们的prepared calls。

    tokens是连续输入的总行数容量，intermediate是本次权重的中间维。
    同一工作区的写入和消费须串行复用；使用它的CUDA Graph存活期间，
    工作区和prepared calls都必须保持存活。
    """
    def __init__(self, sym_buffer, tokens=256, intermediate=3072, compute_sms=None):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('create K3 shared workspace before CUDA graph capture')
        if sym_buffer.buffer is None or sym_buffer.handle is None:
            raise ValueError('routed GC buffer has been destroyed')
        self.group = sym_buffer.group
        if compute_sms is None:
            from ..runtime.gc_moe import get_context

            compute_sms = get_context(num_ranks=self.group.size()).compute_sms
        self.layout = K3SharedLayout(compute_sms=compute_sms, tokens=tokens,
                                     intermediate=intermediate)
        self.routed_buffer = sym_buffer
        self.buffer = torch.empty(self.layout.nbytes, dtype=torch.int8,
                                  device=sym_buffer.buffer.device)
        self.buffer.zero_()

        def view(offset, shape):
            count = 1
            for dim in shape:
                count *= dim
            return self.buffer.narrow(0, offset, count * self.layout.bf16_bytes).view(torch.bfloat16).view(shape)

        self.activations = view(self.layout.act, (self.layout.tokens, self.layout.intermediate))
        self.partial = view(self.layout.partial, (self.layout.tokens, self.layout.hidden))
        self._prepared = []
        # 构造返回时，初始epoch和数据区的清零已经完成。
        torch.cuda.synchronize(self.buffer.device)
