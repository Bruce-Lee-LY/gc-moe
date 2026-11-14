# K3 shared：TP2 外部通信调用示例

GC 的 shared 接口只处理连续输入矩阵 `[n,7168]` 和本卡权重 `[2I,7168]`、`[7168,I]`，不知道 TP 大小或 owner。本文用 TP2 的 AllGather → GC → ReduceScatter 说明调用侧如何组织通信；TP2 不是 kernel 的参数。矩阵布局、工作区与 Graph 契约见 [连续矩阵接口](k3-shared-external-rs.md)。

routed 固定 EP8、128 tokens/rank、H3584/I3072/E896/top16。routed 和 shared 均使用 SiTU(4,25)，保留 FC1 的 BF16 舍入边界；原 GC routed 权重、FP8 输入和 scale 准备方式不变。通用 `fp8_fp4_gc_moe` 的公开调用签名及默认 SwiGLU 数学语义保持不变。

## 调用示例

此例在框架侧创建一个有两个 rank 的 `tp_group`，每个 rank 有 q 行输入，AG 按该 group 的 rank 顺序生成 `[2*q,7168]`。K3 shared 总中间维为 6144 时，TP2 每卡权重的 I 为 3072；这只是调用方选择的权重形状，GC 根据实际 W2 校验 I。

```python
import torch
import torch.distributed as dist
from gc_moe import K3SharedWorkspace, prepare_k3_shared_gc
from gc_moe.runtime.gc_moe import get_context

# 使用上游原有 context 和实际分区。
context = get_context(num_ranks=routed_buffer.group.size())
q = x_bf16.shape[0]
gathered_x = torch.empty((2 * q, 7168), device=x_bf16.device, dtype=torch.bfloat16)
shared_local = torch.empty_like(x_bf16)
workspace = K3SharedWorkspace(
    routed_buffer, tokens=gathered_x.shape[0],
    intermediate=shared_w2.shape[1], compute_sms=context.compute_sms,
)
# prepare 在 capture 外执行；TensorMap 绑定整个连续输入的固定地址。
prepared = prepare_k3_shared_gc(
    routed_y, routed_w1, routed_w2, routed_buffer,
    gathered_x, shared_w13, shared_w2, workspace, context=context,
)
# routed staging 也须先完成；跨 stream 时由调用侧建立依赖。
dist.all_gather_into_tensor(gathered_x, x_bf16, group=tp_group)
prepared.launch()
dist.reduce_scatter_tensor(shared_local, prepared.shared_partial, group=tp_group)
```

`prepared.shared_partial` 为连续 `[2*q,7168]`，顺序与 AG 输入完全相同，因此同一 group 的 RS 可按该顺序分发。若调用方主动重排 AG 行，必须在 RS 前恢复其要求的 rank 顺序。kernel 不维护另一套 owner 顺序。

q=25 时输入 n=50，内部 tileN=64；q=176 时输入 n=352，内部为两个 N256 token tile。两者都可用一次 prepare/launch 的完整输入接口，无需按 owner 切子段。这里仅描述 shared 输入覆盖；routed 仍是固定 128 行，框架对 routed 的分块及 shared 调用次数须保持各自输出只计算一次。

工作区容量须满足 `tokens * max(7168, I) <= 2**32`，保证 ACT/partial 的 32 位元素索引不溢出。所有输出消费者完成后才能复用 workspace。必须保留 workspace、prepared call 和输入缓冲直到 Graph 退役；同一工作区不得在不同 stream 并发重放。Graph 内采集通信时使用支持 capture 的 collective，并保留原有跨 stream/event 依赖。

## 维护验证入口

离线编译脚本的 `--shared-rows` 表示输入矩阵的总 n，不是每 rank 的 q；中间维由 `--shared-intermediate` 指定：

```bash
python scripts/compile_k3_shared_gc.py --shared-rows 50 --shared-intermediate 3072 \
  --output-dir runs/k3_shared_compile
```

GPU 验证必须先取得全部 8 卡锁。直接验证连续输入、无 shared AG/RS 的例子：

```bash
python -m torch.distributed.run --standalone \
  --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
  tests/test_gc_moe_k3_shared.py --flat-shared-tokens 513 \
  --shared-capacity 640 --shared-intermediate 768 --steps 4
```

`--flat-shared-tokens` 路径用于数值验证，不能与 `--benchmark` 合用。它检查本卡连续输入的 ACT/partial、routed 参考、交替输入 Graph 回放，以及工作区多余容量未被覆盖。I=768 是可传入的权重形状，不代表该测试建立了 shared TP8 通信。

保留的外部 TP2 通信维护入口：

```bash
python -m torch.distributed.run --standalone \
  --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
  tests/test_gc_moe_k3_shared.py --steps 8 --benchmark --warmup 10 --iters 20 --inner 20
```

该模式每 rank q=128，shared 总输入 n=256；测试侧执行 TP2 AG/RS，kernel 接口仍只有矩阵。反序 pair 的行顺序处理也在测试侧完成。

**此 benchmark 只计三 role GC 调用和外置 TP2 RS，外部 AG 与输入 staging 在计时范围之外。** 指标名为 `complete_gc_plus_external_tp2_rs_l2_cold_cuda_graph_us`，不能代替框架完整层测量，也不能与旧内置 AG 融合测量混算。每张 Graph 只有一次 GC+RS 调用，`--inner` 表示每轮独立计时次数；每次调用前各 rank 读 506MiB 缓冲驱逐 L2，同步后 EP8 汇合，再记录 start event，驱逐与汇合不计时。

以上命令是维护入口，不表示已经执行这些配置。实际验证范围见 [融合方案摘要](k3-fusion-summary.md#验证与代码位置)。
