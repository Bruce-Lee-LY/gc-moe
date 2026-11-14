# K3 shared 连续矩阵接口

公开入口：`from gc_moe import K3SharedWorkspace, prepare_k3_shared_gc`。

shared 部分只接收本卡上的输入矩阵和权重矩阵。接口不传 TP 大小、owner、pair 或按 owner 的偏移；AllGather、ReduceScatter 及通信顺序由调用方决定。设备实现融合 routed 与 shared 的 FC1/SiTU/FC2，返回与 shared 输入行序一致的 partial。

## 文件分工

| 层 | 文件 | 职责 |
|---|---|---|
| 公开 API | `gc_moe/apis/k3_gc_moe.py` | 参数与 capture 校验、SiTU/输出契约、prepare 入口 |
| 工作区 | `gc_moe/gc/k3_shared.py` | 本地分配、tensor views 与引用生命周期 |
| 布局 | `gc_moe/layout/k3_shared.py` | FC1 标记、activation 和 partial 的容量与偏移 |
| JIT | `gc_moe/jit_kernels/impls/sm100_fp8_fp4_gc_moe_k3_shared_external_rs.py` | TensorMap、PTX ABI 桥接、ptxas、cache、prepared 参数 |
| 设备实现 | `gc_moe/impls/sm100_fp8_fp4_gc_moe_k3_shared.py` | CuTe kernel、SMEM/TMEM 与 warp 分工 |

routed 仍固定 EP8、128 tokens/rank、H3584/I3072/E896/top16。shared 输入是连续 BF16 `gathered_x[n,7168]`，权重是 `shared_w13[2*I,7168]` 和 `shared_w2[7168,I]`。API 从 W2 得到 I，并校验 W13 和工作区；I 必须是正的 64 倍数，与 routed 的 I 独立。W13 默认前半 gate、后半 up；若行已交错，传 `shared_w13_interleaved=True` 复用。

运行资源由上游 `get_context(num_ranks=...)` 分配；workspace 使用该 context 实际获得的 compute SM 数。

```python
from gc_moe import K3SharedWorkspace, prepare_k3_shared_gc
from gc_moe.runtime.gc_moe import get_context

context = get_context(num_ranks=sym_buffer.group.size())
n = gathered_input.shape[0]  # 本次输入矩阵的总行数，包含已有 Graph padding
workspace = K3SharedWorkspace(
    sym_buffer, tokens=n, intermediate=shared_w2.shape[1], compute_sms=context.compute_sms,
)
prepared = prepare_k3_shared_gc(
    routed_output, routed_w1, routed_w2, sym_buffer,
    gathered_input, shared_w13, shared_w2, workspace, context=context,
)
# 输入和 routed staging 就绪后，在调用方安排的 stream 上启动。
prepared.launch()
shared_partial = prepared.shared_partial  # [n,7168]，与输入行序相同
```

`tokens` 是工作区的总行数容量，可以大于实际 n。设备使用 32 位元素索引，因此支持范围须满足 `tokens * max(7168, I) <= 2**32`；超出该容量会在分配前被拒绝。`prepared.shared_partial` 是工作区前 n 行的 view，不新增拷贝。所有输出消费者完成后才能复用工作区；同一工作区的 launch 必须串行。workspace、prepared call 及其输入缓冲须存活到相关 CUDA Graph 退役。

## 单输入 TensorMap 与连续 token tile

JIT 创建四张 TensorMap，顺序为输入、W13、ACT、W2。整张输入只有一个 TensorMap，ACT 同样只有一个；两者的实际行边界都是 n，ACT 使用 `workspace.activations[:n]`，不会把额外容量当作有效行。

```python
tileN = min(256, ((n + 15) // 16) * 16)
rows_per_cta = tileN // 2
token_tiles = (n + tileN - 1) // tileN
```

shared MMA 为 swapAB M128/N=tileN、K64、两级输入缓冲和两个 accumulator。M 是输出通道维，N 是 token 维。一个 token tile 中，两 CTA 的输入位置分别为 `tile_start` 和 `tile_start + rows_per_cta`，最后一个 tile 超出 n 的 TMA 行由硬件填零。n 可以为奇数，也可以超过 256；超过 256 时增加 token tile 数量。

CTA 的计算结果按通道划分，各 CTA 的 epilogue 都处理两个 token 半区，不能把 CTA 输入的半区理解为该 CTA 独占的输出行。FC1 ACT 连续存为 `[n,I]`，FC2 partial 连续存为 `[n,7168]`，global store 只写 n 以内的行。每个 token tile 有 `I/32` 个 FC1 完成标记，FC2 等待当前 token tile 的全部通道完成。

例如输入总共 50 行，tileN=64：两 CTA 分别搬入 32 行和 18 个有效行，剩余 14 个位置由 TMA 填零。kernel 不解释其中哪些行来自哪个 rank。调用方若要拆分输入，必须传入连续行视图，并按同样的连续位置收集输出；原先从每个 owner 各抽一段的分块接口已经移除。

## 本地工作区、ABI 与 Graph

工作区仅包含 FC1 标记、ACT 和 partial，起点各按 256B 对齐：

| 区域 | 容量 |
|---|---|
| fc1_done | `ceil(tokens/256) * (I/32)` 个 uint64 |
| act | BF16 `[tokens,I]` |
| partial | BF16 `[tokens,7168]` |

shared 工作区使用本地 CUDA 分配，不建立 peer 映射或 rendezvous；routed 的 GC 对称缓冲仍由原有管理器提供。构造时清零并等待本 GPU 初始化完成。

role2 有 26 个 CuTe 参数：原 routed 参数不变，shared 尾部仅为工作区与 TensorMap 两个指针。compile arguments、PTX 参数校验/重排和 runtime arguments 使用同一约定；没有额外 shared 输入指针、peer 指针或 owner 参数。

prepare 必须在 CUDA Graph capture 之前执行。二进制身份和缓存键包含实际 n、I、工作区布局及源码身份；prepared call 持有描述符、权重和工作区引用。调用方按已有 Graph 形状准备和缓存，运行时更新输入内容并回放对应 Graph。地址、形状或布局变化时须重新 prepare/capture。

验证范围见 [融合方案摘要](k3-fusion-summary.md#验证与代码位置)。
