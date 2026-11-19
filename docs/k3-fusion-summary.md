# K3 Routed / Shared Expert 融合方案

在 GC-MoE 的 **role2 计算 kernel** 中加入独立的 BF16 shared expert 流水线，将 shared 的 FC1、SiTU、FC2 与 routed expert 放在同一组 CTA 内执行。保留原有 dispatch / compute / finalize 三阶段结构；两条计算路径通过 warp 分工并行推进，各自维护数据与 accumulator 缓冲。

## 执行结构

每个 CTA 使用 640 线程、20 个 warp，两个 CTA 组成一个 cluster：

| Warp | 职责 |
|---|---|
| 0～1 | Routed activation / weight TMA |
| 2～3 | Routed MMA / 任务调度，仅 cluster 的 CTA0 执行 |
| 4～11 | Routed FC1 / FC2 epilogue |
| 12～15 | Shared SiTU、BF16 转换、转置与写回 |
| 16～17 | Shared activation / weight TMA |
| 18 | Shared MMA，由 CTA0 发射 2-CTA MMA |
| 19 | 无独立计算任务，参与公共初始化和最终汇合 |

Shared 无需等待 routed dispatch 完成；routed 按原调度器消费就绪任务。两者可在等待和计算期间重叠，共用 CTA 的资源预算，通过 `setmaxnreg` 按 warpgroup 分配寄存器。退出前完成 CTA / cluster 同步并释放 TMEM。

## Shared 计算与同步

```text
BF16 X[n,7168] → FC1 → SiTU(4,25) → BF16 ACT[n,I] → FC2 → BF16 partial[n,7168]
```

- 使用 **swapAB、2-CTA BF16 MMA**：M=128 为输出通道维，N 为 token 维；K 按 64 分块，输入和 accumulator 均使用双缓冲。FC1 的 M128 包含 gate/up，SiTU 后得到 64 个 ACT 通道。
- `tileN = min(256, ceil(n/16)*16)`。两 CTA 分别搬运一个 token tile 的前后半区；计算结果按通道划分，各 CTA 的 epilogue 均覆盖两个 token 半区。
- 输入和 ACT 各使用一张 TensorMap，真实行边界为 n。尾块越界读取由 TMA 填零，写回只覆盖 n 行；支持奇数行及超过 256 行的多个 token tile。
- FC1 每 CTA 完成一个任务后发布一个 GPU 作用域的 epoch 标记，覆盖 32 个 ACT 通道。FC2 activation warp 等待当前 token tile 的全部 `I/32` 个标记，再读取 ACT。
- 保留 FC1 accumulator 的 BF16 舍入边界。Routed 在 SiTU 后乘 FP32 路由权重并量化；shared 将 SiTU 结果存为 BF16 ACT，再供 FC2 使用。

## 接口与集成边界

公开入口为 `K3SharedWorkspace` 和 `prepare_k3_shared_gc`。Shared 接收连续输入 `[n,7168]`、本卡权重 `[2I,7168]` / `[7168,I]`，返回同输入行序的 `[n,7168]` partial。**接口不感知 TP 大小、rank owner 或 token 来源**；调用方按需要组织 AllGather、ReduceScatter 和依赖，使已有通信重叠可以保留。

Prepare 在 CUDA Graph capture 前完成权重布局、TensorMap 和编译缓存准备。n 使用 Graph 的物理行数，包含已有 padding；按对应形状准备并回放，不在 kernel 内判断 token 是否有效。工作区由调用方持有，所有输出消费者完成后才能复用。

当前 routed 专用范围为 **EP8、128 tokens/rank、H3584、I3072、896 experts、top-16**；shared 的 I 独立取自权重形状，须为正的 64 倍数。dispatch/compute 分区由上游 runtime 分配，K3 使用实际返回的 compute SM 数。已验证平台为 B300，编译入口要求 CUDA 13.2 或更新的 ptxas。

## 性能对比

以下为用户提供的移植前测试结果，原样保留。它们不是基于上游 `c0d60d8` 的本次测量；本次移植不以这些数字声明收益。

| Tokens/rank | 总 tokens | Shared TP | Baseline | GC-MoE |
|---:|---:|---:|---:|---:|
| 16 | 128 | 1 | 381.78 | 363.52 |
| 16 | 128 | 2 | 391.55 | 372.48 |
| 16 | 128 | 4 | 412.61 | 380.64 |
| 16 | 128 | 8 | 412.77 | 384.22 |
| 32 | 256 | 1 | 407.25 | 381.07 |
| 32 | 256 | 2 | 410.98 | 387.54 |
| 32 | 256 | 4 | 416.26 | 411.92 |
| 32 | 256 | 8 | 448.72 | 414.85 |
| 64 | 512 | 1 | 418.42 | 384.45 |
| 64 | 512 | 2 | 424.75 | 395.22 |
| 64 | 512 | 4 | 441.86 | 419.20 |
| 64 | 512 | 8 | 477.54 | 461.34 |
| 128 | 1024 | 1 | 429.74 | 388.46 |
| 128 | 1024 | 2 | 457.10 | 422.16 |
| 128 | 1024 | 4 | 504.29 | 489.57 |
| 128 | 1024 | 8 | 528.06 | 505.95 |

## 验证与代码位置

基于上游 `c0d60d813a3cffee874ffc5e2224263fc40d8ee7` 的本次移植已完成 B300 EP8 验证，使用上游原生 dispatch12/compute136 分区：

| shared 输入 | I | 验证内容 |
|---|---:|---|
| n=256 | 3072 | 外部 TP2 AG/RS、routed 参考、shared ACT/partial |
| n=50，容量64 | 3072 | 尾部行、额外容量未被覆盖 |
| n=300，容量320 | 1536 | 多 token tile、额外容量未被覆盖 |

每组每 rank 检查两次 eager 调用与两次 CUDA Graph 回放，共 96 次 rank 调用全部通过；参考实现和数值容差保持不变。实际 SM103f 编译为 96 registers/thread，stack 与 spill store/load 均为 0。公开接口通过无 CUDA context 的导入检查，独立审查无遗留阻塞问题。

上游原有文件仅 `gc_moe/__init__.py` 增加 K3 导出；其余通用实现保持上游内容。本次没有重新测量性能，也未进行框架完整层或 TPOT 验证。

实现按 API、工作区、布局、JIT 和设备代码分层，主 kernel 位于 [sm100_fp8_fp4_gc_moe_k3_shared.py](../gc_moe/impls/sm100_fp8_fp4_gc_moe_k3_shared.py)。接口细节见 [连续矩阵接口](k3-shared-external-rs.md)，通信接入见 [TP2 调用示例](k3-shared-tp2-usage.md)。
