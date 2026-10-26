# H20 DSV4.1-Flash 融合SE精度与性能

8卡47/47精度用例通过，覆盖全部8 rank。gc_se相对gc_unfused的decode九点平均加速为 **1.036768×**，相对mega_unfused为 **1.192780×**。

gc_se在6/9点快于gc_unfused，在9/9点快于mega_unfused，全部正负结果均列出。

测量日期：2026-10-29。8张NVIDIA H20，每卡1个rank；DSV4.1-Flash，H5120 / I2304 / 384 routed experts / top-k6，容量256。

完整算子采用CUDA Graph计时：10个warmup窗口，30个测量窗口，每窗口100次调用；先取各rank窗口中位数，再取8 rank最大值，单位µs。decode平均加速为九点延迟比的算术平均。权重准备和静态输入整理不计时。

每个测试阶段前均等待全部8卡util=0、无其他计算进程、空闲显存至少51200MiB，至少3次采样且持续10秒。

1个shared expert；routed使用FP4权重；shared使用FP8 E4M3权重，原始N32/K32 block scale沿N无损展开为逐行K32 scale。gc_unfused和mega_unfused均计入完整独立SE的两次GEMM、中间计算及最终BF16相加。

## gc_se 相对 gc_unfused

| Tokens/rank | Total tokens | gc_se (µs) | gc_unfused (µs) | gc_unfused/gc_se |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 145.917120 | 157.582240 | 1.079943× |
| 2 | 16 | 178.749924 | 189.080000 | 1.057791× |
| 4 | 32 | 237.300806 | 248.926563 | 1.048992× |
| 8 | 64 | 301.060324 | 330.562725 | 1.097995× |
| 16 | 128 | 356.527843 | 385.160313 | 1.080309× |
| 32 | 256 | 427.482395 | 421.267033 | 0.985461× |
| 64 | 512 | 472.225456 | 464.926720 | 0.984544× |
| 128 | 1024 | 568.884964 | 554.995689 | 0.975585× |
| 256 | 2048 | 775.905266 | 791.649132 | 1.020291× |

**decode平均加速：1.036768×。**

## gc_se 相对 mega_unfused

| Tokens/rank | Total tokens | gc_se (µs) | mega_unfused (µs) | mega_unfused/gc_se |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 145.917120 | 161.710081 | 1.108232× |
| 2 | 16 | 178.749924 | 213.014727 | 1.191691× |
| 4 | 32 | 237.300806 | 299.556007 | 1.262347× |
| 8 | 64 | 301.060324 | 409.762077 | 1.361063× |
| 16 | 128 | 356.527843 | 483.384953 | 1.355813× |
| 32 | 256 | 427.482395 | 511.919365 | 1.197522× |
| 64 | 512 | 472.225456 | 549.671345 | 1.164002× |
| 128 | 1024 | 568.884964 | 614.928474 | 1.080936× |
| 256 | 2048 | 775.905266 | 786.314392 | 1.013415× |

**decode平均加速：1.192780×。**

## 精度

原始阈值：cosine≥0.99、processed diff≤0.002。9例default、23例boundary、15例asymmetric，共47例、376个rank-case全部通过；以下极值覆盖所有用例、rank与重复。

| 比较 | 最小cosine | 最大processed diff |
| --- | ---: | ---: |
| gc_se / independent reference | 0.999982774258 | 1.7180379096e-05 |
| gc_unfused / independent reference | 0.999982774258 | 1.7180379096e-05 |
| mega_unfused / independent reference | 0.999982774258 | 1.71812484605e-05 |
| gc_se / gc_unfused | 0.999999701977 | 0 |
| gc_se / mega_unfused | 0.999999701977 | 1.68642055876e-09 |

| 比较 | 最大absolute error | 最大relative L2 |
| --- | ---: | ---: |
| gc_se / gc_unfused | 0 | 0 |
| gc_se / mega_unfused | 0.0625 | 5.80761807214e-05 |

| 其他精度检查 | 最大processed diff |
| --- | ---: |
| shared_diff | 1.64508212493e-05 |
| standalone_shared_diff | 1.64508212493e-05 |
| shared_gc_unfused_diff | 0 |
| changing_input_diff | 6.18343259051e-06 |
| half_changing_input_diff | 6.18343259051e-06 |

8 rank的非对齐输出及非法shared权重pair拒绝检查通过。输出全部有限，精确接收统计、eager/graph重复一致、固定地址变输入及半/全uint32 epoch wrap检查通过。
