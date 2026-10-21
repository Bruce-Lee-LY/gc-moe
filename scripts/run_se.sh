#!/bin/bash
# Compare Mega MoE and GC MoE operators with fused shared experts.
# Keep both model decode scenarios and per-stage GPU idle gates in run.sh.
set -euo pipefail
WORK_PATH=$(cd "$(dirname "$0")" && pwd)
export GC_MOE_TEST_SCRIPT=tests/test_gc_moe_se.py
export GC_MOE_BENCH_SCRIPT=benchmarks/benchmark_gc_moe_se.py
export GC_MOE_RESULT_SUFFIX=_gc_se
export GC_MOE_SE_MODE=1
exec bash "$WORK_PATH/run.sh" "$@"
