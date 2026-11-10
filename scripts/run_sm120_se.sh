#!/bin/bash
# Run SM120 fused shared-expert accuracy and decode benchmarks.
set -euo pipefail
WORK_PATH=$(cd "$(dirname "$0")" && pwd)
export GC_MOE_TEST_SCRIPT=tests/test_sm120_gc_moe_se.py
export GC_MOE_BENCH_SCRIPT=benchmarks/benchmark_sm120_gc_moe_se.py
export GC_MOE_RESULT_SUFFIX=_sm120_gc_se
export GC_MOE_SE_MODE=1
exec bash "$WORK_PATH/run_sm120.sh" "$@"
