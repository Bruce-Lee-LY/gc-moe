#!/bin/bash
# SM120 (RTX PRO 5000) decode tests and benchmarks; reuse run.sh GPU idle checks.
set -euo pipefail
WORK_PATH=$(cd "$(dirname "$0")" && pwd)

export GC_MOE_TEST_SCRIPT=${GC_MOE_TEST_SCRIPT:-tests/test_sm120_gc_moe.py}
export GC_MOE_BENCH_SCRIPT=${GC_MOE_BENCH_SCRIPT:-benchmarks/benchmark_sm120_gc_moe.py}
export GC_MOE_RESULT_SUFFIX=${GC_MOE_RESULT_SUFFIX:-_sm120_gc}
export GC_MOE_SE_MODE=${GC_MOE_SE_MODE:-0}
exec bash "$WORK_PATH/run.sh" "$@"
