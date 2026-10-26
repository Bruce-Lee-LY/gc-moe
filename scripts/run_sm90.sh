#!/bin/bash
# SM90 DSV4.1 Flash decode tests; reuse run.sh GPU idle checks.
set -euo pipefail
WORK_PATH=$(cd "$(dirname "$0")" && pwd)
export GC_MOE_TEST_SCRIPT=${GC_MOE_TEST_SCRIPT:-tests/test_sm90_gc_moe.py}
export GC_MOE_BENCH_SCRIPT=${GC_MOE_BENCH_SCRIPT:-benchmarks/benchmark_sm90_gc_moe.py}
export GC_MOE_RESULT_SUFFIX=${GC_MOE_RESULT_SUFFIX:-_sm90_gc}
export GC_MOE_SE_MODE=${GC_MOE_SE_MODE:-0}
exec bash "$WORK_PATH/run.sh" "$@" --models dsv4.1-flash
