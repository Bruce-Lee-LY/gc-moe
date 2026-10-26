#!/bin/bash
# DSV4.1 Flash: gc_unfused and native mega_unfused versus gc_se.
set -euo pipefail
WORK_PATH=$(cd "$(dirname "$0")" && pwd)
export GC_MOE_TEST_SCRIPT=tests/test_sm90_gc_moe_se.py
export GC_MOE_BENCH_SCRIPT=benchmarks/benchmark_sm90_gc_moe_se.py
export GC_MOE_RESULT_SUFFIX=_sm90_gc_se
export GC_MOE_SE_MODE=1
exec bash "$WORK_PATH/run_sm90.sh" "$@"
