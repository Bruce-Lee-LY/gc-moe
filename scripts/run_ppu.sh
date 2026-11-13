#!/usr/bin/env bash
set -eo pipefail

usage() {
    echo 'Usage: scripts/run_ppu.sh SDK_DIR [test|bench] [ARGS...]'
    echo 'Builds GC MoE and tests GLM 5.2 and DSV 4.1 Flash on eight PPUs.'
    echo 'Defaults to bench with no arguments. Pass test --readiness --steps 160 for workspace reuse.'
    echo 'Pass --num-tokens-per-rank-list 256 for accuracy only.'
    echo 'Pass --case-category boundary or asymmetric for the additional accuracy suites.'
    echo 'Pass --contract to test the public API on a non-default stream.'
    echo 'Pass bench --shared-expert to compare fused SE with both serial SE baselines and routed GC MoE.'
}

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    usage
    exit 0
fi
if [[ $# == 0 ]]; then
    usage >&2
    exit 2
fi

sdk_dir=$(cd "$1" && pwd)
shift
stage=test
case "${1:-}" in
    '') stage=bench ;;
    test|bench) stage=$1; shift ;;
esac

repo=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo"
source scripts/ppu_sdk22_env.sh "$sdk_dir"

num_ranks=${GC_MOE_NUM_GPUS:-8}
shared_suffix=
readiness_suffix=
test_args=()
while (( $# )); do
    case "$1" in
        --ranks)
            num_ranks=${2:-}
            if (( $# > 1 )); then shift; fi ;;
        --ranks=*) num_ranks=${1#*=} ;;
        --perf) stage=bench ;;
        --shared-expert|--num-shared-experts=1) shared_suffix=_se; test_args+=("$1") ;;
        --num-shared-experts)
            if [[ ${2:-} == 1 ]]; then shared_suffix=_se; fi
            test_args+=("$1" "${2:-}")
            if (( $# > 1 )); then shift; fi ;;
        --readiness) readiness_suffix=_readiness ;;
        *) test_args+=("$1") ;;
    esac
    shift
done
if [[ ! "$num_ranks" =~ ^[1-8]$ ]]; then
    echo 'Expected one to eight PPU ranks.' >&2
    exit 2
fi
if [[ "$stage" == bench ]]; then
    if [[ -n "$readiness_suffix" ]]; then
        echo 'Workspace readiness requires the test stage.' >&2
        exit 2
    fi
    entry="benchmarks/benchmark_ppu_gc_moe${shared_suffix}.py"
else
    entry="tests/test_ppu_gc_moe${shared_suffix}${readiness_suffix}.py"
fi
mkdir -p build/ppu
result_dir=${GC_MOE_RESULT_DIR:-$(mktemp -d build/ppu/results.XXXXXX)}
mkdir -p "$result_dir"
echo "[run_ppu.sh] result directory: $result_dir"

for model in glm52 dsv41flash; do
    torchrun --standalone --nproc-per-node="$num_ranks" "$entry" \
        --cache-dir build/ppu --model "$model" \
        --output-json "$result_dir/$model.json" "${test_args[@]}"
done
