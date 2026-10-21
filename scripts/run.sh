#!/bin/bash
# GLM5.2 and DSV4.1 Flash decode accuracy and latency comparison.
# Usage: bash scripts/run.sh [GC_MOE_NUM_GPUS] [both|test|bench] [Python overrides...]
# Compare Mega MoE and GC MoE operators.
# Sweep 1..256 tokens per rank by powers of two with CUDA Graph timing.
# Set TMPDIR and GC_MOE_RESULT_DIR within the authorized ASI task directory.

set -euo pipefail

WORK_PATH=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$WORK_PATH/.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# PyTorch symmetric memory uses Unix-domain sockets (sun_path is
# limited to 108 bytes). Keep files in the requested task directory while
# exposing a short /proc path through this live shell's directory descriptor.
if [[ -n "${TMPDIR:-}" && ${#TMPDIR} -gt 40 ]]; then
    mkdir -p "$TMPDIR"
    exec {GC_TMP_FD}<"$TMPDIR"
    export TMPDIR="/proc/$$/fd/$GC_TMP_FD"
fi

# --- Parse positional args: [GC_MOE_NUM_GPUS] [STAGE] [extra args...] ---------------
NUM_GPUS_ARG=""
STAGE="both"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: scripts/run.sh [GC_MOE_NUM_GPUS] [both|test|bench|benchmark|all] [--models <glm5.2,dsv4.1-flash>] [--test-categories <default,boundary,asymmetric>] [Python overrides...]"
    exit 0
fi
if [[ $# -ge 1 && "$1" =~ ^[0-9]+$ ]]; then
    NUM_GPUS_ARG="$1"; shift
fi
if [[ $# -ge 1 && "$1" =~ ^(test|bench|benchmark|both|all)$ ]]; then
    case "$1" in
        bench|benchmark) STAGE="bench" ;;
        all) STAGE="both" ;;
        *) STAGE="$1" ;;
    esac
    shift
fi
EXTRA_ARGS=("$@")

GC_MOE_NUM_GPUS=${NUM_GPUS_ARG:-${GC_MOE_NUM_GPUS:-8}}
if [[ ! "$GC_MOE_NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[run.sh] ERROR: GC_MOE_NUM_GPUS must be a positive integer (got: $GC_MOE_NUM_GPUS)" >&2
    exit 2
fi

# One rank per selected GPU. An explicit GPU count takes precedence over GC_MOE_NUM_GPUS.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    GPU_DEVICES=()
    for (( gpu=0; gpu<GC_MOE_NUM_GPUS; gpu++ )); do GPU_DEVICES+=("$gpu"); done
else
    IFS=',' read -ra GPU_DEVICES <<<"$CUDA_VISIBLE_DEVICES"
    if (( ${#GPU_DEVICES[@]} < GC_MOE_NUM_GPUS )); then
        echo "[run.sh] ERROR: fewer visible GPUs than GC_MOE_NUM_GPUS=$GC_MOE_NUM_GPUS" >&2
        exit 2
    fi
fi
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPU_DEVICES[*]:0:GC_MOE_NUM_GPUS}")

GC_MOE_PYTHON_BIN=${GC_MOE_PYTHON_BIN:-/opt/conda310/bin/python}
GC_MOE_TEST_SCRIPT=${GC_MOE_TEST_SCRIPT:-tests/test_gc_moe.py}
GC_MOE_BENCH_SCRIPT=${GC_MOE_BENCH_SCRIPT:-benchmarks/benchmark_gc_moe.py}
GC_MOE_RESULT_SUFFIX=${GC_MOE_RESULT_SUFFIX:-}
GC_MOE_RESULT_DIR=${GC_MOE_RESULT_DIR:-$(mktemp -d "${TMPDIR:-$REPO_ROOT}/gc-moe-results.XXXXXX")}
mkdir -p "$GC_MOE_RESULT_DIR"
echo "[run.sh] raw result directory: $GC_MOE_RESULT_DIR"
export PATH=/opt/conda310/bin:/usr/local/cuda/bin:/usr/local/cmake/bin:${PATH:-}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

GC_MOE_TEST_PORT=${GC_MOE_TEST_PORT:-${MASTER_PORT:-29501}}
GC_MOE_BENCH_PORT=${GC_MOE_BENCH_PORT:-29502}

# --- Model presets: "tag|hidden|intermediate|experts|topk|shared|sharedK" ---
BENCH_MODELS=(
    "glm5.2|6144|2048|256|8|1|128"
    "dsv4.1-flash|5120|2304|384|6|1|32"
)

# --- Model selection ---------------------------------------------------------
# `--models <csv>` / `--models=<csv>` is consumed here (never forwarded to the
# python scripts); GC_MOE_MODELS env var is the alternative. Empty => all models.
MODELS_CSV="${GC_MOE_MODELS:-}"
TEST_CATEGORIES_CSV="${GC_MOE_TEST_CATEGORIES:-default,boundary,asymmetric}"
FILTERED_ARGS=()
_i=0
while (( _i < ${#EXTRA_ARGS[@]} )); do
    _arg="${EXTRA_ARGS[$_i]}"
    case "$_arg" in
        --models)
            MODELS_CSV="${EXTRA_ARGS[$((_i + 1))]:-}"
            _i=$((_i + 2))
            ;;
        --models=*)
            MODELS_CSV="${_arg#--models=}"
            _i=$((_i + 1))
            ;;
        --test-categories)
            TEST_CATEGORIES_CSV="${EXTRA_ARGS[$((_i + 1))]:-}"
            _i=$((_i + 2))
            ;;
        --test-categories=*)
            TEST_CATEGORIES_CSV="${_arg#--test-categories=}"
            _i=$((_i + 1))
            ;;
        *)
            FILTERED_ARGS+=("$_arg")
            _i=$((_i + 1))
            ;;
    esac
done
EXTRA_ARGS=("${FILTERED_ARGS[@]}")

IFS=',' read -ra GC_MOE_TEST_CATEGORIES <<<"$TEST_CATEGORIES_CSV"
if (( ${#GC_MOE_TEST_CATEGORIES[@]} == 0 )); then
    echo "[run.sh] ERROR: select at least one test category" >&2
    exit 2
fi
for category in "${GC_MOE_TEST_CATEGORIES[@]}"; do
    case "$category" in
        default|boundary|asymmetric) ;;
        *) echo "[run.sh] ERROR: unknown test category '$category'" >&2; exit 2 ;;
    esac
done
test_category_selected() {
    local selected
    for selected in "${GC_MOE_TEST_CATEGORIES[@]}"; do
        [[ "$selected" == "$1" ]] && return 0
    done
    return 1
}
echo "[run.sh] test categories: ${GC_MOE_TEST_CATEGORIES[*]}"


# Accuracy tests and benchmarks both run every BENCH_MODELS entry by default.
# An explicit `--models <csv>` / GC_MOE_MODELS selection overrides both stages.
ALL_TAGS=()
for _spec in "${BENCH_MODELS[@]}"; do
    ALL_TAGS+=("${_spec%%|*}")
done
if [[ -n "$MODELS_CSV" ]]; then
    IFS=',' read -ra TEST_TAGS <<<"$MODELS_CSV"
    BENCH_TAGS=("${TEST_TAGS[@]}")
else
    TEST_TAGS=("${ALL_TAGS[@]}")
    BENCH_TAGS=("${ALL_TAGS[@]}")
fi

get_model_spec() {
    # Args: <tag>; sets REPLY to the complete model preset.
    local _tag="$1" _spec
    for _spec in "${BENCH_MODELS[@]}"; do
        if [[ "${_spec%%|*}" == "$_tag" ]]; then
            REPLY="$_spec"
            return 0
        fi
    done
    return 1
}

for _tag in "${TEST_TAGS[@]}" "${BENCH_TAGS[@]}"; do
    if ! get_model_spec "$_tag"; then
        echo "[run.sh] ERROR: unknown model tag '$_tag' (valid: ${ALL_TAGS[*]})" >&2
        exit 2
    fi
    IFS='|' read -r _ _hidden _intermediate _experts _topk _shared _shared_k <<<"$REPLY"
    if (( _experts % GC_MOE_NUM_GPUS != 0 )); then
        echo "[run.sh] ERROR: $_tag experts=$_experts must be divisible by GC_MOE_NUM_GPUS=$GC_MOE_NUM_GPUS" >&2
        exit 2
    fi
done
echo "[run.sh] accuracy test models: ${TEST_TAGS[*]}"
echo "[run.sh] benchmark models:     ${BENCH_TAGS[*]}"

is_selected_model() {
    local _tag="$1" _t
    for _t in "${BENCH_TAGS[@]}"; do
        [[ "$_t" == "$_tag" ]] && return 0
    done
    return 1
}

# --- Decode token counts per rank ------------------------------------------
DECODE_TOKENS_PER_RANK=(1 2 4 8 16 32 64 128 256)
DECODE_PER_RANK_CSV=$(IFS=,; echo "${DECODE_TOKENS_PER_RANK[*]}")
DECODE_MAX_TOKENS_PER_RANK=${DECODE_TOKENS_PER_RANK[$((${#DECODE_TOKENS_PER_RANK[@]} - 1))]}
DECODE_GLOBAL_TOKENS=()
for tokens_per_rank in "${DECODE_TOKENS_PER_RANK[@]}"; do
    DECODE_GLOBAL_TOKENS+=("$(( tokens_per_rank * GC_MOE_NUM_GPUS ))")
done
echo "[run.sh] decode tokens_per_rank: $DECODE_PER_RANK_CSV"
echo "[run.sh] decode total_tokens (tokens_per_rank * $GC_MOE_NUM_GPUS ranks): ${DECODE_GLOBAL_TOKENS[*]}"
echo "[run.sh] decode num_max_tokens_per_rank=$DECODE_MAX_TOKENS_PER_RANK"

# --- Wait for the chosen visible GPUs to be idle before each run -------------
# Guards against picking up noise from other workloads sharing the node.
# Waits indefinitely until idle (proceeding on busy GPUs corrupts timing and
# can OOM); GC_MOE_GPU_IDLE_WAIT_S is only the progress-report interval.
GC_MOE_GPU_IDLE_WAIT_S=${GC_MOE_GPU_IDLE_WAIT_S:-30}
GC_MOE_GPU_IDLE_UTIL_THRESHOLD=0
GC_MOE_GPU_IDLE_MIN_FREE_MIB=${GC_MOE_GPU_IDLE_MIN_FREE_MIB:-51200}
GC_MOE_GPU_IDLE_MIN_SAMPLES=3
GC_MOE_GPU_IDLE_STABLE_S=10

wait_gpus_idle() {
    local indices="${CUDA_VISIBLE_DEVICES:-0}"
    local -a devices
    IFS=',' read -ra devices <<<"$indices"
    local waited=0 stable_since=0 idle_samples=0
    echo "[run.sh] waiting for GPUs [$indices]: util <= ${GC_MOE_GPU_IDLE_UTIL_THRESHOLD}%, free >= ${GC_MOE_GPU_IDLE_MIN_FREE_MIB} MiB, no compute process, at least ${GC_MOE_GPU_IDLE_MIN_SAMPLES} samples over ${GC_MOE_GPU_IDLE_STABLE_S}s"
    while :; do
        local snapshot compute_processes valid=1
        snapshot=$(nvidia-smi -i "$indices" --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits 2>/dev/null) || valid=0
        compute_processes=$(nvidia-smi -i "$indices" --query-compute-apps=pid,used_memory,process_name --format=csv,noheader,nounits 2>/dev/null) || valid=0
        local max_util=0 min_free=2147483647 count=0 index util free extra
        local -A seen=()
        while IFS=',' read -r index util free extra; do
            index=${index// /}; util=${util// /}; free=${free// /}
            if [[ ! "$index" =~ ^[0-9]+$ || ! "$util" =~ ^[0-9]+$ || ! "$free" =~ ^[0-9]+$ || -n "$extra" ]]; then
                valid=0
                continue
            fi
            if [[ -n "${seen[$index]:-}" ]]; then valid=0; fi
            seen[$index]=1
            count=$(( count + 1 ))
            if (( util > 100 )); then valid=0; fi
            if (( util > max_util )); then max_util=$util; fi
            if (( free < min_free )); then min_free=$free; fi
        done <<<"$snapshot"
        if (( count != ${#devices[@]} )); then valid=0; fi
        local now
        now=$(date +%s)
        if (( valid )) && [[ -z "$compute_processes" ]] && \
           (( max_util <= GC_MOE_GPU_IDLE_UTIL_THRESHOLD && min_free >= GC_MOE_GPU_IDLE_MIN_FREE_MIB )); then
            if (( stable_since == 0 )); then stable_since=$now; fi
            idle_samples=$(( idle_samples + 1 ))
            echo "[run.sh] idle sample time=$now count=$count snapshot=${snapshot//$'\n'/;} no compute process"
            if (( idle_samples >= GC_MOE_GPU_IDLE_MIN_SAMPLES && now - stable_since >= GC_MOE_GPU_IDLE_STABLE_S )); then
                echo "[run.sh] GPUs [$indices] idle (samples=${idle_samples}, span=$(( now - stable_since ))s, max util=${max_util}%, min free=${min_free} MiB, no compute process); proceeding"
                return 0
            fi
        else
            stable_since=0
            idle_samples=0
        fi
        if (( waited > 0 && waited % GC_MOE_GPU_IDLE_WAIT_S == 0 )); then
            echo "[run.sh] still waiting for GPUs [$indices] to idle (waited ${waited}s, valid=$valid, count=$count, max util=${max_util}%, min free=${min_free} MiB, compute_processes=$([[ -n "$compute_processes" ]] && echo yes || echo no))"
        fi
        sleep 2
        waited=$(( waited + 2 ))
    done
}

run_torchrun() {
    # Args: <label> <script> <port> <extra args ...>
    local label="$1" script="$2" port="$3"; shift 3
    echo "================================================================"
    echo "[run.sh] $label  (GC_MOE_NUM_GPUS=$GC_MOE_NUM_GPUS, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, port=$port)"
    echo "================================================================"
    local script_path="$script"
    [[ "$script_path" = /* ]] || script_path="$REPO_ROOT/$script_path"
    local cmd=("$GC_MOE_PYTHON_BIN" -m torch.distributed.run --nproc_per_node="$GC_MOE_NUM_GPUS" --master_port="$port" "$script_path" "$@")
    wait_gpus_idle
    "${cmd[@]}"

}

build_boundary_csv() {
    # Default powers of two plus every supported BLOCK_M/capacity boundary.
    local centers=(1 2 4 8 16 32 64 96 128 192 240 256)
    local -A normal=() selected=()
    local center delta value
    for value in "${DECODE_TOKENS_PER_RANK[@]}"; do normal[$value]=1; done
    for center in "${centers[@]}"; do
        for delta in -1 0 1; do
            value=$(( center + delta ))
            if (( value >= 0 && value <= DECODE_MAX_TOKENS_PER_RANK )) && [[ -z "${normal[$value]:-}" ]]; then
                selected[$value]=1
            fi
        done
    done
    local values=()
    for (( value=0; value<=DECODE_MAX_TOKENS_PER_RANK; value++ )); do
        [[ -n "${selected[$value]:-}" ]] && values+=("$value")
    done
    REPLY=$(IFS=,; echo "${values[*]}")
}

build_asym_vectors() {
    if (( GC_MOE_NUM_GPUS < 2 )); then REPLY=""; return; fi
    local vectors=() vector=() points=()
    local -A seen=()
    local rank pivot kind offset csv value
    # Empty global input; every rank as sole active/idle rank; alternating
    # zero/min/max loads; skew and reverse power-of-two ramps.
    local kinds=(all_zero alternating_zero_max alternating_max_zero alternating_zero_one alternating_one_zero skew reverse_skew ramp reverse_ramp)
    for kind in "${kinds[@]}"; do
        vector=()
        for (( rank=0; rank<GC_MOE_NUM_GPUS; rank++ )); do
            case "$kind" in
                all_zero) value=0 ;;
                alternating_zero_max) value=$(( rank % 2 == 0 ? 0 : DECODE_MAX_TOKENS_PER_RANK )) ;;
                alternating_max_zero) value=$(( rank % 2 == 0 ? DECODE_MAX_TOKENS_PER_RANK : 0 )) ;;
                alternating_zero_one) value=$(( rank % 2 )) ;;
                alternating_one_zero) value=$(( 1 - rank % 2 )) ;;
                skew) value=$(( rank < GC_MOE_NUM_GPUS / 2 ? 1 : DECODE_MAX_TOKENS_PER_RANK )) ;;
                reverse_skew) value=$(( rank < GC_MOE_NUM_GPUS / 2 ? DECODE_MAX_TOKENS_PER_RANK : 1 )) ;;
                ramp) value=${DECODE_TOKENS_PER_RANK[$(( rank % ${#DECODE_TOKENS_PER_RANK[@]} ))]} ;;
                reverse_ramp) value=${DECODE_TOKENS_PER_RANK[$(( (${#DECODE_TOKENS_PER_RANK[@]} - 1 - rank % ${#DECODE_TOKENS_PER_RANK[@]}) ))]} ;;
            esac
            vector+=("$value")
        done
        csv=$(IFS=,; echo "${vector[*]}")
        if [[ -z "${seen[$csv]:-}" ]]; then vectors+=("$csv"); seen[$csv]=1; fi
    done
    for (( pivot=0; pivot<GC_MOE_NUM_GPUS; pivot++ )); do
        for kind in one_active one_idle; do
            vector=()
            for (( rank=0; rank<GC_MOE_NUM_GPUS; rank++ )); do
                if [[ "$kind" == "one_active" ]]; then
                    value=$(( rank == pivot ? DECODE_MAX_TOKENS_PER_RANK : 0 ))
                else
                    value=$(( rank == pivot ? 0 : DECODE_MAX_TOKENS_PER_RANK ))
                fi
                vector+=("$value")
            done
            csv=$(IFS=,; echo "${vector[*]}")
            if [[ -z "${seen[$csv]:-}" ]]; then vectors+=("$csv"); seen[$csv]=1; fi
        done
    done
    # Every rank holds a boundary value; cycle through all boundary centers
    # and their neighbors, including 0/1 and 255/256 capacity endpoints.
    local boundary_begin=${#vectors[@]}
    local -A boundary_values=()
    IFS=',' read -ra points <<<"${1:-}"
    for value in "${points[@]}" "${DECODE_TOKENS_PER_RANK[@]}"; do boundary_values[$value]=1; done
    points=()
    for (( value=0; value<=DECODE_MAX_TOKENS_PER_RANK; value++ )); do
        [[ -n "${boundary_values[$value]:-}" ]] && points+=("$value")
    done
    for (( offset=0; offset<${#points[@]}; offset+=GC_MOE_NUM_GPUS )); do
        vector=()
        for (( rank=0; rank<GC_MOE_NUM_GPUS; rank++ )); do
            vector+=("${points[$(( (offset + rank) % ${#points[@]} ))]}")
        done
        csv=$(IFS=,; echo "${vector[*]}")
        if [[ -z "${seen[$csv]:-}" ]]; then vectors+=("$csv"); seen[$csv]=1; fi
    done
    # Keep about half of the unique cases, sampled across core patterns and
    # boundary combinations. Preserve every boundary combination when it fits.
    local original_count=${#vectors[@]}
    local target_count=$(( (original_count + 1) / 2 ))
    local boundary_count=$(( original_count - boundary_begin ))
    local keep_boundary=$boundary_count
    if (( keep_boundary > target_count / 2 )); then keep_boundary=$(( target_count / 2 )); fi
    local keep_core=$(( target_count - keep_boundary ))
    local reduced=() index selected
    for (( selected=0; selected<keep_core; selected++ )); do
        if (( keep_core > 1 )); then index=$(( selected * (boundary_begin - 1) / (keep_core - 1) )); else index=0; fi
        reduced+=("${vectors[$index]}")
    done
    for (( selected=0; selected<keep_boundary; selected++ )); do
        if (( keep_boundary > 1 )); then index=$(( boundary_begin + selected * (boundary_count - 1) / (keep_boundary - 1) )); else index=$boundary_begin; fi
        reduced+=("${vectors[$index]}")
    done
    REPLY=$(IFS=';'; echo "${reduced[*]}")
}

run_test_stage() {
    build_boundary_csv
    local boundary_csv="$REPLY"
    build_asym_vectors "$boundary_csv"
    local asym_vectors="$REPLY"
    echo "[run.sh] boundary tokens_per_rank: $boundary_csv"
    echo "[run.sh] asymmetric tokens_per_rank vectors: $asym_vectors"
    local tag spec hidden intermediate experts topk shared shared_k
    for tag in "${TEST_TAGS[@]}"; do
        get_model_spec "$tag"
        spec="$REPLY"
        IFS='|' read -r _ hidden intermediate experts topk shared shared_k <<<"$spec"
        local shape_args=(
            --hidden "$hidden"
            --intermediate-hidden "$intermediate"
            --num-experts "$experts"
            --num-topk "$topk"
        )
        if [[ "${GC_MOE_SE_MODE:-0}" == "1" ]]; then
            shape_args+=(--num-shared-experts "$shared" --shared-weight-gran-k "$shared_k")
        fi
        if test_category_selected default; then
            run_torchrun "Default accuracy test decode ($tag)" "$GC_MOE_TEST_SCRIPT" "$GC_MOE_TEST_PORT" \
                "${shape_args[@]}" --case-category default \
                --num-tokens-per-rank-list "$DECODE_PER_RANK_CSV" \
                --num-max-tokens-per-rank "$DECODE_MAX_TOKENS_PER_RANK" \
                --output-json "${GC_MOE_RESULT_DIR}/_accuracy_results_${tag}${GC_MOE_RESULT_SUFFIX}_decode_default.json" \
                "${EXTRA_ARGS[@]}"
        fi
        if test_category_selected boundary; then
            run_torchrun "Boundary accuracy test decode ($tag)" "$GC_MOE_TEST_SCRIPT" "$GC_MOE_TEST_PORT" \
                "${shape_args[@]}" --case-category boundary \
                --num-tokens-per-rank-list "$boundary_csv" \
                --num-max-tokens-per-rank "$DECODE_MAX_TOKENS_PER_RANK" \
                --output-json "${GC_MOE_RESULT_DIR}/_accuracy_results_${tag}${GC_MOE_RESULT_SUFFIX}_decode_boundary.json" \
                "${EXTRA_ARGS[@]}"
        fi
        if test_category_selected asymmetric; then
            if [[ -n "$asym_vectors" ]]; then
                run_torchrun "Asymmetric accuracy test decode ($tag)" "$GC_MOE_TEST_SCRIPT" "$GC_MOE_TEST_PORT" \
                    "${shape_args[@]}" --case-category asymmetric \
                    --per-rank-tokens-list "$asym_vectors" \
                    --num-max-tokens-per-rank "$DECODE_MAX_TOKENS_PER_RANK" \
                    --output-json "${GC_MOE_RESULT_DIR}/_accuracy_results_${tag}${GC_MOE_RESULT_SUFFIX}_decode_asymmetric.json" \
                    "${EXTRA_ARGS[@]}"
            else
                echo "[run.sh] asymmetric tests need at least two ranks; skipped for GC_MOE_NUM_GPUS=$GC_MOE_NUM_GPUS"
            fi
        fi
    done
}

run_bench_stage() {
    for spec in "${BENCH_MODELS[@]}"; do
        IFS='|' read -r tag hidden intermediate experts topk shared shared_k <<<"$spec"
        if ! is_selected_model "$tag"; then
            continue
        fi
        if (( experts % GC_MOE_NUM_GPUS != 0 )); then
            echo "[run.sh] Skip $tag: experts=$experts not divisible by GC_MOE_NUM_GPUS=$GC_MOE_NUM_GPUS"
            continue
        fi
        local model_args=()
        if [[ "${GC_MOE_SE_MODE:-0}" == "1" ]]; then
            model_args+=(--num-shared-experts "$shared" --shared-weight-gran-k "$shared_k")
        fi
        run_torchrun "Benchmark decode [$tag]" "$GC_MOE_BENCH_SCRIPT" "$GC_MOE_BENCH_PORT" \
            --hidden "$hidden" \
            --intermediate-hidden "$intermediate" \
            --num-experts "$experts" \
            --num-topk "$topk" \
            "${model_args[@]}" \
            --num-tokens-per-rank-list "$DECODE_PER_RANK_CSV" \
            --num-max-tokens-per-rank "$DECODE_MAX_TOKENS_PER_RANK" \
            --cuda-graph-max-tokens-per-rank "$DECODE_MAX_TOKENS_PER_RANK" \
            --output-json "${GC_MOE_RESULT_DIR}/_performance_results_${tag}${GC_MOE_RESULT_SUFFIX}_decode.json" \
            "${EXTRA_ARGS[@]}"
    done
}

case "$STAGE" in
    test)  run_test_stage ;;
    bench) run_bench_stage ;;
    both)
        run_test_stage
        run_bench_stage
        ;;
esac

echo "[run.sh] Done."
