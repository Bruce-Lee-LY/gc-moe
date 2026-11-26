#!/usr/bin/env bash
set -euo pipefail

stage=${1:-}
if [[ "$stage" != test && "$stage" != readiness && "$stage" != bench ]]; then
    echo 'Usage: bash scripts/run_ppu.sh {test|readiness|bench} [--shared] [ARGS...]' >&2
    exit 2
fi
shift
suffix=
if [[ "${1:-}" == --shared ]]; then
    suffix=_se
    shift
fi
case "$stage" in
    test) entry=tests/test_ppu_gc_moe${suffix}.py ;;
    readiness) entry=tests/test_ppu_gc_moe${suffix}_readiness.py ;;
    bench) entry=benchmarks/benchmark_ppu_gc_moe${suffix}.py ;;
esac

cd "$(dirname "$0")/.."
export DEEPEP_DISABLE_NETWORK=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
if [[ ${GC_MOE_NUM_NODES:-1} == 1 ]]; then
    export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
    rendezvous=(--standalone)
else
    rendezvous=(--nnodes="$GC_MOE_NUM_NODES" --node-rank="${GC_MOE_NODE_RANK:?set the PPU node rank}"
                --master-addr="${MASTER_ADDR:?set the PPU rendezvous address}"
                --master-port="${MASTER_PORT:-29500}")
fi
exec "${GC_MOE_PYTHON_BIN:-python}" -m torch.distributed.run \
    "${rendezvous[@]}" --nproc-per-node="${GC_MOE_NUM_PPUS:-8}" "$entry" "$@"
