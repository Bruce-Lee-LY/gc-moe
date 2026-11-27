#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo 'Usage: bash scripts/run_ppu_se.sh {test|readiness|bench} [--w4a4|--w4a8] [ARGS...]' >&2
    exit 2
fi

stage=$1
shift
exec bash "$(dirname "$0")/run_ppu.sh" "$stage" --shared "$@"
