#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/vasp_local_env.sh"

VASP_NP="${VASP_NP:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

exec prterun -n "${VASP_NP}" vasp_std "$@"
