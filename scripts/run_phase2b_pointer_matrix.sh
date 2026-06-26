#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TASK="${TASK:-pointer_chasing}"
export DIFFICULTY="${DIFFICULTY:-standard}"
export DATA_DIR="${DATA_DIR:-data/phase2b_pointer}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/phase2b_pointer_matrix}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_DIR}/checkpoints}"
export CONFIGS="${CONFIGS:-direct:- cot:- soft:0 soft:4 soft:8 latent:0 latent:4 latent:8}"
export SEEDS="${SEEDS:-0 1 2}"
export STEPS="${STEPS:-1000}"
export EVAL_EXAMPLES="${EVAL_EXAMPLES:-100}"
export D_MODEL="${D_MODEL:-32}"
export N_LAYERS="${N_LAYERS:-1}"
export N_HEADS="${N_HEADS:-2}"
export LR="${LR:-0.0003}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

exec "${SCRIPT_DIR}/run_phase1d_ladder_matrix.sh"
