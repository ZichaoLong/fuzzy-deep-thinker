#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TASK="${TASK:-graph_reachability}"
export DIFFICULTY="${DIFFICULTY:-easy_ladder}"
export DATA_DIR="${DATA_DIR:-data/phase2a_ladder_scale}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/phase2a_ladder_scale}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_DIR}/checkpoints}"
export CONFIGS="${CONFIGS:-direct:- cot:- soft:8 latent:8}"
export SEEDS="${SEEDS:-0 1 2}"
export STEPS="${STEPS:-3000}"
export EVAL_EXAMPLES="${EVAL_EXAMPLES:-200}"
export D_MODEL="${D_MODEL:-64}"
export N_LAYERS="${N_LAYERS:-2}"
export N_HEADS="${N_HEADS:-4}"
export LR="${LR:-0.0003}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"

exec "${SCRIPT_DIR}/run_phase1d_ladder_matrix.sh"
