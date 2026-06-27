#!/usr/bin/env bash
set -euo pipefail

BASELINE_CONFIGS="${BASELINE_CONFIGS:-direct:-:0 cot:-:0 masked_cot:-:0}"
BASELINE_MAX_TRAIN_SECONDS="${BASELINE_MAX_TRAIN_SECONDS:-21600}"

read -r -a configs <<< "${BASELINE_CONFIGS}"
for config in "${configs[@]}"; do
  IFS=":" read -r method k seed <<< "${config}"
  echo "Running packed baseline: method=${method} k=${k} seed=${seed} max_seconds=${BASELINE_MAX_TRAIN_SECONDS}"
  METHOD="${method}" \
  K="${k}" \
  SEED="${seed}" \
  MAX_TRAIN_SECONDS="${BASELINE_MAX_TRAIN_SECONDS}" \
  scripts/run_qwen_long_single.sh
done
