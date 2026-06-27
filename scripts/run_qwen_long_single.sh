#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-graph_reachability}"
DIFFICULTY="${DIFFICULTY:-hard_ladder}"
DATA_PRESET="${DATA_PRESET:-debug}"
DATA_DIR="${DATA_DIR:-data/qwen_long_hard_ladder}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen_long_hard_ladder}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_DIR}/checkpoints}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-0.6B-Base}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-auto}"
DEVICE="${DEVICE:-npu:0}"
DTYPE="${DTYPE:-bfloat16}"
METHOD="${METHOD:-direct}"
K="${K:--}"
SEED="${SEED:-0}"
STEPS="${STEPS:-100000000}"
MAX_TRAIN_SECONDS="${MAX_TRAIN_SECONDS:-172800}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
TRAIN_SAMPLING="${TRAIN_SAMPLING:-balanced_answer}"
LOG_INTERVAL_STEPS="${LOG_INTERVAL_STEPS:-100}"
CHECKPOINT_INTERVAL_STEPS="${CHECKPOINT_INTERVAL_STEPS:-20000}"
if [[ -z "${TRAIN_PROBE_EXAMPLES+x}" ]]; then
  case "${METHOD}" in
    cot|masked_cot) TRAIN_PROBE_EXAMPLES=0 ;;
    *) TRAIN_PROBE_EXAMPLES=64 ;;
  esac
fi
TRAIN_PROBE_INTERVAL_STEPS="${TRAIN_PROBE_INTERVAL_STEPS:-5000}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-200}"
DIAGNOSTIC_METADATA_KEYS="${DIAGNOSTIC_METADATA_KEYS:-answer,num_nodes,path_length}"
DIAGNOSTIC_CASE_EXAMPLES="${DIAGNOSTIC_CASE_EXAMPLES:-0}"
LR="${LR:-0.0001}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
CASE_EXAMPLES="${CASE_EXAMPLES:-3}"
PYTHON="${PYTHON:-/home/zlong/anaconda3/envs/fdt-npu-py39/bin/python}"

if [[ "${LOCAL_FILES_ONLY}" == "auto" && "${MODEL_NAME_OR_PATH}" == "Qwen/Qwen3-0.6B-Base" ]]; then
  snapshot="$(find "${HOME}/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "${snapshot}" ]]; then
    MODEL_NAME_OR_PATH="${snapshot}"
    LOCAL_FILES_ONLY="1"
  else
    LOCAL_FILES_ONLY="0"
  fi
fi
local_files_args=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  local_files_args=(--local-files-only)
fi

mkdir -p "${OUTPUT_DIR}" "${CHECKPOINT_DIR}"
if [[ ! -f "${DATA_DIR}/train/${TASK}.jsonl" || "${BUILD_DATA:-0}" == "1" ]]; then
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH="src:${PYTHONPATH:-}" "${PYTHON}" -m fdt.build_dataset \
    --task "${TASK}" \
    --preset "${DATA_PRESET}" \
    --difficulty "${DIFFICULTY}" \
    --out-dir "${DATA_DIR}"
fi

suffix="${METHOD}"
extra_k_args=()
if [[ "${K}" != "-" ]]; then
  suffix="${METHOD}_k${K}"
  extra_k_args=(--k "${K}")
fi

output_path="${OUTPUT_DIR}/${suffix}_seed${SEED}.json"
checkpoint_path="${CHECKPOINT_DIR}/${suffix}_seed${SEED}.pt"

echo "Running long Qwen job: method=${METHOD} k=${K} seed=${SEED} device=${DEVICE} visible=${ASCEND_RT_VISIBLE_DEVICES:-unset}"
echo "Output: ${output_path}"

PYTHONUNBUFFERED=1 PYTHONPATH="src:${PYTHONPATH:-}" "${PYTHON}" -m fdt.train_qwen \
  --model-name-or-path "${MODEL_NAME_OR_PATH}" \
  "${local_files_args[@]}" \
  --task "${TASK}" \
  --method "${METHOD}" \
  --difficulty "${DIFFICULTY}" \
  --data-preset "${DATA_PRESET}" \
  --data-dir "${DATA_DIR}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --steps "${STEPS}" \
  --max-train-seconds "${MAX_TRAIN_SECONDS}" \
  --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}" \
  --train-sampling "${TRAIN_SAMPLING}" \
  --log-interval-steps "${LOG_INTERVAL_STEPS}" \
  --checkpoint-interval-steps "${CHECKPOINT_INTERVAL_STEPS}" \
  --train-probe-examples "${TRAIN_PROBE_EXAMPLES}" \
  --train-probe-interval-steps "${TRAIN_PROBE_INTERVAL_STEPS}" \
  --eval-examples "${EVAL_EXAMPLES}" \
  --diagnostic-metadata-keys "${DIAGNOSTIC_METADATA_KEYS}" \
  --diagnostic-case-examples "${DIAGNOSTIC_CASE_EXAMPLES}" \
  --eval-mode binary_choice \
  --lr "${LR}" \
  --seed "${SEED}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --case-examples "${CASE_EXAMPLES}" \
  --use-lora \
  --lora-r "${LORA_R}" \
  --lora-alpha "${LORA_ALPHA}" \
  --lora-dropout "${LORA_DROPOUT}" \
  --lora-target-modules "${LORA_TARGET_MODULES}" \
  "${extra_k_args[@]}" \
  --save-checkpoint "${checkpoint_path}" \
  --output "${output_path}"
