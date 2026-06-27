#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-/home/zlong/anaconda3/envs/fdt-npu-py39/bin/python}"

RUN_NAME="${RUN_NAME:-qwen_48h_large_hard_ladder_npu_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${RUN_NAME}}"
DATA_DIR="${DATA_DIR:-data/qwen_48h_large_hard_ladder}"
DIFFICULTY="${DIFFICULTY:-hard_ladder}"
DATA_PRESET="${DATA_PRESET:-large}"
MAX_TRAIN_SECONDS="${MAX_TRAIN_SECONDS:-172800}"
BASELINE_MAX_TRAIN_SECONDS="${BASELINE_MAX_TRAIN_SECONDS:-21600}"
STEPS="${STEPS:-100000000}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
BASELINE_MICRO_BATCH_SIZE="${BASELINE_MICRO_BATCH_SIZE:-16}"
TRAIN_SAMPLING="${TRAIN_SAMPLING:-balanced_answer}"
LOG_INTERVAL_STEPS="${LOG_INTERVAL_STEPS:-100}"
CHECKPOINT_INTERVAL_STEPS="${CHECKPOINT_INTERVAL_STEPS:-5000}"
TRAIN_PROBE_EXAMPLES="${TRAIN_PROBE_EXAMPLES:-256}"
TRAIN_PROBE_INTERVAL_STEPS="${TRAIN_PROBE_INTERVAL_STEPS:-2000}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-1000}"
DIAGNOSTIC_METADATA_KEYS="${DIAGNOSTIC_METADATA_KEYS:-answer,num_nodes,path_length}"
DIAGNOSTIC_CASE_EXAMPLES="${DIAGNOSTIC_CASE_EXAMPLES:-2}"
CASE_EXAMPLES="${CASE_EXAMPLES:-5}"
LR="${LR:-0.0001}"

START_BASELINE_PACK="${START_BASELINE_PACK:-1}"
BASELINE_ASCEND_DEVICE_ID="${BASELINE_ASCEND_DEVICE_ID:-0}"
BASELINE_CONFIGS="${BASELINE_CONFIGS:-direct:-:0 cot:-:0 masked_cot:-:0}"

# Avoid physical chips that currently have external workloads on this host.
ASCEND_DEVICE_IDS="${ASCEND_DEVICE_IDS:-1 2 3 4 5 6 7 10 11 12 13}"
CONFIGS="${CONFIGS:-latent:1:0 latent:1:1 latent:4:0 latent:4:1 latent:8:0 latent:8:1 soft:4:0 soft:4:1 soft:8:0 soft:8:1 latent:16:0:2}"

cd "${REPO_DIR}"
mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/logs"

TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH="src:${PYTHONPATH:-}" "${PYTHON}" -m fdt.build_dataset \
  --task graph_reachability \
  --preset "${DATA_PRESET}" \
  --difficulty "${DIFFICULTY}" \
  --out-dir "${DATA_DIR}"

manifest="${OUTPUT_DIR}/manifest.tsv"
printf "session\tascend_device_id\tmethod\tk\tseed\toutput_json\tlog\tmicro_batch_size\n" > "${manifest}"

if [[ "${START_BASELINE_PACK}" == "1" ]]; then
  baseline_session="fdt48l_baseline_d${BASELINE_ASCEND_DEVICE_ID}"
  baseline_log="${OUTPUT_DIR}/logs/baseline_pack_d${BASELINE_ASCEND_DEVICE_ID}.log"
  if tmux has-session -t "${baseline_session}" 2>/dev/null; then
    echo "Session already exists: ${baseline_session}" >&2
    exit 1
  fi
  tmux new-session -d -s "${baseline_session}" \
    "cd '${REPO_DIR}' && ASCEND_RT_VISIBLE_DEVICES='${BASELINE_ASCEND_DEVICE_ID}' PYTHON='${PYTHON}' OUTPUT_DIR='${OUTPUT_DIR}' DATA_DIR='${DATA_DIR}' DIFFICULTY='${DIFFICULTY}' DATA_PRESET='${DATA_PRESET}' STEPS='${STEPS}' BASELINE_MAX_TRAIN_SECONDS='${BASELINE_MAX_TRAIN_SECONDS}' GRAD_ACCUM_STEPS='${GRAD_ACCUM_STEPS}' MICRO_BATCH_SIZE='${BASELINE_MICRO_BATCH_SIZE}' TRAIN_SAMPLING='${TRAIN_SAMPLING}' LOG_INTERVAL_STEPS='${LOG_INTERVAL_STEPS}' CHECKPOINT_INTERVAL_STEPS='${CHECKPOINT_INTERVAL_STEPS}' TRAIN_PROBE_EXAMPLES='${TRAIN_PROBE_EXAMPLES}' TRAIN_PROBE_INTERVAL_STEPS='${TRAIN_PROBE_INTERVAL_STEPS}' EVAL_EXAMPLES='${EVAL_EXAMPLES}' DIAGNOSTIC_METADATA_KEYS='${DIAGNOSTIC_METADATA_KEYS}' DIAGNOSTIC_CASE_EXAMPLES='${DIAGNOSTIC_CASE_EXAMPLES}' CASE_EXAMPLES='${CASE_EXAMPLES}' LR='${LR}' BASELINE_CONFIGS='${BASELINE_CONFIGS}' scripts/with_conda_npu.sh scripts/run_qwen_baseline_pack.sh 2>&1 | tee '${baseline_log}'"
  read -r -a baseline_configs <<< "${BASELINE_CONFIGS}"
  for config in "${baseline_configs[@]}"; do
    IFS=":" read -r method k seed <<< "${config}"
    suffix="${method}"
    if [[ "${k}" != "-" ]]; then
      suffix="${method}_k${k}"
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${baseline_session}" "${BASELINE_ASCEND_DEVICE_ID}" "${method}" "${k}" "${seed}" "${OUTPUT_DIR}/${suffix}_seed${seed}.json" "${baseline_log}" "${BASELINE_MICRO_BATCH_SIZE}" >> "${manifest}"
  done
  echo "Started ${baseline_session} on ASCEND_RT_VISIBLE_DEVICES=${BASELINE_ASCEND_DEVICE_ID}: ${BASELINE_CONFIGS}"
fi

read -r -a devices <<< "${ASCEND_DEVICE_IDS}"
read -r -a configs <<< "${CONFIGS}"
if (( ${#configs[@]} > ${#devices[@]} )); then
  echo "Need at least ${#configs[@]} ASCEND_DEVICE_IDS, got ${#devices[@]}." >&2
  exit 1
fi

for idx in "${!configs[@]}"; do
  IFS=":" read -r method k seed job_micro_batch_size <<< "${configs[$idx]}"
  job_micro_batch_size="${job_micro_batch_size:-${MICRO_BATCH_SIZE}}"
  device_id="${devices[$idx]}"
  suffix="${method}"
  if [[ "${k}" != "-" ]]; then
    suffix="${method}_k${k}"
  fi
  session="fdt48l_${idx}_${suffix}_s${seed}_d${device_id}"
  log_path="${OUTPUT_DIR}/logs/${suffix}_seed${seed}_d${device_id}.log"
  output_json="${OUTPUT_DIR}/${suffix}_seed${seed}.json"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "Session already exists: ${session}" >&2
    exit 1
  fi

  tmux new-session -d -s "${session}" \
    "cd '${REPO_DIR}' && ASCEND_RT_VISIBLE_DEVICES='${device_id}' PYTHON='${PYTHON}' OUTPUT_DIR='${OUTPUT_DIR}' DATA_DIR='${DATA_DIR}' DIFFICULTY='${DIFFICULTY}' DATA_PRESET='${DATA_PRESET}' METHOD='${method}' K='${k}' SEED='${seed}' STEPS='${STEPS}' MAX_TRAIN_SECONDS='${MAX_TRAIN_SECONDS}' GRAD_ACCUM_STEPS='${GRAD_ACCUM_STEPS}' MICRO_BATCH_SIZE='${job_micro_batch_size}' TRAIN_SAMPLING='${TRAIN_SAMPLING}' LOG_INTERVAL_STEPS='${LOG_INTERVAL_STEPS}' CHECKPOINT_INTERVAL_STEPS='${CHECKPOINT_INTERVAL_STEPS}' TRAIN_PROBE_EXAMPLES='${TRAIN_PROBE_EXAMPLES}' TRAIN_PROBE_INTERVAL_STEPS='${TRAIN_PROBE_INTERVAL_STEPS}' EVAL_EXAMPLES='${EVAL_EXAMPLES}' DIAGNOSTIC_METADATA_KEYS='${DIAGNOSTIC_METADATA_KEYS}' DIAGNOSTIC_CASE_EXAMPLES='${DIAGNOSTIC_CASE_EXAMPLES}' CASE_EXAMPLES='${CASE_EXAMPLES}' LR='${LR}' scripts/with_conda_npu.sh scripts/run_qwen_long_single.sh 2>&1 | tee '${log_path}'"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${session}" "${device_id}" "${method}" "${k}" "${seed}" "${output_json}" "${log_path}" "${job_micro_batch_size}" >> "${manifest}"
  echo "Started ${session} on ASCEND_RT_VISIBLE_DEVICES=${device_id}: ${method} k=${k} seed=${seed} micro_batch=${job_micro_batch_size}"
done

echo "${OUTPUT_DIR}" > outputs/latest_qwen_long_run.txt
echo "Manifest: ${manifest}"
echo "Latest pointer: outputs/latest_qwen_long_run.txt"
