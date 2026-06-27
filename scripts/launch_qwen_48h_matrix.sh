#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-/home/zlong/anaconda3/envs/fdt-npu-py39/bin/python}"

RUN_NAME="${RUN_NAME:-qwen_48h_hard_ladder_npu_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${RUN_NAME}}"
DATA_DIR="${DATA_DIR:-data/qwen_48h_hard_ladder}"
DIFFICULTY="${DIFFICULTY:-hard_ladder}"
DATA_PRESET="${DATA_PRESET:-debug}"
MAX_TRAIN_SECONDS="${MAX_TRAIN_SECONDS:-172800}"
STEPS="${STEPS:-100000000}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
TRAIN_SAMPLING="${TRAIN_SAMPLING:-balanced_answer}"
LOG_INTERVAL_STEPS="${LOG_INTERVAL_STEPS:-100}"
CHECKPOINT_INTERVAL_STEPS="${CHECKPOINT_INTERVAL_STEPS:-20000}"
TRAIN_PROBE_INTERVAL_STEPS="${TRAIN_PROBE_INTERVAL_STEPS:-5000}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-200}"
DIAGNOSTIC_METADATA_KEYS="${DIAGNOSTIC_METADATA_KEYS:-answer,num_nodes,path_length}"

# Physical Ascend device IDs. The default uses fully idle NPU rows 0, 1, 2, 3, 5, and 6.
ASCEND_DEVICE_IDS="${ASCEND_DEVICE_IDS:-0 1 2 3 4 5 6 7 10 11 12 13}"
CONFIGS="${CONFIGS:-direct:-:0 direct:-:1 masked_cot:-:0 masked_cot:-:1 latent:1:0 latent:1:1 latent:4:0 latent:4:1 latent:8:0 latent:8:1 soft:4:0 soft:4:1}"

cd "${REPO_DIR}"
mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/logs"

TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH="src:${PYTHONPATH:-}" "${PYTHON}" -m fdt.build_dataset \
  --task graph_reachability \
  --preset "${DATA_PRESET}" \
  --difficulty "${DIFFICULTY}" \
  --out-dir "${DATA_DIR}"

read -r -a devices <<< "${ASCEND_DEVICE_IDS}"
read -r -a configs <<< "${CONFIGS}"
if (( ${#configs[@]} > ${#devices[@]} )); then
  echo "Need at least ${#configs[@]} ASCEND_DEVICE_IDS, got ${#devices[@]}." >&2
  exit 1
fi

manifest="${OUTPUT_DIR}/manifest.tsv"
printf "session\tascend_device_id\tmethod\tk\tseed\toutput_json\tlog\n" > "${manifest}"

for idx in "${!configs[@]}"; do
  IFS=":" read -r method k seed <<< "${configs[$idx]}"
  device_id="${devices[$idx]}"
  suffix="${method}"
  if [[ "${k}" != "-" ]]; then
    suffix="${method}_k${k}"
  fi
  session="fdt48_${idx}_${suffix}_s${seed}_d${device_id}"
  log_path="${OUTPUT_DIR}/logs/${suffix}_seed${seed}_d${device_id}.log"
  output_json="${OUTPUT_DIR}/${suffix}_seed${seed}.json"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "Session already exists: ${session}" >&2
    exit 1
  fi

  tmux new-session -d -s "${session}" \
    "cd '${REPO_DIR}' && ASCEND_RT_VISIBLE_DEVICES='${device_id}' PYTHON='${PYTHON}' OUTPUT_DIR='${OUTPUT_DIR}' DATA_DIR='${DATA_DIR}' DIFFICULTY='${DIFFICULTY}' DATA_PRESET='${DATA_PRESET}' METHOD='${method}' K='${k}' SEED='${seed}' STEPS='${STEPS}' MAX_TRAIN_SECONDS='${MAX_TRAIN_SECONDS}' GRAD_ACCUM_STEPS='${GRAD_ACCUM_STEPS}' TRAIN_SAMPLING='${TRAIN_SAMPLING}' LOG_INTERVAL_STEPS='${LOG_INTERVAL_STEPS}' CHECKPOINT_INTERVAL_STEPS='${CHECKPOINT_INTERVAL_STEPS}' TRAIN_PROBE_INTERVAL_STEPS='${TRAIN_PROBE_INTERVAL_STEPS}' EVAL_EXAMPLES='${EVAL_EXAMPLES}' DIAGNOSTIC_METADATA_KEYS='${DIAGNOSTIC_METADATA_KEYS}' scripts/with_conda_npu.sh scripts/run_qwen_long_single.sh 2>&1 | tee '${log_path}'"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${session}" "${device_id}" "${method}" "${k}" "${seed}" "${output_json}" "${log_path}" >> "${manifest}"
  echo "Started ${session} on ASCEND_RT_VISIBLE_DEVICES=${device_id}: ${method} k=${k} seed=${seed}"
done

echo "${OUTPUT_DIR}" > outputs/latest_qwen_long_run.txt
echo "Manifest: ${manifest}"
echo "Latest pointer: outputs/latest_qwen_long_run.txt"
