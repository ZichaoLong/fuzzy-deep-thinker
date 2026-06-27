#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-graph_reachability}"
DIFFICULTY="${DIFFICULTY:-easy_ladder}"
DATA_PRESET="${DATA_PRESET:-debug}"
DATA_DIR="${DATA_DIR:-data/qwen_lora_matrix}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen_lora_matrix}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_DIR}/checkpoints}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-0.6B-Base}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-auto}"
DEVICE="${DEVICE:-npu:0}"
DTYPE="${DTYPE:-bfloat16}"
CONFIGS="${CONFIGS:-direct:- cot:- masked_cot:- soft:1 latent:1}"
SEEDS="${SEEDS:-0 1}"
STEPS="${STEPS:-80}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-16}"
LR="${LR:-0.0001}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
CASE_EXAMPLES="${CASE_EXAMPLES:-2}"
PYTHON="${PYTHON:-python3}"

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

PYTHONPATH="src:${PYTHONPATH:-}" "${PYTHON}" -m fdt.build_dataset \
  --task "${TASK}" \
  --preset "${DATA_PRESET}" \
  --difficulty "${DIFFICULTY}" \
  --out-dir "${DATA_DIR}"

for seed in ${SEEDS}; do
  for config in ${CONFIGS}; do
    method="${config%%:*}"
    k="${config##*:}"
    suffix="${method}"
    extra_k_args=()
    if [[ "${k}" != "-" ]]; then
      suffix="${method}_k${k}"
      extra_k_args=(--k "${k}")
    fi

    output_path="${OUTPUT_DIR}/${suffix}_seed${seed}.json"
    checkpoint_path="${CHECKPOINT_DIR}/${suffix}_seed${seed}.pt"
    echo "Running Qwen LoRA matrix point: config=${suffix} seed=${seed} steps=${STEPS}"
    PYTHONUNBUFFERED=1 PYTHONPATH="src:${PYTHONPATH:-}" "${PYTHON}" -m fdt.train_qwen \
      --model-name-or-path "${MODEL_NAME_OR_PATH}" \
      "${local_files_args[@]}" \
      --task "${TASK}" \
      --method "${method}" \
      --difficulty "${DIFFICULTY}" \
      --data-preset "${DATA_PRESET}" \
      --data-dir "${DATA_DIR}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}" \
      --steps "${STEPS}" \
      --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}" \
      --eval-examples "${EVAL_EXAMPLES}" \
      --eval-mode binary_choice \
      --lr "${LR}" \
      --seed "${seed}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --case-examples "${CASE_EXAMPLES}" \
      --use-lora \
      --lora-r "${LORA_R}" \
      --lora-alpha "${LORA_ALPHA}" \
      --lora-dropout "${LORA_DROPOUT}" \
      --lora-target-modules "${LORA_TARGET_MODULES}" \
      "${extra_k_args[@]}" \
      --save-checkpoint "${checkpoint_path}" \
      --output "${output_path}" \
      2>&1 | tee "${OUTPUT_DIR}/${suffix}_seed${seed}.log"
  done
done

OUTPUT_DIR="${OUTPUT_DIR}" CONFIGS="${CONFIGS}" PYTHONPATH="src:${PYTHONPATH:-}" "${PYTHON}" - <<'PY'
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

root = Path(os.environ["OUTPUT_DIR"])
method_order = {config.split(":", 1)[0]: i for i, config in enumerate(os.environ["CONFIGS"].split())}

rows = []
for path in root.glob("*_seed*.json"):
    payload = json.loads(path.read_text())
    rows.append(
        {
            "config": path.stem.rsplit("_seed", 1)[0],
            "method": payload["method"],
            "k": "" if payload["k"] is None else payload["k"],
            "seed": int(path.stem.rsplit("_seed", 1)[1]),
            "steps": payload["steps"],
            "grad_accum": payload["gradient_accumulation_steps"],
            "trainable_parameters": payload["trainable_parameters"],
            "dev": payload["dev"]["accuracy"],
            "id_test": payload["id_test"]["accuracy"],
            "ood_test": payload["ood_test"]["accuracy"],
            "loss": payload["train_loss_last"],
            "elapsed_sec": payload["elapsed_sec"],
            "checkpoint": payload["checkpoint_saved"],
        }
    )
rows.sort(key=lambda row: (method_order.get(row["method"], 999), str(row["k"]), row["seed"]))

print("\nQwen LoRA matrix")
print("config\tseed\tid_test\tood_test\tloss\tsec")
for row in rows:
    loss = "" if row["loss"] is None else f"{row['loss']:.3f}"
    print(f"{row['config']}\t{row['seed']}\t{row['id_test']:.3f}\t{row['ood_test']:.3f}\t{loss}\t{row['elapsed_sec']:.1f}")

groups = defaultdict(list)
for row in rows:
    groups[row["config"]].append(row)

def mean(values):
    return sum(values) / len(values) if values else 0.0

def std(values):
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))

aggregate = []
for config, group in sorted(groups.items(), key=lambda item: (method_order.get(item[1][0]["method"], 999), str(item[1][0]["k"]))):
    aggregate.append(
        {
            "config": config,
            "method": group[0]["method"],
            "k": group[0]["k"],
            "steps": group[0]["steps"],
            "n": len(group),
            "dev_mean": mean([row["dev"] for row in group]),
            "dev_std": std([row["dev"] for row in group]),
            "id_test_mean": mean([row["id_test"] for row in group]),
            "id_test_std": std([row["id_test"] for row in group]),
            "ood_test_mean": mean([row["ood_test"] for row in group]),
            "ood_test_std": std([row["ood_test"] for row in group]),
            "loss_mean": mean([row["loss"] for row in group if row["loss"] is not None]),
            "elapsed_sec_mean": mean([row["elapsed_sec"] for row in group]),
            "trainable_parameters": group[0]["trainable_parameters"],
        }
    )

(root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
(root / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

if rows:
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
if aggregate:
    with (root / "aggregate.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate)
PY
