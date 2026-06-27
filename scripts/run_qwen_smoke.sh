#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-graph_reachability}"
DIFFICULTY="${DIFFICULTY:-easy_ladder}"
DATA_DIR="${DATA_DIR:-data/qwen_smoke}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen_smoke}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-0.6B-Base}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-auto}"
DEVICE="${DEVICE:-npu:0}"
DTYPE="${DTYPE:-bfloat16}"
CONFIGS="${CONFIGS:-direct:- cot:- masked_cot:- soft:4 latent:4}"
SEED="${SEED:-0}"
STEPS="${STEPS:-2}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-4}"
LR="${LR:-0.00001}"
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

mkdir -p "${OUTPUT_DIR}"

PYTHONPATH="src:${PYTHONPATH:-}" "${PYTHON}" -m fdt.build_dataset \
  --task "${TASK}" \
  --preset smoke \
  --difficulty "${DIFFICULTY}" \
  --out-dir "${DATA_DIR}"

for config in ${CONFIGS}; do
  method="${config%%:*}"
  k="${config##*:}"
  suffix="${method}"
  extra_k_args=()
  if [[ "${k}" != "-" ]]; then
    suffix="${method}_k${k}"
    extra_k_args=(--k "${k}")
  fi

  echo "Running Qwen smoke point: model=${MODEL_NAME_OR_PATH} config=${suffix} steps=${STEPS}"
  PYTHONUNBUFFERED=1 PYTHONPATH="src:${PYTHONPATH:-}" "${PYTHON}" -m fdt.train_qwen \
    --model-name-or-path "${MODEL_NAME_OR_PATH}" \
    "${local_files_args[@]}" \
    --task "${TASK}" \
    --method "${method}" \
    --difficulty "${DIFFICULTY}" \
    --data-dir "${DATA_DIR}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --steps "${STEPS}" \
    --eval-examples "${EVAL_EXAMPLES}" \
    --eval-mode binary_choice \
    --lr "${LR}" \
    --seed "${SEED}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --case-examples "${CASE_EXAMPLES}" \
    "${extra_k_args[@]}" \
    --output "${OUTPUT_DIR}/${suffix}_seed${SEED}.json" \
    2>&1 | tee "${OUTPUT_DIR}/${suffix}_seed${SEED}.log"
done

OUTPUT_DIR="${OUTPUT_DIR}" CONFIGS="${CONFIGS}" PYTHONPATH="src:${PYTHONPATH:-}" "${PYTHON}" - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_DIR"])
method_order = {config.split(":", 1)[0]: i for i, config in enumerate(os.environ["CONFIGS"].split())}
rows = []
for path in root.glob("*_seed*.json"):
    payload = json.loads(path.read_text())
    rows.append(
        {
            "method": payload["method"],
            "k": "" if payload["k"] is None else payload["k"],
            "steps": payload["steps"],
            "dev": payload["dev"]["accuracy"],
            "id_test": payload["id_test"]["accuracy"],
            "ood_test": payload["ood_test"]["accuracy"],
            "loss": payload["train_loss_last"],
            "elapsed_sec": payload["elapsed_sec"],
        }
    )
rows.sort(key=lambda row: (method_order.get(row["method"], 999), str(row["k"])))

print("\nQwen smoke summary")
print("method\tk\tdev\tid_test\tood_test\tloss\tsec")
for row in rows:
    loss = "" if row["loss"] is None else f"{row['loss']:.3f}"
    print(
        f"{row['method']}\t"
        f"{row['k']}\t"
        f"{row['dev']:.3f}\t"
        f"{row['id_test']:.3f}\t"
        f"{row['ood_test']:.3f}\t"
        f"{loss}\t"
        f"{row['elapsed_sec']:.1f}"
    )

(root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
with (root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
    if rows:
        writer.writeheader()
        writer.writerows(rows)
PY
