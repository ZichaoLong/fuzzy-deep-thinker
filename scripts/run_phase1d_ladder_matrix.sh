#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-graph_reachability}"
DIFFICULTY="${DIFFICULTY:-easy_ladder}"
DATA_DIR="${DATA_DIR:-data/phase1d_ladder}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/phase1d_ladder_matrix}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_DIR}/checkpoints}"
DEVICE="${DEVICE:-cpu}"
SEEDS="${SEEDS:-0 1 2}"
STEPS="${STEPS:-1000}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-100}"
D_MODEL="${D_MODEL:-32}"
N_LAYERS="${N_LAYERS:-1}"
N_HEADS="${N_HEADS:-2}"
LR="${LR:-0.0003}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
DIAGNOSTIC_NODES="${DIAGNOSTIC_NODES:-}"
DIAGNOSTIC_EXAMPLES="${DIAGNOSTIC_EXAMPLES:-0}"
PYTHON="${PYTHON:-python3}"

CONFIGS="${CONFIGS:-direct:- cot:- soft:0 soft:8 soft:16 latent:0 latent:8 latent:16}"

mkdir -p "${OUTPUT_DIR}" "${CHECKPOINT_DIR}"

PYTHONPATH="src:${PYTHONPATH:-}" "${PYTHON}" -m clt.build_dataset \
  --task "${TASK}" \
  --preset debug \
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
    echo "Running ladder point: config=${suffix} seed=${seed} steps=${STEPS}"
    PYTHONUNBUFFERED=1 PYTHONPATH="src:${PYTHONPATH:-}" "${PYTHON}" -m clt.train_tiny \
      --task "${TASK}" \
      --method "${method}" \
      --difficulty "${DIFFICULTY}" \
      --data-dir "${DATA_DIR}" \
      --device "${DEVICE}" \
      --steps "${STEPS}" \
      --eval-examples "${EVAL_EXAMPLES}" \
      --eval-mode binary_choice \
      --lr "${LR}" \
      --seed "${seed}" \
      --d-model "${D_MODEL}" \
      --n-layers "${N_LAYERS}" \
      --n-heads "${N_HEADS}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --easy-graph-diagnostic-nodes "${DIAGNOSTIC_NODES}" \
      --diagnostic-examples "${DIAGNOSTIC_EXAMPLES}" \
      --save-checkpoint "${checkpoint_path}" \
      "${extra_k_args[@]}" \
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
config_order = {}
for i, config in enumerate(os.environ["CONFIGS"].split()):
    method, k = config.split(":", 1)
    name = method if k == "-" else f"{method}_k{k}"
    config_order[name] = i

rows = []
for path in root.glob("*_seed*.json"):
    payload = json.loads(path.read_text())
    config_name, seed_text = path.stem.rsplit("_seed", 1)
    row = {
        "config": config_name,
        "method": payload["method"],
        "k": "" if payload["k"] is None else payload["k"],
        "seed": int(seed_text),
        "steps": payload["steps"],
        "dev": payload["dev"]["accuracy"],
        "id_test": payload["id_test"]["accuracy"],
        "ood_test": payload["ood_test"]["accuracy"],
        "loss": payload["train_loss_last"],
        "elapsed_sec": payload["elapsed_sec"],
        "checkpoint": payload.get("checkpoint_saved") or "",
    }
    for name, metric in payload.get("diagnostics", {}).items():
        row[name] = metric["accuracy"]
    rows.append(row)

rows.sort(key=lambda row: (config_order.get(row["config"], 999), row["seed"]))

def mean(values):
    return sum(values) / len(values)

def std(values):
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))

groups = defaultdict(list)
for row in rows:
    groups[(row["config"], row["method"], row["k"], row["steps"])].append(row)

aggregate = []
for (config, method, k, steps), group in groups.items():
    aggregate.append(
        {
            "config": config,
            "method": method,
            "k": k,
            "steps": steps,
            "n": len(group),
            "dev_mean": mean([row["dev"] for row in group]),
            "dev_std": std([row["dev"] for row in group]),
            "id_test_mean": mean([row["id_test"] for row in group]),
            "id_test_std": std([row["id_test"] for row in group]),
            "ood_test_mean": mean([row["ood_test"] for row in group]),
            "ood_test_std": std([row["ood_test"] for row in group]),
            "loss_mean": mean([row["loss"] for row in group if row["loss"] is not None]),
            "elapsed_sec_mean": mean([row["elapsed_sec"] for row in group]),
        }
    )
aggregate.sort(key=lambda row: config_order.get(row["config"], 999))

print("\nPhase 1d ladder matrix aggregate")
print("config\tn\tid_mean\tid_std\tood_mean\tood_std\tloss_mean\tsec_mean")
for row in aggregate:
    print(
        f"{row['config']}\t"
        f"{row['n']}\t"
        f"{row['id_test_mean']:.3f}\t"
        f"{row['id_test_std']:.3f}\t"
        f"{row['ood_test_mean']:.3f}\t"
        f"{row['ood_test_std']:.3f}\t"
        f"{row['loss_mean']:.3f}\t"
        f"{row['elapsed_sec_mean']:.1f}"
    )

(root / "summary.json").write_text(json.dumps({"runs": rows, "aggregate": aggregate}, indent=2), encoding="utf-8")
for name, payload in [("runs.csv", rows), ("aggregate.csv", aggregate)]:
    fieldnames = sorted({key for row in payload for key in row.keys()})
    preferred = [
        "config",
        "method",
        "k",
        "seed",
        "steps",
        "n",
        "dev",
        "dev_mean",
        "dev_std",
        "id_test",
        "id_test_mean",
        "id_test_std",
        "ood_test",
        "ood_test_mean",
        "ood_test_std",
        "loss",
        "loss_mean",
        "elapsed_sec",
        "elapsed_sec_mean",
        "checkpoint",
    ]
    ordered = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with (root / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(payload)
PY
