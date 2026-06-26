#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-graph_reachability}"
DIFFICULTY="${DIFFICULTY:-easy}"
DATA_DIR="${DATA_DIR:-data/phase1b_easy}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/phase1b_direct_curve}"
DEVICE="${DEVICE:-cpu}"
STEPS_LIST="${STEPS_LIST:-100 300 1000}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-200}"
D_MODEL="${D_MODEL:-32}"
N_LAYERS="${N_LAYERS:-1}"
N_HEADS="${N_HEADS:-2}"
LR="${LR:-0.0003}"

mkdir -p "${OUTPUT_DIR}"

PYTHONPATH="src:${PYTHONPATH:-}" python3 -m clt.build_dataset \
  --task "${TASK}" \
  --preset debug \
  --difficulty "${DIFFICULTY}" \
  --out-dir "${DATA_DIR}"

for steps in ${STEPS_LIST}; do
  echo "Running direct curve point: steps=${steps}"
  PYTHONUNBUFFERED=1 PYTHONPATH="src:${PYTHONPATH:-}" python3 -m clt.train_tiny \
    --task "${TASK}" \
    --method direct \
    --difficulty "${DIFFICULTY}" \
    --data-dir "${DATA_DIR}" \
    --device "${DEVICE}" \
    --steps "${steps}" \
    --eval-examples "${EVAL_EXAMPLES}" \
    --eval-mode binary_choice \
    --lr "${LR}" \
    --d-model "${D_MODEL}" \
    --n-layers "${N_LAYERS}" \
    --n-heads "${N_HEADS}" \
    --max-new-tokens 8 \
    --output "${OUTPUT_DIR}/direct_${steps}.json" \
    2>&1 | tee "${OUTPUT_DIR}/direct_${steps}.log"
done

OUTPUT_DIR="${OUTPUT_DIR}" PYTHONPATH="src:${PYTHONPATH:-}" python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_DIR"])
rows = []
for path in sorted(root.glob("direct_*.json"), key=lambda p: int(p.stem.split("_")[1])):
    payload = json.loads(path.read_text())
    rows.append(
        {
            "steps": payload["steps"],
            "dev": payload["dev"]["accuracy"],
            "id_test": payload["id_test"]["accuracy"],
            "ood_test": payload["ood_test"]["accuracy"],
            "loss": payload["train_loss_last"],
            "elapsed_sec": payload["elapsed_sec"],
        }
    )

print("\nPhase 1b direct learning curve")
print("steps\tdev\tid_test\tood_test\tloss\tsec")
for row in rows:
    print(
        f"{row['steps']}\t"
        f"{row['dev']:.3f}\t"
        f"{row['id_test']:.3f}\t"
        f"{row['ood_test']:.3f}\t"
        f"{row['loss']:.3f}\t"
        f"{row['elapsed_sec']:.1f}"
    )

(root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
PY
