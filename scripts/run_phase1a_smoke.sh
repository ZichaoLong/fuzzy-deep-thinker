#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-graph_reachability}"
DATA_DIR="${DATA_DIR:-data/phase1a_smoke}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/phase1a_smoke}"
DEVICE="${DEVICE:-cpu}"
STEPS="${STEPS:-20}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-8}"
K="${K:-4}"
D_MODEL="${D_MODEL:-48}"
N_LAYERS="${N_LAYERS:-1}"
N_HEADS="${N_HEADS:-2}"

mkdir -p "${OUTPUT_DIR}"

PYTHONPATH="src:${PYTHONPATH:-}" python3 -m clt.build_dataset \
  --task "${TASK}" \
  --preset smoke \
  --out-dir "${DATA_DIR}"

for method in direct cot latent; do
  PYTHONPATH="src:${PYTHONPATH:-}" python3 -m clt.train_tiny \
    --task "${TASK}" \
    --method "${method}" \
    --data-dir "${DATA_DIR}" \
    --device "${DEVICE}" \
    --steps "${STEPS}" \
    --eval-examples "${EVAL_EXAMPLES}" \
    --k "${K}" \
    --d-model "${D_MODEL}" \
    --n-layers "${N_LAYERS}" \
    --n-heads "${N_HEADS}" \
    --max-new-tokens 32 \
    --output "${OUTPUT_DIR}/${method}.json"
done

PYTHONPATH="src:${PYTHONPATH:-}" python3 - <<'PY'
import json
from pathlib import Path

root = Path("outputs/phase1a_smoke")
print("\nPhase 1a smoke summary")
print("method\tdev\tid_test\tood_test\tloss")
for path in sorted(root.glob("*.json")):
    payload = json.loads(path.read_text())
    print(
        f"{payload['method']}\t"
        f"{payload['dev']['accuracy']:.3f}\t"
        f"{payload['id_test']['accuracy']:.3f}\t"
        f"{payload['ood_test']['accuracy']:.3f}\t"
        f"{payload['train_loss_last']:.3f}"
    )
PY
