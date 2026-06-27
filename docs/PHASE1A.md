# Phase 1a Vertical Slice

Phase 1a runs a minimal end-to-end loop:

```text
synthetic data -> training format -> tiny decoder training -> dev/ID/OOD evaluation
```

The goal is not to prove the FDT hypothesis yet. The goal is to verify that the experiment pipeline is executable and that Direct/CoT/Latent variants use the intended loss surfaces.

## Build Smoke Splits

```bash
PYTHONPATH=src python3 -m clt.build_dataset \
  --task graph_reachability \
  --preset smoke \
  --out-dir data/phase1a_smoke
```

This writes:

```text
train:    128 examples
dev:       32 examples
id_test:   32 examples
ood_test:  32 examples
```

## Run One Method

```bash
PYTHONPATH=src python3 -m clt.train_tiny \
  --task graph_reachability \
  --method latent \
  --data-dir data/phase1a_smoke \
  --device cpu \
  --steps 20 \
  --eval-examples 8 \
  --k 4 \
  --d-model 48 \
  --n-layers 1 \
  --n-heads 2
```

Supported methods:

```text
direct
cot
masked_cot
soft
latent
```

## Run Smoke Matrix

```bash
scripts/run_phase1a_smoke.sh
```

This runs:

```text
direct
cot
latent
```

and writes JSON metrics to:

```text
outputs/phase1a_smoke/
```

The default smoke settings are intentionally tiny:

```text
steps=20
eval_examples=8
d_model=48
n_layers=1
```

With these settings, generated answers are usually still random and accuracy can be 0. This is expected. The smoke matrix verifies that the data pipeline, loss construction, continuous-step forward pass, generation, parsing, and metric export all run end to end.

Example smoke summary:

```text
method  dev    id_test  ood_test  loss
cot     0.000  0.000    0.000     3.962
direct  0.000  0.000    0.000     3.595
latent  0.000  0.000    0.000     3.739
```

Use longer runs for any accuracy claim.

## Ascend NPU Note

The tiny decoder can execute on NPU:

```bash
ASCEND_RT_VISIBLE_DEVICES=5 PYTHONPATH=src scripts/with_ascend_env.sh \
  python3 -m clt.train_tiny \
  --task graph_reachability \
  --method direct \
  --data-dir data/phase1a_smoke \
  --device npu:0 \
  --steps 1 \
  --eval-examples 1
```

On this machine, `torch_npu` currently falls back to CPU for `aten::_transformer_encoder_layer_fwd`, so this smoke confirms execution compatibility rather than accelerator throughput.
