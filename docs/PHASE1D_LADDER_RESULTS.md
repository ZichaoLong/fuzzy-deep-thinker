# Phase 1d Graph Ladder Results

Date: 2026-06-26

Environment:

- Wrapper: `scripts/with_conda_npu.sh`
- Python: `/home/zlong/anaconda3/envs/clt-npu-py39/bin/python`
- Toolkit path: `/usr/local/Ascend/ascend-toolkit/8.2.RC1`
- Device: `ASCEND_RT_VISIBLE_DEVICES=5`, `DEVICE=npu:0`
- Task: `graph_reachability`
- Difficulty: `easy_ladder`
- Model: `d_model=32`, `n_layers=1`, `n_heads=2`

## Purpose

Phase 1c showed that the original `easy` graph task had a sharp ID/OOD artifact: training examples used a very small graph while OOD examples jumped to larger graphs. Phase 1d replaces that with a smoother graph-size ladder so method comparisons are less dominated by one distribution discontinuity.

The new `easy_ladder` split uses:

| split | graph sizes | labels |
|---|---|---|
| `train` / `dev` / `id_test` | `n=4,5,6` | balanced `YES` / `NO` by seed pattern |
| `ood_test` | `n=7,8` | balanced `YES` / `NO` by seed pattern |

The run also adds checkpoint support to `train_tiny.py`:

- `--save-checkpoint PATH`
- `--load-checkpoint PATH`
- `--eval-only`

This lets follow-up diagnostics reuse the same trained models instead of retraining every time.

## Command

```bash
DEVICE=npu:0 \
DATA_DIR=data/phase1d_ladder_npu \
OUTPUT_DIR=outputs/phase1d_ladder_matrix_npu \
SEEDS='0 1 2' \
STEPS=1000 \
EVAL_EXAMPLES=100 \
scripts/with_conda_npu.sh \
scripts/run_phase1d_ladder_matrix.sh
```

Configs:

```text
direct:- cot:- soft:0 soft:8 soft:16 latent:0 latent:8 latent:16
```

All configs used the unified `binary_choice` evaluator. `soft_k0` and `latent_k0` are expected to match the direct answer path, because they run zero continuous recurrent steps before answer scoring.

## Aggregate Results

| config | seeds | id_mean | id_std | ood_mean | ood_std | loss_mean | sec_mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct | 3 | 0.557 | 0.072 | 0.510 | 0.026 | 0.231 | 3.6 |
| cot | 3 | 0.517 | 0.046 | 0.533 | 0.035 | 1.004 | 3.6 |
| soft_k0 | 3 | 0.557 | 0.072 | 0.510 | 0.026 | 0.231 | 3.6 |
| soft_k8 | 3 | 0.520 | 0.017 | 0.483 | 0.127 | 0.272 | 19.0 |
| soft_k16 | 3 | 0.523 | 0.015 | 0.457 | 0.012 | 0.289 | 34.9 |
| latent_k0 | 3 | 0.557 | 0.072 | 0.510 | 0.026 | 0.231 | 3.6 |
| latent_k8 | 3 | 0.537 | 0.046 | 0.487 | 0.061 | 0.224 | 18.6 |
| latent_k16 | 3 | 0.563 | 0.061 | 0.480 | 0.030 | 0.236 | 34.2 |

Raw outputs are under `outputs/phase1d_ladder_matrix_npu/` in the local workspace:

- `runs.csv`
- `aggregate.csv`
- `summary.json`
- per-run `.json` and `.log` files
- checkpoints under `checkpoints/`

## Readout

What this result supports:

1. The smoother ladder removes the most obvious Phase 1c graph-size cliff, but the current tiny model and 1000-step budget still leave the task near chance.
2. In this setting, continuous thinking has not produced a reliable gain over the direct baseline.
3. Runtime grows roughly linearly with `K`: `K=8` is about 5x slower than direct, and `K=16` is about 10x slower.
4. `soft_k0` and `latent_k0` reproduce direct behavior, which is a useful implementation sanity check.

What this result does not prove:

1. It does not prove that continuous latent thinking is ineffective in general.
2. It does not yet test a model large enough, a training budget long enough, or a task hard enough to require iterative latent computation.
3. It does not isolate whether the bottleneck is representation, optimization, data scale, or the graph task design.

The next useful step is either to increase capacity/training budget on this ladder task, or to switch to a synthetic task with a clearer algorithmic-depth requirement and a less brittle prompt surface.
