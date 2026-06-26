# Phase 2 Dual-Track Experiments

Date: 2026-06-27

Phase 1d did not show a benefit from continuous latent steps in the tiny graph-ladder setup. Phase 2 pushes two directions at the same time:

1. Scale the existing graph ladder by increasing model capacity and training steps.
2. Add a cleaner algorithmic-depth task where each latent step can plausibly represent one state update.

## Track A: Larger Graph Ladder

Runner:

```bash
scripts/run_phase2a_ladder_scale.sh
```

Defaults:

| setting | value |
|---|---|
| task | `graph_reachability` |
| difficulty | `easy_ladder` |
| configs | `direct:- cot:- soft:8 latent:8` |
| seeds | `0 1 2` |
| steps | `3000` |
| eval examples | `200` |
| model | `d_model=64`, `n_layers=2`, `n_heads=4` |

Use on the Ascend NPU:

```bash
DEVICE=npu:0 scripts/with_conda_npu.sh scripts/run_phase2a_ladder_scale.sh
```

Purpose: test whether Phase 1d was limited mainly by tiny model capacity or too short a training budget.

## Track B: Pointer Chasing

New task: `pointer_chasing`

The model receives deterministic transition rules such as `A->C, B->F, ...`, a start state, a number of steps, and a target state. It must answer whether following the transition rule for exactly `T` steps ends at the target.

Split design:

| split | depths | states | labels |
|---|---|---:|---|
| `train` / `dev` / `id_test` | `2,3,4` | 12 | balanced `YES` / `NO` by seed pattern |
| `ood_test` | `5,6,7,8` | 12 | balanced `YES` / `NO` by seed pattern |

Runner:

```bash
scripts/run_phase2b_pointer_matrix.sh
```

Defaults:

| setting | value |
|---|---|
| task | `pointer_chasing` |
| difficulty | `standard` |
| configs | `direct:- cot:- soft:0 soft:4 soft:8 latent:0 latent:4 latent:8` |
| seeds | `0 1 2` |
| steps | `1000` |
| eval examples | `100` |
| model | `d_model=32`, `n_layers=1`, `n_heads=2` |

Use on the Ascend NPU:

```bash
DEVICE=npu:0 scripts/with_conda_npu.sh scripts/run_phase2b_pointer_matrix.sh
```

Purpose: make the compute-depth hypothesis sharper. If latent steps are useful, this task is a better place to look than the previous graph-size OOD setup because the required computation is an explicit repeated state transition.

## Status

NPU smoke checks passed for both tracks:

- Track A: `direct` and `latent_k2`, 1 seed, 5 training steps.
- Track B: `direct` and `latent_k2`, 1 seed, 5 training steps.

These smoke results only verify plumbing. They are not evidence for or against the research hypothesis.
