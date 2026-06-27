# Phase 2 双线实验结果报告

日期：2026-06-27

## 摘要

本轮同时推进了两个方向：

1. 在原有 `graph_reachability/easy_ladder` 任务上扩大模型容量和训练步数，验证 Phase 1d 是否只是模型太小或训练不充分。
2. 新增 `pointer_chasing` 任务，构造一个更直接需要多步状态更新的算法任务，验证 latent steps 是否能带来收益。

当前结果没有观察到 `soft` 或 `latent` continuous thinking 相比 direct baseline 的稳定收益。扩大后的 graph ladder 能把 ID 拟合到 100%，但 OOD 仍接近随机；pointer chasing 目前所有方法基本停在随机水平。

## 实验环境

| 项目 | 配置 |
|---|---|
| 机器 | 本机 Ascend 910 NPU |
| 执行封装 | `scripts/with_conda_npu.sh` |
| Python | `/home/zlong/anaconda3/envs/clt-npu-py39/bin/python` |
| CANN | `/usr/local/Ascend/ascend-toolkit/8.2.RC1` |
| PyTorch | `torch 2.7.1`, `torch_npu 2.7.1.post2` |
| 设备 | `DEVICE=npu:0`, `ASCEND_RT_VISIBLE_DEVICES=5` |

完整运行日志：

```text
outputs/phase2_background/tmux_dual_track_20260627_005518.log
```

## 实验 A：Pointer Chasing

### 任务设置

任务输入是一组确定性状态转移规则，例如 `A->C, B->F, ...`，以及起点、步数和目标状态。模型需要判断：从起点开始，严格执行给定步数的状态转移后，是否到达目标状态。

| split | depth | 状态数 | 标签 |
|---|---:|---:|---|
| `train/dev/id_test` | `2,3,4` | 12 | YES/NO 基本平衡 |
| `ood_test` | `5,6,7,8` | 12 | YES/NO 基本平衡 |

实验配置：

| 项目 | 值 |
|---|---|
| runner | `scripts/run_phase2b_pointer_matrix.sh` |
| seeds | `0,1,2` |
| steps | `1000` |
| eval examples | `100` |
| model | `d_model=32`, `n_layers=1`, `n_heads=2` |
| eval mode | `binary_choice` |

原始结果：

```text
outputs/phase2b_pointer_matrix_npu/aggregate.csv
```

### 聚合结果

| config | ID mean | ID std | OOD mean | OOD std | loss mean | sec mean |
|---|---:|---:|---:|---:|---:|---:|
| direct | 0.503 | 0.012 | 0.493 | 0.023 | 0.197 | 3.4 |
| cot | 0.503 | 0.012 | 0.493 | 0.023 | 0.452 | 3.5 |
| soft_k0 | 0.503 | 0.012 | 0.493 | 0.023 | 0.197 | 3.5 |
| soft_k4 | 0.503 | 0.012 | 0.493 | 0.023 | 0.193 | 10.9 |
| soft_k8 | 0.503 | 0.012 | 0.493 | 0.023 | 0.195 | 18.7 |
| latent_k0 | 0.503 | 0.012 | 0.493 | 0.023 | 0.197 | 3.5 |
| latent_k4 | 0.503 | 0.012 | 0.493 | 0.023 | 0.193 | 11.1 |
| latent_k8 | 0.503 | 0.012 | 0.493 | 0.023 | 0.194 | 18.5 |

### 观察

1. 所有方法在 ID 和 OOD 上都接近 50%，说明当前 tiny model 没有学会 pointer chasing。
2. `soft_k4/k8` 和 `latent_k4/k8` 的 loss 略低，但 accuracy 没有提升，说明 loss 改善没有转化为正确的二分类决策。
3. `K=8` 的连续方法运行时间约为 direct 的 5.4 倍，但没有带来收益。
4. `soft_k0`、`latent_k0` 与 direct 结果一致，说明 zero latent step 的实现路径是合理的 sanity check。

当前结论：这版 pointer chasing 任务还不能作为有效验证 continuous latent thinking 的主任务。它更像是暴露了一个训练/任务设计问题：模型可能学到了回答先验或 candidate scoring 偏置，而不是学到状态转移算法。

## 实验 B：Graph Ladder Scale

### 任务设置

该实验沿用 Phase 1d 的 `graph_reachability/easy_ladder`：

| split | 图规模 | 标签 |
|---|---|---|
| `train/dev/id_test` | `n=4,5,6` | YES/NO 平衡 |
| `ood_test` | `n=7,8` | YES/NO 平衡 |

相对于 Phase 1d，本轮扩大模型和训练预算：

| 项目 | Phase 1d | Phase 2 |
|---|---:|---:|
| steps | 1000 | 3000 |
| d_model | 32 | 64 |
| n_layers | 1 | 2 |
| n_heads | 2 | 4 |
| eval examples | 100 | 200 |

实验配置：

| 项目 | 值 |
|---|---|
| runner | `scripts/run_phase2a_ladder_scale.sh` |
| seeds | `0,1,2` |
| configs | `direct`, `cot`, `soft_k8`, `latent_k8` |
| eval mode | `binary_choice` |

原始结果：

```text
outputs/phase2a_ladder_scale_npu/aggregate.csv
```

### 聚合结果

| config | ID mean | ID std | OOD mean | OOD std | loss mean | sec mean |
|---|---:|---:|---:|---:|---:|---:|
| direct | 1.000 | 0.000 | 0.500 | 0.000 | 0.002 | 14.1 |
| cot | 0.737 | 0.224 | 0.613 | 0.120 | 0.015 | 13.6 |
| soft_k8 | 0.577 | 0.099 | 0.455 | 0.035 | 0.249 | 91.3 |
| latent_k8 | 1.000 | 0.000 | 0.468 | 0.055 | 0.002 | 88.3 |

### 观察

1. 扩容和更长训练后，direct 和 latent_k8 都可以把 ID accuracy 拟合到 1.0。
2. direct 的 OOD 仍为 0.500，latent_k8 的 OOD 为 0.468，二者都没有表现出图规模外推能力。
3. soft_k8 在 ID 和 OOD 上都较弱，说明当前 soft token 路径在该设置下优化更困难。
4. cot 的 OOD mean 最高，为 0.613，但 ID mean 只有 0.737，且 OOD std 为 0.120，说明这个信号有较大随机性，需要进一步复查。
5. continuous 方法成本明显更高：`latent_k8` 约为 direct 的 6.2 倍，`soft_k8` 约为 direct 的 6.5 倍。

当前结论：扩大模型和训练预算可以解决 ID 拟合问题，但没有解决 OOD generalization。continuous latent steps 在这轮 graph ladder scale 中仍未体现出稳定收益。

## 总体结论

当前实验可以支持以下判断：

1. `soft/latent` continuous thinking 路径已经能在 Ascend NPU 上稳定训练、保存 checkpoint、生成聚合结果。
2. `K=0` 对照与 direct 基本一致，说明实验实现的基础 sanity check 通过。
3. 在当前 tiny/small model 与 synthetic task 设置下，增加 continuous latent steps 会显著增加运行时间，但没有带来稳定 accuracy 提升。
4. graph ladder 的主要问题是 ID 可拟合但 OOD 不外推。
5. pointer chasing 的主要问题是任务还没有被当前训练设置学起来，因此不能直接用于判断 latent thinking 是否有效。

当前实验不能证明：

1. 不能证明 continuous latent thinking 整体方向无效。
2. 不能证明 soft token 或 latent thought 在更大模型、更强优化、更合适任务上没有收益。
3. 不能证明 CoT 在 graph ladder 上确实优于 direct；当前 cot 的 OOD 高一些，但方差较大且 ID 表现不稳定。

## 建议下一步

优先做三个小而明确的修正：

1. 给 `pointer_chasing` 增加按 label 和 depth 的诊断评估，确认模型是在学任务，还是坍缩到固定 YES/NO 倾向。
2. 调整 pointer 任务格式，降低字符串解析负担，例如减少状态数、固定规则顺序、加入更短 prompt，先确保 direct baseline 能在 ID 上超过 90%。
3. 对 graph ladder 的 cot 信号做复验：增加 seeds，记录按图规模 `n=4..8` 的 accuracy，确认 0.613 OOD mean 是否可重复。

如果目标是继续验证“latent step 是否能代表 slow thinking step”，下一轮应先把任务做成 direct baseline 可以学习、OOD 需要更多步数才可能外推的形式。否则连续 latent 路径即使没有收益，也难以判断是方法问题还是任务/优化问题。
