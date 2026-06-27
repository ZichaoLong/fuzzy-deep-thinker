# Phase 3 诊断与复验报告

日期：2026-06-27

## 摘要

本轮完成三件事：

1. 项目正式改名为 **Fuzzy Deep Thinker (FDT)**，本地目录改为 `~/llm/fuzzy-deep-thinker`，GitHub remote 指向 `git@github.com:ZichaoLong/fuzzy-deep-thinker.git`。
2. 按 Phase 2 报告建议，为评估脚本增加按 metadata 分组诊断，并在 JSON 结果中保留成功/失败案例。
3. 复查两个关键问题：
   - `pointer_chasing` 到底是按 label 坍缩，还是按 depth 有规律失败？
   - Phase 2 中 `cot` 在 graph ladder 上的 OOD 提升是否可重复？

结论：

- 标准 `pointer_chasing` 的失败主要是 label 坍缩：多数方法倾向预测 `NO`，depth 维度没有明显算法学习迹象。
- 三版简化 pointer prompt 仍未让 direct baseline 学起来，当前 tiny char decoder 不适合把 pointer chasing 作为主验证任务。
- graph ladder 复验中，`cot` 的 OOD mean 仍高于 direct，但差距从 Phase 2 的 `0.613 vs 0.500` 降到 `0.546 vs 0.494`，且方差仍大；这个信号只能算“值得继续复查”，不能作为稳定结论。

## 缩写说明

| 缩写 | 全称 | 含义 |
|---|---|---|
| FDT | Fuzzy Deep Thinker | 本项目名，研究连续空间慢思考 |
| CoT | Chain of Thought | 自然语言链式思考 |
| SFT | Supervised Fine-Tuning | 监督微调 |
| CE | Cross Entropy | 交叉熵损失 |
| ID | In-Distribution | 与训练分布同范围的测试 |
| OOD | Out-of-Distribution | 超出训练分布的测试 |
| NPU | Neural Processing Unit | 本机使用 Ascend 910 NPU |
| K | latent/soft thinking steps | 连续思考步数 |
| soft token | soft vocabulary embedding | 用 vocab 概率分布加权 embedding 作为下一步输入 |
| latent thought | hidden-state feedback | 绕过 vocab，把 hidden state 投影后作为下一步输入 |

## 本轮代码变更

训练脚本新增：

- `--diagnostic-metadata-keys`: 按 metadata 字段分组评估，例如 `answer,depth` 或 `answer,num_nodes`。
- `--case-examples`: 每个 eval 结果保留若干成功/失败案例。

每个 run 的 JSON 现在包含：

```text
dev / id_test / ood_test:
  accuracy
  samples
  cases:
    success
    failure

diagnostics:
  id_test_answer_YES
  id_test_answer_NO
  id_test_depth_2
  ...
```

runner 的 `aggregate.csv` 也会聚合这些诊断字段的 mean/std。

## 实验 A：标准 Pointer Chasing 诊断

目的：确认 Phase 2 中 pointer 任务接近随机，是因为模型没有学会算法，还是因为某些 depth/label 子集失败。

配置：

| 项目 | 值 |
|---|---|
| task | `pointer_chasing` |
| difficulty | `standard` |
| configs | `direct`, `cot`, `soft_k0`, `soft_k4`, `soft_k8`, `latent_k0`, `latent_k4`, `latent_k8` |
| seeds | `0,1,2` |
| steps | `1000` |
| eval examples | `100` |
| diagnostics | `answer,depth` |
| output | `outputs/phase3_pointer_diagnostics_npu/` |

### 总体统计

| config | ID mean | OOD mean | sec mean |
|---|---:|---:|---:|
| direct | 0.503 | 0.493 | 3.5 |
| cot | 0.503 | 0.493 | 3.6 |
| soft_k0 | 0.503 | 0.493 | 3.5 |
| soft_k4 | 0.503 | 0.493 | 11.1 |
| soft_k8 | 0.503 | 0.493 | 18.4 |
| latent_k0 | 0.503 | 0.493 | 3.6 |
| latent_k4 | 0.503 | 0.493 | 11.0 |
| latent_k8 | 0.503 | 0.493 | 18.6 |

### 分组诊断

| config | ID NO | ID YES | OOD NO | OOD YES | ID depth 2 | ID depth 3 | ID depth 4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct | 0.667 | 0.333 | 0.667 | 0.333 | 0.505 | 0.505 | 0.500 |
| cot | 0.706 | 0.293 | 0.750 | 0.256 | 0.505 | 0.505 | 0.500 |
| soft_k8 | 0.667 | 0.333 | 0.667 | 0.333 | 0.505 | 0.505 | 0.500 |
| latent_k8 | 0.667 | 0.333 | 0.667 | 0.333 | 0.505 | 0.505 | 0.500 |

解读：

- depth 维度几乎完全随机，说明模型没有学到“走 T 步状态转移”。
- label 维度明显不均衡：`NO` 子集高、`YES` 子集低，说明模型偏向输出 `NO`。
- CoT 的 generated trace 多数不可读或与题目不一致，没有提供可靠推理过程。

### 直观案例

成功案例：direct 正确预测 `NO`。

```text
Rules: A->K, B->E, C->H, D->F, E->H, F->D, G->F, H->A, I->G, J->C, K->G, L->F
Start: A
Steps: 2
Question: after exactly 2 transitions, are you at K?

Expected: NO
Predicted: NO
Scores: YES=0.293, NO=0.185
```

失败案例：direct 把真实 `YES` 判成 `NO`。

```text
Rules: A->K, B->D, C->G, D->E, E->E, F->B, G->G, H->D, I->K, J->C, K->B, L->A
Start: I
Steps: 4
Question: after exactly 4 transitions, are you at E?
Path result: E

Expected: YES
Predicted: NO
Scores: YES=0.294, NO=0.184
```

CoT 失败案例：trace 本身已失真。

```text
Expected: YES
Predicted: NO
Generated trace:
Start at A. Step 1: A->C. Step 2: G->H. Step 3: steps the state is J. ...
```

这个案例说明：当前 tiny 模型的 CoT 不是可靠的可读推理，只是生成了形式相似但语义错误的文本。

## 实验 B：简化 Pointer Chasing 尝试

目的：按 Phase 2 建议降低 pointer 任务难度，让 direct baseline 先能在 ID 上学到 90% 以上，再用它测试 latent steps。

我们尝试了三版 simple prompt：

1. `simple`: 4 个状态，ID depth=1，短 prompt。
2. `simple + Next`: 增加第一步结果 `Next: X`。
3. `simple + Compare`: 增加同一行比较 `Compare: X=Y`。

结果：

| variant | model | steps | ID mean | OOD mean | 结论 |
|---|---|---:|---:|---:|---|
| simple | d32/l1 | 3000 | 0.500 | 0.500 | 未学会 |
| simple + Next | d32/l1 | 3000 | 0.500 | 0.500 | 未学会 |
| simple + Next | d64/l2 | 5000 | 0.500 | 0.500 | 未学会 |
| simple + Compare | d32/l1 | 3000 | 0.515 | 0.497 | 基本未学会 |

代表失败案例：

```text
Rules: A->C, B->D, C->B, D->A
Start: A
Next: C
Steps: 3
Target: D
Compare: C=D

Expected: YES
Predicted: NO
Scores: YES=0.227, NO=0.181
```

解读：

- 即使把 ID 任务降到 “depth=1 + Compare 同行比较”，tiny char decoder 仍然没有可靠学会。
- 因此 pointer chasing 目前不适合继续作为 FDT 的主要验证任务。
- 这不是 continuous thinking 的负证据，而是任务/模型组合不合格：direct baseline 都没有先学起来。

下一步若继续 pointer 方向，需要换更合适的建模方式，例如：

- 使用 token-level 而不是 char-level tokenizer；
- 使用更大的 base LM；
- 或设计更适合 tiny char decoder 的算法任务。

## 实验 C：Graph Ladder CoT 信号复验

目的：Phase 2 中 `cot` 在 graph ladder 上 OOD mean 达到 0.613，高于 direct 的 0.500。本轮增加 seeds，并按 `answer,num_nodes` 分组复验。

配置：

| 项目 | 值 |
|---|---|
| task | `graph_reachability` |
| difficulty | `easy_ladder` |
| configs | `direct`, `cot` |
| seeds | `0..7` |
| steps | `3000` |
| eval examples | `100` |
| model | `d_model=64`, `n_layers=2`, `n_heads=4` |
| diagnostics | `answer,num_nodes` |
| output | `outputs/phase3_ladder_cot_recheck_fast_npu/` |

### 总体统计

| config | seeds | ID mean | ID std | OOD mean | OOD std | sec mean |
|---|---:|---:|---:|---:|---:|---:|
| direct | 8 | 1.000 | 0.000 | 0.494 | 0.018 | 14.3 |
| cot | 8 | 0.718 | 0.150 | 0.546 | 0.131 | 14.3 |

### 分组诊断

| config | OOD NO | OOD YES | OOD n=7 | OOD n=8 |
|---|---:|---:|---:|---:|
| direct | 0.988 | 0.000 | 0.500 | 0.488 |
| cot | 0.740 | 0.353 | 0.510 | 0.583 |

解读：

- direct 在 ID 上完全拟合，但 OOD 几乎变成“预测 NO”：`OOD YES = 0.000`。
- CoT 的 OOD mean 高于 direct，主要来自它对 `YES` 样本不再完全失败：`OOD YES = 0.353`。
- 但 CoT 的 ID 显著下降，且 OOD std 仍高；它不是稳定优势，只是一个需要进一步复查的信号。
- 按图规模看，CoT 在 `n=8` 上好于 direct，但该提升是否来自真实推理、trace 噪声还是评分偏置，还不能下结论。

### 直观案例

direct 的 ID 成功案例：

```text
Nodes: A, B, C, D, E, F
Edges: A->B, B->F, E->D
Question: Is there a path from A to F?

Expected: YES
Predicted: YES
Scores: YES=0.002, NO=2.358
```

direct 的 OOD 失败案例：

```text
Nodes: A, B, C, D, E, F, G
Edges: A->E, D->B, E->G, F->D
Question: Is there a path from A to G?

Expected: YES
Predicted: NO
Scores: YES=1.672, NO=0.002
```

cot 的 OOD 成功案例：

```text
Nodes: A, B, C, D, E, F, G
Edges: A->E, D->B, E->G, F->D
Question: Is there a path from A to G?

Expected: YES
Predicted: YES
Generated trace: "St A. St A. Visit Visis E ..."
Scores: YES=0.013, NO=2.822
```

cot 的 OOD 失败案例：

```text
Nodes: A, B, C, D, E, F, G, H
Edges: A->H, E->F, F->G
Question: Is there a path from A to H?

Expected: YES
Predicted: NO
Generated trace: "Start at A. Visited nodes: A. F is not reached."
Scores: YES=1.475, NO=0.007
```

这个失败案例很关键：图里明明有 `A->H`，但模型生成了 “F is not reached”。这说明当前 CoT trace 仍不可靠，不能直接解释为真实推理。

## 当前判断

1. FDT 的训练和诊断管线已经更完整：不仅有均值统计，还有分组统计与样例。
2. `pointer_chasing` 暂时应降级为“失败诊断任务”，不适合继续作为主实验。
3. graph ladder 中 direct 的 OOD 失败模式非常清楚：几乎所有 OOD YES 都失败。
4. CoT 在 graph ladder 上确实缓解了一部分 OOD YES 失败，但代价是 ID 下降和高方差。
5. 本轮仍没有看到 `soft/latent` continuous thinking 的正向证据；下一步应该先换任务或换 tokenizer/model，而不是继续在当前 pointer 任务上堆 K。

## 建议下一步

优先级从高到低：

1. 放弃当前 char-level pointer chasing 作为主任务，改用 token-level tokenizer 或更大 base LM 复查。
2. 继续分析 graph ladder 的 OOD YES 失败，设计一个更平滑的图任务，使 direct baseline 不至于靠表面模式在 ID 上过拟合。
3. 若继续 synthetic tiny 路线，任务必须满足两个条件：direct ID 可学到 90% 以上，OOD 又确实需要更多计算步数。
4. 报告和后续实验默认都保留成功/失败案例，避免只看 aggregate 造成误判。
