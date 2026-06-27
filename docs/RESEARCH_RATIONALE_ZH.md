# FDT 当前研究依据

本文是 Fuzzy Deep Thinker 当前工程实验的研究依据摘要。完整原始研究笔记见 `docs/research/FUZZY_DEEP_THINKER_RESEARCH_NOTE_ZH.md`；本文只保留与当前 repo 已实现或近期可验证实验直接相关的内容。

## 核心问题

复杂任务需要 slow thinking，但 slow thinking 不一定必须用人类可读的自然语言 token 承载。

传统路径：

```text
prompt -> textual thinking tokens -> answer
```

FDT 路径：

```text
prompt -> K continuous thinking steps -> answer
```

输出给人的 answer 仍然是离散 token；中间 thinking steps 可以停留在 embedding 或 hidden-state 空间。

## 当前假设

1. CoT 的收益至少部分来自额外 test-time compute，而不一定来自“自然语言是最优中间计算格式”。
2. Thinking token 离散化会带来表达瓶颈：模型每一步丰富的 hidden state 被压成一个词表项，再查 embedding 表进入下一步。
3. 对 thinking trace 做逐 token CE 监督可能过约束：中间 trace 通常没有唯一正确写法，强制模仿某条 oracle trace 可能只是学习文字格式。
4. Continuous thinking 可能在相同 K 或相似 compute 下取得更好的 accuracy-compute tradeoff。

## 两种离散化损害

当前项目把问题拆成两个轴：

| 轴 | 损害 | 当前实验对应 |
|---|---|---|
| 表示轴 | forward feedback bottleneck：`hidden -> logits -> token id -> embedding` 会丢失连续状态信息 | `soft` 和 `latent` 方法 |
| 监督轴 | thinking trace CE：强制对齐某条离散 trace，可能不必要 | `cot` vs `masked_cot` 对照 |

因此当前五组 SFT 对照不是简单“谁准确率更高”，而是在拆分这两个因素：

| 方法 | 中间计算 | thinking loss | 角色 |
|---|---|---|---|
| `direct` | 无显式 thinking | 无 | 无 thinking baseline |
| `cot` | 自然语言 trace | trace CE + answer CE | 离散 CoT baseline |
| `masked_cot` | 自然语言 trace | only answer CE | 测试 trace CE 是否必要 |
| `soft` | vocab distribution 加权 embedding | only answer CE | 连续化词表 token |
| `latent` | hidden-state feedback | only answer CE | FDT 主实验路径 |

## Soft Token 与 Latent Thought

Soft token 仍然绕不开词表：

```text
hidden state -> vocab logits -> softmax distribution -> weighted token embedding
```

Latent thought 直接绕开词表：

```text
hidden state -> normalization/projection -> next input embedding
```

所以 soft token 是“词表空间的连续松弛”，latent thought 是“hidden-state 空间的连续思考”。二者都可能比离散 token 更平滑，但 latent thought 更接近核心假设，也更容易出现 drift 或训练不稳定。

## 当前工程约束

当前 repo 的第一阶段刻意不引入 RL，把问题限制在 SFT / supervised auxiliary losses 内：

- 使用同一 base model；
- 使用同一批 synthetic tasks；
- 使用相同 final answer 格式；
- 使用 deterministic evaluator；
- 优先 fixed K，不先做 adaptive thinking；
- 先证明模型能在可控任务上学到稳定信号，再扩大到更复杂任务。

RL、`thinking_status` 自适应门控、ELF 式 flow head 等仍然保留为 future work，不作为当前 Qwen LoRA 长训练的结论前提。

## 当前验证路径

当前 Qwen 实验使用 `Qwen/Qwen3-0.6B-Base` + LoRA，在 `graph_reachability` 上验证：

1. `easy_ladder`：先确认 pipeline 可运行、基础模型能否快速学到简单任务。
2. `hard_ladder`：扩大节点数和路径长度，测试是否学到超过标签偏置的图搜索能力。
3. 48h long run：在 balanced answer sampling、训练期 probe、多 K continuous thinking 下观察是否能稳定超过 direct / masked_cot baseline。

当前活跃长训练记录：

```text
docs/ACTIVE_EXPERIMENTS.md
docs/QWEN_48H_HARD_LADDER_RUN_20260627_ZH.md
```

## 成功标准

支持 FDT 假设的最低标准：

- `latent` 或 `soft` 在 dev/id/OOD 上稳定超过 `direct` 和相近 compute 的离散 baseline；
- 多 seed 下不是单次偶然；
- `prediction_counts` 不再塌缩成全 YES 或全 NO；
- diagnostics 按 answer、num_nodes、path_length 分组后没有明显只靠标签偏置获胜；
- 在相同或更低 K / latency / wall-clock 下达到更高 accuracy。

若 continuous methods 只降低 loss，但最终预测仍标签坍缩，则只能说明训练目标被优化，不能说明模型学会了任务。

## 文档关系

- 完整理论草案：`docs/research/FUZZY_DEEP_THINKER_RESEARCH_NOTE_ZH.md`
- 文献笔记：`docs/LITERATURE_NOTES_ZH.md`
- 当前实验入口：`README.md` / `README.zh-CN.md`
- Qwen 实现说明：`docs/QWEN_INTEGRATION.md`
- 活跃实验记录：`docs/ACTIVE_EXPERIMENTS.md`
