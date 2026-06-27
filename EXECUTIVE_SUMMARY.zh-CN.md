# Fuzzy Deep Thinker

> slow thinking 仍然保留，但 thinking phase 是否必须用自然语言 token 表达？

传统路径：

```text
prompt -> textual thinking tokens -> answer
```

FDT 路径：

```text
prompt -> K continuous thinking steps -> answer
```

只有最终 answer 需要 decode 成人类可读文字，中间 thinking steps 可以停留在 embedding 或 hidden-state 空间。

## 核心假设

复杂任务需要 test-time compute，但这些 compute 不一定要通过自然语言 CoT 承载。连续 latent thinking 可能带来三类收益：

- 更少 thinking steps 表达更丰富的中间状态；
- 减少长 CoT 的输出 token 成本；
- 在相同 compute budget 下提高最终答案准确率或 OOD 泛化。

## 实验设计

第一阶段只做 SFT 和 supervised auxiliary losses，不引入 RL，避免 reward design 和训练不稳定性干扰结论。

基础模型：

```text
debug: Qwen3-0.6B-Base
main:  Qwen3-1.7B-Base
```

优先使用 Base 模型，避免 Instruct 模型已有 thinking style 污染机制对比。

所有方法使用同一批训练题、同样最终答案格式和同样评估器，只改变“思考载体”：

| 方法 | 中间计算形式 | 训练序列 | Loss | 作用 |
|---|---|---|---|---|
| Direct Answer | 无显式思考 | `prompt -> answer` | answer CE | 无 thinking baseline |
| Standard CoT | 自然语言 reasoning trace | `prompt -> oracle trace -> answer` | trace CE + answer CE | 主流 CoT baseline |
| Masked CoT | trace 存在但不算 loss | `prompt -> oracle trace -> answer` | only answer CE, trace masked | 诊断模仿推理文字是否必要 |
| Soft Token | vocab distribution 加权 embedding | `prompt -> K soft steps -> answer` | only answer CE | 离散 token 的连续松弛 |
| Latent Thought | hidden state 回灌为下一步输入 | `prompt -> K latent steps -> answer` | only answer CE | FDT 主实验 |

其中 Soft Token 和 Latent Thought 的 K 个 thinking steps 不来自数据文件，而是在训练 forward pass 中插入。

关键比较：

```text
Standard CoT vs Soft Token vs Latent Thought
```

如果 Latent Thought 在相同或更低 compute 下达到更高准确率，就说明自然语言 CoT 不是最优 thinking carrier。

## Fixed K 设置

第一阶段使用固定 thinking steps：

```text
K = 0, 4, 8, 16, 32
```

每个 K 单独训练和测试，用来画出：

```text
accuracy vs thinking budget
accuracy vs latency / FLOPs
```

暂不做 adaptive thinking，避免把“连续表示能力”和“动态算力分配”混在一起。

## 数据与任务

第一阶段使用 synthetic、可自动判分的数据。这样可以用确定性 solver 生成标准答案和 oracle trace，公平比较不同 thinking 机制。

优先任务：

| 任务 | 问题 | Solver | 输出 |
|---|---|---|---|
| Graph Reachability | A 到 B 是否可达 | BFS / DFS | YES / NO |
| Shortest Path | 最短路径长度 | BFS | 整数或 INF |
| Maze Planning | 网格 S 到 G 的最短路径 | BFS | 整数或 INF |
| Symbolic Arithmetic | 表达式求值 | AST evaluator | 整数 |

ID test 与训练同难度但 seed 不同；OOD test 使用更大图、更长路径、更深表达式或更复杂 maze。

## 主要风险

- final-answer-only loss 可能不足以训练稳定 latent dynamics；
- hidden state 回灌可能偏离模型原始 input embedding 分布；
- 小模型效果可能弱，需要在 1.7B 规模复验；
- CoT baseline 必须控制 token budget，否则比较不公平。

总体判断：FDT 是一个成本可控、问题清晰、可自动评估的研究方向。它保留 slow thinking 的价值，但探索比自然语言 token 更高效的内部表示。
