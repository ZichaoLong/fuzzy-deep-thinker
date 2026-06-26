# Continuous Latent Thought 中文研究计划

Continuous Latent Thought，简称 CLT，是一个研究项目，目标是把语言模型的 thinking phase 从人类可读的离散链式思考 token，替换或放松为连续的 embedding / hidden-state 计算步骤。

核心假设是：

> slow thinking 很有用，但自然语言 thinking token 可能是一种高噪声、高开销、受 vocabulary 约束的中间计算载体。如果把 thinking phase 放松到连续 latent space，模型可能在复杂任务上获得更好的 accuracy-compute tradeoff。

## 背景

当前的 chain-of-thought 和 long thinking sequence 在编程、数学、规划、多步推理任务中确实有明显效果。实践上，长思考序列对复杂问题的帮助很难否认。

但另一个问题也越来越清楚：这些中间 token 不一定是模型内部计算的忠实解释。自然语言 thinking token 可能同时承担了几件不同的事：

- 提供 test-time compute；
- 作为外部 scratchpad；
- 改变后续 token 的条件分布；
- 模仿人类或 teacher 的推理文字；
- 给用户提供一种看似可解释的过程。

这些功能混在一起后，很难判断性能提升到底来自哪里。本项目关心的不是“CoT 是否无效”，而是一个更具体的问题：

> 复杂问题需要 slow thinking，但 slow thinking 是否必须用自然语言 token 表达？

## 研究问题

我们希望比较以下两类路径：

```text
传统路径:
prompt tokens -> textual thinking tokens -> answer tokens

CLT 路径:
prompt tokens -> K continuous thinking steps -> answer tokens
```

在 CLT 路径中，只有最终输出给人的 answer tokens 需要 decode 到离散 vocabulary。内部 thinking steps 可以停留在连续 embedding 或 hidden-state 空间。

## 主要假设

1. 标准 CoT 的收益主要来自额外 test-time compute，不一定来自自然语言本身是最优思考格式。
2. 离散 vocabulary token 会给 thinking phase 增加表达瓶颈和序列化开销。
3. 连续 latent thought 可能用更少 step 表达更丰富的中间状态。
4. 在相同或接近 compute budget 下，continuous thinking 可能得到比 standard CoT 更好的准确率曲线。
5. 第一阶段应先研究 fixed K，再研究 adaptive K；否则机制差异和动态算力分配会混在一起。

## 实验范围

本计划只研究 SFT 和 supervised auxiliary losses。

这里要回答的问题应尽量干净：

> 同一个 base model、同一批训练题、同样最终答案监督、相近 compute budget 下，continuous thinking 是否优于离散 CoT？

## 基础模型

建议模型：

```text
debug:       Qwen3-0.6B-Base
main:        Qwen3-1.7B-Base
scale check: Qwen3-4B-Base
```

第一阶段优先使用 Base 模型，而不是 Instruct 模型。

理由：

- Instruct 模型已经通过后训练学到了特定 thinking style；
- 这些风格会污染对机制本身的判断；
- Base 模型更适合做可归因实验；
- 小模型能显著降低迭代成本。

后续可以在 Instruct 模型上复现实验，用来检查结论是否稳健。

## 五组 SFT 对照

所有对照都使用同一批题和同样的最终答案标签。区别只在中间计算载体。

### 1. Direct Answer

没有显式 thinking。

```text
prompt -> answer
```

训练 loss：

```text
answer CE
```

用途：

- 作为无 test-time thinking 的下限；
- 判断任务到底多大程度依赖额外思考计算。

### 2. Standard CoT

传统自然语言链式思考。

```text
prompt -> textual reasoning trace -> answer
```

训练 loss：

```text
reasoning CE + answer CE
```

用途：

- 主离散 thinking baseline；
- 测量完整监督自然语言 reasoning trace 的收益。

### 3. Masked CoT

训练输入仍包含自然语言 reasoning trace，但 reasoning tokens 不算 loss。

```text
prompt -> textual reasoning trace -> answer
           no loss                  CE loss
```

用途：

- 诊断实验；
- 测试“强迫模型模仿 reasoning text”本身是否有帮助。

注意：

- 该 baseline 存在明显 train/test mismatch；
- 训练时 answer conditioned on gold reasoning；
- 测试时没有 gold reasoning；
- 因此 Masked CoT 不应作为主要结论，只适合作为辅助诊断。

### 4. Soft Token

Soft Token 是对离散 vocabulary token 的连续松弛。

普通 decoding：

```text
hidden state -> vocab logits -> sampled token id -> token embedding
```

Soft-token thinking：

```text
hidden state -> vocab logits -> softmax distribution -> weighted embedding
```

公式：

```text
p_t = softmax(logits_t / temperature)
e_t = p_t @ embedding_matrix
```

然后把 `e_t` 作为下一步 input embedding。

训练格式：

```text
prompt -> K soft-token steps -> answer
```

训练 loss：

```text
answer CE
```

用途：

- 测试 vocabulary token 的连续化是否有收益；
- 仍然受 vocabulary embedding manifold 约束；
- 作为 discrete token 与 latent thought 之间的中间方案。

### 5. Latent Thought

Latent Thought 不经过 vocabulary logits，而是直接把模型 hidden state 反馈为下一步输入。

```text
h_t = transformer_last_hidden_state
e_{t+1} = projection_or_normalization(h_t)
```

训练格式：

```text
prompt -> K latent steps -> answer
```

训练 loss：

```text
answer CE
```

用途：

- CLT 的主实验；
- 测试 hidden-state-space thinking 是否比自然语言 thinking 更适合作为中间计算载体。

这是最贴近本项目核心假设的设置。

## Soft Token 与 Latent Thought 的区别

Soft Token 仍然经过 vocabulary distribution：

```text
hidden state -> vocab logits -> soft vocabulary distribution -> embedding
```

Latent Thought 直接绕开 vocabulary：

```text
hidden state -> continuous hidden representation -> next input embedding
```

预期表达能力：

```text
discrete token < soft token < latent thought
```

预期训练稳定性：

```text
discrete token > soft token > latent thought
```

也就是说，Latent Thought 更可能提高表达能力，但也更可能遇到训练不稳定、latent drift、credit assignment 困难等问题。

## Fixed K 优先

第一阶段使用固定 thinking 长度：

```text
K = 0, 4, 8, 16, 32
```

每个 fixed-K 模型训练和测试时都使用同一个 K。

例如：

```text
Latent-K8:
  training: prompt -> 8 latent steps -> answer
  testing:  prompt -> 8 latent steps -> answer
```

固定 K 的好处是归因清楚：

> 给定同样数量的 internal thinking steps，continuous thought 是否比离散 CoT 更有效？

Adaptive K 应放到后续阶段。

## Adaptive Thinking 后置

后续可以引入：

```text
thinking_status = THINKING | OUTPUTTING
```

或者：

```text
continue_prob < threshold -> switch to answer decoding
```

这样可以测试模型是否能动态分配 thinking compute。

但第一阶段不应混入 adaptive K，因为它会同时改变两个变量：

- thinking representation；
- compute allocation policy。

## 数据

第一阶段应使用 synthetic、可自动判分的数据。

建议顺序是：

```text
先做：graph reachability、shortest path、maze planning、symbolic arithmetic
后做：Game of 24、bAbI-style multi-hop QA、simple logic problems
```

前四个任务更适合第一阶段，因为它们都有确定性 solver，final answer 容易自动验证，难度也容易连续调节。

### Task 1: Graph Reachability

这是图可达性任务。

问题形式：

```text
给定一个有向图。
Nodes: A, B, C, D, E
Edges: A->B, B->D, C->E
Question: Is there a path from A to D?
Return YES or NO.
```

最终答案：

```text
YES
```

solver：

```text
从 source 到 target 做 BFS 或 DFS
```

给 Standard CoT 用的 oracle trace：

```text
Start at A. Visit B from A. Visit D from B. D is reached.
```

难度变量：

- node 数量；
- edge density；
- 最短路径长度；
- distractor edges 数量；
- YES / NO 标签平衡。

这个任务适合最先做，因为最终答案是二分类，判分稳定，而且 reasoning depth 可以直接用路径长度控制。

### Task 2: Shortest Path

这是最短路径任务。

问题形式：

```text
给定一个无权有向图。
Nodes: A, B, C, D, E
Edges: A->B, A->C, B->D, C->D, D->E
Question: What is the shortest path distance from A to E?
Return an integer, or INF if unreachable.
```

最终答案：

```text
3
```

solver：

```text
无权图用 BFS
```

oracle trace：

```text
Distance(A)=0. From A set B=1 and C=1. From B set D=2. From D set E=3. The shortest distance is 3.
```

难度变量：

- node 数量；
- 最短路径长度；
- branching factor；
- unreachable cases；
- 比最短路更长的 distractor paths。

建议先只做无权图。加权图需要 Dijkstra，会额外引入数值计算噪声，不适合第一版。

### Task 3: Maze Planning

这是二维网格规划任务，本质上也是图搜索，但输入更接近空间结构。

问题形式：

```text
Find the shortest path from S to G in the grid.
S..#
.#..
..#G
Return the shortest path length, or INF if no path exists.
```

最终答案：

```text
5
```

solver：

```text
把每个可走 cell 当作节点，在 grid 上做 BFS
```

oracle trace：

```text
Expand S at distance 0. Add reachable neighbors at distance 1. Continue BFS until G is reached at distance 5.
```

难度变量：

- grid size；
- wall density；
- 最短路径长度；
- dead ends 数量；
- solvable / unsolvable 标签平衡。

这个任务测试模型是否能在结构化输入上维护隐式状态。它比纯图可达性更接近 planning。

### Task 4: Symbolic Arithmetic

这是符号算术任务。

问题形式：

```text
Evaluate the expression:
((3 + 5) - 2) + 4
Return the integer result.
```

最终答案：

```text
10
```

solver：

```text
把表达式 parse 成 AST，然后确定性求值
```

oracle trace：

```text
3 + 5 = 8. 8 - 2 = 6. 6 + 4 = 10.
```

难度变量：

- expression depth；
- 数字范围；
- operators；
- parentheses depth；
- intermediate value range。

建议先只用 `+`、`-` 和小整数。等训练和验证流程稳定后，再加入乘法。不要一开始就加入除法，因为答案格式和整数性会变复杂。

### 后续任务

Game of 24：

- 输入四个数字；
- 输出一个能得到 24 的表达式，或者 `NO SOLUTION`；
- 用 exhaustive search 生成答案；
- 测试时用 parser + evaluator 验证模型输出；
- 这个任务更难，因为一个问题可能有很多合法答案。

bAbI-style multi-hop QA：

- 生成短故事和事实更新；
- 问题需要检索两个或多个事实；
- 用 symbolic state tracker 求解；
- 适合测试语言形式的多跳推理。

Simple logic：

- 生成 facts 和 Horn-style rules；
- 问一个 query 是否可以被推出；
- 用 forward chaining 求解；
- 用 proof depth 和 distractor rules 控制难度。

这些任务应在前四个任务得到稳定训练曲线后再加入。

## 数据生成

每条样本保存：

```json
{
  "prompt": "problem statement",
  "trace": "oracle reasoning trace from a deterministic solver",
  "answer": "final answer",
  "metadata": {
    "task": "graph_reachability",
    "difficulty": "medium",
    "seed": 123
  }
}
```

`trace` 应优先来自确定性 solver，而不是 teacher LLM。

原因：

- solver trace 更可控；
- 可以避免引入 teacher 的语言噪声；
- Standard CoT baseline 更干净；
- final answer verifier 更容易实现。

具体生成流程：

```text
1. 从 difficulty config 采样任务参数。
2. 用 random seed 生成一个 problem instance。
3. 调用确定性 solver 得到 trace 和 answer。
4. 保存 prompt、canonical trace、final answer、metadata、seed。
5. 用 answer parser / verifier 检查保存的 answer 是否可判分。
```

建议第一版每个任务的数据规模：

```text
debug train:   2k examples
debug dev:     200 examples
main train:    50k examples
main dev:      2k examples
ID test:       2k examples
OOD test:      2k examples
```

训练、dev、ID test、OOD test 使用互不重叠的 random seeds。不要先生成大量近似重复样本后再随机切分，否则 train/test 之间可能有很强近邻泄漏。

建议第一版 difficulty config：

| Task | Train / ID test | OOD test |
|---|---|---|
| Graph reachability | 6-10 个 nodes，path length 1-4，YES/NO 各 50% | 12-18 个 nodes，path length 5-8，更多 distractor edges |
| Shortest path | 6-10 个 nodes，distance 2-5，无权图 | 12-18 个 nodes，distance 6-10，更多 distractor paths |
| Maze planning | 5x5 到 8x8 grid，path length 4-12，wall density 0.15-0.30 | 10x10 到 14x14 grid，path length 14-28，wall density 0.20-0.35 |
| Symbolic arithmetic | expression depth 2-4，整数 0-20，只用 `+` 和 `-` | expression depth 5-8，整数 0-50，更多括号 |

建议 seed protocol：

```text
train seeds: 0 to 49,999
dev seeds: 1,000,000 to 1,001,999
ID test seeds: 2,000,000 to 2,001,999
OOD test seeds: 3,000,000 to 3,001,999
```

ID test 和训练集使用相同 difficulty config，但 seed 不重叠。OOD test 使用更大、更深、更长的配置。

最终答案格式要尽量 canonical：

```text
Answer: YES
Answer: NO
Answer: 3
Answer: INF
Answer: (3+5)*(6-3)
```

五种模型对同一条样本的使用方式不同：

- Direct Answer 使用 `prompt` 和 `answer`；
- Standard CoT 使用 `prompt`、`trace` 和 `answer`；
- Masked CoT 使用 `prompt`、`trace` 和 `answer`，但 trace 部分 loss mask；
- Soft Token 使用 `prompt` 和 `answer`，K 个 soft steps 由 training forward pass 插入；
- Latent Thought 使用 `prompt` 和 `answer`，K 个 latent steps 由 training forward pass 插入。

## 训练与测试

训练时，不同方法使用同一批题，只改变格式。

Direct Answer：

```text
prompt -> answer
loss: answer CE
```

Standard CoT：

```text
prompt -> oracle trace -> answer
loss: trace CE + answer CE
```

Masked CoT：

```text
prompt -> oracle trace -> answer
loss: only answer CE
```

Soft Token：

```text
prompt -> K soft steps -> answer
loss: only answer CE
```

Latent Thought：

```text
prompt -> K latent steps -> answer
loss: only answer CE
```

测试时，所有模型都只拿到 prompt，不提供 oracle trace。

评价只看最终答案是否正确。

## 测试集设计

测试集分为两类。

In-distribution：

```text
与训练集同任务类型、同难度范围
```

Out-of-distribution：

```text
更大图
更长路径
更深表达式
更多 distractors
更长 dependency chain
```

这样可以同时测试：

- 同分布准确率；
- 复杂度外推能力。

## 评估指标

主指标：

```text
final answer correctness
```

按任务选择：

- exact match；
- solver-based verifier；
- executable verifier。

辅助指标：

```text
accuracy vs K
accuracy vs generated output tokens
accuracy vs wall-clock latency
accuracy vs FLOPs
accuracy vs memory
```

后续如果进入代码任务，再使用：

```text
pass@1
unit test pass rate
execution correctness
```

不要用人类可解释性评价 latent thought。Latent Thought 的目标本来就不是可读解释，而是中间计算效率和能力。

## Compute Fairness

Standard CoT 应控制 reasoning token budget：

```text
reasoning token budget = 32, 64, 128, 256
```

Continuous methods 应控制 fixed K：

```text
K = 4, 8, 16, 32
```

关键图表是：

```text
accuracy vs wall-clock latency
accuracy vs FLOPs
accuracy vs output token count
```

不能只比较 accuracy。否则一个方法用 500 个 thinking tokens，另一个方法只用 16 个 latent steps，结果很难解释。

## 训练协议

跨方法固定以下变量：

- 同一个 base model；
- 同一批训练样本；
- 同样 answer format；
- 同样 optimizer；
- 同样 learning rate schedule；
- 同样 batch size 或 token budget；
- 同样 evaluation parser；
- 同样 maximum final-answer token length；
- 同样 random seed 设置。

建议第一版协议：

```text
model: Qwen3-0.6B-Base
training: SFT
tasks: graph reachability + maze planning + symbolic arithmetic
K: 0, 4, 8, 16, 32
seeds: at least 3
```

看到信号后，在以下模型上复现：

```text
Qwen3-1.7B-Base
```

## 实现要点

Direct Answer 和 Standard CoT 可以使用普通 tokenized SFT。

Soft Token 和 Latent Thought 需要自定义 forward path，并使用 `inputs_embeds`。

Soft Token forward：

```text
1. Run model on current prefix.
2. Take last-position logits.
3. Compute softmax over vocabulary.
4. Multiply by input embedding matrix.
5. Append resulting embedding as the next input.
6. Repeat for K steps.
7. Decode answer tokens and apply answer CE loss.
```

Latent Thought forward：

```text
1. Run model on current prefix.
2. Take last-position hidden state.
3. Project or normalize it into input-embedding dimension.
4. Append it as the next input embedding.
5. Repeat for K steps.
6. Decode answer tokens and apply answer CE loss.
```

如果 hidden size 等于 embedding size，先尝试只做 normalization。只有在不稳定时再加 learned projection。

可能有用的稳定化技巧：

- latent vector norm control；
- RMSNorm before feedback；
- residual mixing with a learned latent marker embedding；
- curriculum over K；
- 从 Standard CoT 做 teacher-distilled warm start。

## 风险点

Credit assignment：

- final-answer-only loss 可能太弱，难以塑造长 latent dynamics。

Representation drift：

- hidden state 直接作为 embedding 回灌，可能偏离模型预训练时见过的 input embedding 分布。

Degenerate latents：

- 模型可能学出重复、无信息或不稳定的 latent state。

Compute fairness：

- 必须严格比较 FLOPs / latency / output tokens，否则结论容易失真。

Interpretability：

- latent thinking 不可读，因此需要外部 verifier，而不是依赖中间过程解释。

## 阶段计划

### Phase 0: Infrastructure

- 实现 synthetic data generators。
- 实现 deterministic answer verifiers。
- 实现 Direct Answer 与 Standard CoT baselines。
- 实现 fixed-K Soft Token 与 Latent Thought forward paths。

### Phase 1: Small SFT Sweep

- 使用 Qwen3-0.6B-Base。
- 固定 K：0, 4, 8, 16, 32。
- 使用 graph reachability、shortest path、maze planning、symbolic arithmetic 任务。
- 比较 accuracy-compute curves。

### Phase 2: Scale Check

- 在 Qwen3-1.7B-Base 上复现实验。
- 提高任务难度。
- 加入 OOD splits。
- 至少跑 3 个 random seeds。

### Phase 3: Adaptive K

- 加入 `thinking_status = THINKING | OUTPUTTING`。
- 训练模型自己决定何时停止 thinking 并开始 output。
- 在相同 compute budget 下比较 fixed K 与 adaptive K。

### Phase 4: Distillation and Stabilization

- 从 Standard CoT 或更强 teacher 模型蒸馏。
- 如果 final-answer-only 训练不稳定，再尝试辅助 latent loss。
- 探索从 textual CoT 到 soft token，再到 latent thought 的 curriculum。

### Phase 5: Real Tasks

- GSM8K / SVAMP：数学；
- ProofWriter / PrOntoQA：逻辑；
- MBPP / HumanEval-style：代码。

代码任务应后置，因为它包含很多额外因素：

- 语法正确性；
- API recall；
- 执行反馈；
- 测试用例覆盖；
- 输出格式。

这些因素会干扰对 latent thinking 本身的判断。

## 成功标准

如果 Latent Thought 满足以下任一条件，就说明第一阶段有正向信号：

- 在 modest K 下高于 Direct Answer；
- accuracy-latency 曲线优于 Standard CoT；
- 在 matched compute 下 OOD 表现优于 Standard CoT；
- 用更少 output tokens 达到接近 Standard CoT 的准确率；
- 随 K 增长的收益强于 Soft Token。

最强结果是：

> 在 matched compute 下，Latent Thought 的 final answer accuracy 超过 Standard CoT，同时只输出最终答案 token。

## 名称

推荐名称：

```text
Continuous Latent Thought (CLT)
```

备选名称：

- Latent Reasoning Flow (LRF)
- Continuous Thought Tokens (CTT)
- Hidden Thought Flow (HTF)
- Soft Internal Reasoning (SIR)
- Latent Deliberation (LD)

推荐使用 Continuous Latent Thought，因为它直接描述机制，同时避免过度宣称 latent states 是忠实的 reasoning traces。
