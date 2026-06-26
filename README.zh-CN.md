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

第一阶段全部使用 SFT，不使用 RL。

原因是 RL 会引入额外变量：

- reward 设计；
- exploration 不稳定；
- policy shortcut；
- credit assignment 难度；
- 难以判断收益来自 continuous thinking 机制还是 RL objective。

第一阶段要回答的问题应尽量干净：

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

Soft Token 仍然绕过 vocabulary distribution：

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

推荐任务：

- graph reachability；
- shortest path；
- maze planning；
- Game of 24；
- symbolic arithmetic；
- bAbI-style multi-hop QA；
- simple logic problems。

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
- 使用 graph、maze、arithmetic 任务。
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

### Phase 5: RL

- 只有在 SFT 出现正向信号后再加入 RL。
- 使用可验证 reward。
- 比较 Direct Answer、Standard CoT、Soft Token、Latent Thought 在 RL 下的收益差异。

### Phase 6: Real Tasks

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
