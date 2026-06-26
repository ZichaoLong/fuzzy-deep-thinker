# Continuous Latent Thought

Chinese version: [README.zh-CN.md](README.zh-CN.md)

Continuous Latent Thought, abbreviated as CLT, is a research project about replacing human-readable chain-of-thought tokens with continuous internal computation steps.

The central hypothesis is:

> Slow thinking is useful, but natural-language thinking tokens are a noisy and expensive carrier. If the thinking phase is relaxed from discrete vocabulary tokens into continuous embedding or hidden-state space, a language model may get a better accuracy-compute tradeoff on complex tasks.

## Background

Recent work raises doubts about treating intermediate tokens as faithful reasoning traces. The key concern is not that chain-of-thought is useless, but that natural-language intermediate text may be:

- noisy as a representation of the model's actual computation;
- expensive because every thought must be serialized into discrete tokens;
- over-constrained by the human vocabulary;
- misleading when interpreted as an explanation of internal reasoning.

At the same time, long thinking sequences clearly improve practical performance in programming, math, planning, and multi-step problem solving. The goal of this project is therefore not to remove test-time thinking, but to study whether the thinking carrier can be improved.

Relevant nearby ideas include:

- Continuous or latent thought methods such as Coconut-style hidden-state recurrence.
- Continuous-token generative models such as Embedded Language Flows.
- Continuous-token autoregressive image generation work, including approaches that avoid vector quantization.

## Research Question

Can a language model improve complex problem solving by using continuous latent thinking steps before producing human-readable output tokens?

More concretely:

```text
prompt tokens -> K continuous thinking steps -> answer tokens
```

Only the answer tokens need to be decoded into the discrete vocabulary for human use. The internal thinking steps may stay in embedding or hidden-state space.

## Main Hypotheses

1. Standard chain-of-thought helps because it provides test-time compute, not necessarily because natural language is the optimal computation format.
2. Discrete thinking tokens impose a representation bottleneck during the thinking phase.
3. Continuous latent thinking can represent intermediate computation more compactly than natural-language thought tokens.
4. Fixed-budget continuous thinking may improve the accuracy-compute curve compared with standard chain-of-thought.
5. Adaptive thinking length should be explored only after fixed-K continuous thinking shows a measurable benefit.

## Experimental Scope

The first phase should use supervised fine-tuning, not reinforcement learning.

RL is deferred because it introduces additional variables:

- reward design;
- exploration instability;
- policy collapse or shortcut learning;
- ambiguity about whether gains come from the continuous thinking mechanism or the RL objective.

The clean first-stage question is:

> Under the same base model, same data, same answer format, and comparable compute budget, does continuous thinking outperform discrete chain-of-thought?

## Base Models

Recommended starting models:

```text
debug:       Qwen3-0.6B-Base
main:        Qwen3-1.7B-Base
scale check: Qwen3-4B-Base
```

Use Base models first, not Instruct models.

Reason:

- Instruct models already contain post-training preferences and learned thinking styles.
- Base models make it easier to attribute changes to the experimental mechanism.
- Small Base models keep iteration cost manageable.

Instruct models can be used later as a robustness check.

## Five SFT Comparisons

All comparisons use the same training problems and the same final answer labels. They differ only in the intermediate computation carrier.

### 1. Direct Answer

No explicit thinking.

```text
prompt -> answer
```

Training loss:

```text
answer CE
```

Purpose:

- lower-bound baseline;
- measures how much the task needs test-time thinking.

### 2. Standard CoT

Traditional natural-language chain-of-thought.

```text
prompt -> textual reasoning trace -> answer
```

Training loss:

```text
reasoning CE + answer CE
```

Purpose:

- main discrete-thinking baseline;
- measures the benefit of fully supervised natural-language reasoning traces.

### 3. Masked CoT

Natural-language reasoning is present during training, but the reasoning tokens are loss-masked.

```text
prompt -> textual reasoning trace -> answer
           no loss                  CE loss
```

Purpose:

- diagnostic baseline;
- tests whether forcing the model to imitate reasoning text is helpful.

Caveat:

- This baseline has train/test mismatch.
- During training, the answer is conditioned on gold reasoning.
- During testing, the model does not have gold reasoning.
- Therefore Masked CoT should not be treated as a primary result.

### 4. Soft Token

Discrete vocabulary selection is relaxed into a soft embedding.

Normal decoding:

```text
hidden state -> vocab logits -> sampled token id -> token embedding
```

Soft-token thinking:

```text
hidden state -> vocab logits -> softmax distribution -> weighted embedding
```

Formula:

```text
p_t = softmax(logits_t / temperature)
e_t = p_t @ embedding_matrix
```

Then `e_t` is fed back as the next input embedding.

Training format:

```text
prompt -> K soft-token steps -> answer
```

Training loss:

```text
answer CE
```

Purpose:

- tests a continuous relaxation of vocabulary tokens;
- still constrained by the vocabulary embedding manifold.

### 5. Latent Thought

The model bypasses vocabulary logits during thinking and directly feeds hidden states back as continuous inputs.

```text
h_t = transformer_last_hidden_state
e_{t+1} = projection_or_normalization(h_t)
```

Training format:

```text
prompt -> K latent steps -> answer
```

Training loss:

```text
answer CE
```

Purpose:

- main CLT experiment;
- tests whether hidden-state-space thinking is a better intermediate computation carrier.

This is closest to the core research hypothesis.

## Soft Token vs Latent Thought

Soft Token still routes through vocabulary space:

```text
hidden state -> vocab logits -> soft vocabulary distribution -> embedding
```

Latent Thought bypasses vocabulary space:

```text
hidden state -> continuous hidden representation -> next input embedding
```

Expected expression capacity:

```text
discrete token < soft token < latent thought
```

Expected training stability:

```text
discrete token > soft token > latent thought
```

## Fixed K First

The first experiments should use fixed thinking length:

```text
K = 0, 4, 8, 16, 32
```

Each fixed-K model should be trained and evaluated with the same K.

Example:

```text
Latent-K8:
  training: prompt -> 8 latent steps -> answer
  testing:  prompt -> 8 latent steps -> answer
```

Fixed K gives a clean accuracy-compute curve. Adaptive stopping should be introduced only after fixed-K results show a signal.

## Adaptive Thinking Later

A later phase can introduce:

```text
thinking_status = THINKING | OUTPUTTING
```

or:

```text
continue_prob < threshold -> switch to answer decoding
```

This tests whether the model can allocate thinking compute dynamically.

However, adaptive K should not be mixed into the first experiment because it confounds two factors:

- continuous thinking representation;
- dynamic compute allocation.

## Data

Phase 1 should use synthetic, automatically verifiable tasks.

Recommended tasks:

- graph reachability;
- shortest path;
- maze planning;
- Game of 24;
- symbolic arithmetic;
- bAbI-style multi-hop QA;
- simple logic problems.

Each example should contain:

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

Use deterministic solvers to generate traces when possible. Avoid using LLM-generated traces in Phase 1 because they add teacher noise.

## Train/Test Split

Use both in-distribution and out-of-distribution tests.

In-distribution:

```text
same problem family and difficulty range as training
```

Out-of-distribution:

```text
larger graphs
longer paths
deeper arithmetic expressions
more distractors
longer dependency chains
```

This measures both:

- same-distribution accuracy;
- complexity extrapolation.

## Evaluation

Primary metric:

```text
final answer correctness
```

Use exact match or a solver-based verifier depending on the task.

Secondary metrics:

```text
accuracy vs K
accuracy vs generated output tokens
accuracy vs wall-clock latency
accuracy vs FLOPs
accuracy vs memory
```

For coding tasks in later phases:

```text
pass@1
unit test pass rate
execution correctness
```

Do not evaluate latent thinking by human interpretability. Latent thought is intentionally not human-readable.

## Compute Fairness

Standard CoT should be evaluated under controlled token budgets:

```text
reasoning token budget = 32, 64, 128, 256
```

Continuous methods should be evaluated under fixed K:

```text
K = 4, 8, 16, 32
```

The important plots are:

```text
accuracy vs wall-clock latency
accuracy vs FLOPs
accuracy vs output token count
```

This avoids an unfair comparison where one method receives much more test-time compute than another.

## Training Protocol

Keep these fixed across methods:

- same base model;
- same training examples;
- same answer format;
- same optimizer;
- same learning rate schedule;
- same batch size or token budget;
- same evaluation parser;
- same maximum final-answer token length.

Suggested first protocol:

```text
model: Qwen3-0.6B-Base
training: SFT
tasks: graph reachability + maze planning + symbolic arithmetic
K: 0, 4, 8, 16, 32
seeds: at least 3
```

Then repeat promising settings on:

```text
Qwen3-1.7B-Base
```

## Implementation Notes

Direct Answer and Standard CoT can use ordinary tokenized training.

Soft Token and Latent Thought need a custom forward path using `inputs_embeds`.

For Soft Token:

```text
1. Run model on current prefix.
2. Take last-position logits.
3. Compute softmax over vocabulary.
4. Multiply by input embedding matrix.
5. Append resulting embedding as the next input.
6. Repeat for K steps.
7. Decode answer tokens and apply answer CE loss.
```

For Latent Thought:

```text
1. Run model on current prefix.
2. Take last-position hidden state.
3. Project or normalize it into input-embedding dimension.
4. Append it as the next input embedding.
5. Repeat for K steps.
6. Decode answer tokens and apply answer CE loss.
```

If hidden size equals embedding size, start with normalization only. Add a learned projection only if needed.

Potential stabilizers:

- latent vector norm control;
- RMSNorm before feedback;
- residual mixing with a learned latent marker embedding;
- curriculum over K;
- teacher-distilled warm start from Standard CoT.

## Risk Areas

Credit assignment:

- Final-answer-only loss may be too weak to shape long latent dynamics.

Representation drift:

- Hidden states fed back as embeddings may leave the distribution expected by the model.

Degenerate latents:

- The model may learn unstable or uninformative repeated latent states.

Fairness:

- CoT and latent methods must be compared under comparable compute budgets.

Interpretability:

- Latent thinking is not human-readable, so external verification is required.

## Phase Plan

### Phase 0: Infrastructure

- Implement synthetic data generators.
- Implement deterministic answer verifiers.
- Implement Direct Answer and Standard CoT baselines.
- Add fixed-K Soft Token and Latent Thought forward paths.

### Phase 1: Small SFT Sweep

- Use Qwen3-0.6B-Base.
- Run fixed K values: 0, 4, 8, 16, 32.
- Use graph, maze, and arithmetic tasks.
- Compare accuracy-compute curves.

### Phase 2: Scale Check

- Repeat strongest settings on Qwen3-1.7B-Base.
- Increase task difficulty.
- Add OOD splits.
- Run at least 3 random seeds.

### Phase 3: Adaptive K

- Add `thinking_status = THINKING | OUTPUTTING`.
- Train the model to stop thinking and start outputting.
- Compare fixed K vs adaptive K under equal compute budgets.

### Phase 4: Distillation and Stabilization

- Distill from Standard CoT or stronger teacher models.
- Test auxiliary latent losses only if final-answer-only training is unstable.
- Explore curriculum from textual CoT to soft token to latent thought.

### Phase 5: RL

- Add reinforcement learning only after SFT shows a clear signal.
- Use verifiable rewards.
- Compare RL gains across Direct Answer, Standard CoT, Soft Token, and Latent Thought.

### Phase 6: Real Tasks

- GSM8K / SVAMP for math.
- ProofWriter / PrOntoQA for logic.
- MBPP / HumanEval-style tasks for code.

Coding tasks should be delayed because many unrelated factors affect results, including syntax, APIs, execution feedback, and answer formatting.

## Success Criteria

The project has a positive first-stage signal if Latent Thought shows at least one of the following:

- higher accuracy than Direct Answer at modest K;
- better accuracy-latency curve than Standard CoT;
- better OOD extrapolation than Standard CoT under comparable compute;
- similar accuracy to Standard CoT with fewer generated output tokens;
- stronger scaling with K than Soft Token.

The strongest result would be:

> Latent Thought beats Standard CoT on accuracy under matched compute, while producing only final answer tokens.

## Working Name Alternatives

Primary name:

```text
Continuous Latent Thought (CLT)
```

Other possible names:

- Latent Reasoning Flow (LRF)
- Continuous Thought Tokens (CTT)
- Hidden Thought Flow (HTF)
- Soft Internal Reasoning (SIR)
- Latent Deliberation (LD)

The recommended name is Continuous Latent Thought because it directly states the mechanism and avoids overclaiming that the latent states are faithful reasoning traces.
