# Fuzzy Deep Thinker

Chinese version: [README.zh-CN.md](README.zh-CN.md)

Fuzzy Deep Thinker, abbreviated as FDT, is a research project about replacing human-readable chain-of-thought tokens with continuous internal computation steps.

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

This plan uses supervised fine-tuning and supervised auxiliary losses only.

The clean experimental question is:

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

- main FDT experiment;
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

The first experiments should use synthetic, automatically verifiable tasks.

The recommended order is:

```text
start with: graph reachability, shortest path, maze planning, symbolic arithmetic
then add:   Game of 24, bAbI-style multi-hop QA, simple logic problems
```

The first group is preferred because each task has a small deterministic solver, a clean final-answer verifier, and a controllable difficulty axis.

### Task 1: Graph Reachability

Prompt:

```text
You are given a directed graph.
Nodes: A, B, C, D, E
Edges: A->B, B->D, C->E
Question: Is there a path from A to D?
Return YES or NO.
```

Answer:

```text
YES
```

Solver:

```text
BFS or DFS from source to target
```

Oracle trace for Standard CoT:

```text
Start at A. Visit B from A. Visit D from B. D is reached.
```

Difficulty controls:

- number of nodes;
- edge density;
- shortest path length;
- number of distractor edges;
- reachable vs unreachable label balance.

This is the cleanest first task because final answers are binary and the reasoning depth can be controlled by path length.

### Task 2: Shortest Path

Prompt:

```text
You are given an unweighted directed graph.
Nodes: A, B, C, D, E
Edges: A->B, A->C, B->D, C->D, D->E
Question: What is the shortest path distance from A to E?
Return an integer, or INF if unreachable.
```

Answer:

```text
3
```

Solver:

```text
BFS for unweighted graphs
```

Oracle trace:

```text
Distance(A)=0. From A set B=1 and C=1. From B set D=2. From D set E=3. The shortest distance is 3.
```

Difficulty controls:

- number of nodes;
- shortest path distance;
- branching factor;
- unreachable cases;
- distractor paths that are longer than the shortest path.

Start with unweighted graphs. Weighted graphs require Dijkstra and add arithmetic noise, so they should be introduced later.

### Task 3: Maze Planning

Prompt:

```text
Find the shortest path from S to G in the grid.
S..#
.#..
..#G
Return the shortest path length, or INF if no path exists.
```

Answer:

```text
5
```

Solver:

```text
BFS over grid cells
```

Oracle trace:

```text
Expand S at distance 0. Add reachable neighbors at distance 1. Continue BFS until G is reached at distance 5.
```

Difficulty controls:

- grid size;
- wall density;
- shortest path length;
- number of dead ends;
- solvable vs unsolvable label balance.

Maze planning is a spatial version of graph search. It tests whether latent thinking helps when the model must maintain an implicit state over a structured input.

### Task 4: Symbolic Arithmetic

Prompt:

```text
Evaluate the expression:
((3 + 5) - 2) + 4
Return the integer result.
```

Answer:

```text
10
```

Solver:

```text
parse expression into an AST, then evaluate it deterministically
```

Oracle trace:

```text
3 + 5 = 8. 8 - 2 = 6. 6 + 4 = 10.
```

Difficulty controls:

- expression depth;
- number range;
- operators;
- parentheses depth;
- intermediate value range.

Start with `+`, `-`, and small positive integers. Add multiplication only after the baseline setup is stable.

### Later Tasks

Game of 24:

- Input four numbers.
- Output an expression that evaluates to 24, or `NO SOLUTION`.
- Generate answers with exhaustive search.
- Verify by parsing and evaluating the returned expression.
- This is harder because there may be many valid answers.

bAbI-style multi-hop QA:

- Generate short synthetic stories with facts and state updates.
- Ask a question that requires retrieving two or more facts.
- Solve with a symbolic state tracker.
- Useful for testing language-heavy multi-hop reasoning.

Simple logic:

- Generate facts and Horn-style rules.
- Ask whether a query is entailed.
- Solve with forward chaining.
- Control difficulty by proof depth and number of distractor rules.

These tasks should be added after the first four tasks produce reliable training and evaluation curves.

## Data Generation

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

Use deterministic solvers to generate traces. Avoid using LLM-generated traces in the first experiments because they add teacher noise.

The generation pipeline should be:

```text
1. Sample task parameters from a difficulty config.
2. Generate a problem instance with a random seed.
3. Run a deterministic solver.
4. Save the prompt, canonical trace, final answer, metadata, and seed.
5. Verify that the final answer parser accepts the saved answer.
```

Suggested initial dataset sizes per task:

```text
debug train:   2k examples
debug dev:     200 examples
main train:    50k examples
main dev:      2k examples
ID test:       2k examples
OOD test:      2k examples
```

Use disjoint random seeds for train, dev, ID test, and OOD test. Do not create the test set by randomly splitting near-duplicate generated examples after the fact.

Suggested initial difficulty config:

| Task | Train / ID test | OOD test |
|---|---|---|
| Graph reachability | 6-10 nodes, path length 1-4, 50/50 reachable labels | 12-18 nodes, path length 5-8, more distractor edges |
| Shortest path | 6-10 nodes, distance 2-5, unweighted graphs | 12-18 nodes, distance 6-10, more distractor paths |
| Maze planning | 5x5 to 8x8 grids, path length 4-12, wall density 0.15-0.30 | 10x10 to 14x14 grids, path length 14-28, wall density 0.20-0.35 |
| Symbolic arithmetic | expression depth 2-4, integers 0-20, `+` and `-` | expression depth 5-8, integers 0-50, more parentheses |

Suggested seed protocol:

```text
train seeds: 0 to 49,999
dev seeds: 1,000,000 to 1,001,999
ID test seeds: 2,000,000 to 2,001,999
OOD test seeds: 3,000,000 to 3,001,999
```

The ID test uses the same difficulty config as training but disjoint seeds. The OOD test uses larger or deeper instances.

The final answer format should be canonical and minimal:

```text
Answer: YES
Answer: NO
Answer: 3
Answer: INF
Answer: (3+5)*(6-3)
```

For the five model variants:

- Direct Answer uses `prompt` and `answer`.
- Standard CoT uses `prompt`, `trace`, and `answer`.
- Masked CoT uses `prompt`, `trace`, and `answer`, but masks trace loss.
- Soft Token uses `prompt` and `answer`; the K soft steps are inserted by the training forward pass.
- Latent Thought uses `prompt` and `answer`; the K latent steps are inserted by the training forward pass.

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
- Use graph reachability, shortest path, maze planning, and symbolic arithmetic tasks.
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

### Phase 5: Real Tasks

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
Fuzzy Deep Thinker (FDT)
```

Other possible names:

- Latent Reasoning Flow (LRF)
- Continuous Thought Tokens (CTT)
- Hidden Thought Flow (HTF)
- Soft Internal Reasoning (SIR)
- Latent Deliberation (LD)

The recommended name is Fuzzy Deep Thinker because it directly states the mechanism and avoids overclaiming that the latent states are faithful reasoning traces.
