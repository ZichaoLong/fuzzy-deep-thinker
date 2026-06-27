# FDT 文献笔记

本文记录与 Fuzzy Deep Thinker 直接相关的文献定位和项目内用法。它不是完整综述，而是为了避免把“理论来源、已有证据、当前实现”混在一起。

## Position: Stop Anthropomorphizing Intermediate Tokens as Reasoning/Thinking Traces

- arXiv: `2504.09762`
- Link: <https://arxiv.org/abs/2504.09762>
- 项目内角色：质疑“中间 token 等同人类可读推理”的假设。
- 对 FDT 的意义：如果 intermediate tokens 不应被拟人化为忠实 reasoning trace，那么 thinking phase 没有必要强制写成人类语言；这为 continuous/latent thinking 提供理论动机。
- 当前实验映射：
  - `cot` vs `masked_cot`：检查 trace CE 是否真的有帮助。
  - `latent` / `soft`：检查中间计算是否可以不落到离散词表。

## Interpretable Traces, Unexpected Outcomes

- arXiv: `2505.13792`
- Link: <https://arxiv.org/abs/2505.13792>
- 项目内角色：与“trace 的可解释性、结构、最终结果之间关系”相关。
- 注意：该编号与附件中提到的 Soft Thinking 编号不同，不能混用。
- 当前处理：作为文献待精读项；正式结论应基于论文原文，而不是二手解读。

## Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space

- 附件中记录的 arXiv 编号：`2505.15778`
- Link: <https://arxiv.org/abs/2505.15778>
- 项目内角色：AR 模型中 continuous concept / soft thinking 的相关工作。
- 对 FDT 的意义：支持“continuous thinking 在 AR 范式下可能 work”的存在性证据。
- 当前实验映射：
  - `soft`：vocab distribution 加权 embedding。
  - `latent`：hidden-state feedback，比 soft token 更直接绕开 vocabulary bottleneck。

## Coconut: Training LLMs to Reason in a Continuous Latent Space

- 项目内角色：最接近 `latent` 方法的已有工作。
- 对 FDT 的意义：把 hidden state 作为下一步输入，绕过 token id 的离散化。
- 当前实验映射：
  - `latent` k sweep。
  - 后续可加入 curriculum：从离散 CoT 逐步替换为连续 steps。

## ELF: Embedded Language Flows

- 附件中记录的 arXiv 编号：`2605.10938`
- Link: <https://arxiv.org/abs/2605.10938>
- 项目内角色：连续 embedding 表达与延迟离散化的理论/技术动机。
- 重要边界：ELF 是 diffusion / flow matching 范式，不是 AR Transformer 的直接实现方案。
- 对 FDT 的意义：借用的是“连续空间表达 + 延迟离散化”的表示层面启发，不是直接照搬 ELF 训练。
- 当前实验映射：
  - 当前只做 SFT / LoRA continuous thinking。
  - ELF 式 continuous denoising head 暂列 future work。

## 当前项目取舍

当前 repo 第一阶段只做：

- Qwen Base 模型；
- synthetic tasks；
- fixed K；
- SFT / supervised auxiliary losses；
- final-answer evaluator；
- success/failure cases + metadata diagnostics。

当前 repo 暂不把以下内容作为已实现实验：

- RL / reward-only thinking shaping；
- adaptive `thinking_status` 门控；
- ELF 式 flow/diffusion head；
- teacher hidden-state distillation。

这些方向保留在原始研究笔记中，但进入实现前需要独立设计和验证。
