# Qwen Integration

本文说明 FDT 当前如何接入 Qwen，以及它和早期 tiny char-level decoder 的关系。

## 当前 Base Model

现在仓库有两条训练路径：

- `fdt.train_tiny`: 随机初始化的 tiny char-level decoder，只用于快速验证数据、loss、evaluation、NPU 环境和 continuous thinking 机制。
- `fdt.train_qwen`: Hugging Face / Qwen causal LM 路径，默认模型是 `Qwen/Qwen3-0.6B-Base`。

因此，真正的 base model 实验应使用 `fdt.train_qwen`。它加载的是预训练过的 Qwen Base 模型，而不是从零训练的小模型。

## 为什么保留 Tiny Decoder

tiny char-level decoder 不是最终研究对象，它的作用是工程诊断：

- 运行快，能在 CPU 或 NPU 上快速检查五组训练格式是否可执行。
- 模型小，容易定位 loss mask、数据格式、evaluation parsing 的错误。
- 不依赖外部模型下载，适合做 CI / smoke test。
- 可以先排除任务生成和实验框架的问题，再把同一套格式迁移到 Qwen。

tiny 结果只能说明实验管线是否有效，不能证明 Qwen 或真实 LLM 上的结论。

## 五组方法

所有方法使用同一批题和同样的最终答案标签，区别只在中间计算载体。

| 方法 | 格式 | Loss |
| --- | --- | --- |
| `direct` | `prompt -> answer` | answer CE |
| `cot` | `prompt -> oracle trace -> answer` | trace CE + answer CE |
| `masked_cot` | `prompt -> oracle trace -> answer` | only answer CE, trace masked |
| `soft` | `prompt -> K soft-token steps -> answer` | only answer CE |
| `latent` | `prompt -> K latent steps -> answer` | only answer CE |

`soft` 通过 `softmax(logits) @ embedding_matrix` 得到下一步连续输入；`latent` 直接把最后一层 hidden state 经过 norm/projection 后反馈为下一步 input embedding。

## 常用缩写

- FDT: Fuzzy Deep Thinker，本项目名。
- SFT: Supervised Fine-Tuning，监督微调。
- CE: Cross Entropy，交叉熵 loss。
- CoT: Chain-of-Thought，链式思考。
- HF: Hugging Face。
- NPU: Neural Processing Unit，这里主要指 Ascend 910。
- ID: in-distribution，同分布测试。
- OOD: out-of-distribution，分布外测试。
- K: continuous thinking 的固定步数。

## 最小运行命令

只检查 CLI 和依赖：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src /home/zlong/anaconda3/envs/fdt-npu-py39/bin/python -m fdt.train_qwen --help
```

在 Ascend NPU 上跑一个 direct Qwen smoke：

```bash
PYTHONPATH=src scripts/with_conda_npu.sh python -m fdt.train_qwen \
  --model-name-or-path Qwen/Qwen3-0.6B-Base \
  --build-data \
  --task graph_reachability \
  --difficulty easy_ladder \
  --method direct \
  --device npu:0 \
  --dtype bfloat16 \
  --steps 1 \
  --eval-examples 2 \
  --output outputs/qwen_smoke/direct_seed0.json
```

运行五组方法的 smoke matrix：

```bash
scripts/with_conda_npu.sh scripts/run_qwen_smoke.sh
```

如果 Hugging Face 直连不可用，但本机 `127.0.0.1:7890` 代理可用：

```bash
HTTP_PROXY=http://127.0.0.1:7890 \
HTTPS_PROXY=http://127.0.0.1:7890 \
ALL_PROXY=socks5h://127.0.0.1:7891 \
NO_PROXY=localhost,127.0.0.1 \
scripts/with_conda_npu.sh scripts/run_qwen_smoke.sh
```

可通过环境变量调整：

```bash
MODEL_NAME_OR_PATH=/path/to/local/qwen \
CONFIGS="direct:- cot:- soft:4 latent:4" \
STEPS=20 \
EVAL_EXAMPLES=16 \
scripts/with_conda_npu.sh scripts/run_qwen_smoke.sh
```

## 当前限制

- `fdt.train_qwen` 支持 `--use-lora`，正式 Qwen 小矩阵应优先使用 LoRA，降低显存、checkpoint 和优化稳定性风险。
- Ascend NPU 上不建议用 full-parameter FP16 AdamW 训练 Qwen，最小 smoke 中观察到它可能导致候选打分变成 `NaN`；当前 runner 默认使用 `bfloat16`。
- `soft` 方法会计算全 vocabulary 的 soft embedding，显存和速度压力比 `latent` 更大。
- `masked_cot` 存在 train/test mismatch，只应作为诊断 baseline。
- Qwen smoke 的小步数结果只能验证代码路径，不能作为研究结论。

## Qwen LoRA Matrix

正式的小规模 Qwen 对照建议先跑 LoRA 矩阵：

```bash
HTTP_PROXY=http://127.0.0.1:7890 \
HTTPS_PROXY=http://127.0.0.1:7890 \
ALL_PROXY=socks5h://127.0.0.1:7891 \
NO_PROXY=localhost,127.0.0.1 \
scripts/with_conda_npu.sh scripts/run_qwen_lora_matrix.sh
```

默认配置：

```text
model:   Qwen/Qwen3-0.6B-Base
task:    graph_reachability / easy_ladder
methods: direct, cot, masked_cot, soft_k1, latent_k1
seeds:   0, 1
steps:   80 optimizer updates
eval:    16 examples per split
dtype:   bfloat16
LoRA:    r=8, alpha=16, dropout=0.05
```

如果本机已经缓存 `Qwen/Qwen3-0.6B-Base`，runner 默认会自动切换到 Hugging Face cache 里的 snapshot 目录，并传入 `--local-files-only`，避免后台长任务被 Hugging Face metadata 请求或代理波动中断。

输出：

```text
outputs/qwen_lora_matrix/
  aggregate.csv
  summary.csv
  *_seed*.json
  checkpoints/*.pt
```

若要判断当前 `easy_ladder` 对 Qwen 是否过于简单，建议先跑 full-eval diagnostics：

```bash
DATA_DIR=data/qwen_lora_full_eval \
OUTPUT_DIR=outputs/qwen_lora_full_eval \
CHECKPOINT_DIR=outputs/qwen_lora_full_eval/checkpoints \
SEEDS="0 1 2" \
EVAL_EXAMPLES=200 \
DIAGNOSTIC_METADATA_KEYS=answer,num_nodes,path_length \
scripts/with_conda_npu.sh scripts/run_qwen_lora_matrix.sh
```

这会在 `aggregate.csv` 中额外写出分组准确率，例如：

```text
ood_test_answer_YES_mean
ood_test_answer_NO_mean
ood_test_num_nodes_7_mean
ood_test_num_nodes_8_mean
ood_test_path_length_1_mean
```

当前 diagnostics 由 dev/id/ood 的同一次评估 records 聚合，不会再对每个 metadata 子组重复调用模型。默认 JSON 只写 split 级统计、samples 和成功/失败案例；diagnostics 分组默认不写完整案例，可通过 `DIAGNOSTIC_CASE_EXAMPLES=1` 打开。若需要完整逐样本 records，可在 `train_qwen` 命令中显式加入 `--include-eval-records`。

如果 `easy_ladder` 已接近满分，优先切换到更难的图可达设置：

```bash
DIFFICULTY=hard_ladder \
DATA_DIR=data/qwen_hard_probe \
OUTPUT_DIR=outputs/qwen_lora_hard_probe_npu \
CHECKPOINT_DIR=outputs/qwen_lora_hard_probe_npu/checkpoints \
CONFIGS="direct:- cot:- latent:1" \
SEEDS="0 1" \
STEPS=80 \
EVAL_EXAMPLES=200 \
DIAGNOSTIC_METADATA_KEYS=answer,num_nodes,path_length \
scripts/with_conda_npu.sh scripts/run_qwen_lora_matrix.sh
```

`hard_ladder` 的训练 split 使用 6-10 个节点、YES 样本最短路径长度 1-3；OOD split 使用 12-16 个节点、YES 样本最短路径长度 4-8。它用于检验方法在更长隐式搜索链和更大图上的泛化，而不是只验证基础问答格式是否可学。

checkpoint 支持：

```bash
PYTHONPATH=src scripts/with_conda_npu.sh python -m fdt.train_qwen \
  --use-lora \
  --load-checkpoint outputs/qwen_lora_matrix/checkpoints/direct_seed0.pt \
  --eval-only \
  --steps 0 \
  --eval-examples 16 \
  --output outputs/qwen_lora_matrix/direct_seed0_eval_reload.json
```
