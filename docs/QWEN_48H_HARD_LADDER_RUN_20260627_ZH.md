# Qwen 48h hard_ladder 长训练记录

## 基本信息

- 项目：Fuzzy Deep Thinker
- 启动时间：2026-06-27 19:06:08 CST
- 记录时间：2026-06-27 19:10:15 CST
- Git commit：`4303f60 Add long-run Qwen training diagnostics`
- 输出目录：`outputs/qwen_48h_hard_ladder_npu_20260627_190607`
- latest pointer：`outputs/latest_qwen_long_run.txt`
- 数据目录：`data/qwen_48h_hard_ladder`
- manifest：`outputs/qwen_48h_hard_ladder_npu_20260627_190607/manifest.tsv`

本轮实验用于验证：在更长训练时间、balanced answer sampling、训练期 probe 和多配置并行下，Qwen LoRA 是否能在 `graph_reachability/hard_ladder` 上学到超过标签偏置的能力，并比较 discrete/continuous thinking 配置。

## 训练设置

- 模型：`Qwen/Qwen3-0.6B-Base`，实际使用本地 Hugging Face cache snapshot。
- 任务：`graph_reachability`
- 难度：`hard_ladder`
- 数据 preset：`debug`
- 训练数据：2000 examples
- dev/id/OOD：各 200 examples
- 训练时长：`MAX_TRAIN_SECONDS=172800`，即 48 小时。
- 训练步数上限：`STEPS=100000000`，实际由 wall-clock time 截断。
- 采样：`TRAIN_SAMPLING=balanced_answer`
- 梯度累积：`GRAD_ACCUM_STEPS=4`
- dtype：`bfloat16`
- LoRA：`r=8, alpha=16, dropout=0.05`
- log interval：每 100 step
- train probe：每 5000 step，64 个训练样本；`masked_cot` 默认不做 train probe。
- checkpoint interval：每 20000 step
- final eval：dev/id/OOD 各 200 examples
- diagnostics：`answer,num_nodes,path_length`

## 配置矩阵

| session | Ascend device id | config | seed | output |
|---|---:|---|---:|---|
| `fdt48_0_direct_s0_d0` | 0 | `direct` | 0 | `direct_seed0.json` |
| `fdt48_1_direct_s1_d1` | 1 | `direct` | 1 | `direct_seed1.json` |
| `fdt48_2_masked_cot_s0_d2` | 2 | `masked_cot` | 0 | `masked_cot_seed0.json` |
| `fdt48_3_masked_cot_s1_d3` | 3 | `masked_cot` | 1 | `masked_cot_seed1.json` |
| `fdt48_4_latent_k1_s0_d4` | 4 | `latent k=1` | 0 | `latent_k1_seed0.json` |
| `fdt48_5_latent_k1_s1_d5` | 5 | `latent k=1` | 1 | `latent_k1_seed1.json` |
| `fdt48_6_latent_k4_s0_d6` | 6 | `latent k=4` | 0 | `latent_k4_seed0.json` |
| `fdt48_7_latent_k4_s1_d7` | 7 | `latent k=4` | 1 | `latent_k4_seed1.json` |
| `fdt48_8_latent_k8_s0_d10` | 10 | `latent k=8` | 0 | `latent_k8_seed0.json` |
| `fdt48_9_latent_k8_s1_d11` | 11 | `latent k=8` | 1 | `latent_k8_seed1.json` |
| `fdt48_10_soft_k4_s0_d12` | 12 | `soft k=4` | 0 | `soft_k4_seed0.json` |
| `fdt48_11_soft_k4_s1_d13` | 13 | `soft k=4` | 1 | `soft_k4_seed1.json` |

说明：这里的 Ascend device id 是传给 `ASCEND_RT_VISIBLE_DEVICES` 的物理 chip id。每个进程内部都使用 `--device npu:0`。

## 记录时状态快照

记录时间：2026-06-27 19:10:15 CST。

```text
session                         device  config      seed  status   step  loss_ema
fdt48_0_direct_s0_d0            0       direct      0     running  200   0.2137
fdt48_1_direct_s1_d1            1       direct      1     running  200   0.1018
fdt48_2_masked_cot_s0_d2        2       masked_cot  0     running  200   0.0000
fdt48_3_masked_cot_s1_d3        3       masked_cot  1     running  200   0.0000
fdt48_4_latent_k1_s0_d4         4       latent_k1   0     running  100   0.3984
fdt48_5_latent_k1_s1_d5         5       latent_k1   1     running  100   0.4050
fdt48_6_latent_k4_s0_d6         6       latent_k4   0     running  1     10.9235
fdt48_7_latent_k4_s1_d7         7       latent_k4   1     running  1     10.9942
fdt48_8_latent_k8_s0_d10        10      latent_k8   0     running  1     11.1611
fdt48_9_latent_k8_s1_d11        11      latent_k8   1     running  1     11.3852
fdt48_10_soft_k4_s0_d12         12      soft_k4     0     running  1     4.1162
fdt48_11_soft_k4_s1_d13         13      soft_k4     1     running  1     4.1106
```

## 如何检查状态

推荐先用汇总脚本：

```bash
cd /home/zlong/llm/fuzzy-deep-thinker
PYTHONPATH=src /home/zlong/anaconda3/envs/fdt-npu-py39/bin/python scripts/summarize_qwen_long_run.py
```

查看 tmux session：

```bash
tmux list-sessions
```

查看 NPU 进程：

```bash
npu-smi info
```

查看单个日志：

```bash
tail -f outputs/qwen_48h_hard_ladder_npu_20260627_190607/logs/latent_k4_seed0_d6.log
```

进入单个 session：

```bash
tmux attach -t fdt48_6_latent_k4_s0_d6
```

## 如何停止

停止全部本轮任务：

```bash
tmux list-sessions | awk -F: '/^fdt48_/ {print $1}' | xargs -r -n1 tmux kill-session -t
```

停止单个任务：

```bash
tmux kill-session -t fdt48_6_latent_k4_s0_d6
```

## 后续读取约定

后续如果需要检查这轮实验状态，先读取：

```text
docs/ACTIVE_EXPERIMENTS.md
docs/QWEN_48H_HARD_LADDER_RUN_20260627_ZH.md
outputs/latest_qwen_long_run.txt
```

然后运行：

```bash
PYTHONPATH=src /home/zlong/anaconda3/envs/fdt-npu-py39/bin/python scripts/summarize_qwen_long_run.py
```
