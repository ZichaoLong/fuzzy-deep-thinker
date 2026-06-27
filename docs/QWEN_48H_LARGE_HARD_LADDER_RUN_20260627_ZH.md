# Qwen 48h large hard_ladder 长训练记录

## 基本信息

- 项目：Fuzzy Deep Thinker
- 启动时间：2026-06-27 19:54:44 CST
- 记录时间：2026-06-27 19:56:57 CST
- Git commit：`0c653cd Scale Qwen long-run data and micro-batching`
- 输出目录：`outputs/qwen_48h_large_hard_ladder_npu_20260627_195444`
- latest pointer：`outputs/latest_qwen_long_run.txt`
- 数据目录：`data/qwen_48h_large_hard_ladder`
- manifest：`outputs/qwen_48h_large_hard_ladder_npu_20260627_195444/manifest.tsv`

本轮替换上一轮 `debug` 数据长跑。上一轮 direct/masked baseline 已快速过拟合，且 NPU AICore 多数只有个位数；本轮将训练集从 2k 扩到 100k，并加入真实 micro-batch 批处理。

## 训练设置

- 模型：`Qwen/Qwen3-0.6B-Base`，实际使用本地 Hugging Face cache snapshot。
- 任务：`graph_reachability`
- 难度：`hard_ladder`
- 数据 preset：`large`
- 训练数据：100000 examples
- dev/id/OOD：各 1000 examples
- 训练时长：连续/soft/latent 配置 `MAX_TRAIN_SECONDS=172800`，即 48 小时。
- baseline：`direct`、`cot`、`masked_cot` 打包在 chip 0 顺序运行，每个最多 21600 秒，即 6 小时。
- 采样：`TRAIN_SAMPLING=balanced_answer`
- 梯度累积：`GRAD_ACCUM_STEPS=4`
- micro-batch：baseline 为 16，有效 batch 64；多数连续配置为 4，有效 batch 16；`latent k=16` 为 2，有效 batch 8。
- dtype：`bfloat16`
- LoRA：`r=8, alpha=16, dropout=0.05`
- learning rate：`LR=0.0001`
- train probe：每 2000 step，256 个训练样本。
- checkpoint interval：每 5000 step。
- final eval：dev/id/OOD 各 1000 examples。
- diagnostics：`answer,num_nodes,path_length`，每组保留成功/失败案例。

## 配置矩阵

| session | Ascend device id | config | seed | micro-batch |
|---|---:|---|---:|---:|
| `fdt48l_baseline_d0` | 0 | `direct` | 0 | 16 |
| `fdt48l_baseline_d0` | 0 | `cot` | 0 | 16 |
| `fdt48l_baseline_d0` | 0 | `masked_cot` | 0 | 16 |
| `fdt48l_0_latent_k1_s0_d1` | 1 | `latent k=1` | 0 | 4 |
| `fdt48l_1_latent_k1_s1_d2` | 2 | `latent k=1` | 1 | 4 |
| `fdt48l_2_latent_k4_s0_d3` | 3 | `latent k=4` | 0 | 4 |
| `fdt48l_3_latent_k4_s1_d4` | 4 | `latent k=4` | 1 | 4 |
| `fdt48l_4_latent_k8_s0_d5` | 5 | `latent k=8` | 0 | 4 |
| `fdt48l_5_latent_k8_s1_d6` | 6 | `latent k=8` | 1 | 4 |
| `fdt48l_6_soft_k4_s0_d7` | 7 | `soft k=4` | 0 | 4 |
| `fdt48l_7_soft_k4_s1_d10` | 10 | `soft k=4` | 1 | 4 |
| `fdt48l_8_soft_k8_s0_d11` | 11 | `soft k=8` | 0 | 4 |
| `fdt48l_9_soft_k8_s1_d12` | 12 | `soft k=8` | 1 | 4 |
| `fdt48l_10_latent_k16_s0_d13` | 13 | `latent k=16` | 0 | 2 |

说明：Ascend device id 是传给 `ASCEND_RT_VISIBLE_DEVICES` 的物理 chip id。每个进程内部都使用 `--device npu:0`。本轮避开已有外部负载的 chip 8、9、15。

## 启动前验证

- `pytest`：20 passed。
- shell/python 语法检查：通过。
- NPU smoke：
  - `latent k=8, micro_batch=4, grad_accum=2` 通过。
  - `direct, micro_batch=16, grad_accum=2` 通过。
  - `latent k=16, micro_batch=2, grad_accum=2` 通过。
- 负例：`latent k=8, micro_batch=8` 在 chip 14 smoke 中 OOM，因此本轮默认连续配置使用 micro-batch 4。

## 初始状态快照

记录时间：2026-06-27 19:56:57 CST。

```text
session                         device  config      seed  status   step  loss_ema
fdt48l_baseline_d0              0       direct      0     running  1     0.8581
fdt48l_baseline_d0              0       cot         0     running  1     0.8581
fdt48l_baseline_d0              0       masked_cot  0     running  1     0.8581
fdt48l_0_latent_k1_s0_d1        1       latent_k1   0     running  1     9.3403
fdt48l_1_latent_k1_s1_d2        2       latent_k1   1     running  1     9.3283
fdt48l_2_latent_k4_s0_d3        3       latent_k4   0     running  1     10.7455
fdt48l_3_latent_k4_s1_d4        4       latent_k4   1     running  1     10.8253
fdt48l_4_latent_k8_s0_d5        5       latent_k8   0     running  1     11.2497
fdt48l_5_latent_k8_s1_d6        6       latent_k8   1     running  1     11.3524
fdt48l_6_soft_k4_s0_d7          7       soft_k4     0     running  1     4.1644
fdt48l_7_soft_k4_s1_d10         10      soft_k4     1     running  1     4.1191
fdt48l_8_soft_k8_s0_d11         11      soft_k8     0     running  1     9.0417
fdt48l_9_soft_k8_s1_d12         12      soft_k8     1     running  1     9.2534
fdt48l_10_latent_k16_s0_d13     13      latent_k16  0     running  1     10.4851
```

NPU 初始占用：本轮 HBM 约 13-36GB，AICore 多数约 7-24%。相比上一轮 debug 配置的 5-14GB HBM、2-8% AICore，有明显提升。

## 如何检查状态

```bash
cd /home/zlong/llm/fuzzy-deep-thinker
PYTHONPATH=src /home/zlong/anaconda3/envs/fdt-npu-py39/bin/python scripts/summarize_qwen_long_run.py
tmux list-sessions
npu-smi info
```

查看单个日志：

```bash
tail -f outputs/qwen_48h_large_hard_ladder_npu_20260627_195444/logs/latent_k8_seed0_d5.log
tail -f outputs/qwen_48h_large_hard_ladder_npu_20260627_195444/logs/baseline_pack_d0.log
```

进入单个 session：

```bash
tmux attach -t fdt48l_4_latent_k8_s0_d5
```

停止全部本轮任务：

```bash
tmux list-sessions | awk -F: '/^fdt48l_/ {print $1}' | xargs -r -n1 tmux kill-session -t
```
