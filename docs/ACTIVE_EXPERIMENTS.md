# Active Experiments

## Qwen 48h large hard_ladder run

- Record: `docs/QWEN_48H_LARGE_HARD_LADDER_RUN_20260627_ZH.md`
- Started: 2026-06-27 19:54:44 CST
- Recorded: 2026-06-27 19:56:57 CST
- Commit: `0c653cd`
- Output directory: `outputs/qwen_48h_large_hard_ladder_npu_20260627_195444`
- Latest pointer: `outputs/latest_qwen_long_run.txt`
- Status at record time: 12 tmux sessions running.

Check status:

```bash
cd /home/zlong/llm/fuzzy-deep-thinker
PYTHONPATH=src /home/zlong/anaconda3/envs/fdt-npu-py39/bin/python scripts/summarize_qwen_long_run.py
tmux list-sessions
npu-smi info
```

Stop all jobs from this run:

```bash
tmux list-sessions | awk -F: '/^fdt48l_/ {print $1}' | xargs -r -n1 tmux kill-session -t
```

## Qwen 48h hard_ladder debug run

- Record: `docs/QWEN_48H_HARD_LADDER_RUN_20260627_ZH.md`
- Started: 2026-06-27 19:06:08 CST
- Recorded: 2026-06-27 19:10:15 CST
- Commit: `4303f60`
- Output directory: `outputs/qwen_48h_hard_ladder_npu_20260627_190607`
- Status: stopped on 2026-06-27 after direct/masked baselines overfit quickly and the run underused NPU compute.

Check status:

```bash
cd /home/zlong/llm/fuzzy-deep-thinker
PYTHONPATH=src /home/zlong/anaconda3/envs/fdt-npu-py39/bin/python scripts/summarize_qwen_long_run.py --output-dir outputs/qwen_48h_hard_ladder_npu_20260627_190607
```
