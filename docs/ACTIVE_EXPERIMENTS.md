# Active Experiments

## Qwen 48h hard_ladder long run

- Record: `docs/QWEN_48H_HARD_LADDER_RUN_20260627_ZH.md`
- Started: 2026-06-27 19:06:08 CST
- Recorded: 2026-06-27 19:10:15 CST
- Commit: `4303f60`
- Output directory: `outputs/qwen_48h_hard_ladder_npu_20260627_190607`
- Latest pointer: `outputs/latest_qwen_long_run.txt`
- Status at record time: 12 tmux sessions running.

Check status:

```bash
cd /home/zlong/llm/fuzzy-deep-thinker
PYTHONPATH=src /home/zlong/anaconda3/envs/fdt-npu-py39/bin/python scripts/summarize_qwen_long_run.py
tmux list-sessions
```

Stop all jobs from this run:

```bash
tmux list-sessions | awk -F: '/^fdt48_/ {print $1}' | xargs -r -n1 tmux kill-session -t
```
