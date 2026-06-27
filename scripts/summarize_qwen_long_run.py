from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize FDT Qwen long-run outputs and live logs.")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        pointer = Path("outputs/latest_qwen_long_run.txt")
        if not pointer.exists():
            raise SystemExit("Pass --output-dir or create outputs/latest_qwen_long_run.txt")
        output_dir = Path(pointer.read_text(encoding="utf-8").strip())

    manifest_path = output_dir / "manifest.tsv"
    rows = read_manifest(manifest_path)
    live_sessions = tmux_sessions()
    print(f"output_dir: {output_dir}")
    print("session\tdevice\tconfig\tseed\tstatus\tstep\tloss_ema\tprobe\tdev\tid\tood")
    for row in rows:
        output_json = Path(row["output_json"])
        log_path = Path(row["log"])
        if output_json.exists():
            status = "complete"
        elif row["session"] in live_sessions:
            status = "running"
        else:
            status = "not_running"
        final = read_json(output_json) if output_json.exists() else {}
        log_state = read_last_progress(log_path)

        method = row["method"]
        k = row["k"]
        config = method if k == "-" else f"{method}_k{k}"
        step = final.get("train_steps_completed") or log_state.get("step", "")
        loss_ema = final.get("train_loss_ema") or log_state.get("loss_ema", "")
        probe = ""
        probes = final.get("train_probe_history") or []
        if probes:
            probe = probes[-1].get("accuracy", "")
        elif "train_probe_accuracy" in log_state:
            probe = log_state["train_probe_accuracy"]
        dev = (final.get("dev") or {}).get("accuracy", "")
        id_test = (final.get("id_test") or {}).get("accuracy", "")
        ood = (final.get("ood_test") or {}).get("accuracy", "")
        print(
            f"{row['session']}\t{row['ascend_device_id']}\t{config}\t{row['seed']}\t"
            f"{status}\t{step}\t{fmt(loss_ema)}\t{fmt(probe)}\t{fmt(dev)}\t{fmt(id_test)}\t{fmt(ood)}"
        )


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing manifest: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:] if line.strip()]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def tmux_sessions() -> set[str]:
    try:
        result = subprocess.run(
            ["tmux", "list-sessions", "-F", "#{session_name}"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return set()
    if result.returncode != 0:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def read_last_progress(path: Path) -> dict:
    if not path.exists():
        return {}
    state = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue
        try:
            payload = ast.literal_eval(line)
        except (SyntaxError, ValueError):
            continue
        if "step" in payload:
            state.update(payload)
    return state


def fmt(value) -> str:
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
