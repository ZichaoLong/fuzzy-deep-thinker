from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .tasks import Example, generate_example, list_tasks, write_jsonl


SPLIT_SEED_START = {
    "train": 0,
    "dev": 1_000_000,
    "id_test": 2_000_000,
    "ood_test": 3_000_000,
}


@dataclass(frozen=True)
class SplitSizes:
    train: int
    dev: int
    id_test: int
    ood_test: int

    def as_dict(self) -> dict[str, int]:
        return {
            "train": self.train,
            "dev": self.dev,
            "id_test": self.id_test,
            "ood_test": self.ood_test,
        }


def debug_split_sizes() -> SplitSizes:
    return SplitSizes(train=2_000, dev=200, id_test=200, ood_test=200)


def smoke_split_sizes() -> SplitSizes:
    return SplitSizes(train=128, dev=32, id_test=32, ood_test=32)


def example_from_dict(payload: dict) -> Example:
    return Example(
        prompt=payload["prompt"],
        trace=payload["trace"],
        answer=payload["answer"],
        metadata=payload["metadata"],
    )


def read_jsonl(path: Path) -> list[Example]:
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(example_from_dict(json.loads(line)))
    return examples


def dataset_path(root: Path, split: str, task: str) -> Path:
    return root / split / f"{task}.jsonl"


def build_split(
    root: Path,
    task: str,
    split: str,
    num_examples: int,
    seed_start: int | None = None,
    difficulty: str = "standard",
) -> Path:
    start = SPLIT_SEED_START[split] if seed_start is None else seed_start
    examples = [generate_example(task, start + i, split, difficulty=difficulty) for i in range(num_examples)]
    path = dataset_path(root, split, task)
    write_jsonl(examples, path)
    return path


def build_dataset(root: Path, tasks: Iterable[str], sizes: SplitSizes, difficulty: str = "standard") -> list[Path]:
    paths = []
    for task in tasks:
        for split, count in sizes.as_dict().items():
            paths.append(build_split(root, task, split, count, difficulty=difficulty))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize FDT synthetic train/dev/test splits.")
    parser.add_argument("--task", choices=[*list_tasks(), "all"], default="graph_reachability")
    parser.add_argument("--preset", choices=["smoke", "debug"], default="smoke")
    parser.add_argument("--difficulty", choices=["standard", "easy", "easy_ladder", "simple"], default="standard")
    parser.add_argument("--out-dir", type=Path, default=Path("data/phase1a"))
    args = parser.parse_args()

    tasks = list_tasks() if args.task == "all" else [args.task]
    sizes = smoke_split_sizes() if args.preset == "smoke" else debug_split_sizes()
    for path in build_dataset(args.out_dir, tasks, sizes, difficulty=args.difficulty):
        print(path)


if __name__ == "__main__":
    main()
