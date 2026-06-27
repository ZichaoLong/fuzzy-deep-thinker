from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time

from .data import build_dataset, dataset_path, read_jsonl, smoke_split_sizes
from .formats import Method, continuous_item, format_text
from .tasks import Example, generate_easy_graph_reachability_fixed_nodes, verify_answer
from .tokenizer import CharTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny decoder on FDT synthetic tasks.")
    parser.add_argument("--task", default="graph_reachability")
    parser.add_argument("--method", choices=["direct", "cot", "masked_cot", "soft", "latent"], default="direct")
    parser.add_argument("--data-dir", type=Path, default=Path("data/phase1a_smoke"))
    parser.add_argument("--build-data", action="store_true")
    parser.add_argument("--difficulty", choices=["standard", "easy", "easy_ladder", "simple"], default="standard")
    parser.add_argument("--device", default="cpu", help="cpu or npu:0")
    parser.add_argument("--eval-mode", choices=["generate", "binary_choice"], default="generate")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--eval-examples", type=int, default=32)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--easy-graph-diagnostic-nodes",
        default="",
        help="Comma-separated fixed node counts for graph_reachability/easy diagnostic evaluation.",
    )
    parser.add_argument("--diagnostic-examples", type=int, default=0)
    parser.add_argument(
        "--diagnostic-metadata-keys",
        default="",
        help="Comma-separated metadata keys to group dev/id/ood evaluation by. Use answer for labels.",
    )
    parser.add_argument("--case-examples", type=int, default=2, help="Success/failure cases to keep per evaluation.")
    parser.add_argument("--save-checkpoint", type=Path, default=None)
    parser.add_argument("--load-checkpoint", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.device == "cpu":
        os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

    import torch

    if args.device.startswith("npu"):
        import torch_npu

        torch_npu.npu.set_device(torch.device(args.device))

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.build_data or not dataset_path(args.data_dir, "train", args.task).exists():
        build_dataset(args.data_dir, [args.task], smoke_split_sizes(), difficulty=args.difficulty)

    train_examples = read_jsonl(dataset_path(args.data_dir, "train", args.task))
    dev_examples = read_jsonl(dataset_path(args.data_dir, "dev", args.task))
    id_examples = read_jsonl(dataset_path(args.data_dir, "id_test", args.task))
    ood_examples = read_jsonl(dataset_path(args.data_dir, "ood_test", args.task))

    from .tiny_model import TinyDecoder, TinyDecoderConfig

    checkpoint = None
    if args.load_checkpoint is not None:
        checkpoint = torch.load(args.load_checkpoint, map_location="cpu")
        tokenizer = CharTokenizer(checkpoint["tokenizer"])
        config = TinyDecoderConfig(**checkpoint["model_config"])
    else:
        tokenizer = build_tokenizer(train_examples + dev_examples + id_examples + ood_examples)
        config = TinyDecoderConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
        )
    model = TinyDecoder(config)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"])
    model = model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_time = time.time()
    losses = []
    if not args.eval_only:
        for step in range(1, args.steps + 1):
            example = random.choice(train_examples)
            loss = _loss_for_example(model, tokenizer, example, args.method, args.k, args.device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if step == 1 or step % max(1, args.steps // 4) == 0:
                print({"step": step, "loss": round(losses[-1], 4)}, flush=True)

    if args.save_checkpoint is not None:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "model_config": asdict(config),
                "tokenizer": tokenizer.token_to_id,
                "task": args.task,
                "method": args.method,
                "difficulty": args.difficulty,
                "k": args.k if args.method in {"soft", "latent"} else None,
                "steps": args.steps,
                "seed": args.seed,
            },
            args.save_checkpoint,
        )

    eval_splits = {
        "dev": dev_examples[: args.eval_examples],
        "id_test": id_examples[: args.eval_examples],
        "ood_test": ood_examples[: args.eval_examples],
    }
    metrics = {
        "task": args.task,
        "method": args.method,
        "difficulty": args.difficulty,
        "eval_mode": args.eval_mode,
        "steps": args.steps,
        "k": args.k if args.method in {"soft", "latent"} else None,
        "train_loss_last": losses[-1] if losses else None,
        "elapsed_sec": round(time.time() - start_time, 3),
        "checkpoint_loaded": str(args.load_checkpoint) if args.load_checkpoint is not None else None,
        "checkpoint_saved": str(args.save_checkpoint) if args.save_checkpoint is not None else None,
        "dev": evaluate(
            model,
            tokenizer,
            eval_splits["dev"],
            args.method,
            args.k,
            args.device,
            args.max_new_tokens,
            args.eval_mode,
            args.case_examples,
        ),
        "id_test": evaluate(
            model,
            tokenizer,
            eval_splits["id_test"],
            args.method,
            args.k,
            args.device,
            args.max_new_tokens,
            args.eval_mode,
            args.case_examples,
        ),
        "ood_test": evaluate(
            model,
            tokenizer,
            eval_splits["ood_test"],
            args.method,
            args.k,
            args.device,
            args.max_new_tokens,
            args.eval_mode,
            args.case_examples,
        ),
    }
    diagnostics = _run_diagnostics(model, tokenizer, args, eval_splits)
    if diagnostics:
        metrics["diagnostics"] = diagnostics
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def build_tokenizer(examples: list[Example]) -> CharTokenizer:
    texts = []
    for example in examples:
        for method in ["direct", "cot", "masked_cot"]:
            texts.append(format_text(example, method).text)
        item = continuous_item(example)
        texts.append(item.prefix + item.answer)
    return CharTokenizer.build(texts)


def _loss_for_example(model, tokenizer: CharTokenizer, example: Example, method: Method, k: int, device: str):
    import torch

    if method in {"direct", "cot", "masked_cot"}:
        item = format_text(example, method)
        ids = torch.tensor(tokenizer.encode(item.text), device=device, dtype=torch.long)
        input_ids = ids[:-1].unsqueeze(0)
        labels = ids[1:].unsqueeze(0)
        char_positions = torch.arange(1, ids.numel(), device=device)
        label_mask = (char_positions >= item.loss_start).unsqueeze(0)
        return model.text_loss(input_ids, labels, label_mask)

    item = continuous_item(example)
    prefix_ids = torch.tensor(tokenizer.encode(item.prefix), device=device, dtype=torch.long)
    answer_ids = torch.tensor(tokenizer.encode(item.answer), device=device, dtype=torch.long)
    return model.continuous_loss(prefix_ids, answer_ids, num_steps=k, mode=method)


def evaluate(
    model,
    tokenizer: CharTokenizer,
    examples: list[Example],
    method: Method,
    k: int,
    device: str,
    max_new_tokens: int,
    eval_mode: str,
    case_examples: int = 2,
) -> dict:
    import torch

    model.eval()
    if eval_mode == "binary_choice" and _is_binary_answer_set(examples):
        result = evaluate_binary_choice(model, tokenizer, examples, method, k, device, max_new_tokens, case_examples)
        model.train()
        return result

    correct = 0
    predictions = []
    cases = {"success": [], "failure": []}
    with torch.no_grad():
        for example in examples:
            if method == "direct":
                prefix = f"Problem:\n{example.prompt}\nAnswer: "
                prefix_ids = torch.tensor(tokenizer.encode(prefix), device=device, dtype=torch.long)
                generated = tokenizer.decode(model.generate_text(prefix_ids, max_new_tokens=max_new_tokens))
            elif method in {"soft", "latent"}:
                item = continuous_item(example)
                prefix_ids = torch.tensor(tokenizer.encode(item.prefix), device=device, dtype=torch.long)
                generated = tokenizer.decode(
                    model.generate_continuous(prefix_ids, num_steps=k, mode=method, max_new_tokens=max_new_tokens)
                )
            else:
                prefix = f"Problem:\n{example.prompt}\nReasoning: "
                prefix_ids = torch.tensor(tokenizer.encode(prefix), device=device, dtype=torch.long)
                generated = tokenizer.decode(model.generate_text(prefix_ids, max_new_tokens=max_new_tokens))

            answer = extract_answer(generated)
            ok = verify_answer(example, answer)
            correct += int(ok)
            if len(predictions) < 5:
                predictions.append({"expected": example.answer, "generated": generated[:120], "parsed": answer, "ok": ok})
            _record_case(
                cases,
                {
                    "prompt": example.prompt[:500],
                    "metadata": example.metadata,
                    "expected": example.answer,
                    "parsed": answer,
                    "generated": generated[:240],
                    "ok": ok,
                },
                ok,
                case_examples,
            )
    model.train()
    return {
        "accuracy": correct / max(1, len(examples)),
        "num_examples": len(examples),
        "samples": predictions,
        "cases": cases,
    }


def evaluate_binary_choice(
    model,
    tokenizer: CharTokenizer,
    examples: list[Example],
    method: Method,
    k: int,
    device: str,
    max_trace_tokens: int,
    case_examples: int = 2,
) -> dict:
    import torch

    choices = ["YES", "NO"]
    candidate_ids = {
        choice: torch.tensor(tokenizer.encode(f"{choice}\n"), device=device, dtype=torch.long) for choice in choices
    }
    correct = 0
    predictions = []
    cases = {"success": [], "failure": []}

    with torch.no_grad():
        for example in examples:
            if method == "direct":
                prefix = f"Problem:\n{example.prompt}\nAnswer: "
                prefix_ids = torch.tensor(tokenizer.encode(prefix), device=device, dtype=torch.long)
                scores = {
                    choice: float(model.candidate_nll(prefix_ids, ids).detach().cpu())
                    for choice, ids in candidate_ids.items()
                }
            elif method in {"soft", "latent"}:
                item = continuous_item(example)
                prefix_ids = torch.tensor(tokenizer.encode(item.prefix), device=device, dtype=torch.long)
                scores = {
                    choice: float(
                        model.continuous_candidate_nll(prefix_ids, ids, num_steps=k, mode=method).detach().cpu()
                    )
                    for choice, ids in candidate_ids.items()
                }
            elif method in {"cot", "masked_cot"}:
                trace_prefix = f"Problem:\n{example.prompt}\nReasoning: "
                trace_prefix_ids = torch.tensor(tokenizer.encode(trace_prefix), device=device, dtype=torch.long)
                generated_trace = tokenizer.decode(model.generate_text(trace_prefix_ids, max_new_tokens=max_trace_tokens))
                generated_trace = generated_trace.split("Answer:", 1)[0].rstrip()
                answer_prefix = f"{trace_prefix}{generated_trace}\nAnswer: "
                answer_prefix_ids = torch.tensor(tokenizer.encode(answer_prefix), device=device, dtype=torch.long)
                scores = {
                    choice: float(model.candidate_nll(answer_prefix_ids, ids).detach().cpu())
                    for choice, ids in candidate_ids.items()
                }
            else:
                raise ValueError(f"binary_choice eval is not supported for method={method}")

            answer = min(scores, key=scores.get)
            ok = verify_answer(example, answer)
            correct += int(ok)
            if len(predictions) < 5:
                sample = {"expected": example.answer, "parsed": answer, "scores": scores, "ok": ok}
                if method in {"cot", "masked_cot"}:
                    sample["generated_trace"] = generated_trace[:160]
                predictions.append(sample)
            case = {
                "prompt": example.prompt[:500],
                "metadata": example.metadata,
                "expected": example.answer,
                "parsed": answer,
                "scores": scores,
                "ok": ok,
            }
            if method in {"cot", "masked_cot"}:
                case["generated_trace"] = generated_trace[:240]
            _record_case(cases, case, ok, case_examples)

    return {
        "accuracy": correct / max(1, len(examples)),
        "num_examples": len(examples),
        "samples": predictions,
        "cases": cases,
    }


def _is_binary_answer_set(examples: list[Example]) -> bool:
    return all(example.answer in {"YES", "NO"} for example in examples)


def _run_diagnostics(model, tokenizer: CharTokenizer, args, eval_splits: dict[str, list[Example]]) -> dict:
    diagnostics = {}
    if args.diagnostic_metadata_keys:
        keys = [key.strip() for key in args.diagnostic_metadata_keys.split(",") if key.strip()]
        for split_name, examples in eval_splits.items():
            for key in keys:
                groups: dict[str, list[Example]] = defaultdict(list)
                for example in examples:
                    value = _diagnostic_value(example, key)
                    groups[value].append(example)
                for value, group in sorted(groups.items()):
                    diagnostics[f"{split_name}_{_slug(key)}_{_slug(value)}"] = evaluate(
                        model,
                        tokenizer,
                        group,
                        args.method,
                        args.k,
                        args.device,
                        args.max_new_tokens,
                        args.eval_mode,
                        args.case_examples,
                    )

    if not args.easy_graph_diagnostic_nodes or args.diagnostic_examples <= 0:
        return diagnostics
    if args.task != "graph_reachability" or args.difficulty not in {"easy", "easy_ladder"}:
        raise ValueError("--easy-graph-diagnostic-nodes is only supported for graph_reachability easy variants")

    for raw_n in args.easy_graph_diagnostic_nodes.split(","):
        raw_n = raw_n.strip()
        if not raw_n:
            continue
        n = int(raw_n)
        examples = [
            generate_easy_graph_reachability_fixed_nodes(4_000_000 + n * 10_000 + i, n=n, difficulty=args.difficulty)
            for i in range(args.diagnostic_examples)
        ]
        diagnostics[f"easy_n{n}"] = evaluate(
            model,
            tokenizer,
            examples,
            args.method,
            args.k,
            args.device,
            args.max_new_tokens,
            args.eval_mode,
            args.case_examples,
        )
    return diagnostics


def _diagnostic_value(example: Example, key: str) -> str:
    if key == "answer":
        return example.answer
    return str(example.metadata.get(key, "missing"))


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")


def _record_case(cases: dict, sample: dict, ok: bool, limit: int) -> None:
    if limit <= 0:
        return
    bucket = "success" if ok else "failure"
    if len(cases[bucket]) < limit:
        cases[bucket].append(sample)


def extract_answer(text: str) -> str:
    if "Answer:" in text:
        text = text.split("Answer:", 1)[1]
    return text.strip().splitlines()[0].strip() if text.strip() else ""


if __name__ == "__main__":
    main()
