from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import time

from .data import build_dataset, dataset_path, debug_split_sizes, read_jsonl, smoke_split_sizes
from .formats import Method, continuous_item, format_text
from .tasks import Example, verify_answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a Qwen/Hugging Face causal LM on FDT tasks.")
    parser.add_argument("--model-name-or-path", default=os.environ.get("MODEL_NAME_OR_PATH", "Qwen/Qwen3-0.6B-Base"))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--task", default="graph_reachability")
    parser.add_argument("--method", choices=["direct", "cot", "masked_cot", "soft", "latent"], default="direct")
    parser.add_argument("--data-dir", type=Path, default=Path("data/qwen_smoke"))
    parser.add_argument("--build-data", action="store_true")
    parser.add_argument("--data-preset", choices=["smoke", "debug"], default="smoke")
    parser.add_argument(
        "--difficulty",
        choices=["standard", "easy", "easy_ladder", "hard_ladder", "simple"],
        default="easy_ladder",
    )
    parser.add_argument("--device", default="cpu", help="cpu or npu:0")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--eval-mode", choices=["generate", "binary_choice"], default="binary_choice")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--eval-examples", type=int, default=16)
    parser.add_argument(
        "--diagnostic-metadata-keys",
        default="",
        help="Comma-separated metadata keys to group dev/id/ood evaluation by. Use answer for labels.",
    )
    parser.add_argument(
        "--diagnostic-case-examples",
        type=int,
        default=0,
        help="Success/failure cases to keep per diagnostic group. Split-level cases still use --case-examples.",
    )
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--train-sampling",
        choices=["random", "balanced_answer"],
        default="random",
        help="Training example sampler. balanced_answer alternates labels when binary labels are available.",
    )
    parser.add_argument(
        "--log-interval-steps",
        type=int,
        default=0,
        help="Training progress log interval. 0 keeps the legacy quarter-run logging.",
    )
    parser.add_argument(
        "--max-train-seconds",
        type=float,
        default=0.0,
        help="Stop training after this many wall-clock seconds, then save/evaluate. 0 disables the limit.",
    )
    parser.add_argument(
        "--checkpoint-interval-steps",
        type=int,
        default=0,
        help="Save intermediate checkpoints every N optimizer steps when --save-checkpoint is set.",
    )
    parser.add_argument(
        "--train-probe-examples",
        type=int,
        default=0,
        help="Number of train examples for periodic binary-choice probe evaluation.",
    )
    parser.add_argument(
        "--train-probe-interval-steps",
        type=int,
        default=0,
        help="Run train probe every N optimizer steps. 0 disables periodic probes.",
    )
    parser.add_argument("--soft-temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--case-examples", type=int, default=2)
    parser.add_argument(
        "--include-eval-records",
        action="store_true",
        help="Write full per-example eval records to JSON. Diagnostics always use records internally.",
    )
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names for LoRA injection.",
    )
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
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
        if args.dtype == "float16" and args.steps > 0:
            print(
                "Warning: full-parameter FP16 AdamW on NPU can produce NaNs; prefer --dtype bfloat16 for Qwen SFT.",
                flush=True,
            )

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be >= 1")

    if args.build_data or not dataset_path(args.data_dir, "train", args.task).exists():
        build_dataset(args.data_dir, [args.task], split_sizes(args.data_preset), difficulty=args.difficulty)

    train_examples = read_jsonl(dataset_path(args.data_dir, "train", args.task))
    dev_examples = read_jsonl(dataset_path(args.data_dir, "dev", args.task))
    id_examples = read_jsonl(dataset_path(args.data_dir, "id_test", args.task))
    ood_examples = read_jsonl(dataset_path(args.data_dir, "ood_test", args.task))

    tokenizer, model = load_model(args)
    load_info = None
    if args.load_checkpoint is not None:
        load_info = load_training_checkpoint(model, args.load_checkpoint)
    model = model.to(args.device)
    set_latent_adapter_trainable(model, args.method == "latent")

    if args.gradient_checkpointing:
        model.model.gradient_checkpointing_enable()
        if hasattr(model.model, "enable_input_require_grads"):
            model.model.enable_input_require_grads()

    if args.freeze_backbone:
        for name, parameter in model.model.named_parameters():
            if "lora_" not in name:
                parameter.requires_grad_(False)

    trainable, total = parameter_counts(model)
    train_sampler = make_training_sampler(train_examples, args.train_sampling, args.seed)
    train_probe_examples = select_balanced_examples(train_examples, args.train_probe_examples, args.seed + 17)
    losses = []
    loss_history = []
    train_probe_history = []
    train_sample_answer_counts: dict[str, int] = defaultdict(int)
    completed_steps = 0
    stop_reason = "eval_only" if args.eval_only else "steps"
    checkpoint_saved = None
    if not args.eval_only:
        model.train()
        trainable_parameters = [p for p in model.parameters() if p.requires_grad]
        if not trainable_parameters:
            raise ValueError("No trainable parameters remain. Disable --freeze-backbone or use a trainable adapter.")
        optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr)

    start_time = time.time()
    loss_ema = None
    log_interval = args.log_interval_steps if args.log_interval_steps > 0 else max(1, args.steps // 4)
    for step in range(1, args.steps + 1):
        if args.eval_only:
            break
        if args.max_train_seconds > 0 and time.time() - start_time >= args.max_train_seconds:
            stop_reason = "max_train_seconds"
            break
        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        for micro_idx in range(args.gradient_accumulation_steps):
            example = train_sampler(step, micro_idx)
            train_sample_answer_counts[example.answer] += 1
            raw_loss = loss_for_example(model, tokenizer, example, args.method, args.k, args.device)
            (raw_loss / args.gradient_accumulation_steps).backward()
            micro_losses.append(float(raw_loss.detach().cpu()))
        optimizer.step()
        losses.append(sum(micro_losses) / len(micro_losses))
        completed_steps = step
        loss_ema = losses[-1] if loss_ema is None else 0.95 * loss_ema + 0.05 * losses[-1]

        if step == 1 or step % log_interval == 0:
            recent = losses[-min(len(losses), log_interval) :]
            progress = {
                "step": step,
                "loss": round(losses[-1], 4),
                "loss_ema": round(loss_ema, 4),
                "loss_recent_mean": round(sum(recent) / len(recent), 4),
                "elapsed_sec": round(time.time() - start_time, 1),
                "sampled_answers": dict(sorted(train_sample_answer_counts.items())),
            }
            loss_history.append(progress)
            print(progress, flush=True)

        if (
            train_probe_examples
            and args.train_probe_interval_steps > 0
            and step % args.train_probe_interval_steps == 0
        ):
            probe = evaluate(
                model,
                tokenizer,
                train_probe_examples,
                args.method,
                args.k,
                args.device,
                args.max_new_tokens,
                args.eval_mode,
                0,
            )
            probe_output = metric_for_output(probe, include_records=False)
            probe_record = {
                "step": step,
                "elapsed_sec": round(time.time() - start_time, 1),
                **probe_output,
            }
            train_probe_history.append(probe_record)
            print(
                {
                    "step": step,
                    "train_probe_accuracy": round(probe_output["accuracy"], 4),
                    "train_probe_predictions": probe_output.get("prediction_counts", {}),
                },
                flush=True,
            )

        if args.save_checkpoint is not None and args.checkpoint_interval_steps > 0 and step % args.checkpoint_interval_steps == 0:
            checkpoint_saved = save_training_checkpoint(
                model,
                args,
                trainable,
                total,
                checkpoint_path=checkpoint_path_for_step(args.save_checkpoint, step),
                completed_steps=step,
            )

    if args.save_checkpoint is not None:
        checkpoint_saved = save_training_checkpoint(
            model,
            args,
            trainable,
            total,
            completed_steps=completed_steps,
        )

    eval_splits = {
        "dev": dev_examples[: args.eval_examples],
        "id_test": id_examples[: args.eval_examples],
        "ood_test": ood_examples[: args.eval_examples],
    }
    eval_results = {
        split_name: evaluate(
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
        for split_name, examples in eval_splits.items()
    }
    output_eval_results = {
        split_name: metric_for_output(metric, include_records=args.include_eval_records)
        for split_name, metric in eval_results.items()
    }
    metrics = {
        "model_name_or_path": args.model_name_or_path,
        "task": args.task,
        "method": args.method,
        "difficulty": args.difficulty,
        "data_preset": args.data_preset,
        "eval_mode": args.eval_mode,
        "steps": args.steps,
        "train_steps_completed": completed_steps,
        "stop_reason": stop_reason,
        "max_train_seconds": args.max_train_seconds if args.max_train_seconds > 0 else None,
        "train_sampling": args.train_sampling,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "k": args.k if args.method in {"soft", "latent"} else None,
        "use_lora": args.use_lora,
        "lora_r": args.lora_r if args.use_lora else None,
        "lora_alpha": args.lora_alpha if args.use_lora else None,
        "lora_dropout": args.lora_dropout if args.use_lora else None,
        "lora_target_modules": lora_target_modules(args) if args.use_lora else None,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "train_loss_last": losses[-1] if losses else None,
        "train_loss_ema": loss_ema,
        "loss_history": loss_history,
        "train_sample_answer_counts": dict(sorted(train_sample_answer_counts.items())),
        "train_probe_examples": len(train_probe_examples),
        "train_probe_interval_steps": args.train_probe_interval_steps,
        "train_probe_history": train_probe_history,
        "elapsed_sec": round(time.time() - start_time, 3),
        "checkpoint_loaded": str(args.load_checkpoint) if args.load_checkpoint is not None else None,
        "checkpoint_load_info": load_info,
        "checkpoint_saved": checkpoint_saved,
        **output_eval_results,
    }
    diagnostics = run_diagnostics(args, eval_results)
    if diagnostics:
        metrics["diagnostics"] = diagnostics
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def load_model(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        dtype=dtype_map[args.dtype],
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if args.use_lora:
        model = apply_lora(model, args)
    return tokenizer, HFContinuousWrapper(model, soft_temperature=args.soft_temperature)


def split_sizes(preset: str):
    if preset == "smoke":
        return smoke_split_sizes()
    if preset == "debug":
        return debug_split_sizes()
    raise ValueError(f"Unknown data preset: {preset}")


def make_training_sampler(examples: list[Example], sampling: str, seed: int):
    rng = random.Random(seed)
    if sampling == "random":
        return lambda step, micro_idx: rng.choice(examples)
    if sampling != "balanced_answer":
        raise ValueError(f"Unknown train sampling mode: {sampling}")

    buckets: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        buckets[example.answer].append(example)
    labels = sorted(label for label, bucket in buckets.items() if bucket)
    if len(labels) < 2:
        return lambda step, micro_idx: rng.choice(examples)

    def sample(step: int, micro_idx: int) -> Example:
        label = labels[((step - 1) + micro_idx) % len(labels)]
        return rng.choice(buckets[label])

    return sample


def select_balanced_examples(examples: list[Example], count: int, seed: int) -> list[Example]:
    if count <= 0:
        return []

    rng = random.Random(seed)
    buckets: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        buckets[example.answer].append(example)
    labels = sorted(label for label, bucket in buckets.items() if bucket)
    if len(labels) < 2:
        sample = list(examples)
        rng.shuffle(sample)
        return sample[:count]

    selected = []
    per_label = max(1, count // len(labels))
    for label in labels:
        bucket = list(buckets[label])
        rng.shuffle(bucket)
        selected.extend(bucket[:per_label])
    remaining = [example for example in examples if example not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, count - len(selected))])
    rng.shuffle(selected)
    return selected[:count]


def checkpoint_path_for_step(path: Path, step: int) -> Path:
    return path.with_name(f"{path.stem}_step{step}{path.suffix}")


def lora_target_modules(args: argparse.Namespace) -> list[str]:
    return [name.strip() for name in args.lora_target_modules.split(",") if name.strip()]


def apply_lora(model, args: argparse.Namespace):
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_target_modules(args),
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def parameter_counts(model) -> tuple[int, int]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return trainable, total


def set_latent_adapter_trainable(model, trainable: bool) -> None:
    for module in (model.latent_norm, model.latent_proj):
        for parameter in module.parameters():
            parameter.requires_grad_(trainable)


def continuous_adapter_state(model) -> dict:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key.startswith("latent_norm.") or key.startswith("latent_proj.")
    }


def save_training_checkpoint(
    model,
    args: argparse.Namespace,
    trainable_parameters: int,
    total_parameters: int,
    checkpoint_path: Path | None = None,
    completed_steps: int | None = None,
) -> str:
    import torch

    path = checkpoint_path or args.save_checkpoint
    if path is None:
        raise ValueError("checkpoint_path is required when args.save_checkpoint is not set")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_format": "fdt_qwen_v1",
        "model_name_or_path": args.model_name_or_path,
        "task": args.task,
        "method": args.method,
        "difficulty": args.difficulty,
        "data_preset": args.data_preset,
        "k": args.k if args.method in {"soft", "latent"} else None,
        "steps": args.steps,
        "completed_steps": completed_steps,
        "train_sampling": getattr(args, "train_sampling", "random"),
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "seed": args.seed,
        "use_lora": args.use_lora,
        "lora_r": args.lora_r if args.use_lora else None,
        "lora_alpha": args.lora_alpha if args.use_lora else None,
        "lora_dropout": args.lora_dropout if args.use_lora else None,
        "lora_target_modules": lora_target_modules(args) if args.use_lora else None,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "continuous_adapter_state": continuous_adapter_state(model),
    }
    if args.use_lora:
        from peft import get_peft_model_state_dict

        payload["lora_state"] = {
            key: value.detach().cpu() for key, value in get_peft_model_state_dict(model.model).items()
        }
        payload["model_state"] = None
    else:
        payload["lora_state"] = None
        payload["model_state"] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(payload, path)
    return str(path)


def load_training_checkpoint(model, checkpoint_path: Path) -> dict:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    info = {
        "checkpoint_format": checkpoint.get("checkpoint_format", "unknown"),
        "use_lora": checkpoint.get("use_lora"),
    }
    if checkpoint.get("lora_state") is not None:
        from peft import set_peft_model_state_dict

        if not hasattr(model.model, "peft_config"):
            raise ValueError("Checkpoint contains LoRA weights, but the current model was not created with --use-lora.")
        set_peft_model_state_dict(model.model, checkpoint["lora_state"])
        info["lora_keys"] = len(checkpoint["lora_state"])
    elif checkpoint.get("model_state") is not None:
        result = model.load_state_dict(checkpoint["model_state"], strict=False)
        info["missing_keys"] = len(result.missing_keys)
        info["unexpected_keys"] = len(result.unexpected_keys)

    continuous_state = checkpoint.get("continuous_adapter_state")
    if continuous_state:
        model.load_state_dict(continuous_state, strict=False)
        info["continuous_adapter_keys"] = sorted(continuous_state.keys())
    return info


def HFContinuousWrapper(model, soft_temperature: float = 1.0):
    """Create a continuous-thinking adapter after torch has been imported."""

    import torch.nn as nn

    class _HFContinuousWrapper(nn.Module):
        def __init__(self, inner_model, temperature: float) -> None:
            super().__init__()
            self.model = inner_model
            self.soft_temperature = temperature
            hidden_size = getattr(inner_model.config, "hidden_size", None) or getattr(inner_model.config, "n_embd")
            self.latent_norm = nn.LayerNorm(hidden_size)
            self.latent_proj = nn.Linear(hidden_size, hidden_size)
            nn.init.eye_(self.latent_proj.weight)
            nn.init.zeros_(self.latent_proj.bias)

        def continuous_loss(self, prefix_ids, answer_ids, num_steps: int, mode: Method):
            import torch.nn.functional as F

            logits = self._continuous_logits(prefix_ids, answer_ids[:-1], num_steps, mode)
            start = prefix_ids.numel() + num_steps - 1
            end = start + answer_ids.numel()
            supervised_logits = logits[:, start:end, :].squeeze(0).float()
            return F.cross_entropy(supervised_logits, answer_ids)

        def continuous_candidate_nll(self, prefix_ids, candidate_ids, num_steps: int, mode: Method):
            import torch.nn.functional as F

            logits = self._continuous_logits(prefix_ids, candidate_ids[:-1], num_steps, mode).squeeze(0)
            start = prefix_ids.numel() + num_steps - 1
            end = start + candidate_ids.numel()
            token_losses = F.cross_entropy(logits[start:end, :].float(), candidate_ids, reduction="none")
            return token_losses.mean()

        def generate_continuous(
            self,
            prefix_ids,
            num_steps: int,
            mode: Method,
            max_new_tokens: int,
            eos_token_id: int | None,
        ):
            import torch

            seq_embeds = self.model.get_input_embeddings()(prefix_ids.unsqueeze(0))
            seq_embeds = self._append_continuous_steps(seq_embeds, num_steps, mode)

            generated = []
            for _ in range(max_new_tokens):
                outputs = self.model(inputs_embeds=seq_embeds, use_cache=False)
                next_id = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())
                generated.append(next_id)
                next_id_tensor = torch.tensor([[next_id]], device=seq_embeds.device, dtype=prefix_ids.dtype)
                next_embed = self.model.get_input_embeddings()(next_id_tensor)
                seq_embeds = torch.cat([seq_embeds, next_embed], dim=1)
                if eos_token_id is not None and next_id == eos_token_id:
                    break
            return generated

        def _continuous_logits(self, prefix_ids, suffix_input_ids, num_steps: int, mode: Method):
            import torch

            seq_embeds = self.model.get_input_embeddings()(prefix_ids.unsqueeze(0))
            seq_embeds = self._append_continuous_steps(seq_embeds, num_steps, mode)
            if suffix_input_ids.numel() > 0:
                suffix_embeds = self.model.get_input_embeddings()(suffix_input_ids.unsqueeze(0))
                seq_embeds = torch.cat([seq_embeds, suffix_embeds], dim=1)
            return self.model(inputs_embeds=seq_embeds, use_cache=False).logits

        def _append_continuous_steps(self, seq_embeds, num_steps: int, mode: Method):
            import torch

            for _ in range(num_steps):
                outputs = self.model(inputs_embeds=seq_embeds, output_hidden_states=True, use_cache=False)
                next_embed = self._next_continuous_embed(outputs, mode)
                seq_embeds = torch.cat([seq_embeds, next_embed.unsqueeze(1)], dim=1)
            return seq_embeds

        def _next_continuous_embed(self, outputs, mode: Method):
            import torch

            if mode == "latent":
                last_hidden = outputs.hidden_states[-1][:, -1, :]
                return self.latent_proj(self.latent_norm(last_hidden))
            if mode == "soft":
                temperature = max(self.soft_temperature, 1e-6)
                probs = torch.softmax(outputs.logits[:, -1, :].float() / temperature, dim=-1)
                embedding_weight = self.model.get_input_embeddings().weight
                return probs.to(embedding_weight.dtype) @ embedding_weight
            raise ValueError(f"Unknown continuous mode: {mode}")

    return _HFContinuousWrapper(model, soft_temperature)


def loss_for_example(model, tokenizer, example: Example, method: Method, k: int, device: str):
    import torch

    if method in {"direct", "cot", "masked_cot"}:
        item = format_text(example, method)
        encoded = encode_with_offsets(tokenizer, item.text)
        ids = torch.tensor(encoded["input_ids"], device=device, dtype=torch.long)
        labels = ids.clone()
        loss_mask = loss_mask_from_offsets(encoded, item.loss_start, device)
        if loss_mask is None:
            prefix_len = len(tokenizer(item.text[: item.loss_start], add_special_tokens=False)["input_ids"])
            labels[:prefix_len] = -100
        else:
            labels[loss_mask] = -100
        outputs = model.model(input_ids=ids.unsqueeze(0), use_cache=False)
        return causal_lm_loss(outputs.logits, labels.unsqueeze(0))

    item = continuous_item(example)
    prefix_ids = encode(tokenizer, item.prefix, device)
    answer_ids = encode(tokenizer, item.answer, device)
    return model.continuous_loss(prefix_ids, answer_ids, num_steps=k, mode=method)


def evaluate(model, tokenizer, examples, method, k, device, max_new_tokens, eval_mode, case_examples):
    model.eval()
    if eval_mode == "binary_choice" and all(example.answer in {"YES", "NO"} for example in examples):
        result = evaluate_binary_choice(model, tokenizer, examples, method, k, device, max_new_tokens, case_examples)
        model.train()
        return result

    import torch

    records = []
    with torch.no_grad():
        for example in examples:
            if method == "direct":
                prefix = f"Problem:\n{example.prompt}\nAnswer: "
                generated = generate_text(model.model, tokenizer, encode(tokenizer, prefix, device), max_new_tokens)
            elif method in {"soft", "latent"}:
                item = continuous_item(example)
                generated_ids = model.generate_continuous(
                    encode(tokenizer, item.prefix, device),
                    num_steps=k,
                    mode=method,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                )
                generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
            else:
                prefix = f"Problem:\n{example.prompt}\nReasoning: "
                generated = generate_text(model.model, tokenizer, encode(tokenizer, prefix, device), max_new_tokens)

            answer = extract_answer(generated)
            ok = verify_answer(example, answer)
            records.append(
                {
                    "expected": example.answer,
                    "generated": generated[:160],
                    "parsed": answer,
                    "ok": ok,
                    "prompt": example.prompt[:500],
                    "metadata": example.metadata,
                }
            )
    model.train()
    return records_to_metric(records, case_examples, include_records=True)


def evaluate_binary_choice(model, tokenizer, examples, method, k, device, max_trace_tokens, case_examples):
    import torch

    choices = ["YES", "NO"]
    candidate_ids = {choice: encode(tokenizer, f"{choice}\n", device) for choice in choices}
    records = []
    with torch.no_grad():
        for example in examples:
            generated_trace = None
            if method == "direct":
                prefix = f"Problem:\n{example.prompt}\nAnswer: "
                scores = {
                    choice: float(candidate_nll(model.model, tokenizer, prefix, ids, device).detach().cpu())
                    for choice, ids in candidate_ids.items()
                }
            elif method in {"soft", "latent"}:
                item = continuous_item(example)
                prefix_ids = encode(tokenizer, item.prefix, device)
                scores = {
                    choice: float(model.continuous_candidate_nll(prefix_ids, ids, num_steps=k, mode=method).detach().cpu())
                    for choice, ids in candidate_ids.items()
                }
            else:
                trace_prefix = f"Problem:\n{example.prompt}\nReasoning: "
                generated_trace = generate_text(model.model, tokenizer, encode(tokenizer, trace_prefix, device), max_trace_tokens)
                generated_trace = generated_trace.split("Answer:", 1)[0].rstrip()
                answer_prefix = f"{trace_prefix}{generated_trace}\nAnswer: "
                scores = {
                    choice: float(candidate_nll(model.model, tokenizer, answer_prefix, ids, device).detach().cpu())
                    for choice, ids in candidate_ids.items()
                }

            answer = min(scores, key=scores.get)
            ok = verify_answer(example, answer)
            record = {
                "expected": example.answer,
                "parsed": answer,
                "scores": scores,
                "ok": ok,
                "prompt": example.prompt[:500],
                "metadata": example.metadata,
            }
            if generated_trace is not None:
                record["generated_trace"] = generated_trace[:160]
            records.append(record)
    return records_to_metric(records, case_examples, include_records=True)


def records_to_metric(records: list[dict], case_examples: int, include_records: bool = False) -> dict:
    correct = sum(1 for record in records if record.get("ok"))
    samples = [sample_from_record(record) for record in records[:5]]
    cases = {"success": [], "failure": []}
    for record in records:
        record_case(cases, record, bool(record.get("ok")), case_examples)
    metric = {
        "accuracy": correct / max(1, len(records)),
        "num_examples": len(records),
        "samples": samples,
        "cases": cases,
        "expected_counts": count_record_values(records, "expected"),
        "prediction_counts": count_record_values(records, "parsed"),
    }
    yes_no_margin = yes_minus_no_nll_margin(records)
    if yes_no_margin is not None:
        metric["mean_yes_minus_no_nll"] = yes_no_margin
    if include_records:
        metric["records"] = records
    return metric


def count_record_values(records: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[str(record.get(key, "missing"))] += 1
    return dict(sorted(counts.items()))


def yes_minus_no_nll_margin(records: list[dict]) -> float | None:
    margins = []
    for record in records:
        scores = record.get("scores")
        if isinstance(scores, dict) and "YES" in scores and "NO" in scores:
            margins.append(float(scores["YES"]) - float(scores["NO"]))
    if not margins:
        return None
    return sum(margins) / len(margins)


def metric_for_output(metric: dict, include_records: bool) -> dict:
    if include_records:
        return metric
    return {key: value for key, value in metric.items() if key != "records"}


def sample_from_record(record: dict) -> dict:
    keys = ["expected", "generated", "parsed", "scores", "generated_trace", "ok"]
    return {key: record[key] for key in keys if key in record}


def run_diagnostics(args: argparse.Namespace, eval_results: dict[str, dict]) -> dict:
    diagnostics = {}
    if not args.diagnostic_metadata_keys:
        return diagnostics

    keys = [key.strip() for key in args.diagnostic_metadata_keys.split(",") if key.strip()]
    for split_name, metric in eval_results.items():
        records = metric.get("records", [])
        for key in keys:
            groups: dict[str, list[dict]] = defaultdict(list)
            for record in records:
                groups[diagnostic_record_value(record, key)].append(record)
            for value, group_records in sorted(groups.items()):
                diagnostics[f"{split_name}_{slug(key)}_{slug(value)}"] = records_to_metric(
                    group_records,
                    getattr(args, "diagnostic_case_examples", 0),
                    include_records=False,
                )
    return diagnostics


def diagnostic_record_value(record: dict, key: str) -> str:
    if key == "answer":
        return str(record.get("expected", "missing"))
    metadata = record.get("metadata") or {}
    value = metadata.get(key, "missing")
    if value is None:
        return "none"
    return str(value)


def slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")


def candidate_nll(model, tokenizer, prefix: str, candidate_ids, device: str):
    ids = encode(tokenizer, prefix, device)
    input_ids = concat_ids(ids, candidate_ids)
    labels = input_ids.clone()
    labels[: ids.numel()] = -100
    outputs = model(input_ids=input_ids.unsqueeze(0), use_cache=False)
    return causal_lm_loss(outputs.logits, labels.unsqueeze(0))


def causal_lm_loss(logits, labels):
    import torch.nn.functional as F

    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    valid = shift_labels != -100
    if not bool(valid.any()):
        raise ValueError("No supervised tokens available for causal LM loss.")
    losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    )
    return losses[valid.reshape(-1)].mean()


def generate_text(model, tokenizer, prefix_ids, max_new_tokens: int) -> str:
    output_ids = model.generate(
        input_ids=prefix_ids.unsqueeze(0),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )[0, prefix_ids.numel() :]
    return tokenizer.decode(output_ids.tolist(), skip_special_tokens=True)


def encode(tokenizer, text: str, device: str):
    import torch

    return torch.tensor(tokenizer(text, add_special_tokens=False)["input_ids"], device=device, dtype=torch.long)


def encode_with_offsets(tokenizer, text: str) -> dict:
    try:
        return tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    except NotImplementedError:
        return tokenizer(text, add_special_tokens=False)


def loss_mask_from_offsets(encoded: dict, loss_start: int, device: str):
    import torch

    offsets = encoded.get("offset_mapping")
    if offsets is None:
        return None
    return torch.tensor([end <= loss_start for _, end in offsets], device=device, dtype=torch.bool)


def concat_ids(left, right):
    import torch

    return torch.cat([left, right], dim=0)


def extract_answer(text: str) -> str:
    if "Answer:" in text:
        text = text.split("Answer:", 1)[1]
    return text.strip().splitlines()[0].strip() if text.strip() else ""


def record_case(cases: dict, sample: dict, ok: bool, limit: int) -> None:
    if limit <= 0:
        return
    bucket = "success" if ok else "failure"
    if len(cases[bucket]) < limit:
        cases[bucket].append(sample)


if __name__ == "__main__":
    main()
