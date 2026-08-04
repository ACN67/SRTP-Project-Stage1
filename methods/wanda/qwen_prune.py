#!/usr/bin/env python3
"""Project wrapper for Wanda/Magnitude Qwen2.5-Coder pruning."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TARGET_LINEAR_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def sha256_jsonl_rows(rows: list[dict[str, Any]]) -> str:
    material = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def load_guide(path: Path, max_samples: int) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= max_samples:
                break
    if not rows:
        raise ValueError(f"guide file is empty: {path}")
    for row in rows:
        if row.get("contains_solution"):
            raise ValueError(f"guide row contains_solution=true: {row.get('task_id')}")
    return rows, sha256_jsonl_rows(rows)


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def build_calibration_text(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            "### Benchmark",
            str(row.get("benchmark", "")),
            "",
            "### Task",
            str(row.get("prompt", "")),
            "",
            "### Instruction",
            "Complete the requested code task. Do not use unavailable tests or reference solutions.",
        ]
    )


def expected_qwen_layout(config: Any) -> dict[str, Any]:
    return {
        "model_type": getattr(config, "model_type", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "intermediate_size": getattr(config, "intermediate_size", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "target_linear_names": sorted(TARGET_LINEAR_NAMES),
    }


def find_target_linears(model: Any) -> list[tuple[str, Any]]:
    import torch.nn as nn

    modules: list[tuple[str, Any]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in TARGET_LINEAR_NAMES:
            modules.append((name, module))
    return modules


def collect_activation_scales(model: Any, tokenizer: Any, guide_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    import torch

    modules = dict(find_target_linears(model))
    sums: dict[str, Any] = {}
    counts: dict[str, int] = {}

    def make_hook(name: str):
        def hook(_module: Any, inputs: tuple[Any, ...], _output: Any) -> None:
            if not inputs:
                return
            x = inputs[0].detach().float()
            x = x.reshape(-1, x.shape[-1])
            value = x.pow(2).sum(dim=0).cpu()
            sums[name] = value if name not in sums else sums[name] + value
            counts[name] = counts.get(name, 0) + x.shape[0]

        return hook

    handles = [module.register_forward_hook(make_hook(name)) for name, module in modules.items()]
    try:
        for row in guide_rows:
            text = build_calibration_text(row)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_seq_len)
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            with torch.no_grad():
                model(**inputs)
    finally:
        for handle in handles:
            handle.remove()

    scales = {}
    for name, value in sums.items():
        denom = max(1, counts.get(name, 0))
        scales[name] = torch.sqrt(value / denom + 1e-12)
    return scales


def prune_linear(weight: Any, ratio: float, metric: Any) -> tuple[int, int]:
    import torch

    if not (0.0 < ratio < 1.0):
        raise ValueError("--sparsity-ratio must be in (0, 1)")
    prune_count = max(1, int(metric.shape[1] * ratio))
    mask = torch.zeros_like(metric, dtype=torch.bool)
    indices = torch.topk(metric.float(), prune_count, dim=1, largest=False).indices
    mask.scatter_(1, indices, True)
    weight.data[mask] = 0
    return int(mask.sum().item()), int(mask.numel())


def sparsity_rows(model: Any) -> list[dict[str, Any]]:
    rows = []
    for name, module in find_target_linears(model):
        weight = module.weight.data
        zeros = int((weight == 0).sum().item())
        total = int(weight.numel())
        rows.append(
            {
                "layer": name,
                "zeros": zeros,
                "total": total,
                "sparsity": zeros / total if total else 0.0,
            }
        )
    return rows


def write_sparsity_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["layer", "zeros", "total", "sparsity"])
        writer.writeheader()
        writer.writerows(rows)


def run_prune(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    guide_file = resolve_path(args.guide_file)
    save_dir = resolve_path(args.save_dir)
    guide_rows, guide_hash = load_guide(guide_file, args.max_guide_samples)

    start = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    target_modules = find_target_linears(model)
    if not target_modules:
        raise RuntimeError("No Qwen target Linear modules were found")

    activation_scales = {}
    if args.method == "wanda":
        activation_scales = collect_activation_scales(model, tokenizer, guide_rows, args)

    pruned = 0
    total = 0
    for name, module in target_modules:
        weight = module.weight.data
        if args.method == "wanda":
            scale = activation_scales.get(name)
            if scale is None:
                raise RuntimeError(f"missing activation scale for {name}")
            metric = torch.abs(weight.float()) * scale.to(weight.device).reshape(1, -1)
        else:
            metric = torch.abs(weight.float())
        layer_pruned, layer_total = prune_linear(weight, args.sparsity_ratio, metric)
        pruned += layer_pruned
        total += layer_total

    sparsity = sparsity_rows(model)
    save_dir.mkdir(parents=True, exist_ok=True)
    write_sparsity_csv(save_dir / "sparsity_by_layer.csv", sparsity)
    if not args.skip_save_model:
        model.save_pretrained(save_dir / "pruned_model")
        tokenizer.save_pretrained(save_dir / "pruned_model")

    result = {
        "status": "success",
        "method": args.method,
        "model": args.model,
        "guide_file": str(guide_file.relative_to(ROOT) if guide_file.is_relative_to(ROOT) else guide_file),
        "guide_sha256": guide_hash,
        "guide_samples_used": len(guide_rows),
        "guide_task_ids": [row.get("task_id") for row in guide_rows],
        "sparsity_ratio_requested": args.sparsity_ratio,
        "actual_target_module_sparsity": pruned / total if total else None,
        "target_modules": len(target_modules),
        "dtype": args.dtype,
        "device_map": args.device_map,
        "saved_model": not args.skip_save_model,
        "duration_sec": round(time.monotonic() - start, 3),
    }
    (save_dir / "wanda_qwen_prune_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Wanda/Magnitude Qwen2.5-Coder pruning wrapper.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--guide-file", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--method", default="wanda", choices=["wanda", "magnitude"])
    parser.add_argument("--sparsity-ratio", type=float, default=0.10)
    parser.add_argument("--max-guide-samples", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--skip-save-model", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    guide_file = resolve_path(args.guide_file)
    save_dir = resolve_path(args.save_dir)
    guide_rows, guide_hash = load_guide(guide_file, args.max_guide_samples)

    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "status": "dry_run" if args.dry_run else "planned_heavy_run",
        "method": args.method,
        "model": args.model,
        "guide_file": str(guide_file.relative_to(ROOT) if guide_file.is_relative_to(ROOT) else guide_file),
        "guide_sha256": guide_hash,
        "guide_samples_used": len(guide_rows),
        "guide_task_ids": [row.get("task_id") for row in guide_rows],
        "sparsity_ratio_requested": args.sparsity_ratio,
        "max_seq_len": args.max_seq_len,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "qwen_layout": expected_qwen_layout(config),
        "tokenizer": {
            "class": tokenizer.__class__.__name__,
            "vocab_size": getattr(tokenizer, "vocab_size", None),
            "pad_token": tokenizer.pad_token,
            "eos_token": tokenizer.eos_token,
        },
        "target_modules": sorted(TARGET_LINEAR_NAMES),
    }
    (save_dir / "wanda_qwen_prune_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, ensure_ascii=False, indent=2))

    if args.dry_run:
        return 0

    result = run_prune(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
