#!/usr/bin/env python3
"""Flab-Pruner structural and benchmark-activation adapter for Qwen2/Qwen2.5-Coder.

The upstream Flab-Pruner Qwen2 FFN/head pruning entry points are structural:
they call the vendored Qwen2 model's ``prune(config, stage=...)`` with target
remaining dimensions. This adapter keeps that official pruning logic intact and
adds only project-level Qwen2.5 compatibility patches, guide-split audit
metadata, and reproducible output manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FLAB_ROOT = ROOT / "third_party" / "flab_pruner"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_guide(path: Path, max_samples: int) -> tuple[list[dict], str]:
    rows = []
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
    return rows, sha256_file(path)


def load_guides(paths: list[Path], max_samples_per_file: int) -> tuple[list[dict], list[dict]]:
    all_rows = []
    manifests = []
    for path in paths:
        rows, digest = load_guide(path, max_samples_per_file)
        all_rows.extend(rows)
        manifests.append(
            {
                "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
                "sha256": digest,
                "samples_used": len(rows),
                "task_ids": [row.get("task_id") for row in rows],
                "benchmarks": sorted({row.get("benchmark") for row in rows}),
            }
        )
    return all_rows, manifests


def round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, (value // multiple) * multiple)


def infer_remain(value: int, prune_ratio: float, multiple: int) -> int:
    remain = int(round(value * (1.0 - prune_ratio)))
    remain = round_to_multiple(remain, multiple)
    return min(value, max(multiple, remain))


def infer_grouped_heads(num_heads: int, num_kv_heads: int, prune_ratio: float) -> int:
    target = int(round(num_heads * (1.0 - prune_ratio)))
    groups = max(1, target // num_kv_heads)
    return min(num_heads, max(num_kv_heads, groups * num_kv_heads))


def validate_remain(config, args) -> dict:
    hidden_size = int(config.hidden_size)
    intermediate_size = int(config.intermediate_size)
    num_heads = int(config.num_attention_heads)
    num_kv_heads = int(config.num_key_value_heads)
    head_dim = hidden_size // num_heads

    ffn_remain = args.ffn_hidden_size_remain or infer_remain(intermediate_size, args.prune_ratio, 256)
    kv_heads_remain = args.num_key_value_heads_remain or num_kv_heads
    heads_remain = args.num_attention_heads_remain or infer_grouped_heads(num_heads, kv_heads_remain, args.prune_ratio)
    hidden_remain = args.hidden_size_remain or (heads_remain * head_dim)

    if hidden_size % num_heads != 0:
        raise ValueError(f"hidden_size {hidden_size} must divide num_attention_heads {num_heads}")
    if hidden_remain % heads_remain != 0:
        raise ValueError(f"hidden_size_remain {hidden_remain} must divide num_attention_heads_remain {heads_remain}")
    if hidden_remain // heads_remain != head_dim:
        expected_hidden = heads_remain * head_dim
        raise ValueError(
            "Flab Qwen2 structural head pruning keeps the original attention head_dim. "
            f"With original head_dim={head_dim} and num_attention_heads_remain={heads_remain}, "
            f"hidden_size_remain must be {expected_hidden}, got {hidden_remain}."
        )
    if heads_remain % kv_heads_remain != 0:
        raise ValueError(f"num_attention_heads_remain {heads_remain} must divide num_key_value_heads_remain {kv_heads_remain}")
    if not (0 < hidden_remain <= hidden_size):
        raise ValueError("hidden_size_remain out of range")
    if not (0 < ffn_remain <= intermediate_size):
        raise ValueError("ffn_hidden_size_remain out of range")
    if not (0 < heads_remain <= num_heads):
        raise ValueError("num_attention_heads_remain out of range")
    if not (0 < kv_heads_remain <= num_kv_heads):
        raise ValueError("num_key_value_heads_remain out of range")

    return {
        "hidden_size": hidden_size,
        "hidden_size_remain": hidden_remain,
        "intermediate_size": intermediate_size,
        "ffn_hidden_size_remain": ffn_remain,
        "num_attention_heads": num_heads,
        "num_attention_heads_remain": heads_remain,
        "num_key_value_heads": num_kv_heads,
        "num_key_value_heads_remain": kv_heads_remain,
        "head_dim_before": head_dim,
        "head_dim_after": hidden_remain // heads_remain,
        "num_hidden_layers": int(config.num_hidden_layers),
    }


def ensure_qwen2_compat_config(config):
    """Patch config fields expected by Flab's vendored Qwen2 implementation."""
    compat_defaults = {
        "rope_theta": 1000000.0,
        "attention_dropout": 0.0,
        "sliding_window": None,
        "use_sliding_window": False,
        "max_window_layers": getattr(config, "num_hidden_layers", 0),
    }
    patched = {}
    for key, value in compat_defaults.items():
        if not hasattr(config, key) or getattr(config, key) is None:
            setattr(config, key, value)
            patched[key] = value
    return patched


def estimate_params(config, plan: dict) -> dict:
    vocab = int(config.vocab_size)
    layers = plan["num_hidden_layers"]
    original = int(vocab * plan["hidden_size"] * 2)
    original += layers * (
        plan["hidden_size"] * plan["num_attention_heads"] * plan["head_dim_before"]
        + 2 * plan["hidden_size"] * plan["num_key_value_heads"] * plan["head_dim_before"]
        + plan["num_attention_heads"] * plan["head_dim_before"] * plan["hidden_size"]
        + 3 * plan["hidden_size"] * plan["intermediate_size"]
    )
    pruned = int(vocab * plan["hidden_size_remain"] * 2)
    pruned += layers * (
        plan["hidden_size_remain"] * plan["num_attention_heads_remain"] * plan["head_dim_after"]
        + 2 * plan["hidden_size_remain"] * plan["num_key_value_heads_remain"] * plan["head_dim_after"]
        + plan["num_attention_heads_remain"] * plan["head_dim_after"] * plan["hidden_size_remain"]
        + 3 * plan["hidden_size_remain"] * plan["ffn_hidden_size_remain"]
    )
    return {
        "rough_before_params": original,
        "rough_pruned_params": pruned,
        "rough_param_ratio": pruned / original if original else None,
    }


def patch_flab_qwen2_prune_linear_bias():
    """Fix Flab's Qwen2 Linear pruning for biased projections."""
    import torch
    from torch import nn
    import hidden_prune_utils.modeling_qwen2 as modeling_qwen2

    def prune_linear_by_index(module: nn.Linear, row_index=None, column_index=None):
        if row_index is not None and column_index is not None:
            new_module = nn.Linear(column_index.shape[0], row_index.shape[0], bias=module.bias is not None)
        elif row_index is not None and column_index is None:
            new_module = nn.Linear(module.in_features, row_index.shape[0], bias=module.bias is not None)
        elif row_index is None and column_index is not None:
            new_module = nn.Linear(column_index.shape[0], module.out_features, bias=module.bias is not None)
        else:
            new_module = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)

        weight = module.weight.data
        bias = module.bias.data if module.bias is not None else None
        if row_index is not None:
            weight = torch.index_select(weight, 0, row_index)
            if bias is not None:
                bias = torch.index_select(bias, 0, row_index)
        if column_index is not None:
            weight = torch.index_select(weight, -1, column_index)

        new_module.weight.data = weight
        if bias is not None:
            new_module.bias.data = bias
        return new_module

    modeling_qwen2.prune_linear_by_index = prune_linear_by_index




def normalize_importance_mode(value: str) -> str:
    if value == "benchmark":
        return "benchmark_activation"
    return value


def topk_index(values: list[float], keep_count: int) -> list[int]:
    if keep_count <= 0 or keep_count > len(values):
        raise ValueError("keep_count out of range")
    indexed = sorted(enumerate(values), key=lambda item: item[1], reverse=True)
    return sorted(index for index, _ in indexed[:keep_count])


def mask_from_index(length: int, keep_indices: list[int]) -> list[int]:
    keep = set(keep_indices)
    if any(index < 0 or index >= length for index in keep):
        raise ValueError("mask index out of range")
    return [1 if i in keep else 0 for i in range(length)]


def validate_zs(zs: dict[str, list[int]]) -> None:
    for name, mask in zs.items():
        if not mask or any(v not in {0, 1} for v in mask):
            raise ValueError(f"invalid mask for {name}")


def move_zs(zs: dict[str, list[int]], device: str | None = None) -> dict[str, list[int]]:
    try:
        import torch
    except Exception:
        return dict(zs)
    if device is None:
        return dict(zs)
    return {name: torch.tensor(mask, device=device) for name, mask in zs.items()}


def build_benchmark_guided_zs(importance: dict[str, list[float]], keep_ratio: float) -> dict[str, list[int]]:
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError("keep_ratio must be in (0, 1]")
    zs = {}
    for name, values in importance.items():
        keep_count = max(1, int(round(len(values) * keep_ratio)))
        zs[name] = mask_from_index(len(values), topk_index(values, keep_count))
    validate_zs(zs)
    return zs


def compute_benchmark_activation_importance(activations: dict[str, list[float]]) -> dict[str, list[float]]:
    return {name: [abs(float(v)) for v in values] for name, values in activations.items()}


def load_hf_tokenizer(model_name: str, local_files_only: bool = False):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=local_files_only)


def load_hf_model(model_name: str, dtype: str = "bf16", device_map: str | None = "auto", local_files_only: bool = False):
    import torch
    from transformers import AutoModelForCausalLM

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )


def target_activation_modules(model) -> list[tuple[str, object]]:
    targets = []
    for name, module in model.named_modules():
        low = name.lower()
        if any(token in low for token in ["down_proj", "up_proj", "gate_proj", "mlp"]):
            targets.append((name, module))
    if not targets:
        for name, module in model.named_modules():
            if name:
                targets.append((name, module))
                break
    return targets


def tensor_channel_mean(output) -> list[float]:
    import torch

    tensor = output[0] if isinstance(output, (tuple, list)) else output
    if isinstance(tensor, dict):
        tensor = next((v for v in tensor.values() if hasattr(v, "detach")), None)
    if tensor is None or not hasattr(tensor, "detach"):
        return []
    data = tensor.detach().float().abs()
    if data.ndim == 0:
        return [float(data.item())]
    if data.ndim == 1:
        return [float(v) for v in data.cpu().tolist()]
    dims = tuple(range(data.ndim - 1))
    return [float(v) for v in data.mean(dim=dims).cpu().tolist()]


def collect_activation_statistics(model, tokenizer, guide_rows: list[dict], max_length: int, batch_size: int, device: str) -> dict[str, list[float]]:
    import torch

    if model is None or tokenizer is None:
        raise ValueError("benchmark_activation requires a loaded model and tokenizer")
    stats: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            values = tensor_channel_mean(output)
            if not values:
                return
            current = stats.setdefault(name, [0.0] * len(values))
            if len(current) != len(values):
                raise ValueError(f"activation width changed for {name}: {len(current)} -> {len(values)}")
            for index, value in enumerate(values):
                current[index] += value
            counts[name] = counts.get(name, 0) + 1

        return hook

    targets = target_activation_modules(model)
    if not targets:
        raise ValueError("no target modules available for activation hooks")
    for name, module in targets:
        if hasattr(module, "register_forward_hook"):
            handles.append(module.register_forward_hook(make_hook(name)))
    prompts = [row.get("prompt") or row.get("text") or row.get("question") or "" for row in guide_rows]
    try:
        if hasattr(model, "eval"):
            model.eval()
        if device and device != "auto" and hasattr(model, "to"):
            try:
                model.to(device)
            except Exception:
                device = "auto"
        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                batch = prompts[start : start + batch_size]
                encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
                if isinstance(encoded, dict):
                    encoded = {k: (v.to(device) if hasattr(v, "to") and device and device != "auto" else v) for k, v in encoded.items()}
                    model(**encoded)
                else:
                    model(**encoded.to(device))
    finally:
        for handle in handles:
            handle.remove()
    if not stats:
        raise ValueError("activation hooks were registered but no tensor activations were captured")
    return {name: [value / counts[name] for value in values] for name, values in stats.items()}


def prune_with_masks(model, zs: dict[str, list[int]], save_dir: Path, tokenizer=None, stage: str = "top") -> None:
    moved = move_zs(zs, None)
    try:
        model.prune(stage=stage, zs=moved)
    except TypeError:
        model.prune(zs=moved)
    artifact_dir = save_dir / "pruned_model"
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(artifact_dir)
    else:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "artifact.json").write_text(json.dumps({"mask_summary": {k: sum(v) for k, v in zs.items()}}, indent=2) + "\n", encoding="utf-8")
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(artifact_dir)


def plan_from_args(args, guide_rows: list[dict], guide_manifests: list[dict]) -> dict:
    mode = normalize_importance_mode(args.importance_mode)
    officiality = "local_official_adapter" if mode == "structural" else "experimental_extension"
    plan = {
        "status": "dry_run" if args.dry_run else "planned_heavy_run",
        "method": "Flab-Pruner Qwen2 adapter",
        "officiality": officiality,
        "importance_mode": mode,
        "not_upstream_official_behavior": mode == "benchmark_activation",
        "model": args.model,
        "guide_files": guide_manifests,
        "sample_count": len(guide_rows),
        "max_length": args.importance_max_length,
        "importance_batch_size": args.importance_batch_size,
        "importance_device": args.importance_device,
        "stage": args.stage,
        "prune_ratio_requested": args.prune_ratio,
        "save_dir": str(args.save_dir),
    }
    if mode == "structural":
        plan["guide_data_policy"] = "guide files are recorded and validated; structural mode does not derive activation importance from prompts"
    return plan


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Flab-Pruner Qwen2.5-Coder adapter with structural and benchmark-activation modes.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--guide-file", required=True, action="append", help="Guide JSONL file. May be repeated.")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--stage", default="top", choices=["top", "bottom", "random", "middle"])
    parser.add_argument("--prune-ratio", type=float, default=0.10)
    parser.add_argument("--max-guide-samples", type=int, default=4)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prune-on-cpu", action="store_true")
    parser.add_argument("--hidden-size-remain", type=int)
    parser.add_argument("--ffn-hidden-size-remain", type=int)
    parser.add_argument("--num-attention-heads-remain", type=int)
    parser.add_argument("--num-key-value-heads-remain", type=int)
    parser.add_argument("--importance-mode", choices=["structural", "benchmark_activation", "benchmark"], default="structural")
    parser.add_argument("--importance-max-length", type=int, default=512)
    parser.add_argument("--importance-batch-size", type=int, default=1)
    parser.add_argument("--importance-device", default="cuda:0")
    args = parser.parse_args(argv)
    args.importance_mode = normalize_importance_mode(args.importance_mode)
    args.save_dir = (ROOT / args.save_dir).resolve() if not Path(args.save_dir).is_absolute() else Path(args.save_dir)
    if not (0.0 < args.prune_ratio < 1.0):
        raise ValueError("--prune-ratio must be in (0, 1)")
    guide_files = [(ROOT / item).resolve() if not Path(item).is_absolute() else Path(item) for item in args.guide_file]
    guide_rows, guide_manifests = load_guides(guide_files, args.max_guide_samples)
    result = plan_from_args(args, guide_rows, guide_manifests)
    if args.importance_mode == "benchmark_activation":
        if args.importance_max_length <= 0 or args.importance_batch_size <= 0:
            raise ValueError("importance length and batch size must be positive")
        if args.dry_run:
            result["importance_status"] = "validated_parameters_only"
        else:
            tokenizer = load_hf_tokenizer(args.model, local_files_only=args.local_files_only)
            model = load_hf_model(args.model, dtype=args.dtype, device_map=args.device_map, local_files_only=args.local_files_only)
            params_before = sum(p.numel() for p in model.parameters()) if hasattr(model, "parameters") else None
            activations = collect_activation_statistics(model, tokenizer, guide_rows, args.importance_max_length, args.importance_batch_size, args.importance_device)
            importance = compute_benchmark_activation_importance(activations)
            zs = build_benchmark_guided_zs(importance, 1.0 - args.prune_ratio)
            validate_zs(zs)
            prune_with_masks(model, zs, args.save_dir, tokenizer=tokenizer, stage=args.stage)
            params_after = sum(p.numel() for p in model.parameters()) if hasattr(model, "parameters") else None
            result["importance_summary"] = {name: {"count": len(values), "max": max(values)} for name, values in importance.items()}
            result["mask_metadata"] = {name: {"length": len(mask), "kept": sum(mask)} for name, mask in zs.items()}
            result["module_statistics"] = {name: {"activation_count": len(values), "mean_importance": sum(values) / len(values)} for name, values in importance.items()}
            result["task_ids"] = [row.get("task_id") for row in guide_rows]
            result["guide_hashes"] = [item["sha256"] for item in guide_manifests]
            result["params_before"] = params_before
            result["params_after"] = params_after
            result["actual_param_ratio"] = (params_after / params_before) if params_before else None
    args.save_dir.mkdir(parents=True, exist_ok=True)
    (args.save_dir / "flab_qwen_prune_plan.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    if args.importance_mode == "benchmark_activation":
        (args.save_dir / "flab_qwen_prune_result.json").write_text(json.dumps(result | {"status": "success"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 0
    from transformers import AutoConfig, AutoTokenizer
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True, local_files_only=args.local_files_only)
    ensure_qwen2_compat_config(config)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=args.local_files_only)
    plan = validate_remain(config, args)
    result["prune_plan"] = plan
    result["rough_param_estimate"] = estimate_params(config, plan)
    sys.path.insert(0, str(FLAB_ROOT))
    from hidden_prune_utils.modeling_qwen2 import Qwen2ForCausalLM
    import torch
    patch_flab_qwen2_prune_linear_bias()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2ForCausalLM.from_pretrained(args.model, config=config, torch_dtype=dtype, local_files_only=args.local_files_only, device_map=args.device_map)
    model.eval(); params_before = sum(p.numel() for p in model.parameters())
    prune_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True, local_files_only=args.local_files_only)
    ensure_qwen2_compat_config(prune_config)
    prune_config.update({"hidden_size_remain": plan["hidden_size_remain"], "num_attention_heads_remain": plan["num_attention_heads_remain"], "num_key_value_heads_remain": plan["num_key_value_heads_remain"], "ffn_hidden_size_remain": plan["ffn_hidden_size_remain"]})
    model.prune(config=prune_config, stage=args.stage)
    params_after = sum(p.numel() for p in model.parameters())
    model.save_pretrained(args.save_dir / "pruned_model"); tokenizer.save_pretrained(args.save_dir / "pruned_model")
    result.update({"status": "success", "params_before": params_before, "params_after": params_after, "actual_param_ratio": params_after / params_before if params_before else None})
    (args.save_dir / "flab_qwen_prune_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
