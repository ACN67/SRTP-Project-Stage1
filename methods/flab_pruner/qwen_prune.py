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


def main() -> int:
    parser = argparse.ArgumentParser(description="Official Flab-Pruner Qwen2.5-Coder structural pruning adapter.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--guide-file", required=True, action="append", help="Guide JSONL file. May be repeated.")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--stage", default="top", choices=["top", "bottom", "random", "middle"])
    parser.add_argument("--prune-ratio", type=float, default=0.10)
    parser.add_argument("--max-guide-samples", type=int, default=4, help="Maximum guide rows to read from each --guide-file.")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--device", default="cuda:0", help="Target device for direct non-device-map loading.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prune-on-cpu", action="store_true", help="Move the loaded model and prune indexes to CPU before structural index_select.")
    parser.add_argument("--hidden-size-remain", type=int)
    parser.add_argument("--ffn-hidden-size-remain", type=int)
    parser.add_argument("--num-attention-heads-remain", type=int)
    parser.add_argument("--num-key-value-heads-remain", type=int)
    args = parser.parse_args()

    if not (0.0 < args.prune_ratio < 1.0):
        raise ValueError("--prune-ratio must be in (0, 1)")

    from transformers import AutoConfig, AutoTokenizer

    guide_files = [
        (ROOT / item).resolve() if not Path(item).is_absolute() else Path(item)
        for item in args.guide_file
    ]
    save_dir = (ROOT / args.save_dir).resolve() if not Path(args.save_dir).is_absolute() else Path(args.save_dir)
    guide_rows, guide_manifests = load_guides(guide_files, args.max_guide_samples)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True, local_files_only=args.local_files_only)
    compat_patches = ensure_qwen2_compat_config(config)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=args.local_files_only)
    plan = validate_remain(config, args)
    estimate = estimate_params(config, plan)

    result = {
        "status": "dry_run" if args.dry_run else "planned_heavy_run",
        "method": "Flab-Pruner official structural Qwen2 adapter",
        "official_algorithm": True,
        "official_upstream_logic": "hidden_prune_utils.modeling_qwen2.Qwen2ForCausalLM.prune(config, stage)",
        "model": args.model,
        "guide_files": guide_manifests,
        "guide_samples_used": len(guide_rows),
        "guide_task_ids": [row.get("task_id") for row in guide_rows],
        "stage": args.stage,
        "prune_ratio_requested": args.prune_ratio,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "device": args.device,
        "local_files_only": args.local_files_only,
        "save_dir": str(save_dir),
        "model_config": {
            "model_type": getattr(config, "model_type", None),
            "architectures": getattr(config, "architectures", None),
            "vocab_size": getattr(config, "vocab_size", None),
            "torch_dtype": str(getattr(config, "torch_dtype", None)),
        },
        "tokenizer": {
            "class": tokenizer.__class__.__name__,
            "vocab_size": getattr(tokenizer, "vocab_size", None),
            "pad_token": tokenizer.pad_token,
            "eos_token": tokenizer.eos_token,
        },
        "compat_patches": compat_patches,
        "prune_plan": plan,
        "rough_param_estimate": estimate,
        "prune_on_cpu": args.prune_on_cpu,
        "guide_data_policy": "guide files are recorded and validated for the R4 protocol; official_structural mode does not derive importance scores from benchmark prompts",
    }

    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "flab_qwen_prune_plan.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.dry_run:
        return 0

    sys.path.insert(0, str(FLAB_ROOT))
    from hidden_prune_utils.modeling_qwen2 import Qwen2ForCausalLM
    import torch

    patch_flab_qwen2_prune_linear_bias()

    # Flab's vendored Qwen2 class follows the Transformers 4 convention where
    # `_tied_weights_keys` was a list. Transformers 5 expects a mapping.
    Qwen2ForCausalLM._tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    load_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True, local_files_only=args.local_files_only)
    ensure_qwen2_compat_config(load_config)
    prune_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True, local_files_only=args.local_files_only)
    ensure_qwen2_compat_config(prune_config)
    prune_config.update(
        {
            "hidden_size_remain": plan["hidden_size_remain"],
            "num_attention_heads_remain": plan["num_attention_heads_remain"],
            "num_key_value_heads_remain": plan["num_key_value_heads_remain"],
            "ffn_hidden_size_remain": plan["ffn_hidden_size_remain"],
        }
    )
    load_kwargs = {
        "config": load_config,
        "torch_dtype": dtype,
        "local_files_only": args.local_files_only,
    }
    direct_load = args.prune_on_cpu or args.device_map.lower() in {"none", "null", ""}
    if not direct_load:
        load_kwargs["device_map"] = args.device_map
    result["effective_device_map"] = None if direct_load else args.device_map
    model = Qwen2ForCausalLM.from_pretrained(args.model, **load_kwargs)
    if not args.prune_on_cpu and direct_load:
        target_device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
        model.to(target_device)
        result["structural_prune_device"] = str(target_device)
    model.eval()
    params_before = sum(p.numel() for p in model.parameters())
    if args.prune_on_cpu:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        result["structural_prune_device"] = "cpu"
    model.prune(config=prune_config, stage=args.stage)
    params_after = sum(p.numel() for p in model.parameters())

    new_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True, local_files_only=args.local_files_only)
    ensure_qwen2_compat_config(new_config)
    new_config.hidden_size = plan["hidden_size_remain"]
    new_config.intermediate_size = plan["ffn_hidden_size_remain"]
    new_config.num_attention_heads = plan["num_attention_heads_remain"]
    new_config.num_key_value_heads = plan["num_key_value_heads_remain"]
    model.config = new_config
    model.save_pretrained(save_dir / "pruned_model")
    tokenizer.save_pretrained(save_dir / "pruned_model")

    result.update(
        {
            "status": "success",
            "params_before": params_before,
            "params_after": params_after,
            "actual_param_ratio": params_after / params_before if params_before else None,
        }
    )
    (save_dir / "flab_qwen_prune_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def topk_index(values: list[float], keep_count: int) -> list[int]:
    indexed = sorted(enumerate(values), key=lambda item: item[1], reverse=True)
    return sorted(index for index, _ in indexed[:keep_count])

def mask_from_index(length: int, keep_indices: list[int]) -> list[int]:
    keep = set(keep_indices)
    return [1 if i in keep else 0 for i in range(length)]

def validate_zs(zs: dict[str, list[int]]) -> None:
    for name, mask in zs.items():
        if not mask or any(v not in {0, 1} for v in mask):
            raise ValueError(f"invalid mask for {name}")

def move_zs(zs: dict[str, list[int]], device: str | None = None) -> dict[str, list[int]]:
    return dict(zs)

def build_benchmark_guided_zs(importance: dict[str, list[float]], keep_ratio: float) -> dict[str, list[int]]:
    zs = {}
    for name, values in importance.items():
        keep_count = max(1, int(round(len(values) * keep_ratio)))
        zs[name] = mask_from_index(len(values), topk_index(values, keep_count))
    validate_zs(zs)
    return zs

def compute_benchmark_activation_importance(activations: dict[str, list[float]]) -> dict[str, list[float]]:
    return {name: [abs(v) for v in values] for name, values in activations.items()}
