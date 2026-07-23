#!/usr/bin/env python3
"""Project wrapper for Flab-Pruner Qwen2/Qwen2.5-Coder pruning."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import types
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


def validate_remain(config, args) -> dict:
    hidden_size = int(config.hidden_size)
    intermediate_size = int(config.intermediate_size)
    num_heads = int(config.num_attention_heads)
    num_kv_heads = int(config.num_key_value_heads)
    head_dim = hidden_size // num_heads

    hidden_remain = args.hidden_size_remain or infer_remain(hidden_size, args.prune_ratio, head_dim)
    ffn_remain = args.ffn_hidden_size_remain or infer_remain(intermediate_size, args.prune_ratio, 256)
    heads_remain = args.num_attention_heads_remain or infer_remain(num_heads, args.prune_ratio, 1)
    kv_heads_remain = args.num_key_value_heads_remain or num_kv_heads

    if hidden_size % num_heads != 0:
        raise ValueError(f"hidden_size {hidden_size} must divide num_attention_heads {num_heads}")
    if hidden_remain % heads_remain != 0:
        raise ValueError(f"hidden_size_remain {hidden_remain} must divide num_attention_heads_remain {heads_remain}")
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
        "head_dim_original": head_dim,
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
        plan["hidden_size"] * plan["num_attention_heads"] * plan["head_dim_original"]
        + 2 * plan["hidden_size"] * plan["num_key_value_heads"] * plan["head_dim_original"]
        + plan["num_attention_heads"] * plan["head_dim_original"] * plan["hidden_size"]
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
        "rough_original_params": original,
        "rough_pruned_params": pruned,
        "rough_param_ratio": pruned / original if original else None,
    }


def topk_index(scores, keep: int):
    import torch

    if keep >= scores.numel():
        return torch.arange(scores.numel(), dtype=torch.long)
    return torch.topk(scores, keep, largest=True).indices.sort().values.to(torch.long)


def mask_from_index(size: int, index, device):
    import torch

    mask = torch.zeros(size, dtype=torch.bool, device=device)
    mask[index.to(device)] = True
    return mask


def move_zs(zs: dict, device: str) -> dict:
    return {
        key: [item.to(device) for item in value] if isinstance(value, list) else value.to(device)
        for key, value in zs.items()
    }


def validate_zs(zs: dict, plan: dict) -> dict:
    checks = {
        "hidden_index": (zs["hidden_index"], plan["hidden_size"]),
    }
    for idx, index in enumerate(zs["head_indexes"]):
        checks[f"head_indexes[{idx}]"] = (index, plan["num_attention_heads"])
    for idx, index in enumerate(zs["kv_head_indexes"]):
        checks[f"kv_head_indexes[{idx}]"] = (index, plan["num_key_value_heads"])
    for idx, index in enumerate(zs["intermediate_indexes"]):
        checks[f"intermediate_indexes[{idx}]"] = (index, plan["intermediate_size"])

    summary = {}
    for name, (index, size) in checks.items():
        index_cpu = index.detach().cpu()
        min_value = int(index_cpu.min().item()) if index_cpu.numel() else None
        max_value = int(index_cpu.max().item()) if index_cpu.numel() else None
        count = int(index_cpu.numel())
        if count and (min_value < 0 or max_value >= size):
            raise ValueError(f"{name} out of bounds: min={min_value} max={max_value} size={size}")
        summary[name] = {"count": count, "min": min_value, "max": max_value, "size": size}
    return summary


def build_benchmark_guided_zs(model, tokenizer, guide_rows: list[dict], plan: dict, max_length: int, save_dir: Path) -> tuple[dict, dict]:
    import torch

    started = time.time()
    device = next(model.parameters()).device
    layer_count = plan["num_hidden_layers"]
    hidden_size = plan["hidden_size"]
    intermediate_size = plan["intermediate_size"]
    head_count = plan["num_attention_heads"]
    kv_head_count = plan["num_key_value_heads"]
    head_dim = plan["head_dim_original"]

    hidden_scores = torch.zeros(hidden_size, dtype=torch.float64)
    intermediate_scores = [torch.zeros(intermediate_size, dtype=torch.float64) for _ in range(layer_count)]
    head_scores = [torch.zeros(head_count, dtype=torch.float64) for _ in range(layer_count)]
    kv_head_scores = [torch.zeros(kv_head_count, dtype=torch.float64) for _ in range(layer_count)]

    hooks = []

    def add_hidden(_, __, output):
        tensor = output[0] if isinstance(output, tuple) else output
        hidden_scores.add_(tensor.detach().float().abs().sum(dim=(0, 1)).cpu().double())

    def make_intermediate_hook(layer_idx: int):
        def hook(_, __, output):
            intermediate_scores[layer_idx].add_(output.detach().float().abs().sum(dim=(0, 1)).cpu().double())

        return hook

    def make_head_hook(layer_idx: int, heads: int, dim: int, store: list):
        def hook(_, __, output):
            values = output.detach().float().abs().sum(dim=(0, 1)).cpu().double()
            store[layer_idx].add_(values.reshape(heads, dim).sum(dim=1))

        return hook

    hooks.append(model.model.embed_tokens.register_forward_hook(add_hidden))
    for layer_idx, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(add_hidden))
        hooks.append(layer.mlp.gate_proj.register_forward_hook(make_intermediate_hook(layer_idx)))
        hooks.append(layer.self_attn.q_proj.register_forward_hook(make_head_hook(layer_idx, head_count, head_dim, head_scores)))
        hooks.append(layer.self_attn.k_proj.register_forward_hook(make_head_hook(layer_idx, kv_head_count, head_dim, kv_head_scores)))

    processed = 0
    token_count = 0
    model.eval()
    with torch.no_grad():
        for row in guide_rows:
            prompt = row.get("prompt") or ""
            if not prompt.strip():
                continue
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            token_count += int(inputs["input_ids"].numel())
            model(**inputs, use_cache=False)
            processed += 1
            if processed % 25 == 0:
                print(json.dumps({"importance_samples_processed": processed, "tokens": token_count}, ensure_ascii=False), flush=True)

    for hook in hooks:
        hook.remove()

    hidden_index = topk_index(hidden_scores, plan["hidden_size_remain"])
    head_indexes = [topk_index(scores, plan["num_attention_heads_remain"]) for scores in head_scores]
    kv_head_indexes = [topk_index(scores, plan["num_key_value_heads_remain"]) for scores in kv_head_scores]
    intermediate_indexes = [topk_index(scores, plan["ffn_hidden_size_remain"]) for scores in intermediate_scores]

    zs = {
        "hidden_mask": mask_from_index(hidden_size, hidden_index, device),
        "hidden_index": hidden_index.to(device),
        "head_masks": [mask_from_index(head_count, index, device) for index in head_indexes],
        "kv_head_masks": [mask_from_index(kv_head_count, index, device) for index in kv_head_indexes],
        "intermediate_masks": [mask_from_index(intermediate_size, index, device) for index in intermediate_indexes],
        "head_indexes": [index.to(device) for index in head_indexes],
        "kv_head_indexes": [index.to(device) for index in kv_head_indexes],
        "intermediate_indexes": [index.to(device) for index in intermediate_indexes],
    }

    summary = {
        "mode": "benchmark_forward_activation_magnitude",
        "samples_processed": processed,
        "tokens_processed": token_count,
        "max_length": max_length,
        "elapsed_seconds": time.time() - started,
        "hidden_keep": int(hidden_index.numel()),
        "head_keep_per_layer": [int(index.numel()) for index in head_indexes],
        "kv_head_keep_per_layer": [int(index.numel()) for index in kv_head_indexes],
        "intermediate_keep_per_layer": [int(index.numel()) for index in intermediate_indexes],
        "hidden_index_preview": hidden_index[:32].tolist(),
        "head_index_preview": [index.tolist() for index in head_indexes[:3]],
        "kv_head_index_preview": [index.tolist() for index in kv_head_indexes[:3]],
        "intermediate_index_preview": [index[:32].tolist() for index in intermediate_indexes[:2]],
    }
    summary["index_validation"] = validate_zs(zs, plan)
    (save_dir / "benchmark_importance_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return zs, summary


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
    parser = argparse.ArgumentParser(description="Flab-Pruner Qwen2.5-Coder wrapper.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--guide-file", required=True, action="append", help="Guide JSONL file. May be repeated.")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--stage", default="top", choices=["top", "bottom", "random", "middle"])
    parser.add_argument("--prune-ratio", type=float, default=0.10)
    parser.add_argument("--max-guide-samples", type=int, default=4, help="Maximum guide rows to read from each --guide-file.")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--importance-mode", default="structural", choices=["structural", "benchmark"])
    parser.add_argument("--importance-max-length", type=int, default=256)
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
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    compat_patches = ensure_qwen2_compat_config(config)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    plan = validate_remain(config, args)
    estimate = estimate_params(config, plan)

    result = {
        "status": "dry_run" if args.dry_run else "planned_heavy_run",
        "method": "Flab-Pruner",
        "model": args.model,
        "guide_files": guide_manifests,
        "guide_samples_used": len(guide_rows),
        "guide_task_ids": [row.get("task_id") for row in guide_rows],
        "stage": args.stage,
        "prune_ratio_requested": args.prune_ratio,
        "dtype": args.dtype,
        "device_map": args.device_map,
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
        "importance_mode": args.importance_mode,
        "importance_max_length": args.importance_max_length,
        "prune_on_cpu": args.prune_on_cpu,
        "benchmark_guidance_status": (
            "guide files recorded and validated only; structural stage selection"
            if args.importance_mode == "structural"
            else "benchmark guide forward activations will determine hidden/head/intermediate keep indexes"
        ),
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
    load_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    ensure_qwen2_compat_config(load_config)
    prune_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    ensure_qwen2_compat_config(prune_config)
    prune_config.update(
        {
            "hidden_size_remain": plan["hidden_size_remain"],
            "num_attention_heads_remain": plan["num_attention_heads_remain"],
            "num_key_value_heads_remain": plan["num_key_value_heads_remain"],
            "ffn_hidden_size_remain": plan["ffn_hidden_size_remain"],
        }
    )
    model = Qwen2ForCausalLM.from_pretrained(args.model, config=load_config, torch_dtype=dtype, device_map=args.device_map)
    model.eval()
    params_before = sum(p.numel() for p in model.parameters())
    if args.importance_mode == "benchmark":
        zs, importance_summary = build_benchmark_guided_zs(model, tokenizer, guide_rows, plan, args.importance_max_length, save_dir)
        if args.prune_on_cpu:
            model.cpu()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            zs = move_zs(zs, "cpu")
            importance_summary["structural_prune_device"] = "cpu"
        importance_summary["post_move_index_validation"] = validate_zs(zs, plan)
        (save_dir / "benchmark_importance_summary.json").write_text(
            json.dumps(importance_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        def init_prune_zs_from_benchmark(self, config, stage):
            return zs

        model.init_prune_zs = types.MethodType(init_prune_zs_from_benchmark, model)
        result["benchmark_importance_summary"] = importance_summary
    model.prune(config=prune_config, stage=args.stage)
    params_after = sum(p.numel() for p in model.parameters())

    new_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
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
