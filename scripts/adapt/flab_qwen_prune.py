#!/usr/bin/env python3
"""Project wrapper for Flab-Pruner Qwen2/Qwen2.5-Coder pruning."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Flab-Pruner Qwen2.5-Coder wrapper.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--guide-file", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--stage", default="top", choices=["top", "bottom", "random", "middle"])
    parser.add_argument("--prune-ratio", type=float, default=0.10)
    parser.add_argument("--max-guide-samples", type=int, default=4)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hidden-size-remain", type=int)
    parser.add_argument("--ffn-hidden-size-remain", type=int)
    parser.add_argument("--num-attention-heads-remain", type=int)
    parser.add_argument("--num-key-value-heads-remain", type=int)
    args = parser.parse_args()

    if not (0.0 < args.prune_ratio < 1.0):
        raise ValueError("--prune-ratio must be in (0, 1)")

    from transformers import AutoConfig, AutoTokenizer

    guide_file = (ROOT / args.guide_file).resolve() if not Path(args.guide_file).is_absolute() else Path(args.guide_file)
    save_dir = (ROOT / args.save_dir).resolve() if not Path(args.save_dir).is_absolute() else Path(args.save_dir)
    guide_rows, guide_hash = load_guide(guide_file, args.max_guide_samples)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    plan = validate_remain(config, args)
    estimate = estimate_params(config, plan)

    result = {
        "status": "dry_run" if args.dry_run else "planned_heavy_run",
        "method": "Flab-Pruner",
        "model": args.model,
        "guide_file": str(guide_file.relative_to(ROOT) if guide_file.is_relative_to(ROOT) else guide_file),
        "guide_sha256": guide_hash,
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
        "prune_plan": plan,
        "rough_param_estimate": estimate,
        "benchmark_guidance_status": "guide file recorded and validated; upstream Flab stage selection is still structural/top-bottom-random unless scoring patch is added",
    }

    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "flab_qwen_prune_plan.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.dry_run:
        return 0

    sys.path.insert(0, str(FLAB_ROOT))
    from hidden_prune_utils.modeling_qwen2 import Qwen2ForCausalLM
    import torch

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    prune_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    prune_config.update(
        {
            "hidden_size_remain": plan["hidden_size_remain"],
            "num_attention_heads_remain": plan["num_attention_heads_remain"],
            "num_key_value_heads_remain": plan["num_key_value_heads_remain"],
            "ffn_hidden_size_remain": plan["ffn_hidden_size_remain"],
        }
    )
    model = Qwen2ForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map=args.device_map)
    model.eval()
    params_before = sum(p.numel() for p in model.parameters())
    model.prune(config=prune_config, stage=args.stage)
    params_after = sum(p.numel() for p in model.parameters())

    new_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
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
