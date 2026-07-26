#!/usr/bin/env python3
"""Run LLM-Pruner's official MetaPruner flow on Qwen2/Qwen2.5 models.

The upstream entry point is LLaMA-specific. This wrapper keeps the official
LLM-Pruner ingredients intact: MetaPruner, Magnitude/Taylor importance, and
block/channel/layer pruning modes. The adaptation layer is the model family
binding: Qwen2 model classes and Qwen2 RMSNorm are registered where upstream
registers LLaMA classes.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2RMSNorm

ROOT = Path(__file__).resolve().parents[2]
LLM_PRUNER_ROOT = ROOT / "third_party" / "llm_pruner"
sys.path.insert(0, str(LLM_PRUNER_ROOT))

import LLMPruner.torch_pruning as tp
from LLMPruner.pruner import hf_llama_pruner as official_pruner


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dtype_from_name(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_guide_rows(paths: list[Path], limit_per_file: int) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    manifests = []
    for path in paths:
        items = read_jsonl(path)
        if limit_per_file > 0:
            items = items[:limit_per_file]
        for item in items:
            if item.get("contains_solution"):
                raise ValueError(f"guide row contains_solution=true: {item.get('task_id')}")
        rows.extend(items)
        manifests.append(
            {
                "path": str(path),
                "samples_used": len(items),
                "benchmarks": sorted({item.get("benchmark") for item in items}),
                "task_ids": [item.get("task_id") for item in items],
            }
        )
    return rows, manifests


def tokenize_rows(tokenizer, rows: list[dict], max_length: int, device: torch.device) -> list[torch.Tensor]:
    batches = []
    for row in rows:
        prompt = row.get("prompt") or ""
        if not prompt.strip():
            continue
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        batches.append(encoded["input_ids"].to(device))
    if not batches:
        raise ValueError("no non-empty benchmark guide prompts after tokenization")
    return batches


def make_importance(kind: str, grouping_strategy: str, taylor: str):
    kind = kind.lower()
    if kind == "random":
        return tp.importance.RandomImportance()
    if kind == "l1":
        return official_pruner.MagnitudeImportance(p=1)
    if kind == "l2":
        return official_pruner.MagnitudeImportance(p=2)
    if kind == "taylor":
        return official_pruner.TaylorImportance(group_reduction=grouping_strategy, taylor=taylor)
    raise ValueError(f"unknown pruner type: {kind}")


def run_taylor_backward(model, batches: list[torch.Tensor], taylor: str, second_order_examples: int) -> list[dict]:
    events = []
    if taylor in ["param_mix", "param_second"]:
        selected = batches[:second_order_examples] if second_order_examples > 0 else batches
        for idx, input_ids in enumerate(selected, 1):
            started = time.time()
            loss = model(input_ids, labels=input_ids).loss
            loss.backward()
            for param in model.parameters():
                if param.grad is None:
                    continue
                param.grad = param.grad * param.grad / max(1, len(selected))
                if hasattr(param, "acc_grad"):
                    param.acc_grad += param.grad
                else:
                    param.acc_grad = copy.deepcopy(param.grad)
            model.zero_grad()
            event = {"event": "taylor_second_order_sample", "index": idx, "loss": float(loss.detach().cpu()), "seconds": time.time() - started}
            print(json.dumps(event, ensure_ascii=False), flush=True)
            events.append(event)

    joined = torch.cat(batches, dim=0) if len({tuple(batch.shape) for batch in batches}) == 1 else None
    if joined is not None:
        loss_inputs = [joined]
    else:
        loss_inputs = batches
    for idx, input_ids in enumerate(loss_inputs, 1):
        started = time.time()
        loss = model(input_ids, labels=input_ids).loss
        loss.backward()
        event = {"event": "taylor_backward", "index": idx, "loss": float(loss.detach().cpu()), "seconds": time.time() - started}
        print(json.dumps(event, ensure_ascii=False), flush=True)
        events.append(event)
    return events


def refresh_qwen_config_after_pruning(model) -> dict:
    first_layer = model.model.layers[0] if len(model.model.layers) else None
    if first_layer is None:
        return {}

    hidden_size = int(model.model.embed_tokens.weight.shape[1])
    q_out = int(first_layer.self_attn.q_proj.out_features)
    k_out = int(first_layer.self_attn.k_proj.out_features)
    head_dim = int(first_layer.self_attn.head_dim)
    num_heads = max(1, q_out // head_dim)
    num_key_value_heads = max(1, k_out // head_dim)
    intermediate_size = int(first_layer.mlp.gate_proj.out_features)

    model.config.hidden_size = hidden_size
    model.config.num_hidden_layers = len(model.model.layers)
    model.config.num_attention_heads = num_heads
    model.config.num_key_value_heads = num_key_value_heads
    model.config.intermediate_size = intermediate_size

    for layer in model.model.layers:
        layer.hidden_size = hidden_size
        layer.self_attn.hidden_size = hidden_size
        layer.self_attn.num_heads = num_heads
        layer.self_attn.num_key_value_heads = num_key_value_heads
        layer.self_attn.num_key_value_groups = max(1, num_heads // num_key_value_heads)
        layer.mlp.hidden_size = hidden_size
        layer.mlp.intermediate_size = int(layer.mlp.gate_proj.out_features)

    return {
        "hidden_size": hidden_size,
        "num_hidden_layers": len(model.model.layers),
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_key_value_heads,
        "intermediate_size": intermediate_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=["block_wise", "channel_wise", "layer_wise"], default="block_wise")
    parser.add_argument("--pruning-ratio", type=float, default=0.20)
    parser.add_argument("--pruner-type", choices=["random", "l1", "l2", "taylor"], default="taylor")
    parser.add_argument("--taylor", choices=["vectorize", "param_second", "param_first", "param_mix"], default="param_first")
    parser.add_argument("--grouping-strategy", default="sum")
    parser.add_argument("--global-pruning", action="store_true")
    parser.add_argument("--iterative-steps", type=int, default=1)
    parser.add_argument("--guide-file", action="append", type=Path, default=[])
    parser.add_argument("--guide-limit-per-file", type=int, default=0)
    parser.add_argument("--importance-max-length", type=int, default=256)
    parser.add_argument("--second-order-examples", type=int, default=10)
    parser.add_argument("--block-attention-layer-start", type=int, default=4)
    parser.add_argument("--block-attention-layer-end", type=int, default=-1)
    parser.add_argument("--block-mlp-layer-start", type=int, default=4)
    parser.add_argument("--block-mlp-layer-end", type=int, default=-1)
    parser.add_argument("--layer-keep", type=int, default=0)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (0.0 <= args.pruning_ratio < 1.0):
        raise ValueError("--pruning-ratio must be in [0, 1)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)
    started = time.time()
    proc = psutil.Process()
    use_cuda = torch.cuda.is_available() and args.device.startswith("cuda")
    device = torch.device(args.device if use_cuda else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    guide_rows, guide_manifests = load_guide_rows(args.guide_file, args.guide_limit_per_file)
    if args.pruner_type == "taylor" and not guide_rows:
        raise ValueError("Taylor benchmark-guided pruning requires at least one --guide-file")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype_from_name(args.dtype),
    )
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(True)
    before_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    layer_count = len(model.model.layers)

    block_attention_end = layer_count if args.block_attention_layer_end < 0 else args.block_attention_layer_end
    block_mlp_end = layer_count if args.block_mlp_layer_end < 0 else args.block_mlp_layer_end
    forward_prompts = tokenizer("def add(a, b):\n    return a + b", return_tensors="pt")["input_ids"].to(device)

    summary = {
        "status": "dry_run" if args.dry_run else "planned",
        "method": "LLM-Pruner official MetaPruner Qwen adaptation",
        "model": args.model,
        "mode": args.mode,
        "pruning_ratio": args.pruning_ratio,
        "pruner_type": args.pruner_type,
        "taylor": args.taylor,
        "grouping_strategy": args.grouping_strategy,
        "guide_files": guide_manifests,
        "guide_samples_used": len(guide_rows),
        "importance_max_length": args.importance_max_length,
        "layers": layer_count,
        "params_before": before_params,
        "dtype": args.dtype,
        "device": str(device),
    }

    if args.dry_run:
        (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    imp = make_importance(args.pruner_type, args.grouping_strategy, args.taylor)
    if args.mode == "block_wise":
        kwargs = {
            "importance": imp,
            "global_pruning": args.global_pruning,
            "iterative_steps": args.iterative_steps,
            "ch_sparsity": args.pruning_ratio,
            "ignored_layers": [],
            "channel_groups": {},
            "consecutive_groups": {
                layer.self_attn.q_proj: layer.self_attn.head_dim for layer in model.model.layers
            },
            "customized_pruners": {
                Qwen2RMSNorm: official_pruner.hf_rmsnorm_pruner,
            },
            "root_module_types": None,
            "root_instances": [model.model.layers[i].self_attn.q_proj for i in range(args.block_attention_layer_start, block_attention_end)]
            + [model.model.layers[i].mlp.gate_proj for i in range(args.block_mlp_layer_start, block_mlp_end)],
        }
    elif args.mode == "channel_wise":
        kwargs = {
            "importance": imp,
            "global_pruning": args.global_pruning,
            "iterative_steps": args.iterative_steps,
            "ch_sparsity": args.pruning_ratio,
            "ignored_layers": [],
            "channel_groups": {},
            "customized_pruners": {
                Qwen2RMSNorm: official_pruner.hf_rmsnorm_pruner,
            },
            "root_module_types": [Qwen2RMSNorm, Qwen2Attention],
        }
    else:
        keep = args.layer_keep or max(1, int(round(layer_count * (1.0 - args.pruning_ratio))))
        model.model.layers = torch.nn.ModuleList(list(model.model.layers[:keep]))
        model.config.num_hidden_layers = keep
        kwargs = None

    events = []
    if args.mode != "layer_wise":
        pruner = tp.pruner.MetaPruner(model, forward_prompts, **kwargs)
        model.zero_grad()
        guide_batches = tokenize_rows(tokenizer, guide_rows, args.importance_max_length, device) if args.pruner_type == "taylor" else []
        for step in range(args.iterative_steps):
            if args.pruner_type == "taylor":
                events.extend(run_taylor_backward(model, guide_batches, args.taylor, args.second_order_examples))
            pruner.step()
            after_step = sum(p.numel() for p in model.parameters() if p.requires_grad)
            event = {"event": "prune_step_done", "step": step + 1, "params": after_step}
            print(json.dumps(event, ensure_ascii=False), flush=True)
            events.append(event)
        del pruner

    model.zero_grad()
    for param in model.parameters():
        param.grad = None
    after_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary.update(
        {
            "status": "success",
            "params_after": after_params,
            "actual_param_ratio": after_params / before_params,
            "events": events,
            "process_rss_mb_after": proc.memory_info().rss / 1024**2,
            "elapsed_seconds": time.time() - started,
            "torch": torch.__version__,
        }
    )

    if args.save_model:
        summary["refreshed_config"] = refresh_qwen_config_after_pruning(model)
        model.save_pretrained(args.out_dir / "pruned_model", safe_serialization=True)
        tokenizer.save_pretrained(args.out_dir / "pruned_model")
        summary["pruned_model"] = str(args.out_dir / "pruned_model")

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
