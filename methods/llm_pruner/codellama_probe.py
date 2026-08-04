#!/usr/bin/env python3
"""Probe official LLM-Pruner MetaPruner on CodeLlama with benchmark guide data."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import psutil
import torch
from transformers import LlamaTokenizer

ROOT = Path(__file__).resolve().parents[2]
LLM_PRUNER_ROOT = ROOT / "third_party" / "llm_pruner"
sys.path.insert(0, str(LLM_PRUNER_ROOT))

import LLMPruner.torch_pruning as tp
from LLMPruner.models.hf_llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaRMSNorm,
)
from LLMPruner.pruner import hf_llama_pruner as llama_pruner


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dtype_from_name(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def first_model_device(model: torch.nn.Module) -> torch.device:
    device_map = getattr(model, "hf_device_map", None) or {}
    for value in device_map.values():
        if isinstance(value, int):
            return torch.device(f"cuda:{value}")
        if isinstance(value, str) and value.startswith("cuda"):
            return torch.device(value)
    return next(model.parameters()).device


def count_params(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def normalize_max_memory(raw: str) -> dict | None:
    if not raw:
        return None
    parsed = json.loads(raw)
    return {
        int(key) if isinstance(key, str) and key.isdigit() else key: value
        for key, value in parsed.items()
    }


def build_benchmark_batch(tokenizer, guide_files: list[Path], limit_per_file: int, max_length: int, device: torch.device):
    texts: list[str] = []
    for guide_file in guide_files:
        rows = read_jsonl(guide_file)
        selected = rows if limit_per_file <= 0 else rows[:limit_per_file]
        texts.extend(row["prompt"] for row in selected)
    if not texts:
        raise ValueError("benchmark mode requires at least one guide sample")
    batch = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {key: value.to(device) for key, value in batch.items()}, len(texts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--guide-file", action="append", type=Path, default=[])
    parser.add_argument("--guide-limit-per-file", type=int, default=1)
    parser.add_argument("--importance-max-length", type=int, default=128)
    parser.add_argument("--pruning-ratio", type=float, default=0.2)
    parser.add_argument("--pruner-type", choices=["l1", "l2", "taylor"], default="taylor")
    parser.add_argument("--taylor", choices=["param_first", "param_second", "param_mix", "vectorize"], default="param_first")
    parser.add_argument("--grouping-strategy", default="sum")
    parser.add_argument("--mode", choices=["channel_wise", "block_wise"], default="block_wise")
    parser.add_argument("--block-attention-layer-start", type=int, default=3)
    parser.add_argument("--block-attention-layer-end", type=int, default=31)
    parser.add_argument("--block-mlp-layer-start", type=int, default=3)
    parser.add_argument("--block-mlp-layer-end", type=int, default=31)
    parser.add_argument("--iterative-steps", type=int, default=1)
    parser.add_argument("--global-pruning", action="store_true")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-memory-json", default='{"0":"5GiB","cpu":"28GiB"}')
    parser.add_argument("--offload-folder", type=Path)
    parser.add_argument("--offload-buffers", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()

    started = time.time()
    proc = psutil.Process()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.offload_folder:
        args.offload_folder.mkdir(parents=True, exist_ok=True)

    tokenizer = LlamaTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "low_cpu_mem_usage": True,
        "torch_dtype": dtype_from_name(args.dtype),
        "local_files_only": True,
        "device_map": args.device_map,
    }
    max_memory = normalize_max_memory(args.max_memory_json)
    if max_memory:
        load_kwargs["max_memory"] = max_memory
    if args.offload_folder:
        load_kwargs["offload_folder"] = str(args.offload_folder)
    # This local transformers/LLM-Pruner model stack does not accept
    # offload_buffers in from_pretrained; keep the CLI flag only for logging.

    print(json.dumps({"event": "loading_model", "load_kwargs": {k: str(v) for k, v in load_kwargs.items()}}, ensure_ascii=False), flush=True)
    model = LlamaForCausalLM.from_pretrained(args.model_path, **load_kwargs)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(True)

    input_device = first_model_device(model)
    params_before = count_params(model)
    layers_before = len(model.model.layers)

    forward_prompts = torch.tensor(
        [
            [1, 306, 4658, 278, 6593, 310, 2834, 338],
            [1, 3439, 17632, 1925, 29892, 278, 6368, 310],
        ],
        device=input_device,
    )

    if args.pruner_type == "l1":
        importance = llama_pruner.MagnitudeImportance(p=1)
    elif args.pruner_type == "l2":
        importance = llama_pruner.MagnitudeImportance(p=2)
    else:
        importance = llama_pruner.TaylorImportance(group_reduction=args.grouping_strategy, taylor=args.taylor)

    if args.mode == "block_wise":
        kwargs = {
            "importance": importance,
            "global_pruning": args.global_pruning,
            "iterative_steps": args.iterative_steps,
            "ch_sparsity": args.pruning_ratio,
            "ignored_layers": [],
            "channel_groups": {},
            "consecutive_groups": {
                layer.self_attn.q_proj: layer.self_attn.head_dim for layer in model.model.layers
            },
            "customized_pruners": {
                LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
            },
            "root_module_types": None,
            "root_instances": [
                model.model.layers[i].self_attn.q_proj
                for i in range(args.block_attention_layer_start, min(args.block_attention_layer_end, layers_before))
            ]
            + [
                model.model.layers[i].mlp.gate_proj
                for i in range(args.block_mlp_layer_start, min(args.block_mlp_layer_end, layers_before))
            ],
        }
    else:
        kwargs = {
            "importance": importance,
            "global_pruning": args.global_pruning,
            "iterative_steps": args.iterative_steps,
            "ch_sparsity": args.pruning_ratio,
            "ignored_layers": [],
            "channel_groups": {},
            "customized_pruners": {
                LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
            },
            "root_module_types": [LlamaRMSNorm, LlamaAttention],
        }

    print(json.dumps({"event": "building_metapruner", "mode": args.mode, "input_device": str(input_device)}, ensure_ascii=False), flush=True)
    pruner = tp.pruner.MetaPruner(model, forward_prompts, **kwargs)
    model.zero_grad(set_to_none=True)

    guide_samples_used = 0
    if args.pruner_type == "taylor":
        print(json.dumps({"event": "benchmark_backward_start"}, ensure_ascii=False), flush=True)
        batch, guide_samples_used = build_benchmark_batch(
            tokenizer,
            args.guide_file,
            args.guide_limit_per_file,
            args.importance_max_length,
            input_device,
        )
        loss = model(**batch, labels=batch["input_ids"]).loss
        print(json.dumps({"event": "benchmark_loss", "loss": float(loss.detach().cpu())}, ensure_ascii=False), flush=True)
        loss.backward()

    print(json.dumps({"event": "official_pruner_step_start"}, ensure_ascii=False), flush=True)
    for step in range(args.iterative_steps):
        pruner.step()
        print(json.dumps({"event": "official_pruner_step_done", "step": step + 1}, ensure_ascii=False), flush=True)

    for layer in model.model.layers:
        if hasattr(layer.self_attn, "num_heads"):
            layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim

    params_after = count_params(model)
    summary = {
        "status": "success",
        "method": "LLM-Pruner official MetaPruner probe",
        "model_path": args.model_path,
        "mode": args.mode,
        "pruner_type": args.pruner_type,
        "taylor": args.taylor if args.pruner_type == "taylor" else None,
        "benchmark_guided": args.pruner_type == "taylor",
        "guide_files": [str(path) for path in args.guide_file],
        "guide_samples_used": guide_samples_used,
        "pruning_ratio": args.pruning_ratio,
        "layers_before": layers_before,
        "layers_after": len(model.model.layers),
        "params_before": params_before,
        "params_after": params_after,
        "actual_param_ratio": params_after / params_before,
        "device_map": getattr(model, "hf_device_map", None),
        "max_memory": max_memory,
        "offload_buffers": args.offload_buffers,
        "process_rss_mb": proc.memory_info().rss / 1024**2,
        "elapsed_seconds": time.time() - started,
        "torch": torch.__version__,
    }

    if args.save_model:
        save_dir = args.out_dir / "pruned_model"
        model.save_pretrained(save_dir, safe_serialization=True)
        tokenizer.save_pretrained(save_dir)
        summary["pruned_model"] = str(save_dir)

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    del pruner
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
