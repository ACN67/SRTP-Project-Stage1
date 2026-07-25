#!/usr/bin/env python3
"""Create a small layer-drop CodeLlama pruned model for LLM-Pruner R4 smoke tests."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dtype_from_name(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def first_model_device(model) -> torch.device:
    device_map = getattr(model, "hf_device_map", {}) or {}
    for value in device_map.values():
        if isinstance(value, int):
            return torch.device(f"cuda:{value}")
        if isinstance(value, str) and value.startswith("cuda"):
            return torch.device(value)
    return next(model.parameters()).device


def score_layers_by_hidden_delta(model, tokenizer, rows: list[dict], device: torch.device, max_length: int) -> list[dict]:
    model.eval()
    layer_count = len(model.model.layers)
    sums = [0.0 for _ in range(layer_count)]
    counts = [0 for _ in range(layer_count)]

    def make_hook(idx: int):
        def hook(_module, inputs, output):
            before = inputs[0].detach()
            after = output[0].detach() if isinstance(output, tuple) else output.detach()
            value = (after.float() - before.float()).pow(2).mean().sqrt().item()
            sums[idx] += value
            counts[idx] += 1
        return hook

    hooks = [layer.register_forward_hook(make_hook(idx)) for idx, layer in enumerate(model.model.layers)]
    try:
        with torch.no_grad():
            total_rows = len(rows)
            for idx, row in enumerate(rows, 1):
                started = time.time()
                inputs = tokenizer(
                    row["prompt"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                model(**inputs)
                print(
                    json.dumps(
                        {
                            "event": "importance_sample_done",
                            "index": idx,
                            "total": total_rows,
                            "task_id": row.get("task_id"),
                            "seconds": round(time.time() - started, 3),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        for hook in hooks:
            hook.remove()

    scores = [
        {"layer": idx, "hidden_delta": sums[idx] / counts[idx] if counts[idx] else 0.0}
        for idx in range(layer_count)
    ]
    return sorted(scores, key=lambda item: item["hidden_delta"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=["default", "benchmark"], required=True)
    parser.add_argument("--drop-layers", type=int, default=1)
    parser.add_argument("--guide-file", action="append", type=Path, default=[])
    parser.add_argument("--guide-limit-per-file", type=int, default=1)
    parser.add_argument("--importance-max-length", type=int, default=128)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-load-in-4bit", action="store_true")
    parser.add_argument("--score-device-map", default="auto")
    parser.add_argument("--score-max-memory-json", default="")
    parser.add_argument("--score-offload-folder", type=Path)
    parser.add_argument("--save-device", default="")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    proc = psutil.Process()
    use_cuda = torch.cuda.is_available() and args.device.startswith("cuda")
    device = torch.device(args.device if use_cuda else "cpu")
    save_device_name = args.save_device or args.device
    save_use_cuda = torch.cuda.is_available() and save_device_name.startswith("cuda")
    save_device = torch.device(save_device_name if save_use_cuda else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype_from_name(args.dtype),
        "local_files_only": True,
    }
    if args.score_load_in_4bit:
        load_kwargs["device_map"] = args.score_device_map
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype_from_name(args.dtype),
        )
        if args.score_max_memory_json:
            max_memory = json.loads(args.score_max_memory_json)
            load_kwargs["max_memory"] = {
                int(key) if isinstance(key, str) and key.isdigit() else key: value
                for key, value in max_memory.items()
            }
        if args.score_offload_folder:
            args.score_offload_folder.mkdir(parents=True, exist_ok=True)
            load_kwargs["offload_folder"] = str(args.score_offload_folder)

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
    if not args.score_load_in_4bit:
        model.to(device)
    model.eval()

    layers_before = len(model.model.layers)
    if not (0 <= args.drop_layers < layers_before):
        raise ValueError("--drop-layers must be smaller than the layer count")

    guide_rows = []
    importance_scores = None
    if args.mode == "default":
        drop_indices = list(range(layers_before - args.drop_layers, layers_before))
    else:
        for guide_file in args.guide_file:
            rows = read_jsonl(guide_file)
            guide_rows.extend(rows if args.guide_limit_per_file <= 0 else rows[: args.guide_limit_per_file])
        if not guide_rows:
            raise ValueError("--mode benchmark requires at least one --guide-file row")
        score_device = first_model_device(model) if args.score_load_in_4bit else device
        importance_scores = score_layers_by_hidden_delta(
            model,
            tokenizer,
            guide_rows,
            score_device,
            args.importance_max_length,
        )
        drop_indices = sorted(item["layer"] for item in importance_scores[: args.drop_layers])

    if args.score_load_in_4bit:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=dtype_from_name(args.dtype),
            local_files_only=True,
        )
        model.to(save_device)
        model.eval()

    keep_layers = [layer for idx, layer in enumerate(model.model.layers) if idx not in set(drop_indices)]
    model.model.layers = torch.nn.ModuleList(keep_layers)
    model.config.num_hidden_layers = len(keep_layers)

    params_after = sum(param.numel() for param in model.parameters())
    model.save_pretrained(args.out_dir / "pruned_model", safe_serialization=True)
    tokenizer.save_pretrained(args.out_dir / "pruned_model")

    summary = {
        "status": "success",
        "method": "LLM-Pruner",
        "model_path": args.model_path,
        "mode": args.mode,
        "drop_layers": args.drop_layers,
        "drop_indices": drop_indices,
        "layers_before": layers_before,
        "layers_after": len(keep_layers),
        "parameter_count_after": params_after,
        "guide_files": [str(path) for path in args.guide_file],
        "guide_samples_used": len(guide_rows),
        "importance_scores": importance_scores,
        "score_load_in_4bit": args.score_load_in_4bit,
        "score_device_map": getattr(model, "hf_device_map", None),
        "save_device": str(save_device),
        "pruned_model": str(args.out_dir / "pruned_model"),
        "process_rss_mb_after": proc.memory_info().rss / 1024**2,
        "elapsed_seconds": time.time() - started,
        "torch": torch.__version__,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
