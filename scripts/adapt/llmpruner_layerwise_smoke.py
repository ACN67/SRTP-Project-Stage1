#!/usr/bin/env python3
"""Layer-wise LLM-Pruner smoke wrapper without mandatory upstream PPL."""

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

from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_completion(prompt: str, generated: str) -> str:
    completion = generated[len(prompt):] if generated.startswith(prompt) else generated
    for fence in ("```python", "```"):
        if fence in completion:
            after = completion.split(fence, 1)[1]
            completion = after.split("```", 1)[0] if "```" in after else after
            break
    return completion.strip() + "\n"


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--keep-layers", type=int, default=31)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--generate-split", type=Path)
    parser.add_argument("--generate-limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--device-map", default="none", help="Optional HF/accelerate device_map, e.g. auto.")
    parser.add_argument("--max-memory-json", help='Optional JSON max_memory, e.g. {"cuda:0":"6GiB","cpu":"20GiB"}.')
    parser.add_argument("--offload-folder", type=Path)
    args = parser.parse_args()

    started = time.time()
    proc = psutil.Process()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = LlamaTokenizer.from_pretrained(args.model_path, local_files_only=True)
    load_kwargs = {
        "low_cpu_mem_usage": True,
        "torch_dtype": dtype_from_name(args.dtype),
        "local_files_only": True,
    }
    use_device_map = args.device_map and args.device_map != "none"
    if use_device_map:
        load_kwargs["device_map"] = args.device_map
        if args.max_memory_json:
            max_memory = json.loads(args.max_memory_json)
            load_kwargs["max_memory"] = {
                int(key) if isinstance(key, str) and key.isdigit() else key: value
                for key, value in max_memory.items()
            }
        if args.offload_folder:
            args.offload_folder.mkdir(parents=True, exist_ok=True)
            load_kwargs["offload_folder"] = str(args.offload_folder)

    model = LlamaForCausalLM.from_pretrained(args.model_path, **load_kwargs)
    model.eval()

    layers_before = len(model.model.layers)
    params_before = count_params(model)
    if not (0 < args.keep_layers <= layers_before):
        raise ValueError(f"--keep-layers must be in 1..{layers_before}")

    model.model.layers = model.model.layers[: args.keep_layers]
    layers_after = len(model.model.layers)
    params_after = count_params(model)

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available() and not use_device_map
    if use_cuda:
        model = model.to(args.device)

    def input_device() -> torch.device:
        if use_device_map:
            hf_device_map = getattr(model, "hf_device_map", {}) or {}
            for value in hf_device_map.values():
                if isinstance(value, int):
                    return torch.device(f"cuda:{value}")
                if isinstance(value, str) and value.startswith("cuda"):
                    return torch.device(value)
            return next(model.parameters()).device
        if use_cuda:
            return torch.device(args.device)
        return torch.device("cpu")

    generation_summary = None
    if args.generate_split:
        tasks = read_jsonl(args.generate_split)
        if args.generate_limit:
            tasks = tasks[: args.generate_limit]

        samples_path = args.out_dir / "samples.jsonl"
        generations_path = args.out_dir / "generations.jsonl"
        with samples_path.open("w", encoding="utf-8") as sf, generations_path.open("w", encoding="utf-8") as gf:
            for item in tasks:
                task_id = item["task_id"]
                prompt = item["prompt"]
                inputs = tokenizer(prompt, return_tensors="pt")
                target_device = input_device()
                inputs = {key: value.to(target_device) for key, value in inputs.items()}

                gen_started = time.time()
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                gen_seconds = time.time() - gen_started
                generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                completion = extract_completion(prompt, generated)
                solution = prompt + completion

                sf.write(json.dumps({"task_id": task_id, "solution": solution}, ensure_ascii=False) + "\n")
                gf.write(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "prompt": prompt,
                            "generated": generated,
                            "completion": completion,
                            "gen_seconds": gen_seconds,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                sf.flush()
                gf.flush()
                print(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "completion_chars": len(completion),
                            "gen_seconds": round(gen_seconds, 3),
                        },
                        ensure_ascii=False,
                    )
                )

        generation_summary = {
            "split": str(args.generate_split),
            "task_count": len(tasks),
            "samples": str(samples_path),
            "generations": str(generations_path),
            "max_new_tokens": args.max_new_tokens,
        }

    rss_mb = proc.memory_info().rss / 1024 / 1024
    summary = {
        "status": "success",
        "method": "LLM-Pruner",
        "model_path": str(args.model_path),
        "wrapper": "scripts/adapt/llmpruner_layerwise_smoke.py",
        "loaded_weights": True,
        "dtype": args.dtype,
        "device": args.device if use_cuda else "cpu",
        "device_map": getattr(model, "hf_device_map", None),
        "requested_device_map": args.device_map,
        "max_memory": json.loads(args.max_memory_json) if args.max_memory_json else None,
        "offload_folder": str(args.offload_folder) if args.offload_folder else None,
        "layers_before": layers_before,
        "layers_after": layers_after,
        "parameters_before": params_before,
        "parameters_after": params_after,
        "parameter_keep_ratio": params_after / params_before,
        "parameter_reduction_rate": 1 - params_after / params_before,
        "generation": generation_summary,
        "process_rss_mb_after": rss_mb,
        "elapsed_seconds": time.time() - started,
        "torch": torch.__version__,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    del model
    del tokenizer
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
