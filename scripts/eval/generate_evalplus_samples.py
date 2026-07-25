#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_completion(prompt: str, generated: str) -> str:
    if generated.startswith(prompt):
        completion = generated[len(prompt):]
    else:
        completion = generated

    fences = ["```python", "```"]
    for fence in fences:
        if fence in completion:
            after = completion.split(fence, 1)[1]
            completion = after.split("```", 1)[0] if "```" in after else after
            break

    return completion.strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--load-mode", choices=["direct", "device_map"], default="direct")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-memory-json", default="")
    parser.add_argument("--offload-folder", type=Path)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--llm-int8-enable-fp32-cpu-offload", action="store_true")
    args = parser.parse_args()
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("Choose only one of --load-in-8bit or --load-in-4bit.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    use_cuda = torch.cuda.is_available() and args.device.startswith("cuda")
    device = torch.device(args.device if use_cuda else "cpu")

    tasks = list(read_jsonl(args.split))
    if args.limit:
        tasks = tasks[: args.limit]

    started = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype_map[args.dtype],
    }
    if args.load_in_8bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=args.llm_int8_enable_fp32_cpu_offload,
        )
    elif args.load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype_map[args.dtype],
        )
    if args.load_mode == "device_map":
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
    else:
        load_kwargs["device_map"] = args.device if use_cuda else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    def input_device() -> torch.device:
        if args.load_mode == "device_map":
            hf_device_map = getattr(model, "hf_device_map", {}) or {}
            for value in hf_device_map.values():
                if isinstance(value, int):
                    return torch.device(f"cuda:{value}")
                if isinstance(value, str) and value.startswith("cuda"):
                    return torch.device(value)
            return next(model.parameters()).device
        return device

    samples_path = args.out_dir / "samples.jsonl"
    generations_path = args.out_dir / "generations.jsonl"

    with samples_path.open("w", encoding="utf-8") as sf, generations_path.open("w", encoding="utf-8") as gf:
        total_tasks = len(tasks)
        for idx, item in enumerate(tasks, 1):
            task_id = item["task_id"]
            prompt = item["prompt"]

            inputs = tokenizer(prompt, return_tensors="pt")
            if use_cuda:
                target_device = input_device()
                inputs = {k: v.to(target_device) for k, v in inputs.items()}

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
            gf.write(json.dumps({
                "task_id": task_id,
                "prompt": prompt,
                "generated": generated,
                "completion": completion,
                "gen_seconds": gen_seconds,
            }, ensure_ascii=False) + "\n")
            sf.flush()
            gf.flush()

            print(json.dumps({
                "event": "generated",
                "index": idx,
                "total": total_tasks,
                "task_id": task_id,
                "completion_chars": len(completion),
                "gen_seconds": round(gen_seconds, 3),
            }, ensure_ascii=False), flush=True)

    summary = {
        "status": "success",
        "model": args.model,
        "adapter": args.adapter,
        "split": str(args.split),
        "samples": str(samples_path),
        "generations": str(generations_path),
        "task_count": len(tasks),
        "elapsed_seconds": time.time() - started,
        "torch": getattr(torch, "__version__"),
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "load_mode": args.load_mode,
        "load_in_8bit": args.load_in_8bit,
        "load_in_4bit": args.load_in_4bit,
        "llm_int8_enable_fp32_cpu_offload": args.llm_int8_enable_fp32_cpu_offload,
        "device_map": getattr(model, "hf_device_map", None),
        "max_memory": json.loads(args.max_memory_json) if args.max_memory_json else None,
        "offload_folder": str(args.offload_folder) if args.offload_folder else None,
    }
    (args.out_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
