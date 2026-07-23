#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    use_cuda = torch.cuda.is_available() and args.device.startswith("cuda")
    device_map = args.device if use_cuda else "cpu"

    tasks = list(read_jsonl(args.split))
    if args.limit:
        tasks = tasks[: args.limit]

    started = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype_map[args.dtype],
        device_map=device_map,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    samples_path = args.out_dir / "samples.jsonl"
    generations_path = args.out_dir / "generations.jsonl"

    with samples_path.open("w", encoding="utf-8") as sf, generations_path.open("w", encoding="utf-8") as gf:
        for item in tasks:
            task_id = item["task_id"]
            prompt = item["prompt"]

            inputs = tokenizer(prompt, return_tensors="pt")
            if use_cuda:
                inputs = {k: v.to(args.device) for k, v in inputs.items()}

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
                "task_id": task_id,
                "completion_chars": len(completion),
                "gen_seconds": round(gen_seconds, 3),
            }, ensure_ascii=False))

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
        "device": device_map,
    }
    (args.out_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
