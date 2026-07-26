#!/usr/bin/env python3
"""Create a guide-half distillation dataset from a teacher model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from completion_extraction import normalize_completion


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--guide-file", required=True, action="append", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-per-file", type=int, default=0)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load teacher model/tokenizer files only from the local Hugging Face cache.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    use_cuda = torch.cuda.is_available() and args.device.startswith("cuda")
    device_map = args.device if use_cuda else "cpu"

    rows = []
    manifests = []
    for guide_file in args.guide_file:
        items = read_jsonl(guide_file)
        if args.limit_per_file:
            items = items[: args.limit_per_file]
        rows.extend(items)
        manifests.append(
            {
                "path": str(guide_file),
                "sha256": sha256_file(guide_file),
                "samples_used": len(items),
                "benchmarks": sorted({item.get("benchmark") for item in items}),
            }
        )

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        trust_remote_code=True,
        torch_dtype=dtype_map[args.dtype],
        device_map=device_map,
        local_files_only=args.local_files_only,
    )
    model.eval()

    output_path = args.out_dir / "distill_train.jsonl"
    generation_path = args.out_dir / "teacher_generations.jsonl"
    with output_path.open("w", encoding="utf-8") as out, generation_path.open("w", encoding="utf-8") as gen_out:
        for idx, row in enumerate(rows, 1):
            prompt = row["prompt"]
            inputs = tokenizer(prompt, return_tensors="pt")
            if use_cuda:
                inputs = {key: value.to(args.device) for key, value in inputs.items()}
            gen_started = time.time()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen_seconds = time.time() - gen_started
            input_token_count = inputs["input_ids"].shape[-1]
            generated_ids = output_ids[0, input_token_count:]
            raw_completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated = prompt + raw_completion
            completion = normalize_completion(raw_completion)
            item = {
                "task_id": row.get("task_id"),
                "benchmark": row.get("benchmark"),
                "prompt": prompt,
                "completion": completion,
                "text": prompt + completion,
            }
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            gen_out.write(
                json.dumps(
                    {
                        "task_id": row.get("task_id"),
                        "benchmark": row.get("benchmark"),
                        "generated": generated,
                        "raw_completion": raw_completion,
                        "completion": completion,
                        "input_tokens": input_token_count,
                        "generated_tokens": int(generated_ids.numel()),
                        "max_new_tokens": args.max_new_tokens,
                        "hit_max_new_tokens": int(generated_ids.numel()) >= args.max_new_tokens,
                        "gen_seconds": gen_seconds,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out.flush()
            gen_out.flush()
            print(
                json.dumps(
                    {
                        "generated": idx,
                        "task_id": row.get("task_id"),
                        "raw_completion_chars": len(raw_completion),
                        "completion_chars": len(completion),
                        "generated_tokens": int(generated_ids.numel()),
                        "hit_max_new_tokens": int(generated_ids.numel()) >= args.max_new_tokens,
                        "seconds": round(gen_seconds, 3),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    summary = {
        "status": "success",
        "teacher_model": args.teacher_model,
        "guide_files": manifests,
        "samples": len(rows),
        "output": str(output_path),
        "teacher_generations": str(generation_path),
        "local_files_only": args.local_files_only,
        "elapsed_seconds": time.time() - started,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
