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
from generate_official_samples import build_lcb_prompt_and_extractor


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benchmark_name(row: dict) -> str:
    name = str(row.get("benchmark") or "").lower()
    if name.startswith("humaneval"):
        return "humaneval"
    if name.startswith("mbpp"):
        return "mbpp_evalplus"
    if name.startswith("livecodebench"):
        return "livecodebench"
    raise ValueError(f"Unsupported guide benchmark for official distillation prompt: {row.get('benchmark')!r}")


def build_prompt_records(
    rows: list[dict],
    lcb_release: str,
    lcb_config: str,
    lcb_lm_style: str,
) -> tuple[list[dict], dict]:
    lcb_rows = [row for row in rows if benchmark_name(row) == "livecodebench"]
    lcb_prompts = None
    lcb_extractor = None
    if lcb_rows:
        lcb_prompts, lcb_extractor = build_lcb_prompt_and_extractor(
            lcb_rows,
            lcb_release,
            lcb_config,
            lcb_lm_style,
        )

    records = []
    for row in rows:
        bench = benchmark_name(row)
        if bench == "livecodebench":
            assert lcb_prompts is not None and lcb_extractor is not None
            model_prompt = lcb_prompts[row["task_id"]]
            extractor_name = "livecodebench.extract_code"
            extractor = lcb_extractor
            prompt_mode = "livecodebench_official"
        elif bench == "humaneval":
            model_prompt = row["prompt"]
            extractor_name = "normalize_completion"
            extractor = normalize_completion
            prompt_mode = "humaneval_official"
        else:
            model_prompt = row["prompt"]
            extractor_name = "normalize_completion"
            extractor = normalize_completion
            prompt_mode = "mbpp_evalplus_official"
        records.append(
            {
                "row": row,
                "benchmark_family": bench,
                "model_prompt": model_prompt,
                "completion_extractor": extractor,
                "completion_extractor_name": extractor_name,
                "prompt_mode": prompt_mode,
            }
        )

    policy = {
        "official_prompt": True,
        "humaneval": "EvalPlus/HumanEval raw prompt",
        "mbpp_evalplus": "EvalPlus MBPP raw prompt",
        "livecodebench": "LiveCodeBench format_prompt_generation + extract_code",
        "lcb_release": lcb_release if lcb_rows else None,
        "lcb_config": lcb_config if lcb_rows else None,
        "lcb_lm_style": lcb_lm_style if lcb_rows else None,
    }
    return records, policy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--guide-file", required=True, action="append", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-per-file", type=int, default=0)
    parser.add_argument("--lcb-release", default="release_v1")
    parser.add_argument("--lcb-config", default="release_latest")
    parser.add_argument("--lcb-lm-style", default="CodeQwenInstruct")
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
        for item in items:
            if item.get("contains_solution"):
                raise ValueError(f"guide row contains_solution=true: {item.get('task_id')}")
        rows.extend(items)
        manifests.append(
            {
                "path": str(guide_file),
                "sha256": sha256_file(guide_file),
                "samples_used": len(items),
                "benchmarks": sorted({item.get("benchmark") for item in items}),
                "task_ids": [item.get("task_id") for item in items],
            }
        )

    started = time.time()
    prompt_records, prompt_policy = build_prompt_records(
        rows,
        args.lcb_release,
        args.lcb_config,
        args.lcb_lm_style,
    )
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
        for idx, record in enumerate(prompt_records, 1):
            row = record["row"]
            source_prompt = row["prompt"]
            model_prompt = record["model_prompt"]
            inputs = tokenizer(model_prompt, return_tensors="pt")
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
            generated = model_prompt + raw_completion
            completion = record["completion_extractor"](raw_completion)
            item = {
                "task_id": row.get("task_id"),
                "benchmark": row.get("benchmark"),
                "benchmark_family": record["benchmark_family"],
                "prompt": model_prompt,
                "source_prompt": source_prompt,
                "prompt_mode": record["prompt_mode"],
                "official_prompt": True,
                "completion": completion,
                "text": model_prompt + completion,
            }
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            gen_out.write(
                json.dumps(
                    {
                        "task_id": row.get("task_id"),
                        "benchmark": row.get("benchmark"),
                        "benchmark_family": record["benchmark_family"],
                        "prompt": source_prompt,
                        "model_prompt": model_prompt,
                        "prompt_mode": record["prompt_mode"],
                        "official_prompt": True,
                        "completion_extractor": record["completion_extractor_name"],
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
        "prompt_policy": prompt_policy,
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
