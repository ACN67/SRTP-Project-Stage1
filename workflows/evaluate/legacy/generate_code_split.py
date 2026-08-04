#!/usr/bin/env python3
"""Generate predictions for a Stage 1 code benchmark split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_prompt(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            "### Benchmark",
            str(row.get("benchmark", "")),
            "",
            "### Task",
            str(row.get("prompt", "")),
            "",
            "### Instruction",
            "Complete the requested code task. Return only the code needed for the solution.",
        ]
    )


def dry_run_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "task_id": row.get("task_id"),
            "benchmark": row.get("benchmark"),
            "prompt": row.get("prompt"),
            "completion": "",
            "status": "dry_run_no_generation",
        }
        for row in rows
    ]


def generate_predictions(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()

    predictions: list[dict[str, Any]] = []
    for row in rows:
        prompt = build_prompt(row)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_tokens)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        completion = tokenizer.decode(generated, skip_special_tokens=True)
        predictions.append(
            {
                "task_id": row.get("task_id"),
                "benchmark": row.get("benchmark"),
                "prompt": row.get("prompt"),
                "completion": completion,
                "status": "generated",
            }
        )
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Stage 1 benchmark split predictions.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--benchmark", required=True, choices=["humaneval", "mbpp", "livecodebench", "swebench_lite"])
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    split_path = Path(args.split)
    if not split_path.is_absolute():
        split_path = ROOT / split_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(split_path)
    start = time.monotonic()
    predictions = dry_run_predictions(rows) if args.dry_run else generate_predictions(args, rows)
    duration = round(time.monotonic() - start, 3)

    write_jsonl(output_dir / "predictions.jsonl", predictions)
    metrics = {
        "benchmark": args.benchmark,
        "model": args.model,
        "split": str(split_path.relative_to(ROOT) if split_path.is_relative_to(ROOT) else split_path),
        "task_count": len(rows),
        "generated_count": sum(1 for row in predictions if row["status"] == "generated"),
        "dry_run": args.dry_run,
        "duration_sec": duration,
        "score_status": "generation_only; official execution metrics are deferred to benchmark harness",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                f"# {args.benchmark} Evaluation Smoke",
                "",
                f"Model: `{args.model}`",
                f"Split: `{metrics['split']}`",
                f"Tasks: {len(rows)}",
                f"Generated: {metrics['generated_count']}",
                f"Dry run: {args.dry_run}",
                "",
                "This Stage 1 entry produces predictions and metadata. Official pass/fail execution is run by the benchmark harness.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
