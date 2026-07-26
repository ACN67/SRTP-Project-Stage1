#!/usr/bin/env python3
"""Generate full-benchmark predictions for formal Pass@1 evaluation."""

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


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if "```" not in text:
        return text
    parts = text.split("```")
    # Prefer fenced body if present.
    if len(parts) >= 3:
        body = parts[1]
        if body.startswith("python"):
            body = body[len("python") :]
        elif body.startswith("py"):
            body = body[len("py") :]
        return body.lstrip("\n").rstrip()
    return text.replace("```", "").strip()


def clean_humaneval_completion(completion: str) -> str:
    """Truncate junk and ensure function-body indentation for prompt+completion eval."""
    text = strip_code_fences(completion)
    cut_markers = [
        "\nif __name__",
        "\n<|",
        "<|fim",
        "\n```",
        "\n# Explanation",
        "\n# Example usage",
    ]
    cut_at = len(text)
    for marker in cut_markers:
        idx = text.find(marker)
        if idx != -1:
            cut_at = min(cut_at, idx)
    text = text[:cut_at].rstrip() + "\n"
    lines = text.splitlines()
    # Drop leading blank lines.
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return ""
    # If the body is not indented, indent all non-empty lines by 4 spaces.
    first = lines[0]
    if first.strip() and not first.startswith((" ", "\t")):
        # Preserve relative indentation; only add function-body base indent.
        lines = [("    " + line) if line.strip() else "" for line in lines]
    return "\n".join(lines) + "\n"


def clean_mbpp_completion(completion: str) -> str:
    text = strip_code_fences(completion)
    cut_markers = ["\nif __name__", "\n<|", "<|fim", "\n```"]
    cut_at = len(text)
    for marker in cut_markers:
        idx = text.find(marker)
        if idx != -1:
            cut_at = min(cut_at, idx)
    return text[:cut_at].rstrip() + "\n"


def build_prompt(row: dict[str, Any], tokenizer: Any | None = None, benchmark: str = "") -> str:
    task = str(row.get("prompt") or "")
    benchmark = benchmark or str(row.get("benchmark") or "")
    # HumanEval: ask Instruct models for a correctly indented function-body continuation.
    if benchmark == "humaneval":
        user = (
            "Continue the following Python function. "
            "Output ONLY the function body continuation (statements inside the function). "
            "Do not repeat the function signature or docstring. "
            "Use 4-space indentation for the body. "
            "Do not use markdown fences.\n\n"
            f"{task}"
        )
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": "You are a careful Python coder."},
                {"role": "user", "content": user},
            ]
            try:
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        return user
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {
                "role": "system",
                "content": "You are a coding assistant. Return only Python code without markdown fences.",
            },
            {
                "role": "user",
                "content": f"Write a Python solution for the following problem. Return only code.\n\n{task}",
            },
        ]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return "\n".join(
        [
            "### Task",
            task,
            "",
            "### Instruction",
            "Return only the Python code solution. Do not use markdown fences.",
        ]
    )


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
    for idx, row in enumerate(rows, start=1):
        prompt = build_prompt(row, tokenizer=tokenizer, benchmark=args.benchmark)
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
        if args.benchmark == "humaneval":
            completion = clean_humaneval_completion(completion)
        else:
            completion = clean_mbpp_completion(completion)
        predictions.append(
            {
                "task_id": row.get("task_id"),
                "benchmark": row.get("benchmark"),
                "prompt": row.get("prompt"),
                "completion": completion,
                "status": "generated",
            }
        )
        if idx % 20 == 0 or idx == len(rows):
            print(f"generated {idx}/{len(rows)}", flush=True)
    return predictions


def to_evalplus_samples(predictions: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            tid = row.get("task_id")
            if tid is None:
                continue
            # MBPP evalplus expects Mbpp/N ids sometimes; keep raw and also write as-is.
            handle.write(
                json.dumps(
                    {"task_id": tid, "completion": row.get("completion") or ""},
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate full formal benchmark predictions.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", required=True, help="Path to eval.jsonl (full set)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--benchmark", required=True, choices=["humaneval", "mbpp"])
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=512)
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
    predictions = generate_predictions(args, rows)
    duration = round(time.monotonic() - start, 3)

    write_jsonl(output_dir / "predictions.jsonl", predictions)
    to_evalplus_samples(predictions, output_dir / "evalplus_samples.jsonl")
    metrics = {
        "benchmark": args.benchmark,
        "model": args.model,
        "split": str(split_path.relative_to(ROOT) if split_path.is_relative_to(ROOT) else split_path),
        "task_count": len(rows),
        "generated_count": sum(1 for row in predictions if row["status"] == "generated"),
        "duration_sec": duration,
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "score_status": "generation_complete; run evalplus next",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
