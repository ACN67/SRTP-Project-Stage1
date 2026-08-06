#!/usr/bin/env python3
"""Convert predictions.jsonl (completion) to evalplus-style samples.jsonl (solution)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def normalize_task_id(benchmark: str, task_id: str) -> str:
    tid = str(task_id)
    if benchmark == "humaneval":
        if not tid.startswith("HumanEval/"):
            tid = f"HumanEval/{tid}"
        return tid
    if benchmark == "mbpp":
        if tid.startswith("Mbpp/"):
            return tid
        if tid.startswith("mbpp/"):
            return "Mbpp/" + tid.split("/", 1)[1]
        return f"Mbpp/{tid}"
    return tid


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if "```" not in text:
        return text
    parts = text.split("```")
    if len(parts) >= 3:
        body = parts[1]
        if body.startswith("python"):
            body = body[len("python") :]
        elif body.startswith("py"):
            body = body[len("py") :]
        return body.lstrip("\n").rstrip()
    return text.replace("```", "").strip()


def clean_humaneval_completion(completion: str) -> str:
    text = strip_code_fences(completion)
    cut_markers = ["\nif __name__", "\n<|", "<|fim", "\n```", "\n# Explanation", "\n# Example usage"]
    cut_at = len(text)
    for marker in cut_markers:
        idx = text.find(marker)
        if idx != -1:
            cut_at = min(cut_at, idx)
    text = text[:cut_at].rstrip() + "\n"
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return ""
    first = lines[0]
    if first.strip() and not first.startswith((" ", "\t")):
        lines = [("    " + line) if line.strip() else "" for line in lines]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmark", required=True, choices=["humaneval", "mbpp"])
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with pred_path.open(encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            tid = row.get("task_id")
            if tid is None:
                continue
            completion = row.get("completion") or ""
            prompt = row.get("prompt") or ""
            if args.benchmark == "humaneval":
                body = clean_humaneval_completion(completion)
                # score_humaneval_smoke / evalplus check_correctness expect full program
                solution = (prompt or "") + body
            else:
                body = strip_code_fences(completion)
                solution = body if not prompt else ((prompt + "\n" + body) if "def " not in body[:80] else body)
            sample = {
                "task_id": normalize_task_id(args.benchmark, str(tid)),
                "solution": solution,
                "completion": body,
            }
            dst.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
    print(json.dumps({"samples": count, "output": str(out_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
