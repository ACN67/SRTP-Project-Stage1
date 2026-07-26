#!/usr/bin/env python3
"""Convert predictions and run evalplus for formal Pass@1."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def normalize_task_id(benchmark: str, task_id: str) -> str:
    tid = str(task_id)
    if benchmark == "humaneval":
        if not tid.startswith("HumanEval/"):
            tid = f"HumanEval/{tid}"
        return tid
    if benchmark == "mbpp":
        # evalplus expects Mbpp/<id>
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


def write_samples(predictions_path: Path, out_path: Path, benchmark: str) -> int:
    count = 0
    with predictions_path.open(encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            tid = row.get("task_id")
            if tid is None:
                continue
            completion = row.get("completion") or ""
            if benchmark == "humaneval":
                completion = clean_humaneval_completion(completion)
            else:
                completion = strip_code_fences(completion)
            sample = {
                "task_id": normalize_task_id(benchmark, str(tid)),
                "completion": completion,
            }
            dst.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Run evalplus Pass@1 on generated predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--benchmark", required=True, choices=["humaneval", "mbpp"])
    parser.add_argument("--python-bin", default=str(ROOT / ".venv-common" / "bin" / "python"))
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.is_absolute():
        pred_path = ROOT / pred_path
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = out_dir / "samples.jsonl"
    n = write_samples(pred_path, samples, args.benchmark)
    dataset = "humaneval" if args.benchmark == "humaneval" else "mbpp"
    stdout_path = out_dir / "evalplus_stdout.log"
    stderr_path = out_dir / "evalplus_stderr.log"

    cmd = [
        args.python_bin,
        "-m",
        "evalplus.evaluate",
        "--dataset",
        dataset,
        "--samples",
        str(samples),
    ]
    with stdout_path.open("w", encoding="utf-8") as out_f, stderr_path.open("w", encoding="utf-8") as err_f:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=out_f, stderr=err_f, text=True)

    summary = {
        "benchmark": args.benchmark,
        "dataset": dataset,
        "sample_count": n,
        "samples": str(samples.relative_to(ROOT)),
        "returncode": proc.returncode,
        "stdout": str(stdout_path.relative_to(ROOT)),
        "stderr": str(stderr_path.relative_to(ROOT)),
    }
    # Best-effort parse pass@1 from stdout
    text = stdout_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        low = line.lower()
        if "pass@1" in low or "pass@1" in line.replace(" ", "").lower():
            summary.setdefault("metric_lines", []).append(line.strip())
    (out_dir / "evalplus_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
