#!/usr/bin/env python3
"""Score a LiveCodeBench code-generation split from local generations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def starter_from_prompt(prompt: str) -> str:
    marker = "\n\nStarter code:\n"
    if marker not in prompt:
        return ""
    return prompt.split(marker, 1)[1]


def candidate_code(row: dict) -> str:
    completion = row.get("completion") or ""
    prompt = row.get("prompt") or ""
    starter = starter_from_prompt(prompt)
    if starter and "class Solution" not in completion and "def " not in completion[:80]:
        return starter + completion
    return completion.strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--generations", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--lcb-release", default="release_v1")
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--num-process-evaluate", type=int, default=4)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_rows = read_jsonl(args.split)
    gen_rows = {row["task_id"]: row for row in read_jsonl(args.generations)}

    problems = {
        item.question_id: item
        for item in load_code_generation_dataset(release_version=args.lcb_release)
    }

    samples = []
    generations = []
    details = []
    for row in split_rows:
        task_id = row["task_id"]
        if task_id not in problems:
            raise KeyError(f"{task_id} not found in LiveCodeBench {args.lcb_release}")
        if task_id not in gen_rows:
            raise KeyError(f"{task_id} not found in generations")
        problem = problems[task_id]
        samples.append(problem.get_evaluation_sample())
        code = candidate_code(gen_rows[task_id])
        generations.append([code])
        details.append(
            {
                "task_id": task_id,
                "platform": problem.platform.value,
                "difficulty": problem.difficulty.value,
                "generated_chars": len(code),
            }
        )

    metrics, results, metadata = codegen_metrics(
        samples,
        generations,
        k_list=[1],
        num_process_evaluate=args.num_process_evaluate,
        timeout=args.timeout,
        debug=False,
    )

    pass_count = 0
    status_counter: Counter[str] = Counter()
    detail_rows = []
    for idx, detail in enumerate(details):
        passed = bool(results[idx][0]) and all(results[idx][0])
        pass_count += int(passed)
        status_counter["pass" if passed else "fail"] += 1
        detail_rows.append(
            {
                **detail,
                "pass": passed,
                "result": results[idx][0],
                "metadata": metadata[idx][0] if idx < len(metadata) and metadata[idx] else None,
            }
        )

    summary = {
        "status": "success",
        "benchmark": "livecodebench",
        "release": args.lcb_release,
        "split": str(args.split),
        "generations": str(args.generations),
        "task_count": len(split_rows),
        "pass_count": pass_count,
        "pass_rate": pass_count / len(split_rows) if split_rows else None,
        "status_counter": dict(status_counter),
        "official_metrics": metrics,
    }

    (args.out_dir / "score_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (args.out_dir / "score_details.jsonl").open("w", encoding="utf-8") as handle:
        for row in detail_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
