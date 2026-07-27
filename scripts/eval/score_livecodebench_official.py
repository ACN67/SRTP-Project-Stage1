#!/usr/bin/env python3
"""Score a LiveCodeBench code-generation split from local generations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from lcb_runner.benchmarks.code_generation import CodeGenerationProblem, load_code_generation_dataset
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


def load_lcb_problems(release: str, config_name: str) -> list[CodeGenerationProblem]:
    if not config_name:
        return load_code_generation_dataset(release_version=release)
    dataset = load_dataset(
        "livecodebench/code_generation_lite",
        config_name,
        split="test",
        version_tag=release,
    )
    problems = [CodeGenerationProblem(**row) for row in dataset]
    print(f"Loaded {len(problems)} problems")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--generations", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--lcb-release", default="release_v1")
    parser.add_argument("--lcb-config", default="release_latest", help="HF dataset config name for LiveCodeBench.")
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--num-process-evaluate", type=int, default=4)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_rows = read_jsonl(args.split)
    gen_rows = {row["task_id"]: row for row in read_jsonl(args.generations)}
    split_ids = {row["task_id"] for row in split_rows}
    generation_ids = set(gen_rows)
    missing_ids = sorted(split_ids - generation_ids)
    extra_ids = sorted(generation_ids - split_ids)

    problems = {
        item.question_id: item
        for item in load_lcb_problems(args.lcb_release, args.lcb_config)
    }
    problem_ids = set(problems)
    missing_problem_ids = sorted(split_ids - problem_ids)

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
                "completion_chars": len(gen_rows[task_id].get("completion") or ""),
                "raw_completion_chars": len(gen_rows[task_id].get("raw_completion") or ""),
                "generated_chars": len(code),
            }
        )

    coverage = {
        "split_task_count": len(split_ids),
        "generation_task_count": len(generation_ids),
        "matched_task_count": len(split_ids & generation_ids),
        "missing_generation_count": len(missing_ids),
        "extra_generation_count": len(extra_ids),
        "missing_problem_count": len(missing_problem_ids),
        "missing_generation_ids": missing_ids[:20],
        "extra_generation_ids": extra_ids[:20],
        "missing_problem_ids": missing_problem_ids[:20],
    }
    print(json.dumps({"event": "id_coverage", **coverage}, ensure_ascii=False), flush=True)
    print(json.dumps({"event": "evaluating", "benchmark": "livecodebench", "task_count": len(samples), "timeout": args.timeout, "num_process_evaluate": args.num_process_evaluate}, ensure_ascii=False), flush=True)

    metrics, results, metadata = codegen_metrics(
        samples,
        generations,
        k_list=[1],
        num_process_evaluate=args.num_process_evaluate,
        timeout=args.timeout,
        debug=False,
    )

    official_detail = metrics.get("detail", {}).get("pass@1", {})
    official_pass_by_index = {
        int(key): bool(value)
        for key, value in official_detail.items()
    }
    pass_count = 0
    status_counter: Counter[str] = Counter()
    detail_rows = []
    for idx, detail in enumerate(details):
        passed = official_pass_by_index.get(idx, False)
        pass_count += int(passed)
        status_counter["pass" if passed else "fail"] += 1
        row = {
            **detail,
            "pass": passed,
            "official_pass_at_1": metrics.get("detail", {}).get("pass@1", {}).get(str(idx)),
            "result": results[idx][0],
            "metadata": metadata[idx][0] if idx < len(metadata) and metadata[idx] else None,
        }
        detail_rows.append(row)
        print(json.dumps({"event": "scored", "index": idx + 1, "total": len(details), "task_id": detail["task_id"], "pass": passed}, ensure_ascii=False), flush=True)

    official_pass_rate = metrics.get("pass@1")
    summary = {
        "status": "success",
        "benchmark": "livecodebench",
        "release": args.lcb_release,
        "config": args.lcb_config,
        "split": str(args.split),
        "generations": str(args.generations),
        "task_count": len(split_rows),
        "pass_count": pass_count,
        "pass_rate": official_pass_rate if official_pass_rate is not None else (pass_count / len(split_rows) if split_rows else None),
        "status_counter": dict(status_counter),
        "id_coverage": coverage,
        "completion_chars_le_1_count": sum(1 for row in details if row["completion_chars"] <= 1),
        "generated_chars_le_1_count": sum(1 for row in details if row["generated_chars"] <= 1),
        "official_metrics": metrics,
    }

    (args.out_dir / "score_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (args.out_dir / "score_details.jsonl").open("w", encoding="utf-8") as handle:
        for row in detail_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
