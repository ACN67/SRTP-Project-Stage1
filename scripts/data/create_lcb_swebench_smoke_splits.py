#!/usr/bin/env python3
"""Create small LiveCodeBench and SWE-bench Lite guide/eval splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]


def write_jsonl(path: Path, rows: Iterable[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    material = ""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True)
            material += line + "\n"
            handle.write(line + "\n")
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def write_manifest(path: Path, benchmark: str, guide_path: Path, eval_path: Path, guide_hash: str, eval_hash: str, notes: str) -> None:
    manifest = {
        "benchmark": benchmark,
        "split_version": "stage1_smoke_v1",
        "guide_path": str(guide_path.relative_to(ROOT)),
        "eval_path": str(eval_path.relative_to(ROOT)),
        "guide_sha256": guide_hash,
        "eval_sha256": eval_hash,
        "guide_count": sum(1 for _ in guide_path.open(encoding="utf-8")),
        "eval_count": sum(1 for _ in eval_path.open(encoding="utf-8")),
        "contains_solution": False,
        "notes": notes,
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def create_split(name: str, rows: list[dict], guide_count: int, eval_count: int, notes: str) -> None:
    guide = [dict(row, split_role="guide") for row in rows[:guide_count]]
    eval_rows = [dict(row, split_role="eval") for row in rows[guide_count : guide_count + eval_count]]
    split_dir = ROOT / "data" / "splits" / name
    guide_path = split_dir / "guide.jsonl"
    eval_path = split_dir / "eval.jsonl"
    guide_hash = write_jsonl(guide_path, guide)
    eval_hash = write_jsonl(eval_path, eval_rows)
    write_manifest(split_dir / "manifest.json", name, guide_path, eval_path, guide_hash, eval_hash, notes)
    print(f"{name}: guide={len(guide)} eval={len(eval_rows)}")
    print(f"{name}: guide_sha256={guide_hash}")
    print(f"{name}: eval_sha256={eval_hash}")


def livecodebench_records(release_version: str) -> list[dict]:
    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset

    problems = load_code_generation_dataset(release_version=release_version)
    problems = sorted(problems, key=lambda item: (item.contest_date, item.question_id))
    rows = []
    for item in problems:
        prompt = item.question_content
        if item.starter_code:
            prompt = prompt.rstrip() + "\n\nStarter code:\n" + item.starter_code
        rows.append(
            {
                "benchmark": "livecodebench",
                "task_id": item.question_id,
                "prompt": prompt,
                "context": "",
                "metadata": {
                    "release_version": release_version,
                    "platform": item.platform.value,
                    "contest_id": item.contest_id,
                    "contest_date": item.contest_date.isoformat(),
                    "difficulty": item.difficulty.value,
                    "public_test_count": len(item.public_test_cases),
                    "private_test_count": len(item.private_test_cases),
                },
                "contains_solution": False,
            }
        )
    return rows


def swebench_records(dataset_name: str, split: str) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)
    rows = []
    for item in dataset:
        task_id = item.get("instance_id")
        prompt_parts = [
            f"Repository: {item.get('repo')}",
            f"Base commit: {item.get('base_commit')}",
            "",
            "Problem statement:",
            item.get("problem_statement") or "",
        ]
        rows.append(
            {
                "benchmark": "swebench_lite",
                "task_id": task_id,
                "prompt": "\n".join(prompt_parts),
                "context": "",
                "metadata": {
                    "dataset": dataset_name,
                    "split": split,
                    "repo": item.get("repo"),
                    "base_commit": item.get("base_commit"),
                    "version": item.get("version"),
                },
                "contains_solution": False,
            }
        )
    rows.sort(key=lambda row: row["task_id"])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Create deterministic LiveCodeBench/SWE-bench Lite smoke splits.")
    parser.add_argument("--guide-count", type=int, default=4)
    parser.add_argument("--eval-count", type=int, default=4)
    parser.add_argument("--lcb-release", default="release_v1")
    parser.add_argument("--swebench-dataset", default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--swebench-split", default="test")
    args = parser.parse_args()

    create_split(
        "livecodebench",
        livecodebench_records(args.lcb_release),
        args.guide_count,
        args.eval_count,
        "Small deterministic smoke split from LiveCodeBench code_generation_lite, sorted by contest date.",
    )
    create_split(
        "swebench_lite",
        swebench_records(args.swebench_dataset, args.swebench_split),
        args.guide_count,
        args.eval_count,
        "Small deterministic smoke split from SWE-bench Lite. Gold patches and tests are not included in prompt/context.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
