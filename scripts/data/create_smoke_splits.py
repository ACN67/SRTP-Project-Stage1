#!/usr/bin/env python3
"""Create small HumanEval/MBPP guide/eval splits for Stage 1 smoke runs."""

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


def write_manifest(path: Path, benchmark: str, guide_path: Path, eval_path: Path, guide_hash: str, eval_hash: str) -> None:
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
        "notes": "Small deterministic smoke split for pipeline checks; not a formal benchmark result split.",
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def humaneval_records() -> list[dict]:
    from human_eval.data import read_problems

    problems = read_problems()
    rows = []
    for task_id in sorted(problems):
        item = problems[task_id]
        rows.append(
            {
                "benchmark": "humaneval",
                "task_id": task_id,
                "prompt": item["prompt"],
                "context": "",
                "metadata": {"entry_point": item.get("entry_point")},
                "contains_solution": False,
            }
        )
    return rows


def mbpp_records() -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    rows = []
    for item in dataset:
        prompt = item.get("prompt") or item.get("text") or ""
        task_id = str(item.get("task_id"))
        rows.append(
            {
                "benchmark": "mbpp",
                "task_id": task_id,
                "prompt": prompt,
                "context": "",
                "metadata": {
                    "source_file": "datasets:google-research-datasets/mbpp/sanitized/test",
                    "test_count": len(item.get("test_list") or []),
                },
                "contains_solution": False,
            }
        )
    rows.sort(key=lambda row: int(row["task_id"]) if row["task_id"].isdigit() else row["task_id"])
    return rows


def split_rows(rows: list[dict], guide_count: int, eval_count: int) -> tuple[list[dict], list[dict]]:
    guide = [dict(row, split_role="guide") for row in rows[:guide_count]]
    eval_rows = [dict(row, split_role="eval") for row in rows[guide_count : guide_count + eval_count]]
    return guide, eval_rows


def create_split(name: str, rows: list[dict], guide_count: int, eval_count: int) -> None:
    guide, eval_rows = split_rows(rows, guide_count, eval_count)
    split_dir = ROOT / "data" / "splits" / name
    guide_path = split_dir / "guide.jsonl"
    eval_path = split_dir / "eval.jsonl"
    manifest_path = split_dir / "manifest.json"
    guide_hash = write_jsonl(guide_path, guide)
    eval_hash = write_jsonl(eval_path, eval_rows)
    write_manifest(manifest_path, name, guide_path, eval_path, guide_hash, eval_hash)
    print(f"{name}: guide={len(guide)} eval={len(eval_rows)}")
    print(f"{name}: guide_sha256={guide_hash}")
    print(f"{name}: eval_sha256={eval_hash}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create deterministic HumanEval/MBPP smoke guide/eval splits.")
    parser.add_argument("--guide-count", type=int, default=4)
    parser.add_argument("--eval-count", type=int, default=4)
    args = parser.parse_args()

    create_split("humaneval", humaneval_records(), args.guide_count, args.eval_count)
    create_split("mbpp", mbpp_records(), args.guide_count, args.eval_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
