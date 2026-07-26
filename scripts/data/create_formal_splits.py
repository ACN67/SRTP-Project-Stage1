#!/usr/bin/env python3
"""Create formal HumanEval/MBPP guide+eval splits for paper-grade Stage 1 runs.

Policy:
- guide: first N tasks used only for pruning calibration
- eval: FULL benchmark used for Pass@1 scoring
- guide is intentionally a subset of eval (documented in manifest)
"""

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


def write_manifest(
    path: Path,
    benchmark: str,
    split_name: str,
    guide_path: Path,
    eval_path: Path,
    heldout_path: Path,
    guide_hash: str,
    eval_hash: str,
    heldout_hash: str,
    guide_count: int,
    eval_count: int,
    heldout_count: int,
) -> None:
    manifest = {
        "benchmark": benchmark,
        "split_name": split_name,
        "split_version": "stage1_formal_v1",
        "split_policy": "guide_subset_of_eval_formal_v1",
        "guide_path": str(guide_path.relative_to(ROOT)),
        "eval_path": str(eval_path.relative_to(ROOT)),
        "heldout_eval_path": str(heldout_path.relative_to(ROOT)),
        "guide_sha256": guide_hash,
        "eval_sha256": eval_hash,
        "heldout_eval_sha256": heldout_hash,
        "guide_count": guide_count,
        "eval_count": eval_count,
        "heldout_eval_count": heldout_count,
        "contains_solution": False,
        "notes": (
            "Formal split: guide is the first N tasks for pruning calibration; "
            "eval is the FULL benchmark for Pass@1; heldout_eval excludes guide tasks."
        ),
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


def create_formal_split(split_name: str, benchmark: str, rows: list[dict], guide_count: int) -> None:
    if guide_count >= len(rows):
        raise ValueError(f"guide_count={guide_count} must be < total rows={len(rows)}")

    guide = [dict(row, split_role="guide") for row in rows[:guide_count]]
    eval_rows = [dict(row, split_role="eval") for row in rows]
    heldout = [dict(row, split_role="heldout_eval") for row in rows[guide_count:]]

    split_dir = ROOT / "data" / "splits" / split_name
    guide_path = split_dir / "guide.jsonl"
    eval_path = split_dir / "eval.jsonl"
    heldout_path = split_dir / "heldout_eval.jsonl"
    manifest_path = split_dir / "manifest.json"

    guide_hash = write_jsonl(guide_path, guide)
    eval_hash = write_jsonl(eval_path, eval_rows)
    heldout_hash = write_jsonl(heldout_path, heldout)
    write_manifest(
        manifest_path,
        benchmark=benchmark,
        split_name=split_name,
        guide_path=guide_path,
        eval_path=eval_path,
        heldout_path=heldout_path,
        guide_hash=guide_hash,
        eval_hash=eval_hash,
        heldout_hash=heldout_hash,
        guide_count=len(guide),
        eval_count=len(eval_rows),
        heldout_count=len(heldout),
    )
    print(f"{split_name}: guide={len(guide)} eval_full={len(eval_rows)} heldout={len(heldout)}")
    print(f"{split_name}: guide_sha256={guide_hash}")
    print(f"{split_name}: eval_sha256={eval_hash}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create formal HumanEval/MBPP splits.")
    parser.add_argument("--guide-count", type=int, default=32)
    args = parser.parse_args()

    create_formal_split("humaneval_formal", "humaneval", humaneval_records(), args.guide_count)
    create_formal_split("mbpp_formal", "mbpp", mbpp_records(), args.guide_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
