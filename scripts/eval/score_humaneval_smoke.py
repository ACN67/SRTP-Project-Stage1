#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
from evalplus.evaluate import check_correctness, get_groundtruth


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--min-time-limit", type=float, default=1.0)
    parser.add_argument("--gt-time-limit-factor", type=float, default=4.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    split_task_ids = [item["task_id"] for item in read_jsonl(args.split)]
    samples = {item["task_id"]: item for item in read_jsonl(args.samples)}

    all_problems = get_human_eval_plus()
    problems = {task_id: all_problems[task_id] for task_id in split_task_ids}
    dataset_hash = get_human_eval_plus_hash()

    expected_outputs_all = get_groundtruth(all_problems, dataset_hash, set())
    expected_outputs = {task_id: expected_outputs_all[task_id] for task_id in split_task_ids}

    details = []
    status_counter = Counter()

    for completion_id, task_id in enumerate(split_task_ids):
        if task_id not in samples:
            result = {
                "task_id": task_id,
                "status": "missing",
                "base_pass": False,
                "plus_pass": None,
            }
            details.append(result)
            status_counter["missing"] += 1
            continue

        solution = samples[task_id]["solution"]
        checked = check_correctness(
            "humaneval",
            completion_id,
            problems[task_id],
            solution,
            expected_outputs[task_id],
            base_only=args.base_only,
            fast_check=False,
            min_time_limit=args.min_time_limit,
            gt_time_limit_factor=args.gt_time_limit_factor,
        )

        base_status = checked["base"][0]
        plus_status = None if args.base_only else checked["plus"][0]
        base_pass = base_status == "pass"
        plus_pass = None if args.base_only else plus_status == "pass"

        details.append({
            "task_id": task_id,
            "base_status": base_status,
            "base_pass": base_pass,
            "plus_status": plus_status,
            "plus_pass": plus_pass,
        })
        status_counter[base_status] += 1

    total = len(split_task_ids)
    passed = sum(1 for item in details if item.get("base_pass"))
    summary = {
        "status": "success",
        "benchmark": "humaneval",
        "split": str(args.split),
        "samples": str(args.samples),
        "base_only": args.base_only,
        "task_count": total,
        "base_pass_count": passed,
        "base_pass_rate": passed / total if total else 0.0,
        "base_status_counter": dict(status_counter),
    }

    (args.out_dir / "score_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.out_dir / "score_details.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in details),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
