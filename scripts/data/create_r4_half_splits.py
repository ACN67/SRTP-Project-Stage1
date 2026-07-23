#!/usr/bin/env python3
"""Create R4 guide/eval splits for HumanEval, MBPP, and LiveCodeBench."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable


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
    guide_path: Path,
    eval_path: Path,
    guide_hash: str,
    eval_hash: str,
    total_count: int,
    source: str,
    selection: str,
) -> None:
    manifest = {
        "benchmark": benchmark,
        "split_version": "stage1_r4_half_v1",
        "source": source,
        "selection": selection,
        "total_count": total_count,
        "guide_path": str(guide_path.relative_to(ROOT)),
        "eval_path": str(eval_path.relative_to(ROOT)),
        "guide_sha256": guide_hash,
        "eval_sha256": eval_hash,
        "guide_count": sum(1 for _ in guide_path.open(encoding="utf-8")),
        "eval_count": sum(1 for _ in eval_path.open(encoding="utf-8")),
        "contains_solution": False,
        "notes": "R4 formal split. Guide is for pruning/calibration only; eval is for post-pruning scoring.",
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def balanced_stratified_split(
    rows: list[dict],
    stratum_key: Callable[[dict], tuple],
    sort_key: Callable[[dict], tuple],
) -> tuple[list[dict], list[dict]]:
    target_guide = len(rows) // 2
    guide: list[dict] = []
    eval_rows: list[dict] = []
    groups: dict[tuple, list[dict]] = defaultdict(list)

    for row in rows:
        groups[stratum_key(row)].append(row)

    for key in sorted(groups):
        group = sorted(groups[key], key=sort_key)
        start_with_guide = len(guide) <= len(eval_rows)
        for idx, row in enumerate(group):
            goes_to_guide = (idx % 2 == 0) if start_with_guide else (idx % 2 == 1)
            if goes_to_guide:
                guide.append(dict(row, split_role="guide"))
            else:
                eval_rows.append(dict(row, split_role="eval"))

    while len(guide) > target_guide:
        row = guide.pop()
        row["split_role"] = "eval"
        eval_rows.append(row)
    while len(guide) < target_guide and eval_rows:
        row = eval_rows.pop()
        row["split_role"] = "guide"
        guide.append(row)

    guide.sort(key=sort_key)
    eval_rows.sort(key=sort_key)
    return guide, eval_rows


def create_split(
    name: str,
    rows: list[dict],
    source: str,
    selection: str,
    guide_eval: tuple[list[dict], list[dict]],
) -> None:
    guide, eval_rows = guide_eval
    split_dir = ROOT / "data" / "splits" / name
    guide_path = split_dir / "guide.jsonl"
    eval_path = split_dir / "eval.jsonl"
    guide_hash = write_jsonl(guide_path, guide)
    eval_hash = write_jsonl(eval_path, eval_rows)
    write_manifest(split_dir / "manifest.json", name, guide_path, eval_path, guide_hash, eval_hash, len(rows), source, selection)
    print(json.dumps({
        "benchmark": name,
        "total": len(rows),
        "guide": len(guide),
        "eval": len(eval_rows),
        "guide_path": str(guide_path.relative_to(ROOT)),
        "eval_path": str(eval_path.relative_to(ROOT)),
        "guide_sha256": guide_hash,
        "eval_sha256": eval_hash,
    }, ensure_ascii=False))


def humaneval_records() -> list[dict]:
    from human_eval.data import read_problems

    problems = read_problems()
    raw_rows = []
    for task_id in sorted(problems):
        item = problems[task_id]
        prompt = item["prompt"]
        prompt_chars = len(prompt)
        example_count = prompt.count(">>>")
        raw_rows.append({
            "benchmark": "humaneval",
            "task_id": task_id,
            "prompt": prompt,
            "context": "",
            "metadata": {
                "entry_point": item.get("entry_point"),
                "prompt_chars": prompt_chars,
                "prompt_lines": prompt.count("\n") + 1,
                "example_count": example_count,
            },
            "contains_solution": False,
        })
    sorted_lengths = sorted(row["metadata"]["prompt_chars"] for row in raw_rows)
    cut1 = sorted_lengths[len(sorted_lengths) // 4]
    cut2 = sorted_lengths[len(sorted_lengths) // 2]
    cut3 = sorted_lengths[(len(sorted_lengths) * 3) // 4]
    rows = []
    for row in raw_rows:
        chars = row["metadata"]["prompt_chars"]
        if chars <= cut1:
            length_bucket = "short"
        elif chars <= cut2:
            length_bucket = "medium_short"
        elif chars <= cut3:
            length_bucket = "medium_long"
        else:
            length_bucket = "long"
        examples = row["metadata"]["example_count"]
        row["metadata"]["strata"] = {
            "length_bucket": length_bucket,
            "example_bucket": "0" if examples == 0 else "1" if examples == 1 else "2plus",
        }
        rows.append(row)
    return rows


def task_num(task_id: str) -> int:
    _, _, suffix = task_id.partition("/")
    return int(suffix) if suffix.isdigit() else 10**9


def mbpp_evalplus_records() -> tuple[list[dict], list[dict], list[dict]]:
    from datasets import load_dataset
    from evalplus.data import get_mbpp_plus

    problems = get_mbpp_plus()
    official = load_dataset("google-research-datasets/mbpp", "sanitized")
    official_split_by_id: dict[int, str] = {}
    for split_name in ["prompt", "train", "validation", "test"]:
        for item in official[split_name]:
            official_split_by_id[int(item["task_id"])] = split_name

    rows = []
    for task_id in sorted(problems, key=lambda value: (task_num(value), value)):
        item = problems[task_id]
        official_split = official_split_by_id.get(task_num(task_id), "unknown")
        rows.append({
            "benchmark": "mbpp_evalplus",
            "task_id": task_id,
            "prompt": item["prompt"],
            "context": "",
            "metadata": {
                "entry_point": item.get("entry_point"),
                "evalplus_dataset": "mbpp",
                "official_split": official_split,
            },
            "contains_solution": False,
        })
    guide = [dict(row, split_role="guide") for row in rows if row["metadata"]["official_split"] in {"prompt", "train", "validation"}]
    eval_rows = [dict(row, split_role="eval") for row in rows if row["metadata"]["official_split"] == "test"]
    return rows, guide, eval_rows


def livecodebench_records(release_version: str) -> list[dict]:
    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset

    problems = load_code_generation_dataset(release_version=release_version)
    problems = sorted(problems, key=lambda item: (item.contest_date, item.question_id))
    rows = []
    for item in problems:
        prompt = item.question_content
        if item.starter_code:
            prompt = prompt.rstrip() + "\n\nStarter code:\n" + item.starter_code
        rows.append({
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
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Create R4 half guide/eval splits.")
    parser.add_argument(
        "--benchmark",
        choices=["all", "humaneval", "mbpp_evalplus", "livecodebench"],
        default="all",
        help="Run only benchmarks available in the active Python environment when needed.",
    )
    parser.add_argument("--lcb-release", default="release_v1")
    args = parser.parse_args()

    if args.benchmark in {"all", "humaneval"}:
        rows = humaneval_records()
        create_split(
            "humaneval_half",
            rows,
            "human_eval.data.read_problems",
            "stratified by prompt length bucket and example count, then stable alternating assignment",
            balanced_stratified_split(
                rows,
                lambda row: (
                    row["metadata"]["strata"]["length_bucket"],
                    row["metadata"]["strata"]["example_bucket"],
                ),
                lambda row: (task_num(row["task_id"]), row["task_id"]),
            ),
        )
    if args.benchmark in {"all", "mbpp_evalplus"}:
        rows, guide, eval_rows = mbpp_evalplus_records()
        create_split(
            "mbpp_evalplus_half",
            rows,
            "evalplus.data.get_mbpp_plus plus google-research-datasets/mbpp sanitized official splits",
            "official MBPP split mapping: prompt/train/validation guide, test eval",
            (guide, eval_rows),
        )
    if args.benchmark in {"all", "livecodebench"}:
        rows = livecodebench_records(args.lcb_release)
        create_split(
            "livecodebench_half",
            rows,
            f"LiveCodeBench code_generation {args.lcb_release}",
            "stratified by platform and difficulty, then stable time-ordered alternating assignment",
            balanced_stratified_split(
                rows,
                lambda row: (row["metadata"]["platform"], row["metadata"]["difficulty"]),
                lambda row: (
                    row["metadata"]["contest_date"],
                    row["task_id"],
                ),
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
