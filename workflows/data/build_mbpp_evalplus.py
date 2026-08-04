#!/usr/bin/env python3
"""Create MBPP eval split aligned with evalplus problem IDs for formal Pass@1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    material = ""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True)
            material += line + "\n"
            handle.write(line + "\n")
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def main() -> int:
    from evalplus.data import get_mbpp_plus

    problems = get_mbpp_plus()
    rows = []
    for task_id in sorted(problems, key=lambda x: int(str(x).split("/")[-1])):
        item = problems[task_id]
        prompt = item.get("prompt") or item.get("text") or ""
        # evalplus uses Mbpp/N
        rows.append(
            {
                "benchmark": "mbpp",
                "task_id": task_id if str(task_id).startswith("Mbpp/") else f"Mbpp/{task_id}",
                "prompt": prompt,
                "context": "",
                "metadata": {"source": "evalplus.get_mbpp_plus"},
                "contains_solution": False,
                "split_role": "eval",
            }
        )

    split_dir = ROOT / "data" / "splits" / "mbpp_evalplus"
    eval_path = split_dir / "eval.jsonl"
    eval_hash = write_jsonl(eval_path, rows)
    manifest = {
        "benchmark": "mbpp",
        "split_name": "mbpp_evalplus",
        "split_version": "stage1_formal_evalplus_v1",
        "eval_path": str(eval_path.relative_to(ROOT)),
        "eval_sha256": eval_hash,
        "eval_count": len(rows),
        "notes": "Full evalplus MBPP(+) problem set for formal Pass@1 scoring.",
    }
    (split_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"mbpp_evalplus: eval={len(rows)} sha256={eval_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
