#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from completion_extraction import extract_completion


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = args.out_dir / "samples.jsonl"
    generations_path = args.out_dir / "generations.jsonl"

    total = 0
    changed = 0
    short_before = 0
    short_after = 0

    with samples_path.open("w", encoding="utf-8") as samples, generations_path.open("w", encoding="utf-8") as generations:
        for row in read_jsonl(args.generations):
            total += 1
            prompt = row.get("prompt") or ""
            generated = row.get("generated") or ""
            old_completion = row.get("completion") or ""
            completion = extract_completion(prompt, generated)

            if len(old_completion) <= 1:
                short_before += 1
            if len(completion) <= 1:
                short_after += 1
            if completion != old_completion:
                changed += 1

            task_id = row["task_id"]
            samples.write(json.dumps({"task_id": task_id, "solution": prompt + completion}, ensure_ascii=False) + "\n")

            new_row = {
                **row,
                "completion": completion,
                "reextract_old_completion_chars": len(old_completion),
                "reextract_completion_chars": len(completion),
            }
            generations.write(json.dumps(new_row, ensure_ascii=False) + "\n")

    summary = {
        "status": "success",
        "source_generations": str(args.generations),
        "samples": str(samples_path),
        "generations": str(generations_path),
        "task_count": total,
        "changed_count": changed,
        "completion_chars_le_1_before": short_before,
        "completion_chars_le_1_after": short_after,
    }
    (args.out_dir / "reextract_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
