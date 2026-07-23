#!/usr/bin/env python3
"""Check visible Stage 1 readiness gaps without running heavy experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def exists(path: str) -> bool:
    return (ROOT / path).exists()


def count_glob(pattern: str) -> int:
    return len(list(ROOT.glob(pattern)))


def main() -> int:
    checks = [
        ("stage1_plan", exists("docs/stage1.md"), "Stage 1 plan present"),
        ("method_notes", exists("docs/methods.md"), "Method status notes present"),
        ("qwen_15b_probe", count_glob("results/raw/qwen25_coder_15b_config_probe_*/qwen_probe.json") > 0, "Qwen 1.5B config probe recorded"),
        ("qwen_3b_probe", count_glob("results/raw/qwen25_coder_3b_config_probe_*/qwen_probe.json") > 0, "Qwen 3B config probe recorded"),
        ("status_table", exists("results/stage1/status.csv"), "Stage 1 status table present"),
        ("laco_blocker", exists("docs/methods.md"), "LaCo blocker recorded in method notes"),
        ("humaneval_guide", exists("data/splits/humaneval/guide.jsonl"), "HumanEval guide split present"),
        ("humaneval_eval", exists("data/splits/humaneval/eval.jsonl"), "HumanEval eval split present"),
        ("mbpp_guide", exists("data/splits/mbpp/guide.jsonl"), "MBPP guide split present"),
        ("mbpp_eval", exists("data/splits/mbpp/eval.jsonl"), "MBPP eval split present"),
        ("livecodebench_guide", exists("data/splits/livecodebench/guide.jsonl"), "LiveCodeBench guide split present"),
        ("livecodebench_eval", exists("data/splits/livecodebench/eval.jsonl"), "LiveCodeBench eval split present"),
        ("swebench_guide", exists("data/splits/swebench_lite/guide.jsonl"), "SWE-bench Lite guide split present"),
        ("swebench_eval", exists("data/splits/swebench_lite/eval.jsonl"), "SWE-bench Lite eval split present"),
    ]

    rows = [{"id": item[0], "ok": item[1], "description": item[2]} for item in checks]
    ready = sum(1 for row in rows if row["ok"])
    total = len(rows)

    output = {
        "ready": ready,
        "total": total,
        "missing": [row for row in rows if not row["ok"]],
        "checks": rows,
    }

    out_dir = ROOT / "results" / "stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "readiness.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (out_dir / "readiness.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "ok", "description"])
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if ready == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
