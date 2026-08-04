#!/usr/bin/env python3
"""Check Stage 1 guide/eval split separation and manifest hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = [
    "humaneval",
    "mbpp",
    "livecodebench",
    "swebench_lite",
    "humaneval_formal",
    "mbpp_formal",
]


def sha256_jsonl_rows(rows: list[dict[str, Any]]) -> str:
    material = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSONL: {exc}") from exc
    return rows


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("task_id") or row.get("instance_id") or "")


def parse_lcb_date(row: dict[str, Any]) -> datetime | None:
    value = (row.get("metadata") or {}).get("contest_date")
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def check_benchmark(name: str) -> dict[str, Any]:
    base = ROOT / "data" / "splits" / name
    manifest_path = base / "manifest.json"
    manifest = read_json(manifest_path)
    guide_path = ROOT / manifest["guide_path"]
    eval_path = ROOT / manifest["eval_path"]
    guide_rows = read_jsonl(guide_path)
    eval_rows = read_jsonl(eval_path)

    issues: list[str] = []
    guide_ids = {row_id(row) for row in guide_rows}
    eval_ids = {row_id(row) for row in eval_rows}
    guide_ids.discard("")
    eval_ids.discard("")

    policy = str(manifest.get("split_policy") or "")
    overlap = sorted(guide_ids & eval_ids)
    if policy == "guide_subset_of_eval_formal_v1":
        if not guide_ids.issubset(eval_ids):
            issues.append("formal policy requires guide task_ids ⊆ eval task_ids")
        heldout_rel = manifest.get("heldout_eval_path")
        if heldout_rel:
            heldout_rows = read_jsonl(ROOT / heldout_rel)
            heldout_ids = {row_id(row) for row in heldout_rows}
            heldout_ids.discard("")
            if guide_ids & heldout_ids:
                issues.append(f"guide overlaps heldout_eval: {sorted(guide_ids & heldout_ids)}")
            if manifest.get("heldout_eval_count") != len(heldout_rows):
                issues.append(
                    f"heldout_eval_count mismatch: manifest={manifest.get('heldout_eval_count')} actual={len(heldout_rows)}"
                )
    elif overlap:
        issues.append(f"guide/eval task_id overlap: {overlap}")

    if any(row.get("contains_solution") for row in guide_rows + eval_rows):
        issues.append("contains_solution=true found in split rows")

    if manifest.get("guide_count") != len(guide_rows):
        issues.append(f"guide_count mismatch: manifest={manifest.get('guide_count')} actual={len(guide_rows)}")
    if manifest.get("eval_count") != len(eval_rows):
        issues.append(f"eval_count mismatch: manifest={manifest.get('eval_count')} actual={len(eval_rows)}")

    guide_hash = sha256_jsonl_rows(guide_rows)
    eval_hash = sha256_jsonl_rows(eval_rows)
    if manifest.get("guide_sha256") != guide_hash:
        issues.append("guide_sha256 mismatch")
    if manifest.get("eval_sha256") != eval_hash:
        issues.append("eval_sha256 mismatch")

    if name == "livecodebench":
        guide_dates = [d for row in guide_rows if (d := parse_lcb_date(row))]
        eval_dates = [d for row in eval_rows if (d := parse_lcb_date(row))]
        if guide_dates and eval_dates and max(guide_dates) > min(eval_dates):
            issues.append("LiveCodeBench guide contains a later contest date than eval")

    if name == "swebench_lite":
        guide_issue_ids = {row_id(row) for row in guide_rows}
        eval_issue_ids = {row_id(row) for row in eval_rows}
        issue_overlap = sorted(guide_issue_ids & eval_issue_ids)
        if issue_overlap:
            issues.append(f"SWE-bench issue overlap: {issue_overlap}")

    return {
        "benchmark": name,
        "ok": not issues,
        "issues": issues,
        "guide_count": len(guide_rows),
        "eval_count": len(eval_rows),
        "guide_sha256": guide_hash,
        "eval_sha256": eval_hash,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Stage 1 split leakage and manifest hashes.")
    parser.add_argument("--output", default="results/auxiliary/pan_full_eval/split_leakage_check_formal.json")
    args = parser.parse_args()

    results = [check_benchmark(name) for name in BENCHMARKS]
    payload = {
        "ok": all(item["ok"] for item in results),
        "checks": results,
    }

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
