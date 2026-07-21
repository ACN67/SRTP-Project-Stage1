#!/usr/bin/env python3
"""Summarize a runner resource.csv file into resource_summary.json."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def summarize_resource_csv(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"resource csv is not a file: {path}")

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    process_values = [v for row in rows if (v := parse_float(row.get("process_rss_mb"))) is not None]
    gpu_values = [v for row in rows if (v := parse_float(row.get("gpu_memory_used_mb"))) is not None]
    elapsed_values = [v for row in rows if (v := parse_float(row.get("elapsed_sec"))) is not None]

    return {
        "status": "success",
        "source": str(path),
        "sample_count": len(rows),
        "duration_observed_sec": max(elapsed_values) if elapsed_values else None,
        "peak_process_rss_mb": max(process_values) if process_values else None,
        "mean_process_rss_mb": sum(process_values) / len(process_values) if process_values else None,
        "peak_gpu_memory_used_mb": max(gpu_values) if gpu_values else None,
        "mean_gpu_memory_used_mb": sum(gpu_values) / len(gpu_values) if gpu_values else None,
        "first_timestamp": rows[0].get("timestamp") if rows else None,
        "last_timestamp": rows[-1].get("timestamp") if rows else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("resource_csv", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summary = summarize_resource_csv(args.resource_csv)
    output = args.output or args.resource_csv.with_name("resource_summary.json")
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
