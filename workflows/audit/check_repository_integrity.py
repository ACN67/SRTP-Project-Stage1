#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def count_csv(path: Path) -> int:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> int:
    checks = {
        "methods": count_csv(ROOT / "results/status/methods.csv") >= 12,
        "runs": count_csv(ROOT / "results/status/runs.csv") >= 77,
        "scores": count_csv(ROOT / "results/status/scores.csv") > 0,
        "formal": count_csv(ROOT / "results/formal/r4_half/scores.csv") > 0,
        "evidence": (ROOT / "results/evidence").exists() and any((ROOT / "results/evidence").iterdir()),
        "data": (ROOT / "data/benchmarks").exists() and any((ROOT / "data/benchmarks").rglob("manifest.json")),
    }
    out = {"repository_integrity": all(checks.values()), "checks": checks}
    print(json.dumps(out, indent=2))
    return 0 if out["repository_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
