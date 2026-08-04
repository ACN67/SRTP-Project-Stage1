#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from workflows.aggregate.registry_utils import ARTIFACT_FIELDS, ROOT, build_artifact_rows, csv_text, write_or_check


def main() -> int:
    parser = argparse.ArgumentParser(description="Build results/status/artifacts.csv.")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results/status/artifacts.csv")
    args = parser.parse_args()
    if not args.write and not args.check:
        args.check = True
    rows = build_artifact_rows()
    text = csv_text(ARTIFACT_FIELDS, rows)
    ok = write_or_check(args.output if args.output.is_absolute() else ROOT / args.output, text, args.write, args.check)
    print("rows=" + str(len(rows)))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
