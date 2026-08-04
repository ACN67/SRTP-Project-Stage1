#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATHS = [
    ROOT / "README.md",
    ROOT / "docs",
    ROOT / "data",
    ROOT / "scripts",
    ROOT / "configs",
]
FORBIDDEN_PATTERNS = [
    "generate_evalplus_samples.py",
    "score_humaneval_smoke.py",
    "score_mbpp_smoke.py",
    "score_livecodebench_split.py",
    "create_smoke_splits.py",
    "create_lcb_swebench_smoke_splits.py",
    "reextract_evalplus_samples.py",
    "lcb_completion",
]
SKIP_DIRS = {
    ".git",
    "__pycache__",
    "results",
    "third_party",
    ".venv-common",
    ".venv-livecodebench",
}


def iter_files(paths: list[Path]):
    for path in paths:
        if path.is_file():
            yield path
            continue
        for item in path.rglob("*"):
            if any(part in SKIP_DIRS for part in item.relative_to(ROOT).parts):
                continue
            if item.is_file():
                yield item


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit official benchmark script policy.")
    parser.add_argument("paths", nargs="*", type=Path, default=DEFAULT_PATHS)
    args = parser.parse_args()

    findings = []
    for path in iter_files([p if p.is_absolute() else ROOT / p for p in args.paths]):
        if path == Path(__file__).resolve():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for pattern in FORBIDDEN_PATTERNS:
                if pattern in line:
                    findings.append(
                        {
                            "path": str(path.relative_to(ROOT)),
                            "line": lineno,
                            "pattern": pattern,
                            "text": line.strip(),
                        }
                    )

    summary = {
        "status": "success" if not findings else "failed",
        "finding_count": len(findings),
        "findings": findings,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
