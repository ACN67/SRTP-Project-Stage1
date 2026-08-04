#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from workflows.aggregate.registry_utils import ROOT, SCORE_FIELDS, build_run_rows, build_score_rows, build_formal_rows, csv_text, write_or_check

def main() -> int:
    parser=argparse.ArgumentParser(description='Build the formal R4-half table from run and score registries.')
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'results/formal/r4_half/scores.csv')
    args=parser.parse_args()
    if not args.write and not args.check: args.check=True
    runs=build_run_rows(); scores=build_score_rows(runs); rows=build_formal_rows(runs, scores)
    text=csv_text(SCORE_FIELDS, rows)
    ok=write_or_check(args.output if args.output.is_absolute() else ROOT/args.output, text, args.write, args.check)
    print('formal_rows='+str(len(rows)))
    return 0 if ok else 1
if __name__=='__main__': raise SystemExit(main())

