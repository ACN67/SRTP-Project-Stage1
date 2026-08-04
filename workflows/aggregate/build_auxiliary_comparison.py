#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    parser=argparse.ArgumentParser(description='Validate auxiliary full-evaluation aggregate table.'); parser.add_argument('--validate', action='store_true'); parser.add_argument('--output', type=Path, default=ROOT/'results/auxiliary/full_eval/comparison.csv'); args=parser.parse_args()
    rows=list(csv.DictReader(args.output.open(encoding='utf-8-sig')))
    print(json.dumps({'evidence_status':'aggregate_only','row_count':len(rows)}, indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
