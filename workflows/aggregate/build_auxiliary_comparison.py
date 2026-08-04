#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main() -> int:
    parser=argparse.ArgumentParser(description='Validate auxiliary full-evaluation aggregate table.')
    parser.add_argument('--write', action='store_true', help='Validate and keep the aggregate table when raw evidence is unavailable.')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'results/auxiliary/full_eval/comparison.csv')
    args=parser.parse_args()
    path=args.output if args.output.is_absolute() else ROOT/args.output
    rows=list(csv.DictReader(path.open(encoding='utf-8-sig')))
    ok=bool(rows) and {'method','model','benchmark','metric','value'} <= set(rows[0])
    print(json.dumps({'evidence_status':'aggregate_only','row_count':len(rows),'ok':ok}, indent=2))
    return 0 if ok else 1
if __name__=='__main__': raise SystemExit(main())
