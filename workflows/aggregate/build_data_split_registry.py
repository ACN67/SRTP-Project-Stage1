#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(p): h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
def main():
    parser=argparse.ArgumentParser(description='Validate data split registry hashes.'); parser.add_argument('--write', action='store_true'); args=parser.parse_args()
    with (ROOT/'results/status/data_splits.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    for r in rows: assert sha(ROOT/r['path'])==r['sha256']
    print(f'splits={len(rows)}'); return 0
if __name__=='__main__': raise SystemExit(main())
