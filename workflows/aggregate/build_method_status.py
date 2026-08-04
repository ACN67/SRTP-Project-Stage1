#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
FIELDS='method,owner,family,primary_model,upstream_status,adapter_status,smoke_status,r4_status,recovery_status,execution_status,validity_status,quality_gate,officiality,evidence_status,primary_code,readme,notes'.split(',')
def main():
    parser=argparse.ArgumentParser(description='Validate method status registry.'); parser.add_argument('--write', action='store_true'); args=parser.parse_args()
    with (ROOT/'results/status/methods.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    assert len(rows)==12 and list(rows[0].keys())==FIELDS
    print(f'methods={len(rows)}'); return 0
if __name__=='__main__': raise SystemExit(main())
