#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
FIELDS='run_id,category,method_scope,model,protocol,variant,round,execution_status,validity_status,quality_gate,officiality,result_completeness,evidence_path,metadata_present,summary_present,superseded_by,notes'.split(',')
def main():
    parser=argparse.ArgumentParser(description='Validate run registry against evidence directories.'); parser.add_argument('--write', action='store_true'); args=parser.parse_args()
    with (ROOT/'results/status/runs.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    evidence={p.relative_to(ROOT).as_posix() for p in (ROOT/'results/evidence').glob('*/*') if p.is_dir()}
    assert {r['evidence_path'] for r in rows}==evidence and list(rows[0].keys())==FIELDS
    print(f'runs={len(rows)}'); return 0
if __name__=='__main__': raise SystemExit(main())
