#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
FIELDS='score_id,run_id,method,model,variant,benchmark,protocol,split,task_count,pass_count,pass_rate,plus_pass_count,plus_pass_rate,metric_name,metric_value,result_completeness,validity_status,evidence_status,source_file,notes'.split(',')
def read(rel):
    with (ROOT/rel).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def main():
    parser=argparse.ArgumentParser(description='Validate score registry and formal table.'); parser.add_argument('--write', action='store_true'); args=parser.parse_args()
    rows=read('results/status/scores.csv'); formal=read('results/formal/r4_half/scores.csv')
    assert list(rows[0].keys())==FIELDS and all(r['result_completeness']!='pilot' for r in formal)
    print(f'scores={len(rows)} formal={len(formal)}'); return 0
if __name__=='__main__': raise SystemExit(main())
