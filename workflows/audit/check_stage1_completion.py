#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def read(name):
    with (ROOT/'results/status'/name).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def main():
    parser=argparse.ArgumentParser(description='Assess repository integrity and Stage 1 completion separately.'); parser.add_argument('--write', action='store_true'); args=parser.parse_args()
    methods=read('methods.csv'); runs=read('runs.csv'); scores=read('scores.csv')
    repository_integrity=len(methods)==12 and len(runs)>=77 and bool(scores)
    completed=[m['method'] for m in methods if m['execution_status']=='completed' and m['validity_status']=='valid']
    partial=[m['method'] for m in methods if m['execution_status']=='partial' or m['r4_status']=='partial']
    pending=[m['method'] for m in methods if m['execution_status']=='planned']
    blocked=[m['method'] for m in methods if m['execution_status']=='blocked' or 'blocked' in m['adapter_status']]
    missing=[]
    if pending: missing.append('planned methods remain without first-stage evidence')
    if blocked: missing.append('blocked methods remain unresolved')
    if partial: missing.append('partial methods remain incomplete')
    out={'repository_integrity':repository_integrity,'stage1_complete':False,'completed_methods':completed,'partial_methods':partial,'pending_methods':pending,'blocked_methods':blocked,'missing_requirements':missing}
    if args.write:
        (ROOT/'results/status/completion_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+chr(10),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2)); return 0 if repository_integrity else 1
if __name__=='__main__': raise SystemExit(main())
