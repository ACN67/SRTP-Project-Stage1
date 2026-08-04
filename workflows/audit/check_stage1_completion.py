#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def read(name):
    with (ROOT/'results/status'/name).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def assess_completion(methods, repository_integrity: bool):
    completed=[m['method'] for m in methods if m['execution_status']=='completed']
    partial=[m['method'] for m in methods if m['execution_status']=='partial' or m.get('r4_status')=='partial']
    pending=[m['method'] for m in methods if m['execution_status']=='planned']
    blocked=[m['method'] for m in methods if m['execution_status']=='blocked' or 'blocked' in m.get('adapter_status','')]
    quality=[m['method'] for m in methods if m.get('quality_gate')=='fail']
    missing=[]
    if pending:
        missing.append('planned methods remain without first-stage evidence')
    if blocked:
        missing.append('blocked methods remain unresolved but have evidence')
    if partial:
        missing.append('partial methods remain incomplete but have evidence')
    execution_closed=repository_integrity and not pending
    all_successful=execution_closed and not blocked and not partial and not quality and len(completed)==len(methods)
    methods_map={m['method']:{'execution_status':m['execution_status'],'validity_status':m['validity_status'],'quality_gate':m.get('quality_gate','')} for m in methods}
    return {'repository_integrity':repository_integrity,'stage1_execution_closed':execution_closed,'stage1_all_methods_successful':all_successful,'stage1_complete':all_successful,'methods':methods_map,'deferred':['SWE-bench-lite formal agent evaluation'],'quality_gate_failures':quality,'completed':completed,'partial':partial,'blocked_with_evidence':blocked,'planned':pending,'completed_methods':completed,'partial_methods':partial,'pending_methods':pending,'blocked_methods':blocked,'missing_requirements':missing}

def main(argv=None):
    parser=argparse.ArgumentParser(description='Assess repository integrity and Stage 1 completion separately.'); parser.add_argument('--write', action='store_true'); args=parser.parse_args(argv)
    methods=read('methods.csv'); runs=read('runs.csv'); scores=read('scores.csv')
    repository_integrity=len(methods)==12 and len(runs)>=77 and bool(scores)
    out=assess_completion(methods, repository_integrity)
    keshu_status = ROOT/'results/status/keshu_completion.json'
    if keshu_status.exists():
        scoped = json.loads(keshu_status.read_text(encoding='utf-8'))
        reason = scoped.get('global_stage', {}).get('reason', 'Owner-scoped completion does not close the global Stage 1 run.')
        out['stage1_execution_closed'] = False
        out['stage1_complete'] = False
        out['owner_scoped_override'] = reason
        if reason not in out['missing_requirements']:
            out['missing_requirements'].append(reason)
    if args.write:
        (ROOT/'results/status/completion_audit.json').write_text(json.dumps(out,ensure_ascii=True,indent=2)+chr(10),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=True,indent=2)); return 0 if repository_integrity else 1
if __name__=='__main__': raise SystemExit(main(sys.argv[1:]))
