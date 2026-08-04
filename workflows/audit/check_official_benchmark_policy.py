#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
DEFAULT_PATHS=[ROOT/'README.md',ROOT/'methods',ROOT/'workflows',ROOT/'docs',ROOT/'data']
FORBIDDEN=['generate_evalplus_samples.py','score_humaneval_smoke.py','score_mbpp_smoke.py','score_livecodebench_split.py','create_smoke_splits.py','create_lcb_swebench_smoke_splits.py','reextract_evalplus_samples.py','workflows/evaluate/legacy']
SKIP={'.git','__pycache__','third_party','results'}
def iter_files(paths):
    for path in paths:
        if path.is_file(): yield path; continue
        if not path.exists(): continue
        for item in path.rglob('*'):
            if any(part in SKIP for part in item.relative_to(ROOT).parts): continue
            if item.is_file(): yield item
def main():
    parser=argparse.ArgumentParser(description='Audit active benchmark policy paths.'); parser.add_argument('paths',nargs='*',type=Path,default=DEFAULT_PATHS); args=parser.parse_args()
    findings=[]
    for path in iter_files([p if p.is_absolute() else ROOT/p for p in args.paths]):
        if path.resolve()==Path(__file__).resolve(): continue
        try: text=path.read_text(encoding='utf-8')
        except UnicodeDecodeError: continue
        for lineno,line in enumerate(text.splitlines(),1):
            for pat in FORBIDDEN:
                if pat in line: findings.append({'path':str(path.relative_to(ROOT)),'line':lineno,'pattern':pat})
    out={'status':'success' if not findings else 'failed','finding_count':len(findings),'findings':findings}
    print(json.dumps(out,ensure_ascii=False,indent=2)); return 0 if not findings else 1
if __name__=='__main__': raise SystemExit(main())
