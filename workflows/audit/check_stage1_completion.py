#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def read(n):
    with (ROOT/'results/status'/n).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def main():
    parser=argparse.ArgumentParser(description='Check Stage 1 completion from status registries.'); parser.add_argument('--write', action='store_true'); args=parser.parse_args()
    checks=[{'id':'methods','ok':len(read('methods.csv'))==12,'description':'methods indexed'},{'id':'runs','ok':len(read('runs.csv'))>=77,'description':'runs indexed'},{'id':'scores','ok':len(read('scores.csv'))>0,'description':'scores indexed'},{'id':'splits','ok':len(read('data_splits.csv'))>0,'description':'splits indexed'}]
    out={'ok':all(c['ok'] for c in checks),'checks':checks}
    if args.write:
        (ROOT/'results/status/completion_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+chr(10),encoding='utf-8')
        with (ROOT/'results/status/completion_audit.csv').open('w',encoding='utf-8',newline='') as f: x=csv.DictWriter(f,fieldnames=['id','ok','description']);x.writeheader();x.writerows(checks)
    print(json.dumps(out,ensure_ascii=False,indent=2)); return 0 if out['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
