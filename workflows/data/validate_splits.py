#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def readj(p): return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
def sha(p): h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
def main():
    parser=argparse.ArgumentParser(description='Validate split overlap.'); parser.add_argument('--protocol',default='r4_half'); parser.add_argument('--output',type=Path); args=parser.parse_args()
    checks=[]
    for ds in sorted((ROOT/'data/benchmarks'/args.protocol).iterdir()):
        if not ds.is_dir(): continue
        g,e=ds/'guide.jsonl',ds/'eval.jsonl'
        if g.exists() and e.exists():
            gi={r.get('task_id') for r in readj(g)}; ei={r.get('task_id') for r in readj(e)}; checks.append({'benchmark':ds.name,'ok':gi.isdisjoint(ei),'guide_count':len(gi),'eval_count':len(ei),'guide_sha256':sha(g),'eval_sha256':sha(e),'issues':[] if gi.isdisjoint(ei) else ['overlap']})
    out={'ok':all(c['ok'] for c in checks),'protocol':args.protocol,'checks':checks}
    if args.output:
        p=args.output if args.output.is_absolute() else ROOT/args.output; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+chr(10),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2)); return 0 if out['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
