#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def readj(p): return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
def sha(p): h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
def ids(rows): return {r.get('task_id') for r in rows}
def check_dataset(protocol, ds):
    g,e,h=ds/'guide.jsonl',ds/'eval.jsonl',ds/'heldout_eval.jsonl'
    if protocol=='auxiliary_full_eval' and g.exists() and e.exists() and h.exists():
        gi,fi,hi=ids(readj(g)),ids(readj(e)),ids(readj(h))
        return {'protocol':protocol,'dataset':ds.name,'guide_count':len(gi),'full_eval_count':len(fi),'heldout_count':len(hi),'guide_subset_of_full':gi<=fi,'guide_heldout_disjoint':gi.isdisjoint(hi),'full_partition_valid':fi==gi|hi,'ok':gi<=fi and gi.isdisjoint(hi) and fi==gi|hi,'guide_sha256':sha(g),'eval_sha256':sha(e),'heldout_eval_sha256':sha(h)}
    if g.exists() and e.exists():
        gi,ei=ids(readj(g)),ids(readj(e)); disjoint=gi.isdisjoint(ei)
        return {'protocol':protocol,'dataset':ds.name,'guide_count':len(gi),'eval_count':len(ei),'guide_eval_disjoint':disjoint,'ok':disjoint,'guide_sha256':sha(g),'eval_sha256':sha(e)}
    if e.exists():
        ei=ids(readj(e)); return {'protocol':protocol,'dataset':ds.name,'eval_count':len(ei),'ok':True,'eval_sha256':sha(e)}
    return {'protocol':protocol,'dataset':ds.name,'ok':False,'issues':['missing split files']}
def main():
    parser=argparse.ArgumentParser(description='Validate split protocol relationships.'); parser.add_argument('--protocol',default='r4_half'); parser.add_argument('--output',type=Path); args=parser.parse_args()
    base=ROOT/'data/benchmarks'/args.protocol
    checks=[check_dataset(args.protocol, ds) for ds in sorted(base.iterdir()) if ds.is_dir()]
    out={'ok':all(c.get('ok') for c in checks),'protocol':args.protocol,'checks':checks}
    if args.output:
        p=args.output if args.output.is_absolute() else ROOT/args.output; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+chr(10),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2)); return 0 if out['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
