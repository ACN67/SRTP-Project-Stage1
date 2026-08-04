#!/usr/bin/env python3

from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BENCHMARK_ROOT=ROOT/'data'/'benchmarks'
def read_jsonl(path: Path): return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
def sha(path: Path): h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()
def rel(path: Path, root: Path):
    try: return path.relative_to(ROOT).as_posix()
    except ValueError: return path.as_posix()
def reject_old(output_root: Path):
    s=output_root.resolve().as_posix()
    if s.endswith('/' + 'data' + '/' + 'splits') or '/' + 'data' + '/' + 'splits' + '/' in s: raise ValueError('refusing to write retired split root')
def write_manifest(split_dir: Path, protocol: str, dataset: str, dry_run: bool=False):
    manifest={'protocol':protocol,'dataset':dataset,'benchmark':dataset.replace('_evalplus',''),'split_version':'stage1_round3_v1','contains_solution':False}
    for role,name in [('guide','guide.jsonl'),('eval','eval.jsonl'),('heldout_eval','heldout_eval.jsonl')]:
        p=split_dir/name
        if p.exists():
            manifest[f'{role}_path']=rel(p, split_dir.parent.parent)
            manifest[f'{role}_sha256']=sha(p)
            manifest[f'{role}_count']=len(read_jsonl(p))
    if protocol=='auxiliary_full_eval': manifest['split_policy']='guide_subset_of_full_eval'
    else: manifest['split_policy']='guide_eval_disjoint'
    if not dry_run: (split_dir/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+chr(10),encoding='utf-8')
    return manifest

def copy_split(source_dir: Path, target_dir: Path, dry_run: bool) -> None:
    if dry_run:
        return
    if source_dir.resolve() == target_dir.resolve():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        if item.name == 'manifest.json':
            continue
        dest = target_dir/item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

def main():
    parser=argparse.ArgumentParser(description='Refresh auxiliary full-evaluation manifests.')
    parser.add_argument('--output-root', type=Path, default=BENCHMARK_ROOT/'auxiliary_full_eval')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--guide-count', type=int, default=32)
    args=parser.parse_args(); out=args.output_root if args.output_root.is_absolute() else ROOT/args.output_root; reject_old(out)
    source=BENCHMARK_ROOT/'auxiliary_full_eval'; plan=[]
    for ds in sorted(p.name for p in source.iterdir() if p.is_dir()):
        source_dir=source/ds
        split_dir=(out/ds) if out!=source else source_dir
        if args.dry_run:
            manifest=write_manifest(source_dir,'auxiliary_full_eval',ds,True)
            for key in ['guide_path','eval_path','heldout_eval_path']:
                if key in manifest:
                    manifest[key]=(split_dir/Path(manifest[key]).name).as_posix()
        else:
            copy_split(source_dir, split_dir, False)
            manifest=write_manifest(split_dir,'auxiliary_full_eval',ds,False)
        plan.append({'dataset':ds,'manifest':manifest})
    print(json.dumps({'dry_run':args.dry_run,'output_root':str(out),'plan':plan},ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
