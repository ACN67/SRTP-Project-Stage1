#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json,subprocess,os,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
FIELDS=['method','venv_name','lock_file','extra_install','notes']
def read_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def atomic(path,text):
    path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=path.name+'.',dir=str(path.parent),text=True)
    with os.fdopen(fd,'w',encoding='utf-8',newline='') as h: h.write(text)
    Path(tmp).replace(path)
def py_for(venv):
    for rel in ['bin/python','Scripts/python.exe']:
        p=venv/rel
        if p.exists(): return p
    return None
def freeze(py):
    out=subprocess.check_output([str(py),'-m','pip','freeze'],text=True)
    return '\n'.join(sorted(x.strip() for x in out.splitlines() if x.strip()))+'\n'
def main():
    ap=argparse.ArgumentParser(description='Capture and check virtual-environment pip freeze locks.'); ap.add_argument('--write',action='store_true'); ap.add_argument('--check',action='store_true'); ap.add_argument('--allow-missing',action='store_true'); ap.add_argument('--venv-root',type=Path,default=ROOT); ap.add_argument('--method-map',type=Path,default=ROOT/'environment/method_env_map.csv'); ap.add_argument('--output-map',type=Path,default=ROOT/'environment/method_env_map.csv'); ap.add_argument('--lock-root',type=Path,default=ROOT/'environment/locks'); ap.add_argument('--method',action='append')
    args=ap.parse_args(); rows=read_rows(args.method_map); selected=set(args.method or [])
    content_to_lock={}; out_rows=[]; report=[]; ok=True
    for row in rows:
        if selected and row['method'] not in selected: continue
        venv=args.venv_root/row['venv_name']; py=py_for(venv)
        if py is None:
            report.append({'method':row['method'],'status':'missing_environment'}); ok=False; out_rows.append(row); continue
        text=freeze(py); digest=hashlib.sha256(text.encode()).hexdigest(); lock_name=content_to_lock.setdefault(digest, 'shared_pruning.txt' if text else row['method']+'.txt')
        lock_path=args.lock_root/lock_name; rel_lock=lock_path.relative_to(ROOT).as_posix() if str(lock_path).startswith(str(ROOT)) else lock_path.as_posix()
        status='match' if lock_path.exists() and lock_path.read_text(encoding='utf-8')==text else 'drift'
        if args.write: atomic(lock_path,text); status='written'
        elif status!='match': ok=False
        out=dict(row); out['lock_file']=rel_lock; out['notes']=(row.get('notes','')+' lock_sha256='+digest).strip(); out_rows.append(out); report.append({'method':row['method'],'status':status,'lock_file':rel_lock,'sha256':digest})
    if args.write:
        with args.output_map.open('w',encoding='utf-8',newline='') as f: w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(out_rows)
    print(json.dumps({'ok':ok or args.allow_missing,'report':report},ensure_ascii=False,indent=2))
    return 0 if (ok or args.allow_missing or args.write) else 1
if __name__=='__main__': raise SystemExit(main())
