#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json,subprocess,os,tempfile,sys
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
def choose_lock(row, digest, text, lock_root: Path, content_to_lock: dict[str,str]) -> str:
    if digest in content_to_lock:
        return content_to_lock[digest]
    existing = row.get('lock_file','').strip()
    if existing:
        candidate = Path(existing)
        lock_path = candidate if candidate.is_absolute() else ROOT/candidate
        if lock_path.exists() and lock_path.read_text(encoding='utf-8') == text:
            name = lock_path.name if lock_path.parent == lock_root else lock_path.as_posix()
            content_to_lock[digest] = name
            return name
    shared = lock_root/'shared_pruning.txt'
    if shared.exists() and shared.read_text(encoding='utf-8') == text:
        content_to_lock[digest] = shared.name
        return shared.name
    name = f'lock_{digest[:12]}.txt'
    content_to_lock[digest] = name
    return name

def main(argv=None):
    ap=argparse.ArgumentParser(description='Capture and check virtual-environment pip freeze locks.'); ap.add_argument('--write',action='store_true'); ap.add_argument('--check',action='store_true'); ap.add_argument('--allow-missing',action='store_true'); ap.add_argument('--venv-root',type=Path,default=ROOT); ap.add_argument('--method-map',type=Path,default=ROOT/'environment/method_env_map.csv'); ap.add_argument('--output-map',type=Path,default=ROOT/'environment/method_env_map.csv'); ap.add_argument('--lock-root',type=Path,default=ROOT/'environment/locks'); ap.add_argument('--method',action='append')
    args=ap.parse_args(argv); rows=read_rows(args.method_map); selected=set(args.method or [])
    content_to_lock={}; out_rows=[]; report=[]; ok=True
    for row in rows:
        if selected and row['method'] not in selected: continue
        venv=args.venv_root/row['venv_name']; py=py_for(venv)
        if py is None:
            report.append({'method':row['method'],'status':'missing_environment'}); ok=False; out_rows.append(row); continue
        text=freeze(py); digest=hashlib.sha256(text.encode()).hexdigest(); lock_name=choose_lock(row,digest,text,args.lock_root,content_to_lock)
        lock_path=args.lock_root/lock_name; rel_lock=lock_path.relative_to(ROOT).as_posix() if str(lock_path).startswith(str(ROOT)) else lock_path.as_posix()
        status='match' if lock_path.exists() and lock_path.read_text(encoding='utf-8')==text else 'drift'
        if args.write:
            if lock_path.exists() and lock_path.read_text(encoding='utf-8') != text:
                report.append({'method':row['method'],'status':'lock_name_conflict','lock_file':rel_lock,'sha256':digest})
                ok=False; out_rows.append(row); continue
            atomic(lock_path,text); status='written'
        elif status!='match': ok=False
        out=dict(row); out['lock_file']=rel_lock; out['notes']=(row.get('notes','')+' lock_sha256='+digest).strip(); out_rows.append(out); report.append({'method':row['method'],'status':status,'lock_file':rel_lock,'sha256':digest})
    if args.write:
        with args.output_map.open('w',encoding='utf-8',newline='') as f: w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(out_rows)
    print(json.dumps({'ok':ok or args.allow_missing,'report':report},ensure_ascii=False,indent=2))
    return 0 if (ok or args.allow_missing or args.write) else 1
if __name__=='__main__': raise SystemExit(main(sys.argv[1:]))
