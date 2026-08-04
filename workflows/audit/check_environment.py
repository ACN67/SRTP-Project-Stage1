#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.util, json, os, shutil, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def run(command):
    try:
        p=subprocess.run(command,text=True,capture_output=True,timeout=30); return {'ok':p.returncode==0,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()}
    except Exception as e: return {'ok':False,'stdout':'','stderr':repr(e)}
def module_status(names): return {n: importlib.util.find_spec(n) is not None for n in names}
def torch_status():
    try:
        import torch
        return {'version':torch.__version__,'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available()}
    except Exception as e: return {'error':repr(e),'cuda_available':False}
def main():
    parser=argparse.ArgumentParser(description='Check local environment and write to an evidence output directory.')
    parser.add_argument('--output-dir', type=Path, default=None)
    args=parser.parse_args()
    out_dir=args.output_dir or (Path(os.environ['RUN_DIR']) if os.environ.get('RUN_DIR') else None)
    if out_dir is None:
        print(json.dumps({'ok':False,'error':'--output-dir or RUN_DIR is required'},indent=2)); return 2
    out_dir=out_dir if out_dir.is_absolute() else ROOT/out_dir
    mods=['torch','transformers','accelerate','datasets','evaluate','safetensors','sentencepiece','peft','yaml','jsonlines','psutil']
    report={'python':sys.version,'executable':sys.executable,'cwd':str(Path.cwd()),'tools':{n:shutil.which(n) for n in ['git','git-lfs','uv','docker','nvidia-smi']},'commands':{'git':run(['git','--version'])},'modules':module_status(mods),'torch':torch_status()}
    out_dir.mkdir(parents=True,exist_ok=True); (out_dir/'environment.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+chr(10),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2)); return 0 if all(report['modules'].values()) else 1
if __name__=='__main__': raise SystemExit(main())
