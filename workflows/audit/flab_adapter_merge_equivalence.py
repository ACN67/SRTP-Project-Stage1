#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parents[2]
FLAB_ROOT=ROOT/'results/evidence/r4_half/flabpruner_qwen25c15b_official_keep80_20260730_015031'
AUDIT_ROOT=sorted((ROOT/'results/evidence').glob('existing_checkpoint_stage_audit_*'))[-1]
FIXED=ROOT/'data/audit/fixed_smoke_20.json'
OUT=ROOT/'reports/flab_adapter_merge_equivalence.json'


def load_rows(path): return json.loads(path.read_text(encoding='utf-8'))

def load_generations(stage):
    rows={}
    p=AUDIT_ROOT/stage/'audit_generations.jsonl'
    for line in p.open(encoding='utf-8'):
        d=json.loads(line); rows[d['sample_id']]=d
    return rows

def load_model(model_path, adapter=None):
    tok=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16, local_files_only=True, device_map='cuda:0')
    if adapter:
        model=PeftModel.from_pretrained(model, adapter)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    model.eval()
    return model,tok

def collect_logits(model,tok,tasks):
    out={}
    dev=next(model.parameters()).device
    for item in tasks:
        prompt=item.get('model_prompt') or item['prompt']
        inputs=tok(prompt, return_tensors='pt').to(dev)
        with torch.no_grad():
            logits=model(**inputs).logits[0,-1].detach().float().cpu()
        out[item['sample_id']]=logits
    return out

def main():
    f2=load_generations('F2_pruned_lora_adapter')
    f3=load_generations('F3_merged')
    tasks=list(f2.values())
    model,tok=load_model(str(FLAB_ROOT/'flabpruner_keep80_v2/pruned_model'), str(FLAB_ROOT/'flabpruner_keep80_lora_v2/lora_adapter'))
    adapter_logits=collect_logits(model,tok,tasks)
    del model; torch.cuda.empty_cache()
    model,tok=load_model(str(FLAB_ROOT/'flabpruner_keep80_merged_v2'))
    merged_logits=collect_logits(model,tok,tasks)
    diffs=[]; agreements=[]; exact=[]
    per=[]
    for sid in adapter_logits:
        a=adapter_logits[sid]; b=merged_logits[sid]
        diff=(a-b).abs()
        row={
            'sample_id':sid,
            'max_abs_logit_diff':float(diff.max()),
            'mean_abs_logit_diff':float(diff.mean()),
            'top1_agreement':int(a.argmax().item()==b.argmax().item()),
            'greedy_completion_exact_match':f2[sid]['raw_completion']==f3[sid]['raw_completion'],
        }
        per.append(row); diffs.append(row['max_abs_logit_diff']); agreements.append(row['top1_agreement']); exact.append(row['greedy_completion_exact_match'])
    summary={
        'status':'success','audit_root':str(AUDIT_ROOT.relative_to(ROOT)),
        'adapter_model':str((FLAB_ROOT/'flabpruner_keep80_v2/pruned_model').relative_to(ROOT)),
        'adapter':str((FLAB_ROOT/'flabpruner_keep80_lora_v2/lora_adapter').relative_to(ROOT)),
        'merged_model':str((FLAB_ROOT/'flabpruner_keep80_merged_v2').relative_to(ROOT)),
        'sample_count':len(per),
        'max_abs_logit_diff_max':max(diffs),
        'max_abs_logit_diff_mean':sum(diffs)/len(diffs),
        'mean_abs_logit_diff_mean':sum(r['mean_abs_logit_diff'] for r in per)/len(per),
        'top1_token_agreement':sum(agreements)/len(agreements),
        'greedy_completion_exact_match':sum(1 for x in exact if x)/len(exact),
        'per_sample':per,
    }
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
if __name__=='__main__': main()
