from __future__ import annotations
import csv, hashlib, json, os, tempfile
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]

METHOD_FIELDS = 'method,owner,family,primary_model,upstream_status,adapter_status,smoke_status,r4_status,recovery_status,execution_status,validity_status,quality_gate,officiality,evidence_status,primary_code,readme,notes'.split(',')
RUN_FIELDS = 'run_id,category,method_scope,model,protocol,variant,round,execution_status,validity_status,quality_gate,officiality,result_completeness,evidence_path,metadata_present,summary_present,superseded_by,notes'.split(',')
SCORE_FIELDS = 'score_id,run_id,method,model,variant,benchmark,protocol,split,task_count,pass_count,pass_rate,plus_pass_count,plus_pass_rate,metric_name,metric_value,result_completeness,validity_status,evidence_status,source_file,notes'.split(',')
ARTIFACT_FIELDS = 'artifact_id,run_id,method,artifact_type,storage_root,relative_locator,size_bytes,sha256,availability,committed_to_git,notes'.split(',')
SPLIT_FIELDS = 'split_id,dataset,protocol,role,path,task_count,sha256,seed,overlap_policy,source,notes'.split(',')
FORMAL_BANNED = {'pilot_keep80_official_all_20260727_174732','qwen25c3b_r4_baseline_evalhalf_20260723_193503','qwen25c3b_r4_baseline_evalhalf_recheck_20260726_181426'}

METHOD_CONFIG = [
('Flab-Pruner','\u5e38\u73c2\u8212','structured','Qwen2.5-Coder','vendored_submodule','qwen_adapter','completed','completed','completed','completed','under_review','fail','experimental_extension','complete','methods/flab_pruner/qwen_prune.py','methods/flab_pruner/README.md','Official structural mode and benchmark activation experimental mode are separated.'),
('LLM-Pruner','\u5e38\u73c2\u8212','structured','Qwen2.5-Coder / CodeLlama','vendored_submodule','qwen_adapter','completed','completed','completed','completed','under_review','fail','local_official_adapter','complete','methods/llm_pruner/qwen_prune.py','methods/llm_pruner/README.md','Local adapter evidence is complete; CodeLlama route is fallback.'),
('SliceGPT','\u5e38\u73c2\u8212','structured','Qwen2.5-Coder / CodeLlama','vendored_submodule','qwen_adapter','completed','partial','completed','partial','under_review','fail','local_official_adapter','partial','methods/slicegpt/qwen_prune.py','methods/slicegpt/README.md','Partial benchmark evidence keeps actual task counts.'),
('LaCo','\u5e38\u73c2\u8212','layer collapse','CodeLlama candidate','vendored_submodule','blocked','planned','blocked','not_applicable','blocked','diagnostic_only','not_applicable','not_run','not_applicable','','methods/laco/README.md','Upstream notebook route did not become a reproducible Stage 1 run.'),
('Magnitude','\u6f58\u9614','unstructured','Qwen2.5-Coder / OPT','vendored_submodule','ready','completed','not_applicable','not_applicable','completed','valid','pass','auxiliary_protocol','aggregate_only','','methods/magnitude/README.md','Auxiliary full evaluation is aggregate only and not directly comparable with r4_half.'),
('Wanda','\u6f58\u9614','unstructured','Qwen2.5-Coder / OPT','vendored_submodule','ready','completed','not_applicable','not_applicable','completed','valid','pass','auxiliary_protocol','aggregate_only','methods/wanda/qwen_prune.py','methods/wanda/README.md','Auxiliary full evaluation is aggregate only and not directly comparable with r4_half.'),
('DSnoT','\u6f58\u9614','unstructured','OPT','vendored_submodule','blocked_qwen','completed','not_applicable','not_applicable','partial','diagnostic_only','not_applicable','auxiliary_protocol','aggregate_only','','methods/dsnot/README.md','OPT PPL aggregate is recorded; Qwen adapter remains unsupported.'),
('OWL','\u6f58\u9614','unstructured','OPT','vendored_submodule','blocked_qwen','completed','not_applicable','not_applicable','partial','diagnostic_only','not_applicable','auxiliary_protocol','aggregate_only','','methods/owl/README.md','Process evidence is recorded; Qwen adapter remains unsupported.'),
('SparseGPT','\u674e\u957f\u9a8f','reconstruction','OPT / CodeLlama candidate','vendored_submodule','planned','planned','planned','not_applicable','planned','not_applicable','pending','not_run','not_applicable','','methods/sparsegpt/README.md','Candidate retained for the first-stage method scope.'),
('MaskLLM','\u674e\u957f\u9a8f','mask learning','candidate','vendored_submodule','planned','planned','planned','not_applicable','planned','not_applicable','pending','not_run','not_applicable','','methods/maskllm/README.md','Candidate retained without first-stage run evidence.'),
('Pruner-Zero','\u674e\u957f\u9a8f','search','candidate','vendored_submodule','planned','planned','planned','not_applicable','planned','not_applicable','pending','not_run','not_applicable','','methods/pruner_zero/README.md','Candidate retained without first-stage run evidence.'),
('FLAP','\u674e\u957f\u9a8f','structured','candidate','vendored_submodule','planned','planned','planned','not_applicable','planned','not_applicable','pending','not_run','not_applicable','','methods/flap/README.md','Candidate retained without first-stage run evidence.'),
]

def sha256_file(path: Path) -> str:
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))

def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def read_csv(path: Path) -> list[dict[str,str]]:
    with path.open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def csv_text(fields: list[str], rows: Iterable[dict]) -> str:
    import io
    buf=io.StringIO(); w=csv.DictWriter(buf, fieldnames=fields, lineterminator='\n'); w.writeheader(); w.writerows(rows); return buf.getvalue()

def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name+'.', dir=str(path.parent), text=True)
    with os.fdopen(fd, 'w', encoding='utf-8', newline='') as handle: handle.write(text)
    Path(tmp).replace(path)

def display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()

def write_or_check(path: Path, text: str, write: bool, check: bool) -> bool:
    if write:
        old = path.read_text(encoding='utf-8') if path.exists() else None
        atomic_write(path, text)
        print(f'wrote {display_path(path)} bytes={len(text)} changed={old != text}')
        return True
    current = path.read_text(encoding='utf-8') if path.exists() else ''
    ok = current == text
    print(f'check {display_path(path)} ok={ok}')
    return ok

def infer_method(run_id: str) -> str:
    low=run_id.lower(); found=[]
    for token,name in [('flab','Flab-Pruner'),('llmpruner','LLM-Pruner'),('llm_pruner','LLM-Pruner'),('slicegpt','SliceGPT'),('magnitude','Magnitude'),('wanda','Wanda'),('dsnot','DSnoT'),('owl','OWL'),('sparsegpt','SparseGPT'),('maskllm','MaskLLM'),('prunerzero','Pruner-Zero'),('pruner_zero','Pruner-Zero'),('flap','FLAP')]:
        if token in low and name not in found: found.append(name)
    if 'pilot_keep80_official_all' in low: return 'Flab-Pruner;LLM-Pruner;SliceGPT'
    if low.startswith('qwen') or 'baseline' in low or 'codellama7b_4bit' in low: return 'baseline'
    return ';'.join(found) if found else 'shared'

def infer_model(run_id: str) -> str:
    low=run_id.lower()
    if 'qwen25c15b' in low or 'qwen15b' in low: return 'Qwen/Qwen2.5-Coder-1.5B-Instruct'
    if 'qwen25c3b' in low or 'qwen3b' in low: return 'Qwen/Qwen2.5-Coder-3B-Instruct'
    if 'codellama' in low: return 'codellama/CodeLlama-7b-hf'
    if 'tiny_llama' in low or 'tinyllama' in low: return 'TinyLlama/TinyLlama-1.1B'
    if 'opt125m' in low: return 'facebook/opt-125m'
    return 'unknown'

def infer_variant(path: Path, run_id: str) -> str:
    s='/'.join(path.relative_to(ROOT).parts).lower()
    r=run_id.lower()
    checks=[
        ('benchguided_keep80_lora_merged','benchmark_guided_keep80_lora_merged'),('benchmark_guided_keep80_lora_merged','benchmark_guided_keep80_lora_merged'),
        ('default_keep80_lora_merged','default_keep80_lora_merged'),('benchguided_keep80_lora','benchmark_guided_keep80_lora'),('benchmark_guided_keep80_lora','benchmark_guided_keep80_lora'),
        ('default_keep80_lora','default_keep80_lora'),('benchguided_keep80','benchmark_guided_keep80'),('benchmark_guided_keep80','benchmark_guided_keep80'),
        ('default_keep80','default_keep80'),('sliced_model','sliced_model'),('layerdrop_keep80','layerdrop_keep80'),('official_keep80','official_keep80'),('keep80','official_keep80'),('baseline','baseline')]
    if 'slicegpt' in r and 'benchguided' in r:
        return 'benchmark_guided_sliced_model'
    if 'slicegpt' in r and ('official_keep80' in r or 'keep80' in r):
        return 'sliced_model'
    for token,value in checks:
        if token in s or token in r: return value
    if 'qwen25c3b_official_evalhalf' in r or 'qwen25c15b_official_evalhalf' in r: return 'baseline'
    return 'unknown'

def build_run_rows() -> list[dict[str,str]]:
    rows=[]
    for d in sorted((ROOT/'results/evidence').glob('*/*'), key=lambda p:(p.parent.name,p.name)):
        if not d.is_dir(): continue
        cat=d.parent.name; rid=d.name; low=rid.lower()
        proto='r4_half' if cat=='r4_half' else ('smoke' if cat=='smoke' else cat.rstrip('s'))
        comp='pilot' if rid=='pilot_keep80_official_all_20260727_174732' else ('not_applicable' if cat in {'diagnostics','infrastructure','superseded'} and not list(d.rglob('score_summary.json')) else 'complete')
        if 'partial' in low or rid in {'slicegpt_codellama7b_r4_benchguided_evalhalf_20260726_053225','slicegpt_codellama7b_r4_offload_probe_20260726_044311'}: comp='partial'
        val='invalid' if cat=='superseded' else ('diagnostic_only' if cat in {'diagnostics','infrastructure'} else 'valid')
        sup='qwen25c3b_official_evalhalf_20260727_135521' if rid in {'qwen25c3b_r4_baseline_evalhalf_20260723_193503','qwen25c3b_r4_baseline_evalhalf_recheck_20260726_181426'} else ('see registry notes' if cat=='superseded' else '')
        method=infer_method(rid)
        rows.append({'run_id':rid,'category':cat,'method_scope':method,'model':infer_model(rid),'protocol':proto,'variant':infer_variant(d,rid),'round':'R4' if cat=='r4_half' else ('smoke' if cat=='smoke' else 'audit'),'execution_status':'superseded' if cat=='superseded' else ('partial' if comp=='partial' else 'completed'),'validity_status':val,'quality_gate':'fail' if method in {'Flab-Pruner','LLM-Pruner','SliceGPT'} and cat=='r4_half' else ('not_applicable' if val!='valid' else 'pass'),'officiality':'fallback_non_official' if 'codellama' in low and 'llmpruner' in low else ('experimental_extension' if 'benchguided' in low else 'local_official_adapter'),'result_completeness':comp,'evidence_path':d.relative_to(ROOT).as_posix(),'metadata_present':str(any(d.glob('metadata.*'))).lower(),'summary_present':str(bool(list(d.rglob('score_summary.json')) or (d/'summary.json').exists())).lower(),'superseded_by':sup,'notes':'5-task pilot excluded from formal table' if comp=='pilot' else ''})
    return rows

def benchmark_from_path(path: Path, data: dict) -> str:
    b=(data.get('benchmark') or '').replace('mbpp_evalplus','mbpp')
    if b: return b
    s=path.as_posix().lower()
    for k in ['humaneval','livecodebench','mbpp','wikitext2']:
        if k in s: return k
    return 'unknown'

def stable_score_id(row: dict) -> str:
    raw='|'.join(row[k] for k in ['run_id','method','variant','benchmark','protocol','split','metric_name','source_file'])
    return hashlib.sha1(raw.encode()).hexdigest()[:16]

def build_score_rows(run_rows: list[dict[str,str]]|None=None) -> list[dict[str,str]]:
    if run_rows is None: run_rows=build_run_rows()
    by={r['run_id']:r for r in run_rows}
    rows=[]
    for p in sorted((ROOT/'results/evidence').rglob('score_summary.json')):
        parts=p.relative_to(ROOT).parts; rid=parts[3]; rr=by[rid]; data=read_json(p)
        task=str(data.get('task_count','')); pc=data.get('base_pass_count',data.get('pass_count','')); pr=data.get('base_pass_rate',data.get('pass_rate',''))
        comp='pilot' if task=='5' else ('partial' if task=='82' and 'slicegpt_codellama7b_r4_benchguided' in rid else rr['result_completeness'])
        variant=infer_variant(p,rid)
        val=rr['validity_status'] if variant!='unknown' else ('under_review' if rr['validity_status']=='valid' else rr['validity_status'])
        row={'score_id':'','run_id':rid,'method':rr['method_scope'],'model':rr['model'],'variant':variant,'benchmark':benchmark_from_path(p,data),'protocol':rr['protocol'],'split':'eval','task_count':task,'pass_count':str(pc) if pc!='' else '','pass_rate':str(pr) if pr!='' else '','plus_pass_count':'','plus_pass_rate':'','metric_name':'pass_rate','metric_value':str(pr) if pr!='' else '','result_completeness':comp,'validity_status':val,'evidence_status':'present','source_file':p.relative_to(ROOT).as_posix(),'notes':'variant inferred from score path' if variant!='unknown' else 'variant could not be inferred'}
        row['score_id']=stable_score_id(row); rows.append(row)
    # Auxiliary aggregate tables are validated by build_auxiliary_comparison.py.
    # They are not emitted into results/status/scores.csv because that table
    # is intentionally one-to-one traceable to evidence run directories.
    seen=set(); unique=[]
    for r in rows:
        if r['score_id'] in seen:
            r['score_id']=r['score_id']+'_'+str(len(seen))
        seen.add(r['score_id']); unique.append(r)
    return unique

def build_formal_rows(run_rows: list[dict[str,str]]|None=None, score_rows: list[dict[str,str]]|None=None) -> list[dict[str,str]]:
    if run_rows is None: run_rows=build_run_rows()
    if score_rows is None: score_rows=build_score_rows(run_rows)
    runs={r['run_id']:r for r in run_rows}
    out=[]; seen=set()
    for r in score_rows:
        run=runs.get(r['run_id'])
        if not run: continue
        if r['protocol']!='r4_half': continue
        if r['run_id'] in FORMAL_BANNED: continue
        if r['result_completeness'] not in {'complete','partial'}: continue
        if r['validity_status']=='invalid' or run['validity_status']=='invalid': continue
        if run['superseded_by'] or run['execution_status']=='superseded': continue
        key=(r['method'],r['model'],r['variant'],r['benchmark'],r['protocol'],r['split'],r['metric_name'],r['run_id'])
        if key in seen: continue
        seen.add(key); out.append(r)
    return out

def build_method_rows(run_rows: list[dict[str,str]]|None=None, score_rows: list[dict[str,str]]|None=None) -> list[dict[str,str]]:
    if run_rows is None:
        run_rows = build_run_rows()
    evidence_by_method: dict[str, list[dict[str,str]]] = {}
    for run in run_rows:
        for method in run['method_scope'].split(';'):
            evidence_by_method.setdefault(method, []).append(run)
    rows=[]
    for config in METHOD_CONFIG:
        row=dict(zip(METHOD_FIELDS,config))
        runs=evidence_by_method.get(row['method'], [])
        if not runs:
            rows.append(row)
            continue
        valid_complete=[r for r in runs if r['execution_status']=='completed' and r['validity_status']=='valid' and r['result_completeness']=='complete']
        partial=[r for r in runs if r['result_completeness']=='partial' or r['execution_status']=='partial']
        blocked=[r for r in runs if r['validity_status']=='diagnostic_only' or 'blocked' in row['adapter_status']]
        if valid_complete:
            row['execution_status']='completed'
            row['evidence_status']='complete'
            row['validity_status']='valid' if row['quality_gate']=='pass' else row['validity_status']
            if any(r['protocol']=='r4_half' for r in valid_complete):
                row['r4_status']='completed'
        elif partial:
            row['execution_status']='partial'
            row['evidence_status']='partial'
            row['r4_status']='partial'
        elif blocked:
            row['execution_status']='blocked'
            row['evidence_status']='diagnostic_only'
        else:
            row['execution_status']='planned'
            row['evidence_status']='not_applicable'
        rows.append(row)
    return rows

def build_split_rows() -> list[dict[str,str]]:
    rows=[]
    for manifest in sorted((ROOT/'data/benchmarks').rglob('manifest.json')):
        data=read_json(manifest); protocol=manifest.relative_to(ROOT).parts[2]; dataset=manifest.parent.name
        for role,path_key in [('guide','guide_path'),('eval','eval_path'),('heldout_eval','heldout_eval_path')]:
            if path_key not in data: continue
            path=ROOT/data[path_key]
            if protocol=='auxiliary_full_eval' and role=='guide': policy='guide_subset_of_full_eval'
            elif protocol=='auxiliary_full_eval' and role=='heldout_eval': policy='guide_disjoint_from_heldout'
            elif protocol=='auxiliary_full_eval': policy='full_eval_contains_guide'
            else: policy='guide_eval_disjoint'
            rows.append({'split_id':f'{protocol}_{dataset}_{role}','dataset':dataset,'protocol':protocol,'role':role,'path':path.relative_to(ROOT).as_posix(),'task_count':str(len(read_jsonl(path))),'sha256':sha256_file(path),'seed':'0','overlap_policy':policy,'source':'manifest','notes':data.get('split_policy',data.get('selection',''))})
    return rows
