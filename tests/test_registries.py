import csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
METHODS = ["Flab-Pruner","LLM-Pruner","SliceGPT","LaCo","Magnitude","Wanda","DSnoT","OWL","SparseGPT","MaskLLM","Pruner-Zero","FLAP"]
METHOD_SCHEMA = "method,owner,family,primary_model,upstream_status,adapter_status,smoke_status,r4_status,recovery_status,execution_status,validity_status,quality_gate,officiality,evidence_status,primary_code,readme,notes".split(",")
RUN_SCHEMA = "run_id,category,method_scope,model,protocol,variant,round,execution_status,validity_status,quality_gate,officiality,result_completeness,evidence_path,metadata_present,summary_present,superseded_by,notes".split(",")
SCORE_SCHEMA = "score_id,run_id,method,model,variant,benchmark,protocol,split,task_count,pass_count,pass_rate,plus_pass_count,plus_pass_rate,metric_name,metric_value,result_completeness,validity_status,evidence_status,source_file,notes".split(",")
ART_SCHEMA = "artifact_id,run_id,method,artifact_type,storage_root,relative_locator,size_bytes,sha256,availability,committed_to_git,notes".split(",")
SPLIT_SCHEMA = "split_id,dataset,protocol,role,path,task_count,sha256,seed,overlap_policy,source,notes".split(",")

def rows(rel):
    with (ROOT/rel).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def header(rel):
    with (ROOT/rel).open(encoding="utf-8-sig", newline="") as f:
        return next(csv.reader(f))

def test_registry_schemas_and_methods():
    assert header("results/status/methods.csv") == METHOD_SCHEMA
    assert header("results/status/runs.csv") == RUN_SCHEMA
    assert header("results/status/scores.csv") == SCORE_SCHEMA
    assert header("results/status/artifacts.csv") == ART_SCHEMA
    assert header("results/status/data_splits.csv") == SPLIT_SCHEMA
    method_rows = rows("results/status/methods.csv")
    assert len(method_rows) == 12
    assert [r["method"] for r in method_rows] == METHODS

def test_runs_match_evidence_directories_one_to_one():
    evidence={p.relative_to(ROOT).as_posix() for p in (ROOT/"results/evidence").glob("*/*") if p.is_dir()}
    run_rows=rows("results/status/runs.csv")
    assert len(run_rows) == len(evidence) >= 77
    assert {r["evidence_path"] for r in run_rows} == evidence
    assert {r["category"] for r in run_rows} <= {"diagnostics","infrastructure","r4_half","smoke","superseded"}
    assert {r["result_completeness"] for r in run_rows} <= {"complete","partial","pilot","aggregate_only","not_applicable"}

def test_scores_trace_to_runs_and_files():
    run_ids={r["run_id"] for r in rows("results/status/runs.csv")}
    for r in rows("results/status/scores.csv"):
        assert r["run_id"] in run_ids
        assert (ROOT/r["source_file"]).is_file(), r["source_file"]
        assert r["result_completeness"] in {"complete","partial","pilot","aggregate_only","not_applicable"}
        assert r["validity_status"] in {"valid","under_review","diagnostic_only","invalid","not_applicable"}

def test_formal_excludes_pilot_and_keeps_partial():
    formal=rows("results/formal/r4_half/scores.csv")
    assert formal
    assert all(r["protocol"] == "r4_half" for r in formal)
    assert all(r["result_completeness"] != "pilot" for r in formal)
    assert not any(r["run_id"] == "pilot_keep80_official_all_20260727_174732" for r in formal)
    scores=rows("results/status/scores.csv")
    assert any(r["run_id"] == "pilot_keep80_official_all_20260727_174732" and r["result_completeness"] == "pilot" for r in scores)
    assert any(r["result_completeness"] == "partial" and r["task_count"] == "82" for r in scores)
    old=[r for r in rows("results/status/runs.csv") if r["run_id"] == "qwen25c3b_r4_baseline_evalhalf_20260723_193503"]
    assert old and old[0]["superseded_by"] == "qwen25c3b_official_evalhalf_20260727_135521"
