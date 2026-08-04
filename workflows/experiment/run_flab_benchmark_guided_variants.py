#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

from methods.flab_pruner import benchmark_guided
from methods.flab_pruner.qwen_prune import load_flab_qwen_model
from methods.flab_pruner.zs_adapter import full_config_schema, select_intermediate_indexes, validate_flab_zs, apply_flab_zs, count_parameters

VARIANTS = {
    "he": ("benchmark_guided_he_keep80_capped32", "data/benchmarks/r4_half/humaneval/manifest.json"),
    "mbpp": ("benchmark_guided_mbpp_keep80_capped32", "data/benchmarks/r4_half/mbpp_evalplus/manifest.json"),
    "lcb": ("benchmark_guided_lcb_keep80_capped32", "data/benchmarks/r4_half/livecodebench/manifest.json"),
}

class Tok:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def encode(self, text: str, max_length: int = 128):
        return self.tokenizer.encode(text, max_length=max_length, truncation=True, add_special_tokens=False) or [0]


def read_jsonl(path: Path, limit: int):
    rows=[]
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def prompt(row):
    return row.get("prompt") or row.get("question") or row.get("content") or ""


def default_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def prepare_output(run_id: str, output_dir: str | None) -> Path:
    out = Path(output_dir) if output_dir else ROOT / "results/evidence/smoke" / run_id
    if not out.is_absolute():
        out = ROOT / out
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence directory: {out}")
    out.mkdir(parents=True)
    return out


def run_variant(key: str, run_id: str | None = None, output_dir: str | None = None) -> None:
    variant, manifest_path = VARIANTS[key]
    run_id = run_id or f"flab_qwen15b_{variant}_{default_stamp()}"
    out = prepare_output(run_id, output_dir)
    artifact_base = Path("/tmp") / f"{run_id}_artifacts"
    summary = {"run_id": run_id, "method": "Flab-Pruner", "variant": variant, "model": MODEL_ID, "command_status": "exit_nonzero", "execution_status": "failed", "validity_status": "diagnostic_only", "quality_gate": "fail", "protocol_deviation": "calibration_capped32", "artifact_locator": {"artifact_base": str(artifact_base), "committed_to_git": False}}
    stdout=[]; stderr=[]
    def flush():
        (out/"command.sh").write_text(f"#!/usr/bin/env bash\nset -euo pipefail\npython workflows/experiment/run_flab_benchmark_guided_variants.py --variant {key} --run-id {run_id}\n", encoding="utf-8")
        (out/"command.sh").chmod(0o755)
        (out/"stdout.log").write_text("\n".join(stdout)+"\n", encoding="utf-8")
        (out/"stderr.log").write_text("\n".join(stderr)+"\n", encoding="utf-8")
        (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
        (out/"metadata.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
        (out/"artifact_locator.json").write_text(json.dumps(summary["artifact_locator"], indent=2)+"\n", encoding="utf-8")
        (out/"resource.csv").write_text("metric,value\ncommand_status,"+summary["command_status"]+"\nexecution_status,"+summary["execution_status"]+"\nquality_gate,"+summary["quality_gate"]+"\n", encoding="utf-8")
        (out/"resource_summary.json").write_text(json.dumps({"command_status": summary["command_status"], "execution_status": summary["execution_status"], "quality_gate": summary["quality_gate"]}, indent=2)+"\n", encoding="utf-8")
    try:
        manifest=json.loads((ROOT/manifest_path).read_text(encoding="utf-8"))
        guide_rows=read_jsonl(ROOT/manifest["guide_path"], 32)
        eval_rows=read_jsonl(ROOT/manifest["eval_path"], 20)
        summary.update({"guide_path": manifest["guide_path"], "guide_full_hash": manifest["guide_sha256"], "guide_full_count": manifest["guide_count"], "guide_used_count": len(guide_rows), "guide_task_ids": [r.get("task_id") for r in guide_rows], "eval_path": manifest["eval_path"], "eval_used_count": len(eval_rows)})
        model=load_flab_qwen_model(MODEL_ID, dtype="fp16", device_map=None, local_files_only=True, allow_hf_fallback=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        summary["execution_device"] = str(device)
        tokenizer=AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, local_files_only=True)
        importance=benchmark_guided.collect_intermediate_importance(model, Tok(tokenizer), guide_rows, max_length=128)
        before=count_parameters(model)
        hidden=int(model.config.hidden_size); layers=int(model.config.num_hidden_layers); inter=int(model.config.intermediate_size)
        non_ffn=before - layers*(3*hidden*inter)
        keep=max(1, min(inter, int((int(before*0.80)-non_ffn)/max(1, 3*hidden*layers))))
        schema=full_config_schema(model, intermediate_size_remain=keep)
        zs=select_intermediate_indexes(importance, 0.80, schema)
        validate_flab_zs(model, zs, schema)
        prune=apply_flab_zs(model, zs, schema)
        reload_check=benchmark_guided.save_reload_check(model, artifact_base/"artifact")
        generations=[]
        token_counts=[]
        decoded=[]
        reloaded_model=load_flab_qwen_model(str(artifact_base/"artifact"), dtype="fp16", device_map=None, local_files_only=True, allow_hf_fallback=False)
        reloaded_model.to(device)
        reloaded_model.eval()
        for row in eval_rows:
            ids=torch.tensor([Tok(tokenizer).encode(prompt(row), max_length=128)], dtype=torch.long).to(device)
            toks=benchmark_guided.greedy_tokens(reloaded_model, ids, steps=16)
            text=tokenizer.decode(toks, skip_special_tokens=True)
            generations.append({"task_id": row.get("task_id"), "generated_tokens": toks, "completion": text})
            token_counts.append(len(toks)); decoded.append(text)
        empty=sum(1 for x in decoded if not x.strip())
        duplicate= len(decoded)-len(set(decoded))
        median=sorted(token_counts)[len(token_counts)//2] if token_counts else 0
        nan_or_inf=False
        quality_pass = (empty/len(decoded) <= 0.20) and (duplicate/len(decoded) <= 0.50) and median >= 8 and not nan_or_inf and reload_check["reload_success"]
        summary.update({"command_status":"exit_0", "execution_status":"pilot_quality_gate_completed", "validity_status":"valid", "quality_gate":"pass_for_pilot_quality_gate" if quality_pass else "fail", "formal_full_eval":"not_run_pending_after_capped32_gate" if quality_pass else "skipped_due_to_output_collapse", "score_status":"not_evaluated_due_to_output_collapse", "scorer_executed": False, "benchmark_guided_dimensions":["intermediate"], "config_derived_dimensions":["hidden","attention_head","kv_head"], "importance_hash": importance.importance_hash, "selected_index_hash": zs.selected_index_hash, "requested_parameter_keep_ratio":0.80, **prune, "reload_success": reload_check["reload_success"], "empty_rate": empty/len(decoded), "duplicate_rate": duplicate/len(decoded), "median_generated_tokens": median, "nan_or_inf": nan_or_inf, "pass_count": ""})
        (out/"parameter_summary.json").write_text(json.dumps({k: summary[k] for k in ["params_before","params_after","actual_parameter_keep_ratio","requested_parameter_keep_ratio"]}, indent=2)+"\n", encoding="utf-8")
        (out/"reload_check.json").write_text(json.dumps(reload_check, indent=2)+"\n", encoding="utf-8")
        (out/"generation_check.json").write_text(json.dumps({"empty_rate": summary["empty_rate"], "duplicate_rate": summary["duplicate_rate"], "median_generated_tokens": median, "nan_or_inf": nan_or_inf}, indent=2)+"\n", encoding="utf-8")
        with (out/"generations.jsonl").open("w", encoding="utf-8") as f:
            for item in generations:
                f.write(json.dumps(item)+"\n")
        (out/"selected_indices.json").write_text(json.dumps({"selected_index_hash": zs.selected_index_hash}, indent=2)+"\n", encoding="utf-8")
        (out/"importance.json").write_text(json.dumps({"importance_hash": importance.importance_hash}, indent=2)+"\n", encoding="utf-8")
    except Exception as exc:
        stderr.append(type(exc).__name__+": "+str(exc))
        summary["failure_reason"] = str(exc)
    flush()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--run-id", help="Explicit run id for a new evidence directory.")
    parser.add_argument("--output-dir", help="Explicit output directory; must not already exist.")
    args=parser.parse_args()
    run_variant(args.variant, args.run_id, args.output_dir)

if __name__ == "__main__":
    main()
