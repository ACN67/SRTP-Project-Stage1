#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
STAMP = "20260804_135032"

from methods.flab_pruner import benchmark_guided
from methods.flab_pruner.qwen_prune import load_flab_qwen_model


def run_tiny() -> None:
    out = ROOT / "results/evidence/smoke" / f"flab_benchmark_guided_tiny_{STAMP}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "command.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\npython workflows/experiment/run_flab_benchmark_guided.py --job tiny\n", encoding="utf-8")
    (out / "command.sh").chmod(0o755)
    artifact_base = Path("/tmp") / f"flab_benchmark_guided_tiny_artifacts_{STAMP}"
    result = benchmark_guided.run_tiny_pair(out, target_parameter_keep_ratio=0.80, artifact_base_dir=artifact_base)
    (out / "artifact_locator.json").write_text(json.dumps({"artifact_base": str(artifact_base), "committed_to_git": False}, indent=2) + "\n", encoding="utf-8")
    (out / "stdout.log").write_text(json.dumps(result["summary"], indent=2) + "\n", encoding="utf-8")
    (out / "stderr.log").write_text("", encoding="utf-8")
    (out / "resource_summary.json").write_text(json.dumps({"command_status": "exit_0", "execution_status": "tiny_smoke_completed"}, indent=2) + "\n", encoding="utf-8")



def run_qwen_smoke() -> None:
    out = ROOT / "results/evidence/smoke" / f"flab_qwen15b_benchmark_guided_smoke_{STAMP}"
    out.mkdir(parents=True, exist_ok=True)
    command = "python workflows/experiment/run_flab_benchmark_guided.py --job qwen15b_smoke"
    (out / "command.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n", encoding="utf-8")
    (out / "command.sh").chmod(0o755)
    stdout = []
    stderr = []
    model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    artifact_base = Path("/tmp") / f"flab_qwen15b_benchmark_guided_artifacts_{STAMP}"
    summary = {
        "run_id": out.name,
        "method": "Flab-Pruner",
        "command_status": "exit_nonzero",
        "execution_status": "failed",
        "validity_status": "diagnostic_only",
        "quality_gate": "not_applicable",
        "target_smoke_closed": False,
        "model": model_id,
        "standard_attempt": "vendored Flab Qwen2ForCausalLM local_files_only load by model id",
        "fallback_attempt": "local Hugging Face snapshot discovery under user cache",
        "artifact_locator": {"artifact_base": str(artifact_base), "committed_to_git": False},
    }

    def flush() -> None:
        (out / "stdout.log").write_text("\n".join(stdout) + "\n", encoding="utf-8")
        (out / "stderr.log").write_text("\n".join(stderr) + "\n", encoding="utf-8")
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (out / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (out / "artifact_locator.json").write_text(json.dumps(summary["artifact_locator"], indent=2) + "\n", encoding="utf-8")
        (out / "resource.csv").write_text("metric,value\ncommand_status," + summary["command_status"] + "\nexecution_status," + summary["execution_status"] + "\n", encoding="utf-8")
        (out / "resource_summary.json").write_text(json.dumps({"command_status": summary["command_status"], "execution_status": summary["execution_status"]}, indent=2) + "\n", encoding="utf-8")

    try:
        import torch
        from transformers import AutoTokenizer
        from methods.flab_pruner.zs_adapter import full_config_schema, select_intermediate_indexes, validate_flab_zs, apply_flab_zs, count_parameters

        stdout.append("standard_attempt: load vendored Flab model local_files_only=True")
        try:
            model = load_flab_qwen_model(model_id, dtype="fp16", device_map=None, local_files_only=True, allow_hf_fallback=False)
            load_source = model_id
        except Exception as exc:
            stderr.append(type(exc).__name__ + ": " + str(exc))
            matches = sorted(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots/*/config.json"))
            summary["local_snapshot_candidates"] = [str(p.parent) for p in matches]
            if not matches:
                summary["first_failing_function"] = "load_flab_qwen_model"
                summary["first_failing_source"] = "methods/flab_pruner/qwen_prune.py"
                summary["failure_reason"] = "Qwen2.5-Coder-1.5B local snapshot is not present; ordinary HF fallback is forbidden for benchmark-guided pruning."
                flush(); return
            stdout.append("fallback_attempt: load discovered snapshot " + str(matches[0].parent))
            model = load_flab_qwen_model(str(matches[0].parent), dtype="fp16", device_map=None, local_files_only=True, allow_hf_fallback=False)
            load_source = str(matches[0].parent)
        model.eval()
        summary["vendored_flab_model_loaded"] = True
        summary["model_load_source"] = load_source
        summary["params_before"] = count_parameters(model)
        flush()

        tokenizer = AutoTokenizer.from_pretrained(load_source, trust_remote_code=True, local_files_only=True)
        class Tok:
            def encode(self, text: str, max_length: int = 64):
                return tokenizer.encode(text, max_length=max_length, truncation=True, add_special_tokens=False) or [0]
        rows=[]
        with (ROOT / "data/benchmarks/r4_half/humaneval/guide.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
                if len(rows) == 4:
                    break
        importance = benchmark_guided.collect_intermediate_importance(model, Tok(), rows, max_length=64)
        hidden = int(model.config.hidden_size)
        layers = int(model.config.num_hidden_layers)
        inter = int(model.config.intermediate_size)
        before = count_parameters(model)
        ffn_before = layers * (3 * hidden * inter)
        non_ffn = before - ffn_before
        target_after = int(before * 0.80)
        keep = max(1, min(inter, int((target_after - non_ffn) / max(1, 3 * hidden * layers))))
        schema = full_config_schema(model, intermediate_size_remain=keep)
        zs = select_intermediate_indexes(importance, 0.80, schema)
        validate_flab_zs(model, zs, schema)
        prune = apply_flab_zs(model, zs, schema)
        summary.update({
            "real_model_forward_called": True,
            "tensor_activation_collected": True,
            "benchmark_importance_computed": True,
            "guide_hash": importance.guide_hash,
            "importance_hash": importance.importance_hash,
            "selected_index_hash": zs.selected_index_hash,
            "benchmark_guided_dimensions": zs.benchmark_guided_dimensions,
            "config_derived_dimensions": zs.config_derived_dimensions,
            "requested_parameter_keep_ratio": 0.80,
            "intermediate_size_before": inter,
            "intermediate_size_after": keep,
            **prune,
        })
        (out / "importance.json").write_text(json.dumps({"guide_hash": importance.guide_hash, "importance_hash": importance.importance_hash}, indent=2) + "\n", encoding="utf-8")
        (out / "selected_indices.json").write_text(json.dumps({"selected_index_hash": zs.selected_index_hash, "intermediate_indexes": [x.tolist() for x in zs.intermediate_indexes]}, indent=2) + "\n", encoding="utf-8")
        (out / "parameter_summary.json").write_text(json.dumps({k: summary[k] for k in ["params_before", "params_after", "requested_parameter_keep_ratio", "actual_parameter_keep_ratio", "intermediate_size_before", "intermediate_size_after"]}, indent=2) + "\n", encoding="utf-8")
        flush()

        reload_check = benchmark_guided.save_reload_check(model, artifact_base / "artifact")
        generation_ok = reload_check["nonempty_generation_after_reload"] and reload_check["generated_token_count"] >= 4
        summary.update({
            "artifact_saved": True,
            "artifact_reloaded": reload_check["reload_success"],
            "forward_after_reload": reload_check["forward_after_reload"],
            "nonempty_generation_after_reload": reload_check["nonempty_generation_after_reload"],
            "generation_check": reload_check,
            "command_status": "exit_0",
            "execution_status": "code_model_smoke_completed" if generation_ok else "failed",
            "validity_status": "valid" if generation_ok else "diagnostic_only",
            "quality_gate": "pass" if generation_ok else "fail",
            "target_smoke_closed": bool(generation_ok and summary["params_after"] < summary["params_before"]),
        })
        (out / "reload_check.json").write_text(json.dumps(reload_check, indent=2) + "\n", encoding="utf-8")
        (out / "generation_check.json").write_text(json.dumps(reload_check, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        stderr.append(type(exc).__name__ + ": " + str(exc))
        summary["first_failing_function"] = "run_qwen_smoke"
        summary["failure_reason"] = str(exc)
    flush()

def main() -> int:
    parser = argparse.ArgumentParser(description="Run Flab benchmark-guided jobs.")
    parser.add_argument("--job", required=True, choices=["tiny", "qwen15b_smoke"])
    args = parser.parse_args()
    if args.job == "tiny":
        run_tiny()
    elif args.job == "qwen15b_smoke":
        run_qwen_smoke()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
