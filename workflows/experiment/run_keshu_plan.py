#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAMP = os.environ.get("KESHU_STAMP", "20260804_130541")
PYTHON = os.environ.get("SRTP_STAGE1_PYTHON", "python")


def write_common(run_dir: Path, summary: dict, stdout: str = "", stderr: str = "") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stdout.log").write_text(stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(stderr, encoding="utf-8")
    (run_dir / "metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(f"# {summary['run_id']}\n\nStatus: {summary['status']}\n\n", encoding="utf-8")
    (run_dir / "resource.csv").write_text("metric,value\nstatus," + summary["status"] + "\n", encoding="utf-8")
    (run_dir / "resource_summary.json").write_text(json.dumps({"status": summary["status"]}, indent=2) + "\n", encoding="utf-8")


def audit_scores(run_id: str) -> dict:
    scores = list(csv.DictReader((ROOT / "results/status/scores.csv").open(encoding="utf-8-sig", newline="")))
    return {row["benchmark"]: row["task_count"] for row in scores if row["run_id"] == run_id}


def run_llmpruner() -> None:
    run_id = f"llmpruner_primary_evidence_audit_{STAMP}"
    run_dir = ROOT / "results/evidence/diagnostics" / run_id
    primary = "llmpruner_qwen25c15b_official_keep80_gpu_20260730_105340"
    fallback = "llmpruner_codellama7b_r4_layerdrop_keep80_full_20260725_182022"
    summary = {
        "status": "success",
        "run_id": run_id,
        "method": "LLM-Pruner",
        "primary_run": primary,
        "primary_scores": audit_scores(primary),
        "fallback_run": fallback,
        "fallback_officiality": "fallback_non_official",
        "quality_gate": "fail",
        "rerun_policy": "not rerun; existing formal evidence audited",
    }
    (run_dir / "command.sh").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\npython workflows/experiment/run_keshu_plan.py --job llmpruner_primary_evidence_audit\n", encoding="utf-8")
    (run_dir / "command.sh").chmod(0o755)
    write_common(run_dir, summary, stdout=json.dumps(summary, indent=2) + "\n")


def run_slicegpt() -> None:
    run_id = f"slicegpt_primary_evidence_audit_{STAMP}"
    run_dir = ROOT / "results/evidence/diagnostics" / run_id
    primary = "slicegpt_qwen25c15b_official_keep80_gpu_20260731_011001"
    legacy = "slicegpt_codellama7b_r4_benchguided_evalhalf_20260726_053225"
    summary = {
        "status": "success",
        "run_id": run_id,
        "method": "SliceGPT",
        "primary_run": primary,
        "primary_scores": audit_scores(primary),
        "legacy_partial_run": legacy,
        "legacy_partial": {"humaneval": 82, "mbpp": 82, "livecodebench": "missing"},
        "quality_gate": "fail",
        "rerun_policy": "not rerun; existing Qwen primary evidence audited",
    }
    (run_dir / "command.sh").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\npython workflows/experiment/run_keshu_plan.py --job slicegpt_primary_evidence_audit\n", encoding="utf-8")
    (run_dir / "command.sh").chmod(0o755)
    write_common(run_dir, summary, stdout=json.dumps(summary, indent=2) + "\n")


def run_laco() -> None:
    run_id = f"laco_upstream_smoke_{STAMP}"
    run_dir = ROOT / "results/evidence/smoke" / run_id
    command = [PYTHON, "methods/laco/run_smoke.py", "--output-dir", str(run_dir), "--max-samples", "2"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + " ".join(command) + "\n", encoding="utf-8")
    (run_dir / "command.sh").chmod(0o755)
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=600)
    (run_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        summary = {"status": "blocked", "run_id": run_id, "method": "LaCo", "returncode": proc.returncode, "blocker": "tiny core smoke failed"}
        write_common(run_dir, summary, proc.stdout, proc.stderr)
    else:
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        summary.update({"run_id": run_id, "method": "LaCo", "protocol": "tiny_llama_core_smoke", "variant": "adjacent_layer_similarity_collapse"})
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (run_dir / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (run_dir / "resource.csv").write_text("metric,value\nstatus,success\n", encoding="utf-8")
        (run_dir / "resource_summary.json").write_text(json.dumps({"status": "success"}, indent=2) + "\n", encoding="utf-8")
        (run_dir / "model_structure.json").write_text(json.dumps({"layers_before": summary["layers_before"], "layers_after": summary["layers_after"]}, indent=2) + "\n", encoding="utf-8")


def run_flab_tiny() -> None:
    run_id = f"flab_benchmark_activation_tiny_smoke_{STAMP}"
    run_dir = ROOT / "results/evidence/smoke" / run_id
    summary = {
        "status": "blocked",
        "run_id": run_id,
        "method": "Flab-Pruner",
        "entered_activation_path": True,
        "activation_source": "real vendored-model path required",
        "blocker_code": "vendored_config_only_no_external_mask_schema",
        "blocker_reason": "vendored Flab Qwen2 prune API accepts config/stage target dimensions but does not expose a verified external per-channel mask schema for benchmark_activation",
        "qwen15b_smoke": "not_run_due_to_config_only_prune_api",
        "twenty_task_quality_gate": "not_run_due_to_config_only_prune_api",
        "formal_guided_variants": "not_run_due_to_config_only_prune_api",
    }
    (run_dir / "command.sh").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\npython workflows/experiment/run_keshu_plan.py --job flab_benchmark_activation_tiny_smoke\n", encoding="utf-8")
    (run_dir / "command.sh").chmod(0o755)
    (run_dir / "prune_api.json").write_text((ROOT / f"results/evidence/diagnostics/flab_prune_api_audit_{STAMP}/prune_api.json").read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "blocker.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_common(run_dir, summary, stdout=json.dumps(summary, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Keshu-scoped bounded jobs.")
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    jobs = {
        "llmpruner_primary_evidence_audit": run_llmpruner,
        "slicegpt_primary_evidence_audit": run_slicegpt,
        "laco_core_smoke": run_laco,
        "flab_benchmark_activation_tiny_smoke": run_flab_tiny,
    }
    jobs[args.job]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
