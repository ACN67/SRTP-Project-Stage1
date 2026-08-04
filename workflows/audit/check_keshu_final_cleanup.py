#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAPPED = [
    "flab_qwen15b_benchmark_guided_he_keep80_capped32_20260804_135032",
    "flab_qwen15b_benchmark_guided_mbpp_keep80_capped32_20260804_135032",
    "flab_qwen15b_benchmark_guided_lcb_keep80_capped32_20260804_135032",
]
ARTIFACT_RUNS = ["flab_qwen15b_benchmark_guided_smoke_20260804_135032", *CAPPED]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def evidence_hash_check(before: Path) -> dict:
    missing: list[str] = []
    changed: list[str] = []
    checked = 0
    with before.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            category = row["category"].strip()
            run_id = row["run_id"].strip()
            internal = row["internal_relative_path"].strip()
            path = ROOT / "results/evidence" / category
            if run_id:
                path = path / run_id
            if internal:
                path = path / internal
            if not path.exists():
                missing.append(path.as_posix())
                continue
            if path.is_dir():
                continue
            checked += 1
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != row["sha256"].strip():
                changed.append(path.as_posix())
    return {"checked": checked, "missing": missing, "changed": changed}


def collect() -> dict:
    runs = {row["run_id"]: row for row in read_csv(ROOT / "results/status/runs.csv")}
    scores = read_csv(ROOT / "results/status/scores.csv")
    formal = read_csv(ROOT / "results/formal/r4_half/scores.csv")
    artifacts = read_csv(ROOT / "results/status/artifacts.csv")
    keshu = json.loads((ROOT / "results/status/keshu_completion.json").read_text(encoding="utf-8"))
    completion = json.loads((ROOT / "results/status/flab_benchmark_guided_completion.json").read_text(encoding="utf-8"))
    artifact_rows = [row for row in artifacts if row["run_id"] in ARTIFACT_RUNS]
    return {
        "artifact_rows": len(artifact_rows),
        "artifact_availability": {row["run_id"]: row["availability"] for row in artifact_rows},
        "capped_protocols": {rid: runs[rid]["protocol"] for rid in CAPPED},
        "capped_execution_status": {rid: runs[rid]["execution_status"] for rid in CAPPED},
        "capped_quality_gate": {rid: runs[rid]["quality_gate"] for rid in CAPPED},
        "capped_formal_rows": [row["run_id"] for row in formal if row["run_id"] in CAPPED],
        "capped_score_rows": [row["run_id"] for row in scores if row["run_id"] in CAPPED],
        "qwen_smoke_quality_gate": runs["flab_qwen15b_benchmark_guided_smoke_20260804_135032"]["quality_gate"],
        "owner_execution_closed": keshu["owner_execution_closed"],
        "global_stage1_execution_closed": keshu["global_stage"]["stage1_execution_closed"],
        "flab_completion": {
            "implementation_closed": completion["implementation_closed"],
            "target_smoke_closed": completion["target_smoke_closed"],
            "experiment_execution_closed": completion["experiment_execution_closed"],
            "quality_gate": completion["quality_gate"],
            "formal_full_evaluation": completion["formal_full_evaluation"],
            "officiality": completion["officiality"],
        },
    }


def validate(report: dict) -> list[str]:
    errors: list[str] = []
    if report["artifact_rows"] != len(ARTIFACT_RUNS):
        errors.append("missing Flab benchmark-guided artifact registry rows")
    if any(v not in {"available_archived", "missing_after_ephemeral_tmp_cleanup"} for v in report["artifact_availability"].values()):
        errors.append("invalid artifact availability value")
    if any(v != "pilot_quality_gate" for v in report["capped_protocols"].values()):
        errors.append("capped variants must use pilot_quality_gate protocol")
    if any(v != "pilot_quality_gate_completed" for v in report["capped_execution_status"].values()):
        errors.append("capped variants must use pilot_quality_gate_completed execution status")
    if report["capped_formal_rows"]:
        errors.append("capped pilot rows leaked into formal table")
    if report["capped_score_rows"]:
        errors.append("capped pilot rows leaked into formal score registry")
    if report["qwen_smoke_quality_gate"] != "pass_for_execution":
        errors.append("Qwen smoke quality gate must be pass_for_execution")
    if report["owner_execution_closed"] is not True:
        errors.append("Keshu owner execution must be closed")
    if report["global_stage1_execution_closed"] is not False:
        errors.append("global stage must remain open")
    expected = {
        "implementation_closed": True,
        "target_smoke_closed": True,
        "experiment_execution_closed": True,
        "quality_gate": "fail",
        "formal_full_evaluation": "skipped_due_to_output_collapse",
        "officiality": "experimental_extension",
    }
    if report["flab_completion"] != expected:
        errors.append("Flab benchmark-guided completion semantics mismatch")
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check final Keshu Stage-1 cleanup semantics.")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--evidence-before", type=Path)
    parser.add_argument("--artifact-archive", action="store_true")
    args = parser.parse_args(argv)
    report = collect()
    if args.evidence_before:
        report["evidence_hash_check"] = evidence_hash_check(args.evidence_before)
    report["errors"] = validate(report)
    if args.write:
        path = ROOT / "results/status/keshu_final_cleanup.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
