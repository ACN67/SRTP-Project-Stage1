#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KESHU = ["Flab-Pruner", "LLM-Pruner", "SliceGPT", "LaCo"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def laco_closed(method: dict) -> bool:
    return bool(method.get("laco_core_smoke") or method.get("laco_core_blocker")) and not method.get("laco_file_probe_only")


def flab_closed(method: dict) -> bool:
    blocker = method.get("activation_blocker", "")
    return bool(method.get("structural_primary")) and bool(method.get("activation_smoke") or blocker == "vendored_config_only_no_external_mask_schema")


def assess_owner_completion(methods: dict[str, dict]) -> dict:
    closed = {
        "Flab-Pruner": flab_closed(methods.get("Flab-Pruner", {})),
        "LLM-Pruner": bool(methods.get("LLM-Pruner", {}).get("primary_audit")),
        "SliceGPT": bool(methods.get("SliceGPT", {}).get("primary_audit")),
        "LaCo": laco_closed(methods.get("LaCo", {})),
    }
    return {
        "owner": "\u5e38\u73c2\u8212",
        "owner_execution_closed": all(closed.values()),
        "methods": methods,
        "closed_checks": closed,
    }


def global_stage_override_reason() -> dict:
    return {
        "stage1_execution_closed": False,
        "reason": "Only the methods owned by Keshu were processed in this run.",
    }


def collect_state() -> dict[str, dict]:
    runs = read_csv(ROOT / "results/status/runs.csv")
    methods = read_csv(ROOT / "results/status/methods.csv")
    run_ids = {row["run_id"]: row for row in runs}
    state = {
        "Flab-Pruner": {
            "structural_primary": any("flabpruner_qwen25c15b_official_keep80" in row["run_id"] for row in runs),
            "activation_smoke": any(row["run_id"].startswith("flab_benchmark_activation_tiny_smoke_") and row["execution_status"] == "completed" for row in runs),
            "activation_blocker": "vendored_config_only_no_external_mask_schema"
            if any(row["run_id"].startswith("flab_benchmark_activation_tiny_smoke_") and row["execution_status"] == "blocked" for row in runs)
            else "",
            "registry": next((m for m in methods if m["method"] == "Flab-Pruner"), {}),
        },
        "LLM-Pruner": {
            "primary_audit": any(row["run_id"].startswith("llmpruner_primary_evidence_audit_") for row in runs),
            "primary_run": "llmpruner_qwen25c15b_official_keep80_gpu_20260730_105340" in run_ids,
            "registry": next((m for m in methods if m["method"] == "LLM-Pruner"), {}),
        },
        "SliceGPT": {
            "primary_audit": any(row["run_id"].startswith("slicegpt_primary_evidence_audit_") for row in runs),
            "primary_run": "slicegpt_qwen25c15b_official_keep80_gpu_20260731_011001" in run_ids,
            "registry": next((m for m in methods if m["method"] == "SliceGPT"), {}),
        },
        "LaCo": {
            "laco_core_smoke": any(row["run_id"].startswith("laco_upstream_smoke_") and row["execution_status"] == "completed" for row in runs),
            "laco_core_blocker": any(row["run_id"].startswith("laco_core_blocker_") for row in runs),
            "laco_file_probe_only": not any(row["run_id"].startswith("laco_upstream_smoke_") for row in runs),
            "registry": next((m for m in methods if m["method"] == "LaCo"), {}),
        },
    }
    return state


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check Keshu-owned Stage-1 method completion.")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    result = assess_owner_completion(collect_state())
    result["global_stage"] = global_stage_override_reason()
    if args.write:
        out = ROOT / "results/status/keshu_completion.json"
        out.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        md = ROOT / "results/status/keshu_completion.md"
        lines = ["# Keshu Completion", "", "Owner: \\u5e38\\u73c2\\u8212", f"Owner execution closed: {str(result['owner_execution_closed']).lower()}", ""]
        for method, closed in result["closed_checks"].items():
            lines.append(f"- {method}: {'closed' if closed else 'open'}")
        lines.append("")
        lines.append("Global stage remains open for this owner-scoped run.")
        md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result["owner_execution_closed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
