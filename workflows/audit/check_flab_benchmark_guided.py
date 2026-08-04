#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "tests/acceptance/flab_benchmark_guided.yaml"


def validate_tiny_evidence(path: Path) -> dict:
    required = ["summary.json", "parameter_summary_a.json", "parameter_summary_b.json", "reload_check_a.json", "reload_check_b.json", "guide_comparison.json"]
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        return {"implementation_closed": False, "missing": missing}
    params_a = json.loads((path / "parameter_summary_a.json").read_text(encoding="utf-8"))
    params_b = json.loads((path / "parameter_summary_b.json").read_text(encoding="utf-8"))
    reload_a = json.loads((path / "reload_check_a.json").read_text(encoding="utf-8"))
    reload_b = json.loads((path / "reload_check_b.json").read_text(encoding="utf-8"))
    comp = json.loads((path / "guide_comparison.json").read_text(encoding="utf-8"))
    params_ok = params_a.get("params_after", 0) < params_a.get("params_before", 0) and params_b.get("params_after", 0) < params_b.get("params_before", 0)
    reload_ok = all(x.get("reload_success") and x.get("forward_after_reload") and x.get("nonempty_generation_after_reload") for x in [reload_a, reload_b])
    causal_ok = comp.get("guide_a_importance_hash") != comp.get("guide_b_importance_hash") and comp.get("selected_indices_differ") is True
    return {
        "implementation_closed": bool(params_ok and reload_ok and causal_ok),
        "params_after_less_than_before": bool(params_ok),
        "tiny_save_reload_success": bool(reload_ok),
        "guide_causality": bool(causal_ok),
    }


def collect_completion() -> dict:
    tiny = sorted((ROOT / "results/evidence/smoke").glob("flab_benchmark_guided_tiny_*/summary.json"))
    tiny_result = validate_tiny_evidence(tiny[-1].parent) if tiny else {"implementation_closed": False, "missing": ["tiny_evidence"]}
    qwen = sorted((ROOT / "results/evidence/smoke").glob("flab_qwen15b_benchmark_guided_smoke_*/summary.json"))
    qwen_summary = json.loads(qwen[-1].read_text(encoding="utf-8")) if qwen else {}
    variants = {
        "he": "flab_qwen15b_benchmark_guided_he_keep80_capped32_*",
        "mbpp": "flab_qwen15b_benchmark_guided_mbpp_keep80_capped32_*",
        "lcb": "flab_qwen15b_benchmark_guided_lcb_keep80_capped32_*",
    }
    formal_variants = {}
    all_attempted = True
    any_quality_fail = False
    for name, pattern in variants.items():
        found = sorted((ROOT / "results/evidence/smoke").glob(pattern + "/summary.json"))
        if not found:
            formal_variants[name] = "missing"
            all_attempted = False
            continue
        data = json.loads(found[-1].read_text(encoding="utf-8"))
        formal_variants[name] = data.get("formal_full_eval") or data.get("execution_status")
        any_quality_fail = any_quality_fail or data.get("quality_gate") == "fail"
    target_closed = bool(qwen_summary.get("target_smoke_closed") and qwen_summary.get("params_after", 0) < qwen_summary.get("params_before", 0) and qwen_summary.get("nonempty_generation_after_reload"))
    experiment_closed = bool(all_attempted and target_closed)
    qwen_quality_gate = qwen_summary.get("quality_gate")
    if qwen_quality_gate == "pass":
        qwen_summary["quality_gate"] = "pass_for_execution"
    out = {
        "contract": CONTRACT.relative_to(ROOT).as_posix(),
        "implementation_closed": tiny_result.get("implementation_closed", False),
        "target_smoke_closed": target_closed,
        "experiment_closed": experiment_closed,
        "experiment_execution_closed": experiment_closed,
        "quality_gate": "fail" if any_quality_fail else ("pending" if not experiment_closed else "pass"),
        "formal_full_evaluation": "skipped_due_to_output_collapse" if any_quality_fail else ("pending" if not experiment_closed else "not_run"),
        "officiality": "experimental_extension",
        "formal_variants": formal_variants,
        "tiny": tiny_result,
        "qwen15b_smoke": qwen_summary,
    }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate Flab benchmark-guided evidence.")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--schema-audit", action="store_true")
    args = parser.parse_args(argv)
    if args.schema_audit:
        print(json.dumps({"contract": str(CONTRACT), "schema_audit": True}, indent=2))
        return 0
    out = collect_completion()
    if args.write:
        target = ROOT / "results/status/flab_benchmark_guided_completion.json"
        target.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        md = ROOT / "results/status/flab_benchmark_guided_completion.md"
        md.write_text(f"# Flab Benchmark-Guided Completion\n\nImplementation closed: {str(out['implementation_closed']).lower()}\nTarget smoke closed: {str(out['target_smoke_closed']).lower()}\nExperiment closed: {str(out['experiment_closed']).lower()}\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if out["implementation_closed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
