from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPPED = [
    "flab_qwen15b_benchmark_guided_he_keep80_capped32_20260804_135032",
    "flab_qwen15b_benchmark_guided_mbpp_keep80_capped32_20260804_135032",
    "flab_qwen15b_benchmark_guided_lcb_keep80_capped32_20260804_135032",
]


def test_flab_benchmark_guided_final_status_is_experimental_failure_not_execution_failure() -> None:
    status = json.loads((ROOT / "results/status/flab_benchmark_guided_completion.json").read_text(encoding="utf-8"))
    assert status["implementation_closed"] is True
    assert status["target_smoke_closed"] is True
    assert status["experiment_execution_closed"] is True
    assert status["quality_gate"] == "fail"
    assert status["formal_full_evaluation"] == "skipped_due_to_output_collapse"
    assert status["officiality"] == "experimental_extension"


def test_capped_runs_are_pilot_not_formal() -> None:
    with (ROOT / "results/status/runs.csv").open(encoding="utf-8-sig", newline="") as handle:
        runs = {row["run_id"]: row for row in csv.DictReader(handle)}
    for run_id in CAPPED:
        row = runs[run_id]
        assert row["protocol"] == "pilot_quality_gate"
        assert row["execution_status"] == "pilot_quality_gate_completed"
        assert row["officiality"] == "experimental_extension"
        assert row["quality_gate"] == "fail"
