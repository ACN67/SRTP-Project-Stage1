from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_flab_plan_has_nonempty_outputs_and_historical_run_ids() -> None:
    plan = yaml.safe_load((ROOT / "workflows/experiment/flab_benchmark_guided_plan.yaml").read_text(encoding="utf-8"))
    for job in plan["jobs"]:
        assert job.get("status") == "historical_completed"
        assert job.get("actual_run_id")
        assert job.get("expected_outputs")
        assert "--run-id" in job["command"] or "--schema-audit" in job["command"]
        for output in job["expected_outputs"]:
            assert "[]" not in output
