from __future__ import annotations

import json
from pathlib import Path

from workflows.audit import check_flab_benchmark_guided


def test_success_validator_rejects_fake_artifact(tmp_path: Path):
    d = tmp_path / "fake"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps({"execution_status": "tiny_smoke_completed"}), encoding="utf-8")
    assert check_flab_benchmark_guided.validate_tiny_evidence(d)["implementation_closed"] is False


def test_success_validator_accepts_real_tiny_evidence(tmp_path: Path):
    from methods.flab_pruner import benchmark_guided

    d = tmp_path / "real"
    benchmark_guided.run_tiny_pair(output_dir=d, target_parameter_keep_ratio=0.80)
    out = check_flab_benchmark_guided.validate_tiny_evidence(d)
    assert out["implementation_closed"] is True
    assert out["params_after_less_than_before"] is True
    assert out["tiny_save_reload_success"] is True
