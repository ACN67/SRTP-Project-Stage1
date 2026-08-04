from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "tests/acceptance/flab_benchmark_guided.yaml"


def test_acceptance_contract_is_frozen_and_strict():
    text = CONTRACT.read_text(encoding="utf-8")
    assert "task_id: flab_benchmark_guided_experimental" in text
    assert "owner: 常珂舒" in text
    for item in [
        "vendored_flab_model_loaded",
        "actual_flab_prune_called",
        "params_after_less_than_before",
        "artifact_reloaded",
        "nonempty_generation_after_reload",
    ]:
        assert item in text
    assert "allowed_blocker_codes: []" in text


def test_contract_forbids_proxy_successes():
    text = CONTRACT.read_text(encoding="utf-8")
    for item in ["help_exit_zero", "file_exists", "dry_run_only", "synthetic_activation", "prompt_length_importance", "expected_outputs_empty"]:
        assert item in text
