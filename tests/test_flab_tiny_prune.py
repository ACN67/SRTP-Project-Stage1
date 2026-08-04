from __future__ import annotations

import json
from pathlib import Path

from methods.flab_pruner import benchmark_guided


def test_tiny_vendored_qwen2_prunes_and_updates_structure(tmp_path: Path):
    out = tmp_path / "tiny"
    result = benchmark_guided.run_tiny_pair(output_dir=out, target_parameter_keep_ratio=0.80)
    for key in ["a", "b"]:
        summary = result[key]
        assert summary["vendored_flab_model_loaded"] is True
        assert summary["real_model_forward_called"] is True
        assert summary["tensor_activation_collected"] is True
        assert summary["actual_flab_prune_called"] is True
        assert summary["params_after"] < summary["params_before"]
        assert summary["actual_parameter_keep_ratio"] < 1.0
        assert summary["artifact_saved"] is True
        assert (out / f"parameter_summary_{key}.json").exists()
    saved = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert saved["execution_status"] == "tiny_smoke_completed"
    assert saved["quality_gate"] == "pass"
