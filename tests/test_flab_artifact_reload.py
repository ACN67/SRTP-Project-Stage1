from __future__ import annotations

import json

from methods.flab_pruner import benchmark_guided


def test_tiny_artifact_reload_forward_and_generation(tmp_path):
    out = tmp_path / "reload"
    benchmark_guided.run_tiny_pair(output_dir=out, target_parameter_keep_ratio=0.80)
    for suffix in ["a", "b"]:
        data = json.loads((out / f"reload_check_{suffix}.json").read_text(encoding="utf-8"))
        assert data["reload_success"] is True
        assert data["forward_after_reload"] is True
        assert data["nonempty_generation_after_reload"] is True
        assert data["generated_token_count"] > 0
