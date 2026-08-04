from __future__ import annotations

from methods.flab_pruner import benchmark_guided


def test_distinct_guides_change_importance_and_selected_indices(tmp_path):
    result = benchmark_guided.run_tiny_pair(output_dir=tmp_path / "causal", target_parameter_keep_ratio=0.80)
    comp = result["guide_comparison"]
    assert comp["same_initial_model_for_guide_comparison"] is True
    assert comp["guide_a_importance_hash"] != comp["guide_b_importance_hash"]
    assert comp["selected_indices_differ"] is True
    assert comp["selected_indices_a_hash"] != comp["selected_indices_b_hash"]
