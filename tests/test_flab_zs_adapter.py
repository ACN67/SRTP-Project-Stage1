from __future__ import annotations

from types import SimpleNamespace

import torch

from methods.flab_pruner import zs_adapter


def test_select_intermediate_indexes_uses_tensor_importance_not_prompt_length():
    schema = zs_adapter.FlabPruneSchema(
        num_hidden_layers=2,
        hidden_size=8,
        intermediate_size=6,
        num_attention_heads=2,
        num_key_value_heads=1,
        hidden_size_remain=8,
        intermediate_size_remain=3,
        num_attention_heads_remain=2,
        num_key_value_heads_remain=1,
    )
    importance = zs_adapter.BenchmarkImportance(
        layer_importance={0: torch.tensor([0.1, 9.0, 0.2, 8.0, 0.3, 7.0]), 1: torch.tensor([6.0, 0.1, 5.0, 0.2, 4.0, 0.3])},
        guide_hash="guide",
        importance_hash="importance",
        sample_count=2,
    )
    zs = zs_adapter.select_intermediate_indexes(importance, 0.80, schema)
    assert [x.tolist() for x in zs.intermediate_indexes] == [[1, 3, 5], [0, 2, 4]]
    assert zs.benchmark_guided_dimensions == ["intermediate"]
    assert "hidden" in zs.config_derived_dimensions


def test_validate_flab_zs_rejects_wrong_schema_lengths():
    schema = zs_adapter.FlabPruneSchema(2, 8, 6, 2, 1, 8, 3, 2, 1)
    zs = zs_adapter.FlabZS(
        hidden_index=torch.arange(8),
        head_indexes=[torch.arange(2)],
        kv_head_indexes=[torch.arange(1), torch.arange(1)],
        intermediate_indexes=[torch.arange(3), torch.arange(3)],
        benchmark_guided_dimensions=["intermediate"],
        config_derived_dimensions=["hidden", "attention_head", "kv_head"],
        selected_index_hash="x",
    )
    try:
        zs_adapter.validate_flab_zs(SimpleNamespace(config=SimpleNamespace(num_hidden_layers=2)), zs, schema)
    except ValueError as exc:
        assert "head_indexes" in str(exc)
    else:
        raise AssertionError("invalid zs unexpectedly accepted")
