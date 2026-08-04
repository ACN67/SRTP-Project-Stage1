from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import torch


@dataclass
class FlabPruneSchema:
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    hidden_size_remain: int
    intermediate_size_remain: int
    num_attention_heads_remain: int
    num_key_value_heads_remain: int


@dataclass
class BenchmarkImportance:
    layer_importance: dict[int, torch.Tensor]
    guide_hash: str
    importance_hash: str
    sample_count: int


@dataclass
class FlabZS:
    hidden_index: torch.Tensor
    head_indexes: list[torch.Tensor]
    kv_head_indexes: list[torch.Tensor]
    intermediate_indexes: list[torch.Tensor]
    benchmark_guided_dimensions: list[str]
    config_derived_dimensions: list[str]
    selected_index_hash: str

    def to_upstream(self) -> dict[str, Any]:
        def mask(length: int, index: torch.Tensor) -> torch.Tensor:
            out = torch.zeros(length, dtype=torch.bool, device=index.device)
            out[index.to(torch.long)] = True
            return out

        return {
            "hidden_mask": mask(int(self.hidden_index.max().item()) + 1 if len(self.hidden_index) else 0, self.hidden_index),
            "hidden_index": self.hidden_index,
            "head_masks": [mask(int(idx.max().item()) + 1 if len(idx) else 0, idx) for idx in self.head_indexes],
            "head_indexes": self.head_indexes,
            "kv_head_masks": [mask(int(idx.max().item()) + 1 if len(idx) else 0, idx) for idx in self.kv_head_indexes],
            "kv_head_indexes": self.kv_head_indexes,
            "intermediate_masks": [mask(int(idx.max().item()) + 1 if len(idx) else 0, idx) for idx in self.intermediate_indexes],
            "intermediate_indexes": self.intermediate_indexes,
        }


def tensor_hash(values: Any) -> str:
    def convert(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    raw = json.dumps(convert(values), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def inspect_flab_schema(model) -> FlabPruneSchema:
    config = model.config
    return FlabPruneSchema(
        num_hidden_layers=int(config.num_hidden_layers),
        hidden_size=int(config.hidden_size),
        intermediate_size=int(config.intermediate_size),
        num_attention_heads=int(config.num_attention_heads),
        num_key_value_heads=int(config.num_key_value_heads),
        hidden_size_remain=int(getattr(config, "hidden_size_remain", config.hidden_size)),
        intermediate_size_remain=int(getattr(config, "ffn_hidden_size_remain", config.intermediate_size)),
        num_attention_heads_remain=int(getattr(config, "num_attention_heads_remain", config.num_attention_heads)),
        num_key_value_heads_remain=int(getattr(config, "num_key_value_heads_remain", config.num_key_value_heads)),
    )


def full_config_schema(model, intermediate_size_remain: int | None = None) -> FlabPruneSchema:
    schema = inspect_flab_schema(model)
    return FlabPruneSchema(
        schema.num_hidden_layers,
        schema.hidden_size,
        schema.intermediate_size,
        schema.num_attention_heads,
        schema.num_key_value_heads,
        schema.hidden_size,
        int(intermediate_size_remain or schema.intermediate_size),
        schema.num_attention_heads,
        schema.num_key_value_heads,
    )


def select_intermediate_indexes(importance: BenchmarkImportance, target_parameter_keep_ratio: float, schema: FlabPruneSchema) -> FlabZS:
    keep = max(1, min(schema.intermediate_size, int(schema.intermediate_size_remain)))
    inter = []
    for layer in range(schema.num_hidden_layers):
        scores = importance.layer_importance[layer].detach().float().cpu()
        if scores.numel() != schema.intermediate_size:
            raise ValueError(f"layer {layer} importance length {scores.numel()} != {schema.intermediate_size}")
        idx = torch.topk(scores, k=keep, largest=True).indices.sort().values.to(torch.long)
        inter.append(idx)
    zs = FlabZS(
        hidden_index=torch.arange(schema.hidden_size, dtype=torch.long),
        head_indexes=[torch.arange(schema.num_attention_heads, dtype=torch.long) for _ in range(schema.num_hidden_layers)],
        kv_head_indexes=[torch.arange(schema.num_key_value_heads, dtype=torch.long) for _ in range(schema.num_hidden_layers)],
        intermediate_indexes=inter,
        benchmark_guided_dimensions=["intermediate"],
        config_derived_dimensions=["hidden", "attention_head", "kv_head"],
        selected_index_hash="",
    )
    zs.selected_index_hash = tensor_hash(zs.to_upstream())
    return zs


def validate_flab_zs(model, zs: FlabZS, schema: FlabPruneSchema) -> None:
    checks = [
        ("hidden_index", len(zs.hidden_index), schema.hidden_size_remain),
        ("head_indexes", len(zs.head_indexes), schema.num_hidden_layers),
        ("kv_head_indexes", len(zs.kv_head_indexes), schema.num_hidden_layers),
        ("intermediate_indexes", len(zs.intermediate_indexes), schema.num_hidden_layers),
    ]
    for name, actual, expected in checks:
        if actual != expected:
            raise ValueError(f"{name} length {actual} != {expected}")
    for layer, idx in enumerate(zs.intermediate_indexes):
        if len(idx) != schema.intermediate_size_remain:
            raise ValueError(f"intermediate_indexes[{layer}] length {len(idx)} != {schema.intermediate_size_remain}")
        if int(idx.min()) < 0 or int(idx.max()) >= schema.intermediate_size:
            raise ValueError(f"intermediate_indexes[{layer}] out of range")
    if len(getattr(model.config, "num_hidden_layers", range(zs.intermediate_indexes))) != schema.num_hidden_layers if False else False:
        raise ValueError("unreachable")


def apply_flab_zs(model, zs: FlabZS, schema: FlabPruneSchema) -> dict[str, Any]:
    before = count_parameters(model)
    device = next(model.parameters()).device
    zs.hidden_index = zs.hidden_index.to(device)
    zs.head_indexes = [idx.to(device) for idx in zs.head_indexes]
    zs.kv_head_indexes = [idx.to(device) for idx in zs.kv_head_indexes]
    zs.intermediate_indexes = [idx.to(device) for idx in zs.intermediate_indexes]
    model.model.prune(zs.to_upstream())
    synchronize_top_level_model(model, zs, schema)
    after = count_parameters(model)
    return {
        "actual_flab_prune_called": True,
        "params_before": before,
        "params_after": after,
        "actual_parameter_keep_ratio": after / before if before else None,
        "params_after_less_than_before": after < before,
    }


def synchronize_top_level_model(model, zs: FlabZS, schema: FlabPruneSchema) -> None:
    model.config.hidden_size = schema.hidden_size_remain
    model.config.intermediate_size = schema.intermediate_size_remain
    model.config.ffn_hidden_size_remain = schema.intermediate_size_remain
    model.config.num_attention_heads = schema.num_attention_heads_remain
    model.config.num_key_value_heads = schema.num_key_value_heads_remain
    # Hidden remains full in this implementation, so lm_head already matches.


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))
