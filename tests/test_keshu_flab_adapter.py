from __future__ import annotations

import json
from pathlib import Path

from methods.flab_pruner import qwen_prune


ROOT = Path(__file__).resolve().parents[1]


def test_flab_prune_api_audit_records_config_only_schema():
    audits = sorted((ROOT / "results/evidence/diagnostics").glob("flab_prune_api_audit_*/prune_api.json"))
    assert audits
    data = json.loads(audits[-1].read_text(encoding="utf-8"))
    assert data["prune_signature"] == "(self, config, stage)"
    assert data["supports_config"] is True
    assert data["supports_external_zs"] is False


def test_benchmark_activation_uses_flab_loader_not_plain_hf():
    text = (ROOT / "methods/flab_pruner/qwen_prune.py").read_text(encoding="utf-8")
    assert "def load_flab_qwen_model" in text
    assert "benchmark_activation" in text
    assert "load_hf_model(args.model" not in text
    assert "unsupported_external_mask_schema" in text


def test_config_only_prune_api_blocks_module_name_masks():
    api = {"supports_external_zs": False, "prune_signature": "(self, config, stage)"}
    blocker = qwen_prune.validate_benchmark_activation_prune_api(api, {"layers.0.mlp": [1, 0, 1]})
    assert blocker["status"] == "blocked"
    assert "does not expose" in blocker["reason"]
