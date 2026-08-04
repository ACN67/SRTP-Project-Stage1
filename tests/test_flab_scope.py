from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ["LLM-Pruner", "SliceGPT", "LaCo", "Magnitude", "Wanda", "DSnoT", "OWL", "SparseGPT", "MaskLLM", "Pruner-Zero", "FLAP", "stage1_final_attempts.py"]


def test_flab_plan_is_scoped_and_not_proxy_based():
    plan = ROOT / "workflows/experiment/flab_benchmark_guided_plan.yaml"
    assert plan.exists()
    text = plan.read_text(encoding="utf-8")
    assert "method: Flab-Pruner" in text
    for item in FORBIDDEN:
        assert item not in text
    assert "expected_outputs: []" not in text
    assert "--help" not in text
    assert "dry-run" not in text.lower()


def test_benchmark_path_does_not_use_plain_hf_pruning_model():
    text = (ROOT / "methods/flab_pruner/qwen_prune.py").read_text(encoding="utf-8")
    marker = 'if args.importance_mode == "benchmark_activation":'
    assert marker in text
    branch = text.split(marker, 1)[1].split('if args.importance_mode == "structural"', 1)[0]
    assert "AutoModelForCausalLM" not in branch
    assert "return load_hf_model" not in branch
