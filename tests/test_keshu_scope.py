from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_OF_SCOPE = {"Magnitude", "Wanda", "DSnoT", "OWL", "SparseGPT", "MaskLLM", "Pruner-Zero", "FLAP"}


def test_keshu_plan_is_owner_scoped():
    plan = ROOT / "workflows/experiment/keshu_plan.yaml"
    assert plan.exists()
    text = plan.read_text(encoding="utf-8")
    for method in ["Flab-Pruner", "LLM-Pruner", "SliceGPT", "LaCo"]:
        assert method in text
    for method in OUT_OF_SCOPE:
        assert method not in text
    assert "stage1_final_attempts.py" not in text


def test_keshu_plan_has_nonempty_expected_outputs_and_no_false_success():
    text = (ROOT / "workflows/experiment/keshu_plan.yaml").read_text(encoding="utf-8")
    assert "expected_outputs: []" not in text
    assert "--help" not in text
    assert "notebook present" not in text
    assert "Path(\"third_party/laco" not in text
