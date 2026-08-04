from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_keshu_owner_closed_but_global_stage_open() -> None:
    data = json.loads((ROOT / "results/status/keshu_completion.json").read_text(encoding="utf-8"))
    assert data["owner"] == "常珂舒"
    assert data["owner_execution_closed"] is True
    assert data["global_stage"]["stage1_execution_closed"] is False
    assert data["methods"]["LaCo"]["registry"]["execution_status"] == "skipped"
    assert data["methods"]["LaCo"]["registry"]["validity_status"] == "not_applicable"
