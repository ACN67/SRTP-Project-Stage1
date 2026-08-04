from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_laco_smoke_executes_core_algorithm(tmp_path: Path):
    output = tmp_path / "laco"
    result = subprocess.run(
        [sys.executable, str(ROOT / "methods/laco/run_smoke.py"), "--output-dir", str(output), "--max-samples", "2"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["entered_core_algorithm"] is True
    assert summary["layers_after"] < summary["layers_before"]
    assert (output / "reload_check.json").exists()


def test_laco_file_presence_probe_does_not_close_method():
    from workflows.audit import check_keshu_completion

    method = {
        "structural_or_primary": False,
        "laco_core_smoke": False,
        "laco_file_probe_only": True,
    }
    assert check_keshu_completion.laco_closed(method) is False
