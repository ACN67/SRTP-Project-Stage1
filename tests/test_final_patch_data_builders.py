from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run_builder(script: str, output_root: Path):
    return subprocess.run([PY, str(ROOT / script), "--output-root", str(output_root)], cwd=ROOT, text=True, capture_output=True)


def assert_manifest_has_local_paths(manifest: Path, output_root: Path) -> None:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    for key in ["guide_path", "eval_path", "heldout_eval_path"]:
        if key in data:
            path = Path(data[key])
            assert str(output_root) in str(path) or path.exists()


def test_data_builders_materialize_external_output(tmp_path: Path):
    r4 = tmp_path / "r4"
    aux = tmp_path / "aux"
    mbpp = tmp_path / "mbpp_evalplus"
    for script, out in [
        ("workflows/data/build_r4_half_splits.py", r4),
        ("workflows/data/build_auxiliary_full_splits.py", aux),
        ("workflows/data/build_mbpp_evalplus.py", mbpp),
    ]:
        result = run_builder(script, out)
        assert result.returncode == 0, result.stderr
        assert list(out.rglob("manifest.json"))
        for manifest in out.rglob("manifest.json"):
            assert_manifest_has_local_paths(manifest, out)


def test_data_builder_dry_run_reports_target_paths(tmp_path: Path):
    out = tmp_path / "dry"
    result = subprocess.run([PY, str(ROOT / "workflows/data/build_r4_half_splits.py"), "--dry-run", "--output-root", str(out)], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0
    assert str(out) in result.stdout
    assert "data/benchmarks/r4_half" not in result.stdout
