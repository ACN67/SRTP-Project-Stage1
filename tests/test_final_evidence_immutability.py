from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_final_cleanup_only_adds_archive_evidence_for_cleanup() -> None:
    archives = sorted((ROOT / "results/evidence/diagnostics").glob("flab_artifact_archive_*/summary.json"))
    assert archives
    summary = archives[-1].read_text(encoding="utf-8")
    assert "artifact_archive_completed" in summary
    assert "old_evidence_modified" in summary
