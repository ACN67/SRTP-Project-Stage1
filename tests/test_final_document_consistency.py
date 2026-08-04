from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_current_docs_do_not_leave_unqualified_flab_benchmark_blocker() -> None:
    text = "\n".join(
        p.read_text(encoding="utf-8")
        for p in [
            ROOT / "README.md",
            ROOT / "methods/flab_pruner/README.md",
            ROOT / "results/reports/reproduction_status.md",
            ROOT / "results/reports/failure_audit.md",
            ROOT / "results/reports/protocol_deviations.md",
        ]
    ).lower()
    assert "benchmark-guided flab remains blocked" not in text
    assert "activation path not implemented" not in text
    assert "no true pruning occurred" not in text
    assert "superseded" in text
    assert "output collapse" in text
