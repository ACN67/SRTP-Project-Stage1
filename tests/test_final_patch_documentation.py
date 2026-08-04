from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_and_failure_audit_are_actionable():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for phrase in ["Project goal", "Quick navigation", "Data protocol", "Method status", "Execution entry points", "Formal results", "Known failures", "Open items"]:
        assert phrase in readme
    methods = (ROOT / "methods/README.md").read_text(encoding="utf-8")
    assert "results/status/methods.csv" in methods
    failure = (ROOT / "results/reports/failure_audit.md").read_text(encoding="utf-8")
    for ref in ["results/status/runs.csv", "results/status/scores.csv", "results/formal/r4_half/scores.csv", "results/status/completion_audit.json"]:
        assert ref in failure
