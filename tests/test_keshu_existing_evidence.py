from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_llmpruner_and_slicegpt_primary_evidence_audits_exist():
    for prefix in ["llmpruner_primary_evidence_audit_", "slicegpt_primary_evidence_audit_"]:
        assert sorted((ROOT / "results/evidence/diagnostics").glob(prefix + "*/summary.json"))


def test_non_keshu_methods_not_planned_by_keshu_scope():
    rows = list(csv.DictReader((ROOT / "results/status/methods.csv").open(encoding="utf-8-sig", newline="")))
    non = {"Magnitude", "Wanda", "DSnoT", "OWL", "SparseGPT", "MaskLLM", "Pruner-Zero", "FLAP"}
    assert {r["method"] for r in rows if r["method"] in non} == non
