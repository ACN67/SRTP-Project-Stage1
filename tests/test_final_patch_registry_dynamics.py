from __future__ import annotations

import csv
from pathlib import Path

from workflows.aggregate import registry_utils


def test_display_path_handles_external_output(tmp_path: Path):
    assert registry_utils.display_path(tmp_path / "x.csv").startswith(str(tmp_path))


def test_run_registry_infers_tinyllama_and_variants():
    rows = registry_utils.build_run_rows()
    tiny = [r for r in rows if "tiny_llama" in r["run_id"]]
    assert tiny
    assert all("TinyLlama" in r["model"] for r in tiny)
    assert any(r["variant"] != "unknown" for r in rows)
    assert any(r["variant"] == "benchmark_guided_sliced_model" for r in rows if "slicegpt_codellama7b_r4_benchguided_evalhalf" in r["run_id"])


def test_method_rows_are_derived_from_runs(monkeypatch):
    base_runs = registry_utils.build_run_rows()
    base_methods = {row["method"]: row for row in registry_utils.build_method_rows(base_runs)}
    extra = dict(base_runs[0])
    extra.update({"run_id": "sparsegpt_new_success", "method_scope": "SparseGPT", "execution_status": "completed", "validity_status": "valid", "result_completeness": "complete", "protocol": "r4_half"})
    new_methods = {row["method"]: row for row in registry_utils.build_method_rows(base_runs + [extra])}
    assert base_methods["SparseGPT"]["execution_status"] != new_methods["SparseGPT"]["execution_status"]


def test_auxiliary_comparison_builder_can_materialize_output(tmp_path: Path):
    from workflows.aggregate import build_auxiliary_comparison

    output = tmp_path / "comparison.csv"
    assert build_auxiliary_comparison.main(["--write", "--output", str(output)]) == 0
    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert rows
    assert {row["evidence_status"] for row in rows} == {"aggregate_only"}
