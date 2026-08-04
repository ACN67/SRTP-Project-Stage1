from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "flab_qwen15b_benchmark_guided_smoke_20260804_135032",
    "flab_qwen15b_benchmark_guided_he_keep80_capped32_20260804_135032",
    "flab_qwen15b_benchmark_guided_mbpp_keep80_capped32_20260804_135032",
    "flab_qwen15b_benchmark_guided_lcb_keep80_capped32_20260804_135032",
}


def rows(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_flab_artifacts_are_registered_with_real_availability() -> None:
    found = [row for row in rows("results/status/artifacts.csv") if row["run_id"] in RUNS]
    assert {row["run_id"] for row in found} == RUNS
    for row in found:
        assert row["availability"] in {"available_archived", "missing_after_ephemeral_tmp_cleanup"}
        assert "$SRTP_ARTIFACT_ROOT" not in row["persistent_path"]
        assert row["committed_to_git"] == "false"
        if row["availability"] == "available_archived":
            assert row["sha256"]
            assert int(row["size_bytes"]) > 0
        else:
            assert row["sha256"] == ""
            assert row["size_bytes"] == ""
