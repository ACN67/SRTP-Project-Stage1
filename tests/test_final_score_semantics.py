from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPPED = {
    "flab_qwen15b_benchmark_guided_he_keep80_capped32_20260804_135032",
    "flab_qwen15b_benchmark_guided_mbpp_keep80_capped32_20260804_135032",
    "flab_qwen15b_benchmark_guided_lcb_keep80_capped32_20260804_135032",
}


def read_rows(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_capped_pilot_has_no_formal_or_pass_count_score_rows() -> None:
    assert not [row for row in read_rows("results/status/scores.csv") if row["run_id"] in CAPPED]
    assert not [row for row in read_rows("results/formal/r4_half/scores.csv") if row["run_id"] in CAPPED]
