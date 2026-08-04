from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def test_methods_registry_has_twelve_methods() -> None:
    rows = list(csv.DictReader((ROOT / "results/registries/methods.csv").open(encoding="utf-8-sig")))
    assert len(rows) == 12
    assert {row["method"] for row in rows} == {
        "Flab-Pruner",
        "LLM-Pruner",
        "SliceGPT",
        "LaCo",
        "Magnitude",
        "Wanda",
        "DSnoT",
        "OWL",
        "SparseGPT",
        "MaskLLM",
        "Pruner-Zero",
        "FLAP",
    }


def test_runs_registry_matches_evidence_directories() -> None:
    expected = {
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in (ROOT / "results/evidence").glob("*/*")
        if path.is_dir()
    }
    rows = list(csv.DictReader((ROOT / "results/registries/runs.csv").open(encoding="utf-8-sig")))
    assert {row["path"] for row in rows} == expected


def test_r4_splits_do_not_overlap() -> None:
    for bench in ["humaneval", "mbpp_evalplus", "livecodebench"]:
        base = ROOT / "data/benchmarks/r4_half" / bench
        guide_ids = {row["task_id"] for row in read_jsonl(base / "guide.jsonl")}
        eval_ids = {row["task_id"] for row in read_jsonl(base / "eval.jsonl")}
        assert guide_ids.isdisjoint(eval_ids), bench


def test_structured_files_parse() -> None:
    for path in ROOT.rglob("*.json"):
        if ".git" not in path.parts:
            json.loads(path.read_text(encoding="utf-8"))
    for path in ROOT.rglob("*.jsonl"):
        if ".git" not in path.parts:
            read_jsonl(path)
    for path in ROOT.rglob("*.csv"):
        if ".git" not in path.parts:
            rows = list(csv.reader(path.open(encoding="utf-8-sig")))
            if rows:
                width = len(rows[0])
                assert all(len(row) == width for row in rows), path


def test_no_legacy_active_paths() -> None:
    forbidden = [
        "scripts" + "/",
        "configs" + "/experiments",
        "results" + "/raw",
        "results" + "/stage1",
        "results" + "/tables",
        "reports" + "/stage1",
        "/home/" + "keshu",
        "/home/" + "xaillor",
        "/mnt/c/Users/" + "Xile",
    ]
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith("results/evidence/"):
            continue
        if rel.startswith("results/auxiliary/pan_full_eval/protocol_snapshot/"):
            continue
        if rel == "results/reports/merge_provenance.md":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert not any(token in text for token in forbidden), rel
