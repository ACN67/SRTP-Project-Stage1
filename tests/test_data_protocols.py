import csv, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def test_data_split_registry_hash_counts_and_overlap():
    rows=list(csv.DictReader((ROOT/"results/status/data_splits.csv").open(encoding="utf-8-sig")))
    assert {r["protocol"] for r in rows} >= {"smoke","r4_half","auxiliary_full_eval"}
    for r in rows:
        path=ROOT/r["path"]
        assert path.is_file(), r["path"]
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert r["sha256"] == digest
        if path.suffix == ".jsonl":
            assert int(r["task_count"]) == len(read_jsonl(path))
    for protocol in ["smoke","r4_half"]:
        for dataset in {r["dataset"] for r in rows if r["protocol"] == protocol}:
            by_role={r["role"]: r for r in rows if r["protocol"] == protocol and r["dataset"] == dataset}
            if "guide" in by_role and "eval" in by_role:
                gids={x.get("task_id") for x in read_jsonl(ROOT/by_role["guide"]["path"])}
                eids={x.get("task_id") for x in read_jsonl(ROOT/by_role["eval"]["path"])}
                assert gids.isdisjoint(eids), (protocol,dataset)

