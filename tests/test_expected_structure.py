from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
REQUIRED_TOP = {"data","methods","workflows","results","docs","environment","tests","third_party"}
def j(*parts): return "/".join(parts)
FORBIDDEN = [j("docs","legacy"), j("results","registries"), j("results","auxiliary","pan"+"_full_eval"), j("workflows","evaluate","legacy"), j("environment","snapshots"), j("environment","common","common"), j("data","audit"), j("results","reports","merge"+"_provenance.md")]
DOCS = {"stage1_protocol.md","benchmark_protocol.md","recovery_protocol.md","runbook.md","environment.md"}
STATUS = {"methods.csv","runs.csv","scores.csv","artifacts.csv","data_splits.csv","stage1_summary.md"}
AGG = {"build_run_registry.py","build_score_registry.py","build_method_status.py","build_data_split_registry.py","build_auxiliary_comparison.py"}
def test_required_layout_and_no_forbidden_paths():
    assert REQUIRED_TOP <= {p.name for p in ROOT.iterdir() if p.is_dir()}
    for rel in FORBIDDEN: assert not (ROOT / rel).exists(), rel
    assert {p.name for p in (ROOT/"docs").iterdir() if p.is_file()} == DOCS
    assert STATUS <= {p.name for p in (ROOT/"results/status").iterdir() if p.is_file()}
    assert AGG <= {p.name for p in (ROOT/"workflows/aggregate").iterdir() if p.is_file()}
    assert (ROOT/"results/formal/r4_half/README.md").is_file()
    assert (ROOT/"results/auxiliary/full_eval/README.md").is_file()
