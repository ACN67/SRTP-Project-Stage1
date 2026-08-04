from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_no_generated_caches_or_obsolete_patches():
    bad=[]
    for p in ROOT.rglob("*"):
        if ".git" in p.parts or "third_party" in p.parts or p.name == ".pytest_cache" or ("tests" in p.parts and "__pycache__" in p.parts): continue
        if p.name in {".pytest_cache","__pycache__"} or p.suffix == ".pyc": bad.append(p.relative_to(ROOT).as_posix())
        if p.name == "wsl"+"_local.patch" or (p.suffix == ".patch" and p.stat().st_size == 0): bad.append(p.relative_to(ROOT).as_posix())
    assert not bad
    assert not (ROOT/"environment/common/common").exists()
    assert not (ROOT/"environment/snapshots").exists()
