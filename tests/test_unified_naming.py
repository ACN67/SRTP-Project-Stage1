import re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["README.md","data","methods","workflows","results","docs","environment","tests"]
FORBIDDEN = re.compile(r"\bPan\b|pan_|pan formal|pan auxiliary|wsl_local|WSL worktree|常珂舒侧|潘阔分支|source branch|merge provenance|_original|pre-stage1-restructure", re.I)
PERSON = re.compile(r"常珂舒|潘阔|李长骏")
ALLOWED_PERSON_MARKERS = ("Owner:", ",owner,", "owner:", "owner")

def iter_active_files():
    for target in TARGETS:
        base = ROOT / target
        if not base.exists():
            continue
        paths = [base] if base.is_file() else base.rglob("*")
        for p in paths:
            if not p.is_file() or ".git" in p.parts or "evidence" in p.parts:
                continue
            if p.suffix.lower() in {".pyc", ".png", ".jpg", ".jpeg", ".pdf"}:
                continue
            yield p

def test_no_branch_or_source_terms_in_active_files():
    offenders=[]
    for p in iter_active_files():
        text=p.read_text(encoding="utf-8", errors="ignore")
        if FORBIDDEN.search(p.as_posix()) or FORBIDDEN.search(text):
            offenders.append(p.relative_to(ROOT).as_posix())
    assert not offenders

def test_member_names_only_as_owner_fields():
    bad=[]
    for p in iter_active_files():
        rel=p.relative_to(ROOT).as_posix()
        for i,line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(),1):
            if PERSON.search(line) and not any(marker in line for marker in ALLOWED_PERSON_MARKERS):
                bad.append(f"{rel}:{i}:{line[:80]}")
    assert not bad

