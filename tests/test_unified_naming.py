import csv, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["README.md","data","methods","workflows","results","docs","environment"]
def s(*codes): return "".join(chr(c) for c in codes)
TERMS = [r"\b"+s(80,97,110)+r"\b", s(112,97,110,95), s(112,97,110,32,102,111,114,109,97,108), s(112,97,110,32,97,117,120,105,108,105,97,114,121), s(119,115,108,95,108,111,99,97,108), s(87,83,76,32,119,111,114,107,116,114,101,101), s(24120,29634,33298,20391), s(28504,38420,20998,25903), s(115,111,117,114,99,101,32,98,114,97,110,99,104), s(109,101,114,103,101,32,112,114,111,118,101,110,97,110,99,101), s(95,111,114,105,103,105,110,97,108), s(112,114,101,45,115,116,97,103,101,49,45,114,101,115,116,114,117,99,116,117,114,101)]
FORBIDDEN = re.compile("|".join(TERMS), re.I)
PERSON = re.compile(r"常珂舒|潘阔|李长骏")
def iter_active_files():
    for target in TARGETS:
        base = ROOT / target
        if not base.exists(): continue
        paths = [base] if base.is_file() else base.rglob("*")
        for p in paths:
            if not p.is_file() or ".git" in p.parts or "evidence" in p.parts: continue
            if p.suffix.lower() in {".pyc", ".png", ".jpg", ".jpeg", ".pdf"}: continue
            yield p
def test_no_branch_or_source_terms_in_active_files():
    offenders=[]
    for p in iter_active_files():
        text=p.read_text(encoding="utf-8", errors="ignore")
        if FORBIDDEN.search(p.as_posix()) or FORBIDDEN.search(text): offenders.append(p.relative_to(ROOT).as_posix())
    assert not offenders
def csv_has_owner_column(p: Path) -> bool:
    if p.suffix.lower() != '.csv': return False
    with p.open(encoding='utf-8-sig', newline='') as f:
        header=next(csv.reader(f), [])
    return 'owner' in header
def test_member_names_only_as_owner_fields():
    bad=[]
    for p in iter_active_files():
        if csv_has_owner_column(p): continue
        rel=p.relative_to(ROOT).as_posix()
        for i,line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(),1):
            if PERSON.search(line) and not (line.startswith('Owner:') or 'owner:' in line): bad.append(f"{rel}:{i}:{line[:80]}")
    assert not bad
