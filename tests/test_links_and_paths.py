import re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def j(*parts): return "/".join(parts)
OLD_PARTS=[j("data","splits"),j("results","raw"),j("results","stage1"),j("results","tables"),j("results","registries"),j("results","auxiliary","pan"+"_full_eval"),j("reports","stage1"),"scr"+"ipts/",j("configs","experiments")]
OLD=re.compile("|".join(re.escape(x) for x in OLD_PARTS))
ABS=re.compile(r"/home/(keshu|xaillor)|/mnt/c/Users/|C:\\Users\\")
LINK=re.compile(r"\[[^\]]+\]\((?!https?://|mailto:|#)([^)]+)\)")
def text_files():
    for base in [ROOT/"README.md",ROOT/"data",ROOT/"methods",ROOT/"workflows",ROOT/"results",ROOT/"docs",ROOT/"environment"]:
        paths=[base] if base.is_file() else base.rglob("*")
        for p in paths:
            if p.is_file() and ".git" not in p.parts and "evidence" not in p.parts and p.suffix.lower() not in {".png",".jpg",".jpeg",".pyc"}: yield p
def test_no_old_paths_abs_paths_or_control_chars():
    bad=[]; allowed={chr(10),chr(13),chr(9)}
    for p in text_files():
        text=p.read_text(encoding='utf-8',errors='ignore')
        if OLD.search(text) or ABS.search(text): bad.append(p.relative_to(ROOT).as_posix())
        if any(ord(ch)<32 and ch not in allowed for ch in text): bad.append(p.relative_to(ROOT).as_posix()+':control')
    assert not bad
def test_markdown_local_links_resolve():
    broken=[]
    for p in text_files():
        if p.suffix.lower() != ".md": continue
        for m in LINK.finditer(p.read_text(encoding="utf-8", errors="ignore")):
            dest=m.group(1).split("#",1)[0].strip()
            if dest and not dest.startswith("<") and not (p.parent/dest).resolve().exists(): broken.append(f"{p.relative_to(ROOT)} -> {dest}")
    assert not broken
