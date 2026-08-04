from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path

from environment.setup import capture_environment_locks as locks


def make_fake_venv(root: Path, name: str, freeze_text: str) -> None:
    bin_dir = root / name / "bin"
    bin_dir.mkdir(parents=True)
    py = bin_dir / "python"
    py.write_text(
        "#!/usr/bin/env sh\n"
        "if [ \"$1\" = \"-m\" ] && [ \"$2\" = \"pip\" ] && [ \"$3\" = \"freeze\" ]; then\n"
        f"  printf '%s\\n' '{freeze_text}'\n"
        "else\n"
        "  exit 1\n"
        "fi\n",
        encoding="utf-8",
    )
    py.chmod(py.stat().st_mode | 0o111)


def read_map(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_distinct_freezes_get_distinct_locks_and_write_is_idempotent(tmp_path: Path):
    venv_root = tmp_path / "venvs"
    make_fake_venv(venv_root, "venv-a", "a==1")
    make_fake_venv(venv_root, "venv-b", "b==2")
    make_fake_venv(venv_root, "venv-c", "a==1")
    method_map = tmp_path / "method_env_map.csv"
    method_map.write_text(
        "method,venv_name,lock_file,extra_install,notes\n"
        "a,venv-a,,,\n"
        "b,venv-b,,,\n"
        "c,venv-c,,,\n",
        encoding="utf-8",
    )
    out_map = tmp_path / "out_map.csv"
    lock_root = tmp_path / "locks"

    assert locks.main(["--write", "--venv-root", str(venv_root), "--method-map", str(method_map), "--output-map", str(out_map), "--lock-root", str(lock_root)]) == 0
    rows = {row["method"]: row for row in read_map(out_map)}
    assert rows["a"]["lock_file"] == rows["c"]["lock_file"]
    assert rows["b"]["lock_file"] != rows["a"]["lock_file"]
    lock_hashes_before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in lock_root.glob("*.txt")}
    assert locks.main(["--check", "--venv-root", str(venv_root), "--method-map", str(out_map), "--lock-root", str(lock_root)]) == 0
    assert locks.main(["--write", "--venv-root", str(venv_root), "--method-map", str(out_map), "--output-map", str(out_map), "--lock-root", str(lock_root)]) == 0
    assert lock_hashes_before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in lock_root.glob("*.txt")}
