import ast, subprocess, sys, tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_pyproject_pythonpath_and_python_ast():
    data=tomllib.loads((ROOT/"pyproject.toml").read_text(encoding="utf-8"))
    assert data["tool"]["pytest"]["ini_options"]["pythonpath"] == ["."]
    for base in [ROOT/"methods", ROOT/"workflows", ROOT/"tests"]:
        for p in base.rglob("*.py"):
            if "__pycache__" not in p.parts:
                ast.parse(p.read_text(encoding="utf-8"), filename=str(p))

def test_activity_cli_help_and_new_paths():
    scripts=["workflows/experiment/run_plan.py","workflows/audit/check_stage1_completion.py","workflows/data/validate_splits.py","workflows/aggregate/build_run_registry.py","workflows/aggregate/build_score_registry.py","workflows/aggregate/build_method_status.py","workflows/aggregate/build_data_split_registry.py","workflows/aggregate/build_auxiliary_comparison.py"]
    for rel in scripts:
        res=subprocess.run([sys.executable, str(ROOT/rel), "--help"], text=True, capture_output=True, timeout=10)
        assert res.returncode == 0, (rel, res.stderr)
    assert "results/stage1" not in (ROOT/"workflows/experiment/run_plan.py").read_text(encoding="utf-8")
    assert "evidence_category" in (ROOT/"workflows/experiment/run_plan.py").read_text(encoding="utf-8")
    assert not (ROOT/"workflows/evaluate/legacy").exists()

def test_shell_scripts_parse():
    for p in list((ROOT/"workflows").rglob("*.sh")) + list((ROOT/"environment/setup").rglob("*.sh")):
        res=subprocess.run(["bash","-n",str(p)], text=True, capture_output=True)
        assert res.returncode == 0, (p, res.stderr)
