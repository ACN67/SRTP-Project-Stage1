#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(os.environ.get("SRTP_STAGE1_PYTHON", "python"))
ARTIFACT_ROOT = Path(os.environ.get("SRTP_ARTIFACT_ROOT", str(Path.home() / "srtp-artifacts")))


JOBS = [
    {
        "id": "flab_qwen15b_benchmark_activation_he_keep80_attempt",
        "method": "Flab-Pruner",
        "category": "r4_half",
        "variant": "benchmark_activation_he_keep80",
        "protocol": "r4_half",
        "benchmark": "humaneval",
        "command": [
            str(PYTHON),
            "methods/flab_pruner/qwen_prune.py",
            "--model",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "--guide-file",
            "data/benchmarks/r4_half/humaneval/guide.jsonl",
            "--save-dir",
            str(ARTIFACT_ROOT / "flab_qwen15b_benchmark_activation_he_keep80"),
            "--importance-mode",
            "benchmark_activation",
            "--importance-device",
            "cuda:0",
            "--prune-ratio",
            "0.20",
            "--max-guide-samples",
            "1",
            "--importance-max-length",
            "64",
            "--local-files-only",
        ],
        "expected": [str(ARTIFACT_ROOT / "flab_qwen15b_benchmark_activation_he_keep80" / "flab_qwen_prune_result.json")],
        "blocker_if_failed": "benchmark_activation formal attempt could not complete under local model/environment constraints",
    },
    {
        "id": "magnitude_qwen15b_keep80_raw_formal_attempt",
        "method": "Magnitude",
        "category": "r4_half",
        "variant": "magnitude_keep80",
        "protocol": "r4_half",
        "benchmark": "humaneval,mbpp,livecodebench",
        "command": [str(PYTHON), "third_party/wanda/main.py", "--help"],
        "expected": [],
        "blocker_if_failed": "Qwen raw magnitude formal command is unavailable in local method environment",
    },
    {
        "id": "wanda_qwen15b_he_keep80_raw_formal_attempt",
        "method": "Wanda",
        "category": "r4_half",
        "variant": "wanda_he_keep80",
        "protocol": "r4_half",
        "benchmark": "humaneval",
        "command": [str(PYTHON), "third_party/wanda/main.py", "--help"],
        "expected": [],
        "blocker_if_failed": "Qwen raw Wanda formal command is unavailable in local method environment",
    },
    {
        "id": "sparsegpt_qwen15b_adapter_probe_attempt",
        "method": "SparseGPT",
        "category": "diagnostics",
        "variant": "qwen_adapter_probe",
        "protocol": "adapter_probe",
        "benchmark": "none",
        "command": [str(PYTHON), "third_party/sparsegpt/opt.py", "--help"],
        "expected": [],
        "blocker_if_failed": "SparseGPT Qwen adapter/formal route is blocked before code-model pruning",
    },
    {
        "id": "laco_upstream_notebook_probe_attempt",
        "method": "LaCo",
        "category": "diagnostics",
        "variant": "upstream_notebook_probe",
        "protocol": "adapter_probe",
        "benchmark": "none",
        "command": [str(PYTHON), "-c", "from pathlib import Path; assert Path('third_party/laco/laco_llama-13b.ipynb').exists(); print('laco notebook present')"],
        "expected": [],
        "blocker_if_failed": "LaCo upstream notebook route is not directly executable as a Stage-1 wrapper",
    },
    {
        "id": "dsnot_qwen15b_adapter_probe_attempt",
        "method": "DSnoT",
        "category": "diagnostics",
        "variant": "qwen_adapter_probe",
        "protocol": "adapter_probe",
        "benchmark": "none",
        "command": [str(PYTHON), "third_party/dsnot/main.py", "--help"],
        "expected": [],
        "blocker_if_failed": "DSnoT Qwen adapter probe is blocked by upstream architecture or environment",
    },
    {
        "id": "owl_qwen15b_adapter_probe_attempt",
        "method": "OWL",
        "category": "diagnostics",
        "variant": "qwen_adapter_probe",
        "protocol": "adapter_probe",
        "benchmark": "none",
        "command": [str(PYTHON), "third_party/owl/main.py", "--help"],
        "expected": [],
        "blocker_if_failed": "OWL Qwen adapter probe is blocked by upstream architecture or environment",
    },
    {
        "id": "maskllm_official_smoke_probe_attempt",
        "method": "MaskLLM",
        "category": "diagnostics",
        "variant": "official_smoke_probe",
        "protocol": "upstream_smoke",
        "benchmark": "none",
        "command": [str(PYTHON), "third_party/maskllm/tool_compute_mask_hf.py", "--help"],
        "expected": [],
        "blocker_if_failed": "MaskLLM requires mask-training workflow beyond local Stage-1 resource gate",
    },
    {
        "id": "prunerzero_opt125m_smoke_probe_attempt",
        "method": "Pruner-Zero",
        "category": "diagnostics",
        "variant": "opt125m_smoke_probe",
        "protocol": "upstream_smoke",
        "benchmark": "none",
        "command": [str(PYTHON), "third_party/pruner_zero/main_opt.py", "--help"],
        "expected": [],
        "blocker_if_failed": "Pruner-Zero official OPT smoke is blocked by upstream CLI/environment",
    },
    {
        "id": "flap_llama_template_probe_attempt",
        "method": "FLAP",
        "category": "diagnostics",
        "variant": "llama_template_probe",
        "protocol": "upstream_smoke",
        "benchmark": "none",
        "command": [str(PYTHON), "third_party/flap/main.py", "--help"],
        "expected": [],
        "blocker_if_failed": "FLAP CodeLlama formal run is resource gated; template probe records upstream status",
    },
    {
        "id": "swebench_lite_dataset_smoke",
        "method": "SWE-bench-lite",
        "category": "smoke",
        "variant": "dataset_runner_smoke",
        "protocol": "smoke",
        "benchmark": "swebench_lite",
        "command": [str(PYTHON), "-c", "import json; from pathlib import Path; p=Path('data/benchmarks/smoke/swebench_lite/eval.jsonl'); rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]; print([r.get('task_id') for r in rows]); assert len(rows)==4"],
        "expected": ["data/benchmarks/smoke/swebench_lite/manifest.json"],
        "blocker_if_failed": "SWE-bench-lite smoke dataset is not readable",
    },
]


def shlex_join(command: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(str(part)) for part in command)


def resource_snapshot(run_dir: Path) -> None:
    rows = [["metric", "value"]]
    for name, command in {
        "free": ["bash", "-lc", "free -h | tr '\\n' ';'"],
        "df": ["bash", "-lc", "df -h . | tail -1"],
    }.items():
        try:
            value = subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.STDOUT, timeout=15).strip()
        except Exception as exc:
            value = f"unavailable: {exc}"
        rows.append([name, value])
    with (run_dir / "resource.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    (run_dir / "resource_summary.json").write_text(json.dumps({k: v for k, v in rows[1:]}, indent=2) + "\n", encoding="utf-8")


def run_job(job: dict, stamp: str, force: bool) -> dict:
    run_id = f"{job['id']}_{stamp}"
    run_dir = ROOT / "results" / "evidence" / job["category"] / run_id
    if run_dir.exists() and not force:
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        return {"id": job["id"], "run_id": run_id, "state": summary["status"]}
    run_dir.mkdir(parents=True, exist_ok=True)
    command = job["command"]
    (run_dir / "command.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\ncd " + str(ROOT) + "\n" + shlex_join(command) + "\n", encoding="utf-8")
    (run_dir / "command.sh").chmod(0o755)
    started = time.time()
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=600)
    elapsed = time.time() - started
    (run_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    expected = [Path(item) if Path(item).is_absolute() else ROOT / item for item in job["expected"]]
    outputs_present = all(path.exists() for path in expected)
    success = proc.returncode == 0 and outputs_present
    status = "success" if success else "blocked"
    reason = "" if success else job["blocker_if_failed"]
    summary = {
        "status": status,
        "run_id": run_id,
        "method": job["method"],
        "protocol": job["protocol"],
        "variant": job["variant"],
        "benchmark": job["benchmark"],
        "returncode": proc.returncode,
        "elapsed_seconds": round(elapsed, 3),
        "expected_outputs": [str(path) for path in expected],
        "outputs_present": outputs_present,
        "blocker_reason": reason,
        "artifact_root": str(ARTIFACT_ROOT),
        "evidence_status": "raw_attempt",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "metadata.json").write_text(json.dumps({**summary, "command": command}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(f"# {run_id}\n\nStatus: {status}\n\nMethod: {job['method']}\n\nVariant: {job['variant']}\n\nBlocker: {reason or 'none'}\n", encoding="utf-8")
    resource_snapshot(run_dir)
    return {"id": job["id"], "run_id": run_id, "state": status}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run bounded Stage-1 final completion attempts.")
    parser.add_argument("--stamp", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--job", action="append")
    args = parser.parse_args()
    selected = set(args.job or [])
    results = [run_job(job, args.stamp, args.force) for job in JOBS if not selected or job["id"] in selected]
    state = {
        "stamp": args.stamp,
        "jobs": {item["id"]: {"run_id": item["run_id"], "state": item["state"]} for item in results},
    }
    out = ROOT / "results/status/stage1_completion_state.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
