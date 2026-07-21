from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shlex
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = ROOT / "reports" / "stage1" / "manual_run_results.jsonl"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def sanitize_id(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    return "".join(ch if ch in allowed else "_" for ch in value)


def command_prefix(env_name: str | None, cwd: Path, command: str) -> str:
    parts = [
        "set -euo pipefail",
        f"cd {shlex.quote(str(ROOT))}",
        "source scripts/setup/env.sh",
    ]
    if env_name:
        parts.append(f"source .venv-{shlex.quote(env_name)}/bin/activate")
    parts.append(f"cd {shlex.quote(str(cwd))}")
    parts.append(command)
    return " && ".join(parts)


def nvidia_memory_mb() -> int | None:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            timeout=5,
        )
        if proc.returncode != 0:
            return None
        values = [int(x.strip()) for x in proc.stdout.splitlines() if x.strip().isdigit()]
        return max(values) if values else None
    except Exception:
        return None


def process_rss_mb(proc: psutil.Process) -> float:
    total = 0
    try:
        procs = [proc] + proc.children(recursive=True)
        for item in procs:
            try:
                total += item.memory_info().rss
            except psutil.Error:
                pass
    except psutil.Error:
        pass
    return total / 1024 / 1024


def monitor_resources(pid: int, csv_path: Path, stop: threading.Event, interval: float) -> None:
    proc = psutil.Process(pid)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "elapsed_sec", "process_rss_mb", "gpu_memory_used_mb"],
        )
        writer.writeheader()
        start = time.monotonic()
        while not stop.is_set():
            writer.writerow(
                {
                    "timestamp": now_iso(),
                    "elapsed_sec": round(time.monotonic() - start, 3),
                    "process_rss_mb": round(process_rss_mb(proc), 2),
                    "gpu_memory_used_mb": nvidia_memory_mb(),
                }
            )
            f.flush()
            stop.wait(interval)


def summarize_resource_csv(path: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    if not path.exists():
        return {
            "status": "missing",
            "source": str(path),
            "sample_count": 0,
        }
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    def parse_float(value: str | None) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except ValueError:
            return None

    process_values = [v for row in rows if (v := parse_float(row.get("process_rss_mb"))) is not None]
    gpu_values = [v for row in rows if (v := parse_float(row.get("gpu_memory_used_mb"))) is not None]
    elapsed_values = [v for row in rows if (v := parse_float(row.get("elapsed_sec"))) is not None]

    return {
        "status": "success",
        "source": str(path),
        "sample_count": len(rows),
        "duration_observed_sec": max(elapsed_values) if elapsed_values else None,
        "peak_process_rss_mb": max(process_values) if process_values else None,
        "mean_process_rss_mb": sum(process_values) / len(process_values) if process_values else None,
        "peak_gpu_memory_used_mb": max(gpu_values) if gpu_values else None,
        "mean_gpu_memory_used_mb": sum(gpu_values) / len(gpu_values) if gpu_values else None,
        "first_timestamp": rows[0].get("timestamp") if rows else None,
        "last_timestamp": rows[-1].get("timestamp") if rows else None,
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_markdown_summary(jsonl_path: Path, md_path: Path) -> None:
    rows = []
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]

    lines = [
        "# Manual Run Results",
        "",
        f"Source: `{jsonl_path.relative_to(ROOT)}`",
        "",
        "| time | owner | group | job | status | seconds | returncode | run_dir |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        run_dir = row.get("run_dir", "")
        try:
            run_dir = str(Path(run_dir).relative_to(ROOT))
        except Exception:
            pass
        lines.append(
            "| {time} | {owner} | {group} | `{job}` | {status} | {sec} | {rc} | `{run_dir}` |".format(
                time=row.get("end_time", ""),
                owner=row.get("owner", ""),
                group=row.get("method_group", ""),
                job=row.get("job_id", ""),
                status=row.get("status", ""),
                sec=row.get("duration_sec", ""),
                rc=row.get("returncode", ""),
                run_dir=run_dir,
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_job(job: dict[str, Any], plan: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    job_id = sanitize_id(job["id"])
    run_id = sanitize_id(job.get("run_id") or f"{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    run_dir = ROOT / job.get("run_dir_template", f"results/raw/{run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)

    cwd = ROOT / job.get("cwd", ".")
    env_name = job.get("env")
    command = job["command"]
    timeout = int(job.get("timeout_seconds", plan.get("default_timeout_seconds", 3600)))
    poll_interval = float(plan.get("resource_poll_interval_seconds", 5))

    effective_command = command_prefix(env_name, cwd, command)
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in (plan.get("env") or {}).items()})
    env.update({str(k): str(v) for k, v in (job.get("env_vars") or {}).items()})
    env["RUN_ID"] = run_id
    env["RUN_DIR"] = str(run_dir)
    env["PROJECT_ROOT"] = str(ROOT)

    metadata = {
        "job_id": job_id,
        "run_id": run_id,
        "description": job.get("description", ""),
        "tags": job.get("tags", []),
        "owner": job.get("owner", "unassigned"),
        "method_group": job.get("method_group", ""),
        "method": job.get("method", ""),
        "role": job.get("role", ""),
        "recommended_machine": job.get("recommended_machine", ""),
        "cross_reproduction_by": job.get("cross_reproduction_by", ""),
        "heavy": bool(job.get("heavy", False)),
        "start_time": now_iso(),
        "cwd": str(cwd),
        "env": env_name,
        "command": command,
        "effective_command": effective_command,
        "run_dir": str(run_dir),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "command.sh").write_text(effective_command + "\n", encoding="utf-8")

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    resource_path = run_dir / "resource.csv"

    start = time.monotonic()
    stop = threading.Event()
    status = "failed"
    returncode = None
    timeout_hit = False

    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.Popen(
            ["/bin/bash", "-lc", effective_command],
            cwd=ROOT,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        monitor = threading.Thread(
            target=monitor_resources,
            args=(proc.pid, resource_path, stop, poll_interval),
            daemon=True,
        )
        monitor.start()
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timeout_hit = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                returncode = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                returncode = proc.wait()
        finally:
            stop.set()
            monitor.join(timeout=10)

    duration = round(time.monotonic() - start, 3)
    if timeout_hit:
        status = "timeout"
    elif returncode == 0:
        status = "success"

    summary = {
        **metadata,
        "end_time": now_iso(),
        "duration_sec": duration,
        "returncode": returncode,
        "status": status,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "resource_csv": str(resource_path),
        "resource_summary": str(run_dir / "resource_summary.json"),
    }
    resource_summary = summarize_resource_csv(resource_path)
    (run_dir / "resource_summary.json").write_text(
        json.dumps(resource_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary.update(
        {
            "peak_process_rss_mb": resource_summary.get("peak_process_rss_mb"),
            "peak_gpu_memory_used_mb": resource_summary.get("peak_gpu_memory_used_mb"),
        }
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    append_jsonl(summary_path, summary)
    write_markdown_summary(summary_path, summary_path.with_suffix(".md"))
    return summary


def select_jobs(plan: dict[str, Any], only: set[str], include_disabled: bool, list_jobs: bool) -> list[dict[str, Any]]:
    jobs = plan.get("jobs") or []
    if list_jobs:
        for job in jobs:
            enabled = bool(job.get("enabled", False))
            heavy = bool(job.get("heavy", False))
            print(f"{job['id']}\towner={job.get('owner', '')}\tgroup={job.get('method_group', '')}\tenabled={enabled}\theavy={heavy}\t{job.get('description', '')}")
        return []
    selected = []
    for job in jobs:
        if only and job["id"] not in only:
            continue
        if not only and not job.get("enabled", False):
            continue
        if only and not job.get("enabled", False) and not include_disabled:
            continue
        selected.append(job)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--only", action="append", default=[], help="Run one job id. Can be repeated.")
    parser.add_argument("--include-disabled", action="store_true", help="Allow --only to run disabled jobs.")
    parser.add_argument("--list", action="store_true", help="List jobs and exit.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    plan_path = args.plan if args.plan.is_absolute() else ROOT / args.plan
    plan = load_yaml(plan_path)
    summary_path = args.summary if args.summary.is_absolute() else ROOT / args.summary

    selected = select_jobs(plan, set(args.only), args.include_disabled, args.list)
    if args.list:
        return 0
    if not selected:
        print("No jobs selected. Use --list to inspect available jobs.", file=sys.stderr)
        return 2

    failed = 0
    for job in selected:
        summary = run_job(job, plan, summary_path)
        print(f"{summary['job_id']}: {summary['status']} ({summary['duration_sec']}s)")
        if summary["status"] != "success":
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
