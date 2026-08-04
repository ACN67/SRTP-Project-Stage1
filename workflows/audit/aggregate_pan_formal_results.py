#!/usr/bin/env python3
"""Aggregate pan formal OPT/Qwen results into results/auxiliary/pan_full_eval/pan_formal_comparison.csv."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_opt_stdout(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    out: dict[str, str] = {}
    m = re.search(r"sparsity sanity check\s+([0-9.]+)", text)
    if m:
        out["sparsity_actual"] = m.group(1)
    m = re.search(r"wikitext perplexity\s+([0-9.]+)", text, re.I)
    if not m:
        m = re.search(r"ppl on wikitext[^\d]*([0-9.]+)", text, re.I)
    if not m:
        m = re.search(r"ppl on [^\n]*?([0-9.]+)", text, re.I)
    if m:
        out["ppl"] = m.group(1)
    return out


def parse_dsnot_results(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    out: dict[str, str] = {}
    m = re.search(r"sparsity sanity check\s+([0-9.]+)", text)
    if m:
        out["sparsity_actual"] = m.group(1)
    m = re.search(r"ppl:\s*([0-9.]+)", text)
    if m:
        out["ppl"] = m.group(1)
    return out


def parse_evalplus_stdout(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    out: dict[str, str] = {}
    # Capture ordered pass@1 lines: base then plus
    vals = re.findall(r"pass@1:\s*([0-9.]+)", text, flags=re.I)
    if vals:
        out["pass_at_1"] = vals[0]
        if len(vals) > 1:
            out["pass_at_1_plus"] = vals[1]
    return out


def parse_prune_result(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "sparsity_actual": str(data.get("actual_target_module_sparsity", "")),
        "status": str(data.get("status", "")),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--output", default="results/auxiliary/pan_full_eval/pan_formal_comparison.csv")
    args = parser.parse_args()

    root = Path(args.formal_root)
    if not root.is_absolute():
        root = ROOT / root
    rows: list[dict[str, str]] = []

    for run_dir in sorted(root.glob("opt125m_*")):
        tag = run_dir.name.replace("opt125m_", "")
        method = "DSnoT" if tag.startswith("dsnot") else ("Magnitude" if tag.startswith("magnitude") else "Wanda")
        ratio = "0.5" if "s0p5" in tag else "0.3"
        parsed = parse_dsnot_results(run_dir / "dsnot_results.txt")
        parsed.update({k: v for k, v in parse_opt_stdout(run_dir / "stdout.log").items() if k not in parsed or not parsed.get(k)})
        rows.append(
            {
                "method": method,
                "model": "facebook/opt-125m",
                "sparsity_target": ratio,
                "sparsity_actual": parsed.get("sparsity_actual", ""),
                "calib": "nsamples=128",
                "benchmark": "wikitext2",
                "metric": "ppl",
                "value": parsed.get("ppl", ""),
                "value_plus": "",
                "run_id": str(run_dir.relative_to(ROOT)),
                "seed": "0",
            }
        )

    for run_dir in sorted(root.glob("qwen15b_*")):
        tag = run_dir.name.replace("qwen15b_", "")
        result = parse_prune_result(run_dir / "pruned" / "wanda_qwen_prune_result.json")
        method = "Magnitude" if tag.startswith("magnitude") else "Wanda"
        ratio = "0.30" if "s0p30" in tag else "0.10"
        guide = "mbpp_formal/guide" if "mbpp" in tag else "humaneval_formal/guide"
        rows.append(
            {
                "method": method,
                "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                "sparsity_target": ratio,
                "sparsity_actual": result.get("sparsity_actual", ""),
                "calib": guide,
                "benchmark": "prune",
                "metric": "target_module_sparsity",
                "value": result.get("sparsity_actual", ""),
                "value_plus": "",
                "run_id": str(run_dir.relative_to(ROOT)),
                "seed": "0",
            }
        )

    for eval_dir in sorted(root.glob("eval_*")):
        tag = eval_dir.name.replace("eval_", "")
        if tag == "dense_baseline":
            method = "Dense"
            ratio = "0.0"
            calib = "none"
        elif tag.startswith("magnitude"):
            method = "Magnitude"
            ratio = "0.30" if "s0p30" in tag else "0.10"
            calib = "mbpp_formal/guide" if "mbpp" in tag else "humaneval_formal/guide"
        else:
            method = "Wanda"
            ratio = "0.30" if "s0p30" in tag else "0.10"
            calib = "mbpp_formal/guide" if "mbpp" in tag else "humaneval_formal/guide"
        for bench in ("humaneval", "mbpp"):
            parsed = parse_evalplus_stdout(eval_dir / bench / "evalplus" / "evalplus_stdout.log")
            metrics_path = eval_dir / bench / "metrics.json"
            gen_count = ""
            if metrics_path.exists():
                gen_count = str(json.loads(metrics_path.read_text(encoding="utf-8")).get("generated_count", ""))
            rows.append(
                {
                    "method": method,
                    "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                    "sparsity_target": ratio,
                    "sparsity_actual": "",
                    "calib": calib,
                    "benchmark": bench,
                    "metric": "pass@1",
                    "value": parsed.get("pass_at_1", ""),
                    "value_plus": parsed.get("pass_at_1_plus", ""),
                    "run_id": str((eval_dir / bench).relative_to(ROOT)),
                    "seed": "0",
                    "generated_count": gen_count,
                }
            )

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "model",
        "sparsity_target",
        "sparsity_actual",
        "calib",
        "benchmark",
        "metric",
        "value",
        "value_plus",
        "run_id",
        "seed",
        "generated_count",
    ]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote {out} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
