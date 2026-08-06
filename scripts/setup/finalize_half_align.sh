#!/usr/bin/env bash
# Sync half results + backfill summary table placeholders from score_summary.json
set -euo pipefail
cd "$HOME/projects/srtp-code-llm-pruning"
WIN=/mnt/c/Users/Xile/.vscode/.code/SRTP-Project-Stage1
ROOT=$(ls -dt results/raw/pan_half_align_* 2>/dev/null | head -1)
[[ -n "$ROOT" ]] || { echo "no root"; exit 1; }
[[ -f "$ROOT/HALF_ALIGN_DONE" ]] || { echo "not done: $ROOT"; exit 2; }

.venv-common/bin/python - "$ROOT" "$WIN" <<'PY'
import csv, json, re, sys
from pathlib import Path
root = Path(sys.argv[1])
win = Path(sys.argv[2])
rows = []
cell = {}
for tag_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    tag = tag_dir.name
    method = "Dense" if tag.startswith("dense") else ("Magnitude" if tag.startswith("magnitude") else "Wanda")
    spars = "0.0" if method == "Dense" else "0.10"
    vals = {}
    for bench in ("humaneval", "mbpp", "livecodebench"):
        summary = tag_dir / bench / "score_summary.json"
        if not summary.exists():
            continue
        data = json.loads(summary.read_text(encoding="utf-8"))
        if bench == "livecodebench":
            pc = data.get("pass_count")
            tc = data.get("task_count") or 200
            rate = data.get("pass_rate")
            disp = f"{pc}/{tc}"
        else:
            pc = data.get("base_pass_count")
            tc = data.get("task_count")
            rate = data.get("base_pass_rate")
            disp = f"{pc}/{tc}"
        vals[bench] = disp
        rows.append({"method": method, "sparsity": spars, "benchmark": bench, "metric": "pass_count", "value": disp, "pass_rate": rate or "", "run_id": str(tag_dir / bench)})
    cell[method] = vals

out = Path("results/tables/pan_half_comparison.csv")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["method","sparsity","benchmark","metric","value","pass_rate","run_id"])
    w.writeheader()
    for r in rows:
        w.writerow(r)
(win / "results/tables").mkdir(parents=True, exist_ok=True)
out_win = win / "results/tables/pan_half_comparison.csv"
out_win.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")

# Update summary markdown half table
md_path = win / "docs" / "第一阶段总结260804.md"
text = md_path.read_text(encoding="utf-8")
def row(method, spars, guide, he, mb, lcb, status):
    return f"| {method} | Qwen2.5-Coder-1.5B-Instruct | {spars} | {guide} | {he} | {mb} | {lcb} | {status} |"

dense = cell.get("Dense", {})
mag = cell.get("Magnitude", {})
wanda = cell.get("Wanda", {})
new_table = "\n".join([
    "| Method | Model | Sparsity | Guide | HumanEval | MBPP | LiveCodeBench | Status |",
    "|---|---|---:|---|---:|---:|---:|---|",
    row("Dense", "0%", "none", dense.get("humaneval","—"), dense.get("mbpp","—"), dense.get("livecodebench","—"), "半集完成"),
    row("Magnitude", "10%", "HumanEval guide", mag.get("humaneval","—"), mag.get("mbpp","—"), mag.get("livecodebench","—"), "半集完成"),
    row("Wanda", "10%", "HumanEval guide", wanda.get("humaneval","—"), wanda.get("mbpp","—"), wanda.get("livecodebench","—"), "半集完成"),
])
pattern = r"#### 半集对齐表（与结构化组同协议，补跑中）\n\n\| Method \|.*?\n\n"
replacement = "#### 半集对齐表（与结构化组同协议）\n\n" + new_table + "\n\n"
text2, n = re.subn(pattern, replacement, text, count=1, flags=re.S)
if n != 1:
    print("WARN: half table pattern not replaced", n)
else:
    md_path.write_text(text2, encoding="utf-8")
    print("updated summary half table")

# also fix LCB column in full formal table for the three methods
text2 = md_path.read_text(encoding="utf-8")
text2 = text2.replace("| 半集补跑中 | 基线完成 |", f"| {dense.get('livecodebench','半集完成')} | 基线完成；半集已对齐 |")
text2 = text2.replace("| 半集补跑中 | 完成 |\n| Magnitude | Qwen2.5-Coder-1.5B-Instruct | 30%", f"| {mag.get('livecodebench','—')} | 完成；半集已对齐 |\n| Magnitude | Qwen2.5-Coder-1.5B-Instruct | 30%")
# Wanda 10% line is trickier - do simple replace for first Wanda 10% LCB cell
text2 = text2.replace(
    "| Wanda | Qwen2.5-Coder-1.5B-Instruct | 10% | HumanEval guide 32 | 0.152 | 0.680 | 半集补跑中 | 完成 |",
    f"| Wanda | Qwen2.5-Coder-1.5B-Instruct | 10% | HumanEval guide 32 | 0.152 | 0.680 | {wanda.get('livecodebench','—')} | 完成；半集已对齐 |",
)
text2 = text2.replace(
    "| Magnitude | Qwen2.5-Coder-1.5B-Instruct | 10% | HumanEval guide 32 | 0.165 | 0.664 | 半集补跑中 | 完成 |",
    f"| Magnitude | Qwen2.5-Coder-1.5B-Instruct | 10% | HumanEval guide 32 | 0.165 | 0.664 | {mag.get('livecodebench','—')} | 完成；半集已对齐 |",
)
md_path.write_text(text2, encoding="utf-8")
print("wrote", out, out_win)
print(json.dumps(cell, ensure_ascii=False, indent=2))
PY
