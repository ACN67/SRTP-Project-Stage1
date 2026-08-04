#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/projects/srtp-code-llm-pruning"
WIN=/mnt/c/Users/Xile/.vscode/.code/SRTP-Project-Stage1
FORMAL_ROOT=results/raw/pan_formal_20260724_203248

cp "$WIN/scripts/audit/aggregate_pan_formal_results.py" scripts/audit/aggregate_pan_formal_results.py
tr -d '\r' < scripts/audit/aggregate_pan_formal_results.py > /tmp/a.py && mv /tmp/a.py scripts/audit/aggregate_pan_formal_results.py

.venv-common/bin/python scripts/audit/aggregate_pan_formal_results.py \
  --formal-root "$FORMAL_ROOT" \
  --output results/tables/pan_formal_comparison.csv

# Sync key artifacts to Windows clone
mkdir -p "$WIN/results/tables" "$WIN/data/splits/mbpp_evalplus" "$WIN/reports/stage1"
cp results/tables/pan_formal_comparison.csv "$WIN/results/tables/"
cp -r data/splits/mbpp_evalplus/. "$WIN/data/splits/mbpp_evalplus/" 2>/dev/null || true
cp reports/stage1/split_leakage_check_formal.json "$WIN/reports/stage1/" 2>/dev/null || true

# Write a concise formal results markdown into formal root and windows
python3 - <<'PY'
from pathlib import Path
import csv
root = Path('/home/xaillor/projects/srtp-code-llm-pruning')
csv_path = root/'results/tables/pan_formal_comparison.csv'
lines = ["# Pan Formal Comparison", "", "Source run: `results/raw/pan_formal_20260724_203248`", ""]
rows = list(csv.DictReader(csv_path.open(encoding='utf-8')))
lines += ["## OPT-125M WikiText PPL (nsamples=128)", "", "| Method | Sparsity | PPL |", "|---|---:|---:|"]
for r in rows:
    if r['benchmark']=='wikitext2':
        lines.append(f"| {r['method']} | {r['sparsity_target']} | {r['value']} |")
lines += ["", "## Qwen2.5-Coder-1.5B-Instruct Pass@1 (evalplus)", "", "| Method | Sparsity | Calib | HumanEval | HumanEval+ | MBPP | MBPP+ |", "|---|---:|---|---:|---:|---:|---:|"]
# pivot he/mbpp
from collections import defaultdict
cell = defaultdict(dict)
for r in rows:
    if r['benchmark'] in ('humaneval','mbpp') and r['metric']=='pass@1':
        key=(r['method'], r['sparsity_target'], r['calib'])
        cell[key][r['benchmark']] = (r['value'], r.get('value_plus',''))
for key, d in sorted(cell.items()):
    method, spars, calib = key
    he = d.get('humaneval', ('',''))
    mb = d.get('mbpp', ('',''))
    lines.append(f"| {method} | {spars} | {calib} | {he[0]} | {he[1]} | {mb[0]} | {mb[1]} |")
text='\n'.join(lines)+'\n'
for p in [root/'results/raw/pan_formal_20260724_203248/FORMAL_RESULTS.md', Path('/mnt/c/Users/Xile/.vscode/.code/SRTP-Project-Stage1/reports/stage1/pan_formal_results.md')]:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding='utf-8')
    print('wrote', p)
print(text)
PY
