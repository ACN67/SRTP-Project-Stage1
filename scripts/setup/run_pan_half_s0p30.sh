#!/usr/bin/env bash
# Half-eval Mag/Wanda 0.30 (HE+MBPP+LCB) to finish paper-protocol coverage.
set -euo pipefail
cd /home/xaillor/projects/srtp-code-llm-pruning
ROOT=results/raw/pan_half_align_20260806_174220
FORMAL=results/raw/pan_formal_20260724_203248
PY=.venv-common/bin/python
LCB_PY=.venv-livecodebench/bin/python
MAX_NEW=256
WIN=/mnt/c/Users/Xile/.vscode/.code/SRTP-Project-Stage1

cp -f "$WIN/scripts/evaluate/predictions_to_samples.py" scripts/evaluate/ 2>/dev/null || true
cp -f "$WIN/scripts/eval/generate_evalplus_samples.py" scripts/eval/ 2>/dev/null || true
cp -f "$WIN/scripts/eval/completion_extraction.py" scripts/eval/ 2>/dev/null || true
for f in scripts/evaluate/predictions_to_samples.py scripts/eval/generate_evalplus_samples.py scripts/eval/completion_extraction.py; do
  [[ -f "$f" ]] || continue
  tr -d '\r' < "$f" > "$f.tmp" && mv "$f.tmp" "$f"
done

run_he_mbpp() {
  local tag="$1" model="$2"
  local out="$ROOT/$tag"
  mkdir -p "$out"
  if [[ ! -f "$out/humaneval/score_summary.json" ]]; then
    echo "=== [$tag] HE ==="
    mkdir -p "$out/humaneval"
    [[ -f "$out/humaneval/predictions.jsonl" ]] || \
      "$PY" scripts/evaluate/generate_full_benchmark.py \
        --model "$model" --benchmark humaneval \
        --split data/splits/humaneval_half/eval.jsonl \
        --output-dir "$out/humaneval" --max-new-tokens "$MAX_NEW" --device-map cuda:0
    "$PY" scripts/evaluate/predictions_to_samples.py \
      --predictions "$out/humaneval/predictions.jsonl" \
      --output "$out/humaneval/samples.jsonl" --benchmark humaneval
    "$PY" scripts/eval/score_humaneval_smoke.py \
      --split data/splits/humaneval_half/eval.jsonl \
      --samples "$out/humaneval/samples.jsonl" \
      --out-dir "$out/humaneval" --base-only | tee "$out/humaneval/score_stdout.log"
  fi
  if [[ ! -f "$out/mbpp/score_summary.json" ]]; then
    echo "=== [$tag] MBPP ==="
    mkdir -p "$out/mbpp"
    [[ -f "$out/mbpp/predictions.jsonl" ]] || \
      "$PY" scripts/evaluate/generate_full_benchmark.py \
        --model "$model" --benchmark mbpp \
        --split data/splits/mbpp_evalplus_half/eval.jsonl \
        --output-dir "$out/mbpp" --max-new-tokens "$MAX_NEW" --device-map cuda:0
    "$PY" scripts/evaluate/predictions_to_samples.py \
      --predictions "$out/mbpp/predictions.jsonl" \
      --output "$out/mbpp/samples.jsonl" --benchmark mbpp
    "$PY" scripts/eval/score_mbpp_smoke.py \
      --split data/splits/mbpp_evalplus_half/eval.jsonl \
      --samples "$out/mbpp/samples.jsonl" \
      --out-dir "$out/mbpp" --base-only | tee "$out/mbpp/score_stdout.log"
  fi
}

run_lcb() {
  local tag="$1" model="$2"
  local out="$ROOT/$tag/livecodebench_v2"
  mkdir -p "$out"
  if [[ ! -f "$out/generations.jsonl" ]]; then
    echo "=== [$tag] LCB gen ==="
    "$PY" scripts/eval/generate_evalplus_samples.py \
      --model "$model" \
      --split data/splits/livecodebench_half/eval.jsonl \
      --out-dir "$out" --max-new-tokens "$MAX_NEW" \
      --dtype bf16 --device cuda:0 --load-mode device_map --device-map cuda:0 \
      --prompt-mode lcb_completion
  fi
  if [[ ! -f "$out/score_summary.json" ]]; then
    echo "=== [$tag] LCB score ==="
    "$LCB_PY" scripts/eval/score_livecodebench_split.py \
      --split data/splits/livecodebench_half/eval.jsonl \
      --generations "$out/generations.jsonl" \
      --out-dir "$out" --lcb-release release_v1 | tee "$out/score_stdout.log"
  fi
  mkdir -p "$ROOT/$tag/livecodebench"
  cp -f "$out/score_summary.json" "$ROOT/$tag/livecodebench/score_summary.json"
  cp -f "$out/generations.jsonl" "$ROOT/$tag/livecodebench/generations.jsonl"
}

for spec in \
  "magnitude_he_s0p30|$FORMAL/qwen15b_magnitude_he_s0p30/pruned/pruned_model" \
  "wanda_he_s0p30|$FORMAL/qwen15b_wanda_he_s0p30/pruned/pruned_model"
do
  tag="${spec%%|*}"; model="${spec#*|}"
  [[ -f "$model/model.safetensors" ]] || { echo "missing $model"; exit 1; }
  run_he_mbpp "$tag" "$model"
  run_lcb "$tag" "$model"
  echo "DONE_TAG=$tag"
done

"$PY" - "$ROOT" <<'PY'
import csv, json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows=[]
for tag_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.')):
    tag=tag_dir.name
    if tag.startswith('dense'): method,spars='Dense','0.0'
    elif 'magnitude' in tag: method,spars=('Magnitude','0.30' if 's0p30' in tag else '0.10')
    elif 'wanda' in tag: method,spars=('Wanda','0.30' if 's0p30' in tag else '0.10')
    else: continue
    for bench in ('humaneval','mbpp','livecodebench'):
        summary=tag_dir/bench/'score_summary.json'
        if not summary.exists():
            continue
        data=json.loads(summary.read_text(encoding='utf-8'))
        if bench=='livecodebench':
            pc,tc,rate=data.get('pass_count'),data.get('task_count') or 200,data.get('pass_rate')
            val=f'{pc}/{tc}'
        else:
            pc,tc,rate=data.get('base_pass_count'),data.get('task_count'),data.get('base_pass_rate')
            val=f'{pc}/{tc}'
        rows.append(dict(method=method,sparsity=spars,benchmark=bench,metric='pass_count',value=val,pass_rate=rate or '',run_id=str(tag_dir/bench)))
out=Path('results/tables/pan_half_comparison.csv'); out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f, fieldnames=['method','sparsity','benchmark','metric','value','pass_rate','run_id']); w.writeheader(); w.writerows(rows)
print('wrote', out, len(rows))
(root/'HALF_S0P30_DONE').write_text('ok\n', encoding='utf-8')
print('HALF_S0P30_DONE')
PY
cp -f results/tables/pan_half_comparison.csv "$WIN/results/tables/" || true
