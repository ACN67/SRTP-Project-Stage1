#!/usr/bin/env bash
# Pause LCB scoring; finish Mag/Wanda HE+MBPP first.
set -euo pipefail
cd "$HOME/projects/srtp-code-llm-pruning"
ROOT=results/raw/pan_half_align_20260806_174220
WIN=/mnt/c/Users/Xile/.vscode/.code/SRTP-Project-Stage1
PY=.venv-common/bin/python
FORMAL=results/raw/pan_formal_20260724_203248
MAX_NEW_TOKENS=256

pkill -f 'score_livecodebench_split.py' 2>/dev/null || true
sleep 2

cp -f "$WIN/scripts/evaluate/predictions_to_samples.py" scripts/evaluate/
tr -d '\r' < scripts/evaluate/predictions_to_samples.py > /tmp/p2s.py && mv /tmp/p2s.py scripts/evaluate/predictions_to_samples.py

run_he_mbpp() {
  local tag="$1"
  local model="$2"
  local out="$ROOT/$tag"
  mkdir -p "$out"
  if [[ ! -f "$out/humaneval/score_summary.json" ]]; then
    echo "=== [$tag] HE ==="
    mkdir -p "$out/humaneval"
    if [[ ! -f "$out/humaneval/predictions.jsonl" ]]; then
      "$PY" scripts/evaluate/generate_full_benchmark.py \
        --model "$model" --benchmark humaneval \
        --split data/splits/humaneval_half/eval.jsonl \
        --output-dir "$out/humaneval" --max-new-tokens "$MAX_NEW_TOKENS" --device-map cuda:0
    fi
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
    if [[ ! -f "$out/mbpp/predictions.jsonl" ]]; then
      "$PY" scripts/evaluate/generate_full_benchmark.py \
        --model "$model" --benchmark mbpp \
        --split data/splits/mbpp_evalplus_half/eval.jsonl \
        --output-dir "$out/mbpp" --max-new-tokens "$MAX_NEW_TOKENS" --device-map cuda:0
    fi
    "$PY" scripts/evaluate/predictions_to_samples.py \
      --predictions "$out/mbpp/predictions.jsonl" \
      --output "$out/mbpp/samples.jsonl" --benchmark mbpp
    "$PY" scripts/eval/score_mbpp_smoke.py \
      --split data/splits/mbpp_evalplus_half/eval.jsonl \
      --samples "$out/mbpp/samples.jsonl" \
      --out-dir "$out/mbpp" --base-only | tee "$out/mbpp/score_stdout.log"
  fi
  echo "HE_MBPP_DONE=$tag"
}

run_he_mbpp magnitude_he_s0p10 "$FORMAL/qwen15b_magnitude_he_s0p10/pruned/pruned_model"
run_he_mbpp wanda_he_s0p10 "$FORMAL/qwen15b_wanda_he_s0p10/pruned/pruned_model"
echo HE_MBPP_ALL_DONE

# Then LCB gens + scores
LCB_PY=.venv-livecodebench/bin/python
for tag_model in \
  "dense_baseline|Qwen/Qwen2.5-Coder-1.5B-Instruct" \
  "magnitude_he_s0p10|$FORMAL/qwen15b_magnitude_he_s0p10/pruned/pruned_model" \
  "wanda_he_s0p10|$FORMAL/qwen15b_wanda_he_s0p10/pruned/pruned_model"
do
  tag="${tag_model%%|*}"
  model="${tag_model#*|}"
  out="$ROOT/$tag/livecodebench"
  mkdir -p "$out"
  if [[ ! -f "$out/predictions.jsonl" ]]; then
    echo "=== [$tag] LCB gen ==="
    "$PY" scripts/evaluate/generate_code_split.py \
      --model "$model" --benchmark livecodebench \
      --split data/splits/livecodebench_half/eval.jsonl \
      --output-dir "$out" --max-new-tokens "$MAX_NEW_TOKENS" --device-map cuda:0
  fi
  if [[ ! -f "$out/score_summary.json" ]]; then
    echo "=== [$tag] LCB score ==="
    cp -f "$out/predictions.jsonl" "$out/generations.jsonl"
    "$LCB_PY" scripts/eval/score_livecodebench_split.py \
      --split data/splits/livecodebench_half/eval.jsonl \
      --generations "$out/generations.jsonl" \
      --out-dir "$out" --lcb-release release_v1 | tee "$out/score_stdout.log"
  fi
done

"$PY" - "$ROOT" <<'PY'
import csv, json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows=[]
for tag_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    tag=tag_dir.name
    method="Dense" if tag.startswith("dense") else ("Magnitude" if tag.startswith("magnitude") else "Wanda")
    spars="0.0" if method=="Dense" else "0.10"
    for bench in ("humaneval","mbpp","livecodebench"):
        summary=tag_dir/bench/"score_summary.json"
        if not summary.exists():
            continue
        data=json.loads(summary.read_text(encoding="utf-8"))
        if bench=="livecodebench":
            pc=data.get("pass_count"); tc=data.get("task_count") or 200; rate=data.get("pass_rate"); val=f"{pc}/{tc}"
        else:
            pc=data.get("base_pass_count"); tc=data.get("task_count"); rate=data.get("base_pass_rate"); val=f"{pc}/{tc}"
        rows.append({"method":method,"sparsity":spars,"benchmark":bench,"metric":"pass_count","value":val,"pass_rate":rate or "","run_id":str(tag_dir/bench)})
out=Path("results/tables/pan_half_comparison.csv"); out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w",encoding="utf-8",newline="") as f:
    w=csv.DictWriter(f, fieldnames=["method","sparsity","benchmark","metric","value","pass_rate","run_id"]); w.writeheader(); w.writerows(rows)
print("wrote", out, len(rows))
(root/"HALF_ALIGN_DONE").write_text("ok\n", encoding="utf-8")
print("HALF_ALIGN_DONE")
PY

tr -d '\r' < "$WIN/scripts/setup/finalize_half_align.sh" > /tmp/finalize_half_align.sh
bash /tmp/finalize_half_align.sh || true
