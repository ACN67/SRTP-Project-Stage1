#!/usr/bin/env bash
# Re-generate LCB half with structured-group prompt (lcb_completion), then score.
set -euo pipefail
cd /home/xaillor/projects/srtp-code-llm-pruning
pkill -f 'run_pan_half_he_mbpp_then_lcb' 2>/dev/null || true
pkill -f 'generate_code_split.py --benchmark livecodebench' 2>/dev/null || true
sleep 2

ROOT=results/raw/pan_half_align_20260806_174220
FORMAL=results/raw/pan_formal_20260724_203248
PY=.venv-common/bin/python
LCB_PY=.venv-livecodebench/bin/python
# ensure eval scripts present
mkdir -p scripts/eval
cp -f /mnt/c/Users/Xile/.vscode/.code/SRTP-Project-Stage1/scripts/eval/generate_evalplus_samples.py scripts/eval/
cp -f /mnt/c/Users/Xile/.vscode/.code/SRTP-Project-Stage1/scripts/eval/completion_extraction.py scripts/eval/
cp -f /mnt/c/Users/Xile/.vscode/.code/SRTP-Project-Stage1/scripts/eval/score_livecodebench_split.py scripts/eval/
for f in scripts/eval/generate_evalplus_samples.py scripts/eval/completion_extraction.py scripts/eval/score_livecodebench_split.py; do
  tr -d '\r' < "$f" > "$f.tmp" && mv "$f.tmp" "$f"
done

run_lcb() {
  local tag="$1"
  local model="$2"
  local out="$ROOT/$tag/livecodebench_v2"
  mkdir -p "$out"
  if [[ ! -f "$out/generations.jsonl" ]]; then
    echo "=== LCB v2 gen $tag ==="
    "$PY" scripts/eval/generate_evalplus_samples.py \
      --model "$model" \
      --split data/splits/livecodebench_half/eval.jsonl \
      --out-dir "$out" \
      --max-new-tokens 256 \
      --dtype bf16 \
      --device cuda:0 \
      --load-mode device_map \
      --device-map cuda:0 \
      --prompt-mode lcb_completion
  fi
  if [[ ! -f "$out/score_summary.json" ]]; then
    echo "=== LCB v2 score $tag ==="
    "$LCB_PY" scripts/eval/score_livecodebench_split.py \
      --split data/splits/livecodebench_half/eval.jsonl \
      --generations "$out/generations.jsonl" \
      --out-dir "$out" \
      --lcb-release release_v1 | tee "$out/score_stdout.log"
  fi
  # also copy as canonical livecodebench dir for aggregator
  mkdir -p "$ROOT/$tag/livecodebench"
  cp -f "$out/score_summary.json" "$ROOT/$tag/livecodebench/score_summary.json"
  cp -f "$out/generations.jsonl" "$ROOT/$tag/livecodebench/generations.jsonl"
  cp -f "$out/generations.jsonl" "$ROOT/$tag/livecodebench/predictions.jsonl"
  echo "LCB_V2_DONE=$tag"
}

run_lcb dense_baseline "Qwen/Qwen2.5-Coder-1.5B-Instruct"
run_lcb magnitude_he_s0p10 "$FORMAL/qwen15b_magnitude_he_s0p10/pruned/pruned_model"
run_lcb wanda_he_s0p10 "$FORMAL/qwen15b_wanda_he_s0p10/pruned/pruned_model"

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

WIN=/mnt/c/Users/Xile/.vscode/.code/SRTP-Project-Stage1
tr -d '\r' < "$WIN/scripts/setup/finalize_half_align.sh" > /tmp/finalize_half_align.sh
bash /tmp/finalize_half_align.sh || true
