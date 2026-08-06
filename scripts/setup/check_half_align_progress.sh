#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/xaillor/projects/srtp-code-llm-pruning/results/raw/pan_half_align_20260806_174220
echo "ROOT=$ROOT"
for tag in dense_baseline magnitude_he_s0p10 wanda_he_s0p10; do
  echo "== $tag =="
  for b in humaneval mbpp livecodebench; do
    s="$ROOT/$tag/$b/score_summary.json"
    if [[ -f "$s" ]]; then
      python3 - "$s" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print(sys.argv[1].split("/")[-2], d.get("base_pass_count", d.get("pass_count")), "/", d.get("task_count"), "rate", d.get("base_pass_rate", d.get("pass_rate")))
PY
    else
      echo "$b pending"
      [[ -f "$ROOT/$tag/$b/predictions.jsonl" ]] && echo "  (has predictions)"
    fi
  done
done
ps -eo pid,cmd | grep -E 'generate_full|generate_code|score_|run_pan_half' | grep -v grep || true
[[ -f "$ROOT/HALF_ALIGN_DONE" ]] && echo HALF_ALIGN_DONE || echo RUNNING
tail -n 15 /home/xaillor/projects/srtp-code-llm-pruning/results/raw/pan_half_align_nohup.log
