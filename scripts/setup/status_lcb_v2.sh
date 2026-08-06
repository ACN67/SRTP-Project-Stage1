#!/usr/bin/env bash
echo '=== poll ==='
tail -n 25 /tmp/poll_lcb_v2_out.log
echo '=== gens ==='
ROOT=/home/xaillor/projects/srtp-code-llm-pruning/results/raw/pan_half_align_20260806_174220
for t in dense_baseline magnitude_he_s0p10 wanda_he_s0p10; do
  d="$ROOT/$t/livecodebench_v2"
  n=0
  [[ -f "$d/generations.jsonl" ]] && n=$(wc -l < "$d/generations.jsonl")
  echo -n "$t gens=$n "
  if [[ -f "$d/score_summary.json" ]]; then
    python3 - "$d/score_summary.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print("score", d.get("pass_count"), "/", d.get("task_count"), d.get("pass_rate"))
PY
  else
    echo "score=no"
  fi
done
[[ -f "$ROOT/HALF_ALIGN_DONE" ]] && echo DONE || echo NOT_DONE
pgrep -af 'rerun_lcb|generate_evalplus|score_live' | head -n 4 || echo NO_PROC
echo '=== log ==='
tail -n 12 /home/xaillor/projects/srtp-code-llm-pruning/results/raw/pan_half_lcb_v2.log | tr '\r' '\n' | tail -n 12
