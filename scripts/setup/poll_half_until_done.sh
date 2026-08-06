#!/usr/bin/env bash
ROOT=/home/xaillor/projects/srtp-code-llm-pruning/results/raw/pan_half_align_20260806_174220
LOG=/home/xaillor/projects/srtp-code-llm-pruning/results/raw/pan_half_align_he_mbpp_lcb.log
while [[ ! -f "$ROOT/HALF_ALIGN_DONE" ]]; do
  date
  for t in dense_baseline magnitude_he_s0p10 wanda_he_s0p10; do
    he=no; mb=no; lcb=no
    [[ -f "$ROOT/$t/humaneval/score_summary.json" ]] && he=yes
    [[ -f "$ROOT/$t/mbpp/score_summary.json" ]] && mb=yes
    [[ -f "$ROOT/$t/livecodebench/score_summary.json" ]] && lcb=yes
    echo "$t he=$he mb=$mb lcb=$lcb"
  done
  pgrep -af 'run_pan_half_he_mbpp|generate_full|generate_code|score_live' | head -n 3 || echo NO_PROC
  tail -n 3 "$LOG" 2>/dev/null | tr '\r' '\n' | tail -n 3
  sleep 300
done
echo POLL_FINISHED
date
