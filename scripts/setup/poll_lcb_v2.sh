#!/usr/bin/env bash
ROOT=/home/xaillor/projects/srtp-code-llm-pruning/results/raw/pan_half_align_20260806_174220
LOG=/home/xaillor/projects/srtp-code-llm-pruning/results/raw/pan_half_lcb_v2.log
while [[ ! -f "$ROOT/HALF_ALIGN_DONE" ]]; do
  date
  for t in dense_baseline magnitude_he_s0p10 wanda_he_s0p10; do
    v2="$ROOT/$t/livecodebench_v2"
    n=0
    [[ -f "$v2/generations.jsonl" ]] && n=$(wc -l < "$v2/generations.jsonl")
    sc=no
    [[ -f "$v2/score_summary.json" ]] && sc=yes
    echo "$t gens=$n score=$sc"
  done
  pgrep -af 'rerun_lcb_v2|generate_evalplus|score_live' | head -n 3 || echo NO_PROC
  sleep 300
done
echo POLL_FINISHED
date
