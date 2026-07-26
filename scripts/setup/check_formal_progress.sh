#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/xaillor/projects/srtp-code-llm-pruning/results/raw/pan_formal_20260724_203248
for t in dense_baseline magnitude_he_s0p10 wanda_he_s0p10 magnitude_he_s0p30 wanda_he_s0p30 wanda_mbpp_s0p10 magnitude_mbpp_s0p10; do
  echo "== $t HE =="
  if [[ -f "$ROOT/eval_$t/humaneval/evalplus/evalplus_stdout.log" ]]; then
    tail -6 "$ROOT/eval_$t/humaneval/evalplus/evalplus_stdout.log"
  else
    echo missing
  fi
  echo "== $t MBPP =="
  if [[ -f "$ROOT/eval_$t/mbpp/evalplus/evalplus_stdout.log" ]]; then
    tail -6 "$ROOT/eval_$t/mbpp/evalplus/evalplus_stdout.log"
  else
    echo missing
  fi
done
ls "$ROOT"/phase2c.done 2>/dev/null || echo "phase2c not done"
ps aux | grep -E "generate_full|evalplus|phase2c" | grep -v grep || true
