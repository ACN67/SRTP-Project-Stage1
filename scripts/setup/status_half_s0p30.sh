#!/usr/bin/env bash
ROOT=/home/xaillor/projects/srtp-code-llm-pruning/results/raw
echo "=== s0p30 log ==="
tail -n 15 "$ROOT/pan_half_s0p30.log" 2>/dev/null | tr '\r' '\n' | tail -n 15
echo "=== scores ==="
BASE=/home/xaillor/projects/srtp-code-llm-pruning/results/raw/pan_half_align_20260806_174220
for t in magnitude_he_s0p30 wanda_he_s0p30; do
  for b in humaneval mbpp livecodebench; do
    f="$BASE/$t/$b/score_summary.json"
    if [[ -f "$f" ]]; then
      python3 - "$t" "$b" "$f" <<'PY'
import json,sys
d=json.load(open(sys.argv[3]))
print(sys.argv[1], sys.argv[2], d.get('base_pass_count', d.get('pass_count')), '/', d.get('task_count'))
PY
    else
      echo "$t $b pending"
    fi
  done
done
[[ -f "$BASE/HALF_S0P30_DONE" ]] && echo HALF_S0P30_DONE || echo RUNNING
pgrep -af 'run_pan_half_s0p30|generate_full|generate_evalplus' | head -n 3 || echo NO_PROC
