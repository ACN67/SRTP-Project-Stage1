#!/usr/bin/env bash
ROOT=/home/xaillor/projects/srtp-code-llm-pruning/results/raw/pan_half_align_20260806_174220
for t in dense_baseline magnitude_he_s0p10 wanda_he_s0p10; do
  for b in humaneval mbpp livecodebench; do
    f="$ROOT/$t/$b/score_summary.json"
    if [[ -f "$f" ]]; then
      python3 - "$t" "$b" "$f" <<'PY'
import json,sys
d=json.load(open(sys.argv[3]))
print(sys.argv[1], sys.argv[2], d.get("base_pass_count", d.get("pass_count")), "/", d.get("task_count"), "rate", d.get("base_pass_rate", d.get("pass_rate")))
PY
    else
      echo "$t $b pending"
    fi
  done
done
[[ -f "$ROOT/HALF_ALIGN_DONE" ]] && echo DONE_FLAG=yes || echo DONE_FLAG=no
pgrep -af 'run_pan_half_he_mbpp|generate_full|generate_code|score_live' | head -n 5 || echo NO_PROC
echo '--- poll ---'
tail -n 12 /tmp/poll_half_out2.log 2>/dev/null || true
echo '--- job log ---'
tail -n 15 /home/xaillor/projects/srtp-code-llm-pruning/results/raw/pan_half_align_he_mbpp_lcb.log 2>/dev/null | tr '\r' '\n' | tail -n 15
