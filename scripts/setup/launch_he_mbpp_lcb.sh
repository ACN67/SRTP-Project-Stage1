#!/usr/bin/env bash
cd /home/xaillor/projects/srtp-code-llm-pruning
pkill -f score_livecodebench_split.py 2>/dev/null || true
pkill -f resume_pan_half_align 2>/dev/null || true
sleep 1
nohup bash scripts/setup/run_pan_half_he_mbpp_then_lcb.sh > results/raw/pan_half_align_he_mbpp_lcb.log 2>&1 &
echo $! > results/raw/pan_half_align_he_mbpp_lcb.pid
sleep 4
echo "PID=$(cat results/raw/pan_half_align_he_mbpp_lcb.pid)"
head -n 30 results/raw/pan_half_align_he_mbpp_lcb.log
pgrep -af 'run_pan_half_he_mbpp|generate_full' | head
