#!/usr/bin/env bash
cd /home/xaillor/projects/srtp-code-llm-pruning
tr -d '\r' < /mnt/c/Users/Xile/.vscode/.code/SRTP-Project-Stage1/scripts/setup/run_pan_half_s0p30.sh > scripts/setup/run_pan_half_s0p30.sh
chmod +x scripts/setup/run_pan_half_s0p30.sh
nohup bash scripts/setup/run_pan_half_s0p30.sh > results/raw/pan_half_s0p30.log 2>&1 &
echo $! > results/raw/pan_half_s0p30.pid
sleep 4
echo "PID=$(cat results/raw/pan_half_s0p30.pid)"
head -n 20 results/raw/pan_half_s0p30.log
pgrep -af 'run_pan_half_s0p30|generate_full|generate_evalplus' | head
