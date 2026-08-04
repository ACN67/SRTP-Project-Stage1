#!/usr/bin/env bash
set -euo pipefail
cd /home/keshu/projects/srtp-code-llm-pruning
/home/keshu/projects/srtp-code-llm-pruning-pre-final-original-20260804-1015/.venv-common/bin/python methods/flab_pruner/qwen_prune.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --guide-file data/benchmarks/r4_half/humaneval/guide.jsonl --save-dir /home/keshu/srtp-artifacts/flab_qwen15b_benchmark_activation_he_keep80 --importance-mode benchmark_activation --importance-device cuda:0 --prune-ratio 0.20 --max-guide-samples 1 --importance-max-length 64 --local-files-only
