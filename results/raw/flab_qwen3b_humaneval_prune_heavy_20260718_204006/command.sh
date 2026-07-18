set -euo pipefail && cd /home/keshu/projects/srtp-code-llm-pruning && source scripts/setup/env.sh && source .venv-flab_pruner/bin/activate && cd /home/keshu/projects/srtp-code-llm-pruning && python scripts/adapt/flab_qwen_prune.py \
  --model Qwen/Qwen2.5-Coder-3B-Instruct \
  --guide-file data/splits/humaneval/guide.jsonl \
  --save-dir "$RUN_DIR/flab_qwen3b_humaneval" \
  --stage top \
  --prune-ratio 0.10 \
  --max-guide-samples 4 \
  --dtype bf16 \
  --device-map cuda:0
