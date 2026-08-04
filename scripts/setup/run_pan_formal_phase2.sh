#!/usr/bin/env bash
# Phase 2: Qwen formal prune + full HE/MBPP generation + evalplus
set -euo pipefail
cd "$HOME/projects/srtp-code-llm-pruning"
source scripts/setup/env.sh
export PATH="$HOME/.local/bin:$PATH"

FORMAL_ROOT="${1:-}"
if [[ -z "$FORMAL_ROOT" ]]; then
  FORMAL_ROOT=$(cat /tmp/pan_formal_root.txt)
fi
echo "FORMAL_ROOT=$FORMAL_ROOT"
mkdir -p "$FORMAL_ROOT"

QWEN=Qwen/Qwen2.5-Coder-1.5B-Instruct
HE_GUIDE=data/splits/humaneval_formal/guide.jsonl
HE_EVAL=data/splits/humaneval_formal/eval.jsonl
MB_GUIDE=data/splits/mbpp_formal/guide.jsonl
MB_EVAL=data/splits/mbpp_formal/eval.jsonl

prune_one() {
  local method="$1"
  local guide="$2"
  local ratio="$3"
  local tag="$4"
  local run_dir="$FORMAL_ROOT/qwen15b_${tag}"
  mkdir -p "$run_dir"
  echo "=== prune $tag ==="
  .venv-wanda/bin/python scripts/adapt/wanda_qwen_prune.py \
    --method "$method" \
    --model "$QWEN" \
    --guide-file "$guide" \
    --save-dir "$run_dir/pruned" \
    --sparsity-ratio "$ratio" \
    --max-guide-samples 32 \
    --max-seq-len 512 \
    --dtype bf16 \
    --device-map cuda:0 \
    | tee "$run_dir/prune_stdout.log"
}

eval_pair() {
  local model_path="$1"
  local tag="$2"
  local out="$FORMAL_ROOT/eval_${tag}"
  mkdir -p "$out"

  echo "=== generate HE $tag ==="
  .venv-common/bin/python scripts/evaluate/generate_full_benchmark.py \
    --model "$model_path" \
    --split "$HE_EVAL" \
    --output-dir "$out/humaneval" \
    --benchmark humaneval \
    --dtype bf16 \
    --device-map cuda:0 \
    --max-new-tokens 512

  echo "=== evalplus HE $tag ==="
  .venv-common/bin/python scripts/evaluate/run_evalplus_passat1.py \
    --predictions "$out/humaneval/predictions.jsonl" \
    --output-dir "$out/humaneval/evalplus" \
    --benchmark humaneval \
    --python-bin "$HOME/projects/srtp-code-llm-pruning/.venv-common/bin/python" || true

  echo "=== generate MBPP $tag ==="
  .venv-common/bin/python scripts/evaluate/generate_full_benchmark.py \
    --model "$model_path" \
    --split "$MB_EVAL" \
    --output-dir "$out/mbpp" \
    --benchmark mbpp \
    --dtype bf16 \
    --device-map cuda:0 \
    --max-new-tokens 512

  echo "=== evalplus MBPP $tag ==="
  .venv-common/bin/python scripts/evaluate/run_evalplus_passat1.py \
    --predictions "$out/mbpp/predictions.jsonl" \
    --output-dir "$out/mbpp/evalplus" \
    --benchmark mbpp \
    --python-bin "$HOME/projects/srtp-code-llm-pruning/.venv-common/bin/python" || true
}

# Dense baseline (HF id)
eval_pair "$QWEN" dense_baseline

# HE-guide prunes
prune_one magnitude "$HE_GUIDE" 0.10 magnitude_he_s0p10
prune_one wanda "$HE_GUIDE" 0.10 wanda_he_s0p10
prune_one magnitude "$HE_GUIDE" 0.30 magnitude_he_s0p30
prune_one wanda "$HE_GUIDE" 0.30 wanda_he_s0p30

eval_pair "$FORMAL_ROOT/qwen15b_magnitude_he_s0p10/pruned/pruned_model" magnitude_he_s0p10
eval_pair "$FORMAL_ROOT/qwen15b_wanda_he_s0p10/pruned/pruned_model" wanda_he_s0p10
eval_pair "$FORMAL_ROOT/qwen15b_magnitude_he_s0p30/pruned/pruned_model" magnitude_he_s0p30
eval_pair "$FORMAL_ROOT/qwen15b_wanda_he_s0p30/pruned/pruned_model" wanda_he_s0p30

# MBPP-guide prunes (Wanda required; Magnitude 0.10)
prune_one wanda "$MB_GUIDE" 0.10 wanda_mbpp_s0p10
prune_one magnitude "$MB_GUIDE" 0.10 magnitude_mbpp_s0p10
eval_pair "$FORMAL_ROOT/qwen15b_wanda_mbpp_s0p10/pruned/pruned_model" wanda_mbpp_s0p10
eval_pair "$FORMAL_ROOT/qwen15b_magnitude_mbpp_s0p10/pruned/pruned_model" magnitude_mbpp_s0p10

echo "PHASE2_DONE" | tee "$FORMAL_ROOT/phase2.done"
echo "FORMAL_ROOT=$FORMAL_ROOT"
