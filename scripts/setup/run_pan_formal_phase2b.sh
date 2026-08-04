#!/usr/bin/env bash
# Phase 2b: finish any missing prunes, then re-run ALL formal gens+evalplus with fixed prompts.
set -euo pipefail
cd "$HOME/projects/srtp-code-llm-pruning"
source scripts/setup/env.sh
export PATH="$HOME/.local/bin:$PATH"

WIN=/mnt/c/Users/Xile/.vscode/.code/SRTP-Project-Stage1
cp "$WIN/scripts/evaluate/generate_full_benchmark.py" scripts/evaluate/generate_full_benchmark.py
cp "$WIN/scripts/evaluate/run_evalplus_passat1.py" scripts/evaluate/run_evalplus_passat1.py
tr -d '\r' < scripts/evaluate/generate_full_benchmark.py > /tmp/g.py && mv /tmp/g.py scripts/evaluate/generate_full_benchmark.py
tr -d '\r' < scripts/evaluate/run_evalplus_passat1.py > /tmp/e.py && mv /tmp/e.py scripts/evaluate/run_evalplus_passat1.py

FORMAL_ROOT=results/raw/pan_formal_20260724_203248
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
  local model_dir="$run_dir/pruned/pruned_model"
  if [[ -f "$model_dir/config.json" ]]; then
    echo "skip prune $tag (exists)"
    return 0
  fi
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
  rm -f "$out/humaneval/evalplus/samples_eval_results.json"
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
  rm -f "$out/mbpp/evalplus/samples_eval_results.json"
  .venv-common/bin/python scripts/evaluate/run_evalplus_passat1.py \
    --predictions "$out/mbpp/predictions.jsonl" \
    --output-dir "$out/mbpp/evalplus" \
    --benchmark mbpp \
    --python-bin "$HOME/projects/srtp-code-llm-pruning/.venv-common/bin/python" || true
}

# Ensure all prunes exist
prune_one magnitude "$HE_GUIDE" 0.10 magnitude_he_s0p10
prune_one wanda "$HE_GUIDE" 0.10 wanda_he_s0p10
prune_one magnitude "$HE_GUIDE" 0.30 magnitude_he_s0p30
prune_one wanda "$HE_GUIDE" 0.30 wanda_he_s0p30
prune_one wanda "$MB_GUIDE" 0.10 wanda_mbpp_s0p10
prune_one magnitude "$MB_GUIDE" 0.10 magnitude_mbpp_s0p10

# Re-evaluate all compared models with fixed generation
eval_pair "$QWEN" dense_baseline
eval_pair "$FORMAL_ROOT/qwen15b_magnitude_he_s0p10/pruned/pruned_model" magnitude_he_s0p10
eval_pair "$FORMAL_ROOT/qwen15b_wanda_he_s0p10/pruned/pruned_model" wanda_he_s0p10
eval_pair "$FORMAL_ROOT/qwen15b_magnitude_he_s0p30/pruned/pruned_model" magnitude_he_s0p30
eval_pair "$FORMAL_ROOT/qwen15b_wanda_he_s0p30/pruned/pruned_model" wanda_he_s0p30
eval_pair "$FORMAL_ROOT/qwen15b_wanda_mbpp_s0p10/pruned/pruned_model" wanda_mbpp_s0p10
eval_pair "$FORMAL_ROOT/qwen15b_magnitude_mbpp_s0p10/pruned/pruned_model" magnitude_mbpp_s0p10

echo "PHASE2B_DONE" | tee "$FORMAL_ROOT/phase2b.done"
