#!/usr/bin/env bash
# Phase 2c: full formal Pass@1 with fixed HE prompting + evalplus MBPP split
set -euo pipefail
cd "$HOME/projects/srtp-code-llm-pruning"
source scripts/setup/env.sh
export PATH="$HOME/.local/bin:$PATH"

WIN=/mnt/c/Users/Xile/.vscode/.code/SRTP-Project-Stage1
for rel in \
  scripts/evaluate/generate_full_benchmark.py \
  scripts/evaluate/run_evalplus_passat1.py \
  scripts/data/create_mbpp_evalplus_split.py \
  scripts/audit/aggregate_pan_formal_results.py
 do
  cp "$WIN/$rel" "$rel"
  tr -d '\r' < "$rel" > "$rel.tmp" && mv "$rel.tmp" "$rel"
done

.venv-common/bin/python scripts/data/create_mbpp_evalplus_split.py
# sync mbpp_evalplus to windows later

FORMAL_ROOT=results/raw/pan_formal_20260724_203248
QWEN=Qwen/Qwen2.5-Coder-1.5B-Instruct
HE_EVAL=data/splits/humaneval_formal/eval.jsonl
MB_EVAL=data/splits/mbpp_evalplus/eval.jsonl

eval_pair() {
  local model_path="$1"
  local tag="$2"
  local out="$FORMAL_ROOT/eval_${tag}"
  mkdir -p "$out"

  echo "=== generate HE $tag ==="
  .venv-common/bin/python scripts/evaluate/generate_full_benchmark.py \
    --model "$model_path" --split "$HE_EVAL" --output-dir "$out/humaneval" \
    --benchmark humaneval --dtype bf16 --device-map cuda:0 --max-new-tokens 512

  echo "=== evalplus HE $tag ==="
  rm -f "$out/humaneval/evalplus/samples_eval_results.json"
  .venv-common/bin/python scripts/evaluate/run_evalplus_passat1.py \
    --predictions "$out/humaneval/predictions.jsonl" \
    --output-dir "$out/humaneval/evalplus" \
    --benchmark humaneval \
    --python-bin "$HOME/projects/srtp-code-llm-pruning/.venv-common/bin/python" || true

  echo "=== generate MBPP $tag ==="
  .venv-common/bin/python scripts/evaluate/generate_full_benchmark.py \
    --model "$model_path" --split "$MB_EVAL" --output-dir "$out/mbpp" \
    --benchmark mbpp --dtype bf16 --device-map cuda:0 --max-new-tokens 512

  echo "=== evalplus MBPP $tag ==="
  rm -f "$out/mbpp/evalplus/samples_eval_results.json"
  .venv-common/bin/python scripts/evaluate/run_evalplus_passat1.py \
    --predictions "$out/mbpp/predictions.jsonl" \
    --output-dir "$out/mbpp/evalplus" \
    --benchmark mbpp \
    --python-bin "$HOME/projects/srtp-code-llm-pruning/.venv-common/bin/python" || true
}

# Ensure prunes exist (from earlier)
for tag in magnitude_he_s0p10 wanda_he_s0p10 magnitude_he_s0p30 wanda_he_s0p30 wanda_mbpp_s0p10 magnitude_mbpp_s0p10; do
  if [[ ! -f "$FORMAL_ROOT/qwen15b_${tag}/pruned/pruned_model/config.json" ]]; then
    echo "MISSING prune $tag" >&2
    exit 1
  fi
  echo "have prune $tag"
done

eval_pair "$QWEN" dense_baseline
eval_pair "$FORMAL_ROOT/qwen15b_magnitude_he_s0p10/pruned/pruned_model" magnitude_he_s0p10
eval_pair "$FORMAL_ROOT/qwen15b_wanda_he_s0p10/pruned/pruned_model" wanda_he_s0p10
eval_pair "$FORMAL_ROOT/qwen15b_magnitude_he_s0p30/pruned/pruned_model" magnitude_he_s0p30
eval_pair "$FORMAL_ROOT/qwen15b_wanda_he_s0p30/pruned/pruned_model" wanda_he_s0p30
eval_pair "$FORMAL_ROOT/qwen15b_wanda_mbpp_s0p10/pruned/pruned_model" wanda_mbpp_s0p10
eval_pair "$FORMAL_ROOT/qwen15b_magnitude_mbpp_s0p10/pruned/pruned_model" magnitude_mbpp_s0p10

.venv-common/bin/python scripts/audit/aggregate_pan_formal_results.py \
  --formal-root "$FORMAL_ROOT" \
  --output results/tables/pan_formal_comparison.csv

echo "PHASE2C_DONE" | tee "$FORMAL_ROOT/phase2c.done"
