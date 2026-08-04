#!/usr/bin/env bash
# Phase 1 fixed: formal splits + OPT nsamples=128 Mag/Wanda/DSnoT
set -euo pipefail
cd "$HOME/projects/srtp-code-llm-pruning"
source scripts/setup/env.sh
export PATH="$HOME/.local/bin:$PATH"

WIN=/mnt/c/Users/Xile/.vscode/.code/SRTP-Project-Stage1
sync_file() {
  local rel="$1"
  mkdir -p "$(dirname "$rel")"
  cp "$WIN/$rel" "$rel"
  tr -d '\r' < "$rel" > "$rel.tmp" && mv "$rel.tmp" "$rel"
}

for rel in \
  scripts/data/create_formal_splits.py \
  workflows/audit/check_split_leakage.py \
  scripts/evaluate/generate_full_benchmark.py \
  scripts/evaluate/run_evalplus_passat1.py \
  methods/wanda/qwen_prune.py
 do
  sync_file "$rel"
done

echo "=== Create formal splits ==="
.venv-common/bin/python scripts/data/create_formal_splits.py --guide-count 32
.venv-common/bin/python workflows/audit/check_split_leakage.py \
  --output results/auxiliary/pan_full_eval/split_leakage_check_formal.json

TS=$(date +%Y%m%d_%H%M%S)
FORMAL_ROOT="results/raw/pan_formal_${TS}"
mkdir -p "$FORMAL_ROOT"
echo "$FORMAL_ROOT" | tee /tmp/pan_formal_root.txt
META="$FORMAL_ROOT/protocol.json"
cat > "$META" <<EOF
{
  "protocol": "pan_formal_v1",
  "opt_model": "facebook/opt-125m",
  "opt_nsamples": 128,
  "qwen_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
  "guide_count": 32,
  "seed": 0
}
EOF

run_wanda_opt() {
  local method="$1"
  local ratio="$2"
  local tag="$3"
  local run_dir="$FORMAL_ROOT/opt125m_${tag}"
  mkdir -p "$run_dir"
  echo "=== OPT $tag ==="
  (
    cd "$HOME/projects/srtp-code-llm-pruning"
    source .venv-wanda/bin/activate
    cd third_party/wanda
    python main_opt.py \
      --model facebook/opt-125m \
      --prune_method "$method" \
      --sparsity_ratio "$ratio" \
      --sparsity_type unstructured \
      --nsamples 128 \
      --save "$HOME/projects/srtp-code-llm-pruning/$run_dir/results" \
      --save_model "$HOME/projects/srtp-code-llm-pruning/$run_dir/pruned_model"
  ) > "$run_dir/stdout.log" 2> "$run_dir/stderr.log"
  echo "finished $tag"
  grep -E "sparsity sanity|ppl on" "$run_dir/stdout.log" | tail -10 || true
}

run_dsnot_opt() {
  local ratio="$1"
  local tag="$2"
  local run_dir="$FORMAL_ROOT/opt125m_${tag}"
  mkdir -p "$run_dir"
  echo "=== OPT $tag ==="
  (
    cd "$HOME/projects/srtp-code-llm-pruning"
    source .venv-dsnot/bin/activate
    cd third_party/dsnot
    python main.py \
      --model facebook/opt-125m \
      --model_type opt \
      --prune_method DSnoT \
      --initial_method wanda \
      --sparsity_ratio "$ratio" \
      --sparsity_type unstructured \
      --nsamples 128 \
      --max_cycle_time 50 \
      --update_threshold 0.1 \
      --pow_of_var_regrowing 1 \
      --output_results_file "$HOME/projects/srtp-code-llm-pruning/$run_dir/dsnot_results.txt" \
      --save_model "$HOME/projects/srtp-code-llm-pruning/$run_dir/pruned_model"
  ) > "$run_dir/stdout.log" 2> "$run_dir/stderr.log"
  echo "finished $tag"
  cat "$run_dir/dsnot_results.txt" || true
  grep -E "sparsity sanity|ppl on" "$run_dir/stdout.log" | tail -10 || true
}

run_wanda_opt magnitude 0.3 magnitude_s0p3
run_wanda_opt wanda 0.3 wanda_s0p3
run_dsnot_opt 0.3 dsnot_s0p3
run_wanda_opt magnitude 0.5 magnitude_s0p5
run_wanda_opt wanda 0.5 wanda_s0p5
run_dsnot_opt 0.5 dsnot_s0p5

echo "PHASE1_DONE" | tee "$FORMAL_ROOT/phase1.done"
echo "FORMAL_ROOT=$FORMAL_ROOT"
