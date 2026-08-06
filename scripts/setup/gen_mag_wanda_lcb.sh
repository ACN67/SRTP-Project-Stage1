#!/usr/bin/env bash
# Generate Mag/Wanda LCB while dense LCB score may still be downloading.
set -euo pipefail
cd /home/xaillor/projects/srtp-code-llm-pruning
ROOT=results/raw/pan_half_align_20260806_174220
FORMAL=results/raw/pan_formal_20260724_203248
PY=.venv-common/bin/python
MAX_NEW_TOKENS=256

for spec in \
  "magnitude_he_s0p10|$FORMAL/qwen15b_magnitude_he_s0p10/pruned/pruned_model" \
  "wanda_he_s0p10|$FORMAL/qwen15b_wanda_he_s0p10/pruned/pruned_model"
do
  tag="${spec%%|*}"
  model="${spec#*|}"
  out="$ROOT/$tag/livecodebench"
  mkdir -p "$out"
  if [[ -f "$out/predictions.jsonl" ]]; then
    echo "skip gen $tag"
    continue
  fi
  echo "=== gen LCB $tag ==="
  "$PY" scripts/evaluate/generate_code_split.py \
    --model "$model" --benchmark livecodebench \
    --split data/splits/livecodebench_half/eval.jsonl \
    --output-dir "$out" --max-new-tokens "$MAX_NEW_TOKENS" --device-map cuda:0
done
echo LCB_GEN_MAG_WANDA_DONE
