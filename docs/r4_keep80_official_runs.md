# R4 Keep80 Official Run Plan

This run plan is for Qwen2.5-Coder-3B-Instruct with three pruning methods:
Flab-Pruner, LLM-Pruner, and SliceGPT. The benchmark guide splits are used only
from `data/splits/*_half/guide.jsonl`; final scores use only
`data/splits/*_half/eval.jsonl`.

Benchmark evaluation must use `scripts/eval/run_official_eval.sh`.

## Method Policy

| Method | Official logic retained | Qwen adaptation boundary |
|---|---|---|
| Flab-Pruner | Vendored Qwen2 structural `model.prune(config, stage)` path. | Qwen2.5 config compatibility and project manifests only. Guide rows are audited, not used to create custom importance scores. |
| LLM-Pruner | Upstream `MetaPruner`, official magnitude/Taylor importance, block/channel/layer modes, including the default protected early-layer range. | Qwen2 modules replace LLaMA modules in the official pruning graph; non-uniform Qwen artifacts use a custom loader before LoRA/eval. |
| SliceGPT | Upstream replace/fuse/rotate/slice flow and official state/config artifact format. | Local Qwen2 `ModelAdapter`/`LayerAdapter` plus loader for the official sliced artifact. |

## Shared Setup

```bash
set -euo pipefail
cd /home/keshu/projects/srtp-code-llm-pruning
source .venv-common/bin/activate

MODEL="Qwen/Qwen2.5-Coder-3B-Instruct"
RUN_TAG="qwen25c3b_keep80_official_$(date +%Y%m%d_%H%M%S)"
ROOT_OUT="results/raw/${RUN_TAG}"
mkdir -p "$ROOT_OUT"

GUIDE_FILES=(
  --guide-file data/splits/humaneval_half/guide.jsonl
  --guide-file data/splits/mbpp_evalplus_half/guide.jsonl
  --guide-file data/splits/livecodebench_half/guide.jsonl
)
```

## Official LoRA Data

```bash
python scripts/recover/create_distill_dataset.py \
  --teacher-model "$MODEL" \
  --guide-file data/splits/humaneval_half/guide.jsonl \
  --guide-file data/splits/mbpp_evalplus_half/guide.jsonl \
  --guide-file data/splits/livecodebench_half/guide.jsonl \
  --out-dir "$ROOT_OUT/shared_lora_data" \
  --max-new-tokens 512 \
  --dtype fp16 \
  --device cuda:0 \
  --local-files-only \
  --lcb-release release_v1 \
  --lcb-config release_latest \
  --lcb-lm-style CodeQwenInstruct
```

## Flab-Pruner Keep80

```bash
python scripts/adapt/flab_qwen_official.py \
  --model "$MODEL" \
  "${GUIDE_FILES[@]}" \
  --save-dir "$ROOT_OUT/flabpruner_keep80" \
  --stage top \
  --prune-ratio 0.20 \
  --max-guide-samples 999999 \
  --dtype fp16 \
  --local-files-only \
  --prune-on-cpu

python scripts/recover/train_lora_recovery.py \
  --base-model "$ROOT_OUT/flabpruner_keep80/pruned_model" \
  --train-file "$ROOT_OUT/shared_lora_data/distill_train.jsonl" \
  --out-dir "$ROOT_OUT/flabpruner_keep80_lora" \
  --dtype fp16 \
  --device cuda:0 \
  --local-files-only \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 8 \
  --max-length 512

python scripts/recover/merge_lora_model.py \
  --base-model "$ROOT_OUT/flabpruner_keep80/pruned_model" \
  --adapter "$ROOT_OUT/flabpruner_keep80_lora/lora_adapter" \
  --out-dir "$ROOT_OUT/flabpruner_keep80_merged" \
  --dtype fp16 \
  --device cuda:0
```

## LLM-Pruner Keep80

```bash
python scripts/adapt/llmpruner_qwen_official.py \
  --model "$MODEL" \
  --out-dir "$ROOT_OUT/llmpruner_keep80" \
  --mode block_wise \
  --pruning-ratio 0.20 \
  --pruner-type taylor \
  --taylor param_first \
  "${GUIDE_FILES[@]}" \
  --guide-limit-per-file 999999 \
  --importance-max-length 256 \
  --dtype fp32 \
  --device cpu \
  --local-files-only \
  --save-model

python scripts/recover/train_lora_recovery.py \
  --base-model "$ROOT_OUT/llmpruner_keep80/pruned_model" \
  --llmpruner-base-model "$MODEL" \
  --train-file "$ROOT_OUT/shared_lora_data/distill_train.jsonl" \
  --out-dir "$ROOT_OUT/llmpruner_keep80_lora" \
  --dtype fp16 \
  --device cuda:0 \
  --local-files-only \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 8 \
  --max-length 512

python scripts/recover/merge_lora_model.py \
  --base-model "$ROOT_OUT/llmpruner_keep80/pruned_model" \
  --adapter "$ROOT_OUT/llmpruner_keep80_lora/lora_adapter" \
  --out-dir "$ROOT_OUT/llmpruner_keep80_merged" \
  --dtype fp16 \
  --device cuda:0
```

## SliceGPT Keep80

```bash
python scripts/adapt/slicegpt_qwen_official.py \
  --model "$MODEL" \
  --out-dir "$ROOT_OUT/slicegpt_keep80" \
  --dtype fp16 \
  --device cuda:0 \
  --local-files-only \
  --cal-guide-file data/splits/humaneval_half/guide.jsonl \
  --cal-guide-file data/splits/mbpp_evalplus_half/guide.jsonl \
  --cal-guide-file data/splits/livecodebench_half/guide.jsonl \
  --cal-guide-limit-per-file 999999 \
  --cal-nsamples 999999 \
  --cal-batch-size 1 \
  --cal-max-seqlen 512 \
  --sparsity 0.20 \
  --round-interval 128 \
  --final-orientation pca \
  --save-sliced-state \
  --save-hf-files

python scripts/recover/train_lora_recovery.py \
  --base-model "$ROOT_OUT/slicegpt_keep80/sliced_model" \
  --slicegpt-base-model "$MODEL" \
  --slicegpt-sparsity 0.20 \
  --slicegpt-round-interval 128 \
  --train-file "$ROOT_OUT/shared_lora_data/distill_train.jsonl" \
  --out-dir "$ROOT_OUT/slicegpt_keep80_lora" \
  --dtype fp16 \
  --device cuda:0 \
  --local-files-only \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 8 \
  --max-length 512
```

## Official Eval

Flab-Pruner and LLM-Pruner evaluate merged LoRA models:

```bash
for METHOD in flabpruner llmpruner; do
  MODEL_DIR="$ROOT_OUT/${METHOD}_keep80_merged"

  scripts/eval/run_official_eval.sh --benchmark humaneval --model "$MODEL_DIR" --split data/splits/humaneval_half/eval.jsonl --out-dir "$ROOT_OUT/${METHOD}_eval/humaneval"
  scripts/eval/run_official_eval.sh --benchmark mbpp_evalplus --model "$MODEL_DIR" --split data/splits/mbpp_evalplus_half/eval.jsonl --out-dir "$ROOT_OUT/${METHOD}_eval/mbpp_evalplus"
  scripts/eval/run_official_eval.sh --benchmark livecodebench --model "$MODEL_DIR" --split data/splits/livecodebench_half/eval.jsonl --out-dir "$ROOT_OUT/${METHOD}_eval/livecodebench" --lcb-release release_v1 --lcb-config release_latest --lcb-lm-style CodeQwenInstruct
done
```

SliceGPT evaluates the official sliced model with the LoRA adapter loaded:

```bash
SLICE_MODEL="$ROOT_OUT/slicegpt_keep80/sliced_model"
SLICE_ADAPTER="$ROOT_OUT/slicegpt_keep80_lora/lora_adapter"

scripts/eval/run_official_eval.sh --benchmark humaneval --model "$SLICE_MODEL" --adapter "$SLICE_ADAPTER" --slicegpt-base-model "$MODEL" --slicegpt-sparsity 0.20 --slicegpt-round-interval 128 --split data/splits/humaneval_half/eval.jsonl --out-dir "$ROOT_OUT/slicegpt_eval/humaneval"
scripts/eval/run_official_eval.sh --benchmark mbpp_evalplus --model "$SLICE_MODEL" --adapter "$SLICE_ADAPTER" --slicegpt-base-model "$MODEL" --slicegpt-sparsity 0.20 --slicegpt-round-interval 128 --split data/splits/mbpp_evalplus_half/eval.jsonl --out-dir "$ROOT_OUT/slicegpt_eval/mbpp_evalplus"
scripts/eval/run_official_eval.sh --benchmark livecodebench --model "$SLICE_MODEL" --adapter "$SLICE_ADAPTER" --slicegpt-base-model "$MODEL" --slicegpt-sparsity 0.20 --slicegpt-round-interval 128 --split data/splits/livecodebench_half/eval.jsonl --out-dir "$ROOT_OUT/slicegpt_eval/livecodebench" --lcb-release release_v1 --lcb-config release_latest --lcb-lm-style CodeQwenInstruct
```
