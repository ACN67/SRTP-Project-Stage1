# R4 Keep80 Official Run Plan

This run plan is for Qwen2.5-Coder-3B-Instruct with three pruning methods:
Flab-Pruner, LLM-Pruner, and SliceGPT. The benchmark guide splits are used only
from `data/splits/*_half/guide.jsonl`; final scores use only
`data/splits/*_half/eval.jsonl`.

Benchmark evaluation must use `workflows/evaluate/run.sh`.

## Method Policy

| Method | Official logic retained | Qwen adaptation boundary |
|---|---|---|
| Flab-Pruner | Vendored Qwen2 structural `model.prune(config, stage)` path. | Qwen2.5 config compatibility and project manifests only. Guide rows are audited, not used to create custom importance scores. |
| LLM-Pruner | Upstream `MetaPruner`, official magnitude/Taylor importance, block/channel/layer modes, including the default protected early-layer range. | Qwen2 modules replace LLaMA modules in the official pruning graph; non-uniform Qwen artifacts use a custom loader before LoRA/eval. |
| SliceGPT | Upstream replace/fuse/rotate/slice flow and official state/config artifact format. | Local Qwen2 `ModelAdapter`/`LayerAdapter` plus loader for the official sliced artifact. |

## Shared Setup

```bash
set -euo pipefail
cd $SRTP_STAGE1_ROOT
source .venv-common/bin/activate

MODEL="Qwen/Qwen2.5-Coder-3B-Instruct"
RUN_TAG="qwen25c3b_keep80_official_$(date +%Y%m%d_%H%M%S)"
ROOT_OUT="results/evidence/${RUN_TAG}"
mkdir -p "$ROOT_OUT"

GUIDE_FILES=(
  --guide-file data/benchmarks/r4_half/humaneval/guide.jsonl
  --guide-file data/benchmarks/r4_half/mbpp_evalplus/guide.jsonl
  --guide-file data/benchmarks/r4_half/livecodebench/guide.jsonl
)
```

## Official LoRA Data

```bash
python workflows/recovery/build_distillation_data.py \
  --teacher-model "$MODEL" \
  --guide-file data/benchmarks/r4_half/humaneval/guide.jsonl \
  --guide-file data/benchmarks/r4_half/mbpp_evalplus/guide.jsonl \
  --guide-file data/benchmarks/r4_half/livecodebench/guide.jsonl \
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
python methods/flab_pruner/qwen_prune.py \
  --model "$MODEL" \
  "${GUIDE_FILES[@]}" \
  --save-dir "$ROOT_OUT/flabpruner_keep80" \
  --stage top \
  --prune-ratio 0.20 \
  --max-guide-samples 999999 \
  --dtype fp16 \
  --local-files-only \
  --prune-on-cpu

python workflows/recovery/train_lora.py \
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

python workflows/recovery/merge_lora.py \
  --base-model "$ROOT_OUT/flabpruner_keep80/pruned_model" \
  --adapter "$ROOT_OUT/flabpruner_keep80_lora/lora_adapter" \
  --out-dir "$ROOT_OUT/flabpruner_keep80_merged" \
  --dtype fp16 \
  --device cuda:0
```

## LLM-Pruner Keep80

```bash
python methods/llm_pruner/qwen_prune.py \
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

python workflows/recovery/train_lora.py \
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

python workflows/recovery/merge_lora.py \
  --base-model "$ROOT_OUT/llmpruner_keep80/pruned_model" \
  --adapter "$ROOT_OUT/llmpruner_keep80_lora/lora_adapter" \
  --out-dir "$ROOT_OUT/llmpruner_keep80_merged" \
  --dtype fp16 \
  --device cuda:0
```

## SliceGPT Keep80

```bash
python methods/slicegpt/qwen_prune.py \
  --model "$MODEL" \
  --out-dir "$ROOT_OUT/slicegpt_keep80" \
  --dtype fp16 \
  --device cuda:0 \
  --local-files-only \
  --cal-guide-file data/benchmarks/r4_half/humaneval/guide.jsonl \
  --cal-guide-file data/benchmarks/r4_half/mbpp_evalplus/guide.jsonl \
  --cal-guide-file data/benchmarks/r4_half/livecodebench/guide.jsonl \
  --cal-guide-limit-per-file 999999 \
  --cal-nsamples 999999 \
  --cal-batch-size 1 \
  --cal-max-seqlen 512 \
  --sparsity 0.20 \
  --round-interval 128 \
  --final-orientation pca \
  --save-sliced-state \
  --save-hf-files

python workflows/recovery/train_lora.py \
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

  workflows/evaluate/run.sh --benchmark humaneval --model "$MODEL_DIR" --split data/benchmarks/r4_half/humaneval/eval.jsonl --out-dir "$ROOT_OUT/${METHOD}_eval/humaneval"
  workflows/evaluate/run.sh --benchmark mbpp_evalplus --model "$MODEL_DIR" --split data/benchmarks/r4_half/mbpp_evalplus/eval.jsonl --out-dir "$ROOT_OUT/${METHOD}_eval/mbpp_evalplus"
  workflows/evaluate/run.sh --benchmark livecodebench --model "$MODEL_DIR" --split data/benchmarks/r4_half/livecodebench/eval.jsonl --out-dir "$ROOT_OUT/${METHOD}_eval/livecodebench" --lcb-release release_v1 --lcb-config release_latest --lcb-lm-style CodeQwenInstruct
done
```

SliceGPT evaluates the official sliced model with the LoRA adapter loaded:

```bash
SLICE_MODEL="$ROOT_OUT/slicegpt_keep80/sliced_model"
SLICE_ADAPTER="$ROOT_OUT/slicegpt_keep80_lora/lora_adapter"

workflows/evaluate/run.sh --benchmark humaneval --model "$SLICE_MODEL" --adapter "$SLICE_ADAPTER" --slicegpt-base-model "$MODEL" --slicegpt-sparsity 0.20 --slicegpt-round-interval 128 --split data/benchmarks/r4_half/humaneval/eval.jsonl --out-dir "$ROOT_OUT/slicegpt_eval/humaneval"
workflows/evaluate/run.sh --benchmark mbpp_evalplus --model "$SLICE_MODEL" --adapter "$SLICE_ADAPTER" --slicegpt-base-model "$MODEL" --slicegpt-sparsity 0.20 --slicegpt-round-interval 128 --split data/benchmarks/r4_half/mbpp_evalplus/eval.jsonl --out-dir "$ROOT_OUT/slicegpt_eval/mbpp_evalplus"
workflows/evaluate/run.sh --benchmark livecodebench --model "$SLICE_MODEL" --adapter "$SLICE_ADAPTER" --slicegpt-base-model "$MODEL" --slicegpt-sparsity 0.20 --slicegpt-round-interval 128 --split data/benchmarks/r4_half/livecodebench/eval.jsonl --out-dir "$ROOT_OUT/slicegpt_eval/livecodebench" --lcb-release release_v1 --lcb-config release_latest --lcb-lm-style CodeQwenInstruct
```
