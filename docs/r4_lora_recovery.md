# R4 LoRA Recovery Runbook

R4 uses the fixed guide/eval split protocol:

- Pruning guide: `data/splits/*_half/guide.jsonl`
- Evaluation: `data/splits/*_half/eval.jsonl`
- Recovery: LoRA distillation on the same guide half only

The LoRA stage is not a reduced method variant. It is parameter-efficient recovery: the pruned base model is frozen and low-rank adapters are trained on teacher-generated guide-half completions.

## Teacher Datasets

Use one teacher per model family.

Qwen family:

```bash
cd ~/projects/srtp-code-llm-pruning
source scripts/setup/env.sh

RUN_DIR="results/raw/qwen25c3b_r4_distill_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

.venv-common/bin/python scripts/recover/create_distill_dataset.py \
  --teacher-model Qwen/Qwen2.5-Coder-3B-Instruct \
  --guide-file data/splits/humaneval_half/guide.jsonl \
  --guide-file data/splits/mbpp_evalplus_half/guide.jsonl \
  --guide-file data/splits/livecodebench_half/guide.jsonl \
  --out-dir "$RUN_DIR" \
  --max-new-tokens 256 \
  --dtype fp16 \
  --device cuda:0 \
  2>&1 | tee "$RUN_DIR/create_distill.log"
```

CodeLlama family:

```bash
cd ~/projects/srtp-code-llm-pruning
source scripts/setup/env.sh

SNAPSHOT="$(find ~/.cache/huggingface/hub/models--codellama--CodeLlama-7b-hf/snapshots -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
RUN_DIR="results/raw/codellama7b_r4_distill_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

.venv-common/bin/python scripts/recover/create_distill_dataset.py \
  --teacher-model "$SNAPSHOT" \
  --guide-file data/splits/humaneval_half/guide.jsonl \
  --guide-file data/splits/mbpp_evalplus_half/guide.jsonl \
  --guide-file data/splits/livecodebench_half/guide.jsonl \
  --out-dir "$RUN_DIR" \
  --max-new-tokens 256 \
  --dtype fp16 \
  --device cuda:0 \
  2>&1 | tee "$RUN_DIR/create_distill.log"
```

## Train LoRA

Example for a pruned model:

```bash
cd ~/projects/srtp-code-llm-pruning
source scripts/setup/env.sh

BASE_MODEL="PATH_TO_PRUNED_MODEL"
TRAIN_FILE="PATH_TO_DISTILL_RUN/distill_train.jsonl"
RUN_DIR="results/raw/METHOD_MODEL_r4_lora_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

.venv-common/bin/python scripts/recover/train_lora_recovery.py \
  --base-model "$BASE_MODEL" \
  --train-file "$TRAIN_FILE" \
  --out-dir "$RUN_DIR" \
  --rank 8 \
  --alpha 16 \
  --dropout 0 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 8 \
  --max-length 512 \
  --lr 2e-4 \
  --dtype fp16 \
  --device cuda:0 \
  2>&1 | tee "$RUN_DIR/train.log"
```

## Merge LoRA

R4 formal evaluation uses merged recovered models, not dynamic adapter evaluation. The adapter
is kept only as an intermediate recovery artifact until the merged model is produced.

```bash
cd ~/projects/srtp-code-llm-pruning
source scripts/setup/env.sh

BASE_MODEL="PATH_TO_PRUNED_MODEL"
ADAPTER="PATH_TO_LORA_RUN/lora_adapter"
MERGED_MODEL="results/raw/METHOD_MODEL_r4_recovered_merged_$(date +%Y%m%d_%H%M%S)/merged_model"
mkdir -p "$MERGED_MODEL"

.venv-common/bin/python scripts/recover/merge_lora_model.py \
  --base-model "$BASE_MODEL" \
  --adapter "$ADAPTER" \
  --out-dir "$MERGED_MODEL" \
  --dtype fp16 \
  --device cuda:0
```

## Evaluate Merged Model

Use the official benchmark wrapper without `--adapter`.

```bash
scripts/eval/run_official_eval.sh \
  --benchmark humaneval \
  --model "$MERGED_MODEL" \
  --split data/splits/humaneval_half/eval.jsonl \
  --out-dir "$EVAL_DIR" \
  --max-new-tokens 256 \
  --dtype fp16
```

## R4 Method Mapping

Flab-Pruner/Qwen:

- default pruned base: `results/raw/flab_qwen25c3b_r4_default_keep80_prune_*/flab_qwen25c3b_r4_default_keep80/pruned_model`
- benchmark-guided pruned base: `results/raw/flab_qwen25c3b_r4_benchguided_keep80_prune_*/flab_qwen25c3b_r4_benchguided_keep80/pruned_model`
- teacher dataset: Qwen family dataset

LLM-Pruner/CodeLlama:

- pruned base: R4 LLM-Pruner output directory
- teacher dataset: CodeLlama family dataset

SliceGPT/CodeLlama:

- pruned base: R4 SliceGPT output directory
- teacher dataset: CodeLlama family dataset
