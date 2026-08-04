# Cross-Method Failure Audit

Local commit: `9117e9c340e1c56bc5bd6560c2418688da430853`. Submodule commits: Flab `e35e7fc4560369f993d736df1ef5429a74ca6983`, LLM-Pruner `128a07d977f9b205d60ab14cfbc6a78f8a8e39d2`, SliceGPT `6b12cdee6ad51791d7c776b3a046bc408b9e77e9`.

## Status

This is an evidence-first checkpoint, not a final root-cause verdict. Existing completed full runs cover S0 dense and final pruned+LoRA checkpoints, but do not yet cover S1-S5 smoke stages required to identify the first break.

## Component Independence

| 组件 | FlabPruner | LLM-Pruner | SliceGPT | 是否共享代码 |
|---|---|---|---|---|
| Dense model loader | `AutoConfig` + vendored `Qwen2ForCausalLM` in `methods/flab_pruner/qwen_prune.py` | `AutoModelForCausalLM.from_pretrained` in `methods/llm_pruner/qwen_prune.py` | `AutoConfig` + local `Qwen2ModelAdapter` in `methods/slicegpt/qwen_prune.py` | no |
| Tokenizer loader | `AutoTokenizer.from_pretrained` | `AutoTokenizer.from_pretrained` | `AutoTokenizer.from_pretrained` | pattern shared, separate calls |
| Pruning implementation | vendored Flab Qwen2 `model.prune(config, stage)` | vendored LLM-Pruner `MetaPruner` + `TaylorImportance` | local Qwen adapter over SliceGPT replace/fuse/rotate/slice | no |
| Save checkpoint | HF `save_pretrained` after Flab prune | HF `save_pretrained` plus `llmpruner_qwen_shapes.json` | SliceGPT state/config + HF files | no |
| Reload checkpoint | normal HF AutoModel | `load_llmpruner_qwen_model` | `load_sliced_qwen_model` | no |
| LoRA dataset builder | `workflows/recovery/build_distillation_data.py` | same | same | yes |
| PEFT configuration | `workflows/recovery/train_lora.py` | same with custom loader | same with custom loader | mostly yes |
| LoRA trainer | `workflows/recovery/train_lora.py` | same | same | yes |
| Merge adapter | `workflows/recovery/merge_lora.py` | not used final | not used final | partly |
| HumanEval evaluator | `workflows/evaluate/run.sh` | same | same | yes |
| MBPP evaluator | `workflows/evaluate/run.sh` | same | same | yes |
| LiveCodeBench evaluator | `workflows/evaluate/run.sh` | same | same | yes |


## Existing Final Scores

Dense Qwen1.5B baseline: HumanEval 18/82 = 0.2195; MBPP 115/224 = 0.5134; LiveCodeBench 28/200 = 0.14.

Flab final: HumanEval 0/82 = 0.0; MBPP 1/224 = 0.00446; LiveCodeBench 0/200 = 0.0.

LLM-Pruner final: HumanEval 2/82 = 0.02439; MBPP 1/224 = 0.00446; LiveCodeBench 0/200 = 0.0.

SliceGPT final: HumanEval 0/82 = 0.0; MBPP 2/224 = 0.00893; LiveCodeBench 0/200 = 0.0.

## Existing Evidence

- Flab path: `results/evidence/r4_half/flabpruner_qwen25c15b_official_keep80_20260730_015031`. Actual parameter ratio is `0.8939371828221396`. The run records that guide prompts are not used for Flab importance.
- LLM-Pruner path: `results/evidence/r4_half/llmpruner_qwen25c15b_official_keep80_gpu_20260730_105340`. Taylor used 436 guide samples; requested ratio `0.28`; actual parameter ratio `0.8202547406077543`.
- SliceGPT path: `results/evidence/r4_half/slicegpt_qwen25c15b_official_keep80_gpu_20260731_011001`. Local Qwen2 adapter run succeeded, but sparsity=0 invariant is still missing.
- Saved solutions show different failure modes: Flab often repeats invalid asserts/hits max; LLM-Pruner often emits natural-language explanations; SliceGPT often emits syntactically invalid code. See `reports/raw_completion_failure_taxonomy.csv`.

## Current Root-Cause Boundary

The final pruned+LoRA checkpoints are much worse than dense under the shared official evaluator. It is not yet justified to conclude all three pruning algorithms intrinsically fail for code models, because dense round-trip, dense LoRA, no-op method transforms, prune-before-save, and prune-save-reload smoke stages are missing.

Most likely causes to test first:

1. Method-specific Qwen architecture adaptation errors. Evidence: LLM-Pruner and SliceGPT require custom Qwen loaders/adapters; SliceGPT zero-sparsity invariant is untested.
2. Pruning structure/ratio mismatch. Evidence: Flab actual keep differs from the rough structure estimate; final generations show EOS/hit-max instability.
3. Shared recovery/eval chain may contribute, but is not proven. Evidence: all three use common LoRA dataset/trainer/evaluator; dense LoRA control is missing.

## Missing Evidence

Required before final conclusion: S1 dense save/reload, S2/S3 dense LoRA adapter/merged, Flab no-op/low-ratio, LLM-Pruner ratio=0 or official-model smoke, SliceGPT sparsity=0 invariant, adapter-vs-merged logit comparisons.

## Next Minimal Validation Order

Run only smoke, not full benchmark: dense round-trip 20 prompts; dense LoRA adapter and merged 20 prompts; SliceGPT sparsity=0 20 prompts; Flab pruned-before-LoRA if reconstructable; LLM-Pruner pruned without adapter; then adapter stages.
