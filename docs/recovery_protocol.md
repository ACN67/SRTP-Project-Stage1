# Recovery Protocol

## LoRA
LoRA recovery is applicable after a pruned model can be loaded and trained without breaking the model contract.

## distillation
Distillation data must come from allowed guide or training sources and must not leak held-out evaluation answers.

## adapter
Adapter artifacts are external artifacts when large. Their status is recorded in `results/status/artifacts.csv`.

## merge
Merge checks verify that adapter weights can be combined with the pruned model and reloaded.

## merged evaluation
Merged evaluation uses the same benchmark protocol as its corresponding run category.

## SliceGPT sliced artifact
SliceGPT sliced artifacts are method-specific model states and are marked partial when evaluation did not cover all intended benchmarks.

## recovery status
Flab-Pruner, LLM-Pruner, and SliceGPT have recovery evidence with quality limitations. Candidate methods without first-stage recovery evidence remain planned or not applicable.

## Evidence closure
Recovery is complete only when adapter, merge, reload, and evaluation evidence are all indexed and linked from the registry.
