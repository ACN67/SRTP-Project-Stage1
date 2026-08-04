# Failure Audit

## Audit question
The audit asks where first-stage pruning pipelines first lose executable code-generation behavior.

## dense baseline
Corrected dense baselines are preferred over old baseline and recheck runs in formal conclusions.

## save/reload
Save/reload checks show that some failures are not simple serialization failures.

## pruning
Pruning can preserve loadability while damaging generation quality.

## LoRA
LoRA recovery evidence did not fully restore quality-gate behavior for the structured routes.

## merge
Merge evidence is retained where present, but merge success alone is not benchmark success.

## evaluation
Evaluation rows retain actual task counts, including partial and pilot runs.

## Flab first break
Flab first break evidence points to output collapse after structured pruning and recovery attempts.

## LLM-Pruner first break
LLM-Pruner first break evidence separates fallback CodeLlama routes from local Qwen adapter behavior.

## SliceGPT first break
SliceGPT first break evidence includes partial benchmark completion and sliced artifact constraints.

## cross-method commonality
The common pattern is that structural validity does not guarantee code-generation quality.

## raw completion failure taxonomy
The raw completion failure taxonomy is preserved in `results/reports/raw_completion_failure_taxonomy.csv`.

## evidence limitation
Auxiliary aggregate-only rows and missing raw auxiliary evidence cannot be treated as formal R4-half proof.

## excluded explanations
Pilot rows, superseded baselines, and auxiliary full-eval rows are excluded from formal conclusions.

## conclusions not supported
The repository cannot claim a universally best pruning method or full completion of all 12 routes.

## next validation
Next steps are targeted reruns for blocked/planned methods and focused recovery checks for partial methods.
