# Protocol Deviations

## fallback
LLM-Pruner CodeLlama evidence is fallback evidence and not the same as a successful local official Qwen adapter.

## SliceGPT requested vs actual reduction
SliceGPT reduction and partial evaluation must be interpreted with actual recorded task counts and artifact state.

## old vs corrected baseline
Old Qwen baseline and recheck runs are superseded by corrected official baseline evidence.

## pilot
The keep80 pilot is a 5-task run and is excluded from formal R4-half tables.

## auxiliary full-eval boundary
auxiliary_full_eval uses guide subset full-eval semantics and is not equivalent to R4-half.

## partial benchmark
Partial benchmark rows remain in the full registry with exact task counts.

## aggregate-only
Aggregate-only evidence is retained for comparison but does not imply raw evidence is present.

## experimental importance
Flab benchmark activation experimental importance is not upstream official behavior; structural mode remains the local official adapter route.

## owner-scoped completion
The Keshu-owned closure file `results/status/keshu_completion.json` closes only Flab-Pruner, LLM-Pruner, SliceGPT, and LaCo checks. It intentionally does not set global Stage-1 execution closed.

## Flab external masks
The benchmark-activation mask path is experimental and blocked because the vendored Qwen prune API accepts config/stage targets rather than a verified external per-channel mask schema.

## LaCo smoke boundary
The LaCo tiny core smoke demonstrates layer similarity scoring and collapse mechanics only. It is diagnostic evidence and not formal CodeLlama R4 evidence.

## Flab capped-32 calibration
The benchmark-guided HE, MBPP, and LCB variants use capped-32 guide calibration and fixed 20-task non-collapse gates. These are pilot quality gates and are not full-guide formal results.
