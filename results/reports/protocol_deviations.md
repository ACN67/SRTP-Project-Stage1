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
