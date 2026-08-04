# Failure Audit

## Audit question
This audit identifies where Stage-1 pruning pipelines first lose executable code-generation behavior. It should be read with `results/status/runs.csv`, `results/status/scores.csv`, `results/formal/r4_half/scores.csv`, and `results/status/completion_audit.json`.

## Evidence map
`results/status/runs.csv` records each evidence directory, execution status, variant, and supersession decision. `results/status/scores.csv` records raw evidence-backed score summaries only. `results/formal/r4_half/scores.csv` filters those rows to comparable R4-half evidence. `results/auxiliary/full_eval/comparison.csv` is aggregate-only and is intentionally outside the raw score registry.

## Dense baselines
Corrected dense baselines supersede older baseline and recheck runs. The supersession field in `results/status/runs.csv` documents which runs are excluded from formal conclusions.

## Save and reload
Save/reload checks show that some failures are not serialization failures: artifacts can load and still fail downstream code-generation benchmarks.

## Pruning
Pruning can preserve structural validity while damaging generation quality. R4-half score rows should therefore be interpreted through both `quality_gate` and `result_completeness`.

## LoRA and merge
LoRA recovery and merged artifacts are retained where evidence exists. Merge success is not treated as benchmark success unless corresponding score rows pass the formal filters.

## Flab first break
Flab-Pruner evidence separates the local official structural adapter from the benchmark-activation experimental extension. The failure pattern is output-quality collapse after structured pruning and attempted recovery, not repository execution failure.

## LLM-Pruner first break
LLM-Pruner evidence separates fallback CodeLlama routes from the local Qwen adapter. Failures are represented in `results/status/scores.csv` and remain visible in formal R4-half filtering.

## SliceGPT first break
SliceGPT formal evidence includes partial benchmark completion and sliced-model constraints. The `benchmark_guided_sliced_model` variant is retained to avoid mixing it with dense or LoRA variants.

## Auxiliary aggregate limitation
Magnitude, Wanda, DSnoT, and OWL auxiliary results are aggregate-only. They are useful for tracking process outcomes, but they are not raw R4-half evidence and must not be presented as comparable formal scores.

## Raw completion failure taxonomy
The raw completion failure taxonomy is preserved in `results/reports/raw_completion_failure_taxonomy.csv`. It should be used with `results/status/scores.csv` when distinguishing syntax/runtime collapse from benchmark pass-rate degradation.

## Excluded explanations
Pilot rows, superseded baselines, and auxiliary full-eval rows are excluded from formal R4-half claims. These exclusions are mechanical registry rules, not manual cherry-picking.

## Current conclusion
The repository can support the next execution phase, but `results/status/completion_audit.json` does not claim all Stage-1 methods are successful. Remaining work is targeted reruns for planned and blocked methods plus focused recovery checks for partial methods.

## Final-completion pass
The final-completion pass added bounded evidence attempts for Flab benchmark activation, Magnitude, Wanda, SparseGPT, LaCo, DSnoT, OWL, MaskLLM, Pruner-Zero, FLAP, and SWE-bench-lite. These attempts close the planned-state gap without changing the older evidence. Flab activation now uses real forward hooks in code, but the Qwen1.5B activation run is blocked by local model/environment availability. Magnitude and Wanda have raw command evidence, but no new raw R4 score summaries; their historical aggregate-only rows remain separate from formal R4 results. SparseGPT and the coverage methods have probe/blocker evidence rather than completed formal code-model results.
