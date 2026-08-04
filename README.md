# SRTP Project Stage 1

## Project goal
This repository is the unified Stage-1 workspace for code LLM pruning reproduction. It preserves raw evidence, documents comparable protocols, and separates successful, partial, blocked, and planned pruning routes before the remaining experiments continue.

## Quick navigation
- `data/benchmarks/`: smoke, R4-half, and auxiliary full-eval split files plus manifests.
- `methods/`: local adapters and method notes for the 12 selected pruning routes.
- `workflows/`: data, experiment, evaluation, aggregation, recovery, and audit entry points.
- `results/evidence/`: immutable raw evidence retained byte-for-byte.
- `results/status/`: dynamic registries for runs, scores, methods, data splits, and completion.
- `results/formal/r4_half/`: formal comparable R4-half score table.
- `results/auxiliary/full_eval/`: aggregate-only auxiliary comparison table.
- `docs/`: protocol, benchmark, and recovery explanations.

## Environment
Use WSL/Ubuntu for experiment execution. Environment locks are captured with `environment/setup/capture_environment_locks.py`; identical `pip freeze` content shares a lock by SHA, and missing local venvs are reported rather than silently treated as success.

## Data protocol
R4-half builders create guide/eval split directories and manifests under `data/benchmarks/r4_half/` or a caller supplied `--output-root`. Auxiliary full-eval builders create guide, full eval, heldout eval, and MBPP EvalPlus manifests under `data/benchmarks/auxiliary_full_eval/`. Dry-runs report the target plan without writing data.

## Method status
The method set is Flab-Pruner, LLM-Pruner, SliceGPT, LaCo, Magnitude, Wanda, DSnoT, OWL, SparseGPT, MaskLLM, Pruner-Zero, and FLAP. Current status is generated in `results/status/methods.csv` from run and score registries plus method metadata; it is not a fixed completion claim.

## Execution entry points
- Data materialization: `python workflows/data/build_r4_half_splits.py --output-root <path>`
- Registry refresh: `python workflows/aggregate/build_run_registry.py --write`
- Formal table refresh: `python workflows/aggregate/build_formal_r4_table.py --write`
- Completion audit: `python workflows/audit/check_stage1_completion.py --write`
- Environment locks: `python environment/setup/capture_environment_locks.py --write`

## Formal results
Formal comparable results are in `results/formal/r4_half/scores.csv`. Pilot rows, superseded baselines, and aggregate-only auxiliary values are excluded from the formal R4-half table.

## Known failures
Structured routes have evidence showing loadable artifacts can still fail code-generation quality gates. The detailed failure audit is `results/reports/failure_audit.md`; raw scores and run metadata are in `results/status/scores.csv` and `results/status/runs.csv`.

## Open items
Stage 1 repository integrity is maintained, but not every planned method has a successful complete run. Remaining experiments should follow the final execution plan after this technical patch is merged.

## Final completion state
The Keshu-owned completion pass records `owner_execution_closed=true` in `results/status/keshu_completion.json` for Flab-Pruner, LLM-Pruner, SliceGPT, and LaCo. The global audit in `results/status/completion_audit.json` remains `stage1_execution_closed=false` because this pass intentionally did not execute or modify other owners' methods.
