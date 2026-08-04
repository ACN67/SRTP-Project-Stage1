# Stage 1 Final Summary

## Goal
Stage 1 aimed to close reproduction coverage for 12 pruning methods while preserving raw evidence and separating formal R4 scores from auxiliary aggregate results.

## Execution closure
`results/status/completion_audit.json` reports `stage1_execution_closed=true` and `stage1_all_methods_successful=false`. The latter is expected: several methods remain quality-gate failures, partial, or blocked with evidence.

## Completed methods
Flab-Pruner, LLM-Pruner, and SliceGPT have completed primary routes. Their quality gates remain failed or under review, so completion is not a claim of benchmark superiority.

## Partial methods
Magnitude and Wanda retain auxiliary aggregate evidence and now have raw formal attempt directories. They do not yet have new raw R4 score summaries for all formal benchmarks.

## Blocked methods
LaCo, DSnoT, OWL, SparseGPT, MaskLLM, Pruner-Zero, and FLAP now have bounded attempt evidence and are no longer planned-only methods.

## Formal table
The formal R4-half table remains `results/formal/r4_half/scores.csv`. It excludes pilots, superseded baselines, and aggregate-only auxiliary results.

## Quality gates
Known quality-gate failures are tracked in `results/status/completion_audit.json` and include Flab-Pruner, LLM-Pruner, and SliceGPT.

## Benchmark policy
HumanEval, MBPP EvalPlus, and LiveCodeBench R4-half remain the formal benchmark set. SWE-bench-lite is limited to dataset/runner smoke in this stage.

## SWE-lite deferral
Formal SWE-bench-lite agent evaluation is deferred to the agent stage because a plain language-model run would not be protocol comparable.

## Resources
Resource inventory is stored in `results/evidence/infrastructure/stage1_final_resource_inventory_20260804_123140/`. Model artifacts, if produced by a future rerun, must stay under `$HOME/srtp-artifacts`.

## Stage 2 inputs
The next stage should start from blocked/partial evidence, prioritize raw R4 scoring for Magnitude/Wanda/SparseGPT, and run Flab benchmark activation again once the local Qwen1.5B model environment is fully available.
