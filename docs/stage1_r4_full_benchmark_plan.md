# Stage 1 R4 Full Benchmark-Guided Plan

Stage: 1
Status: active plan
Updated: 2026-07-22

## Purpose

R3 smoke runs prove that pruning, generation, scoring, logging, and result archival can execute end to end. They are not enough for the final Stage 1 evidence.

R4 is the first formal benchmark-guided reproduction layer. It requires a larger, fixed benchmark subset and enough resource/quality metrics to support Stage 1 reporting.

## Required Benchmarks

| Benchmark | R4 status | Role |
|---|---|---|
| HumanEval | required | primary code-generation correctness |
| MBPP | required | primary code-generation correctness |
| LiveCodeBench | optional extension | more realistic code tasks if time and resource allow |
| SWE-bench Lite | not required for R4 | keep data interface for later software-engineering task work |

## Split Policy

R4 should use fixed half-set splits:

```text
data/splits/humaneval_half/
data/splits/mbpp_evalplus_half/
data/splits/livecodebench_half/        optional
```

Each split must include a manifest with:

- source benchmark and source version;
- random seed or deterministic selection rule;
- guide/eval counts;
- task IDs;
- whether task solutions are excluded from prompts.

## Minimum R4 Run Requirements

For each R4 method/model pair:

1. Run baseline generation/evaluation on the same model family.
2. Run benchmark-guided pruning using the guide split.
3. Save or manifest the pruned artifact.
4. Reload the pruned model.
5. Run eval split generation/evaluation.
6. Preserve resource traces and summaries.
7. Produce a final method summary with retention and reduction metrics.

## Required Metrics

| Metric | Required |
|---|---|
| baseline pass rate | yes |
| pruned pass rate | yes |
| ability retention rate | yes |
| parameter count before/after | yes |
| parameter reduction rate | yes |
| peak GPU memory | yes |
| peak process RSS | yes |
| runtime | yes |
| artifact size/hash | yes |

If a baseline pass rate is zero, ability retention must be reported as `N/A` rather than divided by zero.

## Current Priority

1. Finish SliceGPT CodeLlama-family R2/R3 feasibility.
2. Finish LLM-Pruner CodeLlama-family R2/R3 feasibility.
3. Generate HumanEval/MBPP half splits.
4. Choose the first R4 representative run:
   - preferred safe option: Flab-Pruner on Qwen2.5-Coder, because R2/R3 already works;
   - optional if resources allow: SliceGPT or LLM-Pruner on CodeLlama-family.

R4 results are the first results that may be used as Stage 1 formal evidence. R3 smoke results remain pipeline evidence only.
