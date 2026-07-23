# Stage 1 Plan

## Goal

Stage 1 builds a reproducible pruning workflow for code large language models. The goal is not to decide the best method from smoke tests. The goal is to make each viable method runnable, benchmark-guided, measurable, and ready for formal comparison.

## Model Policy

Methods use the model family they can realistically prune:

| Case | Model choice |
|---|---|
| Official LLaMA-family support | `codellama/CodeLlama-7b-hf` |
| Explicit Qwen/Qwen2 support | `Qwen/Qwen2.5-Coder-3B-Instruct` |
| No CodeLlama or Qwen route | Official supported model |
| Notebook-only or too narrow | Mark blocked/skipped |

This policy replaced the earlier plan that tried to force every method onto Qwen.

## Benchmarks

| Benchmark | Stage 1 role |
|---|---|
| HumanEval | Required R3 smoke and R4 half-set evaluation. |
| MBPP | Required R3 smoke and R4 half-set evaluation. |
| LiveCodeBench | Smoke interface exists; R4 optional if time/resources allow. |
| SWE-bench Lite | Interface proof only for Stage 1 unless separately assigned. |

Every benchmark split has two roles:

- `guide`: used during pruning or pruning-choice calibration.
- `eval`: used only after pruning.

## R0-R4

| Round | Purpose | Evidence |
|---|---|---|
| R0 | Repository, dependency, model-family, and benchmark-entry audit. | import checks, config probes, split manifests. |
| R1 | Official or minimum upstream reproduction on a supported small model. | official smoke logs and summaries. |
| R2 | Prune the selected Stage 1 model family for the method. | pruning logs, parameter/resource summaries, artifact manifests. |
| R3 | Four-task HumanEval/MBPP benchmark smoke after pruning. | generations, score summaries, resource notes. |
| R4 | Formal half-set benchmark-guided run. | baseline/pruned scores, resources, retention and reduction metrics. |

R3 only proves the pipeline runs. R4 is required before writing formal Stage 1 conclusions.

## R4 Minimum Output

Each R4 method run should store:

- generation outputs for HumanEval and MBPP eval halves.
- score summaries and score details.
- pruning summary with before/after model size or parameter counts.
- resource summary with elapsed time, peak GPU memory, peak process RSS, and artifact size.
- local artifact manifest with paths and hashes when model weights are retained.

Derived reporting metrics:

| Metric | Formula |
|---|---|
| Ability retention | `pruned_pass_rate / baseline_pass_rate`, or `N/A` if baseline is zero. |
| Parameter reduction | `1 - parameter_count_after / parameter_count_before`. |
| Artifact reduction | `1 - artifact_size_after / artifact_size_before`. |
| Runtime reduction | comparable baseline/pruned duration only. |
| VRAM reduction | comparable baseline/pruned peak GPU memory only. |

Cross-family raw scores are not direct quality comparisons. Compare them through normalized retention and resource-reduction indicators.
