# Flab-Pruner

Owner: 常珂舒

Family: structured
Primary model: Qwen2.5-Coder
Execution status: completed
Validity status: under_review
Quality gate: fail
Officiality: experimental_extension
Evidence status: complete
Primary code: `methods/flab_pruner/qwen_prune.py`

## Evidence
- Structural primary route: `results/evidence/r4_half/flabpruner_qwen25c15b_official_keep80_20260730_015031/`.
- Benchmark activation attempt retained as immutable diagnostics: `results/evidence/r4_half/flab_qwen15b_benchmark_activation_he_keep80_attempt_20260804_123140/`.
- Prune API audit: `results/evidence/diagnostics/flab_prune_api_audit_20260804_130541/`.
- Activation blocker smoke: `results/evidence/smoke/flab_benchmark_activation_tiny_smoke_20260804_130541/`.

## Current Interpretation
The local official structural adapter remains the primary completed route. The benchmark-activation path now loads the vendored Flab Qwen model path instead of a plain Hugging Face model, but the vendored `prune(config, stage)` API has no verified external per-channel mask schema for the activation masks produced by the benchmark branch. That branch is therefore closed as a code-level blocker, not as a successful benchmark run.
