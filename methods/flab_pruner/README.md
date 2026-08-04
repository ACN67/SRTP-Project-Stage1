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

That blocker is historical: it applied to the first implementation attempt that used a standard Hugging Face model or stopped at the top-level Flab API. It is superseded by the later internal-zs adapter implementation, which drives `model.model.prune(zs)` with benchmark-derived FFN `intermediate_indexes`.

## Benchmark-Guided Experimental Path

Evidence level L3/L4/L5 is recorded for the experimental benchmark-guided extension. This is not an upstream official Flab-Pruner mode.

The enabled benchmark-guided structural dimension is `intermediate_indexes`: guide examples are forwarded through the vendored Qwen2 model, tensor activations are collected from MLP intermediate projections, and the retained FFN intermediate indices are selected from activation importance. The config-derived dimensions are `hidden`, `attention_head`, and `kv_head`, which remain full-size in this implementation.

Primary evidence:
- Schema audit: `results/evidence/diagnostics/flab_zs_schema_audit_20260804_135032/`.
- Tiny guide causality smoke: `results/evidence/smoke/flab_benchmark_guided_tiny_20260804_135032/`.
- Qwen1.5B target smoke: `results/evidence/smoke/flab_qwen15b_benchmark_guided_smoke_20260804_135032/`.
- Capped quality gates: `results/evidence/smoke/flab_qwen15b_benchmark_guided_he_keep80_capped32_20260804_135032/`, `results/evidence/smoke/flab_qwen15b_benchmark_guided_mbpp_keep80_capped32_20260804_135032/`, and `results/evidence/smoke/flab_qwen15b_benchmark_guided_lcb_keep80_capped32_20260804_135032/`.

The Qwen1.5B smoke reached an actual parameter keep ratio of `0.7999755104980733` (`1777088000` to `1421626880` parameters), saved an external artifact, reloaded it, and produced nonempty greedy output. All three capped-32 variants produced guide-specific artifacts and ran the fixed 20-task non-collapse gate, but each failed quality because duplicate rate was `0.95`; full-guide formal evaluation was therefore skipped by the predefined resource and quality rule.

Final archive status: the `/tmp` artifact directories are no longer present, so `results/evidence/diagnostics/flab_artifact_archive_20260804_191623/` records them as `missing_after_ephemeral_tmp_cleanup`. This does not change the prior save/reload/generation evidence, but it means the pruned artifacts are not currently downloadable from a persistent artifact root.
