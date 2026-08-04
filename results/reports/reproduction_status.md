# Stage 1 Reproduction Status

This report summarizes the final-completion pass. Detailed machine-readable state is in `results/status/methods.csv`, `results/status/runs.csv`, and `results/status/completion_audit.json`.

## Completed primary routes
- Flab-Pruner. Owner metadata is in `results/status/methods.csv`. Upstream: vendored Flab-Pruner. Missing work: successful benchmark-activation R4 rerun after local model availability. Structural route has R4 and recovery evidence; benchmark activation implementation is real, but the final Qwen1.5B activation attempt is resource/model-environment blocked in `results/evidence/r4_half/flab_qwen15b_benchmark_activation_he_keep80_attempt_20260804_123140/`.
- LLM-Pruner. Owner metadata is in `results/status/methods.csv`. Upstream: vendored LLM-Pruner. Missing work: quality recovery. Existing Qwen/CodeLlama formal and recovery evidence retained; quality gate remains failed.
- SliceGPT. Owner metadata is in `results/status/methods.csv`. Upstream: vendored SliceGPT. Missing work: CodeLlama legacy completion. Qwen1.5B official keep80 route remains the primary completed route; CodeLlama partial evidence is retained separately.

## Partial routes
- Magnitude. Owner metadata is in `results/status/methods.csv`. Upstream: Wanda-style magnitude baseline. Missing work: raw R4 score summaries. Historical auxiliary aggregate retained; this pass added raw formal command evidence at `results/evidence/r4_half/magnitude_qwen15b_keep80_raw_formal_attempt_20260804_123140/`, but no raw R4 score was produced.
- Wanda. Owner metadata is in `results/status/methods.csv`. Upstream: vendored Wanda. Missing work: raw R4 score summaries. Historical auxiliary aggregate retained; this pass added raw guided formal command evidence at `results/evidence/r4_half/wanda_qwen15b_he_keep80_raw_formal_attempt_20260804_123140/`, but no raw R4 score was produced.

## Blocked with evidence
- LaCo. Owner metadata is in `results/status/methods.csv`. Upstream: notebook route. Missing work: executable wrapper/formal route. Evidence is recorded in `results/evidence/diagnostics/laco_upstream_notebook_probe_attempt_20260804_123140/`.
- DSnoT. Owner metadata is in `results/status/methods.csv`. Upstream: vendored DSnoT. Missing work: Qwen raw code-model support. Evidence is recorded in `results/evidence/diagnostics/dsnot_qwen15b_adapter_probe_attempt_20260804_123140/`.
- OWL. Owner metadata is in `results/status/methods.csv`. Upstream: vendored OWL. Missing work: Qwen raw code-model support. Blocker is recorded in `results/evidence/diagnostics/owl_qwen15b_adapter_probe_attempt_20260804_123140/`.
- SparseGPT. Owner metadata is in `results/status/methods.csv`. Upstream: vendored SparseGPT. Missing work: code-model formal run. Evidence is recorded in `results/evidence/diagnostics/sparsegpt_qwen15b_adapter_probe_attempt_20260804_123140/`.
- MaskLLM. Owner metadata is in `results/status/methods.csv`. Upstream: vendored MaskLLM. Missing work: mask training and code-model formal run. Evidence is recorded in `results/evidence/diagnostics/maskllm_official_smoke_probe_attempt_20260804_123140/`.
- Pruner-Zero. Owner metadata is in `results/status/methods.csv`. Upstream: vendored Pruner-Zero. Missing work: OPT smoke repair and code-model probe. Blocker is recorded in `results/evidence/diagnostics/prunerzero_opt125m_smoke_probe_attempt_20260804_123140/`.
- FLAP. Owner metadata is in `results/status/methods.csv`. Upstream: vendored FLAP. Missing work: LLaMA-compatible formal route under resource gate. Evidence is recorded in `results/evidence/diagnostics/flap_llama_template_probe_attempt_20260804_123140/`.

## SWE-bench-lite
SWE-bench-lite dataset smoke is recorded in `results/evidence/smoke/swebench_lite_dataset_smoke_20260804_123140/`. Formal agent evaluation is deferred because plain-model generation is not comparable with a SWE-agent protocol.
