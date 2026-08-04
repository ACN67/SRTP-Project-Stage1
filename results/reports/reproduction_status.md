# Stage 1 Reproduction Status

This report summarizes the repository status after the Keshu-owned owner-scoped completion pass. Detailed machine-readable state is in `results/status/methods.csv`, `results/status/runs.csv`, and `results/status/completion_audit.json`.

## Completed primary routes
- Flab-Pruner. Owner metadata is in `results/status/methods.csv`. Structural route has R4 and recovery evidence. Benchmark activation is closed for this owner pass as a code-level blocker because the vendored `prune(config, stage)` API has no verified external per-channel mask schema; see `results/evidence/diagnostics/flab_prune_api_audit_20260804_130541/` and `results/evidence/smoke/flab_benchmark_activation_tiny_smoke_20260804_130541/`.
- LLM-Pruner. Owner metadata is in `results/status/methods.csv`. Existing Qwen primary evidence is audited in `results/evidence/diagnostics/llmpruner_primary_evidence_audit_20260804_130541/`; CodeLlama layer-drop remains fallback non-official evidence. Quality gate remains failed.
- SliceGPT. Owner metadata is in `results/status/methods.csv`. Qwen primary evidence is audited in `results/evidence/diagnostics/slicegpt_primary_evidence_audit_20260804_130541/`; CodeLlama legacy partial evidence is retained separately with exact task counts. Quality gate remains failed.

## Partial routes
- Magnitude. Owner metadata is in `results/status/methods.csv`. Upstream: Wanda-style magnitude baseline. Missing work: raw R4 score summaries. Historical auxiliary aggregate retained; this pass added raw formal command evidence at `results/evidence/r4_half/magnitude_qwen15b_keep80_raw_formal_attempt_20260804_123140/`, but no raw R4 score was produced.
- Wanda. Owner metadata is in `results/status/methods.csv`. Upstream: vendored Wanda. Missing work: raw R4 score summaries. Historical auxiliary aggregate retained; this pass added raw guided formal command evidence at `results/evidence/r4_half/wanda_qwen15b_he_keep80_raw_formal_attempt_20260804_123140/`, but no raw R4 score was produced.

## Blocked with evidence
- LaCo. Owner metadata is in `results/status/methods.csv`. The old notebook probe remains diagnostic only. The new tiny LLaMA-compatible core smoke is recorded in `results/evidence/smoke/laco_upstream_smoke_20260804_130541/`; formal CodeLlama R4 remains not run.
- DSnoT. Owner metadata is in `results/status/methods.csv`. Upstream: vendored DSnoT. Missing work: Qwen raw code-model support. Evidence is recorded in `results/evidence/diagnostics/dsnot_qwen15b_adapter_probe_attempt_20260804_123140/`.
- OWL. Owner metadata is in `results/status/methods.csv`. Upstream: vendored OWL. Missing work: Qwen raw code-model support. Blocker is recorded in `results/evidence/diagnostics/owl_qwen15b_adapter_probe_attempt_20260804_123140/`.
- SparseGPT. Owner metadata is in `results/status/methods.csv`. Upstream: vendored SparseGPT. Missing work: code-model formal run. Evidence is recorded in `results/evidence/diagnostics/sparsegpt_qwen15b_adapter_probe_attempt_20260804_123140/`.
- MaskLLM. Owner metadata is in `results/status/methods.csv`. Upstream: vendored MaskLLM. Missing work: mask training and code-model formal run. Evidence is recorded in `results/evidence/diagnostics/maskllm_official_smoke_probe_attempt_20260804_123140/`.
- Pruner-Zero. Owner metadata is in `results/status/methods.csv`. Upstream: vendored Pruner-Zero. Missing work: OPT smoke repair and code-model probe. Blocker is recorded in `results/evidence/diagnostics/prunerzero_opt125m_smoke_probe_attempt_20260804_123140/`.
- FLAP. Owner metadata is in `results/status/methods.csv`. Upstream: vendored FLAP. Missing work: LLaMA-compatible formal route under resource gate. Evidence is recorded in `results/evidence/diagnostics/flap_llama_template_probe_attempt_20260804_123140/`.

## SWE-bench-lite
SWE-bench-lite dataset smoke is recorded in `results/evidence/smoke/swebench_lite_dataset_smoke_20260804_123140/`. Formal agent evaluation is deferred because plain-model generation is not comparable with a SWE-agent protocol.

## Keshu-owned closure pass
Owner: 常珂舒

`results/status/keshu_completion.json` records owner_execution_closed=true for Flab-Pruner, LLM-Pruner, SliceGPT, and LaCo. `results/status/completion_audit.json` keeps stage1_execution_closed=false because this was intentionally owner-scoped.
