# Reproduction Status

This report expands the method registry into a method-by-method first-stage status view.

## Flab-Pruner
Owner: 常珂舒
Upstream: vendored_submodule
Primary model: Qwen2.5-Coder
Adapter status: qwen_adapter
Smoke status: completed
R4 status: completed
Recovery status: completed
Officiality: experimental_extension
Validity: under_review
Quality gate: fail
Evidence: complete
Missing work: Official structural mode and benchmark activation experimental mode are separated.

## LLM-Pruner
Owner: 常珂舒
Upstream: vendored_submodule
Primary model: Qwen2.5-Coder / CodeLlama
Adapter status: qwen_adapter
Smoke status: completed
R4 status: completed
Recovery status: completed
Officiality: local_official_adapter
Validity: under_review
Quality gate: fail
Evidence: complete
Missing work: Local adapter evidence is complete; CodeLlama route is fallback.

## SliceGPT
Owner: 常珂舒
Upstream: vendored_submodule
Primary model: Qwen2.5-Coder / CodeLlama
Adapter status: qwen_adapter
Smoke status: completed
R4 status: partial
Recovery status: completed
Officiality: local_official_adapter
Validity: under_review
Quality gate: fail
Evidence: partial
Missing work: Partial benchmark evidence keeps actual task counts.

## LaCo
Owner: 常珂舒
Upstream: vendored_submodule
Primary model: CodeLlama candidate
Adapter status: blocked
Smoke status: planned
R4 status: blocked
Recovery status: not_applicable
Officiality: not_run
Validity: diagnostic_only
Quality gate: not_applicable
Evidence: not_applicable
Missing work: Upstream notebook route did not become a reproducible Stage 1 run.

## Magnitude
Owner: 潘阔
Upstream: vendored_submodule
Primary model: Qwen2.5-Coder / OPT
Adapter status: ready
Smoke status: completed
R4 status: not_applicable
Recovery status: not_applicable
Officiality: auxiliary_protocol
Validity: valid
Quality gate: pass
Evidence: aggregate_only
Missing work: Auxiliary full evaluation is aggregate only and not directly comparable with r4_half.

## Wanda
Owner: 潘阔
Upstream: vendored_submodule
Primary model: Qwen2.5-Coder / OPT
Adapter status: ready
Smoke status: completed
R4 status: not_applicable
Recovery status: not_applicable
Officiality: auxiliary_protocol
Validity: valid
Quality gate: pass
Evidence: aggregate_only
Missing work: Auxiliary full evaluation is aggregate only and not directly comparable with r4_half.

## DSnoT
Owner: 潘阔
Upstream: vendored_submodule
Primary model: OPT
Adapter status: blocked_qwen
Smoke status: completed
R4 status: not_applicable
Recovery status: not_applicable
Officiality: auxiliary_protocol
Validity: diagnostic_only
Quality gate: not_applicable
Evidence: aggregate_only
Missing work: OPT PPL aggregate is recorded; Qwen adapter remains unsupported.

## OWL
Owner: 潘阔
Upstream: vendored_submodule
Primary model: OPT
Adapter status: blocked_qwen
Smoke status: completed
R4 status: not_applicable
Recovery status: not_applicable
Officiality: auxiliary_protocol
Validity: diagnostic_only
Quality gate: not_applicable
Evidence: aggregate_only
Missing work: Process evidence is recorded; Qwen adapter remains unsupported.

## SparseGPT
Owner: 李长骏
Upstream: vendored_submodule
Primary model: OPT / CodeLlama candidate
Adapter status: planned
Smoke status: planned
R4 status: planned
Recovery status: not_applicable
Officiality: not_run
Validity: not_applicable
Quality gate: pending
Evidence: not_applicable
Missing work: Candidate retained for the first-stage method scope.

## MaskLLM
Owner: 李长骏
Upstream: vendored_submodule
Primary model: candidate
Adapter status: planned
Smoke status: planned
R4 status: planned
Recovery status: not_applicable
Officiality: not_run
Validity: not_applicable
Quality gate: pending
Evidence: not_applicable
Missing work: Candidate retained without first-stage run evidence.

## Pruner-Zero
Owner: 李长骏
Upstream: vendored_submodule
Primary model: candidate
Adapter status: planned
Smoke status: planned
R4 status: planned
Recovery status: not_applicable
Officiality: not_run
Validity: not_applicable
Quality gate: pending
Evidence: not_applicable
Missing work: Candidate retained without first-stage run evidence.

## FLAP
Owner: 李长骏
Upstream: vendored_submodule
Primary model: candidate
Adapter status: planned
Smoke status: planned
R4 status: planned
Recovery status: not_applicable
Officiality: not_run
Validity: not_applicable
Quality gate: pending
Evidence: not_applicable
Missing work: Candidate retained without first-stage run evidence.

## Missing work summary
LaCo is blocked. SparseGPT, MaskLLM, Pruner-Zero, and FLAP remain planned. SliceGPT is partial. These facts are preserved rather than hidden.
