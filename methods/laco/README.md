# LaCo

Owner: 常珂舒

Family: layer collapse
Primary model: CodeLlama candidate
Execution status: skipped
Validity status: not_applicable
Quality gate: not_applicable
Officiality: not_run
Evidence status: diagnostic_only
Primary code: `not_applicable`

## Evidence
- Previous notebook probe retained as diagnostic only: `results/evidence/diagnostics/laco_upstream_notebook_probe_attempt_20260804_123140/`.
- Upstream notebook audit: `methods/laco/upstream_audit.md`.
- Core smoke evidence: `results/evidence/smoke/laco_upstream_smoke_20260804_130541/`.

## Current Interpretation
The previous file-presence probe does not close LaCo. The new wrapper executes a tiny LLaMA-compatible core path: forward pass, adjacent-layer similarity scoring, layer collapse by parameter averaging, layer removal, and post-collapse forward check. This proves a minimal algorithmic unit, but it is not a formal CodeLlama R4 run and remains diagnostic-only evidence.

## Stage-1 Scope Decision

LaCo is skipped at method scope for Stage 1 because the upstream notebook/model support is insufficient for a faithful unified code-model reproduction. The tiny layer-collapse smoke remains diagnostic-only evidence and does not change this method-level decision.
