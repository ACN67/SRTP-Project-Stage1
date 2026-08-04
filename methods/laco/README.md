# LaCo

Owner: 常珂舒

Family: layer collapse
Primary model: CodeLlama candidate
Execution status: partial
Validity status: diagnostic_only
Quality gate: not_applicable
Officiality: not_run
Evidence status: partial
Primary code: `methods/laco/run_smoke.py`

## Evidence
- Previous notebook probe retained as diagnostic only: `results/evidence/diagnostics/laco_upstream_notebook_probe_attempt_20260804_123140/`.
- Upstream notebook audit: `methods/laco/upstream_audit.md`.
- Core smoke evidence: `results/evidence/smoke/laco_upstream_smoke_20260804_130541/`.

## Current Interpretation
The previous file-presence probe does not close LaCo. The new wrapper executes a tiny LLaMA-compatible core path: forward pass, adjacent-layer similarity scoring, layer collapse by parameter averaging, layer removal, and post-collapse forward check. This proves a minimal algorithmic unit, but it is not a formal CodeLlama R4 run and remains diagnostic-only evidence.
