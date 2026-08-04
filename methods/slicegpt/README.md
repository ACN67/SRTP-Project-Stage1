# SliceGPT

Owner: 常珂舒

Family: structured
Primary model: Qwen2.5-Coder / CodeLlama
Execution status: completed
Validity status: under_review
Quality gate: fail
Officiality: local_official_adapter
Evidence status: complete
Primary code: `methods/slicegpt/qwen_prune.py`

## Evidence
- Qwen primary local-adapter route: `results/evidence/r4_half/slicegpt_qwen25c15b_official_keep80_gpu_20260731_011001/`.
- CodeLlama legacy partial route: `results/evidence/r4_half/slicegpt_codellama7b_r4_benchguided_evalhalf_20260726_053225/`.
- Primary evidence audit: `results/evidence/diagnostics/slicegpt_primary_evidence_audit_20260804_130541/`.

## Current Interpretation
The Qwen primary route is the completed owner-scoped evidence path. The older CodeLlama route remains partial with exact task counts preserved and is not upgraded into the primary claim. The quality gate remains failed, so registry completion is evidence closure, not pass-rate success.
