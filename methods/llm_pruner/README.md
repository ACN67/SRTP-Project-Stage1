# LLM-Pruner

Owner: 常珂舒

Family: structured
Primary model: Qwen2.5-Coder / CodeLlama
Execution status: completed
Validity status: under_review
Quality gate: fail
Officiality: local_official_adapter
Evidence status: complete
Primary code: `methods/llm_pruner/qwen_prune.py`

## Evidence
- Qwen primary local-adapter route: `results/evidence/r4_half/llmpruner_qwen25c15b_official_keep80_gpu_20260730_105340/`.
- CodeLlama fallback route: `results/evidence/r4_half/llmpruner_codellama7b_r4_layerdrop_keep80_full_20260725_182022/`.
- Primary evidence audit: `results/evidence/diagnostics/llmpruner_primary_evidence_audit_20260804_130541/`.

## Current Interpretation
The Qwen route is the primary local adapter evidence for this method. The CodeLlama layer-drop route is retained as fallback non-official evidence and must not be mixed with the Qwen primary claim. The quality gate remains failed, so completion here means evidence closure rather than quality success.
