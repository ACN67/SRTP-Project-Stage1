# Stage 1 Protocol

## Stage 1 goals
Stage 1 establishes a reproducible repository for code LLM pruning methods. The goal is not to prove a final best method; it is to preserve evidence, expose quality gates, and make every completed, partial, blocked, and planned route traceable.

## Method scope
The fixed method scope is Flab-Pruner, LLM-Pruner, SliceGPT, LaCo, Magnitude, Wanda, DSnoT, OWL, SparseGPT, MaskLLM, Pruner-Zero, and FLAP. Member names appear only as owner metadata in method README files, registry rows, and the experiment plan.

## R0-R4
R0 covers environment, model manifest, upstream repository, and split setup. R1 covers upstream smoke checks. R2 covers code-model adapter validation. R3 covers benchmark-guided or recovery preparation when applicable. R4 covers formal R4-half scoring and registry generation.

## Model policy
Qwen2.5-Coder models are the primary code-model targets. CodeLlama and OPT evidence is retained when it explains method behavior, fallback routes, or upstream feasibility. Large model artifacts are tracked through `results/status/artifacts.csv` rather than committed.

## Smoke and formal
Smoke runs are small functional checks. Formal R4-half runs use guide/eval splits under `data/benchmarks/r4_half/` and are summarized by `results/formal/r4_half/scores.csv`.

## Execution, validity, and quality gate
Execution status records whether a run or method completed, is partial, blocked, planned, skipped, or superseded. Validity status separates valid, under-review, diagnostic-only, invalid, and not-applicable evidence. Quality gate records whether evidence is suitable for conclusions.

## Recovery
Recovery includes distillation data, LoRA adapters, merge checks, and post-merge evaluation. Recovery status is stored in `results/status/methods.csv` and discussed in `docs/recovery_protocol.md`.

## Completion criteria
Completion criteria require traceable code, benchmark protocol, registry rows, evidence preservation, and clear failure or missing-work status. Current stage incomplete items do not mean repository organization failed; they mean the registry is honestly representing first-stage facts.

## Registry and evidence relationship
`results/evidence/` is immutable raw evidence. `results/status/` is the generated status index. The registry can be rebuilt from evidence, manifests, method configuration, and explicit inference rules.
