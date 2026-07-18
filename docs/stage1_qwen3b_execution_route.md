# Stage 1 Qwen 3B Execution Route

This document is the operational route for Stage 1 after aligning with the execution book.

## Final Target

Stage 1 aims to reproduce multiple pruning methods, then run representative benchmark-guided pruning on Qwen2.5-Coder-3B and compare it with Qwen2.5-Coder-1.5B under the same evaluation protocol.

The core comparison set is:

- Original Qwen2.5-Coder-1.5B / Instruct.
- Original Qwen2.5-Coder-3B / Instruct.
- Benchmark-guided pruned Qwen2.5-Coder-3B / Instruct.

Qwen 1.5B is both a baseline and a low-cost compatibility/debug target. Qwen 3B is the actual pruning adaptation target for representative methods.

## Execution Layers

| Layer | Purpose | Model | Expected output |
|---|---|---|---|
| R0 | Inspect upstream method and command shape | official repos | method notes and blockers |
| R1 | Reproduce official minimum run | official small model | logs, params/sparsity, saved artifact or manifest |
| R2-debug | Check Qwen compatibility cheaply | Qwen 1.5B | adapter evidence or failure report |
| Q3-target | Prune target comparison model | Qwen 3B | pruned artifact/manifest and metrics |
| R3 | Use project benchmark guide/eval splits | Qwen 3B and 1.5B baseline | guide hash, eval results, report |

## Benchmark-Guided Pruning Requirement

The benchmark guide split is used during pruning. It is not only for evaluation.

Minimum guide usage:

- HumanEval and MBPP smoke guide samples for early checks.
- HumanEval, MBPP, LiveCodeBench, and SWE-bench Lite guide/eval split files for Stage 1 evidence.
- Same-source guide/eval is the minimum formal setup, such as HumanEval-guide pruning followed by HumanEval-eval validation.

## Current Best Path For 常珂舒

1. Finish Flab-Pruner R0/R1 inspection because it already includes Qwen2 modeling and pruning utilities.
2. Build or document the Qwen2.5-Coder-3B Flab-Pruner patch requirements:
   - remove hard-coded local paths;
   - parameterize model id/path;
   - compute remain dimensions from the 3B config instead of using 7B constants;
   - accept project benchmark guide text as calibration/input data.
3. Keep LLM-Pruner and SliceGPT as secondary structured-method candidates:
   - LLM-Pruner needs a Qwen/AutoModel path because current code imports custom LLaMA classes.
   - SliceGPT needs a Qwen2 adapter because current adapters cover OPT/LLaMA/Phi/Phi3.
4. Treat LaCo as blocked until notebook logic is extracted.

## Immediate Small Tasks

- Maintain method status table.
- Add lightweight checks for Qwen 1.5B and Qwen 3B config/tokenizer.
- Add Flab-Pruner method notes and R0 blocker list.
- Add a readiness audit so missing Stage 1 deliverables are visible.

## Long Tasks To Run Manually

These should not be continuously supervised by Codex:

- Download Qwen 1.5B / 3B weights.
- Run original Qwen 1.5B and 3B benchmark baselines.
- Run benchmark-guided pruning on Qwen 3B.
- Run full HumanEval, MBPP, LiveCodeBench, or SWE-bench Lite evaluation.

Codex should prepare commands and inspect output files afterward.
