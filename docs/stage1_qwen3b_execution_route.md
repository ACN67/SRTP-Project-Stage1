# Stage 1 Qwen 3B Execution Route

This document is retained as the Flab-Pruner/Qwen route and as historical context for the earlier Qwen-only plan.

For the active cross-method model policy, see:

```text
docs/stage1_model_selection_and_metrics.md
```

## Final Target

For Flab-Pruner, Stage 1 aims to run representative benchmark-guided pruning on Qwen2.5-Coder-3B and compare it with Qwen2.5-Coder-1.5B under the same evaluation protocol.

For methods whose official implementations support LLaMA-family models better than Qwen, the active target is now CodeLlama-family pruning rather than Qwen adapter development.

The core comparison set is:

- Original Qwen2.5-Coder-1.5B / Instruct.
- Original Qwen2.5-Coder-3B / Instruct.
- Benchmark-guided pruned Qwen2.5-Coder-3B / Instruct.

Qwen 1.5B is both a baseline and a low-cost compatibility/debug target for Flab-Pruner. Qwen 3B is the actual Flab-Pruner pruning adaptation target.

## Execution Layers

| Layer | Purpose | Model | Expected output |
|---|---|---|---|
| R0 | Inspect upstream method and command shape | official repos | method notes and blockers |
| R1 | Reproduce official minimum run | official small model | logs, params/sparsity, saved artifact or manifest |
| R2-debug | Check method-family compatibility cheaply | Qwen 1.5B for Flab-Pruner; CodeLlama/tiny official model for LLaMA-family methods | adapter evidence or failure report |
| R3-target | Prune target comparison model | Qwen 3B for Flab-Pruner; CodeLlama or method-supported model for others | pruned artifact/manifest and metrics |
| R3 | Use project benchmark guide/eval splits | method-family baseline and pruned model | guide hash, eval results, resource report |

## Benchmark-Guided Pruning Requirement

The benchmark guide split is used during pruning. It is not only for evaluation.

Minimum guide usage:

- HumanEval and MBPP smoke guide samples for early checks.
- HumanEval, MBPP, LiveCodeBench, and SWE-bench Lite guide/eval split files for Stage 1 evidence.
- Same-source guide/eval is the minimum formal setup, such as HumanEval-guide pruning followed by HumanEval-eval validation.

## Current Best Path For 常珂舒

1. Keep Flab-Pruner as the Qwen2.5-Coder route because it already includes Qwen2 modeling and pruning utilities and has a successful Qwen3B run.
2. Preserve the Qwen2.5-Coder-3B Flab-Pruner patch requirements:
   - remove hard-coded local paths;
   - parameterize model id/path;
   - compute remain dimensions from the 3B config instead of using 7B constants;
   - accept project benchmark guide text as calibration/input data.
3. Move LLM-Pruner and SliceGPT to CodeLlama-family targets instead of spending Stage 1 effort on Qwen adapters:
   - LLM-Pruner is LLaMA-bound in its official `hf_prune.py`.
   - SliceGPT already has a LLaMA adapter and lacks Qwen2 support.
4. Treat LaCo as blocked because official support is too limited: upstream is notebook-only and does not provide a CodeLlama route.

## Immediate Small Tasks

- Maintain method status table.
- Add lightweight checks for Qwen 1.5B/3B and CodeLlama-family config/tokenizer.
- Add Flab-Pruner method notes and R0 blocker list.
- Add a readiness audit so missing Stage 1 deliverables are visible.

## Long Tasks To Run Manually

These should not be continuously supervised by Codex:

- Download Qwen 1.5B / 3B weights for Flab-Pruner.
- Download CodeLlama weights for LLaMA-family methods when resources allow.
- Run original same-family baselines for every pruned model.
- Run benchmark-guided pruning on Qwen 3B for Flab-Pruner and CodeLlama/method-supported models for other methods.
- Run full HumanEval, MBPP, LiveCodeBench, or SWE-bench Lite evaluation.

Codex should prepare commands and inspect output files afterward.
