# Keshu Stage 1 Final Summary

## 1. Responsible Methods
Owner: 常珂舒.

Responsible methods: Flab-Pruner, LLM-Pruner, SliceGPT, and LaCo.

## 2. Reproduction Scope
The owner-scoped closure covers execution evidence, registries, quality gates, failure semantics, and method-level scope decisions. It does not close methods owned by other contributors.

## 3. Flab Official Structural
The local official structural adapter completed real pruning, parameter reduction, artifact save evidence, formal benchmark evidence, and partial recovery evidence. Its quality gate remains failed.

## 4. Flab Benchmark-Guided Experimental
The experimental extension forwards benchmark guide prompts through Qwen2.5-Coder-1.5B, collects FFN activation importance, maps that importance to `intermediate_indexes`, and calls the vendored internal `model.model.prune(zs)` path. HumanEval, MBPP, and LiveCodeBench guide variants produced distinct importance and selected-index hashes.

The Qwen1.5B smoke reduced parameters from `1777088000` to `1421626880`, for an actual keep ratio of `0.7999755104980733`. The smoke saved and reloaded the artifact during execution. The later archive audit records the ephemeral `/tmp` artifacts as unavailable after cleanup, so artifact reconstruction would be required to download them now.

## 5. LLM-Pruner
The Qwen primary path is completed and recorded. CodeLlama remains fallback evidence. The quality gate remains failed.

## 6. SliceGPT
The Qwen primary route is completed. The legacy CodeLlama route is retained as partial evidence. The quality gate remains failed.

## 7. LaCo Scope Decision
LaCo is skipped for Stage 1 because the official notebook, model, and environment support are too narrow for a faithful unified code-model reproduction without large manual reimplementation. The tiny layer-collapse smoke remains diagnostic only.

## 8. Formal Results
Formal score rows remain limited to runs with actual scorer output. The capped-32 Flab benchmark-guided pilots are excluded from formal R4-half tables.

## 9. Quality Gates
The three Flab benchmark-guided pilots all failed the predefined generation-quality gate with nonempty but highly duplicated output.

## 10. Failure Cause
The current Flab benchmark-guided first break is post-pruning generation distribution collapse, not prune API failure and not serialization failure.

## 11. Artifact Status
The Qwen1.5B smoke and three capped-32 pilot artifacts were written under `/tmp` during execution. The final archive audit found those directories missing and records `missing_after_ephemeral_tmp_cleanup`.

## 12. Stage 1 Conclusion
Owner-scoped Stage 1 execution has reached an auditable closure. Closure means every responsible method now has an evidence-backed completed, quality-failed, or scope-skipped conclusion; it does not mean every method is performance-effective.

## 13. Conclusions Not Supported
This archive does not claim benchmark-guided Flab is upstream official, does not claim the capped pilots are formal benchmark successes, and does not claim the global Stage 1 project is closed.

## 14. Follow-Up Research
Future work should preserve the internal-zs adapter, redesign the benchmark importance signal or pruning allocation to avoid generation collapse, and reconstruct artifacts only under a persistent artifact root.
