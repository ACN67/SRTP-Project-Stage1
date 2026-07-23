# R4 Half-Set Runs

R4 is the formal Stage 1 run level.

Goal: complete one benchmark-guided pruning run per viable method using the selected model-family policy, then evaluate retained coding ability and runtime/resource reduction.

## Required Outputs Per Method

Each method should produce:

- `summary.md`: human-readable method result.
- `summary.json`: machine-readable status and key metrics.
- generation files for HumanEval and MBPP eval halves.
- score summaries for HumanEval and MBPP.
- resource summary with elapsed time, peak memory, peak GPU memory, disk artifact size, and local/offload notes.
- artifact manifest with local-only model paths and hashes when a model artifact is retained.

## Current Status

R4 is pending. R3 smoke confirms the pipeline is runnable for:

- Flab-Pruner on Qwen2.5-Coder-3B.
- LLM-Pruner on CodeLlama-7B.
- SliceGPT on CodeLlama-7B, with very slow local generation under offload.

