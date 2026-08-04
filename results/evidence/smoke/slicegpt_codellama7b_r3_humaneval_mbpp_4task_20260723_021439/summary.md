# SliceGPT CodeLlama-7B R3 HumanEval/MBPP 4-task Smoke

Status: success

Method: SliceGPT
Owner: 常珂舒
Model: codellama/CodeLlama-7b-hf
Run ID: slicegpt_codellama7b_r3_humaneval_mbpp_4task_20260723_021439

This run performs SliceGPT minimal CodeLlama-7B rotate-and-slice pruning separately for HumanEval and MBPP 4-task smoke evaluation, then reports split-aware base-only pass rates.

Key result:

- Requested sparsity: 0.01
- Hidden size: 4096 -> 4048
- Effective hidden-size reduction: 1.1719%
- HumanEval base pass: 0/4 (0.00%)
- MBPP base pass: 0/4 (0.00%)
- HumanEval elapsed: 3970.017 seconds
- MBPP elapsed: 4798.458 seconds

Note:

The naive parameter count is not used as the SliceGPT compression metric because compressed-layer shortcut parameters increase the raw count after layer replacement/fusion.
