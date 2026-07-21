# Flab-Pruner Qwen3B Pruned HumanEval Smoke Eval

Status: success

Method: Flab-Pruner
Owner: 常珂舒
Model: Qwen2.5-Coder-3B-Instruct pruned by Flab-Pruner
Benchmark: HumanEval
Split: data/splits/humaneval/eval.jsonl
Run ID: flab_qwen3b_pruned_humaneval_smoke_eval_20260721_182050

This run generates solutions for the 4-task HumanEval smoke eval split using the locally saved pruned Qwen3B model and scores them with the split-aware HumanEval scorer in base-only mode.

Key result:

- Task count: 4
- Base pass count: 0
- Base pass rate: 0.0
- Base status counter: fail 4

Note:

- This confirms the HumanEval smoke eval pipeline works for the pruned Qwen3B artifact.
- The current purpose is pipeline verification, not final model-quality evaluation.
