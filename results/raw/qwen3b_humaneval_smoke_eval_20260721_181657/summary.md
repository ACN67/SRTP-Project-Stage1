# Qwen2.5-Coder-3B HumanEval Smoke Eval

Status: success

Method: Baseline
Owner: 常珂舒
Model: Qwen/Qwen2.5-Coder-3B-Instruct
Benchmark: HumanEval
Split: data/splits/humaneval/eval.jsonl
Run ID: qwen3b_humaneval_smoke_eval_20260721_181657

This run generates solutions for the 4-task HumanEval smoke eval split and scores them with a split-aware EvalPlus-based HumanEval scorer in base-only mode.

Key result:

- Task count: 4
- Base pass count: 0
- Base pass rate: 0.0
- Base status counter: fail 4

Note:

- This confirms the HumanEval smoke eval pipeline works for the Qwen3B baseline.
- The current generation uses plain prompt completion without chat-template prompting, so the 0/4 score should be treated as a pipeline smoke result rather than a final model-quality result.
