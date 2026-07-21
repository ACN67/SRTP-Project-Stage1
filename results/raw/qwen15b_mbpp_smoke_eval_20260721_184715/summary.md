# Qwen2.5-Coder-1.5B MBPP Smoke Eval

Status: success

Method: Baseline
Owner: 常珂舒
Model: Qwen/Qwen2.5-Coder-1.5B-Instruct
Benchmark: MBPP EvalPlus-compatible smoke
Split: data/splits/mbpp_evalplus/eval.jsonl
Run ID: qwen15b_mbpp_smoke_eval_20260721_184715

This run generates solutions for the 4-task MBPP smoke eval split and scores them with the split-aware MBPP scorer in base-only mode.

Key result:

- Task count: 4
- Base pass count: 1
- Base pass rate: 0.25
- Base status counter: fail 3, pass 1
- Passed task: Mbpp/19

Note:

- `data/splits/mbpp_evalplus/` was added to align the original MBPP smoke task IDs and prompts with EvalPlus MBPP+ scoring.
