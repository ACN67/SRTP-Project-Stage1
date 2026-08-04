# Qwen2.5-Coder-1.5B HumanEval Smoke Eval

Status: success

Method: Baseline
Owner: 常珂舒
Model: Qwen/Qwen2.5-Coder-1.5B-Instruct
Benchmark: HumanEval
Split: data/splits/humaneval/eval.jsonl
Run ID: qwen15b_humaneval_smoke_eval_20260721_180903

This run generates solutions for the 4-task HumanEval smoke eval split and scores them with a split-aware EvalPlus-based HumanEval scorer in base-only mode.

Key result:

- Task count: 4
- Base pass count: 0
- Base pass rate: 0.0
- Base status counter: fail 4

Note:

- The standard EvalPlus full-dataset entry rejected the 4-task smoke samples because it expects all HumanEval tasks.
- A split-aware HumanEval smoke scorer was added for project-local smoke evaluation.
