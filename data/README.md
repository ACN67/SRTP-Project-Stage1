# Data

## Manifests

| Path | Purpose |
|---|---|
| `manifests/upstream_repos.csv` | Upstream repository URLs and commits. |
| `manifests/stage1_model_policy.csv` | Method-to-model policy. |
| `manifests/models/` | Probed model configuration snapshots. |

## Splits

| Split | Purpose |
|---|---|
| `splits/humaneval/` | Four-task HumanEval smoke guide/eval split. |
| `splits/mbpp/` | Original MBPP smoke split. |
| `splits/mbpp_evalplus/` | EvalPlus-compatible MBPP smoke eval split. |
| `splits/livecodebench/` | LiveCodeBench smoke split. |
| `splits/swebench_lite/` | SWE-bench Lite smoke split. |

R4 uses these required guide/eval splits:

```text
splits/humaneval_half/
splits/mbpp_evalplus_half/
splits/livecodebench_half/
```

HumanEval and LiveCodeBench are stratified half splits. MBPP uses the official sanitized split mapping through EvalPlus-compatible task IDs: prompt/train/validation are guide, and test is eval.
