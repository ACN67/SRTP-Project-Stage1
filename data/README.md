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

R4 half-set splits should be added beside these smoke splits with a `_half` suffix.
