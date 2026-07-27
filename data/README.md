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
| `splits/humaneval/` | Historical four-task HumanEval smoke split retained for provenance only. |
| `splits/mbpp/` | Historical original MBPP smoke split retained for provenance only. |
| `splits/mbpp_evalplus/` | Historical EvalPlus-compatible MBPP smoke split retained for provenance only. |
| `splits/livecodebench/` | Historical LiveCodeBench smoke split retained for provenance only. |
| `splits/swebench_lite/` | Historical SWE-bench Lite smoke split retained for provenance only. |

R4 uses these required guide/eval splits:

```text
splits/humaneval_half/
splits/mbpp_evalplus_half/
splits/livecodebench_half/
```

HumanEval and LiveCodeBench are stratified half splits. MBPP uses the official sanitized split mapping through EvalPlus-compatible task IDs: prompt/train/validation are guide, and test is eval.

New formal runs must use `docs/official_benchmarks.md` and `scripts/eval/run_official_eval.sh`; do not create new paper-facing results from the historical smoke splits.
