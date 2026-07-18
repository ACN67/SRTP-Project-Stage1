# Flab-Pruner Qwen3B Heavy Run

This document contains the long-running command for the first Flab-Pruner Qwen2.5-Coder-3B pruning attempt.

Run this only when the machine can spend time downloading/loading Qwen3B weights.

## Enter Project

If you are in PowerShell:

```powershell
wsl -d Ubuntu
```

Then in Ubuntu:

```bash
cd ~/projects/srtp-code-llm-pruning
source scripts/setup/env.sh
```

## Recommended Precheck

The dry-run has already succeeded once, but you can rerun it cheaply:

```bash
scripts/run/run_plan.sh \
  --plan configs/experiments/stage1_manual_plan.yaml \
  --only flab_qwen3b_humaneval_dry_run \
  --include-disabled
```

## Heavy Pruning Run

This command loads Qwen2.5-Coder-3B-Instruct weights and attempts to save a pruned model:

```bash
scripts/run/run_plan.sh \
  --plan configs/experiments/stage1_manual_plan.yaml \
  --only flab_qwen3b_humaneval_prune_heavy \
  --include-disabled
```

Expected success line:

```text
flab_qwen3b_humaneval_prune_heavy: success (...)
```

## Files To Review Afterward

Send Codex the final terminal lines or ask it to inspect:

```text
reports/stage1/manual_run_results.jsonl
results/raw/<run_id>/summary.json
results/raw/<run_id>/stdout.log
results/raw/<run_id>/stderr.log
results/raw/<run_id>/flab_qwen3b_humaneval/flab_qwen_prune_plan.json
results/raw/<run_id>/flab_qwen3b_humaneval/flab_qwen_prune_result.json
```

The pruned model directory may be large:

```text
results/raw/<run_id>/flab_qwen3b_humaneval/pruned_model/
```

Do not commit large model weights until we decide whether to store them through Git LFS or keep only a manifest.

## Known Limitation

This first heavy run validates the Qwen3B pruning path and records the HumanEval guide split hash. The current upstream Flab Qwen2 pruning function still uses structural stage selection (`top`, `bottom`, `random`, `middle`). A later patch is needed for fully benchmark-scored mask selection.
