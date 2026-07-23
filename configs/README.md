# Configs

`experiments/stage1_manual_plan.yaml` is the current manual experiment plan.

Heavy jobs are disabled by default and should be run explicitly with:

```bash
scripts/run/run_plan.sh \
  --plan configs/experiments/stage1_manual_plan.yaml \
  --only <job_id> \
  --include-disabled
```
