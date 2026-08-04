# Magnitude Pruning

Owner: 潘阔  
Stage: 1  
Method group: activation_mask_allocation  
Current target: R1 official reproduction through the Wanda repository

## R0 Summary

Magnitude pruning is the lowest unstructured sparsity baseline for Stage 1. It removes weights by absolute value only and does not use activation statistics. In this project it is run through the Wanda upstream repository so that Magnitude and Wanda share the same model loading, calibration, logging, and saving path.

## Upstream

- Repository: `https://github.com/locuslab/wanda.git`
- Pinned commit: `8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6`
- Local path: `third_party/wanda`
- License: MIT
- Paper context: Wanda paper uses Magnitude as a direct baseline against activation-aware pruning.

## Official Command Shape

The upstream Wanda README exposes `--prune_method magnitude` as one of the supported methods. The Stage 1 smoke job uses `facebook/opt-125m` with reduced calibration samples:

```bash
workflows/experiment/run_plan.sh \
  --plan workflows/experiment/stage1_plan.yaml \
  --only wanda_opt125m_magnitude_smoke \
  --include-disabled
```

Underlying upstream command:

```bash
python main_opt.py \
  --model facebook/opt-125m \
  --prune_method magnitude \
  --sparsity_ratio 0.3 \
  --sparsity_type unstructured \
  --nsamples 8 \
  --save "$RUN_DIR/magnitude_results" \
  --save_model "$RUN_DIR/magnitude_model"
```

## Stage 1 Acceptance Checks

- The run directory contains `metadata.json`, `command.sh`, `stdout.log`, `stderr.log`, `resource.csv`, and `summary.json`.
- The saved result records target sparsity and actual sparsity.
- The saved model directory can be reloaded or represented by a manifest.
- The run is preserved even if the perplexity or task score is poor.

## Qwen Notes

Magnitude is expected to transfer to Qwen once a Qwen-compatible Linear-layer enumeration path is available. It should share the same Qwen wrapper as Wanda, but without activation hooks.

## Status

**Smoke (Stage 1 week):** flow / trend only — do not cite as paper main table.

- R0: done
- R1 smoke: OPT-125M 30%/50%/2:4 with `nsamples=8` (`..._165739`, `..._magnitude_50pct_*`, `..._magnitude_2to4_*`)
- R2 smoke: `qwen15b_magnitude_humaneval_r2_20260724_194150` + 4-task HE/MBPP generation

**Formal (paper-grade comparison):** `external_evidence_missing/pan_formal_20260724_203248/` → [`results/auxiliary/pan_full_eval/pan_formal_comparison.csv`](../../results/auxiliary/pan_full_eval/pan_formal_comparison.csv)

- OPT-125M `nsamples=128`: Mag 30% PPL 34.51; Mag 50% PPL 193.34
- Qwen1.5B Mag 0.10/0.30 (HE+MBPP guides) + full evalplus Pass@1 (HE 164, MBPP 378)
