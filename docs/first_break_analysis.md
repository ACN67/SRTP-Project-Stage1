# First Break Analysis

## Existing Checkpoint Stage Metrics

| Method | Stage | n | pass_rate | hit_max_rate | eos_rate | parse_success_rate | mean_generated_tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| FlabPruner | F0_dense | 20 | 0.400 | 0.100 | 0.200 | 0.550 | 154.4 |
| FlabPruner | F1_pruned_no_lora | 20 | 0.000 | 1.000 | 0.000 | 0.400 | 614.4 |
| FlabPruner | F2_pruned_lora_adapter | 20 | 0.000 | 1.000 | 0.000 | 0.550 | 614.4 |
| FlabPruner | F3_merged | 20 | 0.000 | 1.000 | 0.000 | 0.550 | 614.4 |
| LLM-Pruner | L0_dense | 20 | 0.400 | 0.100 | 0.200 | 0.550 | 154.4 |
| LLM-Pruner | L1_pruned_no_lora | 20 | 0.150 | 0.050 | 0.150 | 0.600 | 118.0 |
| LLM-Pruner | L2_pruned_lora_adapter | 20 | 0.000 | 0.800 | 0.150 | 0.200 | 499.4 |
| SliceGPT | S0_dense | 20 | 0.400 | 0.100 | 0.200 | 0.550 | 154.4 |
| SliceGPT | S1_sliced_no_lora | 20 | 0.000 | 0.800 | 0.200 | 0.200 | 409.8 |
| SliceGPT | S2_sliced_lora_adapter | 20 | 0.000 | 1.000 | 0.000 | 0.200 | 614.4 |

## First Observable Break

- FlabPruner: F1_pruned_no_lora. Dense smoke pass_rate=0.400; F1 pass_rate=0.000, hit_max_rate=1.000, eos_rate=0.000. LoRA adapter and merge do not recover it.
- LLM-Pruner: L1_pruned_no_lora is damaged but not fully collapsed: dense pass_rate=0.400, L1=0.150, hit_max_rate=0.050. First zero-pass stage is L2_pruned_lora_adapter: pass_rate=0.000, hit_max_rate=0.800. This points to both pruning damage and a large additional LoRA/recovery degradation.
- SliceGPT: S1_sliced_no_lora. Dense pass_rate=0.400; S1 pass_rate=0.000, hit_max_rate=0.800. LoRA adapter worsens hit_max_rate to 1.000.

## Flab Adapter vs Merge

- max_abs_logit_diff_max: 0.05419921875
- mean_abs_logit_diff_mean: 0.006351229595020413
- top1_token_agreement: 1.0
- greedy_completion_exact_match: 1.0
- Interpretation: Flab adapter-loaded and merged models are effectively equivalent on the fixed 20 prompts; merge is not the first observable failure source.

## Damage Attribution From Existing Checkpoints

- FlabPruner: pruning/save-reload checkpoint already collapses before LoRA; observed LoRA contribution is recovery failure/no improvement, not root cause.
- LLM-Pruner: pruning alone reduces 20-sample pass_rate from 0.400 to 0.150; LoRA adapter further reduces it to 0.000 and increases hit_max_rate to 0.800.
- SliceGPT: sliced checkpoint already collapses before LoRA; LoRA adapter further increases hit_max behavior.

## Common Pipeline Controls

- Evaluator/generation pipeline is partially controlled by repeated dense stages F0/L0/S0, all with pass_rate 0.400 on the same fixed 20 prompts.
- Dense save/reload and dense LoRA controls are NOT RUN in this pass.
- Flab merge equivalence PASSED.

## Stop/Go Decision

- Do not enter low-compression curve experiments yet. Existing checkpoints show Flab and SliceGPT collapse before recovery, and LLM-Pruner recovery adds a separate collapse.
- Next minimal checks should be no-op invariants and dense LoRA, not wider full benchmarks.

## Renaming Existing Experiments

- Flab: relabel from keep80 to keep89.39 (`actual_param_ratio=0.8939371828221396`).
- LLM-Pruner: relabel from keep80 to keep82.03 (`actual_param_ratio=0.8202547406077543`).
- SliceGPT: relabel from keep80 to keep67.74 (`parameter_keep_ratio=0.677403188718526`).

## Outputs

- `data/audit/fixed_smoke_20.json`
- `reports/existing_checkpoint_stage_metrics.csv`
- `reports/existing_checkpoint_stage_summary.json`
- `reports/flab_adapter_merge_equivalence.json`
- `results/evidence/diagnostics/existing_checkpoint_stage_audit_20260802_194533/`
