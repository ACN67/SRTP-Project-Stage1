# DSnoT

Owner: 潘阔  
Stage: 1  
Method group: activation_mask_allocation  
Current target: R1 official flow or reproducible blocker

## R0 Summary

DSnoT, Dynamic Sparse No Training, is a training-free sparse-mask correction method. It starts from a sparse model produced by a base pruning method, such as Wanda, then iteratively prunes and regrows weights to reduce reconstruction error without gradient training.

Stage 1 treats DSnoT as a combination strategy rather than a standalone base pruner. Every DSnoT result must preserve the base mask/result and the post-DSnoT mask/result.

## Upstream

- Paper: `Dynamic Sparse No Training: Training-Free Fine-tuning for Sparse LLMs`
- Repository: `https://github.com/zyxxmu/DSnoT.git`
- Pinned commit: `26162b71aef5c8fee7775d5a32546d7243c6cb88`
- Local path: `third_party/dsnot`
- Official environment file: `third_party/dsnot/environment.yaml`
- Official examples: LLaMA-family models with Wanda initialization

## Official Command Shape

The upstream README example is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
  --model decapoda-research/llama-7b-hf \
  --prune_method DSnoT \
  --initial_method wanda \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --max_cycle_time 50 \
  --update_threshold 0.1 \
  --pow_of_var_regrowing 1
```

Stage 1 uses a reduced template:

```bash
workflows/experiment/run_plan.sh \
  --plan workflows/experiment/stage1_plan.yaml \
  --only dsnot_llama_official_template \
  --include-disabled
```

Set `DSNOT_MODEL` before running if the default LLaMA model is inaccessible:

```bash
export DSNOT_MODEL="trl-internal-testing/tiny-random-LlamaForCausalLM"
```

## Required Evidence

- Base pruning method and base mask/result.
- DSnoT optimized mask/result.
- Change ratio between the base mask and DSnoT mask.
- Iteration count, `max_cycle_time`, `update_threshold`, and `pow_of_var_regrowing`.
- Failure classification if the method cannot run: environment, model access, model structure, memory, or pruning algorithm.

## Qwen Notes

Qwen work should wait until Wanda's Qwen R2 path is stable, because DSnoT depends on a valid base sparse mask. If the official DSnoT code cannot load Qwen directly, Stage 1 should record an R1 official result and a Qwen compatibility blocker rather than hiding the failure.

## Status

**Smoke:** OPT R1 with lower `nsamples`; Qwen = `unsupported_without_adapter` (no paper-grade Qwen Pass@1).

- R0: done
- R1 smoke: `dsnot_llama_official_template_20260724_173106` (OPT-125M, init=wanda, PPL 29.1652 @ lower nsamples)
- Qwen R2: unsupported without adapter — `results/evidence/dsnot_owl_qwen_probe_*/dsnot_qwen_probe.json`
- HE/MBPP smoke: process evidence on OPT pruned model (`dsnot_opt125m_he_mbpp_smoke_*`)

**Formal:** OPT-125M Mag/Wanda/DSnoT same protocol, `nsamples=128` — DSnoT 30% PPL 28.20; 50% PPL 41.67 (see `pan_formal_comparison.csv`). No Qwen formal Pass@1.
