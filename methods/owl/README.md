# OWL

Owner: 潘阔  
Stage: 1  
Method group: activation_mask_allocation  
Current target: R1 official flow with Wanda combination

## R0 Summary

OWL, Outlier Weighed Layerwise Sparsity, assigns non-uniform sparsity ratios across layers according to outlier statistics. It is a layer-allocation strategy that should be applied on top of a base pruning method such as Wanda or SparseGPT.

Stage 1 treats OWL as a combination strategy. A valid OWL result must include both the uniform base-pruner result and the OWL non-uniform allocation result.

## Upstream

- Paper: `Outlier Weighed Layerwise Sparsity: A Missing Secret Sauce for Pruning LLMs`
- Repository: `https://github.com/luuyin/OWL.git`
- Pinned commit: `dddb7a4bffe27c73e4c8cf692b3a5e36401532c8`
- Local path: `third_party/owl`
- License: MIT
- Official supported base methods: `wanda_owl`, `wanda_owl_structure`, `sparsegpt_owl`, `magnitude_owl`, plus uniform `magnitude`, `wanda`, and `sparsegpt`

## Official Command Shape

The upstream README example for OWL + Wanda is:

```bash
python main.py \
  --model_name_or_path decapoda-research/llama-7b-hf \
  --Lamda 0.08 \
  --Hyper_m 5 \
  --model decapoda-research/llama-7b-hf \
  --prune_method wanda_owl \
  --sparsity_ratio 0.7 \
  --sparsity_type unstructured \
  --save save_test/
```

Stage 1 adds a reduced smoke job, `owl_llama_wanda_smoke`, in the manual experiment plan. It defaults to a tiny LLaMA-compatible model for flow testing and can be pointed to a real accessible model with `OWL_MODEL`.

## Required Evidence

- Uniform Wanda baseline for the same model, split, sparsity, and sample count.
- OWL non-uniform allocation result.
- Per-layer allocation table.
- Total sparsity check proving the global target is still respected.
- Outlier or layer-sensitivity summary used to derive the allocation.

## Qwen Notes

OWL should follow Wanda's Qwen adaptation. Once Wanda can collect Qwen activation statistics, OWL can reuse those hooks to produce non-uniform layer ratios. If Qwen support is not completed during Stage 1, OWL should at least keep the official R1 result and the Qwen compatibility notes.

## Status

**Smoke only for Stage 1 / this formal round:** no paper-grade Qwen Pass@1; OWL remains `unsupported_without_adapter` on Qwen.

- R0: done
- R1 smoke: `owl_llama_wanda_smoke_20260724_172616` (tiny-random Llama, layer 0.195/0.375, overall 0.2852)
- Env note: pin `transformers==4.40.2` in `.venv-owl`
- Qwen R2: unsupported without adapter — `results/evidence/dsnot_owl_qwen_probe_*/owl_qwen_probe.json`
- HE/MBPP smoke: pipeline dry-run under `owl_he_mbpp_smoke_*`
- Formal rerun: **no OWL Qwen numbers**; if paper needs OWL digits, prefer LLaMA-family uniform Wanda vs OWL PPL (deferred)
