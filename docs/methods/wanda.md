# Wanda

Owner: 潘阔  
Stage: 1  
Method group: activation_mask_allocation  
Current target: R1 official reproduction, then Qwen2.5-Coder-1.5B R2/R3

## R0 Summary

Wanda prunes each output channel by combining weight magnitude with the input activation norm. It is the representative activation-aware pruning method for 潘阔's Stage 1 method group and is the preferred path for Qwen R2/R3 and benchmark-guided pruning.

## Upstream

- Paper: `A Simple and Effective Pruning Approach for Large Language Models`
- Repository: `https://github.com/locuslab/wanda.git`
- Pinned commit: `8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6`
- Local path: `third_party/wanda`
- License: MIT
- Official supported families in README: LLaMA/LLaMA-2 and OPT

## Official Command Shape

The upstream README example for LLaMA uses:

```bash
python main.py \
  --model decapoda-research/llama-7b-hf \
  --prune_method wanda \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --save out/llama_7b/unstructured/wanda/
```

Stage 1 starts with the smaller OPT path:

```bash
scripts/run/run_plan.sh \
  --plan configs/experiments/stage1_manual_plan.yaml \
  --only wanda_opt125m_wanda_smoke \
  --include-disabled
```

Underlying command:

```bash
python main_opt.py \
  --model facebook/opt-125m \
  --prune_method wanda \
  --sparsity_ratio 0.3 \
  --sparsity_type unstructured \
  --nsamples 8 \
  --save "$RUN_DIR/wanda_results" \
  --save_model "$RUN_DIR/wanda_model"
```

## Qwen R2 Path

The Qwen2.5-Coder-1.5B config probe records the expected module layout:

- Decoder layers: `model.layers`
- Attention projections: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- MLP projections: `gate_proj`, `up_proj`, `down_proj`
- Norms: `input_layernorm`, `post_attention_layernorm`, `model.norm`

The Qwen wrapper should use project split files as calibration text, starting with `data/splits/humaneval/guide.jsonl`. The first R2 target is intentionally small: Qwen2.5-Coder-1.5B-Instruct, four HumanEval guide samples, unstructured 10% sparsity, batch size 1, and sequence length 512.

## Benchmark-Guided R3 Path

For Stage 1 formal evidence, Wanda should run at least:

- HumanEval guide pruning followed by HumanEval eval validation.
- MBPP guide/eval if time permits.
- LiveCodeBench and SWE-bench Lite remain required at the pipeline level even if full evaluation is deferred.

## Acceptance Checks

- Activation hooks fire for all targeted Linear layers.
- Activation statistics contain no NaN/Inf values.
- Actual sparsity matches the target within a small tolerance.
- Per-layer sparsity is saved as CSV.
- The pruned model can be reloaded or represented by a reproducible manifest.

## Status

**Smoke (Stage 1 week):** flow / trend only — do not cite as paper main table.

- R0: done
- R1 smoke: OPT-125M 30%/50%/2:4 with `nsamples=8`
- R2/R3 smoke: four-benchmark guide prune + 4-task generation; evalplus full-set N/A on smoke split

**Formal (paper-grade comparison):** `results/raw/pan_formal_20260724_203248/` → [`results/tables/pan_formal_comparison.csv`](../../results/tables/pan_formal_comparison.csv)

- OPT-125M `nsamples=128`: Wanda 30% PPL 28.11; 50% PPL 38.96
- Qwen1.5B Wanda 0.10/0.30 (HE+MBPP guides) + full evalplus Pass@1 vs Dense/Magnitude
