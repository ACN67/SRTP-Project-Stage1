# 潘阔 Stage 1 一周计划执行摘要

Date: 2026-07-24  
Owner: 潘阔  
Scope: Magnitude, Wanda, DSnoT, OWL, Benchmark pipeline, cross-repro  
Execution host: WSL `~/projects/srtp-code-llm-pruning` (RTX 4060 Laptop 8GB)

## Verdict

潘阔 Stage 1 责任范围内的必须项已完成：四方法 R1、Magnitude/Wanda Qwen R2、Wanda 四 Benchmark 代表覆盖、HE/MBPP smoke、LiveCodeBench CLI、交叉复跑，以及 DSnoT/OWL 的 Qwen 失败证据 + process smoke。

## Real Experiment Results

### Methods R1

| Method | Run | Key metric |
|---|---|---|
| Magnitude 30% | `wanda_opt125m_magnitude_smoke_20260724_165739` | sparsity 0.3001, PPL 34.5106 |
| Wanda 30% | `wanda_opt125m_wanda_smoke_20260724_170444` | sparsity 0.2996, PPL 29.2267 |
| Magnitude 50% | `wanda_opt125m_magnitude_50pct_20260724_194822` | sparsity 0.5001 |
| Wanda 50% | `wanda_opt125m_wanda_50pct_20260724_194822` | sparsity 0.5000 |
| Magnitude 2:4 | `wanda_opt125m_magnitude_2to4_20260724_194822` | sparsity 0.5000 |
| Wanda 2:4 | `wanda_opt125m_wanda_2to4_20260724_194822` | sparsity 0.5000 |
| DSnoT | `dsnot_llama_official_template_20260724_173106` | OPT-125M, sparsity 0.2996, PPL 29.1652 |
| OWL | `owl_llama_wanda_smoke_20260724_172616` | layer 0.195/0.375, overall 0.2852 |

### Qwen R2 / R3 / four-benchmark (Wanda representative)

| Track | Run | Result |
|---|---|---|
| Magnitude Qwen R2 + HE/MBPP | `qwen15b_magnitude_humaneval_r2_20260724_194150` | sparsity ≈0.0997; HE/MBPP generation |
| Wanda HE R2/R3 | `qwen15b_wanda_humaneval_r2_smoke_20260724_170645` / `wanda_qwen15b_humaneval_r3_eval_20260724_173440` | prune + 4/4 gen |
| Wanda MBPP R2/R3 | `qwen15b_wanda_mbpp_r2_20260724_194241` | prune + generation |
| Wanda LCB R2 + gen | `qwen15b_wanda_lcb_r2_20260724_194241` | prune + 4/4 gen |
| Wanda SWE R2 + gen | `qwen15b_wanda_swe_r2_20260724_194241` | prune + 4/4 gen |
| Unpruned baselines | `qwen15b_{humaneval,mbpp}_baseline_20260724_173332` | 4/4 each |

### Benchmark pipeline

- Splits + leakage audit: done (`results/auxiliary/pan_full_eval/split_leakage_check_formal.json`)
- LiveCodeBench CLI: `livecodebench_import_and_help` success (WSL `.venv-livecodebench` + torch)
- evalplus full-set Pass@1: smoke week N/A on 4-task split; **formal** full HE/MBPP Pass@1 completed (see Formal Rerun)

### DSnoT / OWL Qwen + HE/MBPP

| Item | Artifact |
|---|---|
| DSnoT Qwen probe | `dsnot_owl_qwen_probe_20260724_194719/dsnot_qwen_probe.json` (`unsupported_without_adapter`) |
| OWL Qwen probe | `.../owl_qwen_probe.json` |
| DSnoT HE/MBPP process smoke | `dsnot_opt125m_he_mbpp_smoke_20260724_194719` |
| OWL HE/MBPP pipeline dry-run | `owl_he_mbpp_smoke_20260724_194719` |

### Cross-reproduction (常珂舒)

| Job | Run | Status |
|---|---|---|
| LLM-Pruner | `llmpruner_tiny_llama_layerwise_smoke_20260724_173756` | success |
| SliceGPT | `slicegpt_opt125m_smoke_20260724_192105` | success |

## Repo / Env Work

- Method notes updated: `docs/methods/{magnitude,wanda,dsnot,owl}.md`
- Plan jobs added: Magnitude Qwen R2, Wanda MBPP R2
- Eval scripts prefer `.venv-common/bin/python`
- Compat pins: OWL/DSnoT/LLM-Pruner `transformers==4.40.2`; SliceGPT `wandb`; LCB torch
- Former blockers marked resolved under `reports/failures/`
- Status table: `results/registries/pan_method_status_original.csv`

## Formal Rerun (paper-grade comparison)

Do **not** treat 4-task smoke generations as main conclusions. Formal numbers live in:

- Table: [`results/auxiliary/pan_full_eval/pan_formal_comparison.csv`](../../results/auxiliary/pan_full_eval/pan_formal_comparison.csv)
- Summary: [`results/auxiliary/pan_full_eval/formal_results.md`](pan_formal_results.md)
- Raw root (WSL): `external_evidence_missing/pan_formal_20260724_203248/`

### Protocol

- Model (code): `Qwen/Qwen2.5-Coder-1.5B-Instruct` only; no 7B/13B; 3B not used as main result
- Splits: `data/benchmarks/auxiliary_full_eval/humaneval/` (guide 32 / eval 164), `mbpp_formal/` (guide 32), plus `mbpp_evalplus/` (378) for evalplus IDs
- Decoding: greedy (`do_sample=false`); HumanEval/MBPP full Pass@1 via evalplus
- OPT LM: `facebook/opt-125m`, `nsamples=128`, sparsity 0.3 / 0.5
- Single seed; claim = fair same-protocol comparison, not Wanda paper absolute SOTA

### OPT-125M WikiText PPL (`nsamples=128`)

| Method | 30% PPL | 50% PPL |
|---|---:|---:|
| Magnitude | 34.51 | 193.34 |
| Wanda | 28.11 | 38.96 |
| DSnoT (init=Wanda) | 28.20 | 41.67 |

### Qwen Pass@1 (evalplus base / plus)

| Method | Sparsity | Calib | HumanEval | MBPP |
|---|---:|---|---:|---:|
| Dense | 0 | — | 0.152 / 0.134 | 0.659 / 0.561 |
| Magnitude | 0.10 | HE guide | 0.165 / 0.140 | 0.664 / 0.561 |
| Wanda | 0.10 | HE guide | 0.152 / 0.134 | 0.680 / 0.577 |
| Magnitude | 0.30 | HE guide | 0.134 / 0.122 | 0.601 / 0.513 |
| Wanda | 0.30 | HE guide | 0.159 / 0.140 | 0.648 / 0.550 |
| Magnitude | 0.10 | MBPP guide | 0.165 / 0.140 | 0.664 / 0.561 |
| Wanda | 0.10 | MBPP guide | 0.152 / 0.134 | 0.669 / 0.571 |

### Boundary (unchanged)

- DSnoT: OPT PPL only vs Wanda under formal protocol; **no** Qwen Pass@1
- OWL: R1 tiny-Llama smoke only; **no** Qwen Pass@1 (`unsupported_without_adapter`)
- Absolute HE Pass@1 (~15%) is decoding/prompt limited; use for relative Mag/Wanda/Dense comparison

## Explicitly Out of Scope / Deferred

- Qwen 3B/7B main results; multi-seed variance
- LCB/SWE full official scoring (pipeline evidence retained from smoke week)
- DSnoT/OWL Qwen adapters
- 李长骏 cross-repro of Wanda (owner: 李长骏, not 潘阔)
- Claiming reproduction of Wanda paper absolute SOTA numbers
