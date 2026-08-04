# Pruning Failure Report for External Analysis

## 1. Executive Summary

This repository evaluated three structured pruning routes on `Qwen/Qwen2.5-Coder-1.5B-Instruct` after the 3B LLM-Pruner path proved impractical on the available 8 GB GPU. The completed final pipelines were: FlabPruner structural Qwen2 width pruning plus LoRA merge; LLM-Pruner Taylor/MetaPruner Qwen adaptation plus LoRA adapter; and SliceGPT local Qwen2 adapter plus LoRA adapter. Dense baseline evaluation is healthy but lower than 3B: HumanEval 18/82, MBPP 115/224, LiveCodeBench 28/200. Final pruned+LoRA scores collapse for all three: Flab 0/82, 1/224, 0/200; LLM-Pruner 2/82, 1/224, 0/200; SliceGPT 0/82, 2/224, 0/200.

They are not proven to fail at the same stage. Existing evidence covers S0 dense and final S6/S7 only; dense save/reload, dense LoRA, no-op method transforms, prune-before-save, and prune-save-reload are NOT RUN. Current evidence favors method-specific Qwen adaptation/structure issues plus possible shared recovery-chain risk, not a single confirmed evaluator bug. Top supported causes: unvalidated Qwen adapters/loaders; requested vs actual compression mismatch, especially Flab and SliceGPT; and missing dense LoRA controls for the shared recovery path. The next minimum experiments are dense round-trip, dense LoRA adapter/merge, SliceGPT sparsity=0, LLM-Pruner zero/no-op, and Flab no-op or very-low-ratio smoke.

## 2. Base Model and Unified Evaluation

| Field | Value | Evidence |
|---|---|---|
| base_model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | `results/raw/qwen25c15b_official_evalhalf_20260729_172949/run_info.txt` |
| model_revision | local HF cache, exact revision NOT RECORDED | no revision field in run info |
| tokenizer_revision | local HF cache, exact revision NOT RECORDED | no revision field in run info |
| dense_parameter_count | 1,543,714,304 trainable params from pruning summaries | `llmpruner_keep80_v3/summary.json`, `flab_qwen_prune_result.json` |
| transformers_version | 5.13.1 | local `.venv-common` check |
| torch_version | 2.11.0+cu128 | local `.venv-common` check |
| peft_version | 0.19.1 | local `.venv-common` check |
| cuda_version | torch CUDA 12.8; driver 596.36 | `nvidia-smi` and torch check |
| gpu | NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB | `nvidia-smi` |
| dtype | eval/prune mostly fp16; base config bf16 | command logs and config |
| quantization | none for final Qwen1.5B runs | commands used no 4bit/8bit flags |

Base config from `AutoConfig`: hidden_size=1536, intermediate_size=8960, num_hidden_layers=28, attention_heads=12, kv_heads=2, vocab_size=151936, tie_word_embeddings=True, eos_token_id=151645, pad_token_id=None.

| Benchmark | Dense pass@1 | Samples | max_new_tokens | Prompt/Scorer | Dense hit_max | Dense mean tokens |
|---|---:|---:|---:|---|---:|---:|
| HumanEval | 0.2195 (18/82) | 82 | 512 | official HumanEval prompt + EvalPlus scorer | 0.08536585365853659 | 262.9390243902439 |
| MBPP | 0.5134 (115/224) | 224 | 512 | MBPP EvalPlus official scorer | 0.11160714285714286 | 196.125 |
| LiveCodeBench | 0.1400 (28/200) | 200 | 1024 | LCB official prompt/extract/metrics | 0.005 | 165.765 |

Dense EOS rate is NOT LOGGED by the current generator. Dense parse success is only approximated in `reports/raw_completion_failure_taxonomy.csv` because saved `samples.jsonl` contains extracted `solution`, not full raw completion.

## 3. Actual Execution of the Three Methods

| Item | FlabPruner | LLM-Pruner | SliceGPT |
|---|---|---|---|
| Upstream commit | `e35e7fc4560369f993d736df1ef5429a74ca6983` | `128a07d977f9b205d60ab14cfbc6a78f8a8e39d2` | `6b12cdee6ad51791d7c776b3a046bc408b9e77e9` |
| Local commit | `9117e9c340e1c56bc5bd6560c2418688da430853` | `9117e9c340e1c56bc5bd6560c2418688da430853` | `9117e9c340e1c56bc5bd6560c2418688da430853` |
| Entry script | `scripts/adapt/flab_qwen_official.py` | `scripts/adapt/llmpruner_qwen_official.py` | `scripts/adapt/slicegpt_qwen_official.py` |
| Qwen native support | Vendored Qwen2 path, Qwen2.5 compatibility patched | Not official native path; local Qwen2.5 adaptation | Not official native path; local Qwen2 adapter |
| Local adaptation files | `scripts/adapt/flab_qwen_official.py`; vendored Flab model | `scripts/adapt/llmpruner_qwen_official.py`, custom loader `load_llmpruner_qwen_model` | `scripts/adapt/slicegpt_qwen_official.py`, `Qwen2ModelAdapter`, `load_sliced_qwen_model` |
| Pruning basis | structural fixed width/head/FFN config; no guide importance | Taylor `param_first` backward over guide prompts | SliceGPT calibration over guide prompts |
| Calibration data | guide files recorded/validated only; LoRA uses guide teacher data | 436 guide samples | 436 guide samples |
| Requested compression | prune_ratio 0.10; explicit hidden/head/FFN targets | pruning_ratio 0.28 | sparsity 0.45 |
| Actual parameter keep | 0.8939371828221396 | 0.8202547406077543 | 0.677403188718526 |
| Recovery method | PEFT LoRA rank 8, then merged | PEFT LoRA rank 8, adapter evaluated without merge | PEFT LoRA rank 8, adapter evaluated without merge |
| Experiment nature | repository-derived Qwen2 structural adaptation | LLM-Pruner adaptation to Qwen2.5-Coder | SliceGPT adaptation to Qwen2.5-Coder |

Reconstructed final commands are in prior terminal history and summarized by run logs: `prune_v2.log`, `prune_v3.log`, `slice.log`, `lora*.log`, `eval_*log` under the three result roots. Key flags were `--local-files-only`, fp16, guide files from `data/splits/*_half/guide.jsonl`, LoRA `epochs=1`, `batch_size=1`, `grad_accum=8`.

## 4. Stage Results

Existing results do not include all requested stages. `NOT RUN` means no checkpoint or metric was found.

| Method | Stage | HumanEval | MBPP | LCB | EOS rate | Hit max tokens | Parse success | Mean length | Notes |
|---|---|---:|---:|---:|---|---|---|---|---|
| Dense | S0 dense | 0.2195 | 0.5134 | 0.1400 | not logged | mean 0.067 | sample taxonomy only | mean 208.3 | baseline full eval |
| Dense | S1 dense save/reload | NOT RUN | NOT RUN | NOT RUN | not logged | mean 0.067 | sample taxonomy only | mean 208.3 | NOT RUN |
| Dense | S2 dense + LoRA adapter | NOT RUN | NOT RUN | NOT RUN | not logged | mean 0.067 | sample taxonomy only | mean 208.3 | NOT RUN |
| Dense | S3 dense + LoRA merged | NOT RUN | NOT RUN | NOT RUN | not logged | mean 0.067 | sample taxonomy only | mean 208.3 | NOT RUN |
| FlabPruner | S4 pruned in memory | NOT RUN | NOT RUN | NOT RUN | not logged | mean 1.000 | sample taxonomy only | mean 682.7 | NOT RUN; no in-memory smoke captured before save |
| FlabPruner | S5 pruned save/reload | NOT RUN | NOT RUN | NOT RUN | not logged | mean 1.000 | sample taxonomy only | mean 682.7 | NOT RUN; pruned_model exists but no no-LoRA score found |
| FlabPruner | S6 pruned + LoRA adapter | NOT RUN | NOT RUN | NOT RUN | not logged | mean 1.000 | sample taxonomy only | mean 682.7 | NOT RUN; final path merged adapter |
| FlabPruner | S7 pruned + LoRA merged | 0.0000 | 0.0045 | 0.0000 | not logged | mean 1.000 | sample taxonomy only | mean 682.7 | final full eval |
| LLM-Pruner | S4 pruned in memory | NOT RUN | NOT RUN | NOT RUN | not logged | mean 0.704 | sample taxonomy only | mean 496.4 | NOT RUN |
| LLM-Pruner | S5 pruned save/reload | NOT RUN | NOT RUN | NOT RUN | not logged | mean 0.704 | sample taxonomy only | mean 496.4 | NOT RUN; pruned_model exists but no adapter-free score found |
| LLM-Pruner | S6 pruned + LoRA adapter | 0.0244 | 0.0045 | 0.0000 | not logged | mean 0.704 | sample taxonomy only | mean 496.4 | final full eval |
| LLM-Pruner | S7 pruned + LoRA merged | NOT RUN | NOT RUN | NOT RUN | not logged | mean 0.704 | sample taxonomy only | mean 496.4 | NOT RUN; non-uniform Qwen artifact evaluated with adapter |
| SliceGPT | S4 sliced in memory | NOT RUN | NOT RUN | NOT RUN | not logged | mean 0.960 | sample taxonomy only | mean 647.9 | NOT RUN |
| SliceGPT | S5 sliced save/reload | NOT RUN | NOT RUN | NOT RUN | not logged | mean 0.960 | sample taxonomy only | mean 647.9 | NOT RUN; no adapter-free score found |
| SliceGPT | S6 sliced + LoRA adapter | 0.0000 | 0.0089 | 0.0000 | not logged | mean 0.960 | sample taxonomy only | mean 647.9 | final full eval |
| SliceGPT | S7 sliced + LoRA merged | NOT RUN | NOT RUN | NOT RUN | not logged | mean 0.960 | sample taxonomy only | mean 647.9 | NOT RUN |

First collapse stage by actual metrics:

- FlabPruner: NOT DETERMINED. Only S0 and S7 scored.
- LLM-Pruner: NOT DETERMINED. Only S0 and S6 scored.
- SliceGPT: NOT DETERMINED. Only S0 and S6 scored.

## 5. Shared Code Paths

| Component | File path / function | Shared by all methods? | Verified normal? | Evidence |
|---|---|---|---|---|
| tokenizer loader | `AutoTokenizer.from_pretrained` in method scripts and eval generator | partially | NOT fully | tokenizer used successfully in dense eval; no round-trip diff yet |
| model save/reload | method-specific loaders | no | NOT fully | Flab normal HF; LLM/Slice custom loaders |
| generation config | `scripts/eval/generate_official_samples.py:309 model.generate` | yes for final eval | partially | dense eval normal-ish; final collapse could be model or loader |
| LoRA dataset builder | `scripts/recover/create_distill_dataset.py:106 main`, generate at line 185 | yes | partially | `reports/lora_data_audit.json`: sampled empty_answer_rate=0 |
| label masking | `scripts/recover/train_lora_recovery.py` training loop | yes | NOT RUN audit | token-label masking stats not persisted |
| PEFT trainer | `scripts/recover/train_lora_recovery.py:59 main` | yes | NOT fully | LoRA summaries show 436 samples, 55 steps |
| adapter merge | `scripts/recover/merge_lora_model.py` | only Flab final | NOT fully | adapter-vs-merged logit equivalence NOT RUN |
| HumanEval evaluator | `scripts/eval/run_official_eval.sh`, `score_humaneval_official.py` | yes | likely | dense HumanEval nonzero; official prompt mode logged |
| MBPP evaluator | `score_mbpp_evalplus_official.py` | yes | likely | dense MBPP 0.5134 |
| LiveCodeBench evaluator | `score_livecodebench_official.py` | yes | likely | dense LCB 0.14; extractor warnings logged |
| raw extractor | `generate_official_samples.py:77 build_lcb_prompt_and_extractor`, `build_model_prompt` | yes | partially | dense extraction mostly succeeds; final LCB extraction degrades |

One shared error could affect all methods through LoRA data/trainer/eval generation. However dense nonzero baseline argues against a totally broken evaluator.

## 6. Dense Round-Trip and Dense LoRA Controls

### 6.1 Dense round-trip

Status: NOT RUN.

Required metrics are unavailable: parameter_count after save/reload, config_diff, tokenizer_diff, generation_config_diff, max_abs_logit_diff, mean_abs_logit_diff, top1_token_agreement, completion_exact_match, smoke_pass_rate. Evidence gap: no directory corresponding to dense `save_pretrained -> reload -> smoke` was found in `results/raw/qwen25c15b*`.

### 6.2 Dense LoRA

Status: NOT RUN.

Required metrics are unavailable: trainable_parameter_count, matched_lora_modules, train_loss comparison to dense, all_labels_ignored_rate, training_truncation_rate, labels_with_eos_rate, adapter_vs_merged_logit_diff. Evidence gap: only pruned/sliced LoRA summaries exist under `flabpruner_*`, `llmpruner_*`, `slicegpt_*`.

## 7. FlabPruner Specific Summary

Facts:

- Entry script: `scripts/adapt/flab_qwen_official.py:206 main`.
- Validation: `validate_remain` at line 83 enforces Qwen head-dim constraints after a local patch.
- Tokenizer loading: line 239.
- Pruning implementation described in the script docstring and result: vendored `hidden_prune_utils.modeling_qwen2.Qwen2ForCausalLM.prune(config, stage)`.
- Result path: `results/raw/flabpruner_qwen25c15b_official_keep80_20260730_015031`.
- Plan: hidden 1536->1280, heads 12->10, KV heads 2->2, FFN 8960->8192, layers 28->28.
- Actual keep: 0.8939371828221396.
- Final score: HE 0.0, MBPP 0.004464285714285714, LCB 0.0.

Answers:

1. Vocabulary pruning: no evidence it was intentionally executed; vocab_size remains a config item, and Flab path is width/head/FFN structural pruning.
2. Iterative layer pruning: no; layers remain 28->28.
3. FFN four-rule comparison: NOT EVIDENCED in local script; current adapter passes fixed target FFN size.
4. Current path: yes, Qwen2 fixed structural width/head/FFN path rather than full benchmark-importance Flab.
5. `stage="top"`: passed to vendored `model.prune(config, stage)`; exact index semantics live in `third_party/flab_pruner/hidden_prune_utils/modeling_qwen2.py` and should be audited there before claiming paper equivalence.
6. Hidden size clipped: yes, 1536->1280.
7. Attention heads clipped: yes, 12->10; KV heads unchanged 2.
8. Guide prompts compute importance: no. The run record says guide files are recorded/validated; official Flab structural pruning does not derive importance scores from benchmark prompts.
9. Pruned before LoRA result: NOT RUN.
10. Official Flab checkpoint in evaluator: NOT RUN.
11. Difference from full paper method: local run is a repository-derived Qwen2 structural adapter, not a confirmed full reproduction of all FlabPruner paper procedures.

Minimal code snippet:

```python
# scripts/adapt/flab_qwen_official.py
model.prune(config=prune_config, stage=args.stage)
```

## 8. LLM-Pruner Specific Summary

Facts:

- Entry script: `scripts/adapt/llmpruner_qwen_official.py:330 main`.
- Taylor backward: `run_taylor_backward` at line 105.
- Custom reload: `load_llmpruner_qwen_model` at line 298.
- Importance type: Taylor `param_first`.
- Mode: block_wise.
- Guide samples: 436.
- Requested pruning_ratio: 0.28.
- Actual keep: 0.8202547406077543.
- Final score: HE 0.024390243902439025, MBPP 0.004464285714285714, LCB 0.0.

Required but missing metrics:

```text
importance_min: NOT RECORDED
importance_max: NOT RECORDED
importance_mean: NOT RECORDED
importance_std: NOT RECORDED
importance_nan_count: NOT RECORDED
gradient_nonzero_ratio: NOT RECORDED
```

Answers:

1. Pruning mode: block_wise.
2. Importance type: Taylor `param_first` via vendored `TaylorImportance`.
3. Backward executed: yes; `prune_v3.log` contains `taylor_backward` events and final summary status success.
4. Gradient nonzero: NOT RECORDED; must instrument gradients.
5. Importance distribution: NOT RECORDED.
6. Dependency graph: upstream MetaPruner graph used; Qwen root modules and output transform are local adaptation.
7. Qwen GQA handling: local script sets `consecutive_groups` for q_proj head_dim, but full GQA correctness is not proven.
8. Actual deleted heads/channels/neurons: non-uniform; see `llmpruner_keep80_v3/pruned_model/llmpruner_qwen_shapes.json`.
9. Request vs actual: 0.28 requested, 0.8203 parameter keep.
10. Pruned before save: NOT RUN.
11. Save/reload without adapter: NOT RUN.
12. Official support path: no; this is `LLM-Pruner adaptation to Qwen2.5-Coder`.

Minimal code snippet:

```python
# scripts/adapt/llmpruner_qwen_official.py
imp = official_pruner.TaylorImportance(group_reduction=grouping_strategy, taylor=taylor)
loss = model(input_ids, labels=input_ids).loss
loss.backward()
pruner = tp.pruner.MetaPruner(model, forward_prompts, **kwargs)
```

## 9. SliceGPT Specific Summary

Facts:

- Entry script: `scripts/adapt/slicegpt_qwen_official.py:323 main`.
- Custom sliced loader: `load_sliced_qwen_model` at line 274.
- Tokenizer load for sliced path: line 284.
- Result path: `results/raw/slicegpt_qwen25c15b_official_keep80_gpu_20260731_011001`.
- Requested sparsity: 0.45.
- Hidden/new embedding dimension: 1536->768.
- Actual keep: 0.677403188718526.
- Final score: HE 0.0, MBPP 0.008928571428571428, LCB 0.0.

Required invariant metrics:

```text
zero_sparsity_status: NOT RUN
orthogonality_error: NOT RECORDED
pre_post_rotation_logit_diff: NOT RUN
sliced_in_memory_pass_rate: NOT RUN
sliced_reloaded_pass_rate: NOT RUN without LoRA
```

Answers:

1. Official Qwen adapter: no evidence of upstream native Qwen2.5 adapter; local adapter is used.
2. Custom path: `scripts/adapt/slicegpt_qwen_official.py`.
3. ModelAdapter/LayerAdapter/compressed layer: implemented locally in that script; exact class definitions should be audited before claiming equivalence.
4. sparsity=0 invariant: NOT RUN.
5. norm fusion invariant: NOT RUN.
6. rotation-before-slicing invariant: NOT RUN.
7. orthogonality: NOT RECORDED.
8. hidden dimension changed 1536->768.
9. Reload requires extra parameters: eval/training command passes `--slicegpt-base-model`, `--slicegpt-sparsity`, `--slicegpt-round-interval`.
10. LoRA recognition of compressed layers: training summary says success, but matched module count is NOT RECORDED.
11. adapter vs merged: NOT RUN; final did not merge.
12. Official support scope: repository-derived Qwen adaptation, not confirmed official reproduction.

## 10. Raw Generation Failure Types

Aggregated taxonomy is in `reports/raw_completion_failure_taxonomy.csv`. It samples first 20 saved `solution` records per method/benchmark; it is not a full raw-completion taxonomy because current `samples.jsonl` stores extracted/solution text rather than all raw model text.

Representative paths:

- Dense samples: `results/raw/qwen25c15b_official_evalhalf_20260729_172949/*/generation/samples.jsonl`.
- Flab final samples: `results/raw/flabpruner_qwen25c15b_official_keep80_20260730_015031/eval_v2/*/generation/samples.jsonl`.
- LLM-Pruner final samples: `results/raw/llmpruner_qwen25c15b_official_keep80_gpu_20260730_105340/eval_v3/*/generation/samples.jsonl`.
- SliceGPT final samples: `results/raw/slicegpt_qwen25c15b_official_keep80_gpu_20260731_011001/eval/*/generation/samples.jsonl`.

Observed high-level pattern:

- Flab: final HumanEval/MBPP/LCB sampled outputs are dominated by `F hit_max_new_tokens`.
- LLM-Pruner: HumanEval/MBPP show heavy hit-max; LCB includes syntax errors.
- SliceGPT: HumanEval/MBPP/LCB mostly hit max, with LCB syntax errors.

## 11. Configuration Differences

Copied configs are under `reports/model_config_diff/`.

Important known values:

| Field | Dense | Flab final/pruned | LLM-Pruner | SliceGPT |
|---|---|---|---|---|
| model_type | qwen2 | qwen2 expected | qwen2 plus custom shape sidecar | qwen2 plus SliceGPT state/config |
| hidden_size | 1536 | 1280 | non-uniform; see shape sidecar | 768 effective new embedding dimension |
| intermediate_size | 8960 | 8192 | non-uniform MLP | sliced; see SliceGPT config |
| num_hidden_layers | 28 | 28 | 28 | 28 |
| attention heads | 12 | 10 | non-uniform/root-dependent | logical config needs loader |
| KV heads | 2 | 2 | custom GQA not fully proven | local adapter |
| vocab_size | 151936 | expected unchanged | expected unchanged | expected unchanged |

Not yet compared: in-memory config vs saved vs reloaded configs; generation_config diff; tokenizer diff; logits diff.

## 12. Cause Ranking

### A. Directly supported by current evidence

1. Final collapse is real under the shared official evaluator.
   - Phenomenon: pass@1 nearly zero after pruning+LoRA.
   - Evidence: score summaries listed in Section 4.
   - Affects: all three final methods.
   - Paths: `results/raw/*qwen25c15b*/*/score_summary.json`.

2. Compression/actual-keep accounting is inconsistent across methods.
   - Phenomenon: Flab actual keep 0.8939 despite keep80 intent; SliceGPT actual keep 0.6774.
   - Evidence: Flab `flab_qwen_prune_result.json`; SliceGPT `slicegpt_keep80/summary.json`.
   - Affects: Flab, SliceGPT; possibly experimental comparison validity.

3. Qwen adaptation is method-specific and unvalidated.
   - Phenomenon: LLM-Pruner and SliceGPT require custom loaders/adapters; SliceGPT sparsity=0 invariant absent.
   - Evidence: `load_llmpruner_qwen_model`, `load_sliced_qwen_model`.
   - Affects: LLM-Pruner, SliceGPT.

### B. High likelihood but still needs validation

1. Shared LoRA trainer/label masking may damage all pruned models.
   - Minimum validation: dense LoRA adapter and merged smoke.

2. Save/reload of custom structures may alter behavior.
   - Minimum validation: pruned in-memory vs saved/reloaded logits/completions.

3. SliceGPT Qwen adapter may fail invariants before slicing.
   - Minimum validation: sparsity=0 and rotation-only smoke.

### C. Not supported yet

- “All three algorithms are unsuitable for code models.” Not supported because first collapse stage is unknown.
- “Qwen2.5-Coder cannot be pruned.” Not supported because no low-ratio/no-op controls are complete.
- “LoRA cannot recover structured pruning.” Not supported because dense LoRA and adapter-vs-merge controls are missing.
- “Evaluator is broken.” Weakly opposed by dense baseline nonzero scores, though not fully disproven.

## 13. Minimal Follow-up Experiments

1. Dense round-trip smoke
   - Purpose: test save/load/tokenizer/generation stability.
   - Command: run 20 fixed HumanEval/MBPP/LCB prompts before and after `save_pretrained`.
   - Cost: low, minutes.
   - Success: logit diff near zero, completion exact match high, smoke pass similar.
   - Failure: dense after reload degrades.
   - Output: `results/raw/audit_dense_roundtrip_*`.

2. Dense LoRA adapter and merged smoke
   - Purpose: isolate shared LoRA data/trainer/merge.
   - Command: train LoRA on dense base using existing `shared_lora_data`, evaluate adapter and merged on 20 prompts.
   - Cost: low-medium.
   - Success: no collapse relative to dense.
   - Failure: dense LoRA collapses.
   - Output: `results/raw/audit_dense_lora_*`.

3. Adapter-vs-merged equivalence for Flab
   - Purpose: test merge correctness.
   - Command: same prompts/logits for adapter-loaded and merged model.
   - Cost: low.
   - Success: top1 agreement high, logit diff small.
   - Failure: merged diverges.
   - Output: `results/raw/audit_flab_merge_equiv_*`.

4. Flab no-op or tiny-ratio smoke
   - Purpose: test method conversion before strong pruning.
   - Success: no-op matches dense.
   - Failure: Flab conversion itself breaks.

5. LLM-Pruner zero/no-op smoke
   - Purpose: test dependency graph/save/reload without deletion.
   - Success: no-op matches dense.
   - Failure: Qwen adaptation or loader breaks.

6. SliceGPT sparsity=0 smoke
   - Purpose: test replace/fuse/rotation adapter invariants.
   - Success: logits/completions preserved.
   - Failure: Qwen SliceGPT adapter incorrect.

7. Extreme low-ratio pruning smoke for each method
   - Purpose: determine sensitivity curve.
   - Success: small degradation only.
   - Failure: any deletion causes collapse.

8. Official supported-model smoke
   - Purpose: separate local environment failure from Qwen migration.
   - Success: official model path works.
   - Failure: common environment/process issue.

## 14. Current Status Labels

| Method | Labels | Explanation |
|---|---|---|
| FlabPruner | `REPOSITORY_DERIVED_ADAPTATION`, `OFFICIAL_REPRODUCTION_INCOMPLETE`, `INSUFFICIENT_EVIDENCE` | Uses vendored Qwen2 structural path; guide importance not used; S4/S5 missing. |
| LLM-Pruner | `QWEN_ADAPTATION_UNVALIDATED`, `REPOSITORY_DERIVED_ADAPTATION`, `INSUFFICIENT_EVIDENCE` | Taylor backward ran, but Qwen support is local and importance stats/S5 missing. |
| SliceGPT | `QWEN_ADAPTATION_UNVALIDATED`, `REPOSITORY_DERIVED_ADAPTATION`, `INSUFFICIENT_EVIDENCE` | Local Qwen adapter; sparsity=0 invariant missing; actual keep is 0.6774. |
| Shared pipeline | `PIPELINE_FAILURE_SUSPECTED` | Shared LoRA/eval path could contribute, but dense LoRA not run. |

## 15. Attachment Index

- `docs/cross_method_failure_audit.md`: first-pass cross-method audit and component map.
- `reports/cross_method_stage_metrics.csv`: dense/final stage metrics derived from score summaries and generate logs.
- `reports/pruning_method_comparison.csv`: method comparison, ratios, support status, shared scripts.
- `reports/pruning_method_comparison.md`: compact comparison table.
- `reports/raw_completion_failure_taxonomy.csv`: sampled failure taxonomy from saved solutions.
- `reports/lora_data_audit.json`: 20-sample LoRA dataset key/empty-answer audit.
- `reports/model_config_diff/`: copied pruned/sliced config files for diffing.
- `patches/flabpruner.patch`: local Flab validation changes.
- `patches/llmpruner.patch`: local LLM-Pruner gradient checkpoint / CPU grad accumulation changes.
- `patches/slicegpt.patch`: currently empty because no uncommitted SliceGPT script diff was detected.
- `results/raw/qwen25c15b_official_evalhalf_20260729_172949/`: dense baseline generations and scores.
- `results/raw/flabpruner_qwen25c15b_official_keep80_20260730_015031/`: Flab prune/LoRA/merge/eval logs.
- `results/raw/llmpruner_qwen25c15b_official_keep80_gpu_20260730_105340/`: LLM-Pruner prune/LoRA/eval logs.
- `results/raw/slicegpt_qwen25c15b_official_keep80_gpu_20260731_011001/`: SliceGPT slice/LoRA/eval logs.

## Handoff summary

The project moved from Qwen2.5-Coder-3B to `Qwen/Qwen2.5-Coder-1.5B-Instruct` because the 3B LLM-Pruner Taylor path was too slow or unstable on the available 8 GB GPU. Dense 1.5B official half-eval results are nonzero: HumanEval 18/82 (0.2195), MBPP 115/224 (0.5134), LiveCodeBench 28/200 (0.14). Three final pruning+LoRA pipelines were completed. FlabPruner used a vendored Qwen2 structural width/head/FFN pruning path through `scripts/adapt/flab_qwen_official.py`, then PEFT LoRA and merge; final scores are HumanEval 0, MBPP 1/224, LCB 0. LLM-Pruner used local Qwen adaptation in `scripts/adapt/llmpruner_qwen_official.py`, Taylor `param_first` over 436 guide samples, requested ratio 0.28, actual parameter keep 0.8203; final adapter scores are HE 2/82, MBPP 1/224, LCB 0. SliceGPT used a local Qwen adapter in `scripts/adapt/slicegpt_qwen_official.py`, requested sparsity 0.45, actual keep 0.6774, and final adapter scores HE 0, MBPP 2/224, LCB 0.

The common abnormal symptom is severe final generation collapse, often hitting `max_new_tokens`; Flab and SliceGPT final sampled outputs are especially dominated by hit-max behavior. However, the first collapse stage is not known. Existing data covers only S0 dense and final S6/S7; dense save/reload, dense LoRA adapter/merged, pruned in-memory, pruned save/reload, and no-op method transforms are missing. Therefore it is not justified to conclude that all three algorithms intrinsically fail on code models or that Qwen cannot be pruned. The evidence most strongly supports three hypotheses: unvalidated Qwen method adapters/loaders, mismatched compression accounting or excessive structural change, and possible shared LoRA recovery-chain problems. The evaluator is less likely to be the only cause because dense baseline scores are nonzero under the same official scripts.

The next analyst should first run smoke tests, not full benchmarks: dense round-trip; dense LoRA adapter and merged; adapter-vs-merged equivalence; Flab no-op/low-ratio; LLM-Pruner zero/no-op; SliceGPT sparsity=0; then official-supported model smoke. Only after these locate S1-S7 first-break stages should the team rerun full benchmarks or adjust pruning ratios.
