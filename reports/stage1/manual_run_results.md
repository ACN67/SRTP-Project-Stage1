# Manual Run Results

Source: `reports/stage1/manual_run_results.jsonl`

| time | owner | group | job | status | seconds | returncode | run_dir |
|---|---|---|---|---:|---:|---:|---|
| 2026-07-18T17:32:45+08:00 | shared | common | `env_check` | success | 3.427 | 0 | `results/raw/env_check_20260718_173242` |
| 2026-07-18T17:33:00+08:00 | shared | common | `import_core_method_envs` | success | 15.27 | 0 | `results/raw/import_core_method_envs_20260718_173245` |
| 2026-07-18T17:38:52+08:00 | 常珂舒 | structured_depth_width | `slicegpt_opt125m_smoke` | success | 126.003 | 0 | `results/raw/slicegpt_opt125m_smoke_20260718_173646` |
| 2026-07-18T17:54:33+08:00 | 常珂舒 | structured_depth_width | `llmpruner_tiny_llama_layerwise_smoke` | failed | 21.95 | 1 | `results/raw/llmpruner_tiny_llama_layerwise_smoke_20260718_175411` |
| 2026-07-18T17:56:46+08:00 | 常珂舒 | structured_depth_width | `llmpruner_tiny_llama_layerwise_smoke` | success | 42.282 | 0 | `results/raw/llmpruner_tiny_llama_layerwise_smoke_20260718_175604` |
| 2026-07-18T17:57:43+08:00 | 常珂舒 | structured_depth_width | `llmpruner_tiny_llama_layerwise_smoke` | success | 34.275 | 0 | `results/raw/llmpruner_tiny_llama_layerwise_smoke_20260718_175708` |
| 2026-07-18T18:21:58+08:00 | shared | qwen_adaptation | `qwen25_coder_15b_config_probe` | success | 14.468 | 0 | `results/raw/qwen25_coder_15b_config_probe_20260718_182144` |
| 2026-07-18T18:30:05+08:00 | shared | qwen_adaptation | `qwen25_coder_3b_config_probe` | success | 11.611 | 0 | `results/raw/qwen25_coder_3b_config_probe_20260718_182954` |
| 2026-07-18T18:40:23+08:00 | shared | common | `stage1_readiness_check` | success | 0.123 | 0 | `results/raw/stage1_readiness_check_20260718_184023` |
| 2026-07-18T18:40:33+08:00 | 常珂舒 | structured_depth_width | `flab_pruner_import_probe` | success | 10.099 | 0 | `results/raw/flab_pruner_import_probe_20260718_184023` |
| 2026-07-18T18:45:01+08:00 | 潘阔 | benchmark_pipeline | `create_humaneval_mbpp_smoke_splits` | success | 5.148 | 0 | `results/raw/create_humaneval_mbpp_smoke_splits_20260718_184456` |
| 2026-07-18T18:45:09+08:00 | shared | common | `stage1_readiness_check` | success | 0.096 | 0 | `results/raw/stage1_readiness_check_20260718_184509` |
| 2026-07-18T18:47:30+08:00 | shared | qwen_adaptation | `write_qwen_model_manifests` | success | 1.118 | 0 | `results/raw/write_qwen_model_manifests_20260718_184729` |
| 2026-07-18T19:33:09+08:00 | 潘阔 | benchmark_pipeline | `create_lcb_swebench_smoke_splits` | success | 1813.722 | 0 | `results/raw/create_lcb_swebench_smoke_splits_20260718_190255` |
| 2026-07-18T19:34:12+08:00 | shared | common | `stage1_readiness_check` | success | 0.125 | 0 | `results/raw/stage1_readiness_check_20260718_193412` |
| 2026-07-18T19:40:16+08:00 | 常珂舒 | structured_depth_width | `flab_qwen3b_humaneval_dry_run` | success | 9.298 | 0 | `results/raw/flab_qwen3b_humaneval_dry_run_20260718_194007` |
| 2026-07-18T20:09:37+08:00 | 常珂舒 | structured_depth_width | `flab_qwen3b_humaneval_prune_heavy` | failed | 1540.226 | 1 | `results/raw/flab_qwen3b_humaneval_prune_heavy_20260718_194356` |
| 2026-07-18T20:15:58+08:00 | 常珂舒 | structured_depth_width | `flab_qwen3b_humaneval_dry_run` | success | 6.345 | 0 | `results/raw/flab_qwen3b_humaneval_dry_run_20260718_201552` |
| 2026-07-18T20:27:33+08:00 | 常珂舒 | structured_depth_width | `flab_qwen3b_humaneval_prune_heavy` | failed | 10.605 | 1 | `results/raw/flab_qwen3b_humaneval_prune_heavy_20260718_202722` |
