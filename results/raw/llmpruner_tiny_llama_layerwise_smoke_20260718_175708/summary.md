# LLM-Pruner Tiny LLaMA Layer-Wise Smoke

Status: success

Method: LLM-Pruner
Owner: 常珂舒
Model: trl-internal-testing/tiny-random-LlamaForCausalLM
Run ID: llmpruner_tiny_llama_layerwise_smoke_20260718_175708

Purpose:

- Repeat the minimal LLM-Pruner layer-wise smoke after the environment fix.

Key output:

- Parameters before pruning: 1,032,272
- Parameters after pruning: 1,028,144
- Parameter ratio after pruning: 99.6001%
- PPL after pruning:
  - wikitext2: 32299.676704110814
  - ptb: 32553.006208238
- Reported memory requirement: 10.1196 MiB
- Duration: 34.275 seconds

This run confirms the tiny LLaMA LLM-Pruner R1 smoke is reproducible on the local RTX 5060 WSL environment.
