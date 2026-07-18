# Qwen2.5-Coder 3B Config Probe

Status: success

Method: Qwen2.5-Coder adapter probe
Owner: shared
Model: Qwen/Qwen2.5-Coder-3B-Instruct
Reproduction level: Q3 pre-check
Run ID: qwen25_coder_3b_config_probe_20260718_182954

Purpose:

- Confirm the Stage 1 Qwen 3B comparison target is reachable from the local WSL environment.
- Record the Qwen 3B structure before benchmark-guided pruning experiments.
- Avoid downloading model weights during the first 3B check.

Key output:

- Model type: qwen2
- Architecture: Qwen2ForCausalLM
- Hidden layers: 36
- Hidden size: 2048
- Intermediate size: 11008
- Attention heads: 16
- KV heads: 2
- Max position embeddings: 32768
- Config dtype: torch.bfloat16
- Tokenizer: Qwen2Tokenizer
- Tokenizer vocab size: 151643
- Model config vocab size: 151936
- Loads weights: false
- Duration: 11.611 seconds

Comparison role:

This model is the Stage 1 Qwen 3B target for representative-method pruning. Formal comparison should keep the original Qwen 1.5B, original Qwen 3B, and pruned Qwen 3B results separate, using the same benchmark eval protocol.
