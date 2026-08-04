# Qwen2.5-Coder 1.5B Config Probe

Status: success

Method: Qwen2.5-Coder adapter probe
Owner: shared
Model: Qwen/Qwen2.5-Coder-1.5B-Instruct
Reproduction level: R2 pre-check
Run ID: qwen25_coder_15b_config_probe_20260718_182144

Purpose:

- Confirm the target Qwen model can be reached from the local WSL environment.
- Record Qwen structure before attempting any pruning adaptation.
- Avoid downloading model weights during the first Qwen check.

Key output:

- Model type: qwen2
- Architecture: Qwen2ForCausalLM
- Hidden layers: 28
- Hidden size: 1536
- Intermediate size: 8960
- Attention heads: 12
- KV heads: 2
- Max position embeddings: 32768
- Config dtype: torch.bfloat16
- Tokenizer: Qwen2Tokenizer
- Tokenizer vocab size: 151643
- Model config vocab size: 151936
- Loads weights: false
- Duration: 14.468 seconds

Adapter implication:

Qwen2 uses LLaMA-like projection names (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) but has its own `Qwen2ForCausalLM` class and grouped-query attention. Methods that hard-code LLaMA classes still need an adapter or patch before Qwen R2 pruning.
