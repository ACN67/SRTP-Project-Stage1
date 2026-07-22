# LLM-Pruner CodeLlama-7B Light Probe

Status: success

Method: LLM-Pruner
Owner: 常珂舒
Model: codellama/CodeLlama-7b-hf
Run ID: llmpruner_codellama7b_light_probe_20260722_200734

Purpose:

- Verify CodeLlama-7B config/tokenizer compatibility with the LLM-Pruner LLaMA-bound path.
- Avoid loading full model weights or constructing the full 7B model.

Key output:

- `model_type`: `llama`
- `architectures`: `LlamaForCausalLM`
- `num_hidden_layers`: 32
- `hidden_size`: 4096
- `num_attention_heads`: 32
- `num_key_value_heads`: 32
- `vocab_size`: 32016
- LLM-Pruner custom LLaMA model, attention, MLP, RMSNorm, and pruner modules imported successfully.

Note:

- Hugging Face warned that the checkpoint tokenizer class is `CodeLlamaTokenizer` while the script explicitly uses `LlamaTokenizer`.
- The vocabulary size still matches, so this is a compatibility warning to track before heavy pruning, not a blocker for this light probe.
