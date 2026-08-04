# Flab Qwen2 Prune Call Graph

- `Qwen2ForCausalLM.prune(config, stage)` calls `init_prune_zs(config, stage)`.
- `init_prune_zs` creates masks and retained indices for hidden, attention heads, KV heads, and FFN intermediate channels.
- Top-level prune calls `self.model.prune(self.zs)`.
- `Qwen2Model.prune(zs)` prunes embeddings, layers, and final norm.
- `Qwen2DecoderLayer.prune(zs, layer_num)` forwards `zs` to attention, MLP, and RMSNorm modules.
- `Qwen2MLP.prune(zs, layer_num)` slices `gate_proj`, `up_proj`, and `down_proj` using `intermediate_indexes[layer_num - 1]`.
- `Qwen2SdpaAttention.prune(zs, layer_num)` also requires `head_masks` and `kv_head_masks`.
- Top-level prune slices `lm_head` by `hidden_index` and updates config fields; the experimental adapter keeps hidden/head/KV full and guides only `intermediate_indexes`.
