# LLM-Pruner Method Notes

Owner: 常珂舒
Group: structured_depth_width
Stage: 1
Status: R2 CodeLlama adaptation in progress

## Current Evidence

R1 official-style tiny LLaMA smoke has succeeded:

```text
results/raw/llmpruner_tiny_llama_layerwise_smoke_20260718_175708/
```

The active Stage 1 policy moves LLM-Pruner R2/R3 to CodeLlama-family models because the official pruning path is deeply tied to LLaMA classes.

## CodeLlama Light Probe

The CodeLlama-7B light probe succeeded without loading full weights:

```text
results/raw/llmpruner_codellama7b_light_probe_20260722_200734/
```

Key findings:

- `codellama/CodeLlama-7b-hf` resolves to `LlamaConfig`.
- `model_type` is `llama`.
- `architectures` is `LlamaForCausalLM`.
- LLM-Pruner custom LLaMA modules import successfully:
  - `LlamaForCausalLM`
  - `LlamaAttention`
  - `LlamaMLP`
  - `LlamaRMSNorm`
  - `hf_llama_pruner`

## Tokenizer Warning

The light probe produced a Hugging Face warning:

```text
The tokenizer class you load from this checkpoint is 'CodeLlamaTokenizer'.
The class this function is called from is 'LlamaTokenizer'.
```

This warning should be tracked before heavy pruning. The vocabulary size matches, so it is not currently treated as a blocker for R2 feasibility.

## Next Step

For R2, attempt a conservative CodeLlama-7B pruning run:

- keep the model unquantized;
- use fp16;
- use layer-wise pruning first because it is structurally simpler than channel/block pruning;
- avoid baseline generation and PPL in the first heavy attempt;
- use minimal sequence length and a very small layer target;
- record termination or OOM as resource evidence if the local machine cannot complete it.
