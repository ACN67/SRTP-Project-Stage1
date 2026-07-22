# SliceGPT Method Notes

Owner: 常珂舒
Group: structured_depth_width
Stage: 1
Status: R2 CodeLlama adaptation in progress

## Current Evidence

R1 official smoke has succeeded on `facebook/opt-125m`:

```text
results/raw/slicegpt_opt125m_smoke_20260718_173646/
```

The active Stage 1 policy moves SliceGPT R2/R3 to CodeLlama-family models because the upstream repository includes a LLaMA adapter but no Qwen2 adapter.

## CodeLlama Adapter Finding

`codellama/CodeLlama-7b-hf` has Hugging Face config:

```text
model_type: llama
architectures: LlamaForCausalLM
num_hidden_layers: 32
hidden_size: 4096
num_attention_heads: 32
```

This structure is compatible with the SliceGPT LLaMA adapter in principle.

However, the upstream SliceGPT LLaMA adapter currently accepts only model names starting with:

```text
meta-llama/Llama-2
meta-llama/Meta-Llama-3
```

The project-side patch below extends this allowlist to `codellama/CodeLlama` without changing the SliceGPT algorithm:

```text
patches/slicegpt/0001-allow-codellama-llama-adapter.patch
```

## Local Resource Finding

After applying the allowlist patch, a CodeLlama-7B `uninitialized=True` adapter probe was still terminated by the local system. This indicates that the upstream uninitialized path avoids weight initialization but still constructs a full 7B model object, which can exceed the current WSL/CPU memory budget.

This is a resource-shape blocker for the light adapter probe, not evidence that CodeLlama is structurally unsupported.

## Next Step

For R2, attempt the actual SliceGPT CodeLlama run with conservative settings and offload-oriented execution where possible:

- fp16 or bf16, not fp32;
- very small calibration sample count;
- short calibration/eval sequence length;
- CPU/disk offload if supported by the route;
- record OOM or termination as resource evidence if the run cannot complete locally.
