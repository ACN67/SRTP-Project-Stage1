# SliceGPT CodeLlama-7B Adapter Probe

Status: failed

Method: SliceGPT
Owner: 常珂舒
Model: codellama/CodeLlama-7b-hf
Run ID: slicegpt_codellama7b_adapter_probe_20260722_200041

Purpose:

- Re-run the SliceGPT CodeLlama adapter probe after adding `codellama/CodeLlama` to the local LLaMA adapter allowlist.
- Avoid loading full model weights by using `uninitialized=True`.

Result:

- The process was terminated by the local system before writing `summary.json`.
- The log shows Hugging Face metadata/cache access warnings, but no model-weight download progress.
- Likely category: local memory/resource termination.

Interpretation:

- The previous failure was caused by SliceGPT's model-name allowlist.
- After the allowlist patch, the probe progressed further but constructing the uninitialized 7B model object still exceeded local resources.
- This does not disprove CodeLlama structural compatibility; the committed `codellama7b_config_probe` confirms `model_type=llama` and `architectures=LlamaForCausalLM`.

Next step:

- Treat this as R2 resource evidence.
- Move to a conservative full-run attempt only when prepared for a long/offload run, or record local 7B infeasibility if it is terminated again.
