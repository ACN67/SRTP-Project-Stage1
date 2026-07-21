# SliceGPT Qwen Adapter Audit

Status: success

Method: SliceGPT
Owner: 常珂舒
Stage: R2 Qwen adaptation audit
Run ID: slicegpt_qwen_adapter_audit_20260722_012640

Conclusion:

- SliceGPT imports successfully in the local `.venv-slicegpt` environment.
- The official adapter directory currently contains adapters for LLaMA, OPT, Phi-2, and Phi-3.
- No Qwen/Qwen2 adapter was found.
- Qwen2.5-Coder-1.5B and Qwen2.5-Coder-3B both report `model_type: qwen2` and `architectures: Qwen2ForCausalLM`.
- Therefore, SliceGPT cannot directly run Qwen2.5-Coder pruning through the current official adapter registry.

Next step:

- Implement a Qwen2 adapter for SliceGPT, likely using the LLaMA adapter as the closest reference, or mark SliceGPT Qwen R2 as adapter-needed if development cost is deferred.
