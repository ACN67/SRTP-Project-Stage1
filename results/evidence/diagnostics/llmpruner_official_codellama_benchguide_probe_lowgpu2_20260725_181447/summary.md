# LLM-Pruner Official CodeLlama Low-GPU Probe

Status: failed

Method: official LLM-Pruner MetaPruner
Model: codellama/CodeLlama-7b-hf
Mode: block-wise
Importance: Taylor param_first
Guide: HumanEval + MBPP + LiveCodeBench guide half, 1 sample each

This conservative offload probe limited GPU memory to 2 GiB. It loaded the model but failed during DependencyGraph tracing when accelerate attempted to move offloaded modules to CUDA, producing `CUDA driver error: device not ready`. This supports treating official MetaPruner on CodeLlama-7B as locally blocked.
