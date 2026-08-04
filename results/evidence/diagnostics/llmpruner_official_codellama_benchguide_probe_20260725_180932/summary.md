# LLM-Pruner Official CodeLlama Benchmark-Guided Probe

Status: failed

Method: official LLM-Pruner MetaPruner
Model: codellama/CodeLlama-7b-hf
Mode: block-wise
Importance: Taylor param_first
Guide: HumanEval + MBPP + LiveCodeBench guide half, 1 sample each

The official MetaPruner path loaded CodeLlama-7B and entered benchmark-guided Taylor backward, but failed under local GPU+CPU offload with CUDA out-of-memory. This records why the local R4 used the layer-drop fallback route.
