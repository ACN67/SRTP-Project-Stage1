# LLM-Pruner CodeLlama-7B R4 Layer-Drop Fallback

Status: success

Method: LLM-Pruner-inspired layer-drop fallback
Official MetaPruner status: blocked locally by CUDA/offload failure
Model: codellama/CodeLlama-7b-hf
Keep target: about 80%
Drop layers: 6
Guide: HumanEval + MBPP + LiveCodeBench guide half
Eval: HumanEval + MBPP + LiveCodeBench eval half
Recovery: QLoRA train, LoRA merge to standalone fp16 model
LoRA: rank 8, alpha 16, dropout 0, epochs 1
Run root: results/raw/llmpruner_codellama7b_r4_layerdrop_keep80_full_20260725_182022

This run closes the local-machine LLM-Pruner R4 fallback route.
