# Flab-Pruner Qwen2.5-Coder-3B R4 LoRA Recovery

Status: pending_merge_eval

This run contains the LoRA recovery materials for the two formal Flab-Pruner R4 keep-80 models:

- `default_keep80_lora/`: LoRA recovery adapter trained on the Qwen3B guide-half distillation set.
- `benchguided_keep80_lora/`: LoRA recovery adapter trained on the same guide-half distillation set.
- `qwen25c3b_distill/`: teacher-generated guide-half distillation data.

Dynamic adapter evaluation is not used as the formal R4 route. The formal recovered models should be produced by merging each LoRA adapter into its corresponding pruned base model, then evaluating the merged standalone models on the fixed eval halves for HumanEval, MBPP, and LiveCodeBench.
