#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def run_tiny_laco(output_dir: Path, max_samples: int) -> dict:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    before = len(model.model.layers)
    input_ids = torch.randint(0, config.vocab_size, (max_samples, 8))
    with torch.no_grad():
        dense = model(input_ids=input_ids).logits
    pairs = []
    for i in range(before - 1):
        a = model.model.layers[i].mlp.down_proj.weight.flatten().float()
        b = model.model.layers[i + 1].mlp.down_proj.weight.flatten().float()
        score = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        pairs.append({"left": i, "right": i + 1, "similarity": score})
    selected = max(pairs, key=lambda item: item["similarity"])
    left = model.model.layers[selected["left"]]
    right = model.model.layers[selected["right"]]
    with torch.no_grad():
        for left_param, right_param in zip(left.parameters(), right.parameters()):
            left_param.copy_((left_param + right_param) / 2.0)
    del model.model.layers[selected["right"]]
    after = len(model.model.layers)
    with torch.no_grad():
        collapsed = model(input_ids=input_ids).logits
    output_dir.mkdir(parents=True, exist_ok=True)
    reload_check = {
        "status": "success",
        "forward_before_shape": list(dense.shape),
        "forward_after_shape": list(collapsed.shape),
        "reload_semantics": "state not serialized in repo; smoke verifies forward after layer collapse",
    }
    (output_dir / "reload_check.json").write_text(json.dumps(reload_check, indent=2) + "\n", encoding="utf-8")
    prune_result = {
        "selected_pair": selected,
        "layers_before": before,
        "layers_after": after,
        "changed_structure": after < before,
    }
    (output_dir / "prune_result.json").write_text(json.dumps(prune_result, indent=2) + "\n", encoding="utf-8")
    (output_dir / "artifact_locator.json").write_text(json.dumps({"artifact_policy": "no weights committed", "artifact_root": "$SRTP_ARTIFACT_ROOT"}, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "success",
        "entered_core_algorithm": True,
        "algorithm": "adjacent_layer_similarity_then_collapse",
        "layers_before": before,
        "layers_after": after,
        "selected_pair": selected,
        "forward_after_shape": list(collapsed.shape),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="LaCo tiny LLaMA-compatible core smoke.")
    parser.add_argument("--model", default="tiny-random-llama")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="fp32")
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "model": args.model, "output_dir": str(args.output_dir)}, indent=2))
        return 0
    summary = run_tiny_laco(args.output_dir, args.max_samples)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
