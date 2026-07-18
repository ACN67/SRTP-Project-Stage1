#!/usr/bin/env python3
"""Lightweight Qwen/Qwen2.5-Coder structure probe for Stage 1 R2 planning."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Qwen model config/tokenizer and expected module layout.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--output", default=None, help="Optional JSON output path. Defaults to $RUN_DIR/qwen_probe.json.")
    parser.add_argument("--skip-tokenizer", action="store_true", help="Only fetch config; do not fetch tokenizer files.")
    args = parser.parse_args()

    try:
        from transformers import AutoConfig, AutoTokenizer
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"failed to import transformers: {exc}", file=sys.stderr)
        return 2

    result: dict[str, object] = {
        "model_id": args.model,
        "probe_type": "config_tokenizer_only",
        "loads_weights": False,
    }

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    result["config"] = {
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "intermediate_size": getattr(config, "intermediate_size", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "torch_dtype": str(getattr(config, "torch_dtype", None)),
        "tie_word_embeddings": getattr(config, "tie_word_embeddings", None),
        "rope_theta": getattr(config, "rope_theta", None),
        "sliding_window": getattr(config, "sliding_window", None),
        "use_sliding_window": getattr(config, "use_sliding_window", None),
    }

    likely_modules = {
        "decoder_layers": "model.layers",
        "attention": "model.layers[i].self_attn",
        "attention_linears": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp_linears": ["gate_proj", "up_proj", "down_proj"],
        "norms": ["input_layernorm", "post_attention_layernorm", "model.norm"],
    }
    result["expected_qwen2_layout"] = likely_modules

    missing_layout_fields = []
    for field in ["num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads"]:
        if getattr(config, field, None) is None:
            missing_layout_fields.append(field)
    result["missing_layout_fields"] = missing_layout_fields

    if not args.skip_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        result["tokenizer"] = {
            "class": tokenizer.__class__.__name__,
            "vocab_size": getattr(tokenizer, "vocab_size", None),
            "model_max_length": getattr(tokenizer, "model_max_length", None),
            "pad_token": tokenizer.pad_token,
            "eos_token": tokenizer.eos_token,
            "bos_token": tokenizer.bos_token,
            "padding_side": tokenizer.padding_side,
        }

    result["adapter_implications"] = {
        "llm_pruner": "current hf_prune.py imports custom LLaMA classes and LlamaTokenizer; direct Qwen support likely requires a Qwen/AutoModel path or adapter patch",
        "slicegpt": "current adapters cover OPT/LLaMA/Phi/Phi3; Qwen likely requires a Qwen2 model adapter before R2 pruning",
        "flab_pruner": "requires separate entry-point inspection before judging Qwen compatibility",
        "laco": "upstream is notebook-only, so Qwen work first requires script extraction",
    }

    output = args.output or os.environ.get("RUN_DIR")
    if output:
        output_path = Path(output)
        if output_path.suffix != ".json":
            output_path = output_path / "qwen_probe.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output_path}")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
