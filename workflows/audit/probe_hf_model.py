#!/usr/bin/env python3
"""Lightweight Hugging Face model config/tokenizer probe without loading weights."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe HF model config/tokenizer for Stage 1 model selection.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-short-name", default=None)
    parser.add_argument("--output", default=None, help="Optional JSON output path. Defaults to $RUN_DIR/model_probe.json.")
    parser.add_argument("--skip-tokenizer", action="store_true")
    args = parser.parse_args()

    try:
        from transformers import AutoConfig, AutoTokenizer
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"failed to import transformers: {exc}", file=sys.stderr)
        return 2

    result: dict[str, object] = {
        "model_short_name": args.model_short_name,
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
    }

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

    output = args.output or os.environ.get("RUN_DIR")
    if output:
        output_path = Path(output)
        if output_path.suffix != ".json":
            output_path = output_path / "model_probe.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output_path}")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
