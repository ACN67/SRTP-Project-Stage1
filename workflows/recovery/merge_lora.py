#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model and save a standalone model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import psutil
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    use_cuda = torch.cuda.is_available() and args.device.startswith("cuda")
    device = torch.device(args.device if use_cuda else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=dtype_map[args.dtype],
    )
    model.to(device)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    merged = model.merge_and_unload()
    merged.save_pretrained(args.out_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.out_dir)

    param_count = sum(param.numel() for param in merged.parameters())
    summary = {
        "status": "success",
        "base_model": args.base_model,
        "adapter": args.adapter,
        "merged_model": str(args.out_dir),
        "dtype": args.dtype,
        "device": str(device),
        "param_count": param_count,
        "process_rss_mb_after": psutil.Process().memory_info().rss / 1024**2,
        "elapsed_seconds": time.time() - started,
        "torch": torch.__version__,
    }
    (args.out_dir / "merge_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
