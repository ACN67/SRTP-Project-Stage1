#!/usr/bin/env python3
"""Train a LoRA recovery adapter on a distillation JSONL dataset."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


class TextDataset(Dataset):
    def __init__(self, path: Path):
        self.rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    self.rows.append(json.loads(line))
        if not self.rows:
            raise ValueError(f"empty dataset: {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


def collate_batch(rows: list[dict], tokenizer, max_length: int) -> dict:
    texts = [row["text"] for row in rows]
    prompts = [row["prompt"] for row in rows]
    batch = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    labels = batch["input_ids"].clone()
    for idx, prompt in enumerate(prompts):
        prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
        prompt_len = min(len(prompt_ids), labels.shape[1])
        labels[idx, :prompt_len] = -100
    labels[batch["attention_mask"] == 0] = -100
    batch["labels"] = labels
    return batch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    use_cuda = torch.cuda.is_available() and args.device.startswith("cuda")
    device_map = args.device if use_cuda else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=dtype_map[args.dtype],
        device_map=device_map,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.train()
    model.print_trainable_parameters()

    dataset = TextDataset(args.train_file)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda rows: collate_batch(rows, tokenizer, args.max_length),
    )

    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = max(1, int(math.ceil(args.epochs * steps_per_epoch)))
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    global_step = 0
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    while global_step < total_steps:
        for batch_idx, batch in enumerate(loader, 1):
            if use_cuda:
                batch = {key: value.to(args.device) for key, value in batch.items()}
            output = model(**batch)
            loss = output.loss / args.grad_accum
            loss.backward()
            running_loss += float(loss.detach().cpu()) * args.grad_accum

            if batch_idx % args.grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                avg_loss = running_loss / global_step
                print(json.dumps({"step": global_step, "total_steps": total_steps, "avg_loss": avg_loss}, ensure_ascii=False), flush=True)
                if global_step >= total_steps:
                    break

    model.save_pretrained(args.out_dir / "lora_adapter")
    tokenizer.save_pretrained(args.out_dir / "lora_adapter")
    summary = {
        "status": "success",
        "base_model": args.base_model,
        "train_file": str(args.train_file),
        "samples": len(dataset),
        "output_adapter": str(args.out_dir / "lora_adapter"),
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "target_modules": [item.strip() for item in args.target_modules.split(",") if item.strip()],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_length": args.max_length,
        "lr": args.lr,
        "steps": global_step,
        "elapsed_seconds": time.time() - started,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
