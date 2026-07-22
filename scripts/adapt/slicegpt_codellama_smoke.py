#!/usr/bin/env python
"""Run a minimal SliceGPT CodeLlama smoke using CPU weights and GPU layer work."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from slicegpt import data_utils, layernorm_fusion, rotate, utils
from slicegpt.adapters.llama_adapter import LlamaModelAdapter
from slicegpt.config import config
from slicegpt.slicing_scheduler import ConstSlicingScheduler


def dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(name)


def count_params(model: torch.nn.Module) -> int:
    return sum(int(param.nelement()) for param in model.parameters())


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_completion(prompt: str, generated: str) -> str:
    return generated[len(prompt) :] if generated.startswith(prompt) else generated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cal-dataset", default="wikitext2", choices=["wikitext2", "ptb", "c4", "alpaca"])
    parser.add_argument("--cal-nsamples", type=int, default=1)
    parser.add_argument("--cal-batch-size", type=int, default=1)
    parser.add_argument("--cal-max-seqlen", type=int, default=32)
    parser.add_argument("--sparsity", type=float, default=0.01)
    parser.add_argument("--round-interval", type=int, default=8)
    parser.add_argument("--final-orientation", choices=["random", "pca"], default="random")
    parser.add_argument("--generate-split", type=Path)
    parser.add_argument("--generate-limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    started = time.time()
    proc = psutil.Process()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype)

    logging.info("Loading tokenizer from %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, local_files_only=True)

    logging.info("Loading CodeLlama weights on CPU")
    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    model.config.torch_dtype = dtype

    model_adapter = LlamaModelAdapter(model)
    model_adapter.use_cache = False
    model_adapter.post_init(tokenizer)

    dataset = data_utils.get_dataset(args.cal_dataset)
    train_loader = data_utils.prepare_dataloader(
        dataset=dataset["train"],
        tokenizer=tokenizer,
        max_seqlen=args.cal_max_seqlen,
        batch_size=args.cal_batch_size,
        nsamples=args.cal_nsamples,
        varied_seqlen=False,
        seed=42,
    )

    logging.info("Replacing and fusing SliceGPT layers")
    layernorm_fusion.replace_layers(model_adapter)
    layernorm_fusion.fuse_modules(model_adapter)

    params_before = count_params(model)
    new_embedding_dimension = int((1 - args.sparsity) * model_adapter.hidden_size)
    new_embedding_dimension -= new_embedding_dimension % args.round_interval
    scheduler = ConstSlicingScheduler(new_embedding_dimension)

    logging.info(
        "Running rotate_and_slice: hidden_size=%s new_embedding_dimension=%s",
        model_adapter.hidden_size,
        new_embedding_dimension,
    )
    rotate.rotate_and_slice(model_adapter, train_loader, scheduler, final_orientation=args.final_orientation)

    params_after = count_params(model)

    generation_summary = None
    if args.generate_split:
        tasks = read_jsonl(args.generate_split)
        if args.generate_limit:
            tasks = tasks[: args.generate_limit]

        samples_path = args.out_dir / "samples.jsonl"
        generations_path = args.out_dir / "generations.jsonl"

        model.to(config.device)
        with samples_path.open("w", encoding="utf-8") as sf, generations_path.open("w", encoding="utf-8") as gf:
            for item in tasks:
                task_id = item["task_id"]
                prompt = item["prompt"]
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {key: value.to(config.device) for key, value in inputs.items()}

                gen_started = time.time()
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                gen_seconds = time.time() - gen_started

                generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                completion = extract_completion(prompt, generated)
                solution = prompt + completion

                sf.write(json.dumps({"task_id": task_id, "solution": solution}, ensure_ascii=False) + "\n")
                gf.write(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "prompt": prompt,
                            "generated": generated,
                            "completion": completion,
                            "gen_seconds": gen_seconds,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                print(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "completion_chars": len(completion),
                            "gen_seconds": round(gen_seconds, 3),
                        },
                        ensure_ascii=False,
                    )
                )

        model.cpu()
        utils.cleanup_memory()
        generation_summary = {
            "split": str(args.generate_split),
            "task_count": len(tasks),
            "samples": str(samples_path),
            "generations": str(generations_path),
            "max_new_tokens": args.max_new_tokens,
        }

    summary = {
        "status": "success",
        "method": "SliceGPT",
        "model_path": str(args.model_path),
        "wrapper": "scripts/adapt/slicegpt_codellama_smoke.py",
        "loaded_weights": True,
        "dtype": args.dtype,
        "device": str(config.device),
        "cal_dataset": args.cal_dataset,
        "cal_nsamples": args.cal_nsamples,
        "cal_batch_size": args.cal_batch_size,
        "cal_max_seqlen": args.cal_max_seqlen,
        "sparsity_requested": args.sparsity,
        "new_embedding_dimension": new_embedding_dimension,
        "parameters_before": params_before,
        "parameters_after": params_after,
        "parameter_keep_ratio": params_after / params_before,
        "parameter_reduction_rate": 1 - params_after / params_before,
        "generation": generation_summary,
        "process_rss_mb_after": proc.memory_info().rss / 1024**2,
        "elapsed_seconds": time.time() - started,
        "torch": torch.__version__,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
