#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from workflows.evaluate.completion import normalize_completion

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


@contextmanager
def pushd(path: Path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def add_lcb_import_paths() -> Path:
    candidates = [
        ROOT / "third_party" / "LiveCodeBench",
        ROOT / "third_party" / "livecodebench",
    ]
    for candidate in candidates:
        if (candidate / "lcb_runner").is_dir():
            path = str(candidate)
            if path not in sys.path:
                sys.path.insert(0, path)
            return candidate
    raise FileNotFoundError("Could not find local LiveCodeBench checkout under third_party/.")


def load_lcb_problems(release: str, config_name: str):
    lcb_root = add_lcb_import_paths()
    with pushd(lcb_root):
        from lcb_runner.benchmarks.code_generation import (
            CodeGenerationProblem,
            load_code_generation_dataset,
        )

    if not config_name:
        with pushd(lcb_root):
            return load_code_generation_dataset(release_version=release)

    dataset = load_dataset(
        "livecodebench/code_generation_lite",
        config_name,
        split="test",
        version_tag=release,
    )
    return [CodeGenerationProblem(**row) for row in dataset]


def build_lcb_prompt_and_extractor(
    tasks: list[dict],
    release: str,
    config_name: str,
    lm_style_name: str,
) -> tuple[dict[str, str], Callable[[str], str]]:
    lcb_root = add_lcb_import_paths()
    with pushd(lcb_root):
        from lcb_runner.lm_styles import LMStyle
        from lcb_runner.prompts.code_generation import format_prompt_generation
        from lcb_runner.utils.extraction_utils import extract_code

    try:
        lm_style = LMStyle[lm_style_name]
    except KeyError as exc:
        valid = ", ".join(style.name for style in LMStyle)
        raise ValueError(f"Unknown LiveCodeBench LMStyle {lm_style_name!r}. Valid values: {valid}") from exc

    requested_ids = {item["task_id"] for item in tasks}
    problems = {
        problem.question_id: problem
        for problem in load_lcb_problems(release, config_name)
        if problem.question_id in requested_ids
    }
    missing_ids = sorted(requested_ids - set(problems))
    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        raise KeyError(f"{len(missing_ids)} split task_ids were not found in LiveCodeBench {release}: {preview}")

    prompts: dict[str, str] = {}
    for task_id, problem in problems.items():
        with pushd(lcb_root):
            formatted = format_prompt_generation(problem, lm_style)
        if not isinstance(formatted, str):
            raise TypeError(
                f"LiveCodeBench LMStyle {lm_style_name} produced chat messages, "
                "but this local generator expects a single text prompt."
            )
        prompts[task_id] = formatted

    def extractor(raw_completion: str) -> str:
        extracted = extract_code(raw_completion, lm_style)
        return extracted.rstrip() + "\n" if extracted else "\n"

    return prompts, extractor


def build_model_prompt(item: dict, prompt_mode: str, lcb_prompts: dict[str, str] | None) -> str:
    prompt = item["prompt"]
    if prompt_mode in {"raw", "humaneval_official", "mbpp_evalplus_official"}:
        return prompt
    if prompt_mode == "livecodebench_official":
        if lcb_prompts is None:
            raise ValueError("LiveCodeBench official prompts were not initialized.")
        return lcb_prompts[item["task_id"]]
    raise ValueError(f"Unknown prompt mode: {prompt_mode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--load-mode", choices=["direct", "device_map"], default="direct")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-memory-json", default="")
    parser.add_argument("--offload-folder", type=Path)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--llm-int8-enable-fp32-cpu-offload", action="store_true")
    parser.add_argument("--llmpruner-base-model", default="", help="Base model name/path for loading an official LLM-Pruner Qwen non-uniform artifact.")
    parser.add_argument("--slicegpt-base-model", default="", help="Base model name/path for loading an official SliceGPT Qwen sliced artifact.")
    parser.add_argument("--slicegpt-sparsity", type=float, default=0.0)
    parser.add_argument("--slicegpt-round-interval", type=int, default=128)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model/tokenizer files only from the local Hugging Face cache.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=[
            "raw",
            "humaneval_official",
            "mbpp_evalplus_official",
            "livecodebench_official",
        ],
        default="raw",
        help=(
            "Prompt format used for generation. HumanEval/MBPP official modes are aliases for "
            "their EvalPlus raw prompts. livecodebench_official uses lcb_runner's official formatter."
        ),
    )
    parser.add_argument("--lcb-release", default="release_v1")
    parser.add_argument("--lcb-config", default="release_latest", help="HF dataset config name for LiveCodeBench.")
    parser.add_argument(
        "--lcb-lm-style",
        default="CodeQwenInstruct",
        help="LiveCodeBench LMStyle name used with --prompt-mode livecodebench_official.",
    )
    args = parser.parse_args()
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("Choose only one of --load-in-8bit or --load-in-4bit.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    use_cuda = torch.cuda.is_available() and args.device.startswith("cuda")
    device = torch.device(args.device if use_cuda else "cpu")

    tasks = list(read_jsonl(args.split))
    if args.limit:
        tasks = tasks[: args.limit]

    started = time.time()
    lcb_prompts = None
    completion_extractor: Callable[[str], str] = normalize_completion
    if args.prompt_mode == "livecodebench_official":
        lcb_prompts, completion_extractor = build_lcb_prompt_and_extractor(
            tasks,
            args.lcb_release,
            args.lcb_config,
            args.lcb_lm_style,
        )

    if args.llmpruner_base_model:
        from methods.llm_pruner.qwen_prune import load_llmpruner_qwen_model

        if args.load_in_8bit or args.load_in_4bit:
            raise ValueError("LLM-Pruner custom artifact loading does not support --load-in-8bit/--load-in-4bit.")
        if args.load_mode != "direct":
            raise ValueError("LLM-Pruner custom artifact loading currently requires --load-mode direct.")
        model, tokenizer = load_llmpruner_qwen_model(
            args.llmpruner_base_model,
            args.model,
            dtype_map[args.dtype],
            args.device,
            args.local_files_only,
        )
    elif args.slicegpt_base_model:
        from methods.slicegpt.qwen_prune import load_sliced_qwen_model

        if args.load_in_8bit or args.load_in_4bit:
            raise ValueError("SliceGPT artifact loading does not support --load-in-8bit/--load-in-4bit.")
        if args.load_mode != "direct":
            raise ValueError("SliceGPT artifact loading currently requires --load-mode direct.")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model, tokenizer = load_sliced_qwen_model(
            args.slicegpt_base_model,
            args.model,
            args.slicegpt_sparsity,
            args.slicegpt_round_interval,
            dtype_map[args.dtype],
            args.device,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype_map[args.dtype],
            "local_files_only": args.local_files_only,
        }
        if args.load_in_8bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=args.llm_int8_enable_fp32_cpu_offload,
            )
        elif args.load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype_map[args.dtype],
            )
        if args.load_mode == "device_map":
            load_kwargs["device_map"] = args.device_map
            if args.max_memory_json:
                max_memory = json.loads(args.max_memory_json)
                load_kwargs["max_memory"] = {
                    int(key) if isinstance(key, str) and key.isdigit() else key: value
                    for key, value in max_memory.items()
                }
            if args.offload_folder:
                args.offload_folder.mkdir(parents=True, exist_ok=True)
                load_kwargs["offload_folder"] = str(args.offload_folder)
        else:
            load_kwargs["device_map"] = args.device if use_cuda else "cpu"
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    def input_device() -> torch.device:
        if args.load_mode == "device_map":
            hf_device_map = getattr(model, "hf_device_map", {}) or {}
            for value in hf_device_map.values():
                if isinstance(value, int):
                    return torch.device(f"cuda:{value}")
                if isinstance(value, str) and value.startswith("cuda"):
                    return torch.device(value)
            return next(model.parameters()).device
        return device

    samples_path = args.out_dir / "samples.jsonl"
    generations_path = args.out_dir / "generations.jsonl"

    with samples_path.open("w", encoding="utf-8") as sf, generations_path.open("w", encoding="utf-8") as gf:
        total_tasks = len(tasks)
        for idx, item in enumerate(tasks, 1):
            task_id = item["task_id"]
            prompt = item["prompt"]
            model_prompt = build_model_prompt(item, args.prompt_mode, lcb_prompts)

            inputs = tokenizer(model_prompt, return_tensors="pt")
            if use_cuda:
                target_device = input_device()
                inputs = {k: v.to(target_device) for k, v in inputs.items()}

            gen_started = time.time()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen_seconds = time.time() - gen_started

            input_token_count = inputs["input_ids"].shape[-1]
            generated_ids = output_ids[0, input_token_count:]
            raw_completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated = prompt + raw_completion
            completion = completion_extractor(raw_completion)
            solution = prompt + completion

            sf.write(json.dumps({"task_id": task_id, "solution": solution}, ensure_ascii=False) + "\n")
            gf.write(json.dumps({
                "task_id": task_id,
                "prompt": prompt,
                "model_prompt": model_prompt,
                "prompt_mode": args.prompt_mode,
                "official_prompt": args.prompt_mode in {
                    "humaneval_official",
                    "mbpp_evalplus_official",
                    "livecodebench_official",
                },
                "lcb_release": args.lcb_release if args.prompt_mode == "livecodebench_official" else None,
                "lcb_config": args.lcb_config if args.prompt_mode == "livecodebench_official" else None,
                "lcb_lm_style": args.lcb_lm_style if args.prompt_mode == "livecodebench_official" else None,
                "generated": generated,
                "raw_completion": raw_completion,
                "completion": completion,
                "input_tokens": input_token_count,
                "generated_tokens": int(generated_ids.numel()),
                "max_new_tokens": args.max_new_tokens,
                "hit_max_new_tokens": int(generated_ids.numel()) >= args.max_new_tokens,
                "gen_seconds": gen_seconds,
            }, ensure_ascii=False) + "\n")
            sf.flush()
            gf.flush()

            print(json.dumps({
                "event": "generated",
                "index": idx,
                "total": total_tasks,
                "task_id": task_id,
                "prompt_mode": args.prompt_mode,
                "raw_completion_chars": len(raw_completion),
                "completion_chars": len(completion),
                "generated_tokens": int(generated_ids.numel()),
                "hit_max_new_tokens": int(generated_ids.numel()) >= args.max_new_tokens,
                "extract_warning": len(completion) <= 1 and len(raw_completion.strip()) > 1,
                "gen_seconds": round(gen_seconds, 3),
            }, ensure_ascii=False), flush=True)

    summary = {
        "status": "success",
        "model": args.model,
        "adapter": args.adapter,
        "split": str(args.split),
        "samples": str(samples_path),
        "generations": str(generations_path),
        "task_count": len(tasks),
        "prompt_mode": args.prompt_mode,
        "official_prompt": args.prompt_mode in {
            "humaneval_official",
            "mbpp_evalplus_official",
            "livecodebench_official",
        },
        "lcb_release": args.lcb_release if args.prompt_mode == "livecodebench_official" else None,
        "lcb_config": args.lcb_config if args.prompt_mode == "livecodebench_official" else None,
        "lcb_lm_style": args.lcb_lm_style if args.prompt_mode == "livecodebench_official" else None,
        "elapsed_seconds": time.time() - started,
        "torch": getattr(torch, "__version__"),
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "load_mode": args.load_mode,
        "load_in_8bit": args.load_in_8bit,
        "load_in_4bit": args.load_in_4bit,
        "llm_int8_enable_fp32_cpu_offload": args.llm_int8_enable_fp32_cpu_offload,
        "local_files_only": args.local_files_only,
        "slicegpt_base_model": args.slicegpt_base_model or None,
        "slicegpt_sparsity": args.slicegpt_sparsity if args.slicegpt_base_model else None,
        "slicegpt_round_interval": args.slicegpt_round_interval if args.slicegpt_base_model else None,
        "device_map": getattr(model, "hf_device_map", None),
        "max_memory": json.loads(args.max_memory_json) if args.max_memory_json else None,
        "offload_folder": str(args.offload_folder) if args.offload_folder else None,
    }
    (args.out_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
