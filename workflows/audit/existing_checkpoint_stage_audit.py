#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from workflows.evaluate.completion import normalize_completion
from methods.llm_pruner.qwen_prune import load_llmpruner_qwen_model
from methods.slicegpt.qwen_prune import load_sliced_qwen_model

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
FLAB_ROOT = ROOT / "results/evidence/r4_half/flabpruner_qwen25c15b_official_keep80_20260730_015031"
LLM_ROOT = ROOT / "results/evidence/r4_half/llmpruner_qwen25c15b_official_keep80_gpu_20260730_105340"
SLICE_ROOT = ROOT / "results/evidence/r4_half/slicegpt_qwen25c15b_official_keep80_gpu_20260731_011001"
LCB_PROMPTS: dict[str, str] = {}
LCB_EXTRACTOR: Callable[[str], str]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def add_lcb_import_paths() -> Path:
    for candidate in [ROOT / "third_party" / "LiveCodeBench", ROOT / "third_party" / "livecodebench"]:
        if (candidate / "lcb_runner").is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise FileNotFoundError("LiveCodeBench checkout not found")


def load_lcb_problems(release: str, config_name: str):
    lcb_root = add_lcb_import_paths()
    with pushd(lcb_root):
        from lcb_runner.benchmarks.code_generation import CodeGenerationProblem, load_code_generation_dataset
    if not config_name:
        with pushd(lcb_root):
            return load_code_generation_dataset(release_version=release)
    dataset = load_dataset("livecodebench/code_generation_lite", config_name, split="test", version_tag=release)
    return [CodeGenerationProblem(**row) for row in dataset]


def build_lcb_prompt_and_extractor(tasks: list[dict], release: str, config_name: str, lm_style_name: str):
    lcb_root = add_lcb_import_paths()
    with pushd(lcb_root):
        from lcb_runner.lm_styles import LMStyle
        from lcb_runner.prompts.code_generation import format_prompt_generation
        from lcb_runner.utils.extraction_utils import extract_code
    lm_style = LMStyle[lm_style_name]
    requested = {row["task_id"] for row in tasks if row["benchmark"] == "livecodebench"}
    problems = {p.question_id: p for p in load_lcb_problems(release, config_name) if p.question_id in requested}
    prompts = {}
    for task_id, problem in problems.items():
        with pushd(lcb_root):
            prompts[task_id] = format_prompt_generation(problem, lm_style)
    def extractor(raw_completion: str) -> str:
        code = extract_code(raw_completion, lm_style)
        return code.rstrip() + "\n" if code else "\n"
    return prompts, extractor


def fixed_smoke_rows(path: Path) -> list[dict]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    specs = [
        ("humaneval", ROOT / "data/benchmarks/r4_half/humaneval/eval.jsonl", 8, "humaneval_official"),
        ("mbpp_evalplus", ROOT / "data/benchmarks/r4_half/mbpp_evalplus/eval.jsonl", 8, "mbpp_evalplus_official"),
        ("livecodebench", ROOT / "data/benchmarks/r4_half/livecodebench/eval.jsonl", 4, "livecodebench_official"),
    ]
    rows=[]
    for bench, split, limit, prompt_mode in specs:
        for row in read_jsonl(split)[:limit]:
            item=dict(row)
            item["benchmark"] = bench
            item["prompt_mode"] = prompt_mode
            item["sample_id"] = f"{bench}:{item['task_id']}"
            item["split_path"] = str(split.relative_to(ROOT))
            rows.append(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


def dtype_from_name(name: str):
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def load_stage(stage: dict, dtype: torch.dtype, device: str, local_files_only: bool):
    kind = stage.get("kind", "hf")
    if kind == "llmpruner":
        model, tokenizer = load_llmpruner_qwen_model(BASE_MODEL, stage["model"], dtype, device, local_files_only)
    elif kind == "slicegpt":
        model, tokenizer = load_sliced_qwen_model(BASE_MODEL, stage["model"], stage["sparsity"], stage.get("round_interval", 128), dtype, device, local_files_only)
    else:
        tokenizer = AutoTokenizer.from_pretrained(stage["model"], trust_remote_code=True, local_files_only=local_files_only)
        model = AutoModelForCausalLM.from_pretrained(
            stage["model"], trust_remote_code=True, torch_dtype=dtype, local_files_only=local_files_only, device_map=device
        )
    if stage.get("adapter"):
        model = PeftModel.from_pretrained(model, stage["adapter"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def build_prompt(item: dict) -> tuple[str, Callable[[str], str], int]:
    if item["benchmark"] == "livecodebench":
        return LCB_PROMPTS[item["task_id"]], LCB_EXTRACTOR, 1024
    return item["prompt"], normalize_completion, 512


def repetition_ratio(text: str) -> float:
    toks = re.findall(r"\w+|\S", text)
    if not toks:
        return 0.0
    ngrams = [tuple(toks[i:i+4]) for i in range(max(0, len(toks)-3))]
    if not ngrams:
        return 0.0
    return 1.0 - (len(set(ngrams)) / len(ngrams))


def parse_success(code: str) -> bool:
    try:
        ast.parse(code or "")
        return True
    except SyntaxError:
        return False


def failure_category(raw: str, extracted: str, hit_max: bool, parse_ok: bool, passed: bool | None) -> str:
    if passed:
        return "K_pass"
    if not raw.strip():
        return "A_empty"
    if hit_max:
        return "F_hit_max_new_tokens"
    if not extracted.strip():
        return "G_code_extraction_failed"
    if "```" in raw and raw.count("```") % 2:
        return "D_unclosed_code_fence"
    if "def " not in extracted and re.search(r"\b(function|snippet|returns|takes|code)\b", raw, re.I):
        return "C_natural_language_only"
    if repetition_ratio(raw) > 0.35:
        return "E_repetition"
    if not parse_ok:
        return "H_syntax_error"
    if passed is False:
        return "J_logic_or_runtime_fail"
    return "J_unknown_unscored"


def generate_stage(stage: dict, tasks: list[dict], out_dir: Path, dtype: torch.dtype, device: str, local_files_only: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_stage(stage, dtype, device, local_files_only)
    rows=[]
    samples_by_bench={"humaneval":[], "mbpp_evalplus":[], "livecodebench":[]}
    generations_by_bench={"humaneval":[], "mbpp_evalplus":[], "livecodebench":[]}
    input_device = next(model.parameters()).device
    for idx,item in enumerate(tasks,1):
        prompt, extractor, max_new = build_prompt(item)
        inputs = tokenizer(prompt, return_tensors="pt").to(input_device)
        started=time.time()
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen_seconds=time.time()-started
        input_n=inputs["input_ids"].shape[-1]
        gen_ids=output_ids[0,input_n:].detach().cpu().tolist()
        raw=tokenizer.decode(gen_ids, skip_special_tokens=True)
        extracted=extractor(raw)
        eos_id=tokenizer.eos_token_id
        eos_emitted=bool(eos_id is not None and eos_id in gen_ids)
        hit_max=len(gen_ids)>=max_new
        stop_reason="max_new_tokens" if hit_max else ("eos" if eos_emitted else "stopped_without_eos")
        parse_ok=parse_success(extracted)
        row={"method": stage["method"], "stage": stage["stage"], "sample_id": item["sample_id"], "task_id": item["task_id"], "benchmark": item["benchmark"], "prompt_mode": item["prompt_mode"], "prompt": item["prompt"], "model_prompt": prompt, "raw_completion": raw, "extracted_code": extracted, "generated_token_ids": gen_ids, "generated_token_count": len(gen_ids), "hit_max_new_tokens": hit_max, "eos_emitted": eos_emitted, "stop_reason": stop_reason, "repetition_ratio": repetition_ratio(raw), "code_extraction_success": bool(extracted.strip()), "parse_success": parse_ok, "gen_seconds": gen_seconds}
        rows.append(row)
        samples_by_bench[item["benchmark"]].append({"task_id": item["task_id"], "solution": item["prompt"] + extracted})
        generations_by_bench[item["benchmark"]].append({"task_id": item["task_id"], "prompt": item["prompt"], "model_prompt": prompt, "raw_completion": raw, "completion": extracted, "generated": item["prompt"] + raw, "generated_tokens": len(gen_ids), "max_new_tokens": max_new, "hit_max_new_tokens": hit_max})
        print(json.dumps({"event":"audit_generated", "stage":stage["stage"], "index":idx, "total":len(tasks), "sample_id":item["sample_id"], "tokens":len(gen_ids), "hit_max":hit_max}, ensure_ascii=False), flush=True)
    write_jsonl(out_dir/"audit_generations.jsonl", rows)
    for bench in samples_by_bench:
        bdir=out_dir/bench
        write_jsonl(bdir/"samples.jsonl", samples_by_bench[bench])
        write_jsonl(bdir/"generations.jsonl", generations_by_bench[bench])
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def score_stage(stage_dir: Path, fixed_split_dir: Path):
    details={}
    commands=[
        [sys.executable, "workflows/evaluate/score_humaneval.py", "--split", str(fixed_split_dir/"humaneval.jsonl"), "--samples", str(stage_dir/"humaneval/samples.jsonl"), "--out-dir", str(stage_dir/"humaneval/score"), "--base-only"],
        [sys.executable, "workflows/evaluate/score_mbpp.py", "--split", str(fixed_split_dir/"mbpp_evalplus.jsonl"), "--samples", str(stage_dir/"mbpp_evalplus/samples.jsonl"), "--out-dir", str(stage_dir/"mbpp_evalplus/score"), "--base-only"],
        [sys.executable, "workflows/evaluate/score_livecodebench.py", "--split", str(fixed_split_dir/"livecodebench.jsonl"), "--generations", str(stage_dir/"livecodebench/generations.jsonl"), "--out-dir", str(stage_dir/"livecodebench/score"), "--lcb-release", "release_v1", "--lcb-config", "release_latest", "--timeout", "6", "--num-process-evaluate", "2"],
    ]
    for cmd in commands:
        subprocess.run(cmd, cwd=ROOT, check=True)
    for bench in ["humaneval","mbpp_evalplus"]:
        for row in read_jsonl(stage_dir/bench/"score/score_details.jsonl"):
            details[(bench,row["task_id"])] = {"passed": bool(row.get("base_pass")), "runtime_success": row.get("base_status") == "pass", "score_status": row.get("base_status")}
    for row in read_jsonl(stage_dir/"livecodebench/score/score_details.jsonl"):
        details[("livecodebench",row["task_id"])] = {"passed": bool(row.get("pass")), "runtime_success": bool(row.get("pass")), "score_status": "pass" if row.get("pass") else "fail"}
    return details


def write_fixed_splits(tasks: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for bench in ["humaneval","mbpp_evalplus","livecodebench"]:
        write_jsonl(out_dir/f"{bench}.jsonl", [row for row in tasks if row["benchmark"]==bench])


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT/"results/evidence/existing_checkpoint_stage_audit")
    parser.add_argument("--fixed-file", type=Path, default=ROOT/"data/audit/fixed_smoke_20.json")
    parser.add_argument("--dtype", choices=["fp16","bf16","fp32"], default="fp16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args=parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tasks=fixed_smoke_rows(args.fixed_file)
    global LCB_PROMPTS, LCB_EXTRACTOR
    LCB_PROMPTS, LCB_EXTRACTOR = build_lcb_prompt_and_extractor(tasks, "release_v1", "release_latest", "CodeQwenInstruct")
    split_dir=args.out_dir/"fixed_splits"
    write_fixed_splits(tasks, split_dir)
    stages=[
        {"method":"FlabPruner", "stage":"F0_dense", "model":BASE_MODEL},
        {"method":"FlabPruner", "stage":"F1_pruned_no_lora", "model":str(FLAB_ROOT/"flabpruner_keep80_v2/pruned_model")},
        {"method":"FlabPruner", "stage":"F2_pruned_lora_adapter", "model":str(FLAB_ROOT/"flabpruner_keep80_v2/pruned_model"), "adapter":str(FLAB_ROOT/"flabpruner_keep80_lora_v2/lora_adapter")},
        {"method":"FlabPruner", "stage":"F3_merged", "model":str(FLAB_ROOT/"flabpruner_keep80_merged_v2")},
        {"method":"LLM-Pruner", "stage":"L0_dense", "model":BASE_MODEL},
        {"method":"LLM-Pruner", "stage":"L1_pruned_no_lora", "kind":"llmpruner", "model":str(LLM_ROOT/"llmpruner_keep80_v3/pruned_model")},
        {"method":"LLM-Pruner", "stage":"L2_pruned_lora_adapter", "kind":"llmpruner", "model":str(LLM_ROOT/"llmpruner_keep80_v3/pruned_model"), "adapter":str(LLM_ROOT/"llmpruner_keep80_lora_v3/lora_adapter")},
        {"method":"SliceGPT", "stage":"S0_dense", "model":BASE_MODEL},
        {"method":"SliceGPT", "stage":"S1_sliced_no_lora", "kind":"slicegpt", "model":str(SLICE_ROOT/"slicegpt_keep80/sliced_model"), "sparsity":0.45, "round_interval":128},
        {"method":"SliceGPT", "stage":"S2_sliced_lora_adapter", "kind":"slicegpt", "model":str(SLICE_ROOT/"slicegpt_keep80/sliced_model"), "sparsity":0.45, "round_interval":128, "adapter":str(SLICE_ROOT/"slicegpt_keep80_lora/lora_adapter")},
    ]
    all_rows=[]
    dtype=dtype_from_name(args.dtype)
    for stage in stages:
        sdir=args.out_dir/stage["stage"]
        if args.skip_existing and (sdir/"audit_generations.jsonl").exists():
            rows=read_jsonl(sdir/"audit_generations.jsonl")
        else:
            rows=generate_stage(stage, tasks, sdir, dtype, args.device, args.local_files_only)
        scored=score_stage(sdir, split_dir)
        for row in rows:
            detail=scored.get((row["benchmark"], row["task_id"]), {})
            row["passed"] = detail.get("passed")
            row["runtime_success"] = detail.get("runtime_success")
            row["score_status"] = detail.get("score_status")
            row["failure_category"] = failure_category(row["raw_completion"], row["extracted_code"], row["hit_max_new_tokens"], row["parse_success"], row["passed"])
            all_rows.append(row)
    csv_path=ROOT/"reports/existing_checkpoint_stage_metrics.csv"
    fields=["method","stage","sample_id","task_id","benchmark","passed","raw_completion","extracted_code","generated_token_ids","generated_token_count","hit_max_new_tokens","eos_emitted","stop_reason","repetition_ratio","code_extraction_success","parse_success","runtime_success","score_status","failure_category"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer=csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(all_rows)
    summary=[]
    for stage in stages:
        rows=[r for r in all_rows if r["stage"]==stage["stage"]]
        summary.append({"method":stage["method"], "stage":stage["stage"], "n":len(rows), "pass_rate":sum(1 for r in rows if r.get("passed"))/len(rows) if rows else None, "hit_max_rate":sum(1 for r in rows if r.get("hit_max_new_tokens"))/len(rows) if rows else None, "eos_rate":sum(1 for r in rows if r.get("eos_emitted"))/len(rows) if rows else None, "parse_success_rate":sum(1 for r in rows if r.get("parse_success"))/len(rows) if rows else None, "mean_generated_tokens":sum(r.get("generated_token_count",0) for r in rows)/len(rows) if rows else None})
    (ROOT/"reports/existing_checkpoint_stage_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"csv":str(csv_path), "summary":summary}, indent=2, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
