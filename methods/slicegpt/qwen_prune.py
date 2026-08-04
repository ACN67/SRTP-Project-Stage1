#!/usr/bin/env python3
"""Run SliceGPT's official rotate-and-slice flow on Qwen2/Qwen2.5 models.

SliceGPT upstream requires a model adapter per architecture. This file provides
a local Qwen2 adapter while keeping the official SliceGPT operations:
layer replacement, layernorm fusion, PCA/random rotation, and slicing.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party" / "slicegpt" / "src"))

import psutil
import torch
from datasets import Dataset
from torch import FloatTensor, LongTensor, Tensor, matmul
from torch.nn import Linear, Module
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM, Qwen2RMSNorm

from slicegpt import data_utils, layernorm_fusion, rotate, utils
from slicegpt.config import config
from slicegpt.model_adapter import LayerAdapter, ModelAdapter
from slicegpt.slicing_scheduler import SlicingConfig
from slicegpt.slicing_scheduler import ConstSlicingScheduler


class CompressedQwen2DecoderLayer(Qwen2DecoderLayer):
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: LongTensor | None = None,
        past_key_value: tuple[Tensor] | None = None,
        past_key_values=None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        **kwargs,
    ) -> Tensor:
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if past_key_values is None:
            past_key_values = past_key_value
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = attn_output[0] if isinstance(attn_output, tuple) else attn_output
        if self.attn_shortcut_Q is not None:
            hidden_states = matmul(residual, self.attn_shortcut_Q) + hidden_states
        else:
            hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.mlp_shortcut_Q is not None:
            hidden_states = matmul(residual, self.mlp_shortcut_Q) + hidden_states
        else:
            hidden_states = residual + hidden_states

        return hidden_states


class Qwen2LayerAdapter(LayerAdapter):
    def __init__(self, layer: Qwen2DecoderLayer) -> None:
        super().__init__()
        self._layer = layer

    @property
    def layer(self) -> Module:
        return self._layer

    @property
    def hidden_states_args_position(self) -> int:
        return 0

    @property
    def hidden_states_output_position(self) -> int:
        return 0

    def get_first_layernorm(self) -> Module:
        return self.layer.input_layernorm

    def get_second_layernorm(self) -> Module:
        return self.layer.post_attention_layernorm

    def get_attention_inputs(self) -> list[Linear]:
        return [self.layer.self_attn.q_proj, self.layer.self_attn.k_proj, self.layer.self_attn.v_proj]

    def get_attention_output(self) -> Linear:
        return self.layer.self_attn.o_proj

    def get_mlp_inputs(self) -> list[Linear]:
        return [self.layer.mlp.gate_proj, self.layer.mlp.up_proj]

    def get_mlp_output(self) -> Linear:
        return self.layer.mlp.down_proj


class Qwen2ModelAdapter(ModelAdapter):
    def __init__(self, model: Qwen2ForCausalLM) -> None:
        super().__init__()
        self._model = model

    @property
    def model(self) -> Module:
        return self._model

    @property
    def config(self):
        return self._model.config

    @property
    def config_type(self) -> type:
        return Qwen2Config

    @property
    def parallel_blocks(self) -> bool:
        return False

    @property
    def seqlen(self) -> int:
        return self.config.max_position_embeddings

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    @property
    def should_bake_mean_into_linear(self) -> bool:
        return False

    @property
    def original_layer_type(self) -> type:
        return Qwen2DecoderLayer

    @property
    def original_layer_norm_type(self) -> type:
        return Qwen2RMSNorm

    @property
    def layer_adapter_type(self) -> type:
        return Qwen2LayerAdapter

    @property
    def compressed_layer_type(self) -> type:
        return CompressedQwen2DecoderLayer

    @property
    def use_cache(self) -> bool:
        return self.config.use_cache

    @use_cache.setter
    def use_cache(self, value: bool) -> None:
        self.config.use_cache = value

    def compute_output_logits(self, input_ids: Tensor) -> FloatTensor:
        return self.model(input_ids=input_ids).logits

    def convert_layer_to_compressed(self, layer: Module, layer_idx: int | None) -> Module:
        compressed_layer = self.compressed_layer_type(self.config, layer_idx).to(self.config.torch_dtype)
        compressed_layer.load_state_dict(layer.state_dict(), strict=True)
        return compressed_layer

    def get_layers(self) -> list[LayerAdapter]:
        return [self.layer_adapter_type(layer) for layer in self.model.model.layers]

    def get_raw_layer_at(self, index: int) -> Module:
        return self.model.model.layers[index]

    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        self.model.model.layers[index] = new_layer

    def get_embeddings(self) -> list[Module]:
        return [self.model.model.embed_tokens]

    def get_pre_head_layernorm(self) -> Module:
        return self.model.model.norm

    def get_lm_head(self) -> Linear:
        return self.model.lm_head

    def post_init(self, tokenizer) -> None:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.config.pad_token_id = tokenizer.pad_token_id

    @classmethod
    def _from_pretrained(
        cls,
        model_name: str,
        model_path: str,
        *,
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> ModelAdapter | None:
        if "qwen" not in model_name.lower():
            return None
        model = Qwen2ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            token=token,
            local_files_only=local_files_only,
        )
        model.config.torch_dtype = dtype
        return cls(model)

    @classmethod
    def _from_uninitialized(
        cls,
        model_name: str,
        model_path: str,
        *,
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> ModelAdapter | None:
        if "qwen" not in model_name.lower():
            return None

        class UninitializedQwen2ForCausalLM(Qwen2ForCausalLM):
            def _init_weights(self, _) -> None:
                pass

        qwen_config = AutoConfig.from_pretrained(
            model_path,
            torch_dtype=dtype,
            token=token,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        model = UninitializedQwen2ForCausalLM(qwen_config).to(dtype=dtype)
        model.config.torch_dtype = dtype
        return cls(model)


def dtype_from_name(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def count_params(model: torch.nn.Module) -> int:
    return sum(int(param.nelement()) for param in model.parameters())


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sliced_artifact_stem(model_name: str, sparsity: float) -> str:
    return f"{Path(model_name).name}_{sparsity}"


def load_sliced_qwen_model(
    base_model: str,
    sliced_model_path: str | Path,
    sparsity: float,
    round_interval: int = 128,
    dtype: torch.dtype = torch.float16,
    device: str = "cpu",
    local_files_only: bool = False,
):
    sliced_model_path = Path(sliced_model_path)
    tokenizer = AutoTokenizer.from_pretrained(sliced_model_path, trust_remote_code=True, local_files_only=local_files_only)
    model = Qwen2ForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    )
    model.config.torch_dtype = dtype
    adapter = Qwen2ModelAdapter(model)
    adapter.use_cache = False
    adapter.post_init(tokenizer)
    layernorm_fusion.replace_layers(adapter)
    layernorm_fusion.fuse_modules(adapter)

    hidden_size = adapter.hidden_size
    for layer_adapter in adapter.get_layers():
        layer_adapter.layer.mlp_shortcut_Q = torch.nn.Parameter(torch.zeros(hidden_size, hidden_size, dtype=dtype))
        layer_adapter.layer.attn_shortcut_Q = torch.nn.Parameter(torch.zeros(hidden_size, hidden_size, dtype=dtype))

    stem = sliced_artifact_stem(base_model, sparsity)
    config_path = sliced_model_path / f"{stem}.json"
    if config_path.exists():
        adapter.slicing_conf = SlicingConfig.from_json_string(config_path.read_text(encoding="utf-8"))
    else:
        new_embedding_dimension = int((1 - sparsity) * hidden_size)
        new_embedding_dimension -= new_embedding_dimension % round_interval
        slicing_conf = SlicingConfig()
        slicing_conf.const_dimension = new_embedding_dimension
        adapter.slicing_conf = slicing_conf

    rotate.slice_rotated_model(adapter)
    state_path = sliced_model_path / f"{stem}.pt"
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    target_device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model.to(target_device)
    model.eval()
    return model, tokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cal-dataset", default="wikitext2", choices=["wikitext2", "ptb", "c4", "alpaca"])
    parser.add_argument("--cal-guide-file", action="append", type=Path, default=[])
    parser.add_argument("--cal-guide-limit-per-file", type=int, default=0)
    parser.add_argument("--cal-nsamples", type=int, default=1)
    parser.add_argument("--cal-batch-size", type=int, default=1)
    parser.add_argument("--cal-max-seqlen", type=int, default=64)
    parser.add_argument("--sparsity", type=float, default=0.20)
    parser.add_argument("--round-interval", type=int, default=128)
    parser.add_argument("--final-orientation", choices=["random", "pca"], default="random")
    parser.add_argument("--save-sliced-state", action="store_true")
    parser.add_argument("--save-hf-files", action="store_true", help="Save tokenizer/config files next to the official SliceGPT state/config artifacts.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not (0.0 <= args.sparsity < 1.0):
        raise ValueError("--sparsity must be in [0, 1)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    started = time.time()
    proc = psutil.Process()
    dtype = dtype_from_name(args.dtype)
    config.device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    logging.info("Loading tokenizer from %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=args.local_files_only)

    logging.info("Loading Qwen2 weights")
    model = Qwen2ForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    )
    model.eval()
    model.config.torch_dtype = dtype
    model_adapter = Qwen2ModelAdapter(model)
    model_adapter.use_cache = False
    model_adapter.post_init(tokenizer)

    cal_guide_rows = []
    if args.cal_guide_file:
        for guide_file in args.cal_guide_file:
            rows = read_jsonl(guide_file)
            selected = rows if args.cal_guide_limit_per_file <= 0 else rows[: args.cal_guide_limit_per_file]
            for row in selected:
                if row.get("contains_solution"):
                    raise ValueError(f"guide row contains_solution=true: {row.get('task_id')}")
            cal_guide_rows.extend(selected)
        if not cal_guide_rows:
            raise ValueError("--cal-guide-file was provided but no guide rows were loaded")
        dataset = {"train": Dataset.from_dict({"text": [row["prompt"] for row in cal_guide_rows]})}
    else:
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

    params_before = count_params(model)
    new_embedding_dimension = int((1 - args.sparsity) * model_adapter.hidden_size)
    new_embedding_dimension -= new_embedding_dimension % args.round_interval
    summary = {
        "status": "dry_run" if args.dry_run else "planned",
        "method": "SliceGPT official Qwen2 adapter",
        "model": args.model,
        "dtype": args.dtype,
        "device": str(config.device),
        "local_files_only": args.local_files_only,
        "cal_dataset": args.cal_dataset,
        "cal_guide_files": [str(path) for path in args.cal_guide_file],
        "cal_guide_samples": len(cal_guide_rows),
        "cal_nsamples": args.cal_nsamples,
        "cal_batch_size": args.cal_batch_size,
        "cal_max_seqlen": args.cal_max_seqlen,
        "sparsity_requested": args.sparsity,
        "hidden_size": model_adapter.hidden_size,
        "new_embedding_dimension": new_embedding_dimension,
        "parameters_before": params_before,
    }
    if args.dry_run:
        (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    layernorm_fusion.replace_layers(model_adapter)
    layernorm_fusion.fuse_modules(model_adapter)
    scheduler = ConstSlicingScheduler(new_embedding_dimension)
    rotate.rotate_and_slice(model_adapter, train_loader, scheduler, final_orientation=args.final_orientation)
    params_after = count_params(model)

    sliced_model_dir = args.out_dir / "sliced_model"
    sliced_state_path = None
    sliced_config_path = None
    if args.save_sliced_state:
        sliced_model_dir.mkdir(parents=True, exist_ok=True)
        stem = sliced_artifact_stem(args.model, args.sparsity)
        sliced_state_path = sliced_model_dir / f"{stem}.pt"
        sliced_config_path = sliced_model_dir / f"{stem}.json"
        torch.save(model.state_dict(), sliced_state_path)
        if model_adapter.slicing_conf is not None:
            sliced_config_path.write_text(model_adapter.slicing_conf.to_json_string(), encoding="utf-8")
        if args.save_hf_files:
            tokenizer.save_pretrained(sliced_model_dir)
            model.config.save_pretrained(sliced_model_dir)

    summary.update(
        {
            "status": "success",
            "parameters_after": params_after,
            "parameter_keep_ratio": params_after / params_before,
            "parameter_reduction_rate": 1 - params_after / params_before,
            "sliced_state_path": str(sliced_state_path) if sliced_state_path else None,
            "slicing_config_path": str(sliced_config_path) if sliced_config_path else None,
            "sliced_model": str(sliced_model_dir) if sliced_state_path else None,
            "sliced_model_loader": "methods.slicegpt.qwen_prune.load_sliced_qwen_model",
            "process_rss_mb_after": proc.memory_info().rss / 1024**2,
            "elapsed_seconds": time.time() - started,
            "torch": torch.__version__,
        }
    )
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    utils.cleanup_memory()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
