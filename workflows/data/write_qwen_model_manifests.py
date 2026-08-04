#!/usr/bin/env python3
"""Write lightweight Qwen model manifests from config probes and HF metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "qwen25_coder_15b_instruct": {
        "model_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "probe_glob": "results/evidence/diagnostics/qwen25_coder_15b_config_probe_*/qwen_probe.json",
        "role": "baseline_and_debug",
    },
    "qwen25_coder_3b_instruct": {
        "model_id": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "probe_glob": "results/evidence/diagnostics/qwen25_coder_3b_config_probe_*/qwen_probe.json",
        "role": "pruning_target_and_comparison",
    },
}


def latest_probe(pattern: str) -> dict:
    matches = sorted(ROOT.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no probe found for {pattern}")
    return json.loads(matches[-1].read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Write Qwen model manifest JSON files.")
    parser.add_argument("--skip-hf-info", action="store_true", help="Do not query HF model_info; revision will be null.")
    args = parser.parse_args()

    if not args.skip_hf_info:
        from huggingface_hub import model_info
    else:
        model_info = None

    out_dir = ROOT / "data" / "manifests" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    for short_name, spec in MODELS.items():
        probe = latest_probe(spec["probe_glob"])
        info = model_info(spec["model_id"]) if model_info else None
        siblings = getattr(info, "siblings", []) if info else []
        files = [
            {
                "name": item.rfilename,
                "size_bytes": getattr(item, "size", None),
                "sha256": None,
            }
            for item in siblings
            if item.rfilename.endswith((".safetensors", ".bin", ".json", ".model", ".txt"))
        ]

        manifest = {
            "model_short_name": short_name,
            "model_id": spec["model_id"],
            "role": spec["role"],
            "revision": getattr(info, "sha", None) if info else None,
            "tokenizer_revision": getattr(info, "sha", None) if info else None,
            "dtype": probe["config"].get("torch_dtype"),
            "local_cache": "not_committed",
            "loads_weights_in_probe": probe.get("loads_weights", False),
            "config": probe["config"],
            "tokenizer": probe.get("tokenizer"),
            "files": files,
            "notes": "Lightweight manifest from config/tokenizer probe and HF metadata; raw model weights are not committed.",
        }
        out_path = out_dir / f"{short_name}.json"
        out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out_path.relative_to(ROOT)} revision={manifest['revision']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
