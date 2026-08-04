from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run(command: list[str]) -> dict[str, object]:
    try:
        proc = subprocess.run(command, text=True, capture_output=True, timeout=30)
        return {
            "ok": proc.returncode == 0,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {"ok": False, "stdout": "", "stderr": repr(exc)}


def module_status(names: list[str]) -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in names}


def torch_status() -> dict[str, object]:
    import torch

    status: dict[str, object] = {
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        status["gpu"] = torch.cuda.get_device_name(0)
        status["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        x = torch.randn(256, 256, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        status["cuda_matmul"] = list(y.shape)
    return status


def main() -> int:
    modules = [
        "torch",
        "transformers",
        "accelerate",
        "datasets",
        "evaluate",
        "safetensors",
        "sentencepiece",
        "peft",
        "yaml",
        "jsonlines",
        "psutil",
        "evalplus",
        "human_eval",
        "swebench",
    ]
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "env": {
            "HF_HOME": os.environ.get("HF_HOME"),
            "HUGGINGFACE_HUB_CACHE": os.environ.get("HUGGINGFACE_HUB_CACHE"),
            "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE"),
            "TORCH_HOME": os.environ.get("TORCH_HOME"),
        },
        "tools": {
            name: shutil.which(name)
            for name in ["git", "git-lfs", "gh", "uv", "micromamba", "ninja", "docker", "nvidia-smi"]
        },
        "commands": {
            "git": run(["git", "--version"]),
            "git_lfs": run(["git-lfs", "version"]),
            "docker": run(["docker", "ps"]),
            "nvidia_smi": run(["nvidia-smi"]),
        },
        "modules": module_status(modules),
        "torch": torch_status(),
    }

    out_dir = ROOT / "results" / "stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "environment.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    required_modules_ok = all(report["modules"].values())
    torch_ok = bool(report["torch"].get("cuda_available"))
    return 0 if required_modules_ok and torch_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
