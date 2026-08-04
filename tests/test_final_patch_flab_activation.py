from __future__ import annotations

import json
from pathlib import Path


from methods.flab_pruner import qwen_prune


class FakeTokenizer:
    def __call__(self, prompts, **kwargs):
        import torch

        self.prompts = prompts
        return {"input_ids": torch.ones((len(prompts), 4), dtype=torch.long)}

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer.json").write_text("{}", encoding="utf-8")


class FakeModule:
    def __init__(self, width: int):
        self.width = width
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)

        class Handle:
            def remove(inner_self):
                self.hooks.remove(hook)

        return Handle()


class FakeModel:
    def __init__(self):
        import torch

        self.forward_calls = 0
        self.saved = False
        self.pruned_zs = None
        self.module = FakeModule(6)
        self.weight = torch.nn.Parameter(torch.ones(6))

    def named_modules(self):
        return [("layers.0.mlp.down_proj", self.module)]

    def eval(self):
        return self

    def to(self, device):
        return self

    def parameters(self):
        return [self.weight]

    def __call__(self, **kwargs):
        import torch

        self.forward_calls += 1
        activation = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
        for hook in list(self.module.hooks):
            hook(self.module, (), activation)
        return {"logits": activation}

    def prune(self, config=None, stage="top", zs=None):
        self.pruned_zs = zs

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "model.bin").write_text("fake", encoding="utf-8")


class FakeAuto:
    model = FakeModel()
    tokenizer = FakeTokenizer()

    @staticmethod
    def tokenizer_from_pretrained(*args, **kwargs):
        return FakeAuto.tokenizer

    @staticmethod
    def model_from_pretrained(*args, **kwargs):
        return FakeAuto.model


def test_benchmark_activation_uses_real_forward_hooks_and_prune(monkeypatch, tmp_path: Path):
    guide = tmp_path / "guide.jsonl"
    guide.write_text(json.dumps({"task_id": "t0", "prompt": "def f(): pass", "contains_solution": False}) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    monkeypatch.setattr(qwen_prune, "load_hf_tokenizer", FakeAuto.tokenizer_from_pretrained)
    monkeypatch.setattr(qwen_prune, "load_flab_qwen_model", FakeAuto.model_from_pretrained)

    rc = qwen_prune.main(
        [
            "--model",
            "fake",
            "--guide-file",
            str(guide),
            "--save-dir",
            str(out),
            "--importance-mode",
            "benchmark_activation",
            "--importance-device",
            "cpu",
            "--prune-ratio",
            "0.5",
        ]
    )

    assert rc == 0
    assert FakeAuto.model.forward_calls == 1
    assert FakeAuto.model.pruned_zs
    mask = next(iter(FakeAuto.model.pruned_zs.values()))
    assert len(mask) == 6
    assert sum(mask) == 3
    result = json.loads((out / "flab_qwen_prune_result.json").read_text(encoding="utf-8"))
    assert result["officiality"] == "experimental_extension"
    assert result["importance_summary"]
    assert (out / "pruned_model" / "model.bin").exists()


def test_structural_mode_does_not_collect_activation(monkeypatch, tmp_path: Path):
    guide = tmp_path / "guide.jsonl"
    guide.write_text(json.dumps({"task_id": "t0", "prompt": "def f(): pass", "contains_solution": False}) + "\n", encoding="utf-8")
    called = False

    def fail_collect(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("structural mode must not collect activation")

    monkeypatch.setattr(qwen_prune, "collect_activation_statistics", fail_collect)
    rc = qwen_prune.main(["--model", "fake", "--guide-file", str(guide), "--save-dir", str(tmp_path / "out"), "--dry-run"])
    assert rc == 0
    assert called is False
