from __future__ import annotations

import re


FENCE_ONLY_LINES = {"```", "```python", "```py"}


def normalize_completion(text: str) -> str:
    completion = text.replace("\r\n", "\n")

    fenced_blocks = re.findall(
        r"```(?:python|py)?\s*\n(.*?)```",
        completion,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for block in fenced_blocks:
        if block.strip():
            completion = block
            break

    lines = [
        line
        for line in completion.splitlines()
        if line.strip().lower() not in FENCE_ONLY_LINES
    ]
    completion = "\n".join(lines)
    return completion.rstrip() + "\n" if completion.strip() else "\n"


def extract_completion(prompt: str, generated: str) -> str:
    if generated.startswith(prompt):
        generated = generated[len(prompt):]
    return normalize_completion(generated)
