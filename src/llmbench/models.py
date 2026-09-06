"""Model registry + GGUF download.

All models are small instruct models with GGUF quants available from
`bartowski/*` (consistent naming, every quant level). Everything here fits in
~22 GB RAM at Q8 or below.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

CACHE = os.path.expanduser("~/.cache/llmbench/models")

QUANTS = ["Q3_K_M", "Q4_K_M", "Q5_K_M", "Q8_0"]


@dataclass(frozen=True)
class Model:
    key: str
    params_b: float          # nominal param count (billions)
    repo: str                # HF repo holding the GGUF files
    stem: str                # filename stem: "{stem}-{QUANT}.gguf"


REGISTRY: dict[str, Model] = {
    m.key: m for m in [
        Model("gemma-2-2b", 2.6, "bartowski/gemma-2-2b-it-GGUF", "gemma-2-2b-it"),
        Model("llama-3.2-3b", 3.2, "bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct"),
        Model("qwen2.5-3b", 3.1, "bartowski/Qwen2.5-3B-Instruct-GGUF", "Qwen2.5-3B-Instruct"),
        Model("phi-3.5-mini", 3.8, "bartowski/Phi-3.5-mini-instruct-GGUF", "Phi-3.5-mini-instruct"),
        Model("mistral-7b", 7.2, "bartowski/Mistral-7B-Instruct-v0.3-GGUF", "Mistral-7B-Instruct-v0.3"),
        Model("qwen2.5-7b", 7.6, "bartowski/Qwen2.5-7B-Instruct-GGUF", "Qwen2.5-7B-Instruct"),
        Model("llama-3.1-8b", 8.0, "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct"),
    ]
}


def gguf_path(model_key: str, quant: str) -> str:
    """Download (cached) the GGUF for model_key at quant, return local path."""
    from huggingface_hub import hf_hub_download

    m = REGISTRY[model_key]
    fn = f"{m.stem}-{quant}.gguf"
    return hf_hub_download(repo_id=m.repo, filename=fn, cache_dir=CACHE)


def file_size_gb(path: str) -> float:
    return round(os.path.getsize(path) / 1e9, 3)
