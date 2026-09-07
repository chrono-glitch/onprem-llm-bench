"""Engine #2 — Ollama. Same models, same GGUF files, same probe as `bench.py`,
so the two are directly comparable: what does the Ollama wrapper cost vs calling
llama.cpp in-process?

Ollama is llama.cpp underneath, so the interesting number isn't "is the kernel
faster" (it's the same kernel) — it's the wrapper overhead (HTTP, its scheduler,
its own KV-cache handling) and what you get back for it (model management, an
always-on server, automatic load/unload).

We import each GGUF we already downloaded into Ollama via a `FROM <path>`
Modelfile — identical weights — then hit `/api/generate` (which returns exact
llama.cpp timing counters: load_duration, prompt_eval_*, eval_*).

    python -m llmbench.ollama_bench --model qwen2.5-3b --quant Q4_K_M --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
import urllib.request

from . import models, quality

HOST = "http://127.0.0.1:11434"
PREFILL_PROMPT_TOKENS = 512
DECODE_TOKENS = 128
N_CTX = 4096


def _post(path: str, body: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(
        HOST + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _tag(model_key: str, quant: str) -> str:
    return f"llmbench-{model_key}-{quant.lower().replace('_', '-')}:latest"


def _ensure_model(model_key: str, quant: str) -> tuple[str, str]:
    """Create the Ollama model from our local GGUF if it isn't there yet.
    Returns (tag, gguf_path)."""
    path = models.gguf_path(model_key, quant)
    tag = _tag(model_key, quant)
    with urllib.request.urlopen(HOST + "/api/tags", timeout=30) as r:
        existing = {m["name"] for m in json.loads(r.read()).get("models", [])}
    if tag not in existing:
        # `ollama create` needs a Modelfile on disk (stdin `-f -` is unreliable
        # in 0.33). Ollama then copies the GGUF blob into ~/.ollama.
        with tempfile.NamedTemporaryFile("w", suffix=".Modelfile", delete=False) as mf:
            mf.write(f"FROM {path}\n")
            mf_path = mf.name
        try:
            subprocess.run(
                ["ollama", "create", tag, "-f", mf_path], check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
        finally:
            os.unlink(mf_path)
    return tag, path


def _gen(tag: str, prompt: str, num_predict: int) -> dict:
    return _post("/api/generate", {
        "model": tag, "prompt": prompt, "stream": False,
        "options": {
            "temperature": 0.0, "num_predict": num_predict,
            "num_ctx": N_CTX, "num_thread": os.cpu_count(),
        },
    })


def _ollama_size_gb(tag: str) -> float | None:
    try:
        with urllib.request.urlopen(HOST + "/api/ps", timeout=15) as r:
            for m in json.loads(r.read()).get("models", []):
                if m["name"] == tag:
                    return round(m["size"] / 1e9, 3)
    except Exception:
        pass
    return None


def run(model_key: str, quant: str) -> dict:
    tag, path = _ensure_model(model_key, quant)
    size_gb = models.file_size_gb(path)

    # unload first so load_duration in the next call is a real cold load
    try:
        _post("/api/generate", {"model": tag, "prompt": "", "keep_alive": 0})
        time.sleep(1)
    except Exception:
        pass

    filler = "The quick brown fox jumps over the lazy dog. " * 200
    # ~4 chars/token; take enough chars for ~PREFILL_PROMPT_TOKENS tokens
    prompt = filler[: PREFILL_PROMPT_TOKENS * 4]

    d = _gen(tag, prompt, DECODE_TOKENS)
    load_s = round(d.get("load_duration", 0) / 1e9, 2)
    pe_n, pe_ns = d.get("prompt_eval_count"), d.get("prompt_eval_duration")
    ev_n, ev_ns = d.get("eval_count"), d.get("eval_duration")
    prefill_tps = round(pe_n / (pe_ns / 1e9), 1) if pe_n and pe_ns else None
    decode_tps = round(ev_n / (ev_ns / 1e9), 1) if ev_n and ev_ns else None

    def chat(p: str) -> str:
        r = _post("/api/chat", {
            "model": tag, "stream": False,
            "messages": [{"role": "user", "content": p}],
            "options": {"temperature": 0.0, "num_predict": 48, "num_ctx": N_CTX},
        })
        return r["message"]["content"] or ""

    q = quality.score(chat)
    size_loaded = _ollama_size_gb(tag)

    return {
        "engine": "ollama",
        "model": model_key,
        "quant": quant,
        "params_b": models.REGISTRY[model_key].params_b,
        "device": "cpu",
        "file_gb": size_gb,
        "load_s": load_s,
        "n_threads": os.cpu_count(),
        "prefill_tok_s": prefill_tps,
        "decode_tok_s": decode_tps,
        "ollama_loaded_gb": size_loaded,
        "prompt_tokens": pe_n,
        "gen_tokens": ev_n,
        **q,
    }


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--quant", default="Q4_K_M")
    p.add_argument("--json")
    a = p.parse_args()
    r = run(a.model, a.quant)
    print(json.dumps(r, indent=2))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f)


if __name__ == "__main__":
    _cli()
