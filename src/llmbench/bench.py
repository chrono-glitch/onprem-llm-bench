"""Measure one (model, quant) on this CPU: load time, prefill throughput,
decode throughput, peak RAM. Run one config per process so RAM is isolated.

    python -m llmbench.bench --model llama-3.2-3b --quant Q4_K_M --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time

from . import models, quality

PREFILL_PROMPT_TOKENS = 512    # target size of the prompt used for prefill timing
DECODE_TOKENS = 128            # tokens to generate for decode timing
N_CTX = 4096


def _peak_ram_gb() -> float:
    # ru_maxrss is KiB on Linux
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3)


def run(model_key: str, quant: str, n_threads: int | None = None) -> dict:
    from llama_cpp import Llama

    n_threads = n_threads or os.cpu_count()
    path = models.gguf_path(model_key, quant)
    size_gb = models.file_size_gb(path)

    t0 = time.perf_counter()
    llm = Llama(model_path=path, n_ctx=N_CTX, n_threads=n_threads, verbose=False)
    load_s = round(time.perf_counter() - t0, 2)

    # a prompt of roughly PREFILL_PROMPT_TOKENS tokens
    filler = ("The quick brown fox jumps over the lazy dog. " * 200)
    toks = llm.tokenize(filler.encode(), add_bos=True)[:PREFILL_PROMPT_TOKENS]
    prompt = llm.detokenize(toks).decode(errors="ignore")
    n_prompt = len(toks)

    # run 1: generate 1 token -> time ~= prefill
    t0 = time.perf_counter()
    llm.create_completion(prompt, max_tokens=1, temperature=0.0)
    prefill_s = time.perf_counter() - t0

    # run 2: generate DECODE_TOKENS -> decode rate = extra tokens / extra time
    llm.reset()
    t0 = time.perf_counter()
    out = llm.create_completion(prompt, max_tokens=DECODE_TOKENS, temperature=0.0)
    total_s = time.perf_counter() - t0
    n_gen = out["usage"]["completion_tokens"]

    decode_s = max(total_s - prefill_s, 1e-6)

    def chat(prompt: str) -> str:
        llm.reset()
        r = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=48, temperature=0.0,
        )
        return r["choices"][0]["message"]["content"] or ""

    q = quality.score(chat)

    return {
        "model": model_key,
        "quant": quant,
        "params_b": models.REGISTRY[model_key].params_b,
        "file_gb": size_gb,
        "load_s": load_s,
        "n_threads": n_threads,
        "prefill_tok_s": round(n_prompt / prefill_s, 1),
        "decode_tok_s": round(max(n_gen - 1, 1) / decode_s, 1),
        "peak_ram_gb": _peak_ram_gb(),
        **q,
    }


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--quant", required=True)
    p.add_argument("--threads", type=int)
    p.add_argument("--json")
    a = p.parse_args()
    r = run(a.model, a.quant, a.threads)
    print(json.dumps(r, indent=2))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f)


if __name__ == "__main__":
    _cli()
