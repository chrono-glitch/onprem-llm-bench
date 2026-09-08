"""Measure one (model, quant) on this CPU: load time, prefill throughput,
decode throughput, peak RAM. Run one config per process so RAM is isolated.

    python -m llmbench.bench --model llama-3.2-3b --quant Q4_K_M --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import time

from . import models, quality

PREFILL_PROMPT_TOKENS = 512    # target size of the prompt used for prefill timing
DECODE_TOKENS = 128            # tokens to generate for decode timing
N_CTX = 4096


def _peak_ram_gb() -> float:
    # ru_maxrss is KiB on Linux
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3)


def _perf_rates(llm):
    """(prefill_tok_s, decode_tok_s) from llama.cpp's own timing counters, or
    (None, None) if the internal API isn't reachable in this build."""
    try:
        import llama_cpp
        ctx = getattr(getattr(llm, "_ctx", None), "ctx", None)
        if ctx is None:
            return None, None
        d = llama_cpp.llama_perf_context(ctx)
        pref = d.n_p_eval / (d.t_p_eval_ms / 1000) if d.t_p_eval_ms else None
        dec = d.n_eval / (d.t_eval_ms / 1000) if d.t_eval_ms else None
        if pref and dec:
            return round(pref, 1), round(dec, 1)
    except Exception:
        pass
    return None, None


def _gpu_name() -> str | None:
    try:
        out = os.popen("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null").read().strip()
        return out.splitlines()[0] if out else None
    except Exception:
        return None


def run(model_key: str, quant: str, n_threads: int | None = None,
        n_gpu_layers: int = 0, perf_repeats: int = 1,
        perf_only: bool = False) -> dict:
    from llama_cpp import Llama

    n_threads = n_threads or os.cpu_count()
    path = models.gguf_path(model_key, quant)
    size_gb = models.file_size_gb(path)
    device = f"gpu:{_gpu_name()}" if n_gpu_layers != 0 else "cpu"

    t0 = time.perf_counter()
    llm = Llama(model_path=path, n_ctx=N_CTX, n_threads=n_threads,
                n_gpu_layers=n_gpu_layers, verbose=False)
    load_s = round(time.perf_counter() - t0, 2)

    # a prompt of roughly PREFILL_PROMPT_TOKENS tokens
    filler = ("The quick brown fox jumps over the lazy dog. " * 200)
    toks = llm.tokenize(filler.encode(), add_bos=True)[:PREFILL_PROMPT_TOKENS]
    prompt = llm.detokenize(toks).decode(errors="ignore")
    n_prompt = len(toks)

    # perf_repeats identical generations; report median rate + spread. temp=0 so
    # the text is deterministic — this measures system/scheduler/thermal noise.
    prefs, decs, n_gen = [], [], 0
    for _ in range(max(perf_repeats, 1)):
        llm.reset()
        t0 = time.perf_counter()
        out = llm.create_completion(prompt, max_tokens=DECODE_TOKENS, temperature=0.0)
        total_s = time.perf_counter() - t0
        n_gen = out["usage"]["completion_tokens"]
        p, d = _perf_rates(llm)
        if p is None:                          # fallback: 2-run wall clock
            llm.reset()
            t1 = time.perf_counter()
            llm.create_completion(prompt, max_tokens=1, temperature=0.0)
            prefill_s = time.perf_counter() - t1
            p = round(n_prompt / prefill_s, 1)
            d = round(max(n_gen - 1, 1) / max(total_s - prefill_s, 1e-6), 1)
        prefs.append(p)
        decs.append(d)

    prefill_tps = round(statistics.median(prefs), 1)
    decode_tps = round(statistics.median(decs), 1)
    decode_spread = (round(min(decs), 1), round(max(decs), 1)) if len(decs) > 1 else None
    decode_cv = (round(statistics.pstdev(decs) / statistics.mean(decs), 3)
                 if len(decs) > 1 and statistics.mean(decs) else None)

    if perf_only:
        q = {}
    else:
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
        "device": device,
        "n_gpu_layers": n_gpu_layers,
        "file_gb": size_gb,
        "load_s": load_s,
        "n_threads": n_threads,
        "prefill_tok_s": prefill_tps,
        "decode_tok_s": decode_tps,
        "decode_spread": decode_spread,      # (min, max) over perf_repeats
        "decode_cv": decode_cv,              # coeff. of variation, or None
        "perf_repeats": max(perf_repeats, 1),
        "peak_ram_gb": _peak_ram_gb(),
        **q,
    }


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--quant", required=True)
    p.add_argument("--threads", type=int)
    p.add_argument("--n-gpu-layers", type=int, default=0, help="0=CPU, -1=all layers on GPU")
    p.add_argument("--perf-repeats", type=int, default=1,
                   help="repeat the throughput measurement N times, report median + spread")
    p.add_argument("--perf-only", action="store_true",
                   help="skip the quality probe (throughput-stability runs)")
    p.add_argument("--json")
    a = p.parse_args()
    r = run(a.model, a.quant, a.threads, a.n_gpu_layers, a.perf_repeats, a.perf_only)
    print(json.dumps(r, indent=2))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f)


if __name__ == "__main__":
    _cli()
