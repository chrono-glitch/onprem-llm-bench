"""Concurrency: what happens when N users hit one model instance at once.

A single `llama-cpp-python` `Llama` serialises calls (one internal lock). This is
what a naive single-instance deployment does — so the useful question is *how bad
does it get*: at concurrency C, each request waits behind the others.

For each C in --levels: fire C identical short generations from a threadpool
against one loaded model, record per-request latency, and report:
  - p50 / p95 request latency
  - aggregate output tok/s (all requests / wall time)
  - latency inflation vs C=1

    python -m llmbench.conc --model qwen2.5-3b --quant Q4_K_M --levels 1,2,4,8
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from . import models

PROMPT = "Write three sentences about why on-premise software can be a good choice."
GEN_TOKENS = 96


def _one(llm) -> tuple[float, int]:
    t0 = time.perf_counter()
    r = llm.create_completion(PROMPT, max_tokens=GEN_TOKENS, temperature=0.0)
    return time.perf_counter() - t0, r["usage"]["completion_tokens"]


def run(model_key: str, quant: str, levels: list[int], n_gpu_layers: int = 0) -> list[dict]:
    from llama_cpp import Llama

    path = models.gguf_path(model_key, quant)
    llm = Llama(model_path=path, n_ctx=4096, n_gpu_layers=n_gpu_layers, verbose=False)
    _one(llm)  # warmup

    rows = []
    base_p50 = None
    for c in levels:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=c) as ex:
            results = list(ex.map(lambda _: _one(llm), range(c)))
        wall = time.perf_counter() - t0
        lat = sorted(x[0] for x in results)
        toks = sum(x[1] for x in results)
        p50 = statistics.median(lat)
        base_p50 = base_p50 or p50
        rows.append({
            "model": model_key, "quant": quant, "concurrency": c,
            "p50_s": round(p50, 2),
            "p95_s": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 2),
            "agg_tok_s": round(toks / wall, 1),
            "latency_x_vs_c1": round(p50 / base_p50, 2),
        })
        print(f"  C={c:>2}  p50 {rows[-1]['p50_s']:>6}s  p95 {rows[-1]['p95_s']:>6}s  "
              f"agg {rows[-1]['agg_tok_s']:>6} tok/s  ({rows[-1]['latency_x_vs_c1']}x latency)")
    return rows


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--quant", default="Q4_K_M")
    p.add_argument("--levels", default="1,2,4,8")
    p.add_argument("--n-gpu-layers", type=int, default=0)
    p.add_argument("--out", default="results/conc.json")
    a = p.parse_args()
    rows = run(a.model, a.quant, [int(x) for x in a.levels.split(",")], a.n_gpu_layers)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"saved: {a.out}")


if __name__ == "__main__":
    _cli()
