"""Concurrency: N users hitting one model instance, done right.

FINDING (the reason this file exists): a bare `llama-cpp-python` `Llama` object
is **not concurrency-safe** — calling `create_completion` from two threads on one
instance segfaults (`GGML_ASSERT(i1 >= 0 && i1 < ne1)`). On-prem you MUST put a
server in front. This benchmark runs `llama_cpp.server` (its request queue) and
fires concurrent HTTP requests at it.

For each C in --levels: C concurrent `POST /v1/completions`, measure per-request
latency + aggregate tok/s + latency inflation vs C=1.

    python -m llmbench.conc --model qwen2.5-3b --quant Q4_K_M --levels 1,2,4,8
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from . import models

PROMPT = "Write three sentences about why on-premise software can be a good choice."
GEN_TOKENS = 96
PORT = 8123
BASE = f"http://127.0.0.1:{PORT}"


def _post(path: str, body: dict, timeout: float = 300) -> dict:
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _one() -> tuple[float, int]:
    t0 = time.perf_counter()
    r = _post("/v1/completions", {"prompt": PROMPT, "max_tokens": GEN_TOKENS, "temperature": 0.0})
    return time.perf_counter() - t0, r["usage"]["completion_tokens"]


def _wait_ready(proc, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("server exited during startup")
        try:
            _post("/v1/completions", {"prompt": "hi", "max_tokens": 1}, timeout=5)
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(1)
    raise TimeoutError("server not ready")


def run(model_key: str, quant: str, levels: list[int], n_gpu_layers: int = 0) -> list[dict]:
    path = models.gguf_path(model_key, quant)
    server = subprocess.Popen(
        [sys.executable, "-m", "llama_cpp.server", "--model", path,
         "--n_ctx", "4096", "--port", str(PORT), "--n_gpu_layers", str(n_gpu_layers)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        _wait_ready(server)
        rows, base_p50 = [], None
        for c in levels:
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=c) as ex:
                res = list(ex.map(lambda _: _one(), range(c)))
            wall = time.perf_counter() - t0
            lat = sorted(x[0] for x in res)
            p50 = statistics.median(lat)
            base_p50 = base_p50 or p50
            row = {
                "model": model_key, "quant": quant, "concurrency": c,
                "p50_s": round(p50, 2),
                "p95_s": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 2),
                "agg_tok_s": round(sum(x[1] for x in res) / wall, 1),
                "latency_x_vs_c1": round(p50 / base_p50, 2),
            }
            rows.append(row)
            print(f"  C={c:>2}  p50 {row['p50_s']:>6}s  p95 {row['p95_s']:>6}s  "
                  f"agg {row['agg_tok_s']:>6} tok/s  ({row['latency_x_vs_c1']}x latency)")
        return rows
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


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
