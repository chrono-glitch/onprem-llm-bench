# onprem-llm-bench

**Serving LLMs on your own hardware: what actually works, and how to choose.**

A business wants an LLM on its own box — data can't leave the building, or a
per-token cloud bill doesn't fit. They have a normal server: 8 CPU cores, ~30 GB
RAM, **no GPU**. Which open model, at which quantization, on which engine — and is
a GPU worth buying?

This benchmarks instruct models (2–32B) at multiple GGUF quant levels across
**CPU and a single 16 GB GPU**, measuring prefill/decode throughput, RAM/VRAM,
load time, concurrency behaviour, and a small quality gauge — then turns it into
a decision guide.

- **[RESULTS.md](RESULTS.md)** — every number: CPU grid, GPU grid, CPU-vs-GPU
  cost model, concurrency curve, Ollama-vs-llama.cpp engine comparison.
- **[DECISION.md](DECISION.md)** — the lookup: your box + your use case → the pick.
- **[WRITEUP.en.md](WRITEUP.en.md)** / **[WRITEUP.es.md](WRITEUP.es.md)** — the story.

## Headline findings

1. **On CPU, `Q4_K_M` is the right quant — full stop.** Equal-or-better quality
   than Q8_0 for every model, 20–40 % faster, less RAM. (On GPU, Q8 is fine.)
2. **Sharp quality step at 7B, and `qwen2.5-7b` owns it** — 0.87 vs ~0.70 for the
   entire 3B class, and it holds 0.87 even at Q3_K_M (5.5 GB, ~7 tok/s on CPU).
3. **`qwen2.5-3b Q4_K_M` is the interactive sweet spot** — 16.7 tok/s, 3.6 GB.
4. **A CPU box is a single-user serving unit** — aggregate throughput is flat
   regardless of concurrency; one request already saturates every core.
5. **The engine doesn't change throughput** — Ollama vs in-process
   `llama-cpp-python` decode ratio is 1.00×. Pick the engine for operations.
6. **A GPU loses on $/token** (≈8× the rent, ≈3.5–4× the decode speed) but wins
   on latency, RAG prefill (20–30×), and 14B-class models CPU can't run usefully.

## Run

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
export PYTHONPATH=$PWD/src

# the CPU grid, 3 throughput repeats per cell
python -m llmbench.run --models qwen2.5-3b,qwen2.5-7b --quants Q3_K_M,Q4_K_M --perf-repeats 3

# CPU vs GPU table + cost model
python -m llmbench.compare results/cpu_all.json results/gpu_grid.json

# concurrency: N parallel requests at one model instance (spawns llama_cpp.server)
python -m llmbench.conc --model qwen2.5-3b --quant Q4_K_M --levels 1,2,4,8

# engine #2: same GGUF via Ollama, then compare
python -m llmbench.ollama_bench --model qwen2.5-3b --quant Q4_K_M --json results/o.json
python -m llmbench.engines results/cpu_all.json results/ollama_grid.json
```

GPU track: `kaggle/` — bundles the harness into one notebook, pushes it to
Kaggle's free GPU, polls, pulls `gpu_grid.json`. See `kaggle/drive.sh`.

## Layout

```
src/llmbench/
  models.py        registry + GGUF download (bartowski/*)
  bench.py         one (model, quant) in an isolated process:
                   load / prefill / decode / RAM / quality, N throughput repeats
  quality.py       30-item sanity probe (MCQ reasoning + format-following)
  run.py           the grid -> results table
  compare.py       CPU vs GPU: decode tok/s, $/1M tokens, the break-even
  conc.py          concurrency: N parallel requests vs one llama_cpp.server
  ollama_bench.py  engine #2 — same GGUF imported into Ollama, same probe
  engines.py       Ollama vs llama-cpp-python comparison
  plot.py          the Pareto plots
kaggle/            push the GPU notebook, poll, pull results
```

## Method & caveats

- Prefill/decode use llama.cpp's own perf counters (exact, no call overhead).
- The quality probe is a **sanity gauge (n=30), not MMLU** — it catches a quant
  that broke a model; it does not rank models finely.
- Published CPU numbers: 8 vCPU AMD EPYC 7B12, 31 GB RAM, no GPU,
  `llama-cpp-python` 0.3.35. GPU numbers: Kaggle Tesla P100 16 GB.
