# Results

**Box:** 8 vCPU AMD EPYC 7B12, 31 GB RAM, no GPU. `llama-cpp-python` 0.3.35.
Prompt ≈ 512 tok, decode 128 tok, 8 threads. Quality = 25 MCQ + 5
format-following items — a **sanity gauge, not a real eval** (n=30).

## CPU — full grid (Runs 1 + 2)  ·  2026-09-06  ·  `results/cpu_all.json` / `.png`

| model | params | quant | file GB | decode tok/s | peak RAM GB | quality |
|---|---|---|---|---|---|---|
| gemma-2-2b | 2.6 | Q4_K_M | 1.7 | 15.4 | 3.3 | 0.53 |
| gemma-2-2b | 2.6 | Q8_0 | 2.8 | 12.1 | 3.8 | 0.50 |
| qwen2.5-3b | 3.1 | Q3_K_M | 1.6 | 17.2 | 2.5 | 0.63 |
| **qwen2.5-3b** | 3.1 | **Q4_K_M** | 1.9 | **16.7** | **3.6** | **0.70** |
| qwen2.5-3b | 3.1 | Q8_0 | 3.3 | 15.3 | 3.7 | 0.70 |
| llama-3.2-3b | 3.2 | Q4_K_M | 2.0 | **18.4** | 4.1 | 0.67 |
| llama-3.2-3b | 3.2 | Q8_0 | 3.4 | 12.7 | 4.6 | 0.67 |
| phi-3.5-mini | 3.8 | Q4_K_M | 2.4 | 13.7 | 5.4 | 0.77 |
| phi-3.5-mini | 3.8 | Q8_0 | 4.1 | 10.3 | 5.7 | 0.73 |
| mistral-7b | 7.2 | Q4_K_M | 4.4 | 9.2 | 8.4 | 0.53 |
| **qwen2.5-7b** | 7.6 | **Q3_K_M** | 3.8 | 7.2 | **5.5** | **0.87** |
| **qwen2.5-7b** | 7.6 | **Q4_K_M** | 4.7 | 9.7 | 8.2 | **0.87** |
| llama-3.1-8b | 8.0 | Q4_K_M | 4.9 | 9.1 | 9.1 | 0.67 |

*(mistral-7b Q3_K_M reported 37 tok/s / 0.57 quality — a measurement artifact,
re-running; excluded above.)*

## What the CPU grid says

1. **qwen2.5-7b is a different tier — 0.87 quality vs ~0.70 for the whole 3B
   class** — and it *holds 0.87 even at Q3_K_M* (5.5 GB, 7 tok/s). If the box has
   ~6 GB free and you can live with ~7 tok/s, this is the pick.
2. **The 3B sweet spot is qwen2.5-3b Q4_K_M** — 16.7 tok/s, 3.6 GB, 0.70. Fast
   enough for interactive chat on a commodity box, no GPU.
3. **On CPU there is no reason to run Q8_0** — quality is equal or worse than Q4
   for every model, at 20–40 % lower throughput and more RAM. Q3_K_M is the real
   floor: fine for qwen (0.87 = 0.87) and phi, costs llama ~4 pts, breaks nothing.
4. **Two models to skip:** `llama-3.1-8b` (0.67 — no better than the 3.2-3B at
   2.5× the size) and `gemma-2-2b` (0.44 on reasoning, though perfect on
   format-following — a niche pick for constrained output).
5. **7B on CPU ≈ 9 tok/s, half the 3B speed.** Usable for batch / agents /
   non-interactive; borderline for live chat.

## GPU — Kaggle, Tesla P100 16 GB  ·  2026-09-06  ·  `results/gpu_grid.json`

11 cells, all layers on GPU. Same models, same probe.

| model | quant | decode tok/s | prefill tok/s | quality |
|---|---|---|---|---|
| gemma-2-2b | Q4_K_M | 68.8 | 1290 | 0.57 |
| gemma-2-2b | Q8_0 | 72.3 | 1760 | 0.50 |
| qwen2.5-3b | Q4_K_M | 58.6 | 1569 | 0.70 |
| qwen2.5-3b | Q8_0 | 61.9 | 1629 | 0.73 |
| llama-3.2-3b | Q4_K_M | 62.7 | 1241 | 0.67 |
| qwen2.5-7b | Q4_K_M | 36.1 | 774 | 0.87 |
| qwen2.5-7b | Q8_0 | 35.2 | 844 | 0.87 |
| llama-3.1-8b | Q4_K_M | 34.1 | 639 | 0.67 |
| llama-3.1-8b | Q8_0 | 32.7 | 696 | 0.70 |
| **qwen2.5-14b** | Q4_K_M | **19.1** | 281 | **0.93** |

## CPU vs GPU — the decision (`python -m llmbench.compare`)

| | CPU (8 vCPU) | GPU (P100) | ratio |
|---|---|---|---|
| decode, 3B | ~17 tok/s | ~60 tok/s | **3.5×** |
| decode, 7–8B | ~9 tok/s | ~35 tok/s | **3.8×** |
| prefill (RAG-relevant) | ~50–60 tok/s | **~800–1700 tok/s** | **20–30×** |
| rent (rough) | ~$0.05/h | ~$0.40/h | 8× |
| $ per 1M output tokens | **~$0.75–1.9** | ~$1.5–3.3 | GPU costs *more* |

**The break-even:** GPU rent is ~8× the CPU box; GPU decode is only ~3.5–4× faster
→ **on pure $/token, the CPU box is cheaper.** The GPU earns its keep on:

1. **Latency / UX** — 60 tok/s vs 17 is a snappy assistant vs a sluggish one.
   For anything a person waits on, get the GPU.
2. **Models that don't run on CPU** — `qwen2.5-14b Q4` scores **0.93** (a new
   high) at a usable 19 tok/s on the P100; on CPU it's ~3 tok/s, i.e. unusable.
3. **RAG / long context** — prefill is 20–30× faster. A 4k-token prompt is
   near-instant on GPU, a multi-second stall on CPU.
4. On GPU, **Q8_0 is fine** — sometimes faster than Q4, quality equal or better.
   The "Q4 only" rule is CPU-specific.

Plot: `results/cpu_vs_gpu.png`.


## Method caveats (harden next)

- Prefill number includes call overhead — move to llama.cpp's timing counters.
- Single run per cell — add 3 repeats + spread.
- Quality probe is directional (n=30), not MMLU.
- All single-request; the concurrency curve (1/2/4 parallel) is next.
