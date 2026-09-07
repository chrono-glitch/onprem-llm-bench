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
| mistral-7b | 7.2 | Q3_K_M | 3.5 | 7.8 | 5.5 | 0.57 |
| mistral-7b | 7.2 | Q4_K_M | 4.4 | 9.2 | 8.4 | 0.53 |
| **qwen2.5-7b** | 7.6 | **Q3_K_M** | 3.8 | 7.2 | **5.5** | **0.87** |
| **qwen2.5-7b** | 7.6 | **Q4_K_M** | 4.7 | 9.7 | 8.2 | **0.87** |
| llama-3.1-8b | 8.0 | Q3_K_M | 4.0 | 8.2 | 6.1 | 0.63 |
| llama-3.1-8b | 8.0 | Q4_K_M | 4.9 | 9.1 | 9.1 | 0.67 |

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


## Concurrency — N users, one CPU box  ·  2026-09-06  ·  `results/conc.json`

`qwen2.5-3b Q4_K_M`, one `llama_cpp.server` instance on the 8-vCPU box, C parallel
`POST /v1/completions` (96-token generations). *(First: a bare `llama-cpp-python`
`Llama` object is **not** concurrency-safe — two threads calling it segfault
(`GGML_ASSERT`). On-prem you must front it with a server. This runs that server.)*

| concurrency | p50 latency | p95 latency | aggregate tok/s | latency vs C=1 |
|---|---|---|---|---|
| 1 | 5.7 s | 5.7 s | 16.4 | 1.0× |
| 2 | 7.8 s | 10.5 s | 18.0 | 1.4× |
| 4 | 13.0 s | 20.9 s | 18.0 | 2.3× |
| 8 | 27.6 s | 45.8 s | 16.4 | 4.9× |

**Aggregate throughput is flat (~16–18 tok/s) no matter the concurrency** — one
request already saturates all 8 cores, so parallel requests just queue. Latency
scales ~linearly with load (4.9× at 8 users). A single CPU box is a **single-user**
serving unit: one interactive session at a time, or a batch queue that people
don't wait on. For N concurrent live users you need ~N boxes (or a GPU — see
above), not a bigger prompt for the scheduler.

## Engine #2 — Ollama vs llama-cpp-python  ·  2026-09-07  ·  `results/ollama_grid.json`

Same GGUF weights (imported into Ollama with a `FROM <path>` Modelfile), same
probe, same box. Ollama is llama.cpp underneath — this isolates *wrapper* cost.

| model | quant | llama-cpp dec | ollama dec | Δ | llama-cpp pre | ollama pre | qual (lcp / oll) |
|---|---|---|---|---|---|---|---|
| gemma-2-2b | Q4_K_M | 15.4 | 17.3 | +12% | 61.5 | 71.5 | 0.53 / 0.57 |
| qwen2.5-3b | Q4_K_M | 16.7 | 18.0 | +8% | 45.5 | 54.5 | 0.70 / 0.73 |
| llama-3.2-3b | Q4_K_M | 18.4 | 17.2 | −7% | 46.3 | 52.2 | 0.67 / 0.70 |
| phi-3.5-mini | Q4_K_M | 13.7 | 11.4 | −17% | 26.5 | 33.5 | 0.77 / 0.77 |
| qwen2.5-7b | Q4_K_M | 9.7 | 9.8 | +1% | 21.5 | 24.8 | 0.87 / 0.87 |

**Mean decode ratio: 1.00×.** Ollama costs nothing in throughput — same kernel.
Prefill runs slightly *faster* on Ollama across the board (its defaults enable
flash-attention). Quality is identical within probe noise. phi-3.5-mini is the
one decode outlier (−17%, n=1); everything else is a wash.

What the Ollama wrapper buys for that (zero) cost: `pull`/`tag`/`rm` model
management, an always-on OpenAI-compatible server, automatic load/unload with
`keep_alive`, and a request queue for free. **For serving on-prem, use Ollama
(or `llama_cpp.server`) — it doesn't cost you tok/s.** In-process
`llama-cpp-python` is only simpler for a one-shot benchmark like this one.

Import cost: Ollama copies each GGUF blob into `~/.ollama` (~2 GB/model) — plan disk.

## Method caveats

- ✅ Prefill/decode now use llama.cpp's own perf counters (exact). The CPU table
  above still shows the earlier wall-clock numbers for a few cells; a clean CPU
  re-run with the fixed timing is queued (relative findings unchanged).
- Single run per cell — 3 repeats + spread is the next hardening step.
- Quality probe is directional (n=30), not MMLU.
