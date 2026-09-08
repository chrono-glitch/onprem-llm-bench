# The decision guide

*You have a box. You have a use case. What do you run?*

All numbers are from this benchmark (`RESULTS.md`) — an 8-vCPU AMD EPYC 7B12 CPU
box with no GPU, and a Kaggle Tesla P100 for the GPU column. Quality is the
30-item probe (directional, not MMLU). "tok/s" is decode throughput unless noted.

---

## 1. Start here — pick your row

| Your box | RAM budget for the model | What runs well | Skip |
|---|---|---|---|
| **Small CPU** (2–4 cores, ≤8 GB) | ~3 GB | `qwen2.5-3b Q4_K_M` — that's the whole menu | anything 7B |
| **Standard CPU** (8 vCPU, 16–32 GB) | 4–9 GB | 3B class interactive; `qwen2.5-7b Q3/Q4` for non-interactive | `llama-3.1-8b`, `mistral-7b` (no quality gain) |
| **Big CPU** (16+ cores, 64 GB) | 5–9 GB | `qwen2.5-7b Q4_K_M` becomes borderline-interactive (~15–18 tok/s expected); 14B still too slow | Q8 anything |
| **CPU + one GPU** (T4/L4/P100, 16 GB) | ≤10 GB VRAM | `qwen2.5-14b Q4_K_M` (0.93 quality, 19 tok/s) — the top pick; everything smaller is snappy | CPU-only inference for anything a person waits on |

The CPU box is a **single-user serving unit** — aggregate throughput is flat
regardless of concurrency (one request saturates all cores). For N concurrent
live users you need ~N boxes, or a GPU.

---

## 2. Then pick your use case

| Use case | What dominates | Pick (CPU box) | Pick (with GPU) |
|---|---|---|---|
| **Interactive chat / assistant** | decode tok/s; a person is waiting | **`qwen2.5-3b Q4_K_M`** — 16.7 tok/s, 3.6 GB, quality 0.70 | `qwen2.5-14b Q4_K_M` — 0.93 at 19 tok/s |
| **Best answer quality, latency tolerant** | quality | **`qwen2.5-7b Q3_K_M`** — 0.87 in 5.5 GB at ~7 tok/s | `qwen2.5-14b Q4_K_M` — 0.93 |
| **RAG / long context** (3k+ token prompts) | *prefill* tok/s | `qwen2.5-3b Q4_K_M` (prefill ~46 tok/s → a 3k prompt ≈ 65 s stall) — or get a GPU | **any** — GPU prefill is 20–30× (a 3k prompt ≈ 2–4 s) |
| **Batch / offline** (summarize a queue overnight) | total throughput, not latency | **`qwen2.5-7b Q4_K_M`** — 0.87, 9.7 tok/s, run it unattended | `qwen2.5-14b` if the queue is huge |
| **Agents** (many short LLM calls in a loop) | decode + low per-call overhead | `qwen2.5-3b Q4_K_M` via a persistent server (Ollama / `llama_cpp.server`) | 3B on GPU — the loop finishes 3–4× sooner |
| **Structured output only** (JSON, one word, classification) | format adherence | **`gemma-2-2b Q4_K_M`** — 1.00 on format, 15.4 tok/s, 3.3 GB (but 0.44 reasoning — don't ask it to think) | same; GPU if volume is high |

---

## 3. Fixed rules from the data

- **On CPU, default to `Q4_K_M`.** Every model scored equal-or-better at Q4 than
  Q8, 20–40 % faster, less RAM. Q8 buys nothing on CPU. On **GPU**, Q8 is fine
  (sometimes faster) — the "Q4 only" rule is CPU-specific.
- **`Q3_K_M` is the real floor.** Costs `qwen2.5-7b` nothing (0.87 = 0.87),
  `llama` ~4 points. Drop to it only when RAM is tight *and* you've spot-checked
  quality on your own prompts.
- **7B ≈ half the decode speed of 3B on CPU** (~9 vs ~17 tok/s).
- **Below ~10 tok/s is not a live chatbot** — it's a batch/agent worker.
- **The engine doesn't change throughput.** Ollama vs in-process
  `llama-cpp-python`: mean decode ratio 1.00×. Choose the engine for operations
  (Ollama = model management + always-on OpenAI-compatible server + auto
  load/unload), not speed.
- **Is a GPU worth it?** On pure $/token, no — a GPU rents for ~8× a CPU box and
  is only ~3.5–4× faster at decode. Get one for: latency (a person is waiting),
  RAG (prefill is 20–30× faster), or models that don't fit CPU at usable speed
  (`qwen2.5-14b` and up).

---

## 4. Sizing on a shared box

The headline tok/s are a **single-tenant ceiling**. Re-measured under real
multi-tenant load, throughput dropped ~40 % and run-to-run variance rose to
±10 % for 7–8B models (`qwen2.5-3b` held to ±1.5 %). If the box does anything
else: **size for ~0.6× the table, leave ±10 % headroom, prefer `qwen2.5-3b`.**

## 5. The short answer

> **Most on-prem CPU deployments should run `qwen2.5-3b Q4_K_M` for anything
> interactive and `qwen2.5-7b Q4_K_M` for anything batch, both behind an Ollama
> server. Add a 16 GB GPU only when you need sub-second RAG, a 14B-class model,
> or more than one concurrent user.**
