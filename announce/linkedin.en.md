# LinkedIn — English post (optional second post, or for an EN-first audience)

> Same rules: image uploaded not linked, repo link in the first comment.

---

A business wants an LLM on its own server. Data can't leave the building, or a
per-token cloud bill doesn't fit. They have a normal box: 8 CPU cores, 30 GB RAM,
**no GPU**.

Which model runs there? At which quantization? Is it good enough to ship? And is
a GPU worth buying?

I couldn't find those numbers measured in one place, so I measured them. 7
instruct models (2–8B), 2–3 GGUF quant levels each, on that exact box. Decode
throughput, RAM, load time, concurrency, run-to-run noise, two engines (llama.cpp
vs Ollama), and a comparison against a GPU (Tesla P100).

Three findings that changed how I'd size these:

**1. On CPU, Q4_K_M is the right quant — full stop.** Every model scored
equal-or-better on quality at Q4_K_M than at Q8_0, running 20–40% faster with
less RAM. Q8 buys nothing on CPU. (On GPU it's fine.)

**2. There's a sharp quality step at 7B, and one model owns it.** The entire 3B
class clusters around 0.70 on my probe. qwen2.5-7b jumps to 0.87 — and holds it
at Q3_K_M (5.5 GB). llama-3.1-8b and mistral-7b do NOT make that jump: 2× the
size, no quality gain.

**3. A CPU box is a single-user serving unit.** Aggregate throughput is flat
regardless of concurrency — one request already saturates every core. For N
concurrent live users: ~N boxes, or a GPU. The GPU loses on cost per token (≈8×
the rent, ≈4× the speed) but wins on latency, RAG prefill (20–30×), and
14B-class models CPU can't run usefully.

Code, the full table, and a decision guide (your box + your use case → model,
quant, engine) are in the repo. Link in the first comment.

If you run on-prem inference and a number doesn't match what you see on your
hardware, I want to hear about it.

#LLM #MLOps #OnPremInference #EdgeAI #Ollama #llamacpp

---

## First comment

Repo: https://github.com/chrono-glitch/onprem-llm-bench
Writeup: https://github.com/chrono-glitch/onprem-llm-bench/blob/main/WRITEUP.en.md
