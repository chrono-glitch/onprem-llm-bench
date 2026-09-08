# X / Twitter thread (English)

> 8 tweets. Keep each ≤280 chars. Attach `results/cpu_all.png` to tweet 1.
> Post the repo link in the LAST tweet (or pin a reply) — links mid-thread hurt reach.

---

**1/**
A business wants an LLM on its own hardware. No data leaving the building. A
normal box: 8 CPU cores, 30GB RAM, no GPU.

Which model? Which quant? Good enough to ship? Is a GPU worth it?

I benchmarked it. 7 models, 2–8B, multiple GGUF quants. Findings 🧵

**2/**
On CPU, Q4_K_M is the right quant. Full stop.

Every model I tested scored equal-or-better on quality at Q4_K_M than Q8_0 —
while running 20–40% faster on less RAM.

Q8 buys you nothing on CPU. (On GPU it's fine.)

**3/**
There's a sharp quality step at 7B — and one model owns it.

The whole 3B class clustered at ~0.70 on my probe.
qwen2.5-7b jumped to 0.87, and held it at Q3_K_M (5.5GB, ~7 tok/s on CPU).

llama-3.1-8b and mistral-7b did NOT make that jump.

**4/**
For interactive chat: qwen2.5-3b Q4_K_M is the sweet spot.
16.7 tok/s (faster than reading), 3.6GB, leaves the box free for real work.

Below ~10 tok/s it's not a chatbot — it's a batch/agent worker.

**5/**
A CPU box is a single-user serving unit.

Fire 1/2/4/8 requests in parallel → aggregate throughput stays flat at ~16–18
tok/s. One request already saturates all 8 cores. p95 latency goes 4.9× at C=8.

N concurrent users = ~N boxes, or a GPU.

**6/**
GPU (Tesla P100) vs this CPU box:
- decode: 3.5–4× faster
- prefill: 20–30× faster
- rent: ~8× the cost

On $/token the CPU wins. Get the GPU for latency, RAG prefill, or 14B+ models
(qwen2.5-14b Q4 = 0.93 quality at a usable 19 tok/s).

**7/**
Ollama vs in-process llama-cpp-python: mean decode ratio 1.00×.

The wrapper is free. Same llama.cpp kernel underneath. Pick your engine for ops
(model mgmt, an always-on endpoint, a request queue), not for speed.

**8/**
Full code, every number, and a decision guide (your box + use case → model +
quant + engine):

https://github.com/chrono-glitch/onprem-llm-bench

If you run on-prem inference and a number looks off vs your hardware, tell me.
