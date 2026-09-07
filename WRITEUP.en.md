# Serving LLMs on-prem with no GPU: the numbers, and how to choose

*Draft — CPU track. The GPU (Kaggle T4/P100) comparison is a follow-up section.*

## The question

A business wants an LLM running on their own hardware — data can't leave the
building, or a cloud bill per token doesn't fit the model. They have a normal
server: 8 CPU cores, ~30 GB RAM, **no GPU**. Which open model, at which
quantization, actually works there — and is it good enough to ship?

I benchmarked 7 instruct models (2–8 B) at 2–3 GGUF quant levels each on exactly
that box (8 vCPU AMD EPYC 7B12, `llama-cpp-python`), measuring decode throughput,
RAM, load time, and a 30-item quality probe (multiple-choice reasoning +
format-following — a sanity gauge, not MMLU).

## The three findings that matter

### 1. On CPU, `Q4_K_M` is the right quantization — full stop

Every model scored **equal or better on quality at Q4_K_M than at Q8_0**, while
running 20–40 % faster and using less RAM. Q8 buys nothing on CPU. `Q3_K_M` is
the real floor: fine for some models (qwen2.5-7b: 0.87 at both Q3 and Q4), a
~4-point hit for others (llama), broke nothing.

### 2. There's a sharp quality step at 7 B — and one model owns it

The entire 3 B class clustered around **0.70** on the probe. `qwen2.5-7b` jumped
to **0.87** — and held it at Q3_K_M, which fits in **5.5 GB** and runs at ~7
tok/s on CPU. If the box has the RAM and the use case tolerates ~7 tok/s
(agents, batch, async), this is the pick. `mistral-7b` and `llama-3.1-8b` did
*not* make that jump — llama-3.1-8b scored 0.67, no better than the 3.2-3B at
2.5× the size.

### 3. For interactive chat, `qwen2.5-3b Q4_K_M` is the sweet spot

16.7 tok/s decode (faster than reading speed), 3.6 GB RAM, quality 0.70. It's
the fastest option that isn't compromised on quality, and it leaves most of a
30 GB box free for everything else.

## The decision guide

| use case | pick | why |
|---|---|---|
| Interactive chat / assistant | **qwen2.5-3b `Q4_K_M`** | 16 tok/s, 3.6 GB, quality 0.70 |
| Best quality, can spare RAM + tolerate ~7 tok/s | **qwen2.5-7b `Q3_K_M`** | 0.87 quality in 5.5 GB |
| Constrained / structured output only (JSON, single-word) | **gemma-2-2b `Q4_K_M`** | 1.00 on format-following, fast, tiny — but 0.44 on reasoning |
| Middle ground | **phi-3.5-mini `Q4_K_M`** | 0.77 quality, 13.7 tok/s, 5.4 GB |
| — | *avoid* llama-3.1-8b, mistral-7b | no quality gain over 3 B at 2× the cost |

Rules of thumb from the data:
- **Default to `Q4_K_M`.** Only drop to `Q3_K_M` when RAM is tight *and* you've
  checked the quality on your own prompts.
- **7 B roughly halves the speed of 3 B** on CPU (~9 vs ~17 tok/s).
- **Below ~10 tok/s, it's not a live chatbot** — it's a batch / agent worker.

## Full results

`results/cpu_all.json`, table in `RESULTS.md`, Pareto plot `results/cpu_all.png`.

## Concurrency: a CPU box is a single-user serving unit

An infra finding first: a `llama-cpp-python` `Llama` object is **not
concurrency-safe** — two threads calling it segfault. On-prem you must put a
server in front (`llama_cpp.server`), which queues requests.

With that server and `qwen2.5-3b Q4_K_M`, firing C requests in parallel:

| concurrency | p50 latency | p95 | aggregate tok/s | latency vs C=1 |
|---|---|---|---|---|
| 1 | 5.7 s | 5.7 s | 16.4 | 1.0× |
| 2 | 7.8 s | 10.5 s | 18.0 | 1.4× |
| 4 | 13.0 s | 20.9 s | 18.0 | 2.3× |
| 8 | 27.6 s | 45.8 s | 16.4 | 4.9× |

**Aggregate throughput is flat (~16–18 tok/s) regardless of concurrency** — one
request already saturates all 8 cores, so parallel requests just queue. Latency
scales ~linearly with load. A CPU box serves **one interactive user at a time**,
or a batch queue nobody waits on live. For N concurrent users: ~N boxes, or a GPU.

## Engine: Ollama vs llama-cpp-python — the wrapper is free

Both are llama.cpp underneath. I imported the same GGUF files into Ollama and ran
the same probe. **Mean decode ratio: 1.00×** — Ollama's HTTP layer and scheduler
cost nothing in throughput. Prefill is actually a touch faster on Ollama (its
defaults turn on flash-attention). Quality identical.

So the choice is about operations, not speed. Ollama gives you model management
(`pull`/`tag`/`rm`), an always-on OpenAI-compatible endpoint, auto load/unload,
and a request queue — for no tok/s penalty. **For serving on-prem, use Ollama or
`llama_cpp.server`.** Raw in-process `llama-cpp-python` is only simpler for a
one-shot benchmark. (Ollama does copy each GGUF into `~/.ollama`, ~2 GB/model.)

## Caveats

30-item quality probe (directional, not a real eval). Single run per cell — 3
repeats + spread is the next hardening step. Prefill/decode numbers use
llama.cpp's own perf counters.

---

*ES: versión en español pendiente — "Servir LLMs on-prem sin GPU: los números y
cómo elegir".*

## GPU (Tesla P100) — the break-even

Same models on a Kaggle P100: decode is **3.5–4× faster** than CPU (3B ~60 tok/s,
7–8B ~35 tok/s) and **prefill is 20–30× faster** (~800–1700 vs ~55 tok/s). But a
GPU rents for ~8× a CPU box. So **on pure cost per token, the CPU box wins.**

Get the GPU when: (1) a person is waiting on the output — 60 tok/s vs 17 is a
different product; (2) you need a model that won't run on CPU — `qwen2.5-14b Q4`
hits **0.93** quality at a usable 19 tok/s on the P100, vs ~3 tok/s (unusable) on
CPU; (3) RAG / long prompts — prefill dominates and the GPU is 20–30× ahead.
On GPU, `Q8_0` is fine — the "Q4 only" rule is CPU-specific.
