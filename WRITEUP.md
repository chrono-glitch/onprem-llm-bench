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

## Caveats

30-item quality probe (directional, not a real eval). Single-request only — the
concurrency curve (2–4 parallel requests, the real shared-box question) is the
next section. Prefill/decode numbers use llama.cpp's own perf counters.

---

*ES: versión en español pendiente — "Servir LLMs on-prem sin GPU: los números y
cómo elegir".*
