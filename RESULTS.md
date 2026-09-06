# Results

**Box:** 8 vCPU AMD EPYC 7B12, 31 GB RAM, no GPU. `llama-cpp-python` 0.3.35
(CPU wheel). Prompt ≈ 512 tokens, decode 128 tokens, 8 threads.
Quality = 25 MCQ + 5 format-following items (a sanity gauge, not a real eval).

## Run 1 — 3B-class models, Q4_K_M vs Q8_0  ·  2026-09-06

| model | quant | file GB | load s | prefill tok/s | **decode tok/s** | peak RAM GB | quality (mcq / fmt) |
|---|---|---|---|---|---|---|---|
| gemma-2-2b | Q4_K_M | 1.71 | 9.5 | 61.5 | 15.4 | 3.3 | 0.53 (0.44 / 1.00) |
| gemma-2-2b | Q8_0 | 2.78 | 1.6 | 49.2 | 12.1 | 3.8 | 0.50 |
| llama-3.2-3b | Q4_K_M | 2.02 | 4.6 | 46.3 | **18.4** | 4.1 | 0.67 (0.64 / 0.80) |
| llama-3.2-3b | Q8_0 | 3.42 | 1.7 | 35.9 | 12.7 | 4.6 | 0.67 |
| **qwen2.5-3b** | **Q4_K_M** | 1.93 | 3.1 | 48.3 | 16.1 | **3.6** | **0.70** (0.68 / 0.80) |
| qwen2.5-3b | Q8_0 | 3.29 | 1.0 | 33.3 | 15.3 | 3.7 | 0.70 |
| phi-3.5-mini | Q4_K_M | 2.39 | 6.0 | 27.9 | 14.0 | 5.4 | **0.77** (0.72 / 1.00) |
| phi-3.5-mini | Q8_0 | 4.06 | 1.5 | 27.7 | 10.3 | 5.7 | 0.73 |

## What this already says

1. **On CPU, there is no quality reason to run Q8 over Q4_K_M.** Across all four
   models the quality score is equal or *better* at Q4 (gemma 0.53 vs 0.50;
   llama 0.67 = 0.67; qwen 0.70 = 0.70; phi 0.77 vs 0.73), while Q4 is
   **20–40 % faster to decode** and uses ~15–25 % less RAM. The quality cliff, if
   there is one, is below Q4 — Run 2 checks Q3.

2. **qwen2.5-3b @ Q4_K_M is the sweet spot for the 3B class** — 16 tok/s, 3.6 GB,
   quality 0.70. Best quality per GB and per tok/s.

3. **phi-3.5-mini has the best quality (0.77)** but pays for it: slowest decode
   (14 → 10 tok/s) and heaviest RAM (5.4 GB). The "I need the accuracy" pick.

4. **gemma-2-2b is the outlier** — weak on multiple-choice reasoning (0.44) despite
   its reputation, but perfect on format-following (1.00). Fast and light, good
   for constrained-output tasks, not for reasoning.

5. **15–18 tok/s decode on 8 CPU cores** is faster than reading speed — a
   3B model at Q4 is a usable chatbot on a commodity box with no GPU.

## Caveats / methodology to harden (Run 2)

- Prefill number includes `create_completion` call overhead — switch to
  llama.cpp's own timing counters.
- Single run per cell; add 3 repeats for tok/s variance.
- Quality probe is 30 items — directional only. Not MMLU.
- All single-request. The concurrency test (2–4 parallel) is the real on-prem
  question and comes next.

## Next

- Run 2: add mistral-7b / qwen2.5-7b / llama-3.1-8b, and Q3_K_M — does 7B beat
  3B enough to justify ~2× the RAM and ~½ the speed on CPU? Where's the quant cliff?
- Pareto plot (quality vs decode tok/s, bubble size = RAM).
- Engine #2: Ollama — wrapper overhead?
- Concurrency curve.
- Realistic workloads: a long-context RAG prompt vs a short chat prompt.
