# NEXT — execution checklist (10 working days)

Commit: 2h/day, no pivot, re-evaluate day 11 with the artifact.

## Done
- [x] Harness: models / bench / quality / run / plot
- [x] Run 1 — 3B class × Q4/Q8 (8 cells). `RESULTS.md`, `results/latest.png`.
      Finding: **Q4_K_M ≥ Q8_0 on quality, every model, while faster + lighter.**

## In flight
- [~] Run 2 — qwen2.5-3b / phi-3.5-mini / mistral-7b / qwen2.5-7b / llama-3.1-8b
      × Q3_K_M / Q4_K_M. Answers: *does 7B beat 3B enough to justify ~2× RAM and
      ~½ speed on CPU? where is the quant cliff (does Q3 break things)?*
      Running in background (~2–3 h). → `results/run2.log`

## Days 1–3 — the benchmark, hardened
- [ ] Fold Run 2 into `RESULTS.md`; regenerate the Pareto plot with all points.
- [ ] Methodology fixes: use llama.cpp's own timing counters for prefill;
      3 repeats per cell → report median + spread.
- [ ] Add `file_gb` for the models missing it in the table.

## Days 3–6 — the on-prem reality
- [ ] **Concurrency**: 1 / 2 / 4 parallel requests → throughput vs p50/p95 latency.
      This is the real question for a shared box. (`bench.py` → an async/threaded mode.)
- [ ] **Realistic workloads**: a long-context RAG prompt (~3k tokens) vs a short
      chat prompt. Prefill dominates RAG; decode dominates chat.
- [ ] **Engine #2 — Ollama** *(needs Dante: `curl -fsSL https://ollama.com/install.sh | sudo sh`)*.
      Same grid via the Ollama API → wrapper overhead? model-management value?

## Days 7–10 — the deliverable
- [ ] **The decision guide**: a lookup — "box with N cores / M GB RAM, use-case
      {chat, RAG, batch, agents} → model + quant + engine + expected tok/s".
- [ ] **Writeup** (ES + EN): the numbers, the Q4-vs-Q8 finding, the guide.
      Target: blog nostalgiasistemas.com + LinkedIn + X.
- [ ] **Push to GitHub** (`chrono-glitch` or `nostalgiasistemas` — decide).
- [ ] Model card / repo README polish.

## Stretch (only if days 1–10 land)
- Follow-on project: "the inference server done right" (auth, proxy, metrics,
  systemd/compose) → then the "reference architecture for on-prem enterprise AI"
  capstone.
- OSS: a fix or eval contribution to `ollama/ollama` or `abetlen/llama-cpp-python`
  off the back of what the benchmark surfaces.

## What Dante does (outward / manual)
1. `curl -fsSL https://ollama.com/install.sh | sudo sh` — for the engine #2 comparison.
2. Decide the GitHub org for the push.
3. Read `RESULTS.md` after each run and sanity-check the findings against intuition.
