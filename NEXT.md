# NEXT — execution checklist (10 working days)

Commit: 2h/day, no pivot, re-evaluate day 11 with the artifact.

## Done
- [x] Harness: models / bench / quality / run / plot
- [x] Run 1 + Run 2 folded → CPU grid, 16 cells, `RESULTS.md` + `results/cpu_all.png`.
      Finding: **Q4_K_M ≥ Q8_0 on quality, every model, while faster + lighter.**
      qwen2.5-7b is a quality tier above the whole 3B class (0.87 vs ~0.70).

## Days 1–3 — the benchmark, hardened
- [x] Prefill/decode use llama.cpp's own perf counters.
- [x] **Throughput repeats → median + spread + CV.** `bench.py --perf-repeats N
      / --perf-only`. Noise pass on 4 cells (5× each): decode CV 1.5–11 %
      (scales with model size). **Contention finding:** under multi-tenant load
      throughput fell ~40 %; qwen2.5-3b degraded most gracefully.
      → `results/stability.json`, folded into RESULTS.md + DECISION.md + writeups.
- [x] Headline grid kept as the single-tenant reference (`cpu_all.png` still valid).

## GPU track (Kaggle T4 / Colab) — bigger models
- [x] Harness is device-aware: `bench.run(..., n_gpu_layers=-1)`, `run.py --n-gpu-layers`.
- [x] `kaggle/kernel.py` + `kaggle/drive.sh` — push a GPU notebook, poll, pull `gpu_grid.json`.
- [ ] **BLOCKED: need a full Kaggle API token.** The `KGAT_`-prefixed token in
      `~/.kaggle/kaggle.json` reads datasets but **can't push kernels** (401).
      Dante: Kaggle → Account → Settings → API → "Create New API Token" → replace
      `~/.kaggle/kaggle.json`. (Or run the notebook by hand on Colab/Kaggle web.)
- [ ] Once unblocked: grid qwen2.5-7b / llama-3.1-8b / **qwen2.5-14b / qwen2.5-32b /
      gemma-2-27b** on T4 → the "is a GPU worth it, and for which models" answer.
- [ ] The headline comparison: **same models, CPU (this box) vs T4** — decode
      tok/s, cost/1M tokens, quality.

## Days 3–6 — the on-prem reality
- [x] **Concurrency**: 1/2/4/8 parallel → `llmbench.conc` (spawns `llama_cpp.server`).
      Finding: **aggregate tok/s is flat (~16–18); a CPU box is a single-user unit.**
      Latency 4.9× at C=8. Also proved bare `Llama` is not concurrency-safe (segfault).
      → `results/conc.json`, folded into RESULTS.md + both writeups.
- [ ] **Realistic workloads**: a long-context RAG prompt (~3k tokens) vs a short
      chat prompt. Prefill dominates RAG; decode dominates chat.
- [x] **Engine #2 — Ollama** (`llmbench.ollama_bench`, `llmbench.engines` compare).
      5 cells, Q4_K_M. Finding: **mean decode ratio 1.00× — the wrapper is free**
      (same llama.cpp kernel; Ollama prefill even slightly faster). Use Ollama /
      `llama_cpp.server` for serving; in-process lcp only simpler for a one-shot.
      → `results/ollama_grid.json`, folded into RESULTS.md + both writeups.

## Days 7–10 — the deliverable
- [x] **The decision guide** → `DECISION.md` (box × use-case → model+quant+engine).
- [x] **Writeups** (ES + EN) — finished, GPU section integrated, ES/EN cross-linked.
- [x] **README** rewritten — headline findings, run commands, layout, caveats.
- [x] **LICENSE** (MIT), `requirements.txt`, `IDEAS.md`, repo cleanup.
- [x] **PUBLIC: https://github.com/chrono-glitch/onprem-llm-bench** (2026-09-08, 23 commits).
- [ ] Post the writeup — blog nostalgiasistemas.com + LinkedIn + X. *(Dante)*
- [ ] Optional: transfer/mirror the repo to `nostalgiasistemas` once that account
      has auth or is converted to an org.

## Stretch (only if days 1–10 land)
- Follow-on project: "the inference server done right" (auth, proxy, metrics,
  systemd/compose) → then the "reference architecture for on-prem enterprise AI"
  capstone.
- OSS: a fix or eval contribution to `ollama/ollama` or `abetlen/llama-cpp-python`
  off the back of what the benchmark surfaces.

## What Dante does (outward / manual)
1. Decide the GitHub org for the push (`chrono-glitch` or `nostalgiasistemas`).
2. Read `RESULTS.md` + `DECISION.md` and sanity-check the findings against intuition.
3. Publish the writeup (blog + LinkedIn + X) once the repo is public.

## Still open (smaller)
- Realistic workloads: a ~3k-token RAG prompt vs a short chat prompt
  (prefill-dominated vs decode-dominated) — partly covered by the prefill numbers.
- Optional: DECISION.md as a shareable HTML artifact for the portfolio.
