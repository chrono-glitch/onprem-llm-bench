# IDEAS — the parking lot

New ideas go here, not into a new repo. Rule: nothing new gets started until
`onprem-llm-bench` is public and the writeup is posted. If an idea is real, it
survives two weeks in this file.

## Sanctioned follow-on (one, in order)
- **The inference server, done right** — Ollama + a thin auth/metrics proxy +
  `compose.yml` + n8n wiring. Ships this benchmark's conclusion as running code;
  becomes the Nostalgia OS on-prem AI module.

## Parked
- Realistic-workload split: a ~3k-token RAG prompt vs a short chat prompt
  (prefill-dominated vs decode-dominated). Partly covered by the prefill numbers.
- `DECISION.md` as a shareable interactive HTML page for the portfolio.
- OSS contribution to `ollama/ollama` or `abetlen/llama-cpp-python` off something
  the benchmark surfaced (e.g. the `websockets`/`uvicorn` server startup bug).
- Quant cliff at Q2_K — does anything survive it?
- ARM / Apple Silicon column (M-series is a real on-prem target).
