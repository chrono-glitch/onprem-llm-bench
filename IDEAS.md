# IDEAS — the parking lot

New ideas go here, not into a new repo. Rule: nothing new gets started until
`onprem-llm-bench` is public and the writeup is posted. If an idea is real, it
survives two weeks in this file.

## Sanctioned follow-on (one, in order)
- **The inference server, done right** — Ollama + a thin auth/metrics proxy +
  `compose.yml` + n8n wiring. Ships this benchmark's conclusion as running code;
  becomes the Nostalgia OS on-prem AI module.

## Parked
- **Nostalgia OS, in Rust** (raised 2026-09-08). Next portfolio project after
  onprem-llm-bench ships. Reuse the "hermes" API keys for cloud free tiers.
  UNSCOPED — "an OS" is the most abstract direction yet (app → platform → OS is
  exactly the escalation pattern). Before starting: pin it to ONE concrete,
  finishable Rust artifact with a demo, not a platform. Candidate framings to
  pick from when it starts: a single on-prem inference-server daemon (auth +
  metrics + model mgmt, the sanctioned follow-on above, written in Rust instead
  of a Python proxy); or a CLI tool; or one systemd-managed service. Decide the
  10-day artifact first, then the name.
- Realistic-workload split: a ~3k-token RAG prompt vs a short chat prompt
  (prefill-dominated vs decode-dominated). Partly covered by the prefill numbers.
- `DECISION.md` as a shareable interactive HTML page for the portfolio.
- OSS contribution to `ollama/ollama` or `abetlen/llama-cpp-python` off something
  the benchmark surfaced (e.g. the `websockets`/`uvicorn` server startup bug).
- Quant cliff at Q2_K — does anything survive it?
- ARM / Apple Silicon column (M-series is a real on-prem target).
