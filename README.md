# onprem-llm-bench

**On-prem / edge LLM serving: what actually works, and how to choose.**

Benchmarks instruct models (2–32B) at multiple GGUF quantization levels across
**CPU and single-GPU (T4)** — throughput (prefill / decode tok/s), RAM/VRAM,
load time, and a small quality gauge — and turns it into a decision guide:
*for a given box and use case, which model + quant + engine — and is a GPU worth it?*

- **CPU track:** this repo, runs anywhere. `python -m llmbench.run ...`
- **GPU track:** `kaggle/` — push a notebook to Kaggle's free T4, pull results.

## Run

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
export PYTHONPATH=$PWD/src
python -m llmbench.run --models gemma-2-2b,llama-3.2-3b,qwen2.5-3b --quants Q4_K_M,Q8_0
```

## Layout

```
src/llmbench/
  models.py   registry + GGUF download (bartowski/*)
  bench.py    one (model, quant) in an isolated process: load / prefill / decode / RAM / quality
  quality.py  ~30-item sanity probe (MCQ + format-following)
  run.py      the grid -> results table
```

Machine for the published numbers: 8 vCPU AMD EPYC 7B12, 31 GB RAM, no GPU.
