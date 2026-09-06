# onprem-llm-bench

**On-prem LLM serving without a GPU: what actually works, and how to choose.**

Benchmarks small instruct models (2–8B) at multiple GGUF quantization levels on
a plain CPU box — throughput (prefill / decode tok/s), RAM, load time, and a
small quality gauge — and turns it into a decision guide: *for a given box and
use case, which model + quant?*

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
