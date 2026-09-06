"""Kaggle kernel body — runs the llmbench grid on the notebook's GPU (T4 free tier).

Push with `kaggle/drive.sh`. Enable: GPU + Internet. Output lands in
/kaggle/working/gpu_grid.json, pulled back by the driver.
"""
import json, os, subprocess, sys, time

# T4 = 16GB VRAM / ~20GB disk. Q4 of 14B ~9GB fits fully; 32B Q4 ~19GB blows the
# disk quota, so it's opt-in via LLMBENCH_MODELS.
MODELS = os.environ.get("LLMBENCH_MODELS",
    "gemma-2-2b,qwen2.5-3b,llama-3.2-3b,qwen2.5-7b,llama-3.1-8b,qwen2.5-14b").split(",")
QUANTS = os.environ.get("LLMBENCH_QUANTS", "Q4_K_M,Q8_0").split(",")

# CUDA wheel for llama-cpp-python (matches Kaggle's CUDA 12.x)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "llama-cpp-python",
    "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu124"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)

# vendor the harness (kernel source includes src/llmbench/)
sys.path.insert(0, "src")
from llmbench import bench, models  # noqa: E402

os.system("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")

rows = []
for mk in MODELS:
    for q in QUANTS:
        pb = models.REGISTRY[mk].params_b
        if pb >= 30 and q != "Q3_K_M":      # only Q3 of 30B+ fits a T4
            continue
        if pb >= 13 and q == "Q8_0":        # 14B Q8 ~16GB, too tight
            continue
        print(f"=== {mk} {q} ===", flush=True)
        t0 = time.time()
        try:
            r = bench.run(mk, q, n_gpu_layers=-1)   # all layers on GPU
            r["wall_s"] = round(time.time() - t0, 1)
            rows.append(r)
            print(json.dumps(r), flush=True)
        except Exception as e:
            print(f"FAILED {mk} {q}: {type(e).__name__}: {e}", flush=True)

with open("/kaggle/working/gpu_grid.json", "w") as f:
    json.dump(rows, f, indent=2)
print(f"\nwrote {len(rows)} cells")
