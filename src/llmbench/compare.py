"""CPU vs GPU comparison + cost model.

    python -m llmbench.compare results/cpu_all.json results/gpu_grid.json

Produces:
- a merged table (same model+quant on CPU vs GPU: decode tok/s, quality, RAM)
- $ per 1M output tokens for each, using the rates below
- the break-even note: when the GPU's speed is worth its rent

Rates are editable; defaults are rough 2026 street prices.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# $/hour. CPU = amortised commodity box; GPU = cheapest reliable T4/L4 rental.
RATE_CPU_HR = 0.05      # an 8-vCPU box, ~$35/mo
RATE_GPU_HR = 0.40      # L4 / on-demand T4 street price
KAGGLE_GPU_HR = 0.0     # free tier, for the "if you can use Kaggle" column


def _cost_per_1m(tok_s: float, rate_hr: float) -> float:
    if not tok_s:
        return float("inf")
    return round(rate_hr / (tok_s * 3600) * 1e6, 3)


def load(p: str) -> dict:
    return {(r["model"], r["quant"]): r for r in json.loads(Path(p).read_text())}


def main(cpu_path: str, gpu_path: str):
    cpu, gpu = load(cpu_path), load(gpu_path)
    keys = sorted(set(cpu) | set(gpu), key=lambda k: (cpu.get(k, gpu[k])["params_b"], *k))

    hdr = (f"{'model':<14}{'quant':<8}"
           f"{'CPU tok/s':>10}{'GPU tok/s':>10}{'speedup':>9}"
           f"{'qual':>6}"
           f"{'$/1M CPU':>10}{'$/1M GPU':>10}")
    print(hdr)
    print("-" * len(hdr))
    for k in keys:
        c, g = cpu.get(k), gpu.get(k)
        ct = c["decode_tok_s"] if c else None
        gt = g["decode_tok_s"] if g else None
        speed = f"{gt / ct:.1f}x" if (ct and gt) else "-"
        qual = (g or c)["quality"]
        cc = _cost_per_1m(ct, RATE_CPU_HR) if ct else "-"
        gc = _cost_per_1m(gt, RATE_GPU_HR) if gt else "-"
        print(f"{k[0]:<14}{k[1]:<8}"
              f"{ct if ct else '-':>10}{gt if gt else '-':>10}{speed:>9}"
              f"{qual:>6.2f}{str(cc):>10}{str(gc):>10}")

    print(f"\nrates: CPU ${RATE_CPU_HR}/h, GPU ${RATE_GPU_HR}/h  (edit in compare.py)")
    print("break-even: GPU beats CPU on $/token only when its speedup > "
          f"{RATE_GPU_HR / RATE_CPU_HR:.0f}x (the rent ratio).")


if __name__ == "__main__":
    a = sys.argv[1:] or ["results/cpu_all.json", "results/gpu_grid.json"]
    main(a[0], a[1])
