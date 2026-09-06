"""Pareto plot from a grid JSON: quality vs decode tok/s, bubble = peak RAM.

    python -m llmbench.plot results/latest.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.loads(Path(path).read_text())
    fig, ax = plt.subplots(figsize=(9, 6))
    for r in rows:
        x, y = r["decode_tok_s"], r["quality"]
        ax.scatter(x, y, s=r["peak_ram_gb"] * 120, alpha=0.5,
                   edgecolors="black", linewidths=0.6)
        ax.annotate(f"{r['model']}\n{r['quant']}", (x, y),
                    fontsize=8, ha="center", va="center")
    ax.set(xlabel="decode throughput (tok/s, higher better)",
           ylabel="quality probe (higher better)",
           title="On-prem LLM serving, CPU only — bubble size = peak RAM")
    ax.grid(alpha=0.3)
    out = Path(path).with_suffix(".png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/latest.json")
