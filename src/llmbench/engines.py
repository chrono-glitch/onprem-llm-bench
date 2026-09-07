"""Engine comparison: llama-cpp-python (in-process) vs Ollama (HTTP wrapper),
same models, same GGUF weights, same probe.

    python -m llmbench.engines results/cpu_all.json results/ollama_grid.json

Both run llama.cpp underneath, so this isolates the *wrapper* cost: HTTP, the
Ollama scheduler, its KV-cache handling — against what the wrapper buys you
(model management, an always-on OpenAI-compatible server, auto load/unload).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(p: str) -> dict:
    return {(r["model"], r["quant"]): r for r in json.loads(Path(p).read_text())}


def main(lcp_path: str, oll_path: str) -> None:
    lcp, oll = load(lcp_path), load(oll_path)
    keys = sorted(set(lcp) & set(oll),
                  key=lambda k: (lcp[k]["params_b"], *k))

    hdr = (f"{'model':<15}{'quant':<8}"
           f"{'lcp dec':>9}{'oll dec':>9}{'Δ dec':>8}"
           f"{'lcp pre':>9}{'oll pre':>9}"
           f"{'lcp q':>7}{'oll q':>7}")
    print(hdr)
    print("-" * len(hdr))
    d_ratios = []
    for k in keys:
        a, b = lcp[k], oll[k]
        ad, bd = a["decode_tok_s"], b["decode_tok_s"]
        delta = f"{(bd / ad - 1) * 100:+.0f}%" if ad and bd else "-"
        if ad and bd:
            d_ratios.append(bd / ad)
        print(f"{k[0]:<15}{k[1]:<8}"
              f"{ad:>9}{bd:>9}{delta:>8}"
              f"{a['prefill_tok_s'] or '-':>9}{b['prefill_tok_s'] or '-':>9}"
              f"{a['quality']:>7.2f}{b['quality']:>7.2f}")

    if d_ratios:
        avg = sum(d_ratios) / len(d_ratios)
        print(f"\nmean decode ratio (ollama / llama-cpp-python): {avg:.2f}×")
        print("→ same kernel; any gap is wrapper overhead / default-setting drift,"
              " not a faster engine.")


if __name__ == "__main__":
    a = sys.argv[1:] or ["results/cpu_all.json", "results/ollama_grid.json"]
    main(a[0], a[1])
