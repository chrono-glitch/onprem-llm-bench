"""Orchestrate the grid: for each (model, quant), run `llmbench.bench` in a
fresh subprocess (isolated RAM), collect the JSON, print + save the table.

    python -m llmbench.run --models gemma-2-2b,llama-3.2-3b,qwen2.5-3b \
        --quants Q4_K_M,Q8_0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from . import models


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="gemma-2-2b,llama-3.2-3b,qwen2.5-3b")
    p.add_argument("--quants", default="Q4_K_M,Q8_0")
    p.add_argument("--threads", type=int)
    p.add_argument("--n-gpu-layers", type=int, default=0, help="0=CPU, -1=all on GPU")
    p.add_argument("--perf-repeats", type=int, default=1)
    p.add_argument("--out", default="results")
    a = p.parse_args()

    mods = a.models.split(",")
    quants = a.quants.split(",")
    outdir = Path(a.out)
    outdir.mkdir(exist_ok=True)
    tmp = outdir / "_cell.json"

    rows = []
    for mk in mods:
        for q in quants:
            print(f"  {mk:14} {q:8} ...", end=" ", flush=True)
            cmd = [sys.executable, "-m", "llmbench.bench", "--model", mk,
                   "--quant", q, "--json", str(tmp),
                   "--n-gpu-layers", str(a.n_gpu_layers),
                   "--perf-repeats", str(a.perf_repeats)]
            if a.threads:
                cmd += ["--threads", str(a.threads)]
            t0 = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0 or not tmp.exists():
                print(f"FAILED ({r.stderr.strip().splitlines()[-1:] or 'no output'})")
                continue
            row = json.loads(tmp.read_text())
            tmp.unlink()
            rows.append(row)
            sp = row.get("decode_spread")
            sp_s = f" ({sp[0]}–{sp[1]})" if sp else ""
            print(f"decode {row['decode_tok_s']:>5} tok/s{sp_s} | ram {row['peak_ram_gb']:>5} GB "
                  f"| quality {row['quality']:.2f}  ({time.time() - t0:.0f}s)")

    if not rows:
        print("no results")
        return
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (outdir / f"grid_{stamp}.json").write_text(json.dumps(rows, indent=2))
    (outdir / "latest.json").write_text(json.dumps(rows, indent=2))

    print("\n=== on-prem LLM serving, this box (8 vCPU AMD EPYC, no GPU) ===")
    hdr = f"{'model':<14}{'quant':<8}{'GB':>6}{'load_s':>8}{'prefill':>9}{'decode':>8}{'RAM_GB':>8}{'qual':>7}"
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: (x["model"], x["quant"])):
        print(f"{r['model']:<14}{r['quant']:<8}{r['file_gb']:>6}{r['load_s']:>8}"
              f"{r['prefill_tok_s']:>9}{r['decode_tok_s']:>8}{r['peak_ram_gb']:>8}{r['quality']:>7.2f}")
    print(f"\nsaved: {outdir}/grid_{stamp}.json")


if __name__ == "__main__":
    main()
