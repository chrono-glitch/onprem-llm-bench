#!/usr/bin/env bash
# Push the GPU benchmark to a Kaggle notebook, wait, pull the results.
#
# Needs a FULL Kaggle API token in ~/.kaggle/kaggle.json (Account -> Settings ->
# API -> "Create New API Token"). The KGAT_-prefixed scoped token can read
# datasets but CANNOT push kernels (401).
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE=$(mktemp -d)
mkdir -p "$STAGE/src"
cp -r src/llmbench "$STAGE/src/"
cp kaggle/kernel.py "$STAGE/main.py"

USER=$(python3 -c "import json;print(json.load(open('$HOME/.kaggle/kaggle.json'))['username'])")
cat > "$STAGE/kernel-metadata.json" <<EOF
{
  "id": "$USER/onprem-llm-bench-gpu",
  "title": "onprem-llm-bench-gpu",
  "code_file": "main.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true
}
EOF

echo "pushing kernel..."
kaggle kernels push -p "$STAGE"

echo "waiting for run to finish (poll every 60s)..."
while :; do
  st=$(kaggle kernels status "$USER/onprem-llm-bench-gpu" 2>&1 || true)
  echo "  $st"
  case "$st" in
    *complete*) break ;;
    *error*|*cancel*) echo "run failed"; exit 1 ;;
  esac
  sleep 60
done

echo "pulling output..."
kaggle kernels output "$USER/onprem-llm-bench-gpu" -p results/
echo "-> results/gpu_grid.json"
