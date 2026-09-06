#!/usr/bin/env bash
# Push the GPU benchmark to a Kaggle notebook, wait, pull the results.
#
# Needs a FULL Kaggle API token in ~/.kaggle/kaggle.json (Account -> Settings ->
# API -> "Create New API Token"). A KGAT_-prefixed scoped token cannot push kernels.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 kaggle/build_kernel.py          # regenerate the self-contained bundle

STAGE=$(mktemp -d)
cp kaggle/_bundle.py "$STAGE/main.py"
USER=$(python3 -c "import json;print(json.load(open('$HOME/.kaggle/kaggle.json'))['username'])")
SLUG="$USER/onprem-llm-bench-gpu"
cat > "$STAGE/kernel-metadata.json" <<EOF
{
  "id": "$SLUG",
  "title": "onprem-llm-bench-gpu",
  "code_file": "main.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true
}
EOF

echo "pushing $SLUG ..."
kaggle kernels push -p "$STAGE"

echo "polling (every 90s; a full grid is ~1-2h on a T4)..."
while :; do
  sleep 90
  st=$(kaggle kernels status "$SLUG" 2>&1 || true)
  echo "  $(date +%H:%M)  $st"
  case "$st" in
    *complete*) break ;;
    *error*)    echo "!! run errored — pulling log"; break ;;
    *cancel*)   echo "!! cancelled"; exit 1 ;;
  esac
done

echo "pulling output..."
kaggle kernels output "$SLUG" -p results/
echo "-> results/gpu_grid.json  (+ .log)"
