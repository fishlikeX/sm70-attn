#!/usr/bin/env bash
# sm70 d256 TC decode kernel — standalone cross-check (V100 / WSL2).
# Standalone: builds the kernel + merge directly (nvcc, sm_70) against random
# data with a CPU reference; no llama.cpp runtime, model, or server needed.
# Covers w=1/2/8 (verify widths incl. two row-passes), splits 1..24, live<<cap
# mask semantics, GQA 6 and 8.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/.." && pwd)"
NVCC=$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc 2>/dev/null | head -1)
[ -n "$NVCC" ] || { echo "nvcc not found"; exit 3; }
"$NVCC" -O2 -std=c++17 -arch=sm_70 \
  -I "$root" -I "$root/ggml/include" -I "$root/ggml/src" -I "$root/ggml/src/ggml-cuda" \
  "$here/sm70_dec_verify.cu" -o "$here/sm70_dec_verify" || exit 4
"$here/sm70_dec_verify"
