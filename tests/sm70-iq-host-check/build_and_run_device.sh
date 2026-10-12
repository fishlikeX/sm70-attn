#!/usr/bin/env bash
# Build and run the IQ3_S device-side codec check (needs a Volta GPU + nvcc).
#
# Needs a CPU-only build of this tree first (for the in-tree reference dequant):
#   cmake -B build-cpu -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release \
#         -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF
#   cmake --build build-cpu -j
#
# Usage: ./build_and_run_device.sh <model.gguf> [max_blocks]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
FORK="$(cd "$HERE/../.." && pwd)"
BUILD="${FORK_BUILD_CPU:-$FORK/build-cpu}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
"$NVCC" -O2 -std=c++17 -arch=sm_70 \
    -I "$FORK/ggml/include" -I "$FORK/ggml/src" -I "$FORK/ggml/src/ggml-cuda" \
    "$HERE/iq_device_check.cu" \
    -L "$BUILD/bin" -lggml-base \
    -o "$HERE/iq_device_check"
LD_LIBRARY_PATH="$BUILD/bin:${LD_LIBRARY_PATH:-}" exec "$HERE/iq_device_check" "$@"
