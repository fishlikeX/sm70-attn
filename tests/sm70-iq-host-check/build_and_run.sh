#!/usr/bin/env bash
# Build and run the IQ3_S host-side codec check.
#
# Needs a CPU-only build of this tree first (no Volta required):
#   cmake -B build-cpu -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release \
#         -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF
#   cmake --build build-cpu -j
#
# Usage: ./build_and_run.sh <model.gguf> [max_blocks] [--selftest]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
FORK="$(cd "$HERE/../.." && pwd)"
BUILD="${FORK_BUILD_CPU:-$FORK/build-cpu}"
g++ -O2 -std=c++17 -o "$HERE/iq3s_host_check" "$HERE/iq3s_host_check.cpp" \
    -I "$FORK/ggml/include" -I "$FORK/ggml/src" \
    -L "$BUILD/bin" -lggml-base -Wl,-rpath,"$BUILD/bin"
exec "$HERE/iq3s_host_check" "$@"
