#!/usr/bin/env bash
# Build and run the IQ* host-side codec check (T2 skinny port line).
#
# Needs a CPU-only build of this tree first (no Volta required):
#   cmake -B build-cpu -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release \
#         -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF
#   cmake --build build-cpu -j
#
# Usage:
#   ./build_and_run.sh <model.gguf> --type <iq3_s|iq3_xxs|iq2_xs|iq2_s> [max_blocks] [--all-tensors] [--selftest]
#   ./build_and_run.sh --synth <n_blocks> --type <...> [--selftest]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
FORK="$(cd "$HERE/../.." && pwd)"
BUILD="${FORK_BUILD_CPU:-$FORK/build-cpu}"
g++ -O2 -std=c++17 -o "$HERE/iq_host_check" "$HERE/iq_host_check.cpp" \
    -I "$FORK/ggml/include" -I "$FORK/ggml/src" \
    -L "$BUILD/bin" -lggml-base -Wl,-rpath,"$BUILD/bin"
exec "$HERE/iq_host_check" "$@"
