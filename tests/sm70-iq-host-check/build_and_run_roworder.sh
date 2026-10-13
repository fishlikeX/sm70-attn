#!/usr/bin/env bash
# Build and run the R1 d256 KV row-order round-trip host check (施工单-2 §1 U2).
# Pure host code — no CUDA/GGML link required. Proves that fork's position-major
# staged-f16 mirror layout and v100's head-major mirror layout encode the SAME
# logical KV matrix (the precondition for the d256 merge), and that the two are
# NOT interchangeable on a raw read (sensitivity to the historical 0.69 scramble).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
g++ -O2 -std=c++17 -Wall -Wextra -o "$HERE/d256_kv_roworder_check" "$HERE/d256_kv_roworder_check.cpp"
exec "$HERE/d256_kv_roworder_check"
