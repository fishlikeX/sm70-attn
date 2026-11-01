# Mask-driven D256 prefill for a unified KV pool

Status: shipped on `kvu-mask-driven` (2026-11-01). Prefill only; decode stays on
the stock path. Gates: kernel harness 6/6, existing kernel harness 28/28, e2e
single card and TP2 both PASS.

## 1. What was broken

`llama-server -kvu -np N` (N > 1) puts several sequences in one ubatch. The
D256 kernels derive visibility from positions:

- kernel: `Mask<true,false,false>(kv_len, kv_len - kv_offset, -1, 0)` applies
  `col <= row + kv_offset` (`fattn-sm70-d256-kernel.cuh`, the vendor
  `sm70-vendor/flash/mask.h:181,220,239`);
- `kv_offset` is one scalar for the whole CTA, recovered by a monotone binary
  search over the LAST mask row;
- the scan window is the contiguous prefix `[0, n_block_max]`.

Under a unified pool none of these hold. `llama-kv-cache.cpp:1664-1719` builds
the mask as: visible iff the cell is not empty, the cell belongs to that query
row's sequence, and its position is `<=` the row's position (plus SWA). Cells
of the other sequences sit between a row's own cells (`find_slot` allows holes,
there is no defrag), so a row's visible set is not a prefix.

Before this change `tools/server/server.cpp` told the backend that state with
`LLAMA_SM70_D256_MULTI_SEQ=1`, which disabled the kernels process-wide.

## 2. What the fix does

The mask tensor is the only truth, so the kernel now reads it.

**`fattn-sm70-d256-maskmd.cuh`** (new, the vendor headers stay untouched):

- `MaskFromKV` reuses the vendored `sm70_mask_row_idx` / `sm70_mask_col_idx` and
  `convert_layout_acc_rowcol` so the fragment-to-(row, col) mapping is the same
  one the position mask uses, and replaces only the value: `tensor(coord) +=
  mask[row][col]`.
- Accumulate, not assign: the vendored mask assigns `-INFINITY`, which would
  also wipe the visible entries. A `0.0` entry leaves the QK result bit-exact.
- `col >= n_kv` is past the tensor and past the loaded KV (the last tile is
  rounded up to `kBlockN`), so those entries are assigned `-INFINITY` instead of
  accumulated - otherwise a NaN in the rounding tail poisons the row.
- Rows `>= q_len` (the padding of the last tile) clamp to `q_len - 1`: the
  launcher passes `q_pad` as `query_len`, so without the clamp the mask read
  runs off the end.
- `sm70_mask_row_bounds_kernel`: one block per query row, first and last visible
  cell. Runs once per attention node (the mask does not depend on the head), so
  the `hkv*gqa` CTAs share it.

**`fattn-sm70-d256-kernel.cuh`**: new `MaskMD` template parameter (default
false, so every existing instantiation keeps its exact code). Under `MaskMD`:

- the scan window is the union of the tile's visible cells:
  `n_block_max = ceil((max row_hi + 1) / kBlockN) - 1`,
  `n_block_min = min row_lo / kBlockN`. Both ends are exact, and the loop
  skeleton (descending over one contiguous range) is unchanged - the prefetch
  chain and the `first_tile` flag keep working;
- `row_max` starts at `-1e30` like the SplitKV3 path, so an empty window cannot
  produce `(-inf) - (-inf)`.

**`fattn-sm70-d256.cu`**: routing, plus one extra instantiation. v1 keeps the
mask-driven path on the staged f16 KV layout (`k_direct` / `v_direct` nodes
fall back to stock) and disables SplitKV3 there: its three segments are an equal
division of the causal window, which would have to follow the mask's visible
runs instead.

**`tools/server/server.cpp`**: still announces `LLAMA_SM70_D256_MULTI_SEQ=1`,
but only when the user has not set the variable, so `=0` stays a working
escape hatch to the stock path.

Decode is untouched on purpose: `ok_d` now rejects when the hint is set. Its
kernel carries one `kv_offset` for all rows, and under `-np N` without
speculation `q_len` is exactly N tokens from N different sequences, so it would
compute wrong values, not just be slow.

## 3. Cost model

The mask is read per (query row, 32-column block) for every block in the window,
at every head CTA: `64 x 32 x 2 B = 4 KB` against `32 KB` of K+V for the same
block, i.e. **+12.5% KV traffic inside the attention kernel**. That is HBM read
bandwidth on the card running the kernel - not NVLink (attention has no
cross-card traffic; TP2 only reduces at the layer boundaries) and not host
memory (the mask itself lives in a host buffer and is copied to the device once
per ubatch by `ggml_backend_sched_copy_input`, which is pre-existing).

Folded into the wall clock it is small, because attention is only part of
prefill: at 22k attention is ~14% of the wall (so ~1.8%), at 174k ~61% (~7.6%).
The scan window is the union of the tile's visible cells, not the pool: with the
steady state (each slot's cells in runs of one ubatch chunk) a 64-row tile sits
inside one run.

Measured, 4 slots on one V100:

| workload | stock (`LLAMA_SM70_D256=0`) | mask-driven |
|---|---:|---:|
| 4 x ~2.6k tokens, burst | 903 tok/s | 956 tok/s |
| 4 x ~2.6k tokens, TP2 | 397 tok/s | 394 tok/s |
| 4 x ~10k tokens, chunked (each prompt spans 3 ubatches) | 799 tok/s | **1031 tok/s (+29%)** |

At 2-3k tokens attention is a small share and the two are within noise; at 10k,
where the prefill is chunked and each slot's cells arrive in several runs
interleaved with the others, the kernel pays. That is also the layout that makes
the mask-driven scan window widest, and it still wins.

## 4. Gates

- `verify/run_maskmd.ps1` (new): 6 synthetic unified-pool cases against a CPU
  f32 reference. Single sequence (regression), 4 slots contiguous, 4 slots with
  holes, 4 slots interleaved in runs of 8 cells (every 32-cell block mixes four
  sequences), `q_len` not a multiple of `kBlockM`, and one row with no visible
  cell. Worst error 2-4e-5, bounds kernel checked against the host in every
  case.
- `verify/run.ps1` / `run.sh`: 28/28 unchanged, i.e. the position path did not
  move. The source md5 pin is updated to the new kernel.
- `verify/kvu_gate.py` (new): end-to-end A/B. Runs the same request pattern on
  `LLAMA_SM70_D256=1` and `=0`, twice each for a control, plus solo runs. Four
  patterns, all run: identical prompts in lockstep (the hard one), different
  lengths, staggered arrivals, and 4 x ~10k tokens that each span three ubatches
  so the slots' cells interleave - the most fragmented layout there is. On the
  lockstep pattern the greedy ids match exactly on both builds and both builds
  drift 0 times out of 4; on the other three, ids move at most one step (a single
  flip cascading) and the logprob deltas stay inside the noise floor.
- Intermediate state (`LLAMA_SM70_D256_DECODE=1`, `-kvu -np 4`): the routing
  ledger shows prefill accepted by the mask-driven path and every decode node
  rejected with `REJECT: decode stays stock under unified KV`.

## 5. Findings worth remembering

**The server-level comparison is not bit-reproducible, and the KV pool is not
the reason.** Running the same build twice with the same 4-request burst moves
the top-5 logprobs by 0 to 1.9 nats and swaps up to 9 ranks - for the stock build
as well - and a greedy stream can move with it. Chased down to the bottom:

| experiment (stock kernels, same prompt) | max abs delta logprob |
|---|---:|
| `-np 1` (per-sequence cache) vs `-np 4 -kvu` with ONE active slot | **0.000** |
| the same request alone vs co-batched with 3 identical requests | 0.133 - 0.298 |
| same request, `-ub 2048` vs `-ub 512`, single sequence, no concurrency | **0.251** |
| same, with `-fa off` and an f16 cache | 0.078 |
| same, with `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` | 0.123 (from 0.251) |

So: the unified pool itself is bit-identical to the per-sequence path; what moves
the logits is the **ubatch shape (M)**. Quantized weights go through cuBLAS in
F16 (ggml-cuda.cu picks `GGML_TYPE_F16` compute whenever
`fast_fp16_hardware_available`, and Volta qualifies), so the accumulation is
fp16; cuBLAS picks a different tiling/split for a different M, the order changes,
and the logits move by O(0.1) nat - two thirds of it in the GEMM stack, the rest
in the shape-dependent attention kernel choice. Concurrency is just one of the
ways M changes, which is why heterogeneous requests drift more than identical
ones. Forcing `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` roughly halves it, at fp32 GEMM
speed.

Consequence for the gate: token identity is absolute only when the composition is
reproducible (identical prompts, all slots in lockstep); with different lengths
either build can flip one token where the top-2 gap is smaller than that drift.
Logprob deltas are judged against the noise floor the same build shows against
itself (2x it), and a rank swap only counts when the swapped pair is further
apart than that floor. A fixed 1e-3 fails stock against itself.

**`kv_len` is not always a multiple of `kBlockN`.** `n_kv = min(cells.size(),
PAD(used_max_p1, 256))`, so a `-c` that is not a multiple of 256 leaves a
rounding tail. The position path survives it because it masks with assignment;
the first mask-driven version accumulated and inherited NaN from whatever the
tail read hit. Fixed by assigning on the out-of-range columns.

**mmap weights make the first build measured look 10x slower.** With mmap, the
page-in of ~11 GB happens inside the first prefill: 28.4 s vs 2.7 s for the same
1682-token prompt. The gate now sends a warm-up request before timing.

**TP2 at 3k context is not faster than one card** (397 vs 903 tok/s aggregate
for 4 concurrent requests) - for both builds, so it is pre-existing: at short
context attention is a small share and TP2 pays the allreduce at two boundaries
per layer.

## 6. Not done (v2 candidates)

- **Decode under a unified pool.** `fattn-sm70-d256-decode.cuh` needs the same
  treatment (per-token mask instead of one `kv_offset`). Worth doing only if the
  measurement says the TC decode kernel still pays under `-kvu`; with MTP on it
  is already rejected by the 2..8 verify-width window.
- **Mask traffic.** Precompute a 2-bit state per (row, 32-column block)
  (all hidden / mixed / all visible) once per node: ~0.5 MB at 8k x 8k, resident
  in L2, and blocks that are all visible skip the f16 read entirely. Drops the
  mask cost by two orders of magnitude. Not needed while 12.5% is affordable.
- **SplitKV3 on the mask-driven path**: split along the visible runs instead of
  an equal division.
- **q4/q8-direct on the mask-driven path**: the raw-block loaders would have to
  carry the same parameters; the staged f16 layout is the default anyway.
