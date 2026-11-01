# Mask-driven D256 prefill for a unified KV pool

Status: shipped on `kvu-mask-driven` (2026-11-01). Prefill and decode both take
the mask-driven path; SplitKV3 and q4/q8-direct compose with it. The original
phase-1 scope (prefill only) is described first, the phase-2 additions follow in
section 7.

<!-- phase-1 scope line kept for history: decode stays on the stock path. -->
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

Decode was left on stock in phase 1 on purpose: that kernel carries one
`kv_offset` for all rows, and under `-np N` without speculation `q_len` is
exactly N tokens from N different sequences, so it would have computed wrong
values, not just been slow. Phase 2 fixes that - see section 7.

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

Measured, 4 slots (prefill, aggregate of the four requests):

| workload | stock (`LLAMA_SM70_D256=0`) | mask-driven |
|---|---:|---:|
| 4 x ~2.6k, one V100 | 903 tok/s | 956 tok/s |
| 4 x ~1.7k, 2 x V100 TP2 | 1491 tok/s | 1530 tok/s |
| 4 x ~10k chunked (each prompt spans 3 ubatches), one V100 | 799 tok/s | **1031 tok/s (+29%)** |
| 4 x 16k, one V100 | 646 tok/s | **902 tok/s (+40%)** |
| 4 x 16k, 2 x V100 TP2 | 966 - 1000 tok/s | **1343 - 1396 tok/s (+39%)** |

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

**TP2 pays, once it is really TP2.** On two V100s over NVLink the same 4 x ~1.7k
workload is 1530 vs 956 tok/s for one card (+60%). An earlier revision of this
section reported the opposite; that came from measuring one V100 plus the RTX
3080 instead of the TP2 pair.

## 6. Mask traffic: one byte per (row, 32-cell block)

The f16 mask is read once per head CTA, so an 8k prefill reads it 24 times: 4 KB
per block against 32 KB of K+V, the 12.5% tax. `sm70_mask_state_kernel` condenses
it once per node into one byte per (row, block):

| state | kernel does | cost |
|---|---|---|
| `VISIBLE` (every cell is 0.0) | nothing - adding 0 leaves the score bit-identical | 1 byte |
| `HIDDEN` (every cell is exactly -inf) | assigns -inf instead of adding it | 1 byte |
| `MIXED` | reads the f16 mask as before | 1 byte + 4 KB |

Every classification is exact, never approximate: only all-zero counts as
VISIBLE and only all -inf as HIDDEN, so a graded mask (ALiBi), a short trailing
block, or anything unrecognised falls into MIXED and behaves exactly as before.
The state array is 1/32 of the mask, is shared by all head CTAs and stays in L2,
which takes the tax from 12.5% to about 0.2%. A 2-bit packing would take it to
0.05%, which is not worth the cross-thread packing.

The harness proves the claim rather than arguing it: each case runs the kernel
twice, once with the state and once with the plain f16 path, and compares the
outputs with `memcmp`. All 8 cases report **0 of N bits differ**, and the
28-case battery still reports the error values it had before.

On the wall clock the change is below the noise of a 10k-token burst (the tax was
worth ~1% there by arithmetic); it should be worth ~3.8% at 46k and ~7.6% at
174k, which needs a longer run than the one that shipped with this.

### A 1.2M-cell pool is not the disaster it looks like

The mask is `[n_kv][q_len]` f16 and is uploaded from host memory every ubatch.
On this box the host link is PCIe 3.0 x4, measured at **3.3 GB/s** (pageable and
pinned identical, so it is link-bound, not driver-bound). If `n_kv` tracked the
pool capacity, a 1.2M-cell pool would upload 1.2M x 2048 x 2 B = 4.9 GB per
ubatch, i.e. 1.5 s each, and prefill would be dead.

It does not. `n_kv = PAD(used_max_p1)` - the highest *used* cell index - and
`find_slot` restarts from index 0 whenever the allocation head drifts past the
live count plus two batches, so a pool that is far from full never accumulates a
high-water mark. Measured with `LLAMA_SM70_KV_OCC=1` over a 131072-cell pool:

```
[kv-occ] size=131072 live=10511 max_p1=10511 ratio=1.00 n_kv=10752
```

Every sample reads ratio 1.00. The mask therefore scales with the data actually
held, not with the pool, and defrag is unnecessary outside a nearly-full pool -
where the problem is capacity, not compaction.

Confirmed end to end on 2 x V100 (TP2) with a 1.2M-cell pool, 4 slots x 16k:
stock 1071 tok/s aggregate, mask-driven **1477 tok/s (+38%)**, the same as the
32k-pool numbers. The single-card run of the same configuration OOMs, which is
expected: 1.2M cells need 1.2M x 40 layers x 512 x 0.56 B = 13.7 GB of KV on top
of an 11 GB model, so that pool size implies TP2.

## 7. Phase 2: decode, SplitKV3, q4/q8-direct

**SplitKV3** now composes with the mask window. The kernel used to split the
causal window into equal thirds, which a window that starts at the first visible
cell cannot use; it now divides *whatever* window the tile got (base + span/3).
For the causal path that is identical by construction (base 0, span =
visible_n_blocks). Under MaskMD a window shorter than 6 blocks is not split at
all - two runs of a fresh pool are a handful of blocks apart, and tripling the
CTA count there costs more than the merge; splits 1 and 2 then write neutral
partials, which the merge already guards. Harness: two new cases (one sequence,
and four slots interleaved in runs of 8), 8/8 pass at 1-3e-5, and the 28-case
battery still reports exactly the same error values, i.e. the causal path did
not move.

**q4/q8-direct** now composes too: the raw-block loaders only change how K/V
words reach the fragments, the masking site is the score fragment either way, and
`get_alloc_size` derives the f16 mirror from the same `sm70_kv_direct` predicate
on both sides. The instantiation table became {dense, SplitKV3} x {position mask,
mask tensor} x {f16, q4, q8}^2 = 36. The routing probe reports the two axes
independently, so a ledger can tell "+ q4-direct + mask-driven" from the staged
one.

**Decode** takes a `MaskMD` template parameter: the per-key predicate reads the
mask row of the token instead of comparing against one `kv_offset`, and the live
probe degrades to the capacity because no single boundary exists any more. Cost:
one 2-byte mask read per (row, key) against 512 B of K+V.

The width ceiling is the part that had to become adaptive. Under `-kvu` the
decode ubatch is `slots x (1 + drafts)` tokens wide (4 slots with MTP-3 is 16),
so a hardcoded `<= 8` would silently send every speculative decode back to
stock. The kernel loops its 32-row passes for any width, so the hard limit is
gone; `LLAMA_SM70_D256_DEC_WMAX` is a measured-performance ceiling (default 16,
0 = none), and llama-server raises it to exactly `n_parallel x (1 + n_max)` for
the launch configuration when a unified pool runs several slots - a value the
user set wins.

Measured, 4 slots (one token per slot, no speculation), **decode with a single
request in flight** - the only number that compares the kernels rather than the
scheduler:

| workload | stock | mask-driven + TC decode |
|---|---:|---:|
| decode at ~0.9k, one V100 | 19.9 tok/s | 18.4 tok/s (**-7%**) |
| decode at ~0.9k, 2 x V100 TP2 | 28.4 tok/s | 24.9 tok/s (**-12%**) |
| decode at ~16k, 2 x V100 TP2 | 41.6 / 41.7 / 41.8 / 41.6 tok/s | 42.0 / 42.1 / 42.0 / 41.9 tok/s (+1%) |

Read that as: under a unified pool the TC decode kernel is **not a win in this
depth range**. Below ~1k it is a clear loss (the weight stream dominates and the
KV split only adds overhead); by 16k it is a wash. The +20-30% this kernel shows
on the single-sequence path was measured at 174k, a depth none of these
configurations reach.

A warning about the method, because it produced a wrong conclusion here once:
with four slots decoding at the same time the per-request rate swings by 3-7x
within one run (3.2 to 23.6 tok/s), because the rate depends on how many slots
are still alive. Those burst numbers look like a large win for either build
depending on which slot you read. Only the solo column means anything.

So the value of this work is coverage and coherence - one code path for every
batch shape, no silent fallback - not decode throughput. What is missing is a
depth threshold: decode should fall back to stock below some measured kv_len, and
none of these runs says where that line is (it is above 16k and below 174k, and
finding it costs a long-context sweep).

Verified widths for the mask-driven decode: w=4 (4 slots x 1 token) and w=8
(2 slots x 4) reproduce bit-identically across repeated runs.

**Wide verify batches are not reproducible, and this kernel is not the cause.**
Two identical bursts of 4 slots x 2 tokens (w=8) gave, on consecutive runs:
tuned 0 / stock 38 token mismatch, then tuned 41 / stock 40. The stock build
never executes a line of this code, so the flakiness belongs to the
multi-slot + speculative-verify combination in llama.cpp - the failing side
changes from run to run, which is the signature of a scheduling effect rather
than an arithmetic one. A likely mechanism is the KV rollback after rejected
drafts: the freed cells are handed out again, the mask layout differs, and a
greedy token flips where the top-2 gap is small. That is a hypothesis, not a
measurement.

KV quantization is not the cause: the same shape with an f16 K and V cache gave
tuned 41 / stock 41, unchanged from the q4_0 numbers.

Where the flips actually happen was measured on three stock requests at their
first divergence (top-2 margin in each run, and the logprob drift over the ids
both runs share):

| request | first flip at | margin A | margin B | drift |
|---|---:|---:|---:|---:|
| req 0 | token 23 | 0.237 | 0.060 | 0.160 |
| req 2 | token 23 | 0.106 | 0.031 | 0.155 |
| req 3 | token 26 | 0.090 | 0.032 | 0.211 |

All three flipped where the drift exceeded the smaller margin, i.e. where the
two candidates were closer than the noise between runs. That is the signature of
a floating-point ordering effect, not of a wrong kernel. Note the magnitude: the
drift that matters is the one measured *before* the streams separate, about
0.2 nat. The 1.9 nat quoted elsewhere in this document was measured after
divergence, when the two runs are answering different contexts, and says nothing
about the mechanism.

Consequence for the ceiling: it is still 8, because widths above it were never
verified for this kernel - but it does not, by itself, make wide verify batches
reproducible, since stock flakes on them too. Single-token multi-slot decode
(w = slots) and narrow verify (2 slots x 4) are the shapes that were shown
stable.

### Making the decision reproducible: LLAMA_NEAR_TIE_EPS

Since the flips happen only where the margin is smaller than the drift, the
cheap fix is at the decision, not at the batching: when the best candidate is
within `eps` of the second, keep the smaller token id and drop the rest. Which
token wins then depends on the logits only through "who is within eps", a set
the drift cannot reorder.

Set `LLAMA_NEAR_TIE_EPS=0.25` (nat). It is off by default, applies only to
greedy requests (`temp == 0`) that do not ask for probabilities, and changes the
output relative to stock wherever a near-tie occurs - it is a reproducibility
patch, not a correctness one, and it is also the only change in this document
that affects the stock path.

Measured on the shape that reliably flaked (4 slots x 2 tokens): without the
variable, two identical bursts differed by 38-41 tokens in 2 of 4 runs; with it,
three consecutive runs all reported 0 of 64. The eps comes from the table above
(drift ~0.2 nat), so it prunes only genuine coin flips; setting it in the 1-2
nat range would rewrite tokens the model actually preferred.
