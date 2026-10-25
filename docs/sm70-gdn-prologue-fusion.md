# GDN l2norm prologue fusion (P1) — findings

Date: 2026-10-25. Status: **implemented, bit-exact, opt-in (default OFF), no measurable
round-time gain at the tested operating points**. Archived as a negative performance
result with three positive side findings (kernel bug, verification methodology,
bottleneck redirection).

## What was built

The q/k l2-normalization feeding every GDN layer (`build_gdn_l2_norm` in
`src/models/models.h`: `scale(rms_norm(x, eps/n), 1/sqrt(n))`) is two kernels per tensor.
The fork already carried a `Prologue` template in `gated_delta_net.cu` plus a
`ggml_cuda_op_gated_delta_net_fused` entry point, but **no caller existed** — the
mechanism was dead code (the mma path never serves decode on sm_70, the chunked path
served prefill only).

P1 wires it for decode:
- `ggml_cuda_match_gdn_prologue` (ggml-cuda.cu): matches the 4-node chain
  `q_rms → q_sc → k_rms → k_sc` and the `GATED_DELTA_NET` that consumes both scales.
  The gate/beta/state nodes sit between k_sc and the GDN, so the chain is **not**
  contiguous; the evaluated graph order for one layer is
  `[rms_q, sc_q, view, rms_k, sc_k, view_v, alpha/beta/gate..., state, GDN, ..., cpy]`.
- A pending-elide list (`g_sm70_gdn_pending`, thread_local): the matcher runs at
  `rms_q` and records the four nodes to skip; the dispatch hook
  (`ggml_cuda_sm70_gdn_pro_dispatch`) elides them and launches the fused kernel at
  the GDN node, once g/beta/state exist. The snapshot cpy that follows the GDN is
  folded in through the existing `ggml_cuda_try_gdn_cache_fusion` and elided too.
- Opt-in: `LLAMA_SM70_GDN_PROLOGUE=1`. Diagnostics: `..._DBG=1`
  (first failed match reasons), `..._NOCACHE=1` (disable the cpy folding for A/B).

## Bug found and fixed in the existing kernel (real finding)

The `Prologue` reduce used `warp_reduce_sum<warp_size>` over 16 strided squares per
lane. With `cols_per_warp=4` the four column segments all load the **same** 128-wide
q/k head (q/k do not depend on the state column), so the full-warp sum over-counts
`ss` by 4x and the normalization comes out 2x small.

- `verify/sm70_gdn_prologue_microtest.cu`: warp32 ss = 1555.66 vs true 388.91 (4x),
  rel err vs stock 5.0e-01; 8-lane segment reduce: rel err 3.3e-07.
- Fix: keep the strided sum but reduce over `lanes_per_col` — or, better, the
  bit-exact scheme below.

This bug is invisible today only because the Prologue path has no callers. Anything
that wires the fused entry (including future prefill work) must include the fix.

## Bit-exactness: the scheme that works

`rms_norm_f32` for ncols=128 dispatches `RMS_NORM_LAUNCH(32, 1, vec)`: 32 threads,
one float4 per thread in the order `x²+y²+z²+w²`, then a 32-wide `shfl_xor`
butterfly (offsets 16,8,4,2,1). Reproducing exactly that tree from inside the GDN
kernel requires loading the head row as **one float4 per lane** (lane = threadIdx.x,
0..31) and running the same butterfly — not the strided per-lane sum (ulp-level, not
bitwise; greedy output diverges at long context).

- `verify/sm70_gdn_prologue_bitexact.cu`: with the float4 scheme, mock rms+scale vs
  prologue output differ in **0 / 16384 elements**.
- Matcher guards: warp_size == 32, raw data 16B-aligned, all nb multiples of 16B.

## Acceptance evidence

- Short context (16k, greedy 300): fused ON vs OFF **byte-identical output**,
  identical draft acceptance 15/21, 73.85 vs 73.50 tok/s.
- Baseline nondeterminism (important, pre-existing): two runs of the **same binary in
  the same mode** at ~30k context produce **different greedy output from the first
  token** (off1 ≠ off2). Text-level A/B is therefore only meaningful at short context
  or single-process logits; all long-context "content differs" observations are
  confounded by this.
- Performance (mid ~30k, same binary, step-prof round walls):
  OFF 2.56/3.42/3.08/4.39/2.84/3.13/2.59/3.07 ms per decode call (median ≈3.07);
  ON 5.21/3.06/3.21/2.80/2.08 (median ≈3.06). **No measurable gain.**
  An apparent 89.3 vs 81.2 tok/s in an earlier ON run was acceptance luck
  (98% vs 90% draft acceptance on different generated text), not fusion speed.

## Why no gain (and what it means)

Eliding 4 kernels x 48 layers = 192 launches per pass costs less than the measurement
noise. Under CUDA graphs those kernels are tiny (a 128-float reduce per head, 16 KB
tiles) and overlap launch latency with the neighbouring GEMMs. The P0 ledger says the
real per-round budget is elsewhere:

- meta host orchestration ≈ 3.5-4.0 ms per forward pass (55 meta-graph computes per
  round; each subgraph goes through ggml_backend_sched + cuda graph launch);
- the weight stream runs at ≈43 % of roofline (≈10.4 ms GPU busy per card per round,
  ~3.2 ms theoretical floor at 900 GB/s).

So the next levers are the one-graph-dual-device capture (P2, ninfer-style) and the
fused dequant TC GEMM (P3), not more glue-kernel elision.

## Reproduce

```
# gates
nvcc -arch=sm_70 -O3 verify/sm70_gdn_prologue_bitexact.cu -o bitexact && ./bitexact
# short A/B (Windows)
p1_short_run.bat on|off   +   p1_short_driver.py on|off
# fused smoke (target only, no speculation)
nospec_run.bat on
```

Tools (Windows side, not in the repo): `D:\zcode-build\{mid_run.bat, mid_driver.py,
nospec_run.bat, p1_short_run.bat, p1_short_driver.py, run_bitexact.bat}`.
