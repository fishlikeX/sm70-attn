# SM70 D256 prefill attention — where the time goes, and what is left on the table

Status: measurement complete (2026-10-26), no kernel change landed. This file
records the evidence behind that decision so the next attempt does not have to
re-derive it.

Harness: `verify/sm70_attn_micro.cu` (built by `D:\zcode-build\build_amicro.bat`).
It launches `sm70_d256_splitd_dense_kernel` alone with the production geometry —
D=256, heads_q=24, heads_kv=4 (GQA 6), q_len=4096, kv_len=174080,
kv_offset=kv_len-q_len, f16 K/V mirror, SplitKV3 — and times it with CUDA
events. Numerics were checked against a naive warp-per-row reference:
max|diff| = 3e-5 on a 128×4096 shape (the harness reports
`[check] max|diff|`).

## Baseline (production shape, 1× V100, clocks at 1530 MHz)

| variant | time | TFLOP/s | % of 125 |
|---|---|---|---|
| SplitKV3 (production) | 357.7 ms | 48.4 | 38.7% |
| dense (no KV split) | 370.0 ms | 46.8 | 37.4% |
| SplitKV3 + q4-direct K/V | 507.0 ms | 34.2 | 27.3% |

SplitKV3 is only 3.4% ahead of dense at this length; the q4-direct load path
(no f16 mirror) is 42% *slower* in the kernel, so production uses the mirror.

Useful FLOPs are counted as `4·D·heads·(q_len·kv_offset + q_len(q_len+1)/2)`
(QK + PV over the causal window). The kernel executes ~1.01× that.

## Ablations (timing-only copies, `verify/ablate.py`)

Each row neutralises one inner-loop phase; results are garbage, the delta is the
phase's marginal cost.

| variant | time | delta |
|---|---|---|
| baseline | 357.7 ms | — |
| no QK gemm | 268.9 ms | −88.8 ms (24.8%) |
| no PV gemm | 312.5 ms | −45.2 ms (12.6%) |
| no causal mask | 353.0 ms | −4.7 ms (1.3%) |
| no exp2 | 357.4 ms | −0.3 ms (0.1%) |
| no o_storage rescale | 361.0 ms | 0 (noise) |
| **no `__syncthreads` at all** | 335.0 ms | −22.7 ms (6.3%) |
| **no QK + no PV gemm** | **225.1 ms** | −132.6 ms = the glue alone |
| kBlockM 64→32, 256→128 thr (2 CTA/SM) | 381.9 ms | +24 ms (worse) |

## The decomposition

The two gemms' marginal cost is 88.8 + 45.2 = 134 ms, and their theoretical time
at 125 TFLOP/s is 138 ms — so when the tensor cores run, they run at ~peak. With
both gemms neutralised the kernel still takes 225 ms. 225 + 133 = 358: **the MMA
work and the glue are exactly additive, there is no overlap at all.**

Glue = 3300 cycles per CTA-iteration, ~346 non-HMMA SASS instructions per warp
(the 169 operand `LDS` disappear with the gemms and are not part of it), i.e. an
issue efficiency of ~21% (the nogemm build is a timing probe only: with both
gemms gone the compiler can fold part of the softmax, so 225 ms is a *lower*
bound on the glue, and the additive split is an upper bound on the overlap).
That is a latency signature, not a throughput one: the
gmem→register→smem staging chains sit on each warp's critical path and nothing
fills them.

Consequence: **the kernel's ceiling with the current instruction mix is 225 ms
(1.59×, 76.9 TFLOP/s, 61.5% of peak)**, reachable only by making the glue of one
KV tile overlap the MMA of another. Note the two measured non-fixes: barriers (6%)
and hoisting the V loads (0%) show the stall is a per-warp dependency chain, so
overlap has to come from having a *second independent block in flight in the same
warp's instruction stream* — the FA3 ping-pong shape.

## Instruction budget of the hot loop

`cuobjdump -sass` on the SplitKV3 f16 instantiation; the loop body is
`0x19b0..0x59d0` = 1027 SASS instructions per warp per 32-key tile:

```
512 HMMA.884        169 LDS.U.64        90 FMUL        28 IADD3
 26 FMNMX            24 MOV             22 STS         22 FADD
 18 LEA              12 SHFL            12 LDG         12 FSETP
 12 BAR.SYNC         11 ISETP           10 MUFU         10 FSEL
  8 IMAD              8 FFMA             8 F2F           4 PRMT
```

All operand loads are 64-bit (256 B/warp). The custom pitch-68 K layout and the
TT-swizzled V layout only admit 64-bit vector loads, so this is a property of
the verified 1CatAI layouts, not a codegen accident.

## What is *not* the bottleneck (each measured, not assumed)

* **Barriers / phase serialisation** — dropping every `__syncthreads` saves 6.3%.
* **Occupancy** — halving the tile to 32 rows so two CTAs fit per SM (253→252
  regs × 128 threads × 2 = 64512 ≤ 65536, 26880 B smem × 2 ≤ 98304) is 6.8%
  *slower*: the extra K/V staging per row costs more than the independence buys.
* **Mask / exp2 / rescale ALU** — 1.4% combined.
* **V-load latency in the PV critical path** — hoisting the V `LDG`s to the top
  of the block (letting the whole QK phase cover them) changed register use
  253→191 and the runtime not at all (359.3 ms).
* **QK MMA efficiency** — QK's marginal cost (88.8 ms) is close to its
  theoretical 69 ms at peak, i.e. the QK tensor-core work already runs at
  ~78-100%. There is no 2× sitting in the QK gemm.

## The hardware fact that explains the 38.7%

`verify/sm70_mma_peak.cu`: back-to-back `mma.sync.m8n8k4.row.col.f32.f16.f16.f32`
(the exact atom the kernel uses), 1 CTA/SM, 8 warps:

| independent accumulator tiles per warp | TFLOP/s | % of 125 |
|---|---|---|
| 1 | 9.8 | 7.9% |
| 2 | 20.2 | 16.2% |
| 4 | 47.6 | 38.1% |
| 8 | 121.6 | 97.2% |

So the 125 TFLOP/s spec is reachable, but only with ~16 independent accumulator
chains per SM partition (8 per warp here). **The QK gemm has exactly 4** — a warp
computes 8 rows × 32 cols = 4 tiles of 8×8, and the 16 K-steps of each tile form
a dependency chain 4 instructions long. 38.1% is exactly what the kernel
measures overall, which is a coincidence of the same arithmetic rather than
proof that QK alone is the limiter (the ablation above shows it is not), but it
does mean any schedule that wants the tensor cores near peak needs ≥8 tiles in
flight per warp.

## Why this matters: attention share of the 174k prefill

Model facts (GGUF `qwen35`): 65 blocks, `full_attention_interval = 4` → 16
full-attention layers (D=256, 24 q heads, 4 kv heads) and 49 gated-delta-net
layers. Layer attention time scales with kv_len, so over a 174k prefill in
4096-token chunks the per-layer average is half the deep-end figure:

* per layer per chunk at 174k (measured) 358 ms → average over the prefill 179 ms
* 16 layers → 2.86 s of the 4.66 s chunk wall (`4096 / 878 t/s`)
* ⇒ attention ≈ **61% of the 174k prefill wall** (at 22k the same arithmetic
  gives ~14%, matching the op census' ~15%)

So the prefill number is mostly an attention number at long context. But the
payoff of kernel work is bounded by the ablation budget above: with the MMA and
the non-MMA work perfectly overlapped the kernel would be
`max(134 ms, 176 ms) ≈ 180–200 ms` instead of 358 ms; the realistic figure for a
ping-pong rewrite is ~265 ms (perfect overlap, same instruction mix), which is
`4096 / (4.66 s − 0.61·0.26·4.66 s) ≈ 1045 t/s` at 174k (+19%). A 2× kernel win
(→ ~1200 t/s) additionally requires cutting the staging/softmax work itself,
which is why the kBlockN=64 route (halves the per-key fixed cost *and* doubles
the accumulator chains) is the more valuable one if its layouts can be
re-derived safely.

## Not measured: per-op census at 174k

`GGML_CUDA_DISABLE_GRAPHS=1` (required for op-prof, whose `end()` hook is not
wired in this tree) segfaults the server during the first 4096-token chunk at
174k; it runs fine at 22k. The layer-share arithmetic above is the substitute and
agrees with the 22k census.

## What a further improvement would require

Everything cheap has been tried and measured. The remaining 1.59× (358 → 225 ms)
needs a different *schedule*, not a different constant:

1. **Two-block ping-pong per CTA.** Keep two KV tiles in flight: run QK for tile
   n-1 while doing softmax/PV for tile n, so the 176 ms of non-MMA work overlaps
   with MMA instead of serialising behind it. Needs K and V in smem without
   aliasing (they alias today: sK and sV0 share `kv_smem_ptr`, and K's pitch-68
   layout even overruns its 2048-half size by 139 halves into V), a second
   acc_s (8 registers) and the phases re-fused so the two blocks interleave in
   one instruction stream. Highest expected value (→ ~225 ms), highest risk.
2. **8 accumulator tiles per warp for QK** (kBlockN 32→64). This also halves the
   per-key fixed cost. Blocked on the hand-derived `SmemLayoutK` (pitch-68, built
   for 32 rows) and `SmemLayoutV` (TT swizzle, `load_v_fragment_tt` hardcodes
   `(k & 0x1e) << 6`, which aliases for k ≥ 32); both must be re-derived and
   re-verified. The QK-gemm-only upside is ≤5% (see the QK marginal cost
   above); the real value is halving the per-key fixed cost, which is what
   stands between ~265 ms and ~200 ms.
3. **Keep the current kernel.** It is a ported, verified kernel at 38.7% of peak
   with no single dominant inefficiency.

Recommendation: (3) for now; (1) only as a dedicated piece of work with the
harness as the gate.
