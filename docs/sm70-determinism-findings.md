# Determinism investigation — ≥30k context (2026-10-26)

## Symptom

The same binary, same prompt (p_mid, 22081 tokens), greedy (temp 0, top_k 1),
`cache_prompt:false` produces **different first tokens across successive requests
on one llama-server process**, and different outputs across repeated server runs.
This voided text-level A/B comparisons at long context during the P1 work.

## Localized boundary

Ruled out (each tested, all still nondeterministic unless noted):

| factor | test | result |
|---|---|---|
| PDL timing | code: `ggml_cuda_pdl_*` is Hopper+ only (`__CUDA_ARCH__ >= 900`) | not applicable on V100 |
| CUDA fusion decisions | `GGML_CUDA_DISABLE_FUSION=1`, TP2+MTP | still nondeterministic (3/3 distinct) |
| TP2 / meta backend / P2P allreduce | single GPU, no `-sm tensor`, no spec | still nondeterministic (2/3) |
| CUDA graph capture/chunking | `GGML_CUDA_DISABLE_GRAPHS=1` single GPU | still nondeterministic (2/3, same outputs as with graphs) |
| speculative decoding (MTP) | `--spec-type none` single GPU | still nondeterministic |
| pure compute path | two independent `llama-cli` processes, same prompt, greedy | **identical generation** (only diagnostic lines differ) |
| q4_0 KV quantization | `-ctk f16 -ctv f16` single GPU | still nondeterministic (pattern changes, 2/4) |

So the nondeterminism lives in the **multi-request server path**, not in the CUDA
kernels, graphs, fusions, quantization or tensor parallelism.

## Within-process pattern

Same prompt, back-to-back requests, n_predict=2, single GPU, no spec:

- q4_0 KV: `computes`, `pipelines`, `computes`, `pipelines`, `computes` — period-2 alternation.
- f16 KV: `ing`, then `pipelines`, `pipelines`, `pipelines` — first request differs, then a fixed point.

**Fresh process, first request: stable.** Four independent server processes (single GPU,
graphs on/off, q4 KV) all produced the same first-request token (" computes the").
llama-cli (one process, one request) is deterministic across processes too.

## Cross-request contamination (decisive test)

Two-slot server (`-np 2`, `-c 81920`), requests forced with `id_slot`:

```
slot0 target (1st)      -> " computes the"
slot1 unrelated short   -> "What is"
slot0 target (2nd)      -> " pipelines the"     <- changed by work on ANOTHER slot
slot1 target (1st)      -> " pipelines the"
slot0 target (3rd)      -> " computes the"      <- flipped back
```

An unrelated request **on a different slot** changes the next target request, and the
pattern alternates per request index, not per slot. So it is a process-global,
per-request toggling factor — not slot-local recurrent-state leakage.

## Remaining suspects (not yet instrumented)

1. Per-request physical cell/offset rotation in the KV cache or the recurrent
   (conv/ssm) state cells — the request's data lands at a different offset each time,
   and some code path is offset/alignment dependent (would be deterministic given the
   allocation history, matching all observations).
2. Server-side checkpoint save/restore (`use_ckpt_tgt` / `ckpt.pos_max` paths in
   tools/server/server-context.cpp) restoring global state across requests.
3. A global (not per-slot) staging buffer whose address alternates between two
   allocations with the request's ubatch loop.

## Practical rules until this is root-caused

- **Long-context A/B: compare first requests of fresh processes only**, or compare
  within one process at the same request index and slot.
- Short-context (few hundred tokens) greedy comparisons were bit-stable (P1 gate used
  a 67-token prompt: ON vs OFF byte-identical).
- Never compare request #k of one process with request #j of another.

## Next instrumentation step

Print, per request: the KV cell range actually allocated, the recurrent state
`i_next`/cell ids, `rs_z`, and the addresses of the state/logits buffers; then check
whether the two alternating outcomes correlate with two offset classes. That decides
suspect 1 vs 2.

## Root cause (2026-10-26): out-of-bounds LUT read in the sm70 iq4_xs dequant

`ggml/src/ggml-cuda/dequant-sm70.cuh`, `dequant_iq4_xs_sm70`: the vectorized rewrite
loads four quant bytes as one `uint32_t` and then took the high nibble as

    v[j+4] = d * kvalues_iq4nl[(q >> (8*j)) >> 4];   // missing & 0xF

For a 32-bit `q` this indexes the 16-entry LUT at up to 2^28. In the model the
out-of-range byte load lands inside the multi-hundred-MB compute buffer, so the
"LUT entry" is whatever that memory holds at the time - which is why the first
prefill of a process and the later ones dequantize IQ4_XS weights differently, and
why the wrong values are stable within a request index.

Confirmed with a standalone probe (plain llama API, no server, no sampler, memory
cleared between runs, same 550-token batch):

    before the fix:  run 0 argmax 107077 logit 9.503096 | run 1..4 argmax 69664 logit 9.863740
    after  the fix:  run 0..3 bit-identical (0 / 248320 logits differ)
    fixed fast kernel vs stock kernels (LLAMA_SM70_DEQUANT_FAST=0):
                     550 tok   logit 8.585686 / logprob -4.927675 (identical)
                     22081 tok logit 6.339583 / logprob -6.046364 (identical)

The first differing graph node was layer 0's qkv matmul with a byte-identical
activation input: the weight operand differs, the activation does not. The converted
F16 weight dump showed 3.3 % of values as negative zero on run 0 only, at every
lane `hi[1]` of the kernel's 8-value store - i.e. exactly the unmasked nibble.

### Why the old "bit-exact" claim was vacuous

`dq_verify.cu` compared random blocks against the stock kernels, but the OOB load
faulted (`illegal memory access`), every later `cudaMemcpy` failed silently, the host
vectors stayed value-initialized to zero, and `memcmp(0, 0)` printed "BIT-EXACT".
No CUDA error was ever checked and no output buffer was poisoned.

`verify/sm70_dequant_parity.cu` replaces it: every launch and copy is checked,
outputs are poisoned with 0xCD first, and the fp16 block scales are forced finite
(random `d` bytes are mostly inf/NaN, and `inf * LUT` collapses a whole block to one
bit pattern, hiding index bugs). Verified in both directions: with the bug restored
it exits 1 with `FAIL iq4_xs half sync: CUDA error: an illegal memory access`; with
the fix it reports `ALL BIT-EXACT` and exits 0.

## Residual (smaller, still open): server-level spread

After the fix the server still shows a small spread across identical requests
(550 tok: -2.0252 / -2.0997 / -1.9574 / -2.0028; 30000 tok: the first request changes
the token, the later two differ by 0.038). The probe above is bit-stable at 550 and
22081 tokens with either dequant path, including metadata-only memory clears, so the
residual is not in the ggml compute path. It correlates with the server's post-prompt
flow: right after a prompt the server sends a second batch that starts at pos n-1
(`[det-dec] n_tokens 4 pos_min 549` for a 550-token prompt), and reproducing that
flow directly fails: `llama_memory_seq_rm(mem, 0, 549, -1)` returns 0 because
`n_rs_seq == 0` (no partial rollback for recurrent memory). Whatever sends that
batch therefore relies on state bookkeeping that is still unaccounted for.
