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
