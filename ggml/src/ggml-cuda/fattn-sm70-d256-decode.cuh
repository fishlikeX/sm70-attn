// SPDX-FileCopyrightText: Copyright 2026 the sm70-attn authors
// SPDX-License-Identifier: MIT
//
// sm70-attn — SM70 D256 tensor-core DECODE kernel (WP1-P1).
//
// Topology ported from ninfer-V100X2 gqa_attention_decode_i8_tc_volta.cuh
// (default profile: WarpsPerCta=4, no SharedSoftmax, no Fp32Pv): two warps
// compute the two independent eight-key QK^T tiles on mma.sync.m8n8k4 and
// publish FP32 scores to shared memory; all four warps are dim-slices of the
// head dim (DimSplit=4, DSlice=64), each running the same online-softmax
// recurrence redundantly and accumulating PV only over its own 64 dims in
// FP32 registers. Row capacity: Br=32 rows per pass over (token x gqa)
// query rows, at most two passes for the widest verify shape (w=8, gqa=6).
//
// Differences from the ninfer source, all deliberate:
//   * KV cache is llama.cpp's contiguous [ctx][head][dim] f16 view (or the
//     launcher's dequantized f16 mirror) — no page tables, no in-kernel
//     quantized-cache append. Strides come in as plain row/head strides.
//   * K/V are already f16 at rest: staging is a raw 16-byte copy, no
//     register dequant on the way in (the int8 source that motivated it does
//     not exist here; q4-direct in-kernel loads are a P2 item and will
//     reintroduce a dequant fill at the same site).
//   * Q is read straight from the f32 op input and converted in-register
//     while staging q_s (decode Q is tiny; no separate staging kernel).
//   * Partials are f32 (ninfer packs bf16 acc for its reducer); our merge
//     kernel is a fresh N-split f32 reducer and writes the normalized f32
//     destination directly.
//
// The Volta mma primitives and fragment addressing below are the llama.cpp
// mma.cuh transcription (same provenance as ninfer's volta_mma.cuh; every
// formula independently validated there on real V100 hardware).
//
// Register/shared-memory budget (from ninfer's load-bearing derivation):
//   * acc_f[8][8] = 64 FP32 PV accumulators/thread (DSlice=64) — the whole
//     reason the head dim is split across warps: one output row across the
//     FULL D=256 would need 256 registers/thread, at Volta's hard cap with
//     nothing left for fragments or loop state.
//   * q_s/k_s/v_s with SmemPad=8: 16KB + 8KB + 8KB = 32KB static shared.
//     The pad shifts consecutive rows 4 banks apart (unpadded 512-byte rows
//     put every row on one bank — ncu measured ~21 replays/instruction).
#pragma once

#include "common.cuh"

#include <cuda_fp16.h>
#include <math_constants.h>

#include <cstdint>

// ---------------------------------------------------------------------------
// Volta fragment addressing + mma.sync wrappers (llama.cpp mma.cuh lineage).
// ---------------------------------------------------------------------------
namespace sm70_d256_decode {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

// Q / P / PV-output: 32 rows x 4 half2 regs, I-major (row = lane).
__device__ __forceinline__ int qp_get_i() { return threadIdx.x & 31; }

// K (QK^T B operand): 8 rows, I-major-mirrored.
__device__ __forceinline__ int k_get_i() {
    const int lane = threadIdx.x & 31;
    return ((lane / 16) * 4) + (lane % 4);
}

// D (QK^T f32 output): 32x8, I-major; 8 regs span two rows (l&2).
__device__ __forceinline__ int d_get_i(int l) {
    const int lane = threadIdx.x & 31;
    return (l & 2) + (lane & ~2);
}
__device__ __forceinline__ int d_get_j(int l) {
    const int lane = threadIdx.x & 31;
    return (lane & 2) + (l & (4 + 1));
}

// V (PV B operand): 8 rows, J-major-mirrored.
__device__ __forceinline__ int v_get_i(int l) {
    const int lane = threadIdx.x & 31;
    return ((l / 2) * 4) + (lane % 4);
}
__device__ __forceinline__ int v_get_j(int l) {
    const int lane = threadIdx.x & 31;
    return ((lane / 16) * 2) + (l % 2);
}

__device__ __forceinline__ void load_qp(half2 (&dst)[4], const half2 * __restrict__ base, int stride) {
    const int row = qp_get_i();
#pragma unroll
    for (int l = 0; l < 4; ++l) { dst[l] = base[row * stride + l]; }
}

__device__ __forceinline__ void load_k(half2 (&dst)[4], const half2 * __restrict__ base, int stride) {
    const int row = k_get_i();
#pragma unroll
    for (int l = 0; l < 4; ++l) { dst[l] = base[row * stride + l]; }
}

__device__ __forceinline__ void load_v(half2 (&dst)[4], const half2 * __restrict__ base, int stride) {
#pragma unroll
    for (int l = 0; l < 4; ++l) { dst[l] = base[v_get_i(l) * stride + v_get_j(l)]; }
}

// QK^T: D[32x8 f32] += Q[32x8] @ K[8x8]^T (two m8n8k4 calls, k=4 each).
__device__ __forceinline__ void mma_qk(float (&d)[8], const half2 (&q)[4], const half2 (&k)[4]) {
    const int * A = reinterpret_cast<const int *>(q);
    const int * B = reinterpret_cast<const int *>(k);
    int * D       = reinterpret_cast<int *>(d);
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3]), "+r"(D[4]), "+r"(D[5]), "+r"(D[6]), "+r"(D[7])
        : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1]));
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3]), "+r"(D[4]), "+r"(D[5]), "+r"(D[6]), "+r"(D[7])
        : "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]));
}

// PV: PV[32x8 f16 out] += P[32x8] @ V[8x8] (row.row f16 accumulate).
__device__ __forceinline__ void mma_pv(half2 (&pv)[4], const half2 (&p)[4], const half2 (&v)[4]) {
    const int * P = reinterpret_cast<const int *>(p);
    const int * V = reinterpret_cast<const int *>(v);
    int * PV      = reinterpret_cast<int *>(pv);
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%0,%1,%2,%3};"
        : "+r"(PV[0]), "+r"(PV[1]), "+r"(PV[2]), "+r"(PV[3])
        : "r"(P[0]), "r"(P[1]), "r"(V[0]), "r"(V[1]));
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%0,%1,%2,%3};"
        : "+r"(PV[0]), "+r"(PV[1]), "+r"(PV[2]), "+r"(PV[3])
        : "r"(P[2]), "r"(P[3]), "r"(V[2]), "r"(V[3]));
}

// Post-softmax D tile -> PV "A" fragment: register-only, one warp shuffle
// (Volta needs no real transpose here, Turing+ would).
__device__ __forceinline__ void softmax_to_half2(half2 (&p)[4], const float (&d)[8]) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 4) {
        p[l0 / 2 + 0] = __floats2half2_rn(d[l0 + 0], d[l0 + 1]);
        p[l0 / 2 + 1] = __floats2half2_rn(d[l0 + 2], d[l0 + 3]);
        const int swap = l0 / 2 + (((lane % 4) / 2) ^ 1);
        p[swap]        = __shfl_xor_sync(0xFFFFFFFFu, p[swap], 2, 32);
    }
}

#endif // __CUDA_ARCH__ == 700

// ---------------------------------------------------------------------------
// Decode kernel.
//
// grid  = (hkv, splits); block = 128 (4 warps).
// rows  = q_len * gqa query rows for ONE (kv_head, batch) — the GQA group is
//         packed so each KV byte is read exactly once per split (the stock
//         vec path re-walks the cache once per query head: gqa x traffic).
// partial_acc[split][row_global][D], partial_m/l[split][row_global] f32;
// row_global = head_q * q_len + token, head_q = kv_head * gqa + c.
// ---------------------------------------------------------------------------
constexpr int kDecBr        = 32;   // rows per pass (one Volta tile)
constexpr int kDecBc        = 16;   // keys per tile (two 8-key QK groups)
constexpr int kDecD         = 256;
constexpr int kDecWarps     = 4;
constexpr int kDecThreads   = kDecWarps * 32;
constexpr int kDecDSlice    = kDecD / kDecWarps;
constexpr int kDecDChunks   = kDecD / 8;        // 32 (QK contraction, unsplit)
constexpr int kDecDLocal    = kDecDSlice / 8;   // 8  (PV chunks per dim warp)
constexpr int kDecPVChunks  = kDecBc / 8;       // 2
constexpr int kDecSmemPad   = 8;
constexpr int kDecSmemStride = kDecD + kDecSmemPad;

// q4_0 in-kernel dequant (8 consecutive dims, one aligned 4-byte nibble load).
// Rounding semantics are pinned to convert.cu dequantize_block_q4_0 /
// fattn-sm70-d256-kernel.cuh sm70_q4_dequant_group: value = (q-8)*scale with
// the product exact in f32, one RN to half - bit-identical to the staged path.
// Block layout: [f16 scale (2B)][32 x 4-bit (16B)] = 18B per 32 dims; row_base
// = cache + token*row_stride + head*head_stride (byte strides, [ctx][head][blk]).
// Canonical ggml q4_0 layout (ggml-quants.c writer / convert.cu reader agree):
// qs byte b holds element b in the LOW nibble and element b+16 in the HIGH
// nibble. An 8-aligned dim chunk never straddles the 16 boundary, so a chunk
// is either all-LOW (d&16==0) or all-HIGH nibbles of 8 consecutive bytes.
// Rounding matches convert.cu under nvcc FMA contraction: RN(exact d*(q-8)).
__device__ __forceinline__ void sm70_dec_q4_dequant8(
        const uint8_t * __restrict__ row_base, int d, half out[8]) {
    const uint8_t * blk  = row_base + (d >> 5) * 18;
    const float     s    = __half2float(*reinterpret_cast<const half *>(blk));
    const int       off  = 2 + (d & 15);        // first byte of this 8-dim chunk
    uint64_t        b8   = 0;                    // 8 bytes via 4 aligned u16 loads
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        b8 |= (uint64_t) *reinterpret_cast<const uint16_t *>(blk + off + 2 * k) << (16 * k);
    }
    const bool low = (d & 16) == 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint8_t byte = (uint8_t) ((b8 >> (8 * j)) & 0xFF);
        const int     q    = low ? (byte & 0xF) : (byte >> 4);
        out[j] = __float2half_rn(((float) q - 8.0f) * s);
    }
}

template <int GQA, bool Kq4 = false, bool Vq4 = false>
__global__ void __launch_bounds__(kDecThreads, 2) sm70_d256_decode_kernel(
        const float * __restrict__ q,             // f32 (D, q_len, heads_q, 1)
        int64_t q_row_stride, int64_t q_head_stride,
        const half * __restrict__ k, const half * __restrict__ v,
        int64_t k_row_stride, int64_t k_head_stride,
        int64_t v_row_stride, int64_t v_head_stride,
        // Under Kq4/Vq4 the k/v pointers are RAW q4_0 block bytes and the
        // row/head strides are BYTES ([ctx][head][block]); otherwise f16
        // element strides. Same convention as the prefill kernel q4-direct.
        int q_len, int kv_len, float scale, int splits,
        const half * __restrict__ mask_last_q, int64_t mask_q_stride,
        float * __restrict__ partial_acc, float * __restrict__ partial_m,
        float * __restrict__ partial_l) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
    __shared__ __align__(16) half q_s[kDecBr * kDecSmemStride];
    __shared__ __align__(16) half k_s[kDecBc * kDecSmemStride];
    __shared__ __align__(16) half v_s[kDecBc * kDecSmemStride];
    __shared__ float scores_s[kDecPVChunks][8][32];

    constexpr float Log2E   = 1.4426950408889634074f;
    constexpr unsigned Full = 0xffffffffu;

    const int kv_head = blockIdx.x;
    const int split   = blockIdx.y;
    const int tid     = threadIdx.x;
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int dim_warp = warp;                    // DimSplit == warps

    const int rows_total = q_len * GQA;           // rows for THIS kv head

    // Live KV length from the mask CONTENT (decode masks span the full cache
    // capacity; keys past the boundary are -inf). The last query row of a
    // bottom-right causal mask sees every live key. Split boundaries stay
    // capacity-derived (static grid, CUDA-graph friendly); only the per-CTA
    // end bound and the causal base use the live length.
    // Binary search for the live boundary: the causal last-query row is
    // monotone (valid ... valid, -inf ... -inf), so log2(capacity) broadcast
    // probes replace the linear scan (which re-read 458KB per CTA — 44MB/step
    // across the grid at 229k capacity, a measured ~10% decode tax).
    int live = 0;
    {
        int lo = 0, hi = kv_len;   // invariant: mask[lo-1] valid, mask[hi] invalid
        while (lo < hi) {
            const int mid = (lo + hi) >> 1;
            if (__half2float(mask_last_q[(int64_t) mid * mask_q_stride]) > -30000.0f) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        live = lo;
    }
    const int kv_offset = live - q_len;           // bottom-right causal base

    // Even split over the CAPACITY (static bounds); every CTA writes its rows
    // regardless (neutral partials when its range is empty) so the merge never
    // reads uninitialized memory.
    const int units   = (kv_len + splits - 1) / splits;
    const int k_start = split * units;
    const int k_end   = min(k_start + units, live);

    const int64_t k_head_off = (int64_t) kv_head * k_head_stride;
    const int64_t v_head_off = (int64_t) kv_head * v_head_stride;

    // per-row stats live in the two softmax rows this thread owns
    // (m/l for the CURRENT row pass; reset per pass).
    float m_lo = -CUDART_INF_F, m_hi = -CUDART_INF_F;
    float l_lo = 0.0f, l_hi = 0.0f;
    float acc_f[kDecDLocal][8] = {};

    for (int row_base = 0; row_base < rows_total; row_base += kDecBr) {
        const int rows_here = min(kDecBr, rows_total - row_base);
        m_lo = m_hi = -CUDART_INF_F;
        l_lo = l_hi = 0.0f;
#pragma unroll
        for (int c = 0; c < kDecDLocal; ++c) {
#pragma unroll
            for (int j = 0; j < 8; ++j) { acc_f[c][j] = 0.0f; }
        }

        // Stage Q rows for this pass: f32 input -> f16 q_s (row = token*GQA + c).
        for (int chunk = tid; chunk < kDecBr * (kDecD / 8); chunk += kDecThreads) {
            const int r    = chunk / (kDecD / 8);
            const int d    = (chunk - r * (kDecD / 8)) * 8;
            const int r_gl = row_base + r;
            half * dst     = &q_s[r * kDecSmemStride + d];
            if (r < rows_here) {
                const int token  = r_gl / GQA;
                const int c      = r_gl - token * GQA;
                const int head_q = kv_head * GQA + c;
                const float * src = q + d + (int64_t) token * q_row_stride
                                      + (int64_t) head_q * q_head_stride;
#pragma unroll
                for (int e = 0; e < 8; ++e) { dst[e] = __float2half_rn(src[e]); }
            } else {
                *(int4 *) dst = make_int4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        for (int k0 = k_start; k0 < k_end; k0 += kDecBc) {
            // Stage K/V (f16 at rest): raw 16B copies, zero outside [k_start,k_end).
            for (int chunk = tid; chunk < kDecBc * (kDecD / 8); chunk += kDecThreads) {
                const int kl = chunk / (kDecD / 8);
                const int d  = (chunk - kl * (kDecD / 8)) * 8;
                const int key = k0 + kl;
                half * kd = &k_s[kl * kDecSmemStride + d];
                half * vd = &v_s[kl * kDecSmemStride + d];
                if (key < k_end) {
                    if constexpr (Kq4) {
                        const uint8_t * row =
                            reinterpret_cast<const uint8_t *>(k) + (int64_t) key * k_row_stride + k_head_off;
                        half tmp[8];
                        sm70_dec_q4_dequant8(row, d, tmp);
                        *(int4 *) kd = *reinterpret_cast<const int4 *>(tmp);
                    } else {
                        *(int4 *) kd = *(const int4 *) (k + ((int64_t) key * k_row_stride + k_head_off + d));
                    }
                    if constexpr (Vq4) {
                        const uint8_t * row =
                            reinterpret_cast<const uint8_t *>(v) + (int64_t) key * v_row_stride + v_head_off;
                        half tmp[8];
                        sm70_dec_q4_dequant8(row, d, tmp);
                        *(int4 *) vd = *reinterpret_cast<const int4 *>(tmp);
                    } else {
                        *(int4 *) vd = *(const int4 *) (v + ((int64_t) key * v_row_stride + v_head_off + d));
                    }
                } else {
                    *(int4 *) kd = make_int4(0, 0, 0, 0);
                    *(int4 *) vd = make_int4(0, 0, 0, 0);
                }
            }
            __syncthreads();

            // QK^T on warps 0..1, scores to smem.
            if (warp < kDecPVChunks) {
                float d_score[8] = {};
#pragma unroll
                for (int c = 0; c < kDecDChunks; ++c) {
                    half2 qf[4], kf[4];
                    load_qp(qf, reinterpret_cast<const half2 *>(&q_s[c * 8]), kDecSmemStride / 2);
                    load_k(kf, reinterpret_cast<const half2 *>(&k_s[warp * 8 * kDecSmemStride + c * 8]),
                           kDecSmemStride / 2);
                    mma_qk(d_score, qf, kf);
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) { scores_s[warp][j][lane] = d_score[j]; }
            }
            __syncthreads();

            // All four dim warps: same softmax recurrence (redundant by design),
            // PV over own DSlice.
            auto prepare = [&](int sub, half2 (&p)[4]) {
                const int sub_k0 = k0 + sub * 8;
                float ds[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) { ds[j] = scores_s[sub][j][lane]; }
                // causal + row bound; this thread's 8 regs span rows (l&2)==0 / ==2
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    const int row = d_get_i(l);
                    const int col = d_get_j(l);
                    const int key = sub_k0 + col;
                    const int token = (row_base + row) / GQA;
                    const bool ok = row < rows_here && key < k_end && key <= kv_offset + token;
                    ds[l] = ok ? ds[l] * scale : -CUDART_INF_F;
                }
                // per-row max over the 8 keys: two rows live in this thread
                // (l&2==0 -> lo, l&2!=0 -> hi); tid and tid^2 hold the rest.
                float bm_lo = fmaxf(fmaxf(ds[0], ds[1]), fmaxf(ds[4], ds[5]));
                float bm_hi = fmaxf(fmaxf(ds[2], ds[3]), fmaxf(ds[6], ds[7]));
                bm_lo = fmaxf(bm_lo, __shfl_xor_sync(Full, bm_lo, 2, 32));
                bm_hi = fmaxf(bm_hi, __shfl_xor_sync(Full, bm_hi, 2, 32));

                const float nm_lo = fmaxf(m_lo, bm_lo);
                const float nm_hi = fmaxf(m_hi, bm_hi);
                const float a_lo = m_lo == -CUDART_INF_F ? 0.0f : exp2f((m_lo - nm_lo) * Log2E);
                const float a_hi = m_hi == -CUDART_INF_F ? 0.0f : exp2f((m_hi - nm_hi) * Log2E);
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    const float nm = ((l & 2) == 0) ? nm_lo : nm_hi;
                    ds[l] = (nm > -CUDART_INF_F && ds[l] > -CUDART_INF_F)
                                ? exp2f((ds[l] - nm) * Log2E) : 0.0f;
                }
                float bl_lo = ds[0] + ds[1] + ds[4] + ds[5];
                float bl_hi = ds[2] + ds[3] + ds[6] + ds[7];
                bl_lo += __shfl_xor_sync(Full, bl_lo, 2, 32);
                bl_hi += __shfl_xor_sync(Full, bl_hi, 2, 32);

                l_lo = l_lo * a_lo + bl_lo;
                l_hi = l_hi * a_hi + bl_hi;
                m_lo = nm_lo;
                m_hi = nm_hi;
                softmax_to_half2(p, ds);
                return ((lane & 2) == 0) ? a_lo : a_hi;
            };

#pragma unroll
            for (int sub = 0; sub < kDecPVChunks; ++sub) {
                half2 p[4];
                const float alpha = prepare(sub, p);
#pragma unroll
                for (int c = 0; c < kDecDLocal; ++c) {
                    half2 vf[4];
                    load_v(vf, reinterpret_cast<const half2 *>(
                                   &v_s[sub * 8 * kDecSmemStride + dim_warp * kDecDSlice + c * 8]),
                           kDecSmemStride / 2);
                    half2 pv[4] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
                    mma_pv(pv, p, vf);
#pragma unroll
                    for (int n = 0; n < 4; ++n) {
                        const float2 contrib = __half22float2(pv[n]);
                        acc_f[c][2 * n + 0] = acc_f[c][2 * n + 0] * alpha + contrib.x;
                        acc_f[c][2 * n + 1] = acc_f[c][2 * n + 1] * alpha + contrib.y;
                    }
                }
            }
            __syncthreads();
        }

        // Partials. row_global = head_q * q_len + token; the split stride is
        // the GLOBAL rows count across ALL kv heads (gridDim.x == hkv), not
        // this CTA's own per-head rows — otherwise split planes collide.
        const int rows_pad = gridDim.x * rows_total;
        const int row = lane;
        if (dim_warp == 0 && row < rows_here) {
            const int token = (row_base + row) / GQA;
            const int c     = (row_base + row) - token * GQA;
            const int row_gl = (kv_head * GQA + c) * q_len + token;
            const int64_t st = (int64_t) split * rows_pad + row_gl;
            partial_m[st] = ((lane & 2) == 0) ? m_lo : m_hi;
            partial_l[st] = ((lane & 2) == 0) ? l_lo : l_hi;
        }
        // The f16 mma_pv accumulator fragment is row=lane with 8 CONSECUTIVE
        // dims in register order (the "Q tile" layout) — NOT the QK^T D-tile
        // mapping. Each dim warp writes its own DSlice column range.
        if (lane < rows_here) {
            const int token  = (row_base + lane) / GQA;
            const int cidx   = (row_base + lane) - token * GQA;
            const int row_gl = (kv_head * GQA + cidx) * q_len + token;
            float * out_row  = partial_acc + ((int64_t) split * rows_pad + row_gl) * kDecD
                             + dim_warp * kDecDSlice;
#pragma unroll
            for (int c = 0; c < kDecDLocal; ++c) {
#pragma unroll
                for (int i = 0; i < 8; ++i) { out_row[c * 8 + i] = acc_f[c][i]; }
            }
        }
        if (row_base + kDecBr < rows_total) {
            __syncthreads();      // q_s reused by the next pass
        }
    }
#endif // __CUDA_ARCH__ == 700
}

// ---------------------------------------------------------------------------
// N-split merge: partials -> normalized f32 dst (D, q_len, heads_q, batch).
// grid.x = rows_total (head_q * q_len per batch... single batch P1); 128 thr.
// ---------------------------------------------------------------------------
constexpr float Log2E_MERGE = 1.4426950408889634074f;

__global__ void sm70_d256_decode_merge_kernel(
        const float * __restrict__ partial_acc,
        const float * __restrict__ partial_m,
        const float * __restrict__ partial_l,
        float * __restrict__ dst,
        int rows_total, int splits,
        int64_t dst_row_stride, int64_t dst_head_stride, int q_len) {
    const int row = blockIdx.x;      // row_global = head_q * q_len + token
    if (row >= rows_total) { return; }
    const int head_q = row / q_len;
    const int token  = row - head_q * q_len;

    float acc[2] = {0.0f, 0.0f};     // this thread's dims (tid, tid + 128)
    float m_run  = -CUDART_INF_F;
    float l_run  = 0.0f;
    for (int s = 0; s < splits; ++s) {
        const int64_t st = (int64_t) s * rows_total + row;
        const float ms = partial_m[st];
        const float ls = partial_l[st];
        const float m_new = fmaxf(m_run, ms);
        const float a_run = m_run == -CUDART_INF_F ? 0.0f : exp2f((m_run - m_new) * Log2E_MERGE);
        const float a_s   = ms == -CUDART_INF_F ? 0.0f : exp2f((ms - m_new) * Log2E_MERGE);
        const float * src = partial_acc + st * kDecD;
        acc[0] = acc[0] * a_run + src[threadIdx.x]        * a_s;
        acc[1] = acc[1] * a_run + src[threadIdx.x + 128]  * a_s;
        l_run  = l_run * a_run + ls * a_s;
        m_run  = m_new;
    }
    const float inv_l = l_run > 0.0f ? 1.0f / l_run : 0.0f;
    float * out = dst + (int64_t) token * dst_row_stride + (int64_t) head_q * dst_head_stride;
    out[threadIdx.x]       = acc[0] * inv_l;
    out[threadIdx.x + 128] = acc[1] * inv_l;
}

} // namespace sm70_d256_decode
