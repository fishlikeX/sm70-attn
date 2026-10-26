// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// sm70_attn_micro.cu — standalone microbenchmark for the SM70 D256 prefill
// flash-attention kernel (fattn-sm70-d256-kernel.cuh).
//
// Why: the server-level t/s numbers fold the kernel, the q4->f16 mirror
// dequant, Q staging, the SplitKV3 merge and the scatter into one number.
// This harness launches the attention kernel alone with the exact production
// geometry so a kernel change can be judged on its own.
//
// Geometry (Qwen3.8-27B dense full-attention layers):
//   D = 256, heads_q = 24, heads_kv = 4 (GQA 6), q_len = 4096 (ub), kv_len
//   = 174080 at the deep end, kv_offset = kv_len - q_len.
//
// Modes:
//   --kv N        kv_len (default 174080)
//   --q N         q_len  (default 4096)
//   --split 0|1   dense kernel vs SplitKV3 (default 1 = production at 174k)
//   --q4K 0|1     K read as raw q4_0 blocks (Kq4 template arg)
//   --q4V 0|1     V read as raw q4_0 blocks (Vq4 template arg)
//   --iters N     timed launches (default 5)
//   --check       small-shape numeric check vs a naive reference kernel
//   --full        correctness: q4-direct vs staged-f16 must be bit-identical
//
// Build: D:\zcode-build\build_amicro.bat
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// same predefine the production launcher uses (softmax.h consumes it)
#ifndef M_LOG2E
#define M_LOG2E 1.4426950408889634f
#endif

// ---- timing-only ablations (never part of the production build) -----------
// ABL_NOBAR: drop every __syncthreads in the kernel. Results are garbage; the
// runtime difference is the barrier/serialization cost.
#ifdef ABL_NOBAR
#define __syncthreads() do { } while (0)
#endif

// Each ablation variant is a generated copy of the production header
// (verify/ablate.py) with one inner-loop phase neutralised.
#if defined(ABL_HEADER_NOV)
#include "abl_nov.cuh"
#elif defined(ABL_HEADER_NOQK)
#include "abl_noqk.cuh"
#elif defined(ABL_HEADER_NOMASK)
#include "abl_nomask.cuh"
#elif defined(ABL_HEADER_NOEXP)
#include "abl_noexp.cuh"
#elif defined(ABL_HEADER_NORESCALE)
#include "abl_norescale.cuh"
#elif defined(ABL_HEADER_NOGEMM)
#include "abl_nogemm.cuh"
#elif defined(ABL_HEADER_BM32)
#include "abl_bm32.cuh"
#else
#include "fattn-sm70-d256-kernel.cuh"
#endif

using Traits = FLASH_NAMESPACE::Sm70D256SplitDTraits;
using El = cutlass::half_t;

static constexpr int D = 256;

static void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        printf("FAIL %s: %s\n", what, cudaGetErrorString(e));
        exit(1);
    }
}
#define CK(x) ck((x), #x)

struct Args {
    int kv_len  = 174080;
    int q_len   = 4096;
    int split   = 1;
    int q4K     = 0;
    int q4V     = 0;
    int iters   = 5;
    int hkv     = 4;
    int gqa     = 6;
    bool check  = false;
    bool full   = false;
};

// ---------------------------------------------------------------------------
// host-side q4_0 packing (canonical ggml layout: byte b = elem b low nibble,
// elem b+16 high nibble; f16 scale per 32-dim block)
// ---------------------------------------------------------------------------
static void pack_q4_row(const float * x, int n, uint8_t * out) {
    // n must be a multiple of 32; out gets n/32 * 18 bytes
    for (int b = 0; b < n / 32; ++b) {
        const float * xb = x + b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) { amax = fmaxf(amax, fabsf(xb[i])); }
        const float d = amax / 7.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        __half hd = __float2half(d);
        memcpy(out + b * 18, &hd, 2);
        uint8_t * qs = out + b * 18 + 2;
        int q[32];
        for (int i = 0; i < 32; ++i) {
            int v = (int) lrintf(xb[i] * id) + 8;
            q[i] = std::min(std::max(v, 0), 15);
        }
        for (int i = 0; i < 16; ++i) {
            qs[i] = (uint8_t) ((q[i] & 0xF) | ((q[i + 16] & 0xF) << 4));
        }
    }
}

static void dequant_q4_row(const uint8_t * blk, int n, float * out) {
    for (int b = 0; b < n / 32; ++b) {
        __half hd;
        memcpy(&hd, blk + b * 18, 2);
        const float d = __half2float(hd);
        const uint8_t * qs = blk + b * 18 + 2;
        for (int i = 0; i < 16; ++i) {
            out[b * 32 + i]      = d * (float) ((qs[i] & 0xF) - 8);
            out[b * 32 + i + 16] = d * (float) ((qs[i] >> 4) - 8);
        }
    }
}

// ---------------------------------------------------------------------------
// naive reference: one warp per (row, head), fp32 online softmax
// ---------------------------------------------------------------------------
__global__ void ref_fa_kernel(
        const __half * __restrict__ q,   // [head][row][D]
        const __half * __restrict__ k,   // [ctx][hkv][D]
        const __half * __restrict__ v,   // [ctx][hkv][D]
        float * __restrict__ out,        // [row][head][D]
        int q_len, int kv_len, int heads_q, int hkv,
        int kv_offset, float scale) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int64_t r  = (int64_t) blockIdx.x * (blockDim.x >> 5) + warp;
    if (r >= (int64_t) q_len * heads_q) { return; }
    const int row  = (int) (r / heads_q);
    const int hq   = (int) (r % heads_q);
    const int hk   = hq / (heads_q / hkv);

    const int visible = kv_offset + row + 1;

    extern __shared__ float sq[];   // D floats
    for (int d = lane; d < D; d += 32) { sq[d] = __half2float(q[((int64_t) hq * q_len + row) * D + d]); }
    __syncwarp();

    float m = -INFINITY;
    float l = 0.0f;
    float acc = 0.0f;   // this lane owns d = lane, lane+32, ... -> accumulate into regs? too many
    // D/32 = 8 values per lane
    float accv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { accv[i] = 0.0f; }

    for (int j = 0; j < visible; ++j) {
        const __half * krow = k + ((int64_t) j * hkv + hk) * D;
        float part = 0.0f;
        for (int d = lane; d < D; d += 32) {
            part += sq[d] * __half2float(krow[d]);
        }
        for (int off = 16; off > 0; off >>= 1) { part += __shfl_down_sync(0xffffffffu, part, off); }
        part = __shfl_sync(0xffffffffu, part, 0);
        const float s = part * scale;
        const float m_new = fmaxf(m, s);
        const float corr = expf(m - m_new);
        const float p = expf(s - m_new);
        l = l * corr + p;
        acc = acc * corr;
        const __half * vrow = v + ((int64_t) j * hkv + hk) * D;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int d = lane + i * 32;
            accv[i] += p * __half2float(vrow[d]);
        }
        m = m_new;
    }
    (void) acc;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int d = lane + i * 32;
        out[((int64_t) row * heads_q + hq) * D + d] = accv[i] / l;
    }
}

struct Buffers {
    // Q: [head][row][D] f16, q_head_stride = q_pad*D (the kernel's layout)
    std::vector<__half> q;
    std::vector<float>  out;
};

template <bool SplitKV3, bool Kq4, bool Vq4>
static void launch_attn(
        const __half * q, const void * k, const void * v, float * out,
        float * partial_out, float * partial_max, float * partial_sum,
        int q_pad, int kv_len, int heads_q, int hkv, int kv_offset,
        const __half * mask_last_q, float softmax_scale_log2,
        int nb_sequences, cudaStream_t stream) {
    auto kfn = FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<El, false, float, SplitKV3, Kq4, Vq4>;
    CK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, Traits::kSmemBytes));
    const dim3 block(Traits::kNThreads);
    const dim3 grid(q_pad / Traits::kBlockM,
                    SplitKV3 ? (unsigned) (nb_sequences * 3) : (unsigned) nb_sequences,
                    heads_q);
    kfn<<<grid, block, Traits::kSmemBytes, stream>>>(
            (const El *) q, (const El *) k, (const El *) v, out,
            /*q_batch_stride*/ (int64_t) heads_q * q_pad * D,
            /*q_row_stride  */ D,
            /*q_head_stride */ (int64_t) q_pad * D,
            /*k_outer_stride*/ 0,
            /*k_row_stride  */ Kq4 ? (int) (hkv * 144) : (int) (hkv * D),
            /*k_head_stride */ Kq4 ? 144 : D,
            /*v_outer_stride*/ 0,
            /*v_row_stride  */ Vq4 ? (int) (hkv * 144) : (int) (hkv * D),
            /*v_head_stride */ Vq4 ? 144 : D,
            q_pad, kv_len, heads_q, hkv, kv_offset,
            (const half *) mask_last_q, softmax_scale_log2,
            nullptr, 0, 0,
            partial_out, partial_max, partial_sum);
    CK(cudaGetLastError());
}

static void print_attrs(const char * tag) {
    cudaFuncAttributes at{};
    auto kfn = FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<El, false, float, true, false, false>;
    CK(cudaFuncGetAttributes(&at, kfn));
    printf("[attrs %s] regs=%d smem_static=%zu max_threads=%d localB=%zu\n",
           tag, at.numRegs, at.sharedSizeBytes, at.maxThreadsPerBlock, at.localSizeBytes);
}

int main(int argc, char ** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const char * s = argv[i];
        auto next = [&]() { return i + 1 < argc ? argv[++i] : ""; };
        if      (!strcmp(s, "--kv"))    { a.kv_len = atoi(next()); }
        else if (!strcmp(s, "--q"))     { a.q_len  = atoi(next()); }
        else if (!strcmp(s, "--split")) { a.split  = atoi(next()); }
        else if (!strcmp(s, "--q4K"))   { a.q4K    = atoi(next()); }
        else if (!strcmp(s, "--q4V"))   { a.q4V    = atoi(next()); }
        else if (!strcmp(s, "--iters")) { a.iters  = atoi(next()); }
        else if (!strcmp(s, "--check")) { a.check  = true; }
        else if (!strcmp(s, "--full"))  { a.full   = true; }
        else { printf("unknown arg %s\n", s); return 2; }
    }
    if (a.kv_len % 32 != 0) { printf("kv_len must be a multiple of 32\n"); return 2; }
    const int BM = Traits::kBlockM;
    const int q_pad = ((a.q_len + BM - 1) / BM) * BM;
    const int heads_q = a.hkv * a.gqa;
    const int kv_offset = std::max(a.kv_len - a.q_len, 0);

    CK(cudaSetDevice(1));   // second V100 (CUDA order; device 0 is the other V100)
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 1));
    printf("[dev] %s  sm_%d%d  smem/SM=%d clock_cap=%d kHz\n",
           prop.name, prop.major, prop.minor, prop.sharedMemPerMultiprocessor, prop.clockRate);
    printf("[cfg] q_len=%d q_pad=%d kv_len=%d kv_offset=%d heads_q=%d hkv=%d gqa=%d split=%d q4K=%d q4V=%d\n",
           a.q_len, q_pad, a.kv_len, kv_offset, heads_q, a.hkv, a.gqa, a.split, a.q4K, a.q4V);
    print_attrs("split3");

    // ---- allocate + fill ---------------------------------------------------
    const size_t q_elems = (size_t) heads_q * q_pad * D;
    __half * d_q = nullptr;
    CK(cudaMalloc(&d_q, q_elems * sizeof(__half)));
    std::vector<__half> h_q(q_elems);
    srand(42);
    auto rnd = []() { return (float) (rand() % 20001 - 10000) / 10000.0f; };
    for (size_t i = 0; i < q_elems; ++i) { h_q[i] = __float2half(rnd() * 0.07f); }
    CK(cudaMemcpy(d_q, h_q.data(), q_elems * sizeof(__half), cudaMemcpyHostToDevice));

    const size_t kv_tokens = (size_t) a.kv_len * a.hkv;
    // f16 K/V ([ctx][hkv][D]) and/or raw q4_0 blocks ([ctx][hkv][8 blocks])
    const size_t f16_bytes = kv_tokens * D * sizeof(__half);
    const size_t q4_bytes  = kv_tokens * 144;
    __half * d_k16 = nullptr; __half * d_v16 = nullptr;
    uint8_t * d_k4 = nullptr; uint8_t * d_v4 = nullptr;
    std::vector<__half> h_k16, h_v16;
    if (!a.q4K || a.full) { CK(cudaMalloc(&d_k16, f16_bytes)); h_k16.resize(kv_tokens * D); }
    if (!a.q4V || a.full) { CK(cudaMalloc(&d_v16, f16_bytes)); h_v16.resize(kv_tokens * D); }
    if (a.q4K) { CK(cudaMalloc(&d_k4, q4_bytes)); }
    if (a.q4V) { CK(cudaMalloc(&d_v4, q4_bytes)); }

    std::vector<uint8_t> h_k4, h_v4;
    if (a.q4K || a.full) { h_k4.resize(q4_bytes); }
    if (a.q4V || a.full) { h_v4.resize(q4_bytes); }

    std::vector<float> row(D);
    for (size_t t = 0; t < kv_tokens; ++t) {
        for (int d = 0; d < D; ++d) { row[d] = rnd() * 0.07f; }
        if (!a.q4K || a.full) {
            for (int d = 0; d < D; ++d) { h_k16[t * D + d] = __float2half(row[d]); }
        }
        if (a.q4K || a.full) { pack_q4_row(row.data(), D, h_k4.data() + t * 144); }
        for (int d = 0; d < D; ++d) { row[d] = rnd() * 0.07f; }
        if (!a.q4V || a.full) {
            for (int d = 0; d < D; ++d) { h_v16[t * D + d] = __float2half(row[d]); }
        }
        if (a.q4V || a.full) { pack_q4_row(row.data(), D, h_v4.data() + t * 144); }
    }
    if (d_k16) { CK(cudaMemcpy(d_k16, h_k16.data(), f16_bytes, cudaMemcpyHostToDevice)); }
    if (d_v16) { CK(cudaMemcpy(d_v16, h_v16.data(), f16_bytes, cudaMemcpyHostToDevice)); }
    if (d_k4)  { CK(cudaMemcpy(d_k4,  h_k4.data(),  q4_bytes,  cudaMemcpyHostToDevice)); }
    if (d_v4)  { CK(cudaMemcpy(d_v4,  h_v4.data(),  q4_bytes,  cudaMemcpyHostToDevice)); }

    const size_t out_elems = (size_t) q_pad * heads_q * D;
    float * d_out = nullptr;
    CK(cudaMalloc(&d_out, out_elems * sizeof(float)));

    const size_t rows3 = (size_t) q_pad * heads_q;
    float * d_po = nullptr; float * d_pm = nullptr; float * d_pl = nullptr;
    if (a.split) {
        CK(cudaMalloc(&d_po, 3 * rows3 * D * sizeof(float)));
        CK(cudaMalloc(&d_pm, 3 * rows3 * sizeof(float)));
        CK(cudaMalloc(&d_pl, 3 * rows3 * sizeof(float)));
    }
    // the kernel only reads this row to resolve kv_offset; all-zero == live
    __half * d_mask_last = nullptr;
    CK(cudaMalloc(&d_mask_last, (size_t) a.kv_len * sizeof(__half)));
    CK(cudaMemset(d_mask_last, 0, (size_t) a.kv_len * sizeof(__half)));

    const float scale = 1.0f / sqrtf((float) D);
    const float sslog2 = scale * 1.4426950408889634f;

    cudaStream_t stream;
    CK(cudaStreamCreate(&stream));

    auto run = [&](int iters, bool timed, float * out) {
        auto go = [&]() {
            if (a.split) {
                if (a.q4K && a.q4V)      launch_attn<true,  true,  true >(d_q, d_k4, d_v4, out, d_po, d_pm, d_pl, q_pad, a.kv_len, heads_q, a.hkv, kv_offset, d_mask_last, sslog2, 1, stream);
                else if (a.q4K)          launch_attn<true,  true,  false>(d_q, d_k4, d_v16, out, d_po, d_pm, d_pl, q_pad, a.kv_len, heads_q, a.hkv, kv_offset, d_mask_last, sslog2, 1, stream);
                else if (a.q4V)          launch_attn<true,  false, true >(d_q, d_k16, d_v4, out, d_po, d_pm, d_pl, q_pad, a.kv_len, heads_q, a.hkv, kv_offset, d_mask_last, sslog2, 1, stream);
                else                     launch_attn<true,  false, false>(d_q, d_k16, d_v16, out, d_po, d_pm, d_pl, q_pad, a.kv_len, heads_q, a.hkv, kv_offset, d_mask_last, sslog2, 1, stream);
            } else {
                if (a.q4K && a.q4V)      launch_attn<false, true,  true >(d_q, d_k4, d_v4, out, nullptr, nullptr, nullptr, q_pad, a.kv_len, heads_q, a.hkv, kv_offset, d_mask_last, sslog2, 1, stream);
                else if (a.q4K)          launch_attn<false, true,  false>(d_q, d_k4, d_v16, out, nullptr, nullptr, nullptr, q_pad, a.kv_len, heads_q, a.hkv, kv_offset, d_mask_last, sslog2, 1, stream);
                else if (a.q4V)          launch_attn<false, false, true >(d_q, d_k16, d_v4, out, nullptr, nullptr, nullptr, q_pad, a.kv_len, heads_q, a.hkv, kv_offset, d_mask_last, sslog2, 1, stream);
                else                     launch_attn<false, false, false>(d_q, d_k16, d_v16, out, nullptr, nullptr, nullptr, q_pad, a.kv_len, heads_q, a.hkv, kv_offset, d_mask_last, sslog2, 1, stream);
            }
            CK(cudaStreamSynchronize(stream));
        };
        go();  // warmup (includes smem opt-in / module load)
        if (!timed) { return 0.0f; }
        std::vector<float> ms;
        for (int it = 0; it < iters; ++it) {
            cudaEvent_t e0, e1;
            CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
            CK(cudaEventRecord(e0, stream));
            go();
            CK(cudaEventRecord(e1, stream));
            CK(cudaEventSynchronize(e1));
            float t = 0.0f;
            CK(cudaEventElapsedTime(&t, e0, e1));
            ms.push_back(t);
            CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
        }
        std::sort(ms.begin(), ms.end());
        return ms[ms.size() / 2];
    };

    // ---- numeric check -----------------------------------------------------
    if (a.check || a.full) {
        if (a.split) {
            printf("-- numeric check needs --split 0 (reference has no split-merge); skipping\n");
        } else {
            CK(cudaMemset(d_out, 0, out_elems * sizeof(float)));
            run(1, false, d_out);
            std::vector<float> got(out_elems);
            CK(cudaMemcpy(got.data(), d_out, out_elems * sizeof(float), cudaMemcpyDeviceToHost));

            // reference dequantizes the *same* q4 blocks when q4-direct is used
            std::vector<__half> rk = h_k16, rv = h_v16;
            if (a.q4K) {
                rk.resize(kv_tokens * D);
                for (size_t t = 0; t < kv_tokens; ++t) {
                    std::vector<float> tmp(D);
                    dequant_q4_row(h_k4.data() + t * 144, D, tmp.data());
                    for (int d = 0; d < D; ++d) { rk[t * D + d] = __float2half(tmp[d]); }
                }
            }
            if (a.q4V) {
                rv.resize(kv_tokens * D);
                for (size_t t = 0; t < kv_tokens; ++t) {
                    std::vector<float> tmp(D);
                    dequant_q4_row(h_v4.data() + t * 144, D, tmp.data());
                    for (int d = 0; d < D; ++d) { rv[t * D + d] = __float2half(tmp[d]); }
                }
            }
            __half * d_rk = nullptr; __half * d_rv = nullptr;
            float * d_ref = nullptr;
            CK(cudaMalloc(&d_rk, rk.size() * sizeof(__half)));
            CK(cudaMalloc(&d_rv, rv.size() * sizeof(__half)));
            CK(cudaMalloc(&d_ref, out_elems * sizeof(float)));
            CK(cudaMemcpy(d_rk, rk.data(), rk.size() * sizeof(__half), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_rv, rv.data(), rv.size() * sizeof(__half), cudaMemcpyHostToDevice));
            const int warps_per_block = 8;
            const int64_t nrows = (int64_t) a.q_len * heads_q;
            ref_fa_kernel<<<(int) ((nrows + warps_per_block - 1) / warps_per_block), warps_per_block * 32,
                            D * sizeof(float), stream>>>(
                    d_q, d_rk, d_rv, d_ref, a.q_len, a.kv_len, heads_q, a.hkv, kv_offset, scale);
            CK(cudaGetLastError());
            CK(cudaStreamSynchronize(stream));
            std::vector<float> ref(out_elems);
            CK(cudaMemcpy(ref.data(), d_ref, out_elems * sizeof(float), cudaMemcpyDeviceToHost));

            double max_abs = 0.0, sum_abs = 0.0, ref_abs = 0.0;
            size_t n_cmp = 0;
            for (int r = 0; r < a.q_len; ++r) {
                for (int hq = 0; hq < heads_q; ++hq) {
                    for (int d = 0; d < D; ++d) {
                        const size_t i = ((size_t) r * heads_q + hq) * D + d;
                        const double diff = fabs((double) got[i] - (double) ref[i]);
                        max_abs = std::max(max_abs, diff);
                        sum_abs += diff;
                        ref_abs += fabs((double) ref[i]);
                        ++n_cmp;
                    }
                }
            }
            printf("[check] rows=%d kv=%d q4K=%d q4V=%d  max|diff|=%.5f  mean|diff|=%.6f  mean|ref|=%.5f\n",
                   a.q_len, a.kv_len, a.q4K, a.q4V, max_abs, sum_abs / n_cmp, ref_abs / n_cmp);
            CK(cudaFree(d_rk)); CK(cudaFree(d_rv)); CK(cudaFree(d_ref));
        }
    }

    // ---- timing ------------------------------------------------------------
    const float ms = run(a.iters, true, d_out);
    const double sum_keys = (double) a.q_len * kv_offset
                          + (double) a.q_len * (a.q_len + 1) / 2.0;
    const double flops = 4.0 * D * heads_q * sum_keys;    // QK + PV
    const double tflops = flops / (ms * 1e-3) / 1e12;
    const double peak = 125.0;   // V100 SXM2 fp16 tensor
    printf("[time] split=%d q4K=%d q4V=%d  %8.2f ms  %6.2f TFLOP/s  (%4.1f%% of 125)\n",
           a.split, a.q4K, a.q4V, ms, tflops, 100.0 * tflops / peak);
    printf("[work] grid=(%d,%d,%d)  %lld CTAs  n_blocks/CTA=%d  stage_kv=%s\n",
           q_pad / Traits::kBlockM, a.split ? 3 : 1, heads_q,
           (long long) (q_pad / Traits::kBlockM) * (a.split ? 3 : 1) * heads_q,
           a.split ? (a.kv_len / 32) / 3 : (a.kv_len / 32),
           (a.q4K && a.q4V) ? "none (q4-direct)" : "f16 mirror");

    if (a.full) {
        // q4-direct must reproduce the staged-f16 result bit-exactly
        CK(cudaMemset(d_out, 0, out_elems * sizeof(float)));
        run(1, false, d_out);
        std::vector<float> ref_f16(out_elems);
        CK(cudaMemcpy(ref_f16.data(), d_out, out_elems * sizeof(float), cudaMemcpyDeviceToHost));
        const bool has_f16 = h_k16.size() && h_v16.size();
        if (has_f16 && a.q4K && a.q4V) {
            // rebuild: staged f16 K/V from the same q4 blocks
            std::vector<__half> sk(kv_tokens * D), sv(kv_tokens * D);
            for (size_t t = 0; t < kv_tokens; ++t) {
                std::vector<float> tmp(D);
                dequant_q4_row(h_k4.data() + t * 144, D, tmp.data());
                for (int d = 0; d < D; ++d) { sk[t * D + d] = __float2half(tmp[d]); }
                dequant_q4_row(h_v4.data() + t * 144, D, tmp.data());
                for (int d = 0; d < D; ++d) { sv[t * D + d] = __float2half(tmp[d]); }
            }
            __half * d_sk = nullptr; __half * d_sv = nullptr;
            CK(cudaMalloc(&d_sk, sk.size() * sizeof(__half)));
            CK(cudaMalloc(&d_sv, sv.size() * sizeof(__half)));
            CK(cudaMemcpy(d_sk, sk.data(), sk.size() * sizeof(__half), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_sv, sv.data(), sv.size() * sizeof(__half), cudaMemcpyHostToDevice));
            float * o16 = nullptr;
            CK(cudaMalloc(&o16, out_elems * sizeof(float)));
            launch_attn<false, false, false>(d_q, d_sk, d_sv, o16, nullptr, nullptr, nullptr,
                                             q_pad, a.kv_len, heads_q, a.hkv, kv_offset, d_mask_last, sslog2, 1, stream);
            CK(cudaStreamSynchronize(stream));
            std::vector<float> got16(out_elems);
            CK(cudaMemcpy(got16.data(), o16, out_elems * sizeof(float), cudaMemcpyDeviceToHost));
            size_t ndiff = 0; double maxd = 0.0;
            for (size_t i = 0; i < out_elems; ++i) {
                if (memcmp(&ref_f16[i], &got16[i], 4) != 0) { ++ndiff; maxd = std::max(maxd, fabs((double) ref_f16[i] - got16[i])); }
            }
            printf("[full] q4-direct vs staged-f16: %zu/%zu differ  max|diff|=%.6f\n",
                   ndiff, out_elems, maxd);
        }
    }
    return 0;
}
