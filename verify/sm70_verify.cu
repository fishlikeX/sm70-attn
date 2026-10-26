// ============================================================================
// sm70 d256 attention — standalone cross-check harness
//
// Feeds the production kernel (fattn-sm70-d256-kernel.cuh, byte-identical to
// the one in the sm70-attn build) the EXACT same layout/strides/params the
// llama.cpp launcher (fattn-sm70-d256.cu) uses, and compares output
// element-wise against a CPU reference: softmax(QK^T/sqrt(256))·V, causal,
// GQA 6:1, head_dim 256.
//
// Row sampling: small cases compare ALL rows/heads; big cases compare a
// boundary-focused subset (tile edges 64/128/256, diagonal rows, tail rows)
// across a few heads — tile-boundary and causal-diagonal bugs show up there.
//
// OOB-GUARD (arg "o" / SM70_VERIFY_OOB=1): K/V allocated at EXACT
// [nb][hkv][kvlen][256] (no headroom). Any read past kvlen rows is an
// illegal access — visible here (crash / NaN leak) and under
// compute-sanitizer --tool memcheck.
//
// Build: see build.sh
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define FLASH_NAMESPACE flash_sm70
#include "fattn-sm70-d256-kernel.cuh"

using El = cutlass::half_t;
using Traits = FLASH_NAMESPACE::Sm70D256SplitDTraits;
static_assert(Traits::kHeadDim == 256 && Traits::kBlockM == 64 && Traits::kBlockN == 32);

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    cudaGetLastError(); \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    exit(1); } } while (0)

static uint32_t s_rng = 0x9E3779B9u;
static float frand() {
    s_rng ^= s_rng << 13; s_rng ^= s_rng >> 7; s_rng ^= s_rng << 17;
    return (float)(s_rng >> 8) / 8388608.0f - 1.0f;
}

// ---------------------------------------------------------------------------
// q4_0 helpers (8/23 q4-direct regression cases): ggml reference quantizer +
// kernel-parity dequant (exact f32 product, one RN to half — identical to
// convert.cu dequantize_block_q4_0 and the in-kernel sm70_q4_dequant_*).
// Block: [f16 scale 2B][32 x 4-bit quants 16B]; row = 8 blocks = 144B.
// ---------------------------------------------------------------------------
static void quant_row_q4_0(const float * x, uint8_t * y, int k) {
    for (int i = 0; i < k / 32; ++i) {
        float amax = 0.0f, mx = 0.0f;
        for (int j = 0; j < 32; ++j) {
            const float v = x[i * 32 + j];
            if (amax < fabsf(v)) { amax = fabsf(v); mx = v; }
        }
        const float d = mx / -8.0f;
        const float id = d ? 1.0f / d : 0.0f;
        const __half dh = __float2half(d);
        memcpy(y + i * 18, &dh, 2);
        for (int j = 0; j < 16; ++j) {
            const float x0 = x[i * 32 + j] * id;
            const float x1 = x[i * 32 + 16 + j] * id;
            const uint8_t q0 = (uint8_t) std::min(15, (int) (x0 + 8.5f));
            const uint8_t q1 = (uint8_t) std::min(15, (int) (x1 + 8.5f));
            y[i * 18 + 2 + j] = (uint8_t) (q0 | (q1 << 4));
        }
    }
}

// canonical {j, j+16} nibble layout - matches the fixture writer above and
// every ggml writer/reader (see docs/q4-layout-audit.md)
static float deq_q4_one(const uint8_t * row_base, int d) {
    const uint8_t * b = row_base + (d >> 5) * 18;
    __half s;
    memcpy(&s, b, 2);
    const uint8_t byte = b[2 + (d & 15)];
    const int q = (d & 16) ? (byte >> 4) : (byte & 0x0F);
    return __half2float(__float2half_rn(((float) q - 8.0f) * __half2float(s)));
}

// ---------------------------------------------------------------------------
// q8_0 helpers (10/22 q8-direct gate): ggml reference quantizer + kernel-parity
// dequant. Block: [f16 scale 2B][32 x int8 quants 32B]; row = 8 blocks = 272B.
// ---------------------------------------------------------------------------
static void quant_row_q8_0(const float * x, uint8_t * y, int k) {
    for (int i = 0; i < k / 32; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) { amax = std::max(amax, fabsf(x[i * 32 + j])); }
        const float d = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;
        const __half dh = __float2half(d);
        memcpy(y + i * 34, &dh, 2);
        for (int j = 0; j < 32; ++j) {
            const int q = std::max(-127, std::min(127, (int) roundf(x[i * 32 + j] * id)));
            y[i * 34 + 2 + j] = (uint8_t) (int8_t) q;
        }
    }
}

// exact f32 product + one RN to half — the in-kernel sm70_q8_dequant_* formula
static float deq_q8_one(const uint8_t * row_base, int d) {
    const uint8_t * b = row_base + (d >> 5) * 34;
    __half s;
    memcpy(&s, b, 2);
    const int8_t q = (int8_t) b[2 + (d & 31)];
    return __half2float(__float2half_rn((float) q * __half2float(s)));
}

// CPU f32 reference for ONE (batch, q-head, q-row): dot with each kv col,
// online-softmax-safe (scale in raw space, exp2 parity with kernel).
static std::vector<float> ref_row(const float* q_row, const float* k, const float* v,
                                  int kvlen, int kv_offset, int r, float out[256]) {
    const float scale = 1.0f / 16.0f;          // 1/sqrt(256)
    const int limit = std::min(kvlen, r + 1 + kv_offset);
    float mx = -1e30f;
    for (int c = 0; c < limit; ++c) {
        float s = 0.0f;
        for (int d = 0; d < 256; ++d) s += q_row[d] * k[(size_t)c * 256 + d];
        mx = std::max(mx, s);
    }
    float sum = 0.0f;
    std::vector<float> pbuf(limit);
    for (int c = 0; c < limit; ++c) {
        float s = 0.0f;
        for (int d = 0; d < 256; ++d) s += q_row[d] * k[(size_t)c * 256 + d];
        pbuf[c] = exp2f((s - mx) * scale * 1.4426950408889634f);
        sum += pbuf[c];
    }
    const float inv = 1.0f / sum;
    for (int d = 0; d < 256; ++d) {
        float o = 0.0f;
        for (int c = 0; c < limit; ++c) o += pbuf[c] * v[(size_t)c * 256 + d];
        out[d] = o * inv;
    }
    return pbuf;
}

struct Case { const char* name; int nb, kvlen, q_len; bool sample;
              bool v_posmajor = false;  // true = production position-major V (v_row=hkv*D, v_head=D)
              bool f32out = false;      // true = ElementOut=float kernel (production since 8/23)
              float amp = 1.0f;         // input amplitude: >1 spikes QK logits (softmax sharpness probe)
              bool k_posmajor = false;  // true = production to_fp16 dequant K layout (pos-major, 8/23 fix)
              bool splitkv3 = false;    // true = 3-way KV split + merge path (8/23 port; implies f32out)
              bool k_q4 = false;        // true = raw q4_0 K cache, in-kernel dequant (8/23 q4-direct)
              bool v_q4 = false;      // true = raw q4_0 V cache, in-kernel dequant
              bool k_q8 = false;      // true = raw q8_0 K cache, in-kernel dequant (10/22)
              bool v_q8 = false; };   // true = raw q8_0 V cache, in-kernel dequant

static int run_case(const Case& tc, bool oob) {
    const int nb = tc.nb, kvlen = tc.kvlen, q_len = tc.q_len;
    const int hkv = 4, heads_q = 24, gqa = 6, D = 256;
    const int q_pad = ((q_len + 63) / 64) * 64;
    const int kv_offset = kvlen - q_len;
    const float scale_log2 = (1.0f / 16.0f) * 1.4426950408889634f;

    // host f32 source (Q rows beyond q_len = pad = zero, per launcher memset).
    // The CPU reference runs on the f16-QUANTIZED values (same bytes the
    // kernel receives) so the only residual difference is fp32 accumulation
    // order — not input rounding.
    std::vector<float> Qf((size_t) nb * heads_q * q_pad * D, 0.0f),
                       Kf((size_t) nb * hkv * kvlen * D),
                       Vf((size_t) nb * hkv * kvlen * D);
    for (int b = 0; b < nb; ++b) for (int h = 0; h < heads_q; ++h)
        for (int r = 0; r < q_len; ++r) for (int d = 0; d < D; ++d)
            Qf[((size_t)b * heads_q + h) * q_pad * D + r * D + d] = frand() * tc.amp;
    for (auto& x : Kf) x = frand() * tc.amp;
    for (auto& x : Vf) x = frand() * tc.amp;

    // f16-quantized copies = exactly what the kernel sees
    std::vector<float> Qqf(Qf.size()), Kqf(Kf.size()), Vqf(Vf.size());
    for (size_t i = 0; i < Qf.size(); ++i) Qqf[i] = __half2float(__float2half(Qf[i]));
    for (size_t i = 0; i < Kf.size(); ++i) Kqf[i] = __half2float(__float2half(Kf[i]));
    for (size_t i = 0; i < Vf.size(); ++i) Vqf[i] = __half2float(__float2half(Vf[i]));

    // Normal mode headroom: the kernel's last visible N-tile legitimately
    // reads up to 31 rows past kv_len (production reads stale-but-valid cache
    // capacity there; causally masked). Round the headroom up to a full tile
    // and zero it, so the garbage read is DETERMINISTIC — raw cudaMalloc
    // garbage can contain NaN/Inf f16 patterns that turn p=0 * V into NaN
    // (bit the q4K-s3 case on 8/23: heap-layout luck made f16 cases pass).
    const int kv_alloc_rows = oob ? kvlen : ((kvlen + 31) & ~31) + 32;
    std::vector<__half> Qh(Qf.size()), Kh((size_t) nb * hkv * kv_alloc_rows * D), Vh((size_t) nb * hkv * kv_alloc_rows * D);
    for (size_t i = 0; i < Qf.size(); ++i) Qh[i] = __float2half(Qf[i]);
    // K device buffer layout MUST match the strides handed to the kernel:
    //   default  : head-major [b][hkv][row][d]
    //   posmajor : ctx-major  [b][row][hkv][d]  — the to_fp16 dequant dst
    //               linearization ([ne1][ne2][ne0]), i.e. production -ctk q4_0
    //               geometry since the 8/23 stride fix. A linear copy here
    //               would misalign head j by j*3 rows when kv_alloc_rows >
    //               kvlen (headroom), so copy per-explicit-stride either way.
    for (int b = 0; b < nb; ++b) {
        if (tc.k_posmajor) {
            for (int c = 0; c < kv_alloc_rows; ++c) for (int j = 0; j < hkv; ++j)
                for (int d = 0; d < D; ++d) {
                    const size_t dst_i = (((size_t)(b * kv_alloc_rows + c) * hkv + j) * D + d);
                    const float val = (c < kvlen)
                        ? Kf[(((size_t)(b * hkv + j) * kvlen + c) * D + d)] : 0.0f;
                    Kh[dst_i] = __float2half(val);
                }
        } else {
            for (int j = 0; j < hkv; ++j)
                for (int c = 0; c < kv_alloc_rows; ++c)
                    for (int d = 0; d < D; ++d) {
                        const size_t dst_i = (((size_t)(b * hkv + j) * kv_alloc_rows + c) * D + d);
                        const float val = (c < kvlen)
                            ? Kf[(((size_t)(b * hkv + j) * kvlen + c) * D + d)] : 0.0f;
                        Kh[dst_i] = __float2half(val);
                    }
        }
    }
    // V device buffer layout MUST match the strides handed to the kernel:
    //   default : head-major [b][hkv][row][d]  (production K-dequant geometry)
    //   posmajor: ctx-major  [b][row][hkv][d]  (production -ctv f16 paged cache:
    //               Vnb1=2048 -> v_row=1024=hkv*D, Vnb2=512 -> v_head=256=D)
    for (int b = 0; b < nb; ++b) {
        if (tc.v_posmajor) {
            for (int c = 0; c < kv_alloc_rows; ++c) for (int j = 0; j < hkv; ++j)
                for (int d = 0; d < D; ++d) {
                    const size_t dst_i = (((size_t)(b * kv_alloc_rows + c) * hkv + j) * D + d);
                    const float val = (c < kvlen)
                        ? Vf[(((size_t)(b * hkv + j) * kvlen + c) * D + d)] : 0.0f;
                    Vh[dst_i] = __float2half(val);
                }
        } else {
            for (int j = 0; j < hkv; ++j)
                for (int c = 0; c < kv_alloc_rows; ++c)
                    for (int d = 0; d < D; ++d) {
                        const size_t dst_i = (((size_t)(b * hkv + j) * kv_alloc_rows + c) * D + d);
                        const float val = (c < kvlen)
                            ? Vf[(((size_t)(b * hkv + j) * kvlen + c) * D + d)] : 0.0f;
                        Vh[dst_i] = __float2half(val);
                    }
        }
    }

    // q4_0 caches (k_q4/v_q4): [b][ctx][hkv][8 blocks x 18B] pos-major — the
    // production raw -ctk/-ctv q4_0 geometry the q4-direct kernel reads
    // (row stride = hkv*144 bytes, head stride = 144 bytes). Quantized from
    // the f32 sources with the ggml reference quantizer; the CPU reference
    // is then re-based on the f16-rounded dequantized values (exactly what
    // the kernel produces in-kernel — same rounding formula).
    std::vector<uint8_t> Kq4b, Vq4b;
    if (tc.k_q4) {
        Kq4b.assign((size_t) nb * kv_alloc_rows * hkv * 144, 0);
        for (int b = 0; b < nb; ++b)
            for (int c = 0; c < kvlen; ++c)
                for (int j = 0; j < hkv; ++j)
                    quant_row_q4_0(&Kf[(((size_t) (b * hkv + j) * kvlen + c) * D)],
                                   &Kq4b[(((size_t) (b * kv_alloc_rows + c) * hkv + j) * 144)], D);
        for (size_t i = 0; i < Kqf.size(); ++i) Kqf[i] = 0.0f;
        for (int b = 0; b < nb; ++b)
            for (int c = 0; c < kvlen; ++c)
                for (int j = 0; j < hkv; ++j) {
                    const uint8_t * row = &Kq4b[(((size_t) (b * kv_alloc_rows + c) * hkv + j) * 144)];
                    for (int d = 0; d < D; ++d)
                        Kqf[(((size_t) (b * hkv + j) * kvlen + c) * D) + d] = deq_q4_one(row, d);
                }
    }
    if (tc.v_q4) {
        Vq4b.assign((size_t) nb * kv_alloc_rows * hkv * 144, 0);
        for (int b = 0; b < nb; ++b)
            for (int c = 0; c < kvlen; ++c)
                for (int j = 0; j < hkv; ++j)
                    quant_row_q4_0(&Vf[(((size_t) (b * hkv + j) * kvlen + c) * D)],
                                   &Vq4b[(((size_t) (b * kv_alloc_rows + c) * hkv + j) * 144)], D);
        for (size_t i = 0; i < Vqf.size(); ++i) Vqf[i] = 0.0f;
        for (int b = 0; b < nb; ++b)
            for (int c = 0; c < kvlen; ++c)
                for (int j = 0; j < hkv; ++j) {
                    const uint8_t * row = &Vq4b[(((size_t) (b * kv_alloc_rows + c) * hkv + j) * 144)];
                    for (int d = 0; d < D; ++d)
                        Vqf[(((size_t) (b * hkv + j) * kvlen + c) * D) + d] = deq_q4_one(row, d);
                }
    }

    // q8_0 caches (k_q8/v_q8): same [b][ctx][hkv][8 blocks x 34B] geometry as
    // the q4 caches above, quantized with the ggml reference quantizer; the CPU
    // reference is re-based on the dequantized (one-RN) values.
    std::vector<uint8_t> Kq8b, Vq8b;
    if (tc.k_q8) {
        Kq8b.assign((size_t) nb * kv_alloc_rows * hkv * 272, 0);
        for (int b = 0; b < nb; ++b)
            for (int c = 0; c < kvlen; ++c)
                for (int j = 0; j < hkv; ++j)
                    quant_row_q8_0(&Kf[(((size_t) (b * hkv + j) * kvlen + c) * D)],
                                   &Kq8b[(((size_t) (b * kv_alloc_rows + c) * hkv + j) * 272)], D);
        for (size_t i = 0; i < Kqf.size(); ++i) Kqf[i] = 0.0f;
        for (int b = 0; b < nb; ++b)
            for (int c = 0; c < kvlen; ++c)
                for (int j = 0; j < hkv; ++j) {
                    const uint8_t * row = &Kq8b[(((size_t) (b * kv_alloc_rows + c) * hkv + j) * 272)];
                    for (int d = 0; d < D; ++d)
                        Kqf[(((size_t) (b * hkv + j) * kvlen + c) * D) + d] = deq_q8_one(row, d);
                }
    }
    if (tc.v_q8) {
        Vq8b.assign((size_t) nb * kv_alloc_rows * hkv * 272, 0);
        for (int b = 0; b < nb; ++b)
            for (int c = 0; c < kvlen; ++c)
                for (int j = 0; j < hkv; ++j)
                    quant_row_q8_0(&Vf[(((size_t) (b * hkv + j) * kvlen + c) * D)],
                                   &Vq8b[(((size_t) (b * kv_alloc_rows + c) * hkv + j) * 272)], D);
        for (size_t i = 0; i < Vqf.size(); ++i) Vqf[i] = 0.0f;
        for (int b = 0; b < nb; ++b)
            for (int c = 0; c < kvlen; ++c)
                for (int j = 0; j < hkv; ++j) {
                    const uint8_t * row = &Vq8b[(((size_t) (b * kv_alloc_rows + c) * hkv + j) * 272)];
                    for (int d = 0; d < D; ++d)
                        Vqf[(((size_t) (b * hkv + j) * kvlen + c) * D) + d] = deq_q8_one(row, d);
                }
    }

    void *dQ, *dK, *dV, *dO;
    // kernel writes O at [batch][row][head][d] with batch stride = q_pad*heads_q*D
    // (kernel lines 841-844; scatter kernel reads the same layout)
    CK(cudaMalloc(&dQ, Qh.size() * sizeof(El)));
    const void * k_host = tc.k_q8 ? (const void *) Kq8b.data()
                        : tc.k_q4 ? (const void *) Kq4b.data() : (const void *) Kh.data();
    const size_t k_bytes = tc.k_q8 ? Kq8b.size() : tc.k_q4 ? Kq4b.size() : Kh.size() * sizeof(El);
    const void * v_host = tc.v_q8 ? (const void *) Vq8b.data()
                        : tc.v_q4 ? (const void *) Vq4b.data() : (const void *) Vh.data();
    const size_t v_bytes = tc.v_q8 ? Vq8b.size() : tc.v_q4 ? Vq4b.size() : Vh.size() * sizeof(El);
    CK(cudaMalloc(&dK, k_bytes));
    CK(cudaMalloc(&dV, v_bytes));
    CK(cudaMalloc(&dO, (size_t) nb * q_pad * heads_q * D * (tc.f32out ? sizeof(float) : sizeof(El))));
    CK(cudaMemcpy(dQ, Qh.data(), Qh.size() * sizeof(El), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK, k_host, k_bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV, v_host, v_bytes, cudaMemcpyHostToDevice));
    auto kernel     = FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<El, false>;
    auto kernel_f32 = FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<El, false, float>;
    auto kernel_s3  = FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<El, false, float, true>;
    // q4-direct instantiations (8/23): K / V / K+V, dense and SplitKV3.
    auto kernel_q4k   = FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<El, false, float, false, true,  false>;
    auto kernel_q4v   = FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<El, false, float, false, false, true>;
    auto kernel_q4kv  = FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<El, false, float, false, true,  true>;
    auto kernel_s3q4k = FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<El, false, float, true,  true,  false>;
    // q8-direct instantiations (10/22): K / V / K+V, dense and SplitKV3.
    auto kernel_q8k   = FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<El, false, float, false, false, false, true,  false>;
    auto kernel_q8v   = FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<El, false, float, false, false, false, false, true>;
    auto kernel_q8kv  = FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<El, false, float, false, false, false, true,  true>;
    auto kernel_s3q8k = FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<El, false, float, true,  false, false, true,  false>;
    static bool smem_raised = false;
    if (!smem_raised) {
        for (const void* kfn : {(const void*) kernel, (const void*) kernel_f32, (const void*) kernel_s3,
                                (const void*) kernel_q4k, (const void*) kernel_q4v, (const void*) kernel_q4kv,
                                (const void*) kernel_s3q4k, (const void*) kernel_q8k, (const void*) kernel_q8v,
                                (const void*) kernel_q8kv, (const void*) kernel_s3q8k}) {
            CK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, Traits::kSmemBytes));
        }
        smem_raised = true;
    }
    const dim3 block(Traits::kNThreads);
    const dim3 grid(q_pad / 64, nb, hkv * gqa);
    const int64_t head_s = (int64_t) kv_alloc_rows * D;  // one kv-head's rows (== launcher dequant path)
    // nb=1: production passes outer_stride=0 (batch offset never used).
    // nb>1: correct value = whole batch = hkv * head_s. NOTE: the production
    // launcher hardcodes 0 here for ALL nb — that latent bug is documented
    // separately and NOT what this harness tests.
    const int k_outer = (nb == 1) ? 0 : (int)(hkv * head_s);
    const int v_outer = (nb == 1) ? 0 : (int)(hkv * head_s);
    // K strides: harness default = head-major (k_row=D, k_head=kv_alloc_rows*D);
    // k_posmajor = production to_fp16 dequant dst since the 8/23 fix
    // (pos-major [ne1][ne2][ne0]: k_row=hkv*D, k_head=D).
    int k_row_stride, k_head_stride;
    if (tc.k_posmajor) {
        k_row_stride  = hkv * D;
        k_head_stride = D;
    } else {
        k_row_stride  = D;
        k_head_stride = (int) head_s;
    }
    // V strides: harness default = head-major (v_row=D, v_head=kv_alloc_rows*D),
    // matching production ONLY for the dequant path (V=Q4_0). Production
    // production -ctv f16 reads the paged F16 cache DIRECTLY:
    //   v_row_stride  = V->nb[1] (ctx stride; page = 4 ctx x 4 kvheads x 256 = 1024 for cc=700)
    //   v_head_stride = V->nb[2] (256: cache is [ctx][head][dim], head-minor)
    // i.e. position-major. v_posmajor=true replicates the 4-ctx-page geometry
    // (kv_alloc_rows >= 4; page = 4*4*D, v_row = 4*D).
    int v_row_stride, v_head_stride;
    if (tc.v_posmajor) {
        v_row_stride  = hkv * D;   // production Vnb1/2 = 4*256 = 1024
        v_head_stride = D;         // production Vnb2/2 = 256
    } else {
        v_row_stride  = D;
        v_head_stride = (int) head_s;
    }
    // q4-direct overrides: byte strides over the raw [ctx][head][block]
    // cache (row = hkv*144B, head = 144B) — what the launcher passes as
    // K->nb[1]/K->nb[2] when K->type == GGML_TYPE_Q4_0.
    if (tc.k_q4) { k_row_stride = hkv * 144; k_head_stride = 144; }
    if (tc.v_q4) { v_row_stride = hkv * 144; v_head_stride = 144; }
    // q8-direct: same raw geometry, 272B rows (8 blocks x 34B)
    if (tc.k_q8) { k_row_stride = hkv * 272; k_head_stride = 272; }
    if (tc.v_q8) { v_row_stride = hkv * 272; v_head_stride = 272; }

    if (tc.splitkv3) {
        // SplitKV3 path: 3-way partial kernel (grid.y = 3) + merge kernel.
        // Output layout is identical to the dense path ([b][row][head][d]),
        // written f32 by the merge kernel.
        const int64_t rows3 = (int64_t) q_pad * heads_q;   // nb == 1 for these cases
        float * dPout, * dPmax, * dPsum;
        CK(cudaMalloc(&dPout, (size_t) (3 * rows3 * D) * sizeof(float)));
        CK(cudaMalloc(&dPmax, (size_t) (3 * rows3) * sizeof(float)));
        CK(cudaMalloc(&dPsum, (size_t) (3 * rows3) * sizeof(float)));
        const dim3 grid3(q_pad / 64, 3, heads_q);
        const auto kfn = tc.k_q8 ? kernel_s3q8k : tc.k_q4 ? kernel_s3q4k : kernel_s3;
        kfn<<<grid3, block, Traits::kSmemBytes, 0>>>(
            (const El*) dQ, (const El*) dK, (const El*) dV, (float*) dO,
            (int64_t) heads_q * q_pad * D, D, (int64_t) q_pad * D,
            k_outer, k_row_stride, k_head_stride,
            v_outer, v_row_stride, v_head_stride,
            q_pad, kvlen, heads_q, hkv, kv_offset, nullptr, scale_log2, nullptr, 0, 0,
            dPout, dPmax, dPsum);
        CK(cudaGetLastError());
        FLASH_NAMESPACE::sm70_d256_splitkv3_merge_kernel
            <<<dim3((unsigned) rows3), D, 0, 0>>>(
                (const float*) dPout, (const float*) dPmax, (const float*) dPsum,
                (float*) dO, rows3, scale_log2);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        CK(cudaFree(dPout));
        CK(cudaFree(dPmax));
        CK(cudaFree(dPsum));
    } else if (tc.f32out) {
        const auto kfn = tc.k_q8 ? (tc.v_q8 ? kernel_q8kv : kernel_q8k)
                       : tc.v_q8 ? kernel_q8v
                       : tc.k_q4 ? (tc.v_q4 ? kernel_q4kv : kernel_q4k)
                                 : (tc.v_q4 ? kernel_q4v : kernel_f32);
        kfn<<<grid, block, Traits::kSmemBytes, 0>>>(
            (const El*) dQ, (const El*) dK, (const El*) dV, (float*) dO,
            (int64_t) heads_q * q_pad * D, D, (int64_t) q_pad * D,
            k_outer, k_row_stride, k_head_stride,
            v_outer, v_row_stride, v_head_stride,
            q_pad, kvlen, heads_q, hkv, kv_offset, nullptr, scale_log2, nullptr, 0, 0,
            nullptr, nullptr, nullptr);
    } else {
        kernel<<<grid, block, Traits::kSmemBytes, 0>>>(
            (const El*) dQ, (const El*) dK, (const El*) dV, (El*) dO,
            (int64_t) heads_q * q_pad * D, D, (int64_t) q_pad * D,
            k_outer, k_row_stride, k_head_stride,
            v_outer, v_row_stride, v_head_stride,
            q_pad, kvlen, heads_q, hkv, kv_offset, nullptr, scale_log2, nullptr, 0, 0,
            nullptr, nullptr, nullptr);
    }
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    // O host copy: always float for comparison (f32out: direct; f16out: exact
    // half->float conversion once on host).
    std::vector<float> Ohost((size_t) nb * q_pad * heads_q * D);
    if (tc.f32out) {
        CK(cudaMemcpy(Ohost.data(), dO, Ohost.size() * sizeof(float), cudaMemcpyDeviceToHost));
    } else {
        std::vector<__half> tmp(Ohost.size());
        CK(cudaMemcpy(tmp.data(), dO, tmp.size() * sizeof(El), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < tmp.size(); ++i) { Ohost[i] = __half2float(tmp[i]); }
    }

    // which rows/heads to check
    std::vector<int> rows, heads;
    if (!tc.sample) {
        for (int r = 0; r < q_len; ++r) rows.push_back(r);
        for (int h = 0; h < heads_q; ++h) heads.push_back(h);
    } else {
        std::vector<int> cand = {0, 1, 2, 3, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025};
        for (int r = q_len - 3; r < q_len; ++r) cand.push_back(r);
        for (int r = 32; r < q_len; r += 1024) for (int e = -1; e <= 1; ++e) cand.push_back(r + e);
        std::sort(cand.begin(), cand.end()); cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
        for (int r : cand) if (r >= 0 && r < q_len) rows.push_back(r);
        heads = {0, 1, 5, 6, 11, 12, 17, 23};  // across kv-head boundaries (0,5|6,11|12,17|23)
    }

    double max_abs = 0.0; int bad = 0, nan_out = 0, ncmp = 0;
    int first_r = -1, first_h = -1, first_b = -1, first_d = -1;
    // top-5 errors (even in PASS cases: tells noise floor apart from near-miss)
    struct ErrRec { double e; int b, h, r, d; };
    std::vector<ErrRec> top;
    auto consider = [&](double e, int b, int h, int r, int d) {
        if (top.size() < 5 || e > top.back().e) {
            if (top.size() < 5) top.push_back({e, b, h, r, d});
            else top.back() = {e, b, h, r, d};
            std::sort(top.begin(), top.end(), [](const ErrRec& a, const ErrRec& b){ return a.e > b.e; });
        }
    };
    for (int b = 0; b < nb; ++b) {
      for (int h : heads) {
        const int hkv_idx = h / gqa;
        const float* qbase = &Qqf[((size_t) b * heads_q + h) * q_pad * D];
        const float* kbase = &Kqf[((size_t) b * hkv + hkv_idx) * kvlen * D];
        const float* vbase = &Vqf[((size_t) b * hkv + hkv_idx) * kvlen * D];
        // kernel O layout: [b][row][head][d], batch stride q_pad*heads_q*D (row stride heads_q*D)
        const float* obase = &Ohost[((size_t) b * q_pad * heads_q) * D];
        for (int r : rows) {
            float ref[256];
            ref_row(qbase + (size_t) r * D, kbase, vbase, kvlen, kv_offset, r, ref);
            const float* orow = obase + ((size_t) r * heads_q + h) * D;   // [b][row][head][d]
            for (int d = 0; d < D; ++d) {
                const float got = orow[d];
                ncmp++;
                if (std::isnan(got) || std::isinf(got)) {
                    nan_out++;
                    if (nan_out <= 5) printf("  [%s] NaN/Inf out b=%d h=%d r=%d d=%d got=%g\n", tc.name, b, h, r, d, got);
                    continue;
                }
                const double err = std::abs((double) got - ref[d]);
                if (err > max_abs) max_abs = err;
                consider(err, b, h, r, d);
                if (err > 1e-2) {
                    bad++;
                    if (bad <= 5) printf("  [%s] big err=%.3e b=%d h=%d r=%d d=%d got=%.5f ref=%.5f\n", tc.name, err, b, h, r, d, got, ref[d]);
                    if (first_r < 0) { first_r = r; first_h = h; first_b = b; first_d = d; }
                }
            }
        }
      }
    }
    printf("case %-13s nb=%d kv=%6d q=%5d pad=%5d | elems=%-7d max|err|=%.3e  >1e-2: %6d  nan/inf: %d%s\n",
           tc.name, nb, kvlen, q_len, q_pad, ncmp, max_abs, bad, nan_out,
           (bad || nan_out) ? "   <-- FAIL" : "");
    for (auto& t : top) printf("   top: err=%.3e b=%d h=%d r=%d d=%d\n", t.e, t.b, t.h, t.r, t.d);
    if (bad > 0) printf("   first big: b=%d head=%d row=%d d=%d\n", first_b, first_h, first_r, first_d);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    return (bad || nan_out) ? 1 : 0;
}

int main(int argc, char** argv) {
    const bool oob = (getenv("SM70_VERIFY_OOB") != nullptr) || (argc > 1 && argv[1][0] == 'o');
    const bool full = (getenv("SM70_VERIFY_FULL") != nullptr) || (argc > 1 && argv[1][0] == 'f');

    static const Case small[] = {
        { "diag-64",     1,   64,  64, false },  // single N-block + M-block
        { "diag-279",    1,  279, 279, false },  // 8/20 production shape (q_len=279)
        { "cap512-q279", 1,  512, 279, false },  // production chunk: kv_offset=233 > 0
        { "tail-31",     1,   31,  31, false },  // kv_len < kBlockN: partial first/only block
        { "tail-95",     1,   95,  95, false },  // 3 N-blocks, last partial
        { "long-3000",   1, 3000, 3000, true  },  // 94 N-blocks, sampled
        { "nb2-279",     2,  279, 279, true  },  // batch=2, sampled
        { "nb2-long",    2, 3000, 3000, true  },
        { "posV-279",    1,  279, 279, false, true },
        { "posV-3000",   1, 3000, 3000, true,  true },
        { "f32out-279",  1,  279, 279, false, false, true },
        { "f32out-pV",   1,  279, 279, false, true,  true },
        // 8/23 cause probe: real-text attention has spiked QK logits (attention
        // sinks, locality). amp=8 multiplies Q/K/V amplitude -> logits x64 ->
        // sharp softmax + big V magnitudes, approximating real-load structure.
        { "spiky8-279",   1,  279, 279, false, false, false, 8.0f },
        { "spiky8-f32",   1,  279, 279, false, false, true,  8.0f },
        { "spiky8-pV",    1,  279, 279, false, true,  true,  8.0f },
        // 8/23 stride-fix regression: production -ctk q4_0 dequant K is
        // pos-major ([ne1][ne2][ne0]); these pin the fixed stride semantics.
        { "posK-279",     1,  279, 279, false, false, false, 1.0f, true },
        { "posKV-279",    1,  279, 279, false, true,  false, 1.0f, true },  // full production geometry (q4_0 K + paged f16 V)
        { "posKV-f32",    1,  279, 279, false, true,  true,  1.0f, true },
        // 8/23 splitkv3 port: 3-way KV split + merge must match the dense path
        // and the CPU reference. splitkv3-600: prefix geometry (all three
        // segments non-empty). splitkv3-edge: kv == q, third segment empty —
        // exercises the empty-split gmem guard (must not crash, zero contribution).
        { "splitkv3-600",  1, 600, 279, false, false, true, 1.0f, false, true },
        { "splitkv3-edge", 1, 279, 279, false, false, true, 1.0f, false, true },
        // 8/23 q4-direct port: raw q4_0 K/V caches read in-kernel (no f16
        // staging). q4K-pV = full production geometry (-ctk q4_0 -ctv f16:
        // q4 K direct + paged f16 V). q4KV = both tensors quantized. q4K-s3
        // = q4-direct combined with the 3-way KV split + merge.
        { "q4K-pV",     1, 279, 279, false, true,  true, 1.0f, false, false, true,  false },
        { "q4KV-279",   1, 279, 279, false, false, true, 1.0f, false, false, true,  true  },
        { "q4KV-3000",  1, 3000, 3000, true,  false, true, 1.0f, false, false, true,  true  },
        { "q4K-s3",     1, 600, 279, false, false, true, 1.0f, false, true,  true,  false },
        // 10/22 q8-direct port (PR #3): raw q8_0 K/V caches read in-kernel.
        // q8K-pV = production-like mixed geometry (-ctk q8_0 -ctv f16),
        // q8KV = both quantized, q8K-s3 = q8-direct + 3-way split + merge.
        { "q8K-pV",     1, 279, 279, false, true,  true, 1.0f, false, false, false, false, true,  false },
        { "q8KV-279",   1, 279, 279, false, false, true, 1.0f, false, false, false, false, true,  true  },
        { "q8KV-3000",  1, 3000, 3000, true,  false, true, 1.0f, false, false, false, false, true,  true  },
        { "q8K-s3",     1, 600, 279, false, false, true, 1.0f, false, true,  false, false, true,  false },
    };
    static const Case big[] = {
        { "full-32k",    1, 32768, 32768, true },
        { "posV-32k",    1, 32768, 32768, true,  true },  // production V strides, big
    };

    int fails = 0;
    printf("=== sm70 d256 cross-check (%s) ===\n", oob ? "OOB-GUARD: exact-size K/V" : "normal: 3-row K/V headroom");
    for (auto& tc : small) fails += run_case(tc, oob);
    if (full) for (auto& tc : big) fails += run_case(tc, oob);
    printf("=== %s: %d case(s) failing ===\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
