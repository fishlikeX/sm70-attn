// ============================================================================
// sm70 d256 mask-driven (unified KV pool) - standalone cross-check harness
//
// Feeds the production kernel (fattn-sm70-d256-kernel.cuh, MaskMD = true) the
// same layout/strides the llama.cpp launcher uses for a unified pool, and
// compares against a CPU f32 reference: softmax over the cells the KQ mask
// leaves visible, times V.
//
// This is the gate for the case llama-server -kvu with several slots creates:
// one batch holds several sequences, the cells of one sequence are scattered
// through the pool, and a query row may not read another sequence's cells.
//
// Also cross-checks sm70_mask_row_bounds_kernel against the host computation.
//
// Build/run: verify/run_maskmd.ps1 (Windows) or the nvcc line in the README.
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// The vendored cutlass headers assume platform.h was pulled in first (the
// ggml-cuda TU reaches it through common.cuh; a standalone harness does not).
#include "sm70-vendor/cutlass/platform/platform.h"

#define FLASH_NAMESPACE flash_sm70
#include "fattn-sm70-d256-kernel.cuh"

using El = cutlass::half_t;
using Traits = FLASH_NAMESPACE::Sm70D256SplitDTraits;
static_assert(Traits::kHeadDim == 256 && Traits::kBlockM == 64 && Traits::kBlockN == 32);

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    cudaGetLastError(); \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    exit(1); } } while (0)

static constexpr int D = 256;
static constexpr int HKV = 4;
static constexpr int GQA = 6;
static constexpr int HQ = HKV * GQA;   // 24

static uint32_t s_rng = 0x9E3779B9u;
static float frand() {
    s_rng ^= s_rng << 13; s_rng ^= s_rng >> 7; s_rng ^= s_rng << 17;
    return (float) (s_rng >> 8) / 8388608.0f - 1.0f;
}

// ---------------------------------------------------------------------------
// Pool layout: which sequence owns each cell, at which position. owner = -1 is
// a hole (llama-kv-cache find_slot allows them). Rows are ubatch entries: one
// query row per (sequence, position) pair, in batch order.
// ---------------------------------------------------------------------------
enum PoolLayout { LAYOUT_CONTIG, LAYOUT_HOLES, LAYOUT_INTERLEAVE8, LAYOUT_SINGLE_CAUSAL };

struct Pool {
    int n_kv = 0;
    std::vector<int> owner;
    std::vector<int> pos;
    std::vector<int> row_seq;
    std::vector<int> row_pos;
};

// Fill `cells` with `per_seq` cells per sequence in the given layout; positions
// run 0..per_seq-1 per sequence.
static void build_pool(Pool & p, int nseq, int per_seq, PoolLayout layout) {
    p.n_kv = nseq * per_seq;
    p.owner.assign(p.n_kv, -1);
    p.pos.assign(p.n_kv, -1);

    int cell = 0;
    if (layout == LAYOUT_CONTIG || layout == LAYOUT_HOLES || layout == LAYOUT_SINGLE_CAUSAL) {
        for (int s = 0; s < nseq; ++s) {
            for (int i = 0; i < per_seq; ++i) {
                // HOLES: every 7th cell is never allocated, so a sequence's own
                // cells are not even contiguous with themselves.
                if (layout == LAYOUT_HOLES && (i % 7) == 3) {
                    ++cell;
                    continue;
                }
                p.owner[cell] = s;
                p.pos[cell] = i;
                ++cell;
            }
        }
    } else {
        // INTERLEAVE8: round robin in runs of 8 cells, the worst case for a
        // contiguous scan (each 32-cell block mixes four sequences).
        for (int i = 0; i < per_seq; i += 8) {
            for (int s = 0; s < nseq; ++s) {
                for (int k = 0; k < 8 && i + k < per_seq; ++k) {
                    p.owner[cell] = s;
                    p.pos[cell] = i + k;
                    ++cell;
                }
            }
        }
    }

    // rows: the last min(per_seq, 48) positions of every sequence
    const int nrows = std::min(per_seq, 48);
    for (int s = 0; s < nseq; ++s) {
        for (int i = per_seq - nrows; i < per_seq; ++i) {
            p.row_seq.push_back(s);
            p.row_pos.push_back(i);
        }
    }
}

// Visible(r, c) exactly as llama-kv-cache fills the mask.
static inline bool visible(const Pool & p, int r, int c) {
    return p.owner[c] >= 0 && p.owner[c] == p.row_seq[r] && p.pos[c] <= p.row_pos[r];
}

struct Case {
    const char * name;
    int nseq;
    int per_seq;
    PoolLayout layout;
    bool empty_row;
    bool split3 = false;   // SplitKV3 on top of the mask window
};

static int run_case(const Case & tc, bool verbose) {
    Pool pool;
    build_pool(pool, tc.nseq, tc.per_seq, tc.layout);

    const int n_kv = pool.n_kv;
    const int q_len = (int) pool.row_seq.size();
    const int q_pad = ((q_len + Traits::kBlockM - 1) / Traits::kBlockM) * Traits::kBlockM;
    const float scale = 1.0f / std::sqrt((float) D);
    const float scale_log2 = scale * 1.4426950408889634f;

    // ---- host tensors -----------------------------------------------------
    // K/V get one extra kBlockN tail: the last tile is rounded up, and a real
    // cache allocates the full kv_size rows, so the tail read stays inside the
    // buffer. It is filled with finite noise on purpose - if the mask ever let
    // a tail cell through, the error would be O(1), not a NaN.
    const int n_kv_alloc = n_kv + Traits::kBlockN;
    std::vector<__half> Qh((size_t) HQ * q_pad * D, __float2half(0.0f));
    std::vector<__half> Kh((size_t) n_kv_alloc * HKV * D);
    std::vector<__half> Vh((size_t) n_kv_alloc * HKV * D);
    for (size_t i = 0; i < Qh.size(); ++i) { Qh[i] = __float2half(frand() * 0.5f); }
    for (size_t i = 0; i < Kh.size(); ++i) { Kh[i] = __float2half(frand() * 0.5f); }
    for (size_t i = 0; i < Vh.size(); ++i) { Vh[i] = __float2half(frand() * 0.5f); }

    std::vector<__half> Mh((size_t) q_len * n_kv);
    for (int r = 0; r < q_len; ++r) {
        for (int c = 0; c < n_kv; ++c) {
            Mh[(size_t) r * n_kv + c] = visible(pool, r, c)
                ? __float2half(0.0f) : __float2half(-INFINITY);
        }
    }
    // one deliberately empty row: no cell is visible, so its softmax is
    // undefined and the merge/scatter must never see it as a real row.
    if (tc.empty_row) {
        const int r = q_len / 2;
        for (int c = 0; c < n_kv; ++c) { Mh[(size_t) r * n_kv + c] = __float2half(-INFINITY); }
    }

    // host bounds, the reference for the bounds kernel
    std::vector<int> lo_h(q_len), hi_h(q_len);
    for (int r = 0; r < q_len; ++r) {
        int lo = n_kv, hi = -1;
        for (int c = 0; c < n_kv; ++c) {
            if (visible(pool, r, c)) { lo = std::min(lo, c); hi = std::max(hi, c); }
        }
        if (tc.empty_row && r == q_len / 2) { lo = n_kv; hi = -1; }
        lo_h[r] = lo;
        hi_h[r] = hi;
    }

    // ---- device ----------------------------------------------------------
    __half * dQ = nullptr; __half * dK = nullptr; __half * dV = nullptr;
    __half * dM = nullptr; float * dO = nullptr;
    int * dLo = nullptr; int * dHi = nullptr;
    CK(cudaMalloc(&dQ, Qh.size() * sizeof(__half)));
    CK(cudaMalloc(&dK, Kh.size() * sizeof(__half)));
    CK(cudaMalloc(&dV, Vh.size() * sizeof(__half)));
    CK(cudaMalloc(&dM, Mh.size() * sizeof(__half)));
    CK(cudaMalloc(&dO, (size_t) q_pad * HQ * D * sizeof(float)));
    CK(cudaMalloc(&dLo, q_len * sizeof(int)));
    CK(cudaMalloc(&dHi, q_len * sizeof(int)));
    CK(cudaMemcpy(dQ, Qh.data(), Qh.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK, Kh.data(), Kh.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV, Vh.data(), Vh.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dM, Mh.data(), Mh.size() * sizeof(__half), cudaMemcpyHostToDevice));

    // ---- 1. bounds kernel -------------------------------------------------
    FLASH_NAMESPACE::sm70_mask_row_bounds_kernel<<<q_len, 256, 0, 0>>>(
            (const half *) dM, (int64_t) n_kv, n_kv, q_len, dLo, dHi);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<int> lo_d(q_len), hi_d(q_len);
    CK(cudaMemcpy(lo_d.data(), dLo, q_len * sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hi_d.data(), dHi, q_len * sizeof(int), cudaMemcpyDeviceToHost));

    int bounds_bad = 0;
    for (int r = 0; r < q_len; ++r) {
        if (lo_d[r] != lo_h[r] || hi_d[r] != hi_h[r]) {
            if (bounds_bad < 5) {
                printf("  [%s] bounds r=%d kernel=(%d,%d) host=(%d,%d)\n",
                       tc.name, r, lo_d[r], hi_d[r], lo_h[r], hi_h[r]);
            }
            ++bounds_bad;
        }
    }

    // ---- 2. attention -----------------------------------------------------
    const int64_t rows3 = (int64_t) q_pad * HQ;
    float * dP_out = nullptr;
    float * dP_max = nullptr;
    float * dP_sum = nullptr;
    if (tc.split3) {
        // SplitKV3: three partial (numerator, max, sum) triples per row, merged
        // back into dO by the merge kernel - the same path the launcher takes
        // when kv_len >= LLAMA_SM70_SPLITKV3_MIN_KV.
        CK(cudaMalloc(&dP_out, (size_t) 3 * rows3 * D * sizeof(float)));
        CK(cudaMalloc(&dP_max, (size_t) 3 * rows3 * sizeof(float)));
        CK(cudaMalloc(&dP_sum, (size_t) 3 * rows3 * sizeof(float)));
    }
    auto kfn = tc.split3
        ? FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<
              El, false, float, true, false, false, false, false, true>
        : FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<
              El, false, float, false, false, false, false, false, true>;
    CK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, Traits::kSmemBytes));
    const dim3 grid(q_pad / Traits::kBlockM, tc.split3 ? 3 : 1, HQ);
    const dim3 block(Traits::kNThreads);
    kfn<<<grid, block, Traits::kSmemBytes, 0>>>(
            (const El *) dQ, (const El *) dK, (const El *) dV, dO,
            /*q_batch_stride*/ (int64_t) HQ * q_pad * D,
            /*q_row_stride  */ D,
            /*q_head_stride */ (int64_t) q_pad * D,
            /*k_outer_stride*/ 0,
            /*k_row_stride  */ HKV * D,
            /*k_head_stride */ D,
            /*v_outer_stride*/ 0,
            /*v_row_stride  */ HKV * D,
            /*v_head_stride */ D,
            q_pad, n_kv, HQ, HKV,
            /*kv_offset  */ 0,
            /*mask_last_q*/ nullptr,
            (const half *) dM, (int64_t) n_kv, q_len,
            dLo, dHi,
            scale_log2,
            nullptr, 0, 0,
            dP_out, dP_max, dP_sum);
    CK(cudaGetLastError());
    if (tc.split3) {
        FLASH_NAMESPACE::sm70_d256_splitkv3_merge_kernel
            <<<dim3((unsigned) rows3), Traits::kHeadDim, 0, 0>>>(
                (const float *) dP_out, (const float *) dP_max,
                (const float *) dP_sum, dO, rows3, scale_log2);
        CK(cudaGetLastError());
    }
    CK(cudaDeviceSynchronize());

    std::vector<float> Oh((size_t) q_pad * HQ * D);
    CK(cudaMemcpy(Oh.data(), dO, Oh.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- 3. CPU reference -------------------------------------------------
    // Q/K/V as the f16 tensor actually stores them
    std::vector<float> Qf(Qh.size()), Kf(Kh.size()), Vf(Vh.size());
    for (size_t i = 0; i < Qh.size(); ++i) { Qf[i] = __half2float(Qh[i]); }
    for (size_t i = 0; i < Kh.size(); ++i) { Kf[i] = __half2float(Kh[i]); }
    for (size_t i = 0; i < Vh.size(); ++i) { Vf[i] = __half2float(Vh[i]); }

    std::vector<float> scores(n_kv), ref(D);
    double max_abs = 0.0;
    int bad = 0, nan_out = 0, ncmp = 0;

    for (int r = 0; r < q_len; ++r) {
        const bool row_empty = (tc.empty_row && r == q_len / 2);
        for (int h = 0; h < HQ; ++h) {
            const int hkv = h / GQA;
            const float * qrow = &Qf[(size_t) h * q_pad * D + (size_t) r * D];
            const float * kbase = &Kf[(size_t) hkv * D];
            const float * vbase = &Vf[(size_t) hkv * D];

            float mx = -INFINITY;
            for (int c = 0; c < n_kv; ++c) {
                scores[c] = -INFINITY;
                if (!visible(pool, r, c)) { continue; }
                float s = 0.0f;
                const float * krow = kbase + (size_t) c * HKV * D;
                for (int d = 0; d < D; ++d) { s += qrow[d] * krow[d]; }
                s *= scale;
                scores[c] = s;
                if (s > mx) { mx = s; }
            }
            std::fill(ref.begin(), ref.end(), 0.0f);
            float l = 0.0f;
            if (!row_empty) {
                for (int c = 0; c < n_kv; ++c) {
                    if (scores[c] == -INFINITY) { continue; }
                    const float p = expf(scores[c] - mx);
                    l += p;
                    const float * vrow = vbase + (size_t) c * HKV * D;
                    for (int d = 0; d < D; ++d) { ref[d] += p * vrow[d]; }
                }
                for (int d = 0; d < D; ++d) { ref[d] /= l; }
            }

            const float * orow = &Oh[((size_t) r * HQ + h) * D];
            for (int d = 0; d < D; ++d) {
                const float got = orow[d];
                ++ncmp;
                if (std::isnan(got) || std::isinf(got)) {
                    // an all-empty row has no defined softmax; anything else is a bug
                    if (!row_empty) {
                        ++nan_out;
                        if (nan_out <= 5) {
                            printf("  [%s] NaN/Inf out r=%d h=%d d=%d\n", tc.name, r, h, d);
                        }
                    }
                    continue;
                }
                if (row_empty) {
                    ++nan_out;
                    if (nan_out <= 5) {
                        printf("  [%s] empty row produced finite output r=%d h=%d d=%d\n", tc.name, r, h, d);
                    }
                    continue;
                }
                const double err = std::abs((double) got - ref[d]);
                if (err > max_abs) { max_abs = err; }
                if (err > 1e-2) {
                    ++bad;
                    if (bad <= 5) {
                        printf("  [%s] big err=%.3e r=%d h=%d d=%d got=%.5f ref=%.5f\n",
                               tc.name, err, r, h, d, got, ref[d]);
                    }
                }
            }
        }
    }

    printf("case %-18s nseq=%d n_kv=%5d q_len=%4d pad=%4d | bounds: %-6s | max|err|=%.3e  >1e-2: %6d  nan/inf: %d %s\n",
           tc.name, tc.nseq, n_kv, q_len, q_pad, bounds_bad ? "FAIL" : "ok",
           max_abs, bad, nan_out,
           (bad || nan_out || bounds_bad) ? "  <-- FAIL" : "");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dM); cudaFree(dO);
    cudaFree(dLo); cudaFree(dHi);
    if (dP_out) { cudaFree(dP_out); cudaFree(dP_max); cudaFree(dP_sum); }
    (void) verbose;
    return (bad || nan_out || bounds_bad) ? 1 : 0;
}

int main(int argc, char ** argv) {
    (void) argc; (void) argv;

    static const Case cases[] = {
        // single sequence, cells 0..n-1, positions 0..n-1: the classic causal
        // case the position formula also handles (regression on the new path)
        { "md-causal-200",   1, 200, LAYOUT_SINGLE_CAUSAL, false },
        // 4 slots, one contiguous run per slot (the steady state)
        { "md-seq4-contig",  4, 128, LAYOUT_CONTIG,        false },
        // 4 slots, holes inside each run (find_slot allows them)
        { "md-seq4-holes",   4, 128, LAYOUT_HOLES,         false },
        // worst case: runs of 8 cells, every 32-cell block mixes four sequences
        { "md-seq4-il8",     4, 128, LAYOUT_INTERLEAVE8,   false },
        // q_len not a multiple of kBlockM: the tail tile is padded
        { "md-tail-100",     2,  73, LAYOUT_CONTIG,        false },
        // one row with no visible cell at all
        { "md-empty-row",    2,  96, LAYOUT_CONTIG,        true  },
        // SplitKV3 composes with the mask window: three partial segments over
        // whatever range the mask left visible, merged back.
        { "md-split3-600",   1, 600, LAYOUT_SINGLE_CAUSAL, false, true },
        { "md-split3-seq4",  4, 160, LAYOUT_INTERLEAVE8,   false, true },
    };

    int fails = 0;
    for (const Case & tc : cases) { fails += run_case(tc, false); }

    printf("\n%s (%d/%zu cases passed)\n", fails ? "FAIL" : "PASS",
           (int) (sizeof(cases) / sizeof(cases[0])) - fails,
           sizeof(cases) / sizeof(cases[0]));
    return fails ? 1 : 0;
}
