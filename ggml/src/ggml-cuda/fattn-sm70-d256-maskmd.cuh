// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// fattn-sm70-d256-maskmd.cuh
//
// Mask-driven masking for the SM70 D256 kernel, used when the KV cache is a
// unified pool shared by several sequences (llama-server -kvu with more than
// one slot). The vendored Mask<> derives visibility from positions
// (col <= row + kv_offset), which only holds while one sequence owns the whole
// batch. In a unified pool the KQ mask tensor is the only truth: a query row
// sees the cells its own sequence owns with a position <= its own, and the
// cells of the other sequences sit between them (find_slot allows holes and
// there is no defrag), so a row's visible set is not a prefix.
//
// The vendored mask.h stays untouched. This header reuses its two index
// helpers (sm70_mask_row_idx / sm70_mask_col_idx) and the same
// fragment-to-(row, col) mapping, and replaces only the value: instead of
// deriving -inf from a position formula it reads the mask element for that
// (row, cell).
//
// The mask is [n_kv, n_q] f16 with the cell index contiguous
// (llama-kv-cache fills it with idst = n_kv * query_row), so one query row is
// a contiguous run and a thread's columns are two adjacent pairs.
// ============================================================================
#pragma once

#include <cuda_fp16.h>

#include "sm70-vendor/flash/mask.h"

namespace FLASH_NAMESPACE {

// Per query row: first and last cell the KQ mask leaves visible over the whole
// padded key range. One block per row, run once per attention node so the
// hkv*gqa head CTAs share the result (the mask does not depend on the head).
// A row with no visible cell reports lo = n_kv, hi = -1.
// static: defined in a header that the launcher and the verify harness both
// include, and only one of them links it per binary.
static __global__ void sm70_mask_row_bounds_kernel(
        const half * __restrict__ mask, int64_t row_stride, int n_kv, int q_len,
        int * __restrict__ row_lo, int * __restrict__ row_hi) {
    const int r = blockIdx.x;
    if (r >= q_len) {
        return;
    }
    const half * row = mask + (int64_t) r * row_stride;
    int lo = n_kv;
    int hi = -1;
    for (int c = threadIdx.x; c < n_kv; c += blockDim.x) {
        // same visibility threshold the live probes use
        if (__half2float(row[c]) > -30000.0f) {
            if (c < lo) { lo = c; }
            if (c > hi) { hi = c; }
        }
    }
    __shared__ int slo[32];
    __shared__ int shi[32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        lo = min(lo, __shfl_down_sync(0xffffffffu, lo, off));
        hi = max(hi, __shfl_down_sync(0xffffffffu, hi, off));
    }
    if (lane == 0) {
        slo[warp] = lo;
        shi[warp] = hi;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const int nwarp = blockDim.x / 32;
        int l = slo[0];
        int h = shi[0];
        for (int i = 1; i < nwarp; ++i) {
            l = min(l, slo[i]);
            h = max(h, shi[i]);
        }
        row_lo[r] = l;
        row_hi[r] = h;
    }
}

struct MaskFromKV {
    const half * base;      // KQ mask, f16, [n_q][n_kv]
    int64_t      row_stride;
    int          n_kv;      // columns >= n_kv are outside the tensor, dropped as -inf
    int          n_q;       // real query rows; padding rows of the last tile clamp to n_q - 1

    __forceinline__ __device__ MaskFromKV(const half * b, const int64_t rs, const int kv, const int q)
        : base(b), row_stride(rs), n_kv(kv), n_q(q) {}

    template <bool Is_even_MN = false, typename Engine, typename Layout>
    __forceinline__ __device__ void apply_mask(Tensor<Engine, Layout> & tensor_,
                                               const int col_idx_offset_,
                                               const int row_idx_offset) {
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        static_assert(decltype(size<0>(tensor_))::value == 8, "First dimension must be 8");

        Tensor tensor = make_tensor(tensor_.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tensor_.layout()));
        static constexpr int kWarpRows = decltype(size<0>(tensor))::value * 4;
        static constexpr int kBlockN = decltype(size<1>(tensor))::value * 8;
        static_assert(kWarpRows == 8 || kWarpRows == 16 || kWarpRows == 32 || kWarpRows == 64,
                      "Unexpected SM70 row layout");
        static_assert(kBlockN == 32 || kBlockN == 64 || kBlockN == 128 || kBlockN == 256,
                      "Unexpected SM70 col layout");
        static_assert(decltype(size<0, 0>(tensor))::value == 2, "Unexpected SM70 row-inner layout");
        static_assert(decltype(size<0, 1>(tensor))::value == kWarpRows / 8, "Unexpected SM70 row-outer layout");
        static_assert(decltype(size<1, 0>(tensor))::value == 2, "Unexpected SM70 col-inner layout");
        static_assert(decltype(size<1, 1>(tensor))::value == 2, "Unexpected SM70 col-middle layout");
        static_assert(decltype(size<1, 2>(tensor))::value == kBlockN / 32, "Unexpected SM70 col-outer layout");

        const int lane_id = threadIdx.x % 32;
        const int lane_row_base = (lane_id & 0x1) | ((lane_id & 0x10) >> 2);
        const int lane_col_base =
            (((lane_id >> 1) & 0x1) << 1) |
            (((lane_id >> 2) & 0x1) << 3) |
            (((lane_id >> 3) & 0x1) << 4);

        // Scores are accumulated, not assigned: a 0.0 mask entry leaves the QK
        // result bit-identical, a -inf entry knocks the column out. The vendored
        // mask assigns, which would wipe the visible entries instead.
        #pragma unroll
        for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
            #pragma unroll
            for (int i = 0; i < size<0, 0>(tensor); ++i) {
                int row_idx = sm70_mask_row_idx<kWarpRows>(lane_row_base, row_idx_offset, i, mi);
                // The last query tile is q_pad rows: the padding rows must not
                // read past the mask. Their output is never scattered back.
                if (row_idx >= n_q) {
                    row_idx = n_q - 1;
                }
                const half * row_ptr = base + (int64_t) row_idx * row_stride;
                #pragma unroll
                for (int n = 0; n < size<1, 2>(tensor); ++n) {
                    #pragma unroll
                    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                        #pragma unroll
                        for (int j = 0; j < size<1, 0>(tensor); ++j) {
                            const int col_idx = sm70_mask_col_idx<kBlockN>(lane_col_base, col_idx_offset_, j, nj, n);
                            auto coord = make_coord(make_coord(i, mi), make_coord(j, nj, n));
                            if (col_idx < n_kv) {
                                tensor(coord) += __half2float(row_ptr[col_idx]);
                            } else {
                                // Past the live key range: the last tile is rounded
                                // up to kBlockN, so those columns were never
                                // loaded (their K may hold anything, NaN
                                // included). Assign, do not accumulate.
                                tensor(coord) = -INFINITY;
                            }
                        }
                    }
                }
            }
        }
    }
};

} // namespace FLASH_NAMESPACE
