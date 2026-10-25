// SPDX-FileCopyrightText: Copyright 2026 the sm70-attn authors
// SPDX-License-Identifier: MIT
//
// sm70-attn — dequantize kernels tuned for large-batch prefill (WP-prefill).
//
// Same per-element math as ggml/src/ggml-cuda/dequantize.cuh, bit-identical by
// construction: identical per-thread element mapping, identical operation order
// (d * grid_byte, then the sign application, then one cast to dst_t) — only the
// memory access pattern changes:
//   * 256-thread CTAs, one warp per 256-element super-block (convert.cu uses
//     32-thread CTAs: 1024-thread/SM occupancy ceiling and small-block launch
//     granularity)
//   * the grid tables staged in shared memory once per CTA (convert.cu reads
//     them from device memory with divergent per-thread indices)
//   * 32-bit grid loads instead of 4 scalar byte loads, vectorized half/float
//     stores instead of 8/4 scalar stores
//
// The kernels read random-ish block bytes; every defined field is consumed and
// undefined padding is never read, so random-byte testing is exact.
#pragma once

#include "convert.cuh"
#include "common.cuh"

#include <cuda_fp16.h>
#include <type_traits>

#include <cstdint>

namespace dequant_sm70 {

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// ---- iq3_s -----------------------------------------------------------------
// block_iq3_s layout (ggml-common.h): d, qs[QK_K/4], qh[QK_K/32], signs[QK_K/32],
// scales[QK_K/64]; QK_K = 256.
template <typename dst_t>
__global__ void dequant_iq3_s_sm70(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb) {
    __shared__ uint32_t s_grid[512];
    for (int i = threadIdx.x; i < 512; i += blockDim.x) {
        s_grid[i] = iq3s_grid[i];
    }
    __syncthreads();

    const int64_t super = (int64_t) blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    if (super >= nb) { return; }
    const int tid = threadIdx.x & 31;

    const block_iq3_s * x = (const block_iq3_s *) vx;
    const int64_t il = tid / 8;   // 0..3
    const int64_t ib = tid % 8;   // 0..7
    dst_t * y = yy + super * 256 + 32 * ib + 8 * il;
    const uint8_t * qs = x[super].qs + 8 * ib;
    const int64_t idx1 = qs[2 * il + 0] | ((x[super].qh[ib] << (8 - 2 * il)) & 256);
    const int64_t idx2 = qs[2 * il + 1] | ((x[super].qh[ib] << (7 - 2 * il)) & 256);
    // one aligned 32-bit load per grid row (4 table bytes)
    const uint32_t g1 = s_grid[idx1];
    const uint32_t g2 = s_grid[idx2];
    const float d = (float) x[super].d * (1 + 2 * ((x[super].scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
    const uint8_t signs = x[super].signs[4 * ib + il];

    float v[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j]   = d * (float) ((g1 >> (8 * j)) & 0xFF) * (signs & kmask_iq2xs[j]     ? -1.f : 1.f);
        v[j+4] = d * (float) ((g2 >> (8 * j)) & 0xFF) * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
    }
    if constexpr (std::is_same_v<dst_t, half>) {
        // 8 contiguous halves = 16 B, 16 B-aligned by construction
        half packed[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) { packed[j] = ggml_cuda_cast<dst_t>(v[j]); }
        *reinterpret_cast<uint4 *>(y) = *reinterpret_cast<const uint4 *>(packed);
    } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) { y[j] = ggml_cuda_cast<dst_t>(v[j]); }
    }
}

// ---- iq3_xxs ----------------------------------------------------------------
template <typename dst_t>
__global__ void dequant_iq3_xxs_sm70(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb) {
    __shared__ uint32_t s_grid[256];         // iq3xxs_grid: 256 x u32
    __shared__ uint8_t  s_signs[128];        // ksigns_iq2xs: 128 x u8
    for (int i = threadIdx.x; i < 256; i += blockDim.x) { s_grid[i] = iq3xxs_grid[i]; }
    for (int i = threadIdx.x; i < 128; i += blockDim.x) { s_signs[i] = ksigns_iq2xs[i]; }
    __syncthreads();

    const int64_t super = (int64_t) blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    if (super >= nb) { return; }
    const int tid = threadIdx.x & 31;

    const block_iq3_xxs * x = (const block_iq3_xxs *) vx;
    const int64_t il = tid / 8;
    const int64_t ib = tid % 8;
    dst_t * y = yy + super * 256 + 32 * ib + 8 * il;
    const uint8_t * q3 = x[super].qs + 8 * ib;
    const uint16_t * gas = (const uint16_t *) (x[super].qs + 64) + 2 * ib;
    const int64_t idx1 = q3[2 * il + 0];
    const int64_t idx2 = q3[2 * il + 1];
    const uint32_t g1 = s_grid[idx1];
    const uint32_t g2 = s_grid[idx2];
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float) x[super].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = s_signs[(aux32 >> 7 * il) & 127];

    float v[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j]   = d * (float) ((g1 >> (8 * j)) & 0xFF) * (signs & kmask_iq2xs[j]     ? -1.f : 1.f);
        v[j+4] = d * (float) ((g2 >> (8 * j)) & 0xFF) * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
    }
    if constexpr (std::is_same_v<dst_t, half>) {
        half packed[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) { packed[j] = ggml_cuda_cast<dst_t>(v[j]); }
        *reinterpret_cast<uint4 *>(y) = *reinterpret_cast<const uint4 *>(packed);
    } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) { y[j] = ggml_cuda_cast<dst_t>(v[j]); }
    }
}

// ---- iq4_xs -----------------------------------------------------------------
// kvalues_iq4nl is int8 (-8..7): values can be negative without a sign table.
// Store pattern: y[0..3] contiguous, y[16..19] at +16 elements.
template <typename dst_t>
__global__ void dequant_iq4_xs_sm70(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb) {
    __shared__ int8_t s_kv[16];
    for (int i = threadIdx.x; i < 16; i += blockDim.x) { s_kv[i] = kvalues_iq4nl[i]; }
    __syncthreads();

    const int64_t super = (int64_t) blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    if (super >= nb) { return; }
    const int tid = threadIdx.x & 31;

    const block_iq4_xs * x = (const block_iq4_xs *) vx;
    const int64_t il = tid / 8;
    const int64_t ib = tid % 8;
    dst_t * y = yy + super * 256 + 32 * ib + 4 * il;
    const uint8_t * q4 = x[super].qs + 16 * ib + 4 * il;
    const uint32_t q = *reinterpret_cast<const uint32_t *>(q4);   // 4-aligned: 16*ib + 4*il
    const float d = (float) x[super].d *
        ((((x[super].scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x[super].scales_h >> 2 * ib) & 3) << 4)) - 32);

    float v[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j]   = d * (float) s_kv[(q >> (8 * j)) & 0xF];
        v[j+4] = d * (float) s_kv[(q >> (8 * j)) >> 4];
    }
    if constexpr (std::is_same_v<dst_t, half>) {
        half lo[4], hi[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            lo[j] = ggml_cuda_cast<dst_t>(v[j]);
            hi[j] = ggml_cuda_cast<dst_t>(v[j + 4]);
        }
        *reinterpret_cast<uint2 *>(y)       = *reinterpret_cast<const uint2 *>(lo);
        *reinterpret_cast<uint2 *>(y + 16)  = *reinterpret_cast<const uint2 *>(hi);
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            y[j]     = ggml_cuda_cast<dst_t>(v[j]);
            y[j+16]  = ggml_cuda_cast<dst_t>(v[j + 4]);
        }
    }
}

#endif // !GGML_USE_HIP && !GGML_USE_MUSA

} // namespace dequant_sm70
