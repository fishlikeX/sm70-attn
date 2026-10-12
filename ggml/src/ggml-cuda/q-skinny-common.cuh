// QPN8 lane/column/physical-K permutations shared by the skinny GEMM kernels for sm_70 (Volta).
//
// Adapted from 1Cat-vLLM csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu (Apache-2.0). The QPN8
// execution layout is derived from dnv2003/v100-skinny (MIT) and its block-scale adaptation
// in haohervchb/sglang-V100. See LICENSE.v100-skinny in this directory.

#pragma once

#include "common.cuh"

// ---- 1Cat fp8_qpn8_sm70.cu:39-50 (lane/column mapping, unchanged) ----

static __device__ __forceinline__ int qpn8_col_from_lane(int lane) {
    return ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
}

static __device__ __forceinline__ int qpn8_lane_from_col(int col) {
    return (col & 3) | (((col >> 3) & 3) << 2) | (((col >> 2) & 1) << 4);
}

static __device__ __forceinline__ int qpn8_physical_k(int logical_k) {
    const int local = logical_k & 7;
    return (logical_k & 8) + (local >> 1) + ((local & 1) << 2);
}

// ---- shared record helpers (used by the k-quant and IQ codec families) ----
//
// One loaded record: the record bytes as two little-endian 64-bit words, hi zero padded.
// The bits of one value sit at bit offset slot*bits_per_value of the record,
// slot = qpn8_physical_k.

struct qskinny_code {
    unsigned long long lo, hi;
};

// record buffers of the sizes that have no matching builtin load type
struct qskinny_rec6  { uint8_t b[6];  };
struct qskinny_rec10 { uint8_t b[10]; };
struct qskinny_rec12 { uint8_t b[12]; };

template <ggml_type T> struct qskinny_codec;

static __device__ __forceinline__ unsigned qskinny_code_get(const qskinny_code rec, const int off, const int bits) {
    const unsigned mask = (1u << bits) - 1;
    if (off + bits <= 64) {
        return (unsigned) (rec.lo >> off) & mask;
    }
    if (off >= 64) {
        return (unsigned) (rec.hi >> (off - 64)) & mask;
    }
    return (unsigned) ((rec.lo >> off) | (rec.hi << (64 - off))) & mask;
}

static __device__ __forceinline__ void qskinny_code_put(qskinny_code & rec, const unsigned code, const int off, const int bits) {
    if (off < 64) {
        rec.lo |= (unsigned long long) code << off;
        if (off + bits > 64) {
            rec.hi |= (unsigned long long) code >> (64 - off);
        }
    } else {
        rec.hi |= (unsigned long long) code << (off - 64);
    }
}
