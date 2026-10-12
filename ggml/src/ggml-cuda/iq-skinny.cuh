// IQ3_S "skinny" record codec (design A) for sm_70 (Volta) - T2 of the v100 port line.
//
// Design A bakes the iq3s_grid lookup into the repacked record at load time: one 4-bit
// code per value = 3-bit magnitude index (values are odd: 1,3,...,15 -> index = (v-1)/2)
// + 1-bit sign, placed at the QPN8 slot offsets the k-quant codecs use. The GPU decode
// therefore needs no table at all - it builds the f16 magnitude with a single
// integer-to-half convert and folds the sign in with an XOR.
//
// Layout (see q4k-skinny.cuh for the family contract):
//   codes [N/32][K/16][32][8] uint8   (16 values * 4 bit = exactly one u64 per record)
//   meta  [K/256][N/32][32][6] uint8  (d f16 + scales[4], copied unchanged)
// 16*8 + 6 = 134 bytes per 256-value super-block vs 110 on disk (+21.8 %), so unlike the
// k-quant codecs this one is NOT byte-neutral: the repack is out-of-place (side buffer or
// load-time repack), it cannot run in place.
//
// Value semantics validated against the in-tree dequantize_row_iq3_s:
//   host   : tests/sm70-iq-host-check/iq_host_check.cpp --type iq3_s
//            (18,874,368 real values, 100 % within 1 f16 ulp)
//   device : tests/sm70-iq-host-check/iq3s_device_check.cu (same comparison, on the GPU)
//
// QPN8 lane/column layout derived from llama.cpp-v100 v100-1.1.1 (MIT); see
// LICENSE.v100-skinny in this directory.

#pragma once

#include "common.cuh"
#include "q-skinny-common.cuh"

template <> struct qskinny_codec<GGML_TYPE_IQ3_S> {
    using record_t = uint64_t; // 16 values * 4 bit fills the low word exactly; hi stays 0
    struct meta_t {
        unsigned long long w; // d (f16, 2 bytes) + scales[4], little-endian
    };
    static constexpr int record_bytes      = 8;
    static constexpr int meta_bytes        = 6;
    static constexpr int bits_per_value    = 4;
    static constexpr int block_bytes       = sizeof(block_iq3_s);
    static constexpr int values_per_record = 16;
    static constexpr int values_per_meta   = 256;
    static constexpr int values_per_sub_block = 32;

    static_assert(block_bytes == 110, "iq3_s block size");
    static_assert(record_bytes * 16 + meta_bytes > block_bytes, "iq3_s is out-of-place by construction");

    static __host__ __device__ constexpr size_t codes_bytes(size_t n, size_t k) {
        return n * k * record_bytes / values_per_record;
    }

    static __device__ __forceinline__ qskinny_code load_record(const record_t * p) {
        return { *p, 0 };
    }

    static __device__ __forceinline__ meta_t load_meta(const uint8_t * p) {
        meta_t m;
        memcpy(&m.w, p, meta_bytes);
        return m;
    }

    static __device__ __forceinline__ void store_record(const qskinny_code rec, uint8_t * p) {
        memcpy(p, &rec.lo, record_bytes);
    }

    static __device__ __forceinline__ void store_meta(const meta_t m, uint8_t * p) {
        memcpy(p, &m.w, meta_bytes);
    }

    // d + scales[4] (block layout: d at 0, scales at 106), copied unchanged
    static __device__ __forceinline__ meta_t pack_meta(const uint8_t * block) {
        meta_t m;
        memcpy(&m.w, block, 2);
        memcpy(((uint8_t *) &m.w) + 2, block + 106, 4);
        return m;
    }

    // magnitude byte + sign bit of one original value, mirrors dequantize_row_iq3_s
    static __device__ __forceinline__ void value_at(const uint8_t * block, const int p, unsigned & mag, unsigned & sign) {
        const uint8_t * qs    = block + 2;
        const uint8_t * qh    = block + 2 + 64;
        const uint8_t * signs = block + 2 + 64 + 8;
        const int sb = p / 32;
        const int l  = (p % 32) / 8;
        const int j  = p % 8;
        const uint8_t * qs_sb = qs + 8 * sb;
        const uint8_t   qh_sb = qh[sb];
        const uint8_t * sg_sb = signs + 4 * sb;
        const uint8_t * grid;
        if (j < 4) {
            grid = (const uint8_t *) (iq3s_grid + (qs_sb[2*l + 0] | (((unsigned) qh_sb << (8 - 2*l)) & 256)));
        } else {
            grid = (const uint8_t *) (iq3s_grid + (qs_sb[2*l + 1] | (((unsigned) qh_sb << (7 - 2*l)) & 256)));
        }
        mag  = grid[j & 3];
        sign = (sg_sb[l] >> j) & 1;
    }

    // one record = 16 values starting at block position 32*(g>>1) + 16*(g&1)
    static __device__ __forceinline__ qskinny_code pack_record(const uint8_t * block, const int g) {
        qskinny_code rec = { 0, 0 };
        const int p0 = 32 * (g >> 1) + 16 * (g & 1);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            unsigned mag, sign;
            value_at(block, p0 + j, mag, sign);
            const unsigned code = ((mag - 1) / 2) | (sign << 3);
            qskinny_code_put(rec, code, bits_per_value * qpn8_physical_k(j), bits_per_value);
        }
        return rec;
    }

    // per-sub-block scale as one half: mirrors d * (1 + 2*s) of dequantize_row_iq3_s
    static __device__ __forceinline__ half scale_half(const meta_t m, const int sub_block) {
        const uint8_t * sc = ((const uint8_t *) &m.w) + 2;
        const unsigned s = (sc[sub_block / 2] >> (4 * (sub_block % 2))) & 0xf;
        const float d = __half2float(*(const half *) &m.w);
        return __float2half_rn(d * (float) (1 + 2 * (int) s));
    }

    // f16 of one code: magnitude = 2*index + 1 (exact in f16), sign via the sign bit
    static __device__ __forceinline__ half value_of(const unsigned code) {
        const half m = __int2half_rn((int) (2 * (code & 7) + 1));
        return __ushort_as_half((unsigned short) (__half_as_ushort(m) ^ (unsigned short) (((code >> 3) & 1) << 15)));
    }

    static __device__ __forceinline__ unsigned code_of(const qskinny_code records[2], const int k) {
        return (unsigned) qskinny_code_get(records[k >> 4], bits_per_value * qpn8_physical_k(k & 15), bits_per_value);
    }

    // 32 values of one sub-block (records 2*sb, 2*sb+1) in original order -> 16 half2;
    // the scale is folded in with a single rounded half multiply (k-quant codec paradigm)
    static __device__ __forceinline__ void decode(
            const qskinny_code records[2], const meta_t m, const int sub_block, half2 out[16]) {
        const half sc = scale_half(m, sub_block);
        const half2 scale2 = __halves2half2(sc, sc);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const half h0 = value_of(code_of(records, 2 * i));
            const half h1 = value_of(code_of(records, 2 * i + 1));
            out[i] = __hmul2(__halves2half2(h0, h1), scale2);
        }
    }
};
