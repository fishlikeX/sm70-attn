// IQ skinny record codecs (T2 of the v100 port line).
//
// TWO DESIGNS, per the 2026-10-13 decision ("A first, B as the fallback"): design A
// bakes the iq3s_grid lookup into the record at repack time (no table in the kernel,
// ~4-5 ops/value to rebuild the magnitude); design B keeps the grid index + expanded
// sign bits in the record and decodes through a one-time built f16 grid table in L1
// (~1.5 ops/value). The measured A/B on this tree led to B for IQ3_S:
//
//   design A (commit 331612208): single-V100 tg128 29.81 vs mmq 29.79 t/s - neutral,
//   the ~4-5 op/value decode keeps the M1 kernel at the mmq issue rate.
//   design B (this file): record keeps 4x9-bit grid indices + 16 sign bits = 52 bits
//   in 7 bytes; decode = 1 LUT load + 1 sign-mask load + 1 XOR + 1 half2 mul per
//   4 values, i.e. ~1.5 ops/value, and drops the on-disk growth to +7.3 %.
//
// Per type (design B where noted, design A otherwise):
//   IQ3_S   (B): record 7 B = 4x9-bit idx + 16 sign bits, meta 6 B   -> 118 B/block (+7.3 %)
//   IQ3_XXS (A): 4-bit codes (3-bit index into {4,12,...,62}), record 8 B, meta 6 B -> +36.7 %
//   IQ2_XS  (A): 3-bit codes (2-bit index into {8,25,43}),     record 6 B, meta 10 B -> +43.2 %
//   IQ2_S   (A): 3-bit codes (2-bit index into {8,25,43}),     record 6 B, meta 10 B -> +29.3 %
//
// Value semantics validated against the in-tree dequantize_row_iq*:
//   host   : tests/sm70-iq-host-check/iq_host_check.cpp (18.9M+ real values, 100% <=1 ulp)
//   device : tests/sm70-iq-host-check/iq_device_check.cu (same comparison on the GPU)
//
// QPN8 lane/column layout derived from llama.cpp-v100 v100-1.1.1 (MIT); see
// LICENSE.v100-skinny in this directory.

#pragma once

#include "common.cuh"
#include "q-skinny-common.cuh"

// ---- one-time LUTs for design B (built by iq_skinny_init_lut, 4 KB + 128 B) ----

// iq3s_grid entry (4 byte values in {1,3,...,15}) pre-converted to 4 f16 halves;
// 4-bit sign pattern -> four f16 sign bits. Defined once (iq-skinny.cu sets
// IQ_SKINNY_LUT_DEFINE before including this header; standalone device tests may
// define their own copy), extern everywhere else.
#if defined(IQ_SKINNY_LUT_DEFINE)
__device__ unsigned long long iq3s_grid_f16[512];
__device__ unsigned long long iq3s_sign_mask16[16];

__global__ void iq_skinny_init_lut_kernel() {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 512) {
        const uint8_t * g = (const uint8_t *) (iq3s_grid + i);
        unsigned long long v = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v |= (unsigned long long) __half_as_ushort(__int2half_rn((int) g[j])) << (16 * j);
        }
        iq3s_grid_f16[i] = v;
    }
    if (i < 16) {
        unsigned long long m = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            m |= (unsigned long long) (((i >> j) & 1) << 15) << (16 * j);
        }
        iq3s_sign_mask16[i] = m;
    }
}
#else
extern __device__ unsigned long long iq3s_grid_f16[512];
extern __device__ unsigned long long iq3s_sign_mask16[16];
__global__ void iq_skinny_init_lut_kernel();
#endif

// must run once per process before any design-B decode
inline void iq_skinny_init_lut(cudaStream_t stream) {
    iq_skinny_init_lut_kernel<<<2, 256, 0, stream>>>();
}

// ============================================================================ IQ3_S (design B)

template <> struct qskinny_codec<GGML_TYPE_IQ3_S> {
    using record_t = qskinny_rec7; // 4x9-bit grid index + 16 sign bits = 52 bits in 7 bytes
    struct meta_t {
        unsigned long long w; // d (f16, 2 bytes) + scales[4], little-endian
    };
    static constexpr int record_bytes      = 7;
    static constexpr int meta_bytes        = 6;
    static constexpr int bits_per_value    = 0; // design B stores indices, not per-value codes
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
        unsigned long long v = 0;
        memcpy(&v, p->b, record_bytes);
        return { v, 0 };
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
    // d + scales[4] (d at 0, scales at 106), copied unchanged
    static __device__ __forceinline__ meta_t pack_meta(const uint8_t * block) {
        meta_t m;
        memcpy(&m.w, block, 2);
        memcpy(((uint8_t *) &m.w) + 2, block + 106, 4);
        return m;
    }
    // record layout: [0,9) [9,18) grid indices of group l (sub-block g>>1, half g&1),
    //                [18,27) [27,36) grid indices of group l+1,
    //                [36,44) [44,52) their sign bytes
    static __device__ __forceinline__ qskinny_code pack_record(const uint8_t * block, const int g) {
        const int sb = g >> 1;
        const int h  = g & 1;
        const uint8_t * qs  = block + 2 + 8 * sb;
        const uint8_t   qh  = block[2 + 64 + sb];
        const uint8_t * sg  = block + 2 + 64 + 8 + 4 * sb;
        unsigned long long rec = 0;
        int bit = 0;
#pragma unroll
        for (int l = 2 * h; l <= 2 * h + 1; ++l) {
            const unsigned i0 = qs[2*l + 0] | (((unsigned) qh << (8 - 2*l)) & 256);
            const unsigned i1 = qs[2*l + 1] | (((unsigned) qh << (7 - 2*l)) & 256);
            rec |= (unsigned long long) i0 << bit; bit += 9;
            rec |= (unsigned long long) i1 << bit; bit += 9;
        }
        rec |= (unsigned long long) sg[2*h + 0] << 36;
        rec |= (unsigned long long) sg[2*h + 1] << 44;
        return { rec, 0 };
    }
    // per-sub-block scale as one half: mirrors d * (1 + 2*s) of dequantize_row_iq3_s
    static __device__ __forceinline__ half scale_half(const meta_t m, const int sub_block) {
        const uint8_t * sc = ((const uint8_t *) &m.w) + 2;
        const unsigned s = (sc[sub_block / 2] >> (4 * (sub_block % 2))) & 0xf;
        const float d = __half2float(*(const half *) &m.w);
        return __float2half_rn(d * (float) (1 + 2 * (int) s));
    }
    // 32 values of one sub-block (records 2*sb, 2*sb+1) in original order -> 16 half2;
    // the scale is folded in with one rounded half multiply (k-quant codec paradigm)
    static __device__ __forceinline__ void decode(
            const qskinny_code records[2], const meta_t m, const int sub_block, half2 out[16]) {
        const half sc = scale_half(m, sub_block);
        const half2 scale2 = __halves2half2(sc, sc);
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const unsigned long long rec = records[r].lo;
#pragma unroll
            for (int grp = 0; grp < 2; ++grp) {
                const unsigned i0 = (unsigned) ((rec >> (18 * grp + 0)) & 511);
                const unsigned i1 = (unsigned) ((rec >> (18 * grp + 9)) & 511);
                const unsigned sg = (unsigned) ((rec >> (36 + 8 * grp)) & 0xff);
                const unsigned long long w0 = iq3s_grid_f16[i0] ^ iq3s_sign_mask16[sg & 0xf];
                const unsigned long long w1 = iq3s_grid_f16[i1] ^ iq3s_sign_mask16[sg >> 4];
                const unsigned long long ws[2] = { w0, w1 };
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const unsigned short * hp = (const unsigned short *) &ws[t >> 1];
                    const half2 pair = __halves2half2(__ushort_as_half(hp[(t & 1) * 2 + 0]),
                                                      __ushort_as_half(hp[(t & 1) * 2 + 1]));
                    out[8 * r + 4 * grp + t] = __hmul2(pair, scale2);
                }
            }
        }
    }
};

// ============================================================================ IQ3_XXS (design A)

// magnitudes {4,12,20,28,36,44,52,62}: 4 + 8*idx for idx < 7, 62 for idx == 7
static __device__ __forceinline__ unsigned iq3xxs_mag_of(const unsigned idx) {
    return idx < 7 ? 4 + 8 * idx : 62;
}
static __device__ __forceinline__ unsigned iq3xxs_idx_of(const unsigned mag) {
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
        if (iq3xxs_mag_of(i) == mag) {
            return i;
        }
    }
    return 0;
}

template <> struct qskinny_codec<GGML_TYPE_IQ3_XXS> {
    using record_t = uint64_t;
    struct meta_t {
        unsigned long long w; // d (f16) + 8 scale nibbles packed in 4 bytes
    };
    static constexpr int record_bytes      = 8;
    static constexpr int meta_bytes        = 6;
    static constexpr int bits_per_value    = 4;
    static constexpr int block_bytes       = sizeof(block_iq3_xxs);
    static constexpr int values_per_record = 16;
    static constexpr int values_per_meta   = 256;
    static constexpr int values_per_sub_block = 32;

    static_assert(block_bytes == 98, "iq3_xxs block size");

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
    // d + the 8 per-sub-block 4-bit scales (aux32 bits 28..31), packed as nibbles
    static __device__ __forceinline__ meta_t pack_meta(const uint8_t * block) {
        meta_t m;
        memset(&m, 0, sizeof(m));
        memcpy(&m.w, block, 2);
        const uint16_t * gas = (const uint16_t *) (block + 2 + 64);
        uint8_t * sc = ((uint8_t *) &m.w) + 2;
#pragma unroll
        for (int sb = 0; sb < 8; ++sb) {
            const uint32_t aux32 = (uint32_t) gas[2*sb] | ((uint32_t) gas[2*sb+1] << 16);
            const unsigned s = (aux32 >> 28) & 0xf;
            sc[sb / 2] |= (uint8_t) (s << (4 * (sb % 2)));
        }
        return m;
    }
    static __device__ __forceinline__ void value_at(const uint8_t * block, const int p, unsigned & mag, unsigned & sign) {
        const uint8_t * qs = block + 2;
        const uint16_t * gas = (const uint16_t *) (qs + 64);
        const int sb = p / 32;
        const int l  = (p % 32) / 8;
        const int j  = p % 8;
        const uint8_t * q3_sb = qs + 8 * sb;
        const uint8_t * grid = (const uint8_t *) (iq3xxs_grid + q3_sb[2*l + (j < 4 ? 0 : 1)]);
        mag = grid[j & 3];
        const uint32_t aux32 = (uint32_t) gas[2*sb] | ((uint32_t) gas[2*sb+1] << 16);
        sign = (ksigns_iq2xs[(aux32 >> 7*l) & 127] >> j) & 1;
    }
    static __device__ __forceinline__ qskinny_code pack_record(const uint8_t * block, const int g) {
        qskinny_code rec = { 0, 0 };
        const int p0 = 32 * (g >> 1) + 16 * (g & 1);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            unsigned mag, sign;
            value_at(block, p0 + j, mag, sign);
            qskinny_code_put(rec, iq3xxs_idx_of(mag) | (sign << 3), bits_per_value * qpn8_physical_k(j), bits_per_value);
        }
        return rec;
    }
    static __device__ __forceinline__ half scale_half(const meta_t m, const int sub_block) {
        const uint8_t * sc = ((const uint8_t *) &m.w) + 2;
        const unsigned s = (sc[sub_block / 2] >> (4 * (sub_block % 2))) & 0xf;
        const float d = __half2float(*(const half *) &m.w);
        return __float2half_rn(d * (0.5f + (float) (int) s) * 0.5f);
    }
    static __device__ __forceinline__ void decode(
            const qskinny_code records[2], const meta_t m, const int sub_block, half2 out[16]) {
        const half sc = scale_half(m, sub_block);
        const half2 scale2 = __halves2half2(sc, sc);
        struct dec { static __device__ __forceinline__ half value(unsigned code) {
            const half h = __int2half_rn((int) iq3xxs_mag_of(code & 7));
            return __ushort_as_half((unsigned short) (__half_as_ushort(h) ^ (unsigned short) (((code >> 3) & 1) << 15)));
        } };
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int r = i >> 3;
            const int j = 2 * (i & 7);
            const unsigned c0 = (unsigned) qskinny_code_get(records[r], 4 * qpn8_physical_k(j + 0), 4);
            const unsigned c1 = (unsigned) qskinny_code_get(records[r], 4 * qpn8_physical_k(j + 1), 4);
            out[i] = __hmul2(__halves2half2(dec::value(c0), dec::value(c1)), scale2);
        }
    }
};

// ============================================================================ IQ2_XS / IQ2_S (design A)

// magnitudes {8,25,43}
static __device__ __forceinline__ unsigned iq2_mag_of(const unsigned idx) {
    return idx == 0 ? 8 : (idx == 1 ? 25 : 43);
}
static __device__ __forceinline__ unsigned iq2_idx_of(const unsigned mag) {
    return mag == 8 ? 0 : (mag == 25 ? 1 : 2);
}

template <> struct qskinny_codec<GGML_TYPE_IQ2_XS> {
    using record_t = qskinny_rec6; // 16 values * 3 bit = 48 bits, same shape as the Q3_K record
    struct meta_t {
        unsigned long long w0, w1; // d (f16) + scales[8] = 10 bytes (Q3_K-style over-sized struct)
    };
    static constexpr int record_bytes      = 6;
    static constexpr int meta_bytes        = 10;
    static constexpr int bits_per_value    = 3;
    static constexpr int block_bytes       = sizeof(block_iq2_xs);
    static constexpr int values_per_record = 16;
    static constexpr int values_per_meta   = 256;
    static constexpr int values_per_sub_block = 32;

    static_assert(block_bytes == 74, "iq2_xs block size");
    static_assert(sizeof(meta_t) >= (size_t) meta_bytes, "meta_t must cover meta_bytes");

    static __host__ __device__ constexpr size_t codes_bytes(size_t n, size_t k) {
        return n * k * record_bytes / values_per_record;
    }
    static __device__ __forceinline__ qskinny_code load_record(const record_t * p) {
        const unsigned short * wp = reinterpret_cast<const unsigned short *>(p);
        return { (unsigned long long) wp[0] | ((unsigned long long) wp[1] << 16) |
                 ((unsigned long long) wp[2] << 32), 0 };
    }
    static __device__ __forceinline__ meta_t load_meta(const uint8_t * p) {
        meta_t m;
        memset(&m, 0, sizeof(m));
        memcpy(&m, p, meta_bytes);
        return m;
    }
    static __device__ __forceinline__ void store_record(const qskinny_code rec, uint8_t * p) {
        unsigned short * wp = reinterpret_cast<unsigned short *>(p);
        wp[0] = (unsigned short) rec.lo;
        wp[1] = (unsigned short) (rec.lo >> 16);
        wp[2] = (unsigned short) (rec.lo >> 32);
    }
    static __device__ __forceinline__ void store_meta(const meta_t m, uint8_t * p) {
        memcpy(p, &m, meta_bytes);
    }
    // d + scales[8], copied unchanged (scale nibble for record r = (scales[sb] >> 4*r) & 0xf)
    static __device__ __forceinline__ meta_t pack_meta(const uint8_t * block) {
        meta_t m;
        memset(&m, 0, sizeof(m));
        memcpy(&m, block, 2);
        memcpy(((uint8_t *) &m) + 2, block + 2 + 64, 8);
        return m;
    }
    static __device__ __forceinline__ void value_at(const uint8_t * block, const int p, unsigned & mag, unsigned & sign) {
        const uint16_t * q2 = (const uint16_t *) (block + 2);
        const int sb = p / 32;
        const int l  = (p % 32) / 8;
        const int j  = p % 8;
        const uint16_t v = q2[4 * sb + l];
        const uint8_t * grid = (const uint8_t *) (iq2xs_grid + (v & 511));
        mag  = grid[j];
        sign = (ksigns_iq2xs[v >> 9] >> j) & 1;
    }
    static __device__ __forceinline__ qskinny_code pack_record(const uint8_t * block, const int g) {
        qskinny_code rec = { 0, 0 };
        const int p0 = 32 * (g >> 1) + 16 * (g & 1);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            unsigned mag, sign;
            value_at(block, p0 + j, mag, sign);
            qskinny_code_put(rec, iq2_idx_of(mag) | (sign << 2), bits_per_value * qpn8_physical_k(j), bits_per_value);
        }
        return rec;
    }
    static __device__ __forceinline__ half scale_half(const meta_t m, const int sb, const int r) {
        const uint8_t * sc = ((const uint8_t *) &m) + 2;
        const unsigned s = (sc[sb] >> (4 * r)) & 0xf;
        const float d = __half2float(*(const half *) &m);
        return __float2half_rn(d * (0.5f + (float) (int) s) * 0.25f);
    }
    static __device__ __forceinline__ void decode(
            const qskinny_code records[2], const meta_t m, const int sub_block, half2 out[16]) {
        const half scale2[2] = { scale_half(m, sub_block, 0), scale_half(m, sub_block, 1) };
        struct dec { static __device__ __forceinline__ half value(unsigned code) {
            const half h = __int2half_rn((int) iq2_mag_of(code & 3));
            return __ushort_as_half((unsigned short) (__half_as_ushort(h) ^ (unsigned short) (((code >> 2) & 1) << 15)));
        } };
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int r = i >> 3;
            const int j = 2 * (i & 7);
            const unsigned c0 = (unsigned) qskinny_code_get(records[r], 3 * qpn8_physical_k(j + 0), 3);
            const unsigned c1 = (unsigned) qskinny_code_get(records[r], 3 * qpn8_physical_k(j + 1), 3);
            out[i] = __hmul2(__halves2half2(dec::value(c0), dec::value(c1)), __halves2half2(scale2[r], scale2[r]));
        }
    }
};

template <> struct qskinny_codec<GGML_TYPE_IQ2_S> {
    using record_t = qskinny_rec6;
    struct meta_t {
        unsigned long long w0, w1; // d (f16) + scales[8] = 10 bytes (Q3_K-style over-sized struct)
    };
    static constexpr int record_bytes      = 6;
    static constexpr int meta_bytes        = 10;
    static constexpr int bits_per_value    = 3;
    static constexpr int block_bytes       = sizeof(block_iq2_s);
    static constexpr int values_per_record = 16;
    static constexpr int values_per_meta   = 256;
    static constexpr int values_per_sub_block = 32;

    static_assert(block_bytes == 82, "iq2_s block size");
    static_assert(sizeof(meta_t) >= (size_t) meta_bytes, "meta_t must cover meta_bytes");

    static __host__ __device__ constexpr size_t codes_bytes(size_t n, size_t k) {
        return n * k * record_bytes / values_per_record;
    }
    static __device__ __forceinline__ qskinny_code load_record(const record_t * p) {
        const unsigned short * wp = reinterpret_cast<const unsigned short *>(p);
        return { (unsigned long long) wp[0] | ((unsigned long long) wp[1] << 16) |
                 ((unsigned long long) wp[2] << 32), 0 };
    }
    static __device__ __forceinline__ meta_t load_meta(const uint8_t * p) {
        meta_t m;
        memset(&m, 0, sizeof(m));
        memcpy(&m, p, meta_bytes);
        return m;
    }
    static __device__ __forceinline__ void store_record(const qskinny_code rec, uint8_t * p) {
        unsigned short * wp = reinterpret_cast<unsigned short *>(p);
        wp[0] = (unsigned short) rec.lo;
        wp[1] = (unsigned short) (rec.lo >> 16);
        wp[2] = (unsigned short) (rec.lo >> 32);
    }
    static __device__ __forceinline__ void store_meta(const meta_t m, uint8_t * p) {
        memcpy(p, &m, meta_bytes);
    }
    // d + scales[8] (d at 0, scales at 74)
    static __device__ __forceinline__ meta_t pack_meta(const uint8_t * block) {
        meta_t m;
        memset(&m, 0, sizeof(m));
        memcpy(&m, block, 2);
        memcpy(((uint8_t *) &m) + 2, block + 2 + 64 + 8, 8);
        return m;
    }
    static __device__ __forceinline__ void value_at(const uint8_t * block, const int p, unsigned & mag, unsigned & sign) {
        const uint8_t * qs = block + 2;
        const uint8_t * qh = block + 2 + 64;
        const int sb = p / 32;
        const int l  = (p % 32) / 8;
        const int j  = p % 8;
        const uint8_t * grid = (const uint8_t *) (iq2s_grid + (qs[4*sb + l] | (((unsigned) qh[sb] << (8 - 2*l)) & 0x300)));
        mag  = grid[j];
        sign = (qs[32 + 4*sb + l] >> j) & 1;
    }
    static __device__ __forceinline__ qskinny_code pack_record(const uint8_t * block, const int g) {
        qskinny_code rec = { 0, 0 };
        const int p0 = 32 * (g >> 1) + 16 * (g & 1);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            unsigned mag, sign;
            value_at(block, p0 + j, mag, sign);
            qskinny_code_put(rec, iq2_idx_of(mag) | (sign << 2), bits_per_value * qpn8_physical_k(j), bits_per_value);
        }
        return rec;
    }
    static __device__ __forceinline__ half scale_half(const meta_t m, const int sb, const int r) {
        const uint8_t * sc = ((const uint8_t *) &m) + 2;
        const unsigned s = (sc[sb] >> (4 * r)) & 0xf;
        const float d = __half2float(*(const half *) &m);
        return __float2half_rn(d * (0.5f + (float) (int) s) * 0.25f);
    }
    static __device__ __forceinline__ void decode(
            const qskinny_code records[2], const meta_t m, const int sub_block, half2 out[16]) {
        const half scale2[2] = { scale_half(m, sub_block, 0), scale_half(m, sub_block, 1) };
        struct dec { static __device__ __forceinline__ half value(unsigned code) {
            const half h = __int2half_rn((int) iq2_mag_of(code & 3));
            return __ushort_as_half((unsigned short) (__half_as_ushort(h) ^ (unsigned short) (((code >> 2) & 1) << 15)));
        } };
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int r = i >> 3;
            const int j = 2 * (i & 7);
            const unsigned c0 = (unsigned) qskinny_code_get(records[r], 3 * qpn8_physical_k(j + 0), 3);
            const unsigned c1 = (unsigned) qskinny_code_get(records[r], 3 * qpn8_physical_k(j + 1), 3);
            out[i] = __hmul2(__halves2half2(dec::value(c0), dec::value(c1)), __halves2half2(scale2[r], scale2[r]));
        }
    }
};

// ---- IQ skinny runtime (out-of-place repack into side buffers), see iq-skinny.cu ----
bool ggml_cuda_iq_skinny_can_repack(const ggml_tensor * t);
bool ggml_cuda_iq_skinny_is_repacked(const ggml_tensor * t);
void ggml_cuda_iq_skinny_repack(ggml_backend_cuda_context & ctx, ggml_tensor * t);
void ggml_cuda_iq_skinny_prepass(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph);
bool ggml_cuda_iq_skinny_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                                 const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cuda_iq_skinny_to_f16(const ggml_tensor * src0, half * dst, cudaStream_t stream);
