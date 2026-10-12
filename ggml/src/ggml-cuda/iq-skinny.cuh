// IQ skinny record codecs (design A) for sm_70 (Volta) - T2 of the v100 port line.
//
// Design A bakes the grid lookup into the repacked record at load time: a fixed-width
// code per value (magnitude index + sign) at the QPN8 slot offsets the k-quant codecs
// use, so the GPU decode needs no table - it builds the f16 magnitude arithmetically
// and folds the sign in with an XOR. Per type:
//
//   IQ3_S   : 4-bit code (3-bit index into {1,3,...,15}), record 8 B,  meta 6 B  (d + scales[4])
//   IQ3_XXS : 4-bit code (3-bit index into {4,12,...,62}), record 8 B,  meta 6 B  (d + 8 nibbles)
//   IQ2_XS  : 3-bit code (2-bit index into {8,25,43}),     record 6 B,  meta 10 B (d + scales[8])
//   IQ2_S   : 3-bit code (2-bit index into {8,25,43}),     record 6 B,  meta 10 B (d + scales[8])
//
// Block cost: IQ3_S 134 B vs 110 on disk (+21.8 %), IQ3_XXS 134 vs 98 (+36.7 %),
// IQ2_XS 106 vs 74 (+43.2 %), IQ2_S 106 vs 82 (+29.3 %) per 256-value super-block.
// None of the four is byte-neutral, so unlike the k-quant codecs the repack is
// out-of-place by construction (side buffer first, load-time CPU repack later).
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

// ---- shared record helpers (shared by the four IQ codecs) ----

// 32 values of one sub-block from its 2 records in original order -> 16 half2.
// scale_of(r) gives the f16 scale of record r (per-record for the IQ2 family,
// the same scale for both records in the IQ3 family).
template <int kBits, typename DecodeT>
static __device__ __forceinline__ void iq_skinny_decode_32(
        const qskinny_code records[2], const half scale_of[2], half2 out[16]) {
    half2 scale2[2] = { __halves2half2(scale_of[0], scale_of[0]),
                        __halves2half2(scale_of[1], scale_of[1]) };
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int r = i >> 3;              // values 0..15 -> record 0, 16..31 -> record 1
        const int j = 2 * (i & 7);
        const unsigned c0 = (unsigned) qskinny_code_get(records[r], kBits * qpn8_physical_k(j + 0), kBits);
        const unsigned c1 = (unsigned) qskinny_code_get(records[r], kBits * qpn8_physical_k(j + 1), kBits);
        out[i] = __hmul2(__halves2half2(DecodeT::value(c0), DecodeT::value(c1)), scale2[r]);
    }
}

// ============================================================================ IQ3_S

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
    // d + scales[4] (d at 0, scales at 106), copied unchanged
    static __device__ __forceinline__ meta_t pack_meta(const uint8_t * block) {
        meta_t m;
        memcpy(&m.w, block, 2);
        memcpy(((uint8_t *) &m.w) + 2, block + 106, 4);
        return m;
    }
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
    // odd magnitudes: index = (mag - 1) / 2
    static __device__ __forceinline__ unsigned mag_of(const unsigned idx) {
        return 2 * idx + 1;
    }
    static __device__ __forceinline__ unsigned idx_of(const unsigned mag) {
        return (mag - 1) / 2;
    }
    static __device__ __forceinline__ qskinny_code pack_record(const uint8_t * block, const int g) {
        qskinny_code rec = { 0, 0 };
        const int p0 = 32 * (g >> 1) + 16 * (g & 1);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            unsigned mag, sign;
            value_at(block, p0 + j, mag, sign);
            qskinny_code_put(rec, idx_of(mag) | (sign << 3), bits_per_value * qpn8_physical_k(j), bits_per_value);
        }
        return rec;
    }
    static __device__ __forceinline__ half scale_half(const meta_t m, const int sub_block) {
        const uint8_t * sc = ((const uint8_t *) &m) + 2;
        const unsigned s = (sc[sub_block / 2] >> (4 * (sub_block % 2))) & 0xf;
        const float d = __half2float(*(const half *) &m);
        return __float2half_rn(d * (float) (1 + 2 * (int) s));
    }
    static __device__ __forceinline__ void decode(
            const qskinny_code records[2], const meta_t m, const int sub_block, half2 out[16]) {
        const half sc = scale_half(m, sub_block);
        const half scale2[2] = { sc, sc };
        struct dec { static __device__ __forceinline__ half value(unsigned code) {
            const half h = __int2half_rn((int) (2 * (code & 7) + 1));
            return __ushort_as_half((unsigned short) (__half_as_ushort(h) ^ (unsigned short) (((code >> 3) & 1) << 15)));
        } };
        iq_skinny_decode_32<4, dec>(records, scale2, out);
    }
};

// ============================================================================ IQ3_XXS

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
        const uint8_t * sc = ((const uint8_t *) &m) + 2;
        const unsigned s = (sc[sub_block / 2] >> (4 * (sub_block % 2))) & 0xf;
        const float d = __half2float(*(const half *) &m);
        return __float2half_rn(d * (0.5f + (float) (int) s) * 0.5f);
    }
    static __device__ __forceinline__ void decode(
            const qskinny_code records[2], const meta_t m, const int sub_block, half2 out[16]) {
        const half sc = scale_half(m, sub_block);
        const half scale2[2] = { sc, sc };
        struct dec { static __device__ __forceinline__ half value(unsigned code) {
            const half h = __int2half_rn((int) iq3xxs_mag_of(code & 7));
            return __ushort_as_half((unsigned short) (__half_as_ushort(h) ^ (unsigned short) (((code >> 3) & 1) << 15)));
        } };
        iq_skinny_decode_32<4, dec>(records, scale2, out);
    }
};

// ============================================================================ IQ2_XS / IQ2_S

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
        iq_skinny_decode_32<3, dec>(records, scale2, out);
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
        iq_skinny_decode_32<3, dec>(records, scale2, out);
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
