// Host-side reference check for the planned IQ* "skinny" record codecs (sm_70 port line, T2).
//
// Validates, per type (IQ3_S / IQ3_XXS / IQ2_XS / IQ2_S):
//   * repack : a raw block -> 16 little-endian records of fixed-width codes
//              (magnitude index + sign bit) placed at the QPN8 slot offsets the GPU
//              codec uses (mirrors qpn8_physical_k from q-skinny-common.cuh)
//   * decode : the planned GPU arithmetic (float scale factor rounded once to half,
//              exact half2 multiply) reproduces the in-tree dequantize_row_iq* up to
//              f16 rounding - the bar the in-tree Q*_K skinny codecs document.
//
// Data sources: a GGUF tensor of the requested type, or --synth (random f32 through the
// in-tree quantizer with an all-ones imatrix). --selftest corrupts the sign and must
// report mismatches (proves the check is not vacuous).
//
// Usage:
//   iq_host_check <model.gguf> --type <iq3_s|iq3_xxs|iq2_xs|iq2_s> [max_blocks]
//                 [--all-tensors] [--selftest]
//   iq_host_check --synth <n_blocks> --type <...> [--selftest]

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "ggml-quants.h"
#include "gguf.h"

// ---- f16 helpers (round-to-nearest-even; matches CUDA __float2half_rn on normal values) ----

static uint16_t f32_to_f16(float x) {
    uint32_t f;
    memcpy(&f, &x, 4);
    const uint32_t sign = (f >> 16) & 0x8000u;
    int32_t  e = (int32_t) ((f >> 23) & 0xff) - 127 + 15;
    uint32_t m = f & 0x7fffffu;
    if (((f >> 23) & 0xff) == 0xff) {
        return (uint16_t) (sign | 0x7c00u | (m ? 0x200u : 0u));
    }
    if (e >= 31) {
        return (uint16_t) (sign | 0x7c00u);
    }
    if (e <= 0) {
        if (e < -10) {
            return (uint16_t) sign;
        }
        m |= 0x800000u;
        const uint32_t shift = (uint32_t) (14 - e);
        uint32_t r = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1);
        const uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (r & 1))) {
            r++;
        }
        return (uint16_t) (sign | r);
    }
    uint32_t r = m >> 13;
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) {
        r++;
        if (r == 0x400u) {
            r = 0;
            e++;
            if (e >= 31) {
                return (uint16_t) (sign | 0x7c00u);
            }
        }
    }
    return (uint16_t) (sign | ((uint32_t) e << 10) | r);
}

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1fu;
    const uint32_t m = h & 0x3ffu;
    uint32_t f;
    if (e == 0) {
        if (m == 0) {
            f = sign;
        } else {
            int shift = 0;
            uint32_t mm = m;
            while (!(mm & 0x400u)) { mm <<= 1; shift++; }
            mm &= 0x3ffu;
            f = sign | ((uint32_t) (113 - shift) << 23) | (mm << 13);
        }
    } else if (e == 31) {
        f = sign | 0x7f800000u | (m << 13);
    } else {
        f = sign | ((e - 15 + 127) << 23) | (m << 13);
    }
    float out;
    memcpy(&out, &f, 4);
    return out;
}

// ---- QPN8 slot mapping; mirrors ggml/src/ggml-cuda/q-skinny-common.cuh (MIT, see
//      LICENSE.v100-skinny in ggml/src/ggml-cuda/) ----

static int qpn8_physical_k_host(int logical_k) {
    const int local = logical_k & 7;
    return (logical_k & 8) + (local >> 1) + ((local & 1) << 2);
}

// ---- per-type codec description -----------------------------------------------------------

enum iq_type {
    IQ_TYPE_IQ3_S,
    IQ_TYPE_IQ3_XXS,
    IQ_TYPE_IQ2_XS,
    IQ_TYPE_IQ2_S,
};

struct iq_codec {
    const char * name;
    int bits_per_value;       // 4 (IQ3_*) or 3 (IQ2_*)
    int n_mag;                // distinct magnitudes (8 or 3)
    const uint8_t * mags;     // valid magnitudes in code order
    size_t block_bytes;
    // value extraction from one block, p in [0,256): magnitude byte and sign bit
    void (*value_at)(const uint8_t * block, int p, int & mag, int & sign, int & scale_nib);
};

// ------------------------------------------------------------------ IQ3_S
static const uint8_t iq3s_mags[8] = {1,3,5,7,9,11,13,15};
static void iq3s_value_at(const uint8_t * block, int p, int & mag, int & sign, int & scale_nib) {
    // block_iq3_s: d(2) qs[64] qh[8] signs[32] scales[4]
    const uint8_t * qs    = block + 2;
    const uint8_t * qh    = block + 2 + 64;
    const uint8_t * signs = block + 2 + 64 + 8;
    const uint8_t * sc    = block + 2 + 64 + 8 + 32;
    const int sb = p / 32;
    const int l  = (p % 32) / 8;
    const int j  = p % 8;
    const uint8_t * qs_sb = qs + 8*sb;
    const uint8_t   qh_sb = qh[sb];
    const uint8_t * sg_sb = signs + 4*sb;
    const uint8_t * grid;
    if (j < 4) {
        grid = (const uint8_t *) (iq3s_grid + (qs_sb[2*l+0] | (((unsigned) qh_sb << (8-2*l)) & 256)));
    } else {
        grid = (const uint8_t *) (iq3s_grid + (qs_sb[2*l+1] | (((unsigned) qh_sb << (7-2*l)) & 256)));
    }
    mag  = grid[j & 3];
    sign = (sg_sb[l] >> j) & 1;
    scale_nib = (sc[sb/2] >> 4*(sb%2)) & 0xf;   // per 32 values
}

// ------------------------------------------------------------------ IQ3_XXS
static const uint8_t iq3xxs_mags[8] = {4,12,20,28,36,44,52,62};
static int mag_index(const uint8_t * mags, int n, int mag) {
    for (int i = 0; i < n; ++i) {
        if (mags[i] == mag) return i;
    }
    return -1;
}
static void iq3xxs_value_at(const uint8_t * block, int p, int & mag, int & sign, int & scale_nib) {
    // block_iq3_xxs: d(2) qs[96]  (gas/scale live in qs[64..71])
    const uint8_t * qs = block + 2;
    const uint16_t * gas = (const uint16_t *) (qs + 64);
    const int sb = p / 32;
    const int l  = (p % 32) / 8;
    const int j  = p % 8;
    const uint8_t * q3_sb = qs + 8*sb;
    const uint8_t * grid = (const uint8_t *) (iq3xxs_grid + q3_sb[2*l + (j < 4 ? 0 : 1)]);
    mag = grid[j & 3];
    const uint32_t aux32 = (uint32_t) gas[2*sb] | ((uint32_t) gas[2*sb+1] << 16);
    const uint8_t signs_byte = ksigns_iq2xs[(aux32 >> 7*l) & 127];
    sign = (signs_byte >> j) & 1;
    scale_nib = (aux32 >> 28) & 0xf;            // per 32 values
}

// ------------------------------------------------------------------ IQ2_XS
static const uint8_t iq2_mags[3] = {8,25,43};
static void iq2xs_value_at(const uint8_t * block, int p, int & mag, int & sign, int & scale_nib) {
    // block_iq2_xs: d(2) qs[32x u16] scales[8]
    const uint8_t * qs = block + 2;
    const uint8_t * sc = block + 2 + 64;
    const uint16_t * q2 = (const uint16_t *) qs;
    const int sb = p / 32;
    const int l  = (p % 32) / 8;
    const int j  = p % 8;
    const uint16_t v = q2[4*sb + l];
    const uint8_t * grid = (const uint8_t *) (iq2xs_grid + (v & 511));
    mag = grid[j];
    sign = (ksigns_iq2xs[v >> 9] >> j) & 1;
    scale_nib = (sc[sb] >> 4*(l/2)) & 0xf;      // per 16 values (record)
}

// ------------------------------------------------------------------ IQ2_S
static void iq2s_value_at(const uint8_t * block, int p, int & mag, int & sign, int & scale_nib) {
    // block_iq2_s: d(2) qs[64] qh[8] scales[8]   (signs live in qs[32..63])
    const uint8_t * qs = block + 2;
    const uint8_t * qh = block + 2 + 64;
    const uint8_t * sc = block + 2 + 64 + 8;
    const int sb = p / 32;
    const int l  = (p % 32) / 8;
    const int j  = p % 8;
    const uint8_t * grid = (const uint8_t *) (iq2s_grid + (qs[4*sb+l] | (((unsigned) qh[sb] << (8-2*l)) & 0x300)));
    mag  = grid[j];
    sign = (qs[32 + 4*sb + l] >> j) & 1;
    scale_nib = (sc[sb] >> 4*(l/2)) & 0xf;
}

static const iq_codec & codec_for(iq_type t) {
    static const iq_codec codecs[4] = {
        { "iq3_s",   4, 8, iq3s_mags,   sizeof(block_iq3_s),   iq3s_value_at   },
        { "iq3_xxs", 4, 8, iq3xxs_mags, sizeof(block_iq3_xxs), iq3xxs_value_at },
        { "iq2_xs",  3, 3, iq2_mags,    sizeof(block_iq2_xs),  iq2xs_value_at  },
        { "iq2_s",   3, 3, iq2_mags,    sizeof(block_iq2_s),   iq2s_value_at   },
    };
    return codecs[t];
}

// scale factor applied by the dequant for a value with scale nibble s (per type)
static float scale_factor(iq_type t, float d, int s) {
    switch (t) {
        case IQ_TYPE_IQ3_S:   return d * (float) (1 + 2*s);
        case IQ_TYPE_IQ3_XXS: return d * (0.5f + (float) s) * 0.5f;
        default:              return d * (0.5f + (float) s) * 0.25f;
    }
}

// original block position of the j-th value of record g (16 values per record)
static int record_value_pos(int g, int j) {
    return 32 * (g >> 1) + 16 * (g & 1) + j;
}

// pack one record: codes at bits_per_value * qpn8_physical_k(j), low bit = magnitude index
static uint64_t pack_record(iq_type t, const uint8_t * block, int g, bool corrupt_sign) {
    const iq_codec & c = codec_for(t);
    uint64_t rec = 0;
    for (int j = 0; j < 16; ++j) {
        int mag, sign, snib;
        c.value_at(block, record_value_pos(g, j), mag, sign, snib);
        if (corrupt_sign) {
            sign ^= 1;
        }
        const int idx = (c.n_mag == 8 && t == IQ_TYPE_IQ3_S) ? (mag - 1) / 2 : mag_index(c.mags, c.n_mag, mag);
        if (idx < 0) {
            fprintf(stderr, "invalid magnitude %d\n", mag);
            abort();
        }
        const unsigned code = (unsigned) idx | ((unsigned) sign << (c.bits_per_value - 1));
        rec |= (uint64_t) code << (c.bits_per_value * qpn8_physical_k_host(j));
    }
    return rec;
}

// planned GPU decode: value = half2_mul( half(mag) XOR sign, half(scale_factor) )
static uint16_t decode_value(iq_type t, uint64_t rec, int j, uint16_t d_h, int scale_nib) {
    const iq_codec & c = codec_for(t);
    const unsigned code = (unsigned) ((rec >> (c.bits_per_value * qpn8_physical_k_host(j)))
            & ((1u << c.bits_per_value) - 1));
    const int idx  = (int) (code & ((1u << (c.bits_per_value - 1)) - 1));
    const int sign = (int) ((code >> (c.bits_per_value - 1)) & 1);

    const uint16_t sc_h = f32_to_f16(scale_factor(t, f16_to_f32(d_h), scale_nib));
    const uint16_t mag_h = f32_to_f16((float) c.mags[idx]); // exact for all magnitudes
    const uint16_t v_h = f32_to_f16(f16_to_f32(mag_h) * f16_to_f32(sc_h));
    return (uint16_t) (v_h ^ (sign << 15));
}

// ---- main ---------------------------------------------------------------------------------

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s <model.gguf> --type <iq3_s|iq3_xxs|iq2_xs|iq2_s> [max_blocks] [--all-tensors] [--selftest]\n"
                "       %s --synth <n_blocks> --type <...> [--selftest]\n", argv[0], argv[0]);
        return 1;
    }

    const char * path = nullptr;
    int64_t max_blocks = 512;
    bool selftest = false, all_tensors = false, synth = false;
    iq_type type = IQ_TYPE_IQ3_S;
    bool have_type = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--type") == 0 && i + 1 < argc) {
            const char * tn = argv[++i];
            if      (strcmp(tn, "iq3_s")   == 0) type = IQ_TYPE_IQ3_S;
            else if (strcmp(tn, "iq3_xxs") == 0) type = IQ_TYPE_IQ3_XXS;
            else if (strcmp(tn, "iq2_xs")  == 0) type = IQ_TYPE_IQ2_XS;
            else if (strcmp(tn, "iq2_s")   == 0) type = IQ_TYPE_IQ2_S;
            else { fprintf(stderr, "unknown type %s\n", tn); return 1; }
            have_type = true;
        } else if (strcmp(argv[i], "--synth") == 0 && i + 1 < argc) {
            synth = true;
            max_blocks = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--selftest") == 0) {
            selftest = true;
        } else if (strcmp(argv[i], "--all-tensors") == 0) {
            all_tensors = true;
        } else if (argv[i][0] != '-' && path == nullptr) {
            path = argv[i];
        } else {
            max_blocks = atoll(argv[i]);
        }
    }
    if (!have_type) {
        fprintf(stderr, "--type is required\n");
        return 1;
    }
    const iq_codec & c = codec_for(type);

    uint64_t n_values = 0, n_exact = 0, n_le1ulp = 0;
    double max_abs = 0.0;
    int n_gt1ulp_shown = 0;
    uint64_t ulp_hist[7] = {};
    std::vector<float> ref(256);

    auto check_blocks = [&](const uint8_t * raw, size_t n_blocks, const char * label) {
        for (size_t ib = 0; ib < n_blocks; ++ib) {
            const uint8_t * blk = raw + ib * c.block_bytes;
            switch (type) {
                case IQ_TYPE_IQ3_S:   dequantize_row_iq3_s  ((const block_iq3_s   *) blk, ref.data(), 256); break;
                case IQ_TYPE_IQ3_XXS: dequantize_row_iq3_xxs((const block_iq3_xxs *) blk, ref.data(), 256); break;
                case IQ_TYPE_IQ2_XS:  dequantize_row_iq2_xs ((const block_iq2_xs  *) blk, ref.data(), 256); break;
                case IQ_TYPE_IQ2_S:   dequantize_row_iq2_s  ((const block_iq2_s   *) blk, ref.data(), 256); break;
            }
            const uint16_t d_h = *(const uint16_t *) blk; // d is the first field of every block
            for (int g = 0; g < 16; ++g) {
                const uint64_t rec = pack_record(type, blk, g, selftest);
                for (int j = 0; j < 16; ++j) {
                    const int p = record_value_pos(g, j);
                    int mag, sign, snib;
                    c.value_at(blk, p, mag, sign, snib);
                    const uint16_t got_h = decode_value(type, rec, j, d_h, snib);
                    const uint16_t ref_h = f32_to_f16(ref[p]);
                    const float got = f16_to_f32(got_h);
                    const float want = f16_to_f32(ref_h);

                    n_values++;
                    if (got_h == ref_h) n_exact++;
                    auto ord = [](uint16_t h) -> int {
                        return (h & 0x8000) ? -(int) (h & 0x7fff) : (int) (h & 0x7fff);
                    };
                    const int ulp = abs(ord(got_h) - ord(ref_h));
                    ulp_hist[ulp > 5 ? 6 : ulp]++;
                    if (ulp <= 1) {
                        n_le1ulp++;
                    } else if (n_gt1ulp_shown < 8) {
                        printf("  >1ulp %s blk=%zu rec=%d j=%2d: mag=%d sign=%d snib=%d d=%.6g ulp=%d got %04x (%.7g) want %04x (%.7g)\n",
                               label, ib, g, j, mag, sign, snib, (double) f16_to_f32(d_h), ulp,
                               got_h, got, ref_h, want);
                        n_gt1ulp_shown++;
                    }
                    const double ad = std::fabs((double) got - (double) want);
                    if (ad > max_abs) max_abs = ad;
                }
            }
        }
    };

    if (synth) {
        // synthesize valid blocks through the in-tree quantizer (all-ones imatrix)
        const int64_t n_per_row = 512;             // 2 super-blocks per row
        const int64_t nrows     = 64;
        std::vector<float> src(n_per_row * nrows);
        for (size_t i = 0; i < src.size(); ++i) {
            src[i] = 0.5f * sinf(0.01f * (float) i) + 0.25f * cosf(0.037f * (float) i);
        }
        std::vector<float> weights(n_per_row, 1.0f);
        std::vector<uint8_t> dst(c.block_bytes * (src.size() / 256));
        const ggml_type qtype = type == IQ_TYPE_IQ3_S   ? GGML_TYPE_IQ3_S   :
                                type == IQ_TYPE_IQ3_XXS ? GGML_TYPE_IQ3_XXS :
                                type == IQ_TYPE_IQ2_XS  ? GGML_TYPE_IQ2_XS  : GGML_TYPE_IQ2_S;
        ggml_quantize_init(qtype); // builds the iq2/iq3 lattice tables the quantizers need
        size_t written = 0;
        switch (type) {
            case IQ_TYPE_IQ3_S:   written = quantize_iq3_s  (src.data(), dst.data(), nrows, n_per_row, weights.data()); break;
            case IQ_TYPE_IQ3_XXS: written = quantize_iq3_xxs(src.data(), dst.data(), nrows, n_per_row, weights.data()); break;
            case IQ_TYPE_IQ2_XS:  written = quantize_iq2_xs (src.data(), dst.data(), nrows, n_per_row, weights.data()); break;
            case IQ_TYPE_IQ2_S:   written = quantize_iq2_s  (src.data(), dst.data(), nrows, n_per_row, weights.data()); break;
        }
        size_t n_blocks = written / c.block_bytes;
        if ((int64_t) n_blocks > max_blocks) n_blocks = (size_t) max_blocks;
        printf("synth       : type=%s rows=%lld blocks=%zu\n", c.name, (long long) nrows, n_blocks);
        check_blocks(dst.data(), n_blocks, "synth");
    } else {
        if (path == nullptr) {
            fprintf(stderr, "need a gguf path or --synth\n");
            return 1;
        }
        gguf_init_params gp = {};
        gp.no_alloc = true;
        gguf_context * gctx = gguf_init_from_file(path, gp);
        if (!gctx) {
            fprintf(stderr, "failed to open %s\n", path);
            return 1;
        }
        const ggml_type want = type == IQ_TYPE_IQ3_S   ? GGML_TYPE_IQ3_S   :
                               type == IQ_TYPE_IQ3_XXS ? GGML_TYPE_IQ3_XXS :
                               type == IQ_TYPE_IQ2_XS  ? GGML_TYPE_IQ2_XS  : GGML_TYPE_IQ2_S;
        std::vector<int64_t> tensors;
        for (int64_t i = 0; i < gguf_get_n_tensors(gctx); ++i) {
            if (gguf_get_tensor_type(gctx, i) == want) {
                tensors.push_back(i);
                if (!all_tensors) break;
            }
        }
        if (tensors.empty()) {
            fprintf(stderr, "no %s tensor in %s\n", c.name, path);
            return 1;
        }
        FILE * f = fopen(path, "rb");
        if (!f) {
            fprintf(stderr, "failed to read %s\n", path);
            return 1;
        }
        const size_t d_off = gguf_get_data_offset(gctx);
        for (int64_t tid : tensors) {
            const char * tn = gguf_get_tensor_name(gctx, tid);
            const size_t t_size = gguf_get_tensor_size(gctx, tid);
            const size_t t_off  = gguf_get_tensor_offset(gctx, tid);
            size_t n_blocks = t_size / c.block_bytes;
            if ((int64_t) n_blocks > max_blocks) n_blocks = (size_t) max_blocks;
            printf("tensor      : %s (%zu blocks total, checking %zu)\n", tn, t_size / c.block_bytes, n_blocks);
            fflush(stdout);
            if (n_blocks == 0) continue;
            if (fseek(f, (long) (d_off + t_off), SEEK_SET) != 0) {
                fprintf(stderr, "seek failed\n");
                return 1;
            }
            std::vector<uint8_t> raw(n_blocks * c.block_bytes);
            if (fread(raw.data(), 1, raw.size(), f) != raw.size()) {
                fprintf(stderr, "short read\n");
                return 1;
            }
            check_blocks(raw.data(), n_blocks, tn);
        }
        fclose(f);
        gguf_free(gctx);
    }

    printf("\ntype           : %s\n", c.name);
    printf("values checked : %llu\n", (unsigned long long) n_values);
    printf("f16 bit-exact  : %llu (%.6f%%)\n", (unsigned long long) n_exact,
           100.0 * (double) n_exact / (double) n_values);
    printf("within 1 ulp   : %llu (%.6f%%)\n", (unsigned long long) n_le1ulp,
           100.0 * (double) n_le1ulp / (double) n_values);
    printf("max abs diff   : %.6g\n", max_abs);
    printf("ulp histogram  : 0:%llu 1:%llu 2:%llu 3:%llu 4:%llu 5:%llu >5:%llu\n",
           (unsigned long long) ulp_hist[0], (unsigned long long) ulp_hist[1],
           (unsigned long long) ulp_hist[2], (unsigned long long) ulp_hist[3],
           (unsigned long long) ulp_hist[4], (unsigned long long) ulp_hist[5],
           (unsigned long long) ulp_hist[6]);

    const bool within_1ulp = (n_le1ulp == n_values) && n_values > 0;
    const bool ok = selftest ? !within_1ulp : within_1ulp;
    printf("\n%s\n", ok ? (selftest ? "[selftest] PASS (mismatches detected)" : "PASS")
                       : (selftest ? "[selftest] FAIL (check is blind)" : "FAIL"));
    return ok ? 0 : 1;
}
