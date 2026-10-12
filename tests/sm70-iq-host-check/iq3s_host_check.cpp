// Host-side reference check for the planned IQ3_S "skinny" record codec (sm_70 port line).
//
// What this validates (CPU only - no Volta needed):
//   * repack  : a raw block_iq3_S block -> 16 little-endian 64-bit records holding 4-bit
//               codes (3-bit magnitude index + 1-bit sign) placed at the QPN8 slot offsets
//               the GPU codec uses (mirrors qpn8_physical_k from q-skinny-common.cuh)
//   * decode  : the planned GPU arithmetic (float scale factor rounded once to half, exact
//               half2 multiply) reproduces dequantize_row_iq3_s from the same tree, up to
//               f16 rounding - the same bar the in-tree Q*_K skinny codecs document.
//
// The checker has a --selftest mode that deliberately corrupts the sign handling; it must
// report mismatches, which proves the check is not vacuously passing.
//
// Usage:
//   iq3s_host_check <model.gguf> [max_blocks] [--selftest]
// The first IQ3_S tensor of the GGUF is used.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "ggml-quants.h"
#include "gguf.h"

// ---- f16 helpers (round-to-nearest-even, matches CUDA __float2half_rn on normal values) ----

static uint16_t f32_to_f16(float x) {
    uint32_t f;
    memcpy(&f, &x, 4);
    const uint32_t sign = (f >> 16) & 0x8000u;
    int32_t  e = (int32_t) ((f >> 23) & 0xff) - 127 + 15;
    uint32_t m = f & 0x7fffffu;
    if (((f >> 23) & 0xff) == 0xff) {
        return (uint16_t) (sign | 0x7c00u | (m ? 0x200u : 0u)); // inf/nan
    }
    if (e >= 31) {
        return (uint16_t) (sign | 0x7c00u);                     // overflow -> inf
    }
    if (e <= 0) {
        if (e < -10) {
            return (uint16_t) sign;                             // -> +-0
        }
        m |= 0x800000u;                                         // implicit bit
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
            // subnormal: value = m * 2^-24 = (1+fr) * 2^(k-24), k = 10 - shift
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
//      LICENSE.v100-skinny) ----

static int qpn8_physical_k_host(int logical_k) {
    const int local = logical_k & 7;
    return (logical_k & 8) + (local >> 1) + ((local & 1) << 2);
}

// ---- IQ3_S record codec, design A: one 4-bit code per value ------------------------------
//
// code = magnitude index | (sign << 3);  magnitude = 2*index + 1, index in 0..7
// (iq3s_grid byte values are exactly {1,3,5,7,9,11,13,15})
// record j holds the code for the j-th of 16 values at bit offset 4*qpn8_physical_k(j)

static constexpr int IQ3S_RECORD_BYTES = 8;   // 16 values * 4 bit
static constexpr int IQ3S_META_BYTES   = 6;   // d (f16) + scales[4]

// original block position of the j-th value of record g (g = 0..15)
static int iq3s_record_value_pos(int g, int j) {
    return 32 * (g >> 1) + 16 * (g & 1) + j;
}

// per-value (magnitude, sign) of the original block, as dequantize_row_iq3_s sees it
static void iq3s_block_value(const block_iq3_s & blk, int p, int & mag, int & sign) {
    const int sb = p / 32;
    const int l  = (p % 32) / 8;
    const int j  = p % 8;
    const uint8_t * qs_sb    = blk.qs  + 8 * sb;
    const uint8_t   qh_sb    = blk.qh[sb];
    const uint8_t * signs_sb = blk.signs + 4 * sb;
    const uint8_t * grid;
    if (j < 4) {
        grid = (const uint8_t *) (iq3s_grid + (qs_sb[2*l+0] | (((unsigned) qh_sb << (8-2*l)) & 256)));
        sign = (signs_sb[l] >> j) & 1;
    } else {
        grid = (const uint8_t *) (iq3s_grid + (qs_sb[2*l+1] | (((unsigned) qh_sb << (7-2*l)) & 256)));
        sign = (signs_sb[l] >> j) & 1;
    }
    mag = grid[j & 3];
}

static uint64_t iq3s_pack_record(const block_iq3_s & blk, int g, bool corrupt_sign) {
    uint64_t rec = 0;
    for (int j = 0; j < 16; ++j) {
        int mag, sign;
        iq3s_block_value(blk, iq3s_record_value_pos(g, j), mag, sign);
        if (corrupt_sign) {
            sign ^= 1; // --selftest: the check must catch this
        }
        const unsigned code = (unsigned) ((mag - 1) / 2) | ((unsigned) sign << 3);
        rec |= (uint64_t) code << (4 * qpn8_physical_k_host(j));
    }
    return rec;
}

// planned GPU decode: value = half2_mul( half(2*idx+1) XOR sign, half(d * (1 + 2*scale)) )
static uint16_t iq3s_decode_value(uint64_t rec, int j, uint16_t d_h, uint8_t scales, int sb) {
    const unsigned code = (unsigned) ((rec >> (4 * qpn8_physical_k_host(j))) & 0xf);
    const int idx  = (int) (code & 7);
    const int sign = (int) ((code >> 3) & 1);

    const float sc_f = f16_to_f32(d_h) * (float) (1 + 2 * ((scales >> 4 * (sb % 2)) & 0xf));
    const uint16_t sc_h = f32_to_f16(sc_f);

    const uint16_t mag_h = f32_to_f16((float) (2 * idx + 1)); // exact
    const uint16_t v_h   = f32_to_f16(f16_to_f32(mag_h) * f16_to_f32(sc_h));
    return (uint16_t) (v_h ^ (sign << 15));
}

// ---- main ----

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [max_blocks] [--selftest]\n", argv[0]);
        return 1;
    }
    const char * path = argv[1];
    int64_t max_blocks = 512;
    bool selftest = false;
    bool all_tensors = false;
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--selftest") == 0) {
            selftest = true;
        } else if (strcmp(argv[i], "--all-tensors") == 0) {
            all_tensors = true;
        } else {
            max_blocks = atoll(argv[i]);
        }
    }

    gguf_init_params gp = {};
    gp.no_alloc = true;
    gguf_context * gctx = gguf_init_from_file(path, gp);
    if (!gctx) {
        fprintf(stderr, "failed to open %s\n", path);
        return 1;
    }

    // collect the IQ3_S tensors to check (all of them with --all-tensors, else the first)
    std::vector<int64_t> tensors;
    for (int64_t i = 0; i < gguf_get_n_tensors(gctx); ++i) {
        if (gguf_get_tensor_type(gctx, i) == GGML_TYPE_IQ3_S) {
            tensors.push_back(i);
            if (!all_tensors) {
                break;
            }
        }
    }
    if (tensors.empty()) {
        fprintf(stderr, "no IQ3_S tensor in %s\n", path);
        return 1;
    }

    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "failed to read %s\n", path);
        return 1;
    }
    const size_t d_off = gguf_get_data_offset(gctx);

    uint64_t n_values = 0, n_exact = 0, n_le1ulp = 0;
    double max_abs = 0.0;
    int n_gt1ulp_shown = 0;
    uint64_t ulp_hist[7] = {}; // 0..5, 6 = ">5"
    std::vector<float> ref(256);

    for (int64_t tensor_id : tensors) {
        const char * tensor_name = gguf_get_tensor_name(gctx, tensor_id);
        const size_t t_size = gguf_get_tensor_size(gctx, tensor_id);
        const size_t t_off  = gguf_get_tensor_offset(gctx, tensor_id);
        const size_t block_bytes = sizeof(block_iq3_s);
        size_t n_blocks = t_size / block_bytes;
        if ((int64_t) n_blocks > max_blocks) {
            n_blocks = (size_t) max_blocks;
        }
        printf("tensor      : %s (%zu blocks total, checking %zu)\n",
               tensor_name, t_size / block_bytes, n_blocks);
        fflush(stdout);
        if (n_blocks == 0) {
            continue;
        }
        if (fseek(f, (long) (d_off + t_off), SEEK_SET) != 0) {
            fprintf(stderr, "seek failed\n");
            return 1;
        }
        std::vector<uint8_t> raw(n_blocks * block_bytes);
        if (fread(raw.data(), 1, raw.size(), f) != raw.size()) {
            fprintf(stderr, "short read\n");
            return 1;
        }

    for (size_t ib = 0; ib < n_blocks; ++ib) {
        const block_iq3_s & blk = *(const block_iq3_s *) (raw.data() + ib * block_bytes);
        dequantize_row_iq3_s(&blk, ref.data(), 256);

        const uint16_t d_h = *(const uint16_t *) &blk.d;
        for (int g = 0; g < 16; ++g) {
            const uint64_t rec = iq3s_pack_record(blk, g, selftest);
            const int sb = g >> 1;
            for (int j = 0; j < 16; ++j) {
                const int p = iq3s_record_value_pos(g, j);
                const uint16_t got_h = iq3s_decode_value(rec, j, d_h, blk.scales[sb / 2], sb);
                const uint16_t ref_h = f32_to_f16(ref[p]);
                const float got = f16_to_f32(got_h);
                const float want = f16_to_f32(ref_h);

                n_values++;
                if (got_h == ref_h) {
                    n_exact++;
                }
                // f16 ulp distance on the ordered integer representation (sign-magnitude)
                auto ord = [](uint16_t h) -> int {
                    return (h & 0x8000) ? -(int) (h & 0x7fff) : (int) (h & 0x7fff);
                };
                const int ulp = abs(ord(got_h) - ord(ref_h));
                ulp_hist[ulp > 5 ? 6 : ulp]++;
                if (ulp <= 1) {
                    n_le1ulp++;
                } else if (n_gt1ulp_shown < 8) {
                    const unsigned code = (unsigned) ((rec >> (4 * qpn8_physical_k_host(j))) & 0xf);
                    const float sc_f = f16_to_f32(d_h) * (float) (1 + 2 * ((blk.scales[sb / 2] >> 4 * (sb % 2)) & 0xf));
                    printf("  >1ulp %s blk=%zu rec=%d sb=%d j=%2d: code=%u (mag=%d sign=%d) d=%.6g scf=%.6g ulp=%d got %04x (%.7g) want %04x (%.7g)\n",
                           tensor_name, ib, g, sb, j, code, (int) (2 * (code & 7) + 1), (int) ((code >> 3) & 1),
                           (double) f16_to_f32(d_h), (double) sc_f, ulp,
                           got_h, got, ref_h, want);
                    n_gt1ulp_shown++;
                }
                const double ad = std::fabs((double) got - (double) want);
                if (ad > max_abs) {
                    max_abs = ad;
                }
            }
        }
    }

    }
    fclose(f);

    printf("\nvalues checked : %llu\n", (unsigned long long) n_values);
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

    const bool within_1ulp = (n_le1ulp == n_values);
    // the gate: every value within 1 f16 ulp of the in-tree reference (the in-tree Q*_K
    // codecs document the same bar); the selftest must instead report mismatches
    const bool ok = selftest ? !within_1ulp : within_1ulp;
    printf("\n%s\n", ok ? (selftest ? "[selftest] PASS (mismatches detected)" : "PASS")
                       : (selftest ? "[selftest] FAIL (check is blind)" : "FAIL"));

    gguf_free(gctx);
    return ok ? 0 : 1;
}
