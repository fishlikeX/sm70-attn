// Device-side check for the IQ skinny record codecs (design A), T2 of the v100 port line.
//
// Runs the real codec header (ggml/src/ggml-cuda/iq-skinny.cuh) on the GPU against real
// IQ blocks from a GGUF (or synthesized blocks) and compares every decoded value with the
// in-tree dequantize_row_iq* reference (host, libggml-base), same bar as the host checker:
// f16 bit-exact within 1 ulp. Also times the repack and decode kernels per type.
//
// Build (from the repo root):
//   nvcc -O2 -std=c++17 -arch=sm_70 -I ggml/include -I ggml/src -I ggml/src/ggml-cuda \
//        tests/sm70-iq-host-check/iq_device_check.cu \
//        -L build-cpu/bin -lggml-base -o tests/sm70-iq-host-check/iq_device_check
//
// Usage: iq_device_check <model.gguf> --type <iq3_s|iq3_xxs|iq2_xs|iq2_s> [max_blocks]
//        iq_device_check --synth <n_blocks> --type <...>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <vector>

#define IQ_SKINNY_LUT_DEFINE 1
#include "iq-skinny.cuh"
#include "ggml-quants.h"
#include "gguf.h"

// the test links only against libggml-base, so it cannot use common.cuh's CUDA_CHECK
// (which pulls ggml_cuda_error from the CUDA backend library)
#define TEST_CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "%s:%d: CUDA error %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return 1; \
    } \
} while (0)

// repack: one thread per record (16 values)
template <ggml_type T>
__global__ void iq_repack_kernel(const uint8_t * __restrict__ blocks, int64_t n_rec,
                                 uint8_t * __restrict__ codes, uint8_t * __restrict__ meta) {
    using codec = qskinny_codec<T>;
    const int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (i >= n_rec) {
        return;
    }
    const int64_t blk = i >> 4;
    const int     g   = (int) (i & 15);
    const uint8_t * b = blocks + blk * codec::block_bytes;
    codec::store_record(codec::pack_record(b, g), codes + (blk * 16 + g) * codec::record_bytes);
    if (g == 0) {
        codec::store_meta(codec::pack_meta(b), meta + blk * codec::meta_bytes);
    }
}

// decode: one thread per (block, sub-block) -> 32 values
template <ggml_type T>
__global__ void iq_decode_kernel(const uint8_t * __restrict__ codes, const uint8_t * __restrict__ meta,
                                 int64_t n_blocks, half * __restrict__ out) {
    using codec = qskinny_codec<T>;
    const int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (i >= n_blocks * 8) {
        return;
    }
    const int64_t blk = i >> 3;
    const int     sb  = (int) (i & 7);

    qskinny_code rec[2];
    rec[0] = codec::load_record((const typename codec::record_t *) (codes + (blk * 16 + 2 * sb + 0) * codec::record_bytes));
    rec[1] = codec::load_record((const typename codec::record_t *) (codes + (blk * 16 + 2 * sb + 1) * codec::record_bytes));
    const typename codec::meta_t m = codec::load_meta(meta + blk * codec::meta_bytes);

    half2 o[16];
    codec::decode(rec, m, sb, o);

    half * dst = out + blk * 256 + 32 * sb;
#pragma unroll
    for (int t = 0; t < 16; ++t) {
        dst[2 * t + 0] = __low2half(o[t]);
        dst[2 * t + 1] = __high2half(o[t]);
    }
}

// ---- host helpers ------------------------------------------------------------------------

static uint16_t f32_to_f16(float x) {
    uint32_t fb;
    memcpy(&fb, &x, 4);
    uint32_t sign = (fb >> 16) & 0x8000u;
    int32_t e = (int32_t) ((fb >> 23) & 0xff) - 127 + 15;
    uint32_t m = fb & 0x7fffffu;
    if (((fb >> 23) & 0xff) == 0xff) {
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
        const uint32_t h2 = 1u << (shift - 1);
        if (rem > h2 || (rem == h2 && (r & 1))) r++;
        return (uint16_t) (sign | r);
    }
    uint32_t r = m >> 13;
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) {
        r++;
        if (r == 0x400u) { r = 0; e++; if (e >= 31) return (uint16_t) (sign | 0x7c00u); }
    }
    return (uint16_t) (sign | ((uint32_t) e << 10) | r);
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu;
    uint32_t m = h & 0x3ffu;
    uint32_t fbits;
    if (e == 0) {
        if (m == 0) { fbits = sign; }
        else {
            int shift = 0;
            uint32_t mm = m;
            while (!(mm & 0x400u)) { mm <<= 1; shift++; }
            mm &= 0x3ffu;
            fbits = sign | ((uint32_t) (113 - shift) << 23) | (mm << 13);
        }
    } else if (e == 31) {
        fbits = sign | 0x7f800000u | (m << 13);
    } else {
        fbits = sign | ((e - 15 + 127) << 23) | (m << 13);
    }
    float out;
    memcpy(&out, &fbits, 4);
    return out;
}

enum iq_kind { KIND_IQ3_S, KIND_IQ3_XXS, KIND_IQ2_XS, KIND_IQ2_S };

struct type_info {
    const char * name;
    ggml_type    gtype;
    size_t       block_bytes;
    size_t       record_bytes;
    size_t       meta_bytes;
};

static const type_info TYPES[4] = {
    { "iq3_s",   GGML_TYPE_IQ3_S,   sizeof(block_iq3_s),   qskinny_codec<GGML_TYPE_IQ3_S>::record_bytes,   qskinny_codec<GGML_TYPE_IQ3_S>::meta_bytes   },
    { "iq3_xxs", GGML_TYPE_IQ3_XXS, sizeof(block_iq3_xxs), qskinny_codec<GGML_TYPE_IQ3_XXS>::record_bytes, qskinny_codec<GGML_TYPE_IQ3_XXS>::meta_bytes },
    { "iq2_xs",  GGML_TYPE_IQ2_XS,  sizeof(block_iq2_xs),  qskinny_codec<GGML_TYPE_IQ2_XS>::record_bytes,  qskinny_codec<GGML_TYPE_IQ2_XS>::meta_bytes  },
    { "iq2_s",   GGML_TYPE_IQ2_S,   sizeof(block_iq2_s),   qskinny_codec<GGML_TYPE_IQ2_S>::record_bytes,   qskinny_codec<GGML_TYPE_IQ2_S>::meta_bytes   },
};

static void dequant_ref(int kind, const uint8_t * blk, float * ref) {
    switch (kind) {
        case KIND_IQ3_S:   dequantize_row_iq3_s  ((const block_iq3_s   *) blk, ref, 256); break;
        case KIND_IQ3_XXS: dequantize_row_iq3_xxs((const block_iq3_xxs *) blk, ref, 256); break;
        case KIND_IQ2_XS:  dequantize_row_iq2_xs ((const block_iq2_xs  *) blk, ref, 256); break;
        case KIND_IQ2_S:   dequantize_row_iq2_s  ((const block_iq2_s   *) blk, ref, 256); break;
    }
}

static int run_check(int kind, const uint8_t * raw, size_t n_blocks) {
    const type_info & ti = TYPES[kind];

    const size_t codes_bytes = n_blocks * 16 * ti.record_bytes;
    const size_t meta_bytes  = n_blocks * ti.meta_bytes;

    uint8_t * d_blocks = nullptr;
    uint8_t * d_codes  = nullptr;
    uint8_t * d_meta   = nullptr;
    half    * d_out    = nullptr;
    TEST_CUDA_CHECK(cudaMalloc(&d_blocks, n_blocks * ti.block_bytes));
    TEST_CUDA_CHECK(cudaMalloc(&d_codes,  codes_bytes));
    TEST_CUDA_CHECK(cudaMalloc(&d_meta,   meta_bytes));
    TEST_CUDA_CHECK(cudaMalloc(&d_out,    n_blocks * 256 * sizeof(half)));
    TEST_CUDA_CHECK(cudaMemcpy(d_blocks, raw, n_blocks * ti.block_bytes, cudaMemcpyHostToDevice));

    cudaEvent_t ev0, ev1, ev2, ev3;
    TEST_CUDA_CHECK(cudaEventCreate(&ev0));
    TEST_CUDA_CHECK(cudaEventCreate(&ev1));
    TEST_CUDA_CHECK(cudaEventCreate(&ev2));
    TEST_CUDA_CHECK(cudaEventCreate(&ev3));

    const int threads = 256;
    const int64_t n_rec = (int64_t) n_blocks * 16;
    const unsigned grid_rec = (unsigned) ((n_rec + threads - 1) / threads);
    const unsigned grid_dec = (unsigned) ((n_blocks * 8 + threads - 1) / threads);

    auto repack = [&]() {
        switch (kind) {
            case KIND_IQ3_S:   iq_repack_kernel<GGML_TYPE_IQ3_S  ><<<grid_rec, threads>>>(d_blocks, n_rec, d_codes, d_meta); break;
            case KIND_IQ3_XXS: iq_repack_kernel<GGML_TYPE_IQ3_XXS><<<grid_rec, threads>>>(d_blocks, n_rec, d_codes, d_meta); break;
            case KIND_IQ2_XS:  iq_repack_kernel<GGML_TYPE_IQ2_XS ><<<grid_rec, threads>>>(d_blocks, n_rec, d_codes, d_meta); break;
            case KIND_IQ2_S:   iq_repack_kernel<GGML_TYPE_IQ2_S  ><<<grid_rec, threads>>>(d_blocks, n_rec, d_codes, d_meta); break;
        }
    };
    auto decode = [&]() {
        switch (kind) {
            case KIND_IQ3_S:   iq_decode_kernel<GGML_TYPE_IQ3_S  ><<<grid_dec, threads>>>(d_codes, d_meta, (int64_t) n_blocks, d_out); break;
            case KIND_IQ3_XXS: iq_decode_kernel<GGML_TYPE_IQ3_XXS><<<grid_dec, threads>>>(d_codes, d_meta, (int64_t) n_blocks, d_out); break;
            case KIND_IQ2_XS:  iq_decode_kernel<GGML_TYPE_IQ2_XS ><<<grid_dec, threads>>>(d_codes, d_meta, (int64_t) n_blocks, d_out); break;
            case KIND_IQ2_S:   iq_decode_kernel<GGML_TYPE_IQ2_S  ><<<grid_dec, threads>>>(d_codes, d_meta, (int64_t) n_blocks, d_out); break;
        }
    };

    iq_skinny_init_lut(0); // design-B LUTs (no-op for design-A types)
    repack();
    decode();
    TEST_CUDA_CHECK(cudaDeviceSynchronize());

    TEST_CUDA_CHECK(cudaEventRecord(ev0));
    for (int r = 0; r < 20; ++r) repack();
    TEST_CUDA_CHECK(cudaEventRecord(ev1));
    for (int r = 0; r < 20; ++r) decode();
    TEST_CUDA_CHECK(cudaEventRecord(ev2));
    TEST_CUDA_CHECK(cudaEventSynchronize(ev2));
    float ms_repack = 0.0f, ms_decode = 0.0f;
    TEST_CUDA_CHECK(cudaEventElapsedTime(&ms_repack, ev0, ev1));
    TEST_CUDA_CHECK(cudaEventElapsedTime(&ms_decode, ev1, ev2));

    std::vector<half> out(n_blocks * 256);
    TEST_CUDA_CHECK(cudaMemcpy(out.data(), d_out, n_blocks * 256 * sizeof(half), cudaMemcpyDeviceToHost));

    uint64_t n_values = 0, n_exact = 0, n_le1ulp = 0;
    uint64_t ulp_hist[7] = {};
    double max_abs = 0.0;
    int shown = 0;
    std::vector<float> ref(256);
    auto ord = [](uint16_t h) -> int {
        return (h & 0x8000) ? -(int) (h & 0x7fff) : (int) (h & 0x7fff);
    };

    for (size_t ib = 0; ib < n_blocks; ++ib) {
        dequant_ref(kind, raw + ib * ti.block_bytes, ref.data());
        const uint16_t * got = (const uint16_t *) (out.data() + ib * 256);
        for (int p = 0; p < 256; ++p) {
            const uint16_t ref_h = f32_to_f16(ref[p]);
            const int ulp = abs(ord(got[p]) - ord(ref_h));
            n_values++;
            if (got[p] == ref_h) n_exact++;
            ulp_hist[ulp > 5 ? 6 : ulp]++;
            if (ulp <= 1) n_le1ulp++;
            else if (shown < 8) {
                printf("  >1ulp blk=%zu p=%d: got %04x (%.7g) want %04x (%.7g) ulp=%d\n",
                       ib, p, got[p], (double) f16_to_f32(got[p]), ref_h, (double) f16_to_f32(ref_h), ulp);
                shown++;
            }
            const double ad = std::fabs((double) f16_to_f32(got[p]) - (double) f16_to_f32(ref_h));
            if (ad > max_abs) max_abs = ad;
        }
    }

    printf("type           : %s\n", ti.name);
    printf("values checked : %llu\n", (unsigned long long) n_values);
    printf("f16 bit-exact  : %llu (%.6f%%)\n", (unsigned long long) n_exact, 100.0 * n_exact / n_values);
    printf("within 1 ulp   : %llu (%.6f%%)\n", (unsigned long long) n_le1ulp, 100.0 * n_le1ulp / n_values);
    printf("max abs diff   : %.6g\n", max_abs);
    printf("ulp histogram  : 0:%llu 1:%llu 2:%llu 3:%llu 4:%llu 5:%llu >5:%llu\n",
           (unsigned long long) ulp_hist[0], (unsigned long long) ulp_hist[1],
           (unsigned long long) ulp_hist[2], (unsigned long long) ulp_hist[3],
           (unsigned long long) ulp_hist[4], (unsigned long long) ulp_hist[5],
           (unsigned long long) ulp_hist[6]);
    printf("repack         : %.3f ms / %zu blocks  (%.2f GB/s of block bytes)\n",
           ms_repack / 20.0, n_blocks,
           (double) n_blocks * ti.block_bytes * 20.0 / (ms_repack / 1000.0) / 1e9);
    printf("decode         : %.3f ms / %zu blocks  (%.2f G values/s, %.2f GB/s f16 out)\n",
           ms_decode / 20.0, n_blocks,
           (double) n_blocks * 256.0 * 20.0 / (ms_decode / 1000.0) / 1e9,
           (double) n_blocks * 256.0 * 2.0 * 20.0 / (ms_decode / 1000.0) / 1e9);

    const bool pass = (n_le1ulp == n_values) && n_values > 0;
    printf("verdict        : %s\n\n", pass ? "PASS" : "FAIL");

    cudaFree(d_blocks); cudaFree(d_codes); cudaFree(d_meta); cudaFree(d_out);
    return pass ? 0 : 1;
}

int main(int argc, char ** argv) {
    const char * path = nullptr;
    int64_t max_blocks = 4096;
    bool synth = false;
    int kind = -1;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--type") == 0 && i + 1 < argc) {
            const char * tn = argv[++i];
            for (int k = 0; k < 4; ++k) {
                if (strcmp(tn, TYPES[k].name) == 0) kind = k;
            }
            if (kind < 0) { fprintf(stderr, "unknown type %s\n", tn); return 1; }
        } else if (strcmp(argv[i], "--synth") == 0 && i + 1 < argc) {
            synth = true;
            max_blocks = atoll(argv[++i]);
        } else if (argv[i][0] != '-' && path == nullptr) {
            path = argv[i];
        } else {
            max_blocks = atoll(argv[i]);
        }
    }
    if (kind < 0) {
        fprintf(stderr, "usage: %s <model.gguf> --type <iq3_s|iq3_xxs|iq2_xs|iq2_s> [max_blocks]\n"
                        "       %s --synth <n_blocks> --type <...>\n", argv[0], argv[0]);
        return 1;
    }
    const type_info & ti = TYPES[kind];

    std::vector<uint8_t> raw;
    size_t n_blocks = 0;

    if (synth) {
        const int64_t n_per_row = 512;
        const int64_t nrows     = 64;
        std::vector<float> src(n_per_row * nrows);
        for (size_t i = 0; i < src.size(); ++i) {
            src[i] = 0.5f * sinf(0.01f * (float) i) + 0.25f * cosf(0.037f * (float) i);
        }
        std::vector<float> weights(n_per_row, 1.0f);
        raw.resize(ti.block_bytes * (src.size() / 256));
        ggml_quantize_init(ti.gtype);
        switch (kind) {
            case KIND_IQ3_S:   quantize_iq3_s  (src.data(), raw.data(), nrows, n_per_row, weights.data()); break;
            case KIND_IQ3_XXS: quantize_iq3_xxs(src.data(), raw.data(), nrows, n_per_row, weights.data()); break;
            case KIND_IQ2_XS:  quantize_iq2_xs (src.data(), raw.data(), nrows, n_per_row, weights.data()); break;
            case KIND_IQ2_S:   quantize_iq2_s  (src.data(), raw.data(), nrows, n_per_row, weights.data()); break;
        }
        n_blocks = raw.size() / ti.block_bytes;
        if ((int64_t) n_blocks > max_blocks) n_blocks = (size_t) max_blocks;
        printf("source         : synth (%zu blocks of %s)\n", n_blocks, ti.name);
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
        int64_t tid = -1;
        for (int64_t i = 0; i < gguf_get_n_tensors(gctx); ++i) {
            if (gguf_get_tensor_type(gctx, i) == ti.gtype) {
                tid = i;
                break;
            }
        }
        if (tid < 0) {
            fprintf(stderr, "no %s tensor in %s\n", ti.name, path);
            return 1;
        }
        const size_t t_size = gguf_get_tensor_size(gctx, tid);
        const size_t t_off  = gguf_get_tensor_offset(gctx, tid);
        const size_t d_off  = gguf_get_data_offset(gctx);
        n_blocks = t_size / ti.block_bytes;
        if ((int64_t) n_blocks > max_blocks) n_blocks = (size_t) max_blocks;
        printf("source         : %s (%s, %zu blocks)\n", gguf_get_tensor_name(gctx, tid), path, n_blocks);
        FILE * f = fopen(path, "rb");
        if (!f) {
            return 1;
        }
        if (fseek(f, (long) (d_off + t_off), SEEK_SET) != 0) {
            return 1;
        }
        raw.resize(n_blocks * ti.block_bytes);
        if (fread(raw.data(), 1, raw.size(), f) != raw.size()) {
            return 1;
        }
        fclose(f);
        gguf_free(gctx);
    }

    printf("device         : sm_70\n");
    return run_check(kind, raw.data(), n_blocks);
}
