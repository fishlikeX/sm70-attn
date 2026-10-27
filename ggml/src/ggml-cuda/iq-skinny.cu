// IQ skinny runtime (T2, design A): out-of-place repack into side buffers.
//
// Unlike the k-quant codecs (byte-neutral, repacked in place inside the tensor), the IQ
// codecs grow by 22-43 % and therefore cannot be repacked in place. This module keeps a
// per-tensor side buffer laid out exactly like the in-place stream, [codes][meta] with
// meta at codes + codes_bytes(n, k), so the GEMM/repack templates in q4k-skinny.cu work
// unchanged; only the base pointer differs.
//
// The original tensor bytes stay intact, which means:
//   - any other consumer of the weight (views, copies, get) keeps seeing the original,
//   - the conservative "not read by another op" prepass rule of the k-quants is not
//     required for correctness, but is kept for now to stay symmetric with q4k/q8.
//
// Enable/disable: env LLAMA_SM70_IQ_SKINNY=0 disables (A/B benchmarking); default on.
// Volta only, like the rest of the skinny family.

#include "iq-skinny.cuh"
#include "convert.cuh"
#include "q4k-skinny.cuh"

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <mutex>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// implemented in q4k-skinny.cu (templates live there, one translation unit)
bool ggml_cuda_q4k_skinny_mul_mat_codes(ggml_type type, const uint8_t * codes, const half * input,
                                        float * output, int n, int k, int m, cudaStream_t stream);
void ggml_cuda_q4k_skinny_to_f16_codes(ggml_type type, const uint8_t * codes, half * dst,
                                       int n, int k, cudaStream_t stream);
void ggml_cuda_q4k_skinny_repack_side(ggml_type type, const void * src, uint8_t * codes,
                                      uint8_t * meta, int n, int k, cudaStream_t stream);
bool ggml_cuda_q4k_skinny_type_is_iq(ggml_type type);

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

namespace {

struct iq_skinny_entry {
    ggml_type type = GGML_TYPE_COUNT;
    uint8_t * data = nullptr; // side buffer: [codes][meta]
    size_t    bytes = 0;
    int       n = 0;
    int       k = 0;
};

std::mutex iq_skinny_mutex;
std::vector<std::unique_ptr<iq_skinny_entry>> iq_skinny_storage; // keeps descriptors alive
std::unordered_map<const ggml_tensor *, iq_skinny_entry *> iq_skinny_map;

size_t iq_skinny_codes_bytes(ggml_type type, int n, int k) {
    return ggml_cuda_skinny_codes_bytes(type, n, k);
}

size_t iq_skinny_meta_bytes(ggml_type type, int n, int k) {
    return ggml_cuda_skinny_meta_bytes(type, n, k);
}

bool iq_skinny_enabled() {
    // default OFF: locked-clock A/B on the production box (2026-10-13, see the T2 report
    // section 6.1) showed +1.9 % pp512 on one card and no change on two cards or on
    // decode, against +7.3 % VRAM on the repacked tensors - not worth a default.
    static const bool enabled = [] {
        const char * e = getenv("LLAMA_SM70_IQ_SKINNY");
        return e != nullptr && atoi(e) != 0;
    }();
    return enabled;
}

const iq_skinny_entry * iq_skinny_get(const ggml_tensor * t) {
    if (t == nullptr || t->extra == nullptr) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(iq_skinny_mutex);
    const auto it = iq_skinny_map.find(t);
    return it == iq_skinny_map.end() ? nullptr : it->second;
}

} // namespace

bool ggml_cuda_iq_skinny_can_repack(const ggml_tensor * t) {
    if (!ggml_cuda_q4k_skinny_side_supported(t->type)) {
        return false;
    }
    // route gate: IQ types follow LLAMA_SM70_IQ_SKINNY (opt-in, off by default),
    // the k-quant types follow the k-quant mode (2 = side-buffer hybrid)
    const bool enabled = ggml_cuda_q4k_skinny_type_is_iq(t->type)
        ? (ggml_cuda_skinny_mode() != 0 && iq_skinny_enabled())
        : (ggml_cuda_skinny_mode() == 2);
    if (!enabled) {
        return false;
    }
    if (t->view_src != nullptr || t->op != GGML_OP_NONE || t->extra != nullptr) {
        return false;
    }
    if (t->buffer == nullptr || ggml_backend_buffer_get_type(t->buffer) != ggml_backend_cuda_buffer_type(ggml_cuda_get_device())) {
        return false;
    }
    if (ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        return false;
    }
    if (ggml_cuda_info().devices[ggml_cuda_get_device()].cc != GGML_CUDA_CC_VOLTA) {
        return false;
    }
    if (!ggml_is_contiguous(t) || t->ne[2] != 1 || t->ne[3] != 1) {
        return false;
    }
    const int64_t k = t->ne[0];
    const int64_t n = t->ne[1];
    return k >= 256 && k % 256 == 0 && n % 32 == 0 && n >= 32;
}

bool ggml_cuda_iq_skinny_is_repacked(const ggml_tensor * t) {
    return iq_skinny_get(t) != nullptr;
}

void ggml_cuda_iq_skinny_repack(ggml_backend_cuda_context & ctx, ggml_tensor * t) {
    GGML_ASSERT(ggml_cuda_iq_skinny_can_repack(t));

    const int n = (int) t->ne[1];
    const int k = (int) t->ne[0];
    const size_t codes_bytes = iq_skinny_codes_bytes(t->type, n, k);
    const size_t meta_bytes  = iq_skinny_meta_bytes(t->type, n, k);

    // design-B decodes read the f16 grid / sign-mask LUTs; build them once per process
    {
        // the LUT is per-device memory, so the flag has to be per device too
        static bool lut_ready[GGML_CUDA_MAX_DEVICES] = {};
        const int lut_dev = ggml_cuda_get_device();
        if (!lut_ready[lut_dev]) {
            iq_skinny_init_lut(ctx.stream());
            lut_ready[lut_dev] = true;
        }
    }

    auto entry = std::make_unique<iq_skinny_entry>();
    entry->type = t->type;
    entry->n = n;
    entry->k = k;
    entry->bytes = codes_bytes + meta_bytes;
    CUDA_CHECK(cudaMalloc(&entry->data, entry->bytes));

    ggml_cuda_q4k_skinny_repack_side(t->type, t->data, entry->data, entry->data + codes_bytes,
                                    n, k, ctx.stream());

    // temporary parity probe: does the side buffer + to_f16 expansion reproduce the
    // stock dequant of the same tensor? prints the worst mismatch and its location
    static bool parity_done[GGML_TYPE_COUNT] = {};
    if (!parity_done[t->type] && getenv("LLAMA_SM70_IQ_PARITY") != nullptr) {
        parity_done[t->type] = true;
        const int64_t ne = n * (int64_t) k;
        std::vector<half> ha(ne), hb(ne);
        half * da = nullptr;
        half * db = nullptr;
        CUDA_CHECK(cudaMalloc(&da, ne * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&db, ne * sizeof(half)));
        ggml_cuda_q4k_skinny_to_f16_codes(t->type, entry->data, da, (int) n, (int) k, ctx.stream());
        CUDA_CHECK(cudaGetLastError());
        const to_fp16_cuda_t stock = ggml_get_to_fp16_cuda(t->type);
        GGML_ASSERT(stock != nullptr);
        stock(t->data, db, ne, ctx.stream());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(ha.data(), da, ne * sizeof(half), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hb.data(), db, ne * sizeof(half), cudaMemcpyDeviceToHost));
        double worst = 0.0;
        int64_t worst_i = -1, nbad = 0;
        for (int64_t i = 0; i < ne; ++i) {
            const double a = __half2float(ha[i]);
            const double b = __half2float(hb[i]);
            const double d = std::fabs(a - b);
            if (d > 1e-3 * std::max(1.0, std::fabs(b))) { ++nbad; }
            if (d > worst) { worst = d; worst_i = i; }
        }
        const int64_t row = worst_i / k, col = worst_i % k;
        fprintf(stderr, "[iqparity] %s n=%d k=%d ntiles=%d nbad=%lld/%lld worst=%.6g "
                        "at row=%lld col=%lld (expansion=%.6g stock=%.6g)%c",
                ggml_type_name(t->type), (int) n, (int) k, (int) (n / 32),
                (long long) nbad, (long long) ne, worst, (long long) row, (long long) col,
                worst_i >= 0 ? __half2float(ha[worst_i]) : 0.0,
                worst_i >= 0 ? __half2float(hb[worst_i]) : 0.0, 10);
        CUDA_CHECK(cudaFree(da));
        CUDA_CHECK(cudaFree(db));
    }

    {
        std::lock_guard<std::mutex> lock(iq_skinny_mutex);
        iq_skinny_map[t] = entry.get();
        iq_skinny_storage.push_back(std::move(entry));
    }
    t->extra = iq_skinny_storage.back().get();
}

void ggml_cuda_iq_skinny_prepass(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph) {
    if (!iq_skinny_enabled()) {
        return;
    }

    // same conservative candidate rule as the k-quant prepasses: MUL_MAT src0 only,
    // nothing that is viewed or read by another op
    std::vector<ggml_tensor *> candidates;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if (node->op == GGML_OP_MUL_MAT && node->src[0] != nullptr &&
                ggml_cuda_q4k_skinny_type_is_iq(node->src[0]->type) &&
                ggml_cuda_iq_skinny_can_repack(node->src[0])) {
            candidates.push_back(node->src[0]);
        }
    }
    if (candidates.empty()) {
        return;
    }
    for (int i = 0; i < cgraph->n_nodes && !candidates.empty(); ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (node->view_src != nullptr) {
            candidates.erase(std::remove(candidates.begin(), candidates.end(), node->view_src), candidates.end());
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const ggml_tensor * src = node->src[j];
            if (src == nullptr) {
                continue;
            }
            if (src->view_src != nullptr) {
                candidates.erase(std::remove(candidates.begin(), candidates.end(), src->view_src), candidates.end());
            }
            if (node->op != GGML_OP_MUL_MAT || j != 0) {
                candidates.erase(std::remove(candidates.begin(), candidates.end(), src), candidates.end());
            }
        }
    }

    for (ggml_tensor * t : candidates) {
        ggml_cuda_iq_skinny_repack(ctx, t);
    }
}

bool ggml_cuda_iq_skinny_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                                 const ggml_tensor * src1, ggml_tensor * dst) {
    const iq_skinny_entry * e = iq_skinny_get(src0);
    if (e == nullptr) {
        return false;
    }
    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    const int64_t m = src1->ne[1];
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst) ||
            src1->ne[2] != 1 || src1->ne[3] != 1 || m < 1 || m > 64) {
        return false;
    }

    ggml_cuda_pool_alloc<half> input(ctx.pool(), m * k);
    const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
    GGML_ASSERT(to_fp16 != nullptr);
    to_fp16(src1->data, input.get(), m * k, ctx.stream());

    return ggml_cuda_q4k_skinny_mul_mat_codes(src0->type, e->data, input.get(), (float *) dst->data,
                                              (int) n, (int) k, (int) m, ctx.stream());
}

void ggml_cuda_iq_skinny_to_f16(const ggml_tensor * src0, half * dst, cudaStream_t stream) {
    const iq_skinny_entry * e = iq_skinny_get(src0);
    GGML_ASSERT(e != nullptr);
    ggml_cuda_q4k_skinny_to_f16_codes(src0->type, e->data, dst, (int) src0->ne[1], (int) src0->ne[0], stream);
}

#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

bool ggml_cuda_iq_skinny_can_repack(const ggml_tensor * t) {
    GGML_UNUSED(t);
    return false;
}

bool ggml_cuda_iq_skinny_is_repacked(const ggml_tensor * t) {
    GGML_UNUSED(t);
    return false;
}

void ggml_cuda_iq_skinny_repack(ggml_backend_cuda_context & ctx, ggml_tensor * t) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(t);
}

void ggml_cuda_iq_skinny_prepass(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(cgraph);
}

bool ggml_cuda_iq_skinny_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                                 const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    return false;
}

void ggml_cuda_iq_skinny_to_f16(const ggml_tensor * src0, half * dst, cudaStream_t stream) {
    GGML_UNUSED(src0);
    GGML_UNUSED(dst);
    GGML_UNUSED(stream);
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
