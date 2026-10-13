// d256_kv_roworder_check.cpp — R1 host round-trip gate for the d256 merge.
//
// WHAT THIS PROVES (施工单-2 §1 U2):
//   The sm_70 d256 flash-attention reads its staged f16 KV mirror through two
//   stride conventions that must encode the SAME logical KV matrix:
//
//     * fork  = POSITION-MAJOR  : the dequant writer (to_fp16_nc) linearizes dst
//                                 as [ctx][hkv][D]; the kernel reader uses
//                                 row_stride = hkv*D, head_stride = D.
//       (commit 58a4ec231 "0.69 divergence" root fix — the old head-major read
//        strides scrambled K against this writer.)
//
//     * v100  = HEAD-MAJOR      : writer linearizes [hkv][ctx][D]; reader uses
//                                 row_stride = D, head_stride = ctx*D.
//       (the layout the KV-windowing merge is built on.)
//
//   R1 asserts, element-by-element over a deterministic logical tensor
//   K[b][h][p][d]:
//     (1) PM-writer + PM-reader  reconstructs K  (fork self-consistency)
//     (2) HM-writer + HM-reader  reconstructs K  (v100 self-consistency)
//     (3) the two reconstructions are IDENTICAL  -> both encodings = ONE matrix
//         (the precondition that makes the d256 merge safe)
//     (4) PM-writer + HM-reader  MISMATCHES (sensitivity guard: this is the
//         historical 0.69 scramble; proves the test can actually catch it)
//
// Includes odd hkv, ctx not a multiple of the tile, and batch nb>1 tails.
// Pure host code, no ggml/CUDA needed. Returns 0 iff every check passes.

#include <cstdint>
#include <cstdio>
#include <vector>
#include <string>

namespace {

// Deterministic injective value for a logical element (b,h,p,d).
static inline int64_t logical(int64_t b, int64_t h, int64_t p, int64_t d) {
    return ((b * 100000 + h) * 100000 + p) * 1000 + d;
}

// ---- index mappers (offset in ELEMENTS within a single (b) plane) ----
// Position-major  [ctx][hkv][D]:   off = p*(hkv*D) + h*D + d
static inline size_t off_pm(int64_t p, int64_t h, int64_t d, int64_t hkv, int64_t D) {
    return (size_t)(p * hkv * D + h * D + d);
}
// Head-major      [hkv][ctx][D]:   off = h*(ctx*D) + p*D + d
static inline size_t off_hm(int64_t p, int64_t h, int64_t d, int64_t ctx, int64_t D) {
    return (size_t)(h * ctx * D + p * D + d);
}

// Reader formulas expressed exactly as the kernel uses row/head strides:
//   addr(p,h,d) = base_plane + p*row_stride + h*head_stride + d
struct Reader {
    int64_t row_stride;   // step per position (p)
    int64_t head_stride;  // step per head (h)
};

static inline size_t read_off(size_t base_plane, int64_t p, int64_t h, int64_t d, const Reader & r) {
    return base_plane + (size_t)(p * r.row_stride + h * r.head_stride + d);
}

} // namespace

int main() {
    // Test shapes: {ctx, hkv, D, nb}
    struct Shape { int64_t ctx, hkv, D, nb; };
    std::vector<Shape> shapes = {
        {1,   1,  256, 1},   // trivial
        {7,   5,  256, 1},   // odd hkv, odd ctx tail
        {33,  8,  128, 2},   // non-int-multiple ctx, batch 2
        {100, 2,  192, 3},   // odd D (192), nb tail
        {256, 4,  256, 4},   // aligned
        {1000,7,  256, 1},   // larger, odd hkv
        {2,   3,  64,  5},   // tiny rows, odd nb
    };

    int failures = 0;
    int checks = 0;

    for (const Shape & s : shapes) {
        const int64_t ctx = s.ctx, hkv = s.hkv, D = s.D, nb = s.nb;
        const int64_t plane_elems = ctx * hkv * D;         // elements per batch plane (both layouts)
        Reader rPM{hkv * D, D};                              // position-major reader
        Reader rHM{D, ctx * D};                              // head-major reader

        std::vector<int64_t> buf_pm(nb * plane_elems, 0);
        std::vector<int64_t> buf_hm(nb * plane_elems, 0);

        // Writer passes (loop every logical element once).
        for (int64_t b = 0; b < nb; ++b) {
            size_t base = (size_t)(b * plane_elems);
            for (int64_t h = 0; h < hkv; ++h)
                for (int64_t p = 0; p < ctx; ++p)
                    for (int64_t d = 0; d < D; ++d) {
                        int64_t v = logical(b, h, p, d);
                        buf_pm[base + off_pm(p, h, d, hkv, D)] = v;  // position-major write
                        buf_hm[base + off_hm(p, h, d, ctx, D)] = v;  // head-major write
                    }
        }

        // (1) PM-writer + PM-reader reconstructs K
        checks++;
        for (int64_t b = 0; b < nb; ++b) {
            size_t base = (size_t)(b * plane_elems);
            for (int64_t h = 0; h < hkv; ++h)
                for (int64_t p = 0; p < ctx; ++p)
                    for (int64_t d = 0; d < D; ++d) {
                        int64_t got = buf_pm[read_off(base, p, h, d, rPM)];
                        if (got != logical(b, h, p, d)) {
                            std::printf("FAIL[1 PM self] ctx=%lld hkv=%lld D=%lld nb=%lld b=%lld h=%lld p=%lld d=%lld got=%lld want=%lld\n",
                                (long long)ctx,(long long)hkv,(long long)D,(long long)nb,(long long)b,(long long)h,(long long)p,(long long)d,
                                (long long)got,(long long)logical(b,h,p,d));
                            ++failures; goto next_shape;
                        }
                    }
        }

        // (2) HM-writer + HM-reader reconstructs K
        checks++;
        for (int64_t b = 0; b < nb; ++b) {
            size_t base = (size_t)(b * plane_elems);
            for (int64_t h = 0; h < hkv; ++h)
                for (int64_t p = 0; p < ctx; ++p)
                    for (int64_t d = 0; d < D; ++d) {
                        int64_t got = buf_hm[read_off(base, p, h, d, rHM)];
                        if (got != logical(b, h, p, d)) {
                            std::printf("FAIL[2 HM self] ctx=%lld hkv=%lld D=%lld nb=%lld b=%lld h=%lld p=%lld d=%lld got=%lld want=%lld\n",
                                (long long)ctx,(long long)hkv,(long long)D,(long long)nb,(long long)b,(long long)h,(long long)p,(long long)d,
                                (long long)got,(long long)logical(b,h,p,d));
                            ++failures; goto next_shape;
                        }
                    }
        }

        // (3) equivalence: reading PM-enc'd buffer with PM accessor yields, for every
        //     logical coord, the same value as reading HM-enc'd buffer with HM accessor.
        checks++;
        for (int64_t b = 0; b < nb; ++b) {
            size_t base = (size_t)(b * plane_elems);
            for (int64_t h = 0; h < hkv; ++h)
                for (int64_t p = 0; p < ctx; ++p)
                    for (int64_t d = 0; d < D; ++d) {
                        int64_t a = buf_pm[read_off(base, p, h, d, rPM)];
                        int64_t c = buf_hm[read_off(base, p, h, d, rHM)];
                        if (a != c) {
                            std::printf("FAIL[3 equivalence] ctx=%lld hkv=%lld D=%lld nb=%lld b=%lld h=%lld p=%lld d=%lld pm=%lld hm=%lld\n",
                                (long long)ctx,(long long)hkv,(long long)D,(long long)nb,(long long)b,(long long)h,(long long)p,(long long)d,
                                (long long)a,(long long)c);
                            ++failures; goto next_shape;
                        }
                    }
        }

        // (4) sensitivity guard: PM-writer + HM-reader MUST mismatch (the 0.69 scramble).
        //     Verify at least one differing coord exists; if none differ, the test cannot
        //     catch the historical bug and is worthless -> FAIL.
        checks++;
        {
            bool any_mismatch = false;
            for (int64_t b = 0; b < nb && !any_mismatch; ++b) {
                size_t base = (size_t)(b * plane_elems);
                for (int64_t h = 0; h < hkv && !any_mismatch; ++h)
                    for (int64_t p = 0; p < ctx && !any_mismatch; ++p)
                        for (int64_t d = 0; d < D && !any_mismatch; ++d) {
                            int64_t a = buf_pm[read_off(base, p, h, d, rPM)]; // intended
                            int64_t wrong = buf_pm[read_off(base, p, h, d, rHM)]; // mismatched read
                            if (a != wrong) any_mismatch = true;
                        }
            }
            if (!any_mismatch && !(ctx == 1 || (hkv == 1 && ctx == 1))) {
                // shapes with only one position never scramble (p always 0) -> skip those
                std::printf("FAIL[4 sensitivity] ctx=%lld hkv=%lld D=%lld nb=%lld : PM-write+HM-read did NOT scramble (test too weak)\n",
                    (long long)ctx,(long long)hkv,(long long)D,(long long)nb);
                ++failures;
            }
        }

    next_shape:;
    }

    if (failures == 0) {
        std::printf("R1 PASS: %d checks across %zu shapes — position-major & head-major encode ONE logical KV matrix; mismatched read still scrambles (sensitivity confirmed).\n",
                    checks, shapes.size());
        return 0;
    }
    std::printf("R1 FAIL: %d failure(s) in %d checks.\n", failures, checks);
    return 1;
}
