// standalone correctness check for sm70_d256_decode_kernel + merge
#include "../ggml/src/ggml-cuda/fattn-sm70-d256-decode.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

using namespace sm70_d256_decode;

static void run_case(int hkv, int gqa, int q_len, int kv_len, int splits, unsigned seed, bool dbg) {
    const int heads_q = hkv * gqa;
    const int D = 256;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);

    std::vector<float> Q((size_t) D * q_len * heads_q);
    std::vector<half> K((size_t) kv_len * hkv * D), V((size_t) kv_len * hkv * D);
    for (auto & x : Q) x = nd(rng);
    for (auto & x : K) x = __float2half(nd(rng));
    for (auto & x : V) x = __float2half(nd(rng));
    const float scale = 1.0f / sqrtf((float) D);

    float * dQ; half * dK, * dV;
    cudaMalloc(&dQ, Q.size() * 4); cudaMalloc(&dK, K.size() * 2); cudaMalloc(&dV, V.size() * 2);
    cudaMemcpy(dQ, Q.data(), Q.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K.data(), K.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V.data(), V.size() * 2, cudaMemcpyHostToDevice);

    const int rows_total = heads_q * q_len;
    float * pa, * pm, * pl, * dOut;
    cudaMalloc(&pa, (size_t) splits * rows_total * D * 4);
    cudaMalloc(&pm, (size_t) splits * rows_total * 4);
    cudaMalloc(&pl, (size_t) splits * rows_total * 4);
    cudaMalloc(&dOut, (size_t) D * q_len * heads_q * 4);
    std::vector<float> out((size_t) D * q_len * heads_q);

    // real decode semantics: capacity-sized mask, live boundary via -inf
    const int cap = kv_len + 512;
    std::vector<half> M((size_t) cap * q_len);
    for (int q = 0; q < q_len; ++q)
        for (int k = 0; k < cap; ++k)
            M[(size_t) q * cap + k] = __float2half(k <= kv_len - q_len + q ? 0.0f : -INFINITY);
    half * dM; cudaMalloc(&dM, M.size() * 2);
    cudaMemcpy(dM, M.data(), M.size() * 2, cudaMemcpyHostToDevice);

    dim3 grid(hkv, splits);
    if (gqa == 6) {
        sm70_d256_decode_kernel<6><<<grid, 128>>>(dQ, (int64_t) D, (int64_t) D * q_len,
            dK, dV, (int64_t) hkv * D, D, (int64_t) hkv * D, D,
            q_len, cap, scale, splits, dM + (int64_t)(q_len - 1) * cap, 1, pa, pm, pl);
    } else {
        sm70_d256_decode_kernel<8><<<grid, 128>>>(dQ, (int64_t) D, (int64_t) D * q_len,
            dK, dV, (int64_t) hkv * D, D, (int64_t) hkv * D, D,
            q_len, cap, scale, splits, dM + (int64_t)(q_len - 1) * cap, 1, pa, pm, pl);
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("KERNEL LAUNCH FAIL: %s\n", cudaGetErrorString(e)); exit(1); }

    // debug: dump partials BEFORE merge for splits==1/q_len==1 and compare m/l/acc[0]
    if (dbg && splits == 1 && q_len == 1) {
        std::vector<float> pm_(rows_total), pl_(rows_total), pa_((size_t) rows_total * D);
        cudaMemcpy(pm_.data(), pm, rows_total * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(pl_.data(), pl, rows_total * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(pa_.data(), pa, pa_.size() * 4, cudaMemcpyDeviceToHost);
        for (int hq = 0; hq < heads_q && hq < 3; ++hq) {
            const int kh = hq / gqa;
            float mm = -1e30f;
            std::vector<float> s(kv_len);
            for (int k = 0; k < kv_len; ++k) {
                float dot = 0;
                for (int d = 0; d < D; ++d)
                    dot += Q[d + (size_t) hq * q_len * D] * __half2float(K[(size_t) k * hkv * D + kh * D + d]);
                s[k] = dot * scale; mm = fmaxf(mm, s[k]);
            }
            float ll = 0;
            for (int k = 0; k < kv_len; ++k) { s[k] = exp2f((s[k] - mm) * 1.4426950408889634074f); ll += s[k]; }
            float a0 = 0;
            for (int k = 0; k < kv_len; ++k) a0 += s[k] * __half2float(V[(size_t) k * hkv * D + kh * D + 0]);
            printf("  dbg row=%d m gpu=%.5f cpu=%.5f | l gpu=%.5f cpu=%.5f | acc[0] gpu=%.5f cpu=%.5f\n",
                   hq, pm_[hq], mm, pl_[hq], ll, pa_[(size_t) hq * D], a0);
        }
    }

    sm70_d256_decode_merge_kernel<<<rows_total, 128>>>(pa, pm, pl, dOut, rows_total, splits,
        (int64_t) D, (int64_t) D * q_len, q_len);
    e = cudaGetLastError();
    if (e != cudaSuccess) { printf("MERGE LAUNCH FAIL: %s\n", cudaGetErrorString(e)); exit(1); }
    cudaMemcpy(out.data(), dOut, out.size() * 4, cudaMemcpyDeviceToHost);

    double max_err = 0; int bad = 0;
    for (int hq = 0; hq < heads_q; ++hq) {
        const int kh = hq / gqa;
        for (int t = 0; t < q_len; ++t) {
            const int qabs = kv_len - q_len + t;
            std::vector<float> acc(D, 0.f); float m = -1e30f;
            std::vector<float> s(std::min(qabs + 1, kv_len));
            for (int k = 0; k <= qabs && k < kv_len; ++k) {
                float dot = 0;
                for (int d = 0; d < D; ++d) {
                    dot += Q[d + (size_t) t * D + (size_t) hq * q_len * D]
                         * __half2float(K[(size_t) k * hkv * D + kh * D + d]);
                }
                s[k] = dot * scale; m = fmaxf(m, s[k]);
            }
            float l = 0;
            for (int k = 0; k <= qabs && k < kv_len; ++k) { s[k] = exp2f((s[k] - m) * 1.4426950408889634074f); l += s[k]; }
            for (int k = 0; k <= qabs && k < kv_len; ++k) {
                for (int d = 0; d < D; ++d) {
                    acc[d] += s[k] * __half2float(V[(size_t) k * hkv * D + kh * D + d]);
                }
            }
            for (int d = 0; d < D; ++d) {
                float ref = acc[d] / l;
                float got = out[d + (size_t) t * D + (size_t) hq * q_len * D];
                double err = fabs((double) ref - got);
                if (err > max_err) max_err = err;
                if (err > 1e-2) ++bad;
            }
        }
    }
    printf("case hkv=%d gqa=%d w=%d kv=%d splits=%d | max|err|=%.3e bad(>1e-2)=%d\n",
           hkv, gqa, q_len, kv_len, splits, max_err, bad);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(pa); cudaFree(pm); cudaFree(pl); cudaFree(dOut);
}

// q4_0 quantize one 32-dim group (ggml-style: d=maxabs/8, q=clamp(round(x/d)+8)).
static void q4_quant8(const half * src32, uint8_t * blk) {
    float mx = 0.f;
    for (int i = 0; i < 32; ++i) mx = fmaxf(mx, fabsf(__half2float(src32[i])));
    const float d = mx / 8.f;
    half * sc = reinterpret_cast<half *>(blk);
    *sc = __float2half_rn(d);
    for (int i = 0; i < 16; ++i) blk[2 + i] = 0;
    if (d == 0.f) return;
    for (int i = 0; i < 32; ++i) {
        int q = (int) lrintf(__half2float(src32[i]) / d) + 8;
        q = q < 0 ? 0 : (q > 15 ? 15 : q);
        // canonical ggml layout: byte b = elem b (low) | elem b+16 (high)
        if (i < 16) blk[2 + i] |= (uint8_t)q;
        else        blk[2 + i - 16] |= (uint8_t)(q << 4);
    }
}

static void run_case_q4(int hkv, int gqa, int q_len, int kv_len, int splits, unsigned seed, bool kq4, bool vq4) {
    const int heads_q = hkv * gqa;
    const int D = 256;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> Q((size_t) D * q_len * heads_q);
    std::vector<half> K((size_t) kv_len * hkv * D), V((size_t) kv_len * hkv * D);
    for (auto & x : Q) x = nd(rng);
    for (auto & x : K) x = __float2half(nd(rng));
    for (auto & x : V) x = __float2half(nd(rng));
    const int nblk = 8, blksz = 18;
    std::vector<uint8_t> Kq((size_t) kv_len * hkv * nblk * blksz), Vq(Kq.size());
    std::vector<half> Kd(K.size()), Vd(V.size());
    for (int k = 0; k < kv_len; ++k)
        for (int h = 0; h < hkv; ++h)
            for (int b = 0; b < nblk; ++b) {
                const size_t off = ((size_t) k * hkv + h) * nblk * blksz + (size_t) b * blksz;
                q4_quant8(&K[((size_t) k * hkv + h) * D + b * 32], &Kq[off]);
                q4_quant8(&V[((size_t) k * hkv + h) * D + b * 32], &Vq[off]);
                const float s  = __half2float(*reinterpret_cast<half *>(&Kq[off]));
                const float sv = __half2float(*reinterpret_cast<half *>(&Vq[off]));
                for (int j = 0; j < 32; ++j) {
                    const int q  = j < 16 ? (Kq[off + 2 + j] & 0xF) : (Kq[off + 2 + j - 16] >> 4);
                    const int qv = j < 16 ? (Vq[off + 2 + j] & 0xF) : (Vq[off + 2 + j - 16] >> 4);
                    Kd[((size_t) k * hkv + h) * D + b * 32 + j] = __float2half_rn(((float) q - 8.f) * s);
                    Vd[((size_t) k * hkv + h) * D + b * 32 + j] = __float2half_rn(((float) qv - 8.f) * sv);
                }
            }
    const float scale = 1.0f / sqrtf((float) D);
    const int cap = kv_len + 512;
    std::vector<half> M((size_t) cap * q_len);
    for (int q = 0; q < q_len; ++q)
        for (int k = 0; k < cap; ++k)
            M[(size_t) q * cap + k] = __float2half(k <= kv_len - q_len + q ? 0.0f : -INFINITY);

    float * dQ; half * dM; uint8_t * dKq, * dVq; half * dK, * dV;
    cudaMalloc(&dQ, Q.size() * 4); cudaMalloc(&dKq, Kq.size()); cudaMalloc(&dVq, Vq.size());
    cudaMalloc(&dK, Kd.size() * 2); cudaMalloc(&dV, Vd.size() * 2); cudaMalloc(&dM, M.size() * 2);
    cudaMemcpy(dQ, Q.data(), Q.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dKq, Kq.data(), Kq.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dVq, Vq.data(), Vq.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, Kd.data(), Kd.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, Vd.data(), Vd.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dM, M.data(), M.size() * 2, cudaMemcpyHostToDevice);

    const int rows_total = heads_q * q_len;
    float * pa, * pm, * pl, * dOut;
    cudaMalloc(&pa, (size_t) splits * rows_total * 256 * 4);
    cudaMalloc(&pm, (size_t) splits * rows_total * 4);
    cudaMalloc(&pl, (size_t) splits * rows_total * 4);
    cudaMalloc(&dOut, (size_t) D * q_len * heads_q * 4);
    std::vector<float> out((size_t) D * q_len * heads_q);

    const half * kp = kq4 ? (const half *) dKq : dK;
    const half * vp = vq4 ? (const half *) dVq : dV;
    const int64_t krs = (int64_t) hkv * 144, khs = 144;     // bytes when q4
    const int64_t krf = (int64_t) hkv * D, khf = D;         // f16 elems
    dim3 grid(hkv, splits);
    if (gqa == 6) {
        if (kq4 && vq4) sm70_d256_decode::sm70_d256_decode_kernel<6, true, true><<<grid, 128>>>(
            dQ, (int64_t) D, (int64_t) D * q_len, kp, vp, krs, khs, krs, khs,
            q_len, cap, scale, splits, dM + (int64_t)(q_len - 1) * cap, 1, pa, pm, pl);
        else if (kq4) sm70_d256_decode::sm70_d256_decode_kernel<6, true, false><<<grid, 128>>>(
            dQ, (int64_t) D, (int64_t) D * q_len, kp, vp, krs, khs, krf, khf,
            q_len, cap, scale, splits, dM + (int64_t)(q_len - 1) * cap, 1, pa, pm, pl);
        else if (vq4) sm70_d256_decode::sm70_d256_decode_kernel<6, false, true><<<grid, 128>>>(
            dQ, (int64_t) D, (int64_t) D * q_len, kp, vp, krf, khf, krs, khs,
            q_len, cap, scale, splits, dM + (int64_t)(q_len - 1) * cap, 1, pa, pm, pl);
        else sm70_d256_decode::sm70_d256_decode_kernel<6, false, false><<<grid, 128>>>(
            dQ, (int64_t) D, (int64_t) D * q_len, kp, vp, krf, khf, krf, khf,
            q_len, cap, scale, splits, dM + (int64_t)(q_len - 1) * cap, 1, pa, pm, pl);
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("Q4 KERNEL FAIL: %s\n", cudaGetErrorString(e)); exit(1); }
    sm70_d256_decode::sm70_d256_decode_merge_kernel<<<rows_total, 128>>>(
        pa, pm, pl, dOut, rows_total, splits, (int64_t) D, (int64_t) D * q_len, q_len);
    cudaMemcpy(out.data(), dOut, out.size() * 4, cudaMemcpyDeviceToHost);

    double max_err = 0; int bad = 0;
    for (int hq = 0; hq < heads_q; ++hq) {
        const int kh = hq / gqa;
        for (int tq = 0; tq < q_len; ++tq) {
            const int qabs = kv_len - q_len + tq;
            std::vector<float> acc(D, 0.f); float m = -1e30f;
            std::vector<float> s(qabs + 1);
            for (int k = 0; k <= qabs; ++k) {
                float dot = 0;
                for (int d = 0; d < D; ++d)
                    dot += Q[d + (size_t) tq * D + (size_t) hq * q_len * D]
                         * __half2float(Kd[(size_t) k * hkv * D + kh * D + d]);
                s[k] = dot * scale; m = fmaxf(m, s[k]);
            }
            float l = 0;
            for (int k = 0; k <= qabs; ++k) { s[k] = exp2f((s[k] - m) * 1.4426950408889634074f); l += s[k]; }
            for (int k = 0; k <= qabs; ++k)
                for (int d = 0; d < D; ++d)
                    acc[d] += s[k] * __half2float(Vd[(size_t) k * hkv * D + kh * D + d]);
            for (int d = 0; d < D; ++d) {
                float ref = acc[d] / l;
                float got = out[d + (size_t) tq * D + (size_t) hq * q_len * D];
                double err = fabs((double) ref - got);
                if (err > max_err) max_err = err;
                if (err > 1e-2) ++bad;
            }
        }
    }
    printf("q4case hkv=%d gqa=%d w=%d kv=%d splits=%d Kq4=%d Vq4=%d | max|err|=%.3e bad=%d\n",
           hkv, gqa, q_len, kv_len, splits, (int) kq4, (int) vq4, max_err, bad);
    if (bad) {
        for (int hq = 0; hq < heads_q; ++hq)
            for (int tq = 0; tq < q_len; ++tq) {
                double me = 0; int nb2 = 0;
                const int kh2 = hq / gqa;
                const int qabs = kv_len - q_len + tq;
                std::vector<float> acc2(D, 0.f); float m2 = -1e30f;
                std::vector<float> s2(qabs + 1);
                for (int k = 0; k <= qabs; ++k) {
                    float dot = 0;
                    for (int d = 0; d < D; ++d)
                        dot += Q[d + (size_t) tq * D + (size_t) hq * q_len * D]
                             * __half2float(Kd[(size_t) k * hkv * D + kh2 * D + d]);
                    s2[k] = dot * scale; m2 = fmaxf(m2, s2[k]);
                }
                float l2 = 0;
                for (int k = 0; k <= qabs; ++k) { s2[k] = exp2f((s2[k] - m2) * 1.4426950408889634074f); l2 += s2[k]; }
                for (int k = 0; k <= qabs; ++k)
                    for (int d = 0; d < D; ++d)
                        acc2[d] += s2[k] * __half2float(Vd[(size_t) k * hkv * D + kh2 * D + d]);
                for (int d = 0; d < D; ++d) {
                    double err = fabs((double) acc2[d] / l2 - out[d + (size_t) tq * D + (size_t) hq * q_len * D]);
                    if (err > me) me = err;
                    if (err > 1e-2) ++nb2;
                }
                if (nb2) printf("  row hq=%d tq=%d max=%.3e bad=%d\n", hq, tq, me, nb2);
            }
    }
    cudaFree(dQ); cudaFree(dKq); cudaFree(dVq); cudaFree(dK); cudaFree(dV); cudaFree(dM);
    cudaFree(pa); cudaFree(pm); cudaFree(pl); cudaFree(dOut);
}

int main() {
    run_case_q4(4, 6, 1, 3000, 6, 21, true, false);
    run_case_q4(4, 6, 1, 3000, 6, 22, true, true);
    run_case_q4(4, 6, 2, 279, 2, 23, true, false);
    run_case_q4(4, 6, 1, 4096, 24, 24, true, false);
    run_case(4, 6, 1, 279, 1, 1, true);
    run_case(4, 6, 2, 2, 1, 11, false);     // server warmup shape: live==q_len
    run_case(4, 6, 1, 8, 24, 12, false);    // tiny live, many empty splits
    run_case(4, 6, 1, 279, 3, 2, false);
    run_case(4, 6, 1, 3000, 6, 3, false);
    run_case(4, 6, 2, 279, 2, 4, false);
    run_case(4, 6, 8, 1000, 2, 5, false);
    run_case(4, 6, 1, 4096, 24, 6, false);
    run_case(8, 8, 1, 1000, 4, 7, false);
    printf("DONE\n");
    return 0;
}
