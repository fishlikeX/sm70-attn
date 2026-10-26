// verify/sm70_dequant_parity.cu — bit-exactness gate for the sm_70 dequant kernels
// (ggml/src/ggml-cuda/dequant-sm70.cuh) against the stock convert.cu kernels.
//
// Why this exists: the first version of this check reported "ALL BIT-EXACT" for
// iq4_xs while the kernel was reading kvalues_iq4nl out of bounds. The out-of-range
// load faulted, every later cudaMemcpy failed, the host buffers stayed zero and
// memcmp(0, 0) "passed". Two rules follow and are enforced here:
//   1. every launch and every memcpy is checked (CHK), so a fault fails the gate;
//   2. the output buffers are poisoned first, so a kernel that writes nothing
//      cannot compare equal by accident.
//
// Build (Windows, from this directory):
//   nvcc -O2 -std=c++17 -arch=sm_70 -I .. -I ../ggml/include -I ../ggml/src \
//        -I ../ggml/src/ggml-cuda sm70_dequant_parity.cu -o sm70_dequant_parity.exe
// Run:  sm70_dequant_parity.exe   (exit code 0 = all bit-exact)
//
// The input is random block bytes: dequant is a pure function of the block bytes
// (all defined fields are consumed, padding is never read), so random bytes are an
// exhaustive-enough probe for the per-element mapping.

#include "ggml/src/ggml-cuda/dequant-sm70.cuh"
#include "ggml/src/ggml-cuda/dequantize.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace dequant_sm70;

static int g_fail = 0;

#define CHK(what)                                                              \
    do {                                                                       \
        cudaError_t err__ = cudaGetLastError();                                 \
        if (err__ != cudaSuccess) {                                             \
            printf("FAIL %s: CUDA error: %s\n", what, cudaGetErrorString(err__)); \
            ++g_fail;                                                          \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ---- stock launchers: same shape as convert.cu (one 32-thread block per block) ----
__global__ void stock_iq3_s_h (const void * vx, half  * yy, int64_t i) { dequantize_iq3_s  (vx, i, yy + i*QK_K, threadIdx.x); }
__global__ void stock_iq3_s_f (const void * vx, float * yy, int64_t i) { dequantize_iq3_s  (vx, i, yy + i*QK_K, threadIdx.x); }
__global__ void stock_iq3_xxs_h(const void * vx, half * yy, int64_t i) { dequantize_iq3_xxs(vx, i, yy + i*QK_K, threadIdx.x); }
__global__ void stock_iq4_xs_h (const void * vx, half  * yy, int64_t i) { dequantize_iq4_xs (vx, i, yy + i*QK_K, threadIdx.x); }
__global__ void stock_iq4_xs_f (const void * vx, float * yy, int64_t i) { dequantize_iq4_xs (vx, i, yy + i*QK_K, threadIdx.x); }

template <typename dst_t>
static void compare(const char * name, int64_t nb, const std::vector<dst_t> & ref, const std::vector<dst_t> & test) {
    const size_t n = ref.size();
    size_t bad = 0, first = (size_t) -1;
    for (size_t i = 0; i < n; ++i) {
        if (memcmp(&ref[i], &test[i], sizeof(dst_t)) != 0) {
            if (first == (size_t) -1) {
                first = i;
            }
            ++bad;
        }
    }
    if (bad != 0) {
        printf("FAIL %-8s nb=%-6lld %-5s: %zu / %zu differ, first at %zu (mod 32 = %zu)\n",
               name, (long long) nb, sizeof(dst_t) == 2 ? "half" : "float",
               bad, n, first, first % 32);
        ++g_fail;
    } else {
        printf("ok   %-8s nb=%-6lld %-5s: bit-exact (%zu elems)\n",
               name, (long long) nb, sizeof(dst_t) == 2 ? "half" : "float", n);
    }
}

// random blocks with finite fp16 scales: random d bytes are mostly inf/NaN, and
// inf*LUT collapses every element of a block to the same value, which hides
// index bugs behind equal bit patterns.
template <typename block_t>
static std::vector<uint8_t> make_blocks(int64_t nb, uint32_t seed) {
    std::vector<uint8_t> in(nb * sizeof(block_t));
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto & b : in) {
        b = (uint8_t) byte(rng);
    }
    for (int64_t i = 0; i < nb; ++i) {
        // d at offset 0: 0x3000..0x3BFF is a finite fp16 in [0.5, 1.0)
        uint8_t * blk = in.data() + i * sizeof(block_t);
        blk[0] = 0x00;
        blk[1] = (uint8_t) (0x30 | (blk[1] & 0x0F));
    }
    return in;
}

int main() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        printf("FAIL: no CUDA device\n");
        return 1;
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    printf("device %d: %s (cc %d.%d)\n", dev, prop.name, prop.major, prop.minor);
    if (prop.major != 7 || prop.minor != 0) {
        printf("NOTE: the sm70 kernels are compiled for sm_70; this device is cc %d.%d\n",
               prop.major, prop.minor);
    }

    for (int64_t nb : { 1LL, 3LL, 8LL, 33LL, 1024LL }) {
        std::vector<uint8_t> in = make_blocks<block_iq4_xs>(nb, (uint32_t) (nb * 15485863 + 33));
        const size_t out_elems = (size_t) nb * QK_K;

        // half
        {
            uint8_t * d_in = nullptr; half * d_ref = nullptr; half * d_test = nullptr;
            cudaMalloc(&d_in, in.size());
            cudaMalloc(&d_ref,  out_elems * sizeof(half));
            cudaMalloc(&d_test, out_elems * sizeof(half));
            cudaMemset(d_ref,  0xCD, out_elems * sizeof(half));
            cudaMemset(d_test, 0xCD, out_elems * sizeof(half));
            cudaMemcpy(d_in, in.data(), in.size(), cudaMemcpyHostToDevice);
            CHK("iq4_xs half setup");
            for (int64_t i = 0; i < nb; ++i) {
                stock_iq4_xs_h<<<1, 32>>>(d_in, d_ref, i);
            }
            CHK("iq4_xs half stock");
            dequant_iq4_xs_sm70<half><<<(int) ((nb + 7) / 8), 256>>>(d_in, d_test, nb);
            CHK("iq4_xs half sm70");
            cudaDeviceSynchronize();
            CHK("iq4_xs half sync");
            std::vector<half> ref(out_elems), test(out_elems);
            cudaMemcpy(ref.data(),  d_ref,  out_elems * sizeof(half), cudaMemcpyDeviceToHost);
            cudaMemcpy(test.data(), d_test, out_elems * sizeof(half), cudaMemcpyDeviceToHost);
            CHK("iq4_xs half d2h");
            compare<half>("iq4_xs", nb, ref, test);
            cudaFree(d_in); cudaFree(d_ref); cudaFree(d_test);
        }
        // float
        {
            uint8_t * d_in = nullptr; float * d_ref = nullptr; float * d_test = nullptr;
            cudaMalloc(&d_in, in.size());
            cudaMalloc(&d_ref,  out_elems * sizeof(float));
            cudaMalloc(&d_test, out_elems * sizeof(float));
            cudaMemset(d_ref,  0xCD, out_elems * sizeof(float));
            cudaMemset(d_test, 0xCD, out_elems * sizeof(float));
            cudaMemcpy(d_in, in.data(), in.size(), cudaMemcpyHostToDevice);
            CHK("iq4_xs float setup");
            for (int64_t i = 0; i < nb; ++i) {
                stock_iq4_xs_f<<<1, 32>>>(d_in, d_ref, i);
            }
            CHK("iq4_xs float stock");
            dequant_iq4_xs_sm70<float><<<(int) ((nb + 7) / 8), 256>>>(d_in, d_test, nb);
            CHK("iq4_xs float sm70");
            cudaDeviceSynchronize();
            CHK("iq4_xs float sync");
            std::vector<float> ref(out_elems), test(out_elems);
            cudaMemcpy(ref.data(),  d_ref,  out_elems * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(test.data(), d_test, out_elems * sizeof(float), cudaMemcpyDeviceToHost);
            CHK("iq4_xs float d2h");
            compare<float>("iq4_xs", nb, ref, test);
            cudaFree(d_in); cudaFree(d_ref); cudaFree(d_test);
        }
    }

    // iq3_s / iq3_xxs: same shape, half destination (their production use)
    for (int64_t nb : { 1LL, 8LL, 1024LL }) {
        const size_t out_elems = (size_t) nb * QK_K;

        {
            std::vector<uint8_t> in = make_blocks<block_iq3_s>(nb, (uint32_t) (nb * 7919 + 11));
            uint8_t * d_in = nullptr; half * d_ref = nullptr; half * d_test = nullptr;
            cudaMalloc(&d_in, in.size());
            cudaMalloc(&d_ref,  out_elems * sizeof(half));
            cudaMalloc(&d_test, out_elems * sizeof(half));
            cudaMemset(d_ref,  0xCD, out_elems * sizeof(half));
            cudaMemset(d_test, 0xCD, out_elems * sizeof(half));
            cudaMemcpy(d_in, in.data(), in.size(), cudaMemcpyHostToDevice);
            CHK("iq3_s setup");
            for (int64_t i = 0; i < nb; ++i) {
                stock_iq3_s_h<<<1, 32>>>(d_in, d_ref, i);
            }
            CHK("iq3_s stock");
            dequant_iq3_s_sm70<half><<<(int) ((nb + 7) / 8), 256>>>(d_in, d_test, nb);
            CHK("iq3_s sm70");
            cudaDeviceSynchronize();
            CHK("iq3_s sync");
            std::vector<half> ref(out_elems), test(out_elems);
            cudaMemcpy(ref.data(),  d_ref,  out_elems * sizeof(half), cudaMemcpyDeviceToHost);
            cudaMemcpy(test.data(), d_test, out_elems * sizeof(half), cudaMemcpyDeviceToHost);
            CHK("iq3_s d2h");
            compare<half>("iq3_s", nb, ref, test);
            cudaFree(d_in); cudaFree(d_ref); cudaFree(d_test);
        }
        {
            std::vector<uint8_t> in = make_blocks<block_iq3_xxs>(nb, (uint32_t) (nb * 104729 + 22));
            uint8_t * d_in = nullptr; half * d_ref = nullptr; half * d_test = nullptr;
            cudaMalloc(&d_in, in.size());
            cudaMalloc(&d_ref,  out_elems * sizeof(half));
            cudaMalloc(&d_test, out_elems * sizeof(half));
            cudaMemset(d_ref,  0xCD, out_elems * sizeof(half));
            cudaMemset(d_test, 0xCD, out_elems * sizeof(half));
            cudaMemcpy(d_in, in.data(), in.size(), cudaMemcpyHostToDevice);
            CHK("iq3_xxs setup");
            for (int64_t i = 0; i < nb; ++i) {
                stock_iq3_xxs_h<<<1, 32>>>(d_in, d_ref, i);
            }
            CHK("iq3_xxs stock");
            dequant_iq3_xxs_sm70<half><<<(int) ((nb + 7) / 8), 256>>>(d_in, d_test, nb);
            CHK("iq3_xxs sm70");
            cudaDeviceSynchronize();
            CHK("iq3_xxs sync");
            std::vector<half> ref(out_elems), test(out_elems);
            cudaMemcpy(ref.data(),  d_ref,  out_elems * sizeof(half), cudaMemcpyDeviceToHost);
            cudaMemcpy(test.data(), d_test, out_elems * sizeof(half), cudaMemcpyDeviceToHost);
            CHK("iq3_xxs d2h");
            compare<half>("iq3_xxs", nb, ref, test);
            cudaFree(d_in); cudaFree(d_ref); cudaFree(d_test);
        }
    }

    printf(g_fail == 0 ? "ALL BIT-EXACT\n" : "%d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
