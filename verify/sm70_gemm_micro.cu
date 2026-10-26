// sm70_gemm_micro: how much of the prefill FFN GEMM time is the per-call F16
// expansion, and what would a fused dequant GEMM have to beat?
//
// Path A (today): dequant the quantized weight into an F16 scratch, then
//                 cublasGemmEx -- exactly what ggml_cuda_mul_mat_cublas_impl does
//                 for a quantized src0 (traits::convert) and what
//                 ggml_cuda_mul_mat_iq_skinny does after its expansion.
// Path C (ceiling): cublasGemmEx straight on an F16 weight.
// A - C is the recoverable cost of the expansion; a fused kernel has to land
// below A and ideally near C.
//
// Shapes are the production per-device TP2 FFN ones (ub 8192 / 4096).
#include "ggml/src/ggml-cuda/dequant-sm70.cuh"
#include "ggml/src/ggml-cuda/dequantize.cuh"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    printf("CUBLAS %d @%d\n", (int) s_, __LINE__); return 1; } } while (0)

using namespace dequant_sm70;

static cublasHandle_t g_h;

static float gemm_f16(int M, int N, int K, const half * A, const half * B, float * C) {
    const float alpha = 1.0f, beta = 0.0f;
    // A = weight [N][K] row-major -> cuBLAS sees it as KxN col-major, so OP_T/N
    // with lda = K: identical to ggml_cuda_mul_mat_cublas_impl
    auto run = [&]() {
        return cublasGemmEx(g_h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                &alpha, A, CUDA_R_16F, K,
                        B, CUDA_R_16F, K,
                &beta,  C, CUDA_R_32F, N,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    };
    CB(run()); CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    for (int i = 0; i < 10; ++i) { if (run() != CUBLAS_STATUS_SUCCESS) { return -1.f; } }
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
    return ms / 10.0f;
}

template <typename KERNEL>
static float timed_kernel(KERNEL launch) {
    CK(launch()); CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    for (int i = 0; i < 10; ++i) { CK(launch()); }
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
    return ms / 10.0f;
}

// ---- one row of the table: a quantized weight of shape [K, N] --------------
template <typename BLOCK>
static int row(const char * name, int M, int N, int K) {
    const int64_t nb = (int64_t) (K / QK_K) * N;          // super-blocks = (K/256)*N
    const size_t wbytes = (size_t) nb * sizeof(BLOCK);

    std::vector<uint8_t> host(wbytes);
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto & b : host) { b = (uint8_t) byte(rng); }
    // finite fp16 scales so the dequantized values are sane
    for (int64_t i = 0; i < nb; ++i) {
        uint8_t * blk = host.data() + i * sizeof(BLOCK);
        blk[0] = 0x00;
        blk[1] = (uint8_t) (0x30 | (blk[1] & 0x0F));
    }

    uint8_t * dW = nullptr; half * dW16 = nullptr; half * dB = nullptr; float * dC = nullptr;
    CK(cudaMalloc(&dW, wbytes));
    CK(cudaMalloc(&dW16, (size_t) N*K*2));
    CK(cudaMalloc(&dB, (size_t) K*M*2));
    CK(cudaMalloc(&dC, (size_t) M*N*4));
    CK(cudaMemcpy(dW, host.data(), wbytes, cudaMemcpyHostToDevice));
    CK(cudaMemset(dB, 0x22, (size_t) K*M*2));

    // (C) ceiling: an F16 weight, GEMM only
    auto dequant_all = [&]() {
        if constexpr (std::is_same_v<BLOCK, block_iq3_xxs>) {
            dequant_iq3_xxs_sm70<half><<<(int) ((nb + 7) / 8), 256>>>(dW, dW16, nb);
        } else if constexpr (std::is_same_v<BLOCK, block_iq3_s>) {
            dequant_iq3_s_sm70<half><<<(int) ((nb + 7) / 8), 256>>>(dW, dW16, nb);
        } else {
            dequant_iq4_xs_sm70<half><<<(int) ((nb + 7) / 8), 256>>>(dW, dW16, nb);
        }
        return cudaGetLastError();
    };
    const float ms_deq = timed_kernel(dequant_all);
    const float ms_ceil = gemm_f16(M, N, K, dW16, dB, dC);
    // (A) today: expansion every call, then the same GEMM
    const float ms_today = ms_deq + ms_ceil;

    const double flops = 2.0 * M * N * K;
    printf("%-9s M=%-5d N=%-5d K=%-5d | today %7.3f ms (%5.1f%% = deq %6.3f + gemm %6.3f) | ceiling %7.3f ms (%5.1f%%) | deq=%.0f GB/s\n",
           name, M, N, K, ms_today, 100.0 * flops / (ms_today / 1e3) / 1e12 / 125.0,
           ms_deq, ms_ceil, ms_ceil, 100.0 * flops / (ms_ceil / 1e3) / 1e12 / 125.0,
           (double) (N*K*2 + wbytes) / (ms_deq / 1e3) / 1e9);

    cudaFree(dW); cudaFree(dW16); cudaFree(dB); cudaFree(dC);
    return 0;
}

int main() {
    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceProp p{}; cudaGetDeviceProperties(&p, dev);
    printf("%s cc%d.%d\n", p.name, p.major, p.minor);
    CB(cublasCreate(&g_h));
    printf("real(GS)    = 125.0 TFLOP/s fp16, 900 GB/s HBM\n");

    // production per-device FFN shapes (TP2 row split, so n = 17408/2)
    row<block_iq3_xxs>("iq3_xxs", 4096, 8704, 5120);
    row<block_iq3_xxs>("iq3_xxs", 8192, 8704, 5120);
    row<block_iq3_s>  ("iq3_s",   4096, 8704, 5120);
    row<block_iq4_xs> ("iq4_xs",  4096, 8704, 5120);
    row<block_iq3_xxs>("iq3_xxs", 4096, 5120, 8704);   // ffn_down (k split)
    return 0;
}
