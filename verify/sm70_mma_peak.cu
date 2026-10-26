// SPDX-License-Identifier: BSD-3-Clause
// sm70_mma_peak.cu — what is the real HMMA.884 issue rate on V100?
//
// The D256 attention kernel runs 512 SASS HMMA.884 per warp per 32-key tile.
// If the SM retires 2/cycle the kernel sits at 39% of the 125 TFLOP/s spec
// number and there is headroom; if it retires 1/cycle the spec number does not
// apply to m8n8k4 and the kernel is already near the hardware limit. This
// probe answers that with nothing but back-to-back MMAs (no smem, no loads).
#include <cstdio>
#include <cuda_runtime.h>

#define MACS 256ull   // m8n8k4

__global__ void mma_f32acc(float * out, int iters, int nacc) {
    // exactly the atom the kernel uses (cute SM70_8x8x4_F32F16F16F32_TN):
    // 8 f32 accumulators per thread, one call == 4 SASS HMMA.884.STEPx
    float c[8][8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { c[j][i] = 0.0f; }
    }
    const uint32_t a0 = threadIdx.x, a1 = threadIdx.x + 1;
    const uint32_t b0 = threadIdx.x + 2, b1 = threadIdx.x + 3;
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j < nacc) {
                asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                    "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
                    : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3]),
                      "+f"(c[j][4]), "+f"(c[j][5]), "+f"(c[j][6]), "+f"(c[j][7])
                    : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
            }
        }
    }
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { s += c[j][i]; }
    }
    if (s == 1234.5678f) { out[threadIdx.x] = s; }
}

template <typename K>
static double run(K kfn, int blocks, int threads, int iters, int nacc, float * d_out) {
    kfn<<<blocks, threads>>>(d_out, iters, nacc);
    cudaDeviceSynchronize();
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    kfn<<<blocks, threads>>>(d_out, iters, nacc);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    const double warps = (double) blocks * (threads / 32.0);
    const double flops = warps * (double) nacc * (double) iters * 4.0 * MACS * 2.0;
    return flops / (ms * 1e-3) / 1e12;
}

int main() {
    int dev = 1;
    cudaSetDevice(dev);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    printf("[dev] %s  SMs=%d  clock=%d kHz\n", prop.name, prop.multiProcessorCount, prop.clockRate);
    float * d_out = nullptr;
    cudaMalloc(&d_out, 4096);
    const int blocks = prop.multiProcessorCount;
    const int iters = 200000;
    printf("-- f32 accumulate, m8n8k4 ----------------------------------\n");
    for (int nacc : {1, 2, 4, 8}) {
        for (int threads : {256, 512, 1024}) {
            const double tf = run(mma_f32acc, blocks, threads, iters, nacc, d_out);
            printf("  1 CTA/SM warps/CTA=%2d acc=%d : %7.2f TFLOP/s  (%5.1f%% of 125)\n",
                   threads / 32, nacc, tf, 100.0 * tf / 125.0);
        }
    }
    for (int nacc : {4, 8}) {
        const double tf = run(mma_f32acc, blocks * 2, 256, iters, nacc, d_out);
        printf("  2 CTA/SM warps/CTA= 8 acc=%d : %7.2f TFLOP/s  (%5.1f%% of 125)\n",
               nacc, tf, 100.0 * tf / 125.0);
    }
    return 0;
}
