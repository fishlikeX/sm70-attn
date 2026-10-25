// Which reduce width makes the GDN Prologue l2norm match stock build_gdn_l2_norm?
// The kernel: each lane sums 16 squares (q[i], i = r*8 + lane_in_col), then
// warp_reduce_sum<warp_size=32>. 4 column segments load the SAME q values
// (lane_in_col = lane - col_in_warp*8), so the 32-wide sum = 4x true ss.
// Candidate fix: reduce over lanes_per_col=8 only.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__device__ __forceinline__ float seg_reduce(float x, int width) {
#pragma unroll
    for (int off = 4; off >= 1; off >>= 1) {
        if (width > off) x += __shfl_xor_sync(0xffffffff, x, off, 32);
    }
    return x;
}

__global__ void fused_variants(const float * q, float * out32, float * out8,
                               float * ss32_out, float * ss8_out, float eps, float scale) {
    const int lane = threadIdx.x;          // 0..31 (one warp = one head row)
    const int col_in_warp = lane / 8;
    const int lane_in_col = lane - col_in_warp * 8;
    const float * q_t = q + blockIdx.x * 128;

    float q_reg[16];
    float x = 0.0f;
#pragma unroll
    for (int r = 0; r < 16; r++) {
        q_reg[r] = q_t[r * 8 + lane_in_col];
    }
#pragma unroll
    for (int r = 0; r < 16; r++) {
        x += q_reg[r] * q_reg[r];
    }

    // variant A: current kernel — full warp reduce
    float x32 = x;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        x32 += __shfl_xor_sync(0xffffffff, x32, off, 32);
    }
    // variant B: segment reduce (8 lanes)
    float x8 = seg_reduce(x, 8);

    const float n32 = rsqrtf(x32 / 128.0f + eps) * scale;
    const float n8  = rsqrtf(x8  / 128.0f + eps) * scale;
#pragma unroll
    for (int r = 0; r < 16; r++) {
        const int i = r * 8 + lane_in_col;
        if (col_in_warp == 0) { // column segment 0 covers the whole head
            out32[blockIdx.x * 128 + i] = q_reg[r] * n32;
            out8[blockIdx.x * 128 + i]  = q_reg[r] * n8;
        }
    }
    if (lane == 0) { ss32_out[blockIdx.x] = x32; ss8_out[blockIdx.x] = x8; }
}

int main() {
    const int NR = 64;
    const size_t N = (size_t) NR * 128;
    float * h = (float *) malloc(N * 4);
    srand(7);
    for (size_t i = 0; i < N; i++) h[i] = (rand() / (float) RAND_MAX - 0.5f) * 6.0f;

    float *dq, *d32, *d8, *ds32, *ds8;
    cudaMalloc(&dq, N * 4); cudaMalloc(&d32, N * 4); cudaMalloc(&d8, N * 4);
    cudaMalloc(&ds32, NR * 4); cudaMalloc(&ds8, NR * 4);
    cudaMemcpy(dq, h, N * 4, cudaMemcpyHostToDevice);

    const float eps = 1e-6f / 128.0f;      // build_gdn_l2_norm: eps/n passed to rms_norm
    const float scale = 1.0f / sqrtf(128.0f);
    fused_variants<<<NR, 32>>>(dq, d32, d8, ds32, ds8, eps, scale);

    float *o32 = (float *) malloc(N * 4), *o8 = (float *) malloc(N * 4);
    float s32[64], s8[64];
    cudaMemcpy(o32, d32, N * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(o8, d8, N * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(s32, ds32, NR * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(s8, ds8, NR * 4, cudaMemcpyDeviceToHost);

    // CPU stock reference: fp32 sequential sumsq per head (rms_norm order differs
    // slightly from any kernel but ~1 ulp; we test structural factor, not ulps)
    double max_rel32 = 0, max_rel8 = 0;
    for (int r = 0; r < NR; r++) {
        float ss = 0;
        for (int c = 0; c < 128; c++) { float v = h[r * 128 + c]; ss += v * v; }
        const float n = rsqrtf(ss / 128.0f + eps) * scale;
        for (int c = 0; c < 128; c++) {
            const float ref = h[r * 128 + c] * n;
            if (ref == 0) continue;
            double d32r = fabs(o32[r * 128 + c] - ref) / fabs(ref);
            double d8r  = fabs(o8[r * 128 + c]  - ref) / fabs(ref);
            if (d32r > max_rel32) max_rel32 = d32r;
            if (d8r > max_rel8) max_rel8 = d8r;
        }
        if (r == 0) {
            double true_ss = 0; for (int c = 0; c < 128; c++) { double v = h[c]; true_ss += v * v; }
            printf("row0 ss: cpu %.4f  warp32 %.4f  seg8 %.4f\n", true_ss, s32[0], s8[0]);
        }
    }
    printf("variant A (warp32, current kernel): max rel vs stock = %.3e\n", max_rel32);
    printf("variant B (seg8,    proposed fix ): max rel vs stock = %.3e\n", max_rel8);
    return 0;
}
