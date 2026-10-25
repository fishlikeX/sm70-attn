// Bit-exactness test for the GDN prologue l2norm, round 2.
// Claim: rms_norm_f32 for ncols=128 uses RMS_NORM_LAUNCH(32,1,true): 32 threads,
// each summing one float4 in the order x^2+y^2+z^2+w^2, then warp_reduce_sum<32>
// (offsets 16,8,4,2,1). A GDN warp that loads the same 128 floats as one float4
// per lane and runs the same butterfly gets a BITWISE identical ss -> identical
// rsqrtf -> identical output (multiply order (x*scale)*s on both sides).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp32_sum(float x) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) x += __shfl_xor_sync(0xffffffff, x, off, 32);
    return x;
}

// mock of rms_norm_f32 for ncols=128 (one warp per row, float4 per lane)
__global__ void mock_rms(const float * x, float * dst, float eps) {
    const float4 v = ((const float4 *) (x + blockIdx.x * 128))[threadIdx.x];
    float tmp = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    tmp = warp32_sum(tmp);
    const float scale = rsqrtf(tmp / 128.0f + eps);
    if (threadIdx.x == 0) printf("  mock_rms   ss bits %08x\n", __float_as_uint(tmp));
    const float4 o = make_float4(v.x * scale, v.y * scale, v.z * scale, v.w * scale);
    ((float4 *) (dst + blockIdx.x * 128))[threadIdx.x] = o;
}

// mock of the ggml SCALE op (y = x*s, elementwise)
__global__ void mock_scale(const float * x, float * dst, float s) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    dst[i] = x[i] * s;
}

// GDN-prologue style: same float4 ss, then multiply in register order (x*norm)*s
__global__ void prologue_like(const float * q, float * dst, float eps, float s) {
    const float4 v = ((const float4 *) (q + blockIdx.x * 128))[threadIdx.x];
    float tmp = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    tmp = warp32_sum(tmp);
    const float norm = rsqrtf(tmp / 128.0f + eps);
    if (threadIdx.x == 0) printf("  prologue   ss bits %08x\n", __float_as_uint(tmp));
    const float4 o = make_float4(v.x * norm * s, v.y * norm * s, v.z * norm * s, v.w * norm * s);
    ((float4 *) (dst + blockIdx.x * 128))[threadIdx.x] = o;
}

int main() {
    const int NR = 128;
    const size_t N = (size_t) NR * 128;
    float * h = (float *) malloc(N * 4);
    srand(11);
    for (size_t i = 0; i < N; i++) h[i] = (rand() / (float) RAND_MAX - 0.5f) * 8.0f;

    float *dq, *d1, *d2, *d3;
    cudaMalloc(&dq, N * 4); cudaMalloc(&d1, N * 4); cudaMalloc(&d2, N * 4); cudaMalloc(&d3, N * 4);
    cudaMemcpy(dq, h, N * 4, cudaMemcpyHostToDevice);

    const float eps = 1e-6f / 128.0f;      // build_gdn_l2_norm passes eps/n
    const float s   = 1.0f / sqrtf(128.0f); // scale(1/sqrt(n))

    printf("row 0:\n");
    mock_rms<<<NR, 32>>>(dq, d1, eps);
    mock_scale<<<N / 256, 256>>>(d1, d2, s);
    prologue_like<<<NR, 32>>>(dq, d3, eps, s);
    cudaDeviceSynchronize();

    float *o2 = (float *) malloc(N * 4), *o3 = (float *) malloc(N * 4);
    cudaMemcpy(o2, d2, N * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(o3, d3, N * 4, cudaMemcpyDeviceToHost);

    int diff = 0;
    for (size_t i = 0; i < N; i++) {
        uint32_t a, b;
        memcpy(&a, &o2[i], 4);
        memcpy(&b, &o3[i], 4);
        if (a != b) {
            if (diff < 5) printf("bit diff at %zu: rms+scale %08x  prologue %08x\n", i, a, b);
            diff++;
        }
    }
    printf("%s: %d / %zu elements differ\n", diff ? "MISMATCH" : "BIT-EXACT", diff, N);
    return diff != 0;
}
