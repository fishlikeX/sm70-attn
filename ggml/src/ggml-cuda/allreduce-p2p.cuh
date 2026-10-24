#pragma once

#include "common.cuh"
#include "ggml-backend-impl.h"

#include <cstddef>

// (ported) the fold args struct upstream of this file lives in common.cuh in
// the source tree; our tree does not carry it, so it is defined here. The fold
// path itself is unused in this port - the struct only needs to resolve.
struct ggml_cuda_ar_fold_args {
    uint4 *        ll[4]     = {};
    uint4 *        own       = nullptr;
    unsigned int * ll_seq    = nullptr;
    int64_t        ll_stride = 0;
    int            rank      = 0;
    int            n_devs    = 0;
};

// One-shot peer-to-peer AllReduce for small messages.
//
// NCCL's ring AllReduce costs ~35 us of fixed overhead plus ~6.75 us per ring step, so a
// hidden-state sized message (tens of KB) is entirely dominated by the fixed term: measured
// 49 us at 2 ranks and 76 us at 4 on NVLink-connected V100s. Batch-1 tensor-parallel decode
// issues two AllReduces per layer, which made the collective ~35% of token generation time.
//
// This path trades bandwidth for latency: every rank pushes its whole payload straight into
// peer-mapped staging on all other ranks, synchronises once, then reduces locally. That is
// n_devices times the traffic of a ring, which is irrelevant at these sizes and removes all
// but one of the round trips. Above the size threshold the ring wins and this returns false
// so the caller falls back to NCCL.

struct ggml_cuda_p2p_ar;

// Returns nullptr if the path is unavailable: fewer than 2 devices, no peer access between
// some pair, or allocation failure. The caller keeps using NCCL in that case.
ggml_cuda_p2p_ar * ggml_cuda_p2p_ar_init(const int * devices, size_t n_devices);

void ggml_cuda_p2p_ar_free(ggml_cuda_p2p_ar * ar);

// In-place AllReduce (sum) over tensors[0..n_devices-1], tensors[i] living on devices[i].
// Ranks whose tensor lacks GGML_TENSOR_FLAG_COMPUTE contribute nothing but still receive the
// sum, matching the NCCL path's zero-fill semantics.
//
// Returns false without enqueueing anything if this call is not eligible (too large, not F32,
// misaligned, non-contiguous); the caller must then fall back. Returns true once the work is
// enqueued on each device's stream.
bool ggml_cuda_p2p_ar_allreduce(ggml_cuda_p2p_ar * ar, ggml_backend_t * backends, ggml_tensor ** tensors);

// AllGather of one row of equal 4-byte slices: dsts[i] receives every srcs[j] at offset j. false without enqueueing
//   anything if the call is not eligible (a slice above the staging size, several rows, misaligned); the caller falls back.
//   GGML_CUDA_P2P_ALLGATHER=0 turns it off.
bool ggml_cuda_p2p_ar_allgather(ggml_cuda_p2p_ar * ar, ggml_backend_t * backends, ggml_tensor ** srcs, ggml_tensor ** dsts);

// The same in two steps: prepare does the eligibility checks and claims a sequence number, launch
// enqueues one rank's kernel. launch may run on a different thread per rank, so the meta backend can
// issue it from the thread that already owns that device's graph launch.
bool ggml_cuda_p2p_ar_prepare(ggml_cuda_p2p_ar * ar, ggml_backend_t * backends, ggml_tensor ** tensors);
void ggml_cuda_p2p_ar_launch (ggml_cuda_p2p_ar * ar, ggml_backend_t * backends, ggml_tensor ** tensors, size_t rank);

// The LL kernel of one rank with its arguments built from the tensors, nothing claimed on the host and the sequence
// numbers kept on the device, so it can be captured into a CUDA graph. capturable tells whether a call takes this path.
bool ggml_cuda_p2p_ar_ll_capturable(const ggml_cuda_p2p_ar * ar, ggml_backend_t * backends, ggml_tensor ** tensors);
void ggml_cuda_p2p_ar_launch_ll    (const ggml_cuda_p2p_ar * ar, ggml_backend_t * backends, ggml_tensor ** tensors, size_t rank);

// The same with this rank's input computed in the kernel: the MoE weighted sum of n_used expert rows of one token
// (experts [n_embd, n_used], weights [n_used]) plus add_x * sigmoid(*add_g) if add_x, as moe_weighted_reduction
// computes it. Only the sum is written, into tensors[rank]; the call must be ll_capturable.
void ggml_cuda_p2p_ar_launch_ll_moe_reduce(const ggml_cuda_p2p_ar * ar, ggml_backend_t * backends, ggml_tensor ** tensors,
                                           size_t rank, const float * experts, const float * weights, int n_used,
                                           const float * add_x, const float * add_g);

// The LL kernel's receive half alone, for a call whose rank pushed its lines from the kernel that made its input (see
//   ggml_cuda_ar_fold_push): polls them, writes the sum into tensors[rank] and moves the sequence numbers on.
void ggml_cuda_p2p_ar_launch_ll_recv(const ggml_cuda_p2p_ar * ar, ggml_backend_t * backends, ggml_tensor ** tensors, size_t rank);

// The arguments of a fold of this call into the kernels around it, false if the call cannot be folded: it must be
//   ll_capturable with every rank contributing, on 2 to 4 devices, and a multiple of 4 values.
bool ggml_cuda_p2p_ar_fold_args(const ggml_cuda_p2p_ar * ar, ggml_backend_t * backends, ggml_tensor ** tensors, size_t rank,
                                ggml_cuda_ar_fold_args & args);

// The LL scheme of the kernels in allreduce-p2p.cu, for the kernels a fold puts it in. The values of a call are float4s;
//   LL block b (GGML_CUDA_P2P_AR_LL_NTHREADS float4s) has its own sequence number ll_seq[b]: call seq = ll_seq[b] + 1
//   uses slot seq % GGML_CUDA_P2P_AR_NSLOTS and flag seq | 1, and its last reader stores seq into ll_seq[b]. Float4 i
//   goes into two lines {d0, flag, d1, flag}; the k-th lines of the 32 float4s of a warp are next to each other.
#define GGML_CUDA_P2P_AR_NSLOTS 2
#define GGML_CUDA_P2P_AR_LL_NTHREADS 128

static __device__ __forceinline__ void ggml_cuda_ar_st_ll(uint4 * p, const uint4 v) {
    asm volatile("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

static __device__ __forceinline__ uint4 ggml_cuda_ar_ld_ll(const uint4 * p) {
    uint4 v;
    asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}

// line k of float4 i of a rank's region
static __device__ __forceinline__ int64_t ggml_cuda_ar_ll_line(const int64_t i, const int k) {
    return (i & ~(int64_t) (WARP_SIZE - 1))*2 + (i & (WARP_SIZE - 1)) + k*WARP_SIZE;
}

// the sequence number of this call for float4 i
static __device__ __forceinline__ uint32_t ggml_cuda_ar_seq(const ggml_cuda_ar_fold_args & a, const int64_t i) {
    return a.ll_seq[i / GGML_CUDA_P2P_AR_LL_NTHREADS] + 1;
}

// values (v0, v1), of line k of float4 i, to every rank
static __device__ __forceinline__ void ggml_cuda_ar_fold_push(const ggml_cuda_ar_fold_args & a, const uint32_t seq,
        const int64_t i, const int k, const float v0, const float v1) {
    const uint32_t flag = seq | 1u;
    const uint4    l    = make_uint4(__float_as_uint(v0), flag, __float_as_uint(v1), flag);
    const int64_t  off  = ((int64_t) (seq % GGML_CUDA_P2P_AR_NSLOTS)*a.n_devs + a.rank)*a.ll_stride + ggml_cuda_ar_ll_line(i, k);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if (j < a.n_devs) {
            ggml_cuda_ar_st_ll(a.ll[j] + off, l);
        }
    }
}

// the sums of the n values cols[0..n-1] over the ranks, in rank order from 0 as the LL kernels add them; the lines of
//   all of them are loaded at once and only those that have not arrived are loaded again
template <int n>
static __device__ __forceinline__ void ggml_cuda_ar_fold_sum(const ggml_cuda_ar_fold_args & a, const int64_t * cols,
        const bool * valid, float * sums) {
    uint32_t       flag[n];
    const uint4 *  p[n];
    uint4          r[n][4];
#pragma unroll
    for (int c = 0; c < n; ++c) {
        const int64_t  i   = cols[c] / 4;
        const uint32_t seq = valid[c] ? ggml_cuda_ar_seq(a, i) : 0;
        flag[c] = seq | 1u;
        p[c]    = a.own + (int64_t) (seq % GGML_CUDA_P2P_AR_NSLOTS)*a.n_devs*a.ll_stride + ggml_cuda_ar_ll_line(i, (cols[c] % 4) / 2);
    }
#pragma unroll
    for (int c = 0; c < n; ++c) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (valid[c] && j < a.n_devs) {
                r[c][j] = ggml_cuda_ar_ld_ll(p[c] + j*a.ll_stride);
            }
        }
    }
    bool done;
    do {
        done = true;
#pragma unroll
        for (int c = 0; c < n; ++c) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (valid[c] && j < a.n_devs && (r[c][j].y != flag[c] || r[c][j].w != flag[c])) {
                    r[c][j] = ggml_cuda_ar_ld_ll(p[c] + j*a.ll_stride);
                    done = false;
                }
            }
        }
    } while (!done);
#pragma unroll
    for (int c = 0; c < n; ++c) {
        float s = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (j < a.n_devs) {
                s += __uint_as_float(cols[c] % 2 == 0 ? r[c][j].x : r[c][j].z);
            }
        }
        sums[c] = valid[c] ? s : 0.0f;
    }
}

// after every reader of the call has polled its lines: ll_seq of the LL blocks of a row of ne values moves on, by
//   threads 0.. of one block
static __device__ __forceinline__ void ggml_cuda_ar_fold_done(const ggml_cuda_ar_fold_args & a, const int64_t ne, const int tid) {
    const int64_t nb = (ne/4 + GGML_CUDA_P2P_AR_LL_NTHREADS - 1) / GGML_CUDA_P2P_AR_LL_NTHREADS;
    if (tid < nb) {
        a.ll_seq[tid] = a.ll_seq[tid] + 1;
    }
}
