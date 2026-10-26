#include "allreduce-p2p.cuh"
#include "unary.cuh"

#include <algorithm>
#include <cstdlib>

// One block index is the unit of cross-device synchronisation: block b on every rank exchanges
// the same slice, so a per-block-index barrier is enough and no grid-wide sync is needed.
// Override the block count with GGML_CUDA_P2P_AR_NBLOCKS.
// 2026-10-26: the staged kernel is latency-bound at 16 blocks on prefill-sized
// messages (168 MB): 16 -> 40 -> 80 blocks measured 1472 -> 1514 -> 1544 t/s
// at 22k, 2x V100. The init clamps to one block per SM, so this is the ceiling.
#define GGML_CUDA_P2P_AR_NBLOCKS_DEFAULT 80
#define GGML_CUDA_P2P_AR_NTHREADS        256

// Two staging slots (GGML_CUDA_P2P_AR_NSLOTS), alternating per call. A rank can run at most one call ahead of its peers
// (it cannot finish call k until every peer published its flag for k, and a peer only publishes
// that flag after it finished reading call k-1), so two slots are enough to keep a fast rank
// from overwriting payload a slow rank has not consumed yet.
// The LL path for the smallest messages takes one float4 per thread (GGML_CUDA_P2P_AR_LL_NTHREADS per block), no
// inter-block coupling.

// local stand-in for the source tree op sigmoid helper (port note)
__device__ __forceinline__ float ggml_cuda_p2p_sigmoid_local(const float x) {
    return 1.0f / (1.0f + expf(-x));
}
struct ggml_cuda_p2p_ar_args {
    float4 *             stage[GGML_CUDA_MAX_DEVICES];
    unsigned long long * flag [GGML_CUDA_MAX_DEVICES];
    uint4 *              ll   [GGML_CUDA_MAX_DEVICES];

    const float4 *     src;
    float4 *           dst;
    unsigned int *     ll_seq; // this rank's sequence number per LL block, see ggml_cuda_p2p_ar_ll_kernel_n
    int64_t            nvec;
    int64_t            stage_stride;
    int64_t            ll_stride;
    int                n_devs;
    int                rank;
    int                slot;
    unsigned int       contrib_mask;
    unsigned long long seq;
};

struct ggml_cuda_p2p_ar {
    int    devices[GGML_CUDA_MAX_DEVICES] = {};
    size_t n_devices                      = 0;

    // Peer-mapped. stage[j] is device j's staging area, laid out [slot][rank][stage_stride].
    // flag[j] is device j's flag array, laid out [slot][rank][nblocks].
    // ll[j] is device j's LL staging, laid out [slot][rank][ll_stride] 16 B lines.
    float4 *             stage[GGML_CUDA_MAX_DEVICES] = {};
    unsigned long long * flag [GGML_CUDA_MAX_DEVICES] = {};
    uint4 *              ll   [GGML_CUDA_MAX_DEVICES] = {};
    unsigned int *       ll_seq[GGML_CUDA_MAX_DEVICES] = {}; // device j's sequence number per LL block

    size_t             max_bytes    = 0; // per-rank payload cap, above which the caller uses NCCL
    size_t             ll_max_bytes = 0; // per-rank payload cap for the LL kernel, 0 = LL off
    int64_t            stage_stride = 0; // float4 elements per (slot, rank) region
    int64_t            ll_stride    = 0; // LL lines per (slot, rank) region, two per float4
    int                nblocks      = 0; // grid size, also the flag array's last dimension
    unsigned long long seq          = 0; // monotonic, identical across ranks for a given call

    ggml_cuda_p2p_ar_args args = {}; // the call prepare claimed, read by launch on every rank
};

static __global__ void ggml_cuda_p2p_ar_kernel(const ggml_cuda_p2p_ar_args a) {
    const int64_t nblocks = gridDim.x;
    const int64_t bid     = blockIdx.x;

    const int64_t chunk = (a.nvec + nblocks - 1) / nblocks;
    const int64_t begin = bid * chunk;
    const int64_t end   = min(begin + chunk, a.nvec);

    const int64_t slot_base = (int64_t) a.slot * a.n_devs * a.stage_stride;

    // 1. push this block's slice into the PEERS' staging areas only. Our own slice is
    //    already in a.src, so staging it into a.stage[rank] as well doubled the local
    //    traffic (write + read back) for nothing; the reduce below starts from a.src.
    if ((a.contrib_mask >> a.rank) & 1u) {
        for (int j = 0; j < a.n_devs; ++j) {
            if (j == a.rank) {
                continue;
            }
            float4 * dst = a.stage[j] + slot_base + (int64_t) a.rank * a.stage_stride;
            for (int64_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
                dst[i] = a.src[i];
            }
        }
    }

    // 2. publish. __syncthreads first, so the fence covers every thread's stores and not just
    //    those of thread 0.
    __syncthreads();
    __threadfence_system();

    if (threadIdx.x == 0) {
        for (int j = 0; j < a.n_devs; ++j) {
            volatile unsigned long long * f =
                a.flag[j] + (((int64_t) a.slot * a.n_devs + a.rank) * nblocks + bid);
            *f = a.seq; // naturally aligned 8 B volatile store, emitted as a single st.volatile
        }

        // 3. wait for every rank's slice to land on our own device. seq is monotonic, so a peer
        //    that already raced ahead to seq+1 still satisfies this.
        for (int j = 0; j < a.n_devs; ++j) {
            volatile const unsigned long long * f =
                a.flag[a.rank] + (((int64_t) a.slot * a.n_devs + j) * nblocks + bid);
            while (*f < a.seq) {
#if __CUDA_ARCH__ >= 700
                __nanosleep(32);
#endif // __CUDA_ARCH__ >= 700
            }
        }
        __threadfence_system();
    }
    __syncthreads();

    // 4. reduce. __ldcg bypasses L1, so a peer's write cannot be shadowed by a stale line, while
    //    still loading 16 B at a time (plain volatile would split this into four 4 B loads).
    const float4 * stage = a.stage[a.rank] + slot_base;
    for (int64_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
        // own contribution first: same addition order as the staged-everywhere version
        // for the 2-rank case, so the result is bit-identical to it
        float4 sum = ((a.contrib_mask >> a.rank) & 1u) ? a.src[i] : make_float4(0.f, 0.f, 0.f, 0.f);
        for (int j = 0; j < a.n_devs; ++j) {
            if (j == a.rank || ((a.contrib_mask >> j) & 1u) == 0) {
                continue;
            }
            const float4 v = __ldcg(stage + (int64_t) j * a.stage_stride + i);
            sum.x += v.x;
            sum.y += v.y;
            sum.z += v.z;
            sum.w += v.w;
        }
        a.dst[i] = sum;
    }
}

// AllGather over the same staging and flags: block b pushes its part of this rank's slice to every peer, publishes,
//   waits for block b of every peer and copies the slices into place. Only copies, so every rank has the same bits.
static __global__ void ggml_cuda_p2p_ag_kernel(const ggml_cuda_p2p_ar_args a) {
    const int64_t nblocks = gridDim.x;
    const int64_t bid     = blockIdx.x;

    const int64_t chunk = (a.nvec + nblocks - 1) / nblocks;
    const int64_t begin = bid * chunk;
    const int64_t end   = min(begin + chunk, a.nvec);

    const int64_t slot_base = (int64_t) a.slot * a.n_devs * a.stage_stride;

    for (int j = 0; j < a.n_devs; ++j) {
        if (j == a.rank) {
            continue;
        }
        float4 * dst = a.stage[j] + slot_base + (int64_t) a.rank * a.stage_stride;
        for (int64_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
            dst[i] = a.src[i];
        }
    }
    for (int64_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
        a.dst[(int64_t) a.rank * a.nvec + i] = a.src[i];
    }

    __syncthreads();
    __threadfence_system();

    if (threadIdx.x == 0) {
        for (int j = 0; j < a.n_devs; ++j) {
            volatile unsigned long long * f =
                a.flag[j] + (((int64_t) a.slot * a.n_devs + a.rank) * nblocks + bid);
            *f = a.seq;
        }
        for (int j = 0; j < a.n_devs; ++j) {
            volatile const unsigned long long * f =
                a.flag[a.rank] + (((int64_t) a.slot * a.n_devs + j) * nblocks + bid);
            while (*f < a.seq) {
#if __CUDA_ARCH__ >= 700
                __nanosleep(32);
#endif // __CUDA_ARCH__ >= 700
            }
        }
        __threadfence_system();
    }
    __syncthreads();

    const float4 * stage = a.stage[a.rank] + slot_base;
    for (int j = 0; j < a.n_devs; ++j) {
        if (j == a.rank) {
            continue;
        }
        for (int64_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
            a.dst[(int64_t) j * a.nvec + i] = __ldcg(stage + (int64_t) j * a.stage_stride + i);
        }
    }
}

static __device__ __forceinline__ void ggml_cuda_p2p_ar_st_ll(uint4 * p, const uint4 v) {
    asm volatile("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

static __device__ __forceinline__ uint4 ggml_cuda_p2p_ar_ld_ll(const uint4 * p) {
    uint4 v;
    asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}

// Same exchange for small messages with NCCL's LL scheme: each 16 B line is {d0, flag, d1, flag} and
// one store, so the flag arrives with its data. The receiver polls the data itself: no fence, no
// barrier, no flag array. It doubles the bytes, which only pays below a few tens of KB.
static __global__ void ggml_cuda_p2p_ar_ll_kernel(const ggml_cuda_p2p_ar_args a) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= a.nvec) {
        return;
    }

    // Never 0, so zeroed staging cannot match. A slot only sees one parity of seq, so seq | 1 stays
    // unique per slot.
    const uint32_t flag = (uint32_t) a.seq | 1u;

    const int64_t slot_base = (int64_t) a.slot * a.n_devs * a.ll_stride;

    // A rank that does not contribute still pushes zeros: the lines are also the barrier.
    const float4 v = ((a.contrib_mask >> a.rank) & 1u) ? a.src[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const uint4 l0 = make_uint4(__float_as_uint(v.x), flag, __float_as_uint(v.y), flag);
    const uint4 l1 = make_uint4(__float_as_uint(v.z), flag, __float_as_uint(v.w), flag);
    for (int j = 0; j < a.n_devs; ++j) {
        uint4 * dst = a.ll[j] + slot_base + (int64_t) a.rank * a.ll_stride + 2*i;
        ggml_cuda_p2p_ar_st_ll(dst + 0, l0);
        ggml_cuda_p2p_ar_st_ll(dst + 1, l1);
    }

    // Sum in rank order, like the kernel above, so every rank gets the same bits.
    const uint4 * src = a.ll[a.rank] + slot_base + 2*i;
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int j = 0; j < a.n_devs; ++j) {
        const uint4 * p = src + (int64_t) j * a.ll_stride;
        uint4 r0;
        uint4 r1;
        do {
            r0 = ggml_cuda_p2p_ar_ld_ll(p + 0);
        } while (r0.y != flag || r0.w != flag);
        do {
            r1 = ggml_cuda_p2p_ar_ld_ll(p + 1);
        } while (r1.y != flag || r1.w != flag);
        if (((a.contrib_mask >> j) & 1u) == 0) {
            continue;
        }
        sum.x += __uint_as_float(r0.x);
        sum.y += __uint_as_float(r0.z);
        sum.z += __uint_as_float(r1.x);
        sum.w += __uint_as_float(r1.z);
    }
    a.dst[i] = sum;
}

// The same for a device count known at compile time: the lines of all ranks are loaded at once, and only the ones
// that have not arrived yet are loaded again. A poll loop per line waits a round trip before the next line's load.
// Each block keeps its sequence number on the device (ll_seq), so a launch takes no per-call argument and can sit in
// a CUDA graph; every rank runs the same calls with the same blocks, so the numbers agree. The slots then alternate
// per block, which keeps the guarantee above: a rank is at most one call ahead of any peer in each block.
// payload 1 (f16) or 2 (bf16): two values per {d, flag} pair instead of one, half the bytes on the links; every rank
//   sums the same rounded values, its own included, so the ranks still agree.
template <int payload>
static __device__ __forceinline__ uint32_t ggml_cuda_p2p_ar_pack2(const float x, const float y) {
    uint32_t bits;
    if constexpr (payload == 1) {
        const half2 h = make_half2(__float2half_rn(x), __float2half_rn(y));
        memcpy(&bits, &h, sizeof(bits));
    } else {
        const nv_bfloat162 h = __floats2bfloat162_rn(x, y);
        memcpy(&bits, &h, sizeof(bits));
    }
    return bits;
}

template <int payload>
static __device__ __forceinline__ float2 ggml_cuda_p2p_ar_unpack2(const uint32_t bits) {
    if constexpr (payload == 1) {
        half2 h;
        memcpy(&h, &bits, sizeof(bits));
        return __half22float2(h);
    } else {
        nv_bfloat162 h;
        memcpy(&h, &bits, sizeof(bits));
        return __bfloat1622float2(h);
    }
}

// This rank's float4 i, v, to the LL lines of every rank, and the sum of all ranks' in rank order, as the kernels
//   above: one line {d0, flag, d1, flag} per two values, or with payload one line per four. The k-th lines of the 32
//   float4s of a warp are next to each other (32 lines apart from its other ones), so that every store of a warp is
//   512 contiguous bytes, which cross the link to a peer in large packets.
// push = false: only the sums, the lines of this rank having been pushed by the kernel that made v (a fold, see
//   ggml_cuda_ar_fold_push)
template <int n_devs, int payload, bool push = true>
static __device__ __forceinline__ float4 ggml_cuda_p2p_ar_ll_exchange(const ggml_cuda_p2p_ar_args & a, const float4 v,
        const int64_t i, const uint32_t flag, const int64_t slot_base) {
    constexpr int nl = payload == 0 ? 2 : 1; // lines per float4
    const int64_t line0 = (i & ~(int64_t) (WARP_SIZE - 1))*nl + (i & (WARP_SIZE - 1)); // its line k is line0 + k*32
    if constexpr (push) {
        uint4 l[nl];
        if constexpr (payload == 0) {
            l[0] = make_uint4(__float_as_uint(v.x), flag, __float_as_uint(v.y), flag);
            l[1] = make_uint4(__float_as_uint(v.z), flag, __float_as_uint(v.w), flag);
        } else {
            l[0] = make_uint4(ggml_cuda_p2p_ar_pack2<payload>(v.x, v.y), flag, ggml_cuda_p2p_ar_pack2<payload>(v.z, v.w), flag);
        }
#pragma unroll
        for (int j = 0; j < n_devs; ++j) {
            uint4 * dst = a.ll[j] + slot_base + (int64_t) a.rank * a.ll_stride + line0;
#pragma unroll
            for (int k = 0; k < nl; ++k) {
                ggml_cuda_p2p_ar_st_ll(dst + k*WARP_SIZE, l[k]);
            }
        }
    } else {
        GGML_UNUSED(v);
    }

    const uint4 * src = a.ll[a.rank] + slot_base + line0;
    uint4 r[n_devs][nl];
#pragma unroll
    for (int j = 0; j < n_devs; ++j) {
#pragma unroll
        for (int k = 0; k < nl; ++k) {
            r[j][k] = ggml_cuda_p2p_ar_ld_ll(src + (int64_t) j * a.ll_stride + k*WARP_SIZE);
        }
    }
    bool done;
    do {
        done = true;
#pragma unroll
        for (int j = 0; j < n_devs; ++j) {
#pragma unroll
            for (int k = 0; k < nl; ++k) {
                if (r[j][k].y != flag || r[j][k].w != flag) {
                    r[j][k] = ggml_cuda_p2p_ar_ld_ll(src + (int64_t) j * a.ll_stride + k*WARP_SIZE);
                    done = false;
                }
            }
        }
    } while (!done);

    // in rank order, as the kernels above
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int j = 0; j < n_devs; ++j) {
        if (((a.contrib_mask >> j) & 1u) == 0) {
            continue;
        }
        if constexpr (payload == 0) {
            sum.x += __uint_as_float(r[j][0].x);
            sum.y += __uint_as_float(r[j][0].z);
            sum.z += __uint_as_float(r[j][1].x);
            sum.w += __uint_as_float(r[j][1].z);
        } else {
            const float2 lo = ggml_cuda_p2p_ar_unpack2<payload>(r[j][0].x);
            const float2 hi = ggml_cuda_p2p_ar_unpack2<payload>(r[j][0].z);
            sum.x += lo.x;
            sum.y += lo.y;
            sum.z += hi.x;
            sum.w += hi.y;
        }
    }
    return sum;
}

// push = false: the receive half alone, see ggml_cuda_p2p_ar_launch_ll_recv
template <int n_devs, int payload = 0, bool push = true>
static __global__ void ggml_cuda_p2p_ar_ll_kernel_n(const ggml_cuda_p2p_ar_args a) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    const uint32_t seq  = a.ll_seq[blockIdx.x] + 1;
    const uint32_t flag = seq | 1u;

    const int64_t slot_base = (int64_t) (seq % GGML_CUDA_P2P_AR_NSLOTS) * n_devs * a.ll_stride;

    if (i < a.nvec) {
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if constexpr (push) {
            if ((a.contrib_mask >> a.rank) & 1u) {
                v = a.src[i];
            }
        }
        a.dst[i] = ggml_cuda_p2p_ar_ll_exchange<n_devs, payload, push>(a, v, i, flag, slot_base);
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        a.ll_seq[blockIdx.x] = seq;
    }
}

// ggml_cuda_p2p_ar_ll_kernel_n with its input computed on the way: the MoE weighted sum of the token's n_used expert
// rows (and add_x times sigmoid(add_g) after it), as moe_weighted_reduction_f32 sums each column, so the values are
// those of the two kernels apart. Same blocks, sequence numbers and slots, so a peer may run either kernel.
template <int n_devs, int payload = 0>
static __global__ void ggml_cuda_p2p_ar_ll_moe_reduce_kernel(const ggml_cuda_p2p_ar_args a, const float * experts,
        const float * weights, const int n_used, const int64_t n_embd, const float * add_x, const float * add_g) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    const uint32_t seq  = a.ll_seq[blockIdx.x] + 1;
    const uint32_t flag = seq | 1u;

    const int64_t slot_base = (int64_t) (seq % GGML_CUDA_P2P_AR_NSLOTS) * n_devs * a.ll_stride;

    if (i < a.nvec) {
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if ((a.contrib_mask >> a.rank) & 1u) {
            // every load is issued before the first sum: one round trip instead of one per 4 experts and the shared expert
            constexpr int ne_pf = 8;
            float4 xs[ne_pf];
            float  ws[ne_pf];
#pragma unroll
            for (int e = 0; e < ne_pf; ++e) {
                if (e < n_used) {
                    xs[e] = ((const float4 *) (experts + e*n_embd))[i];
                    ws[e] = weights[e];
                }
            }
            float4 xa = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float  ga = 0.0f;
            if (add_x != nullptr) {
                xa = ((const float4 *) add_x)[i];
                ga = *add_g;
            }
            for (int e0 = 0; e0 < n_used; e0 += ne_pf) {
                if (e0 > 0) {
#pragma unroll
                    for (int e = 0; e < ne_pf; ++e) {
                        if (e0 + e < n_used) {
                            xs[e] = ((const float4 *) (experts + (e0 + e)*n_embd))[i];
                            ws[e] = weights[e0 + e];
                        }
                    }
                }
#pragma unroll
                for (int e = 0; e < ne_pf; ++e) {
                    if (e0 + e < n_used) {
                        const float4 x = xs[e];
                        const float  w = ws[e];
                        const float4 p = make_float4(__fmul_rn(x.x, w), __fmul_rn(x.y, w), __fmul_rn(x.z, w), __fmul_rn(x.w, w));
                        v = e0 + e == 0 ? p : make_float4(__fadd_rn(v.x, p.x), __fadd_rn(v.y, p.y), __fadd_rn(v.z, p.z), __fadd_rn(v.w, p.w));
                    }
                }
            }
            if (add_x != nullptr) {
                const float sig = ggml_cuda_p2p_sigmoid_local(ga);
                v = make_float4(__fadd_rn(v.x, __fmul_rn(xa.x, sig)), __fadd_rn(v.y, __fmul_rn(xa.y, sig)),
                                __fadd_rn(v.z, __fmul_rn(xa.z, sig)), __fadd_rn(v.w, __fmul_rn(xa.w, sig)));
            }
        }
        a.dst[i] = ggml_cuda_p2p_ar_ll_exchange<n_devs, payload>(a, v, i, flag, slot_base);
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        a.ll_seq[blockIdx.x] = seq;
    }
}

void ggml_cuda_p2p_ar_free(ggml_cuda_p2p_ar * ar) {
    if (ar == nullptr) {
        return;
    }
    for (size_t i = 0; i < ar->n_devices; ++i) {
        if (cudaSetDevice(ar->devices[i]) != cudaSuccess) {
            continue;
        }
        cudaFree(ar->stage[i]);
        cudaFree(ar->flag [i]);
        cudaFree(ar->ll   [i]);
        cudaFree(ar->ll_seq[i]);
    }
    (void) cudaGetLastError();
    delete ar;
}

ggml_cuda_p2p_ar * ggml_cuda_p2p_ar_init(const int * devices, size_t n_devices) {
    if (n_devices < 2 || n_devices > GGML_CUDA_MAX_DEVICES) {
        return nullptr;
    }

    // 2026-10-26: the staged kernel writes the WHOLE message into the staging area,
    // so max_bytes has to cover it. At 1 MiB the [n_embd x ub] f32 partials of a
    // prefill (168 MB at ub 8192, 84 MB at 4096) missed this path and fell into the
    // host-staged copy-engine reduce: measured 739 -> 1726 t/s at 22k on 2x V100
    // when they take this one instead (NVLink peer writes, no PCIe round trip).
    // Cost: NSLOTS(2) * n_devices * max_bytes of VRAM per device (~704 MB on 2 GPUs),
    // only when tensor parallelism is actually in use. Override with the env knob.
    size_t max_bytes = 176ull*1024*1024;
    if (const char * env = getenv("GGML_CUDA_P2P_AR_MAX_BYTES")) {
        max_bytes = (size_t) atoll(env);
    }
    if (max_bytes == 0) {
        GGML_LOG_INFO("%s: one-shot P2P AllReduce disabled by GGML_CUDA_P2P_AR_MAX_BYTES=0\n", __func__);
        return nullptr;
    }
    max_bytes = GGML_PAD(max_bytes, sizeof(float4));

    // Decode and speculative verify sizes (tokens x n_embd x 4 B). In the model the LL kernel wins up to 128 KB (4x V100,
    //   -sm tensor, 4096 cells: Qwen3.8-27B 2/4 tokens = 40/80 KB +4.3%/+2.0%, Ornith 8 tokens = 64 KB +3.2%) and ties at
    //   160 KB; a lock-step harness of the older kernel had it losing above ~48 KB.
    size_t ll_max_bytes = 128*1024;
    if (const char * env = getenv("GGML_CUDA_P2P_AR_LL_MAX_BYTES")) {
        ll_max_bytes = (size_t) atoll(env);
    }
    ll_max_bytes = std::min(GGML_PAD(ll_max_bytes, sizeof(float4)), max_bytes);

    int nblocks = GGML_CUDA_P2P_AR_NBLOCKS_DEFAULT;
    if (const char * env = getenv("GGML_CUDA_P2P_AR_NBLOCKS")) {
        nblocks = atoi(env);
    }
    // Block b waits on block b of every other rank, so all blocks must be resident at the same
    // time -- more blocks than SMs deadlocks. One block per SM is the safe ceiling.
    int max_nblocks = ggml_cuda_info().devices[devices[0]].nsm;
    for (size_t i = 1; i < n_devices; ++i) {
        const int nsm = ggml_cuda_info().devices[devices[i]].nsm;
        if (nsm < max_nblocks) {
            max_nblocks = nsm;
        }
    }
    if (nblocks < 1 || nblocks > max_nblocks) {
        GGML_LOG_WARN("%s: GGML_CUDA_P2P_AR_NBLOCKS=%d outside [1, %d], clamping\n",
            __func__, nblocks, max_nblocks);
        nblocks = nblocks < 1 ? 1 : max_nblocks;
    }

    // Every pair has to be reachable, a partial mesh would leave some ranks unable to push.
    for (size_t i = 0; i < n_devices; ++i) {
        for (size_t j = 0; j < n_devices; ++j) {
            if (i == j) {
                continue;
            }
            int can_access = 0;
            if (cudaDeviceCanAccessPeer(&can_access, devices[i], devices[j]) != cudaSuccess || !can_access) {
                (void) cudaGetLastError();
                GGML_LOG_INFO("%s: no peer access %d -> %d, keeping NCCL for AllReduce\n",
                    __func__, devices[i], devices[j]);
                return nullptr;
            }
        }
    }

    ggml_cuda_p2p_ar * ar = new ggml_cuda_p2p_ar;
    ar->n_devices    = n_devices;
    ar->max_bytes    = max_bytes;
    ar->ll_max_bytes = ll_max_bytes;
    ar->stage_stride = (int64_t) (max_bytes / sizeof(float4));
    // whole warps of lines, see ggml_cuda_p2p_ar_ll_exchange
    ar->ll_stride    = (int64_t) (2 * GGML_PAD(ll_max_bytes, WARP_SIZE*sizeof(float4)) / sizeof(uint4));
    ar->nblocks      = nblocks;

    const size_t stage_bytes = (size_t) GGML_CUDA_P2P_AR_NSLOTS * n_devices * max_bytes;
    const size_t flag_bytes  = (size_t) GGML_CUDA_P2P_AR_NSLOTS * n_devices *
                               ar->nblocks * sizeof(unsigned long long);
    const size_t ll_bytes    = (size_t) GGML_CUDA_P2P_AR_NSLOTS * n_devices * 2 * GGML_PAD(ll_max_bytes, WARP_SIZE*sizeof(float4));
    const size_t ll_seq_bytes = (ll_max_bytes/sizeof(float4) + GGML_CUDA_P2P_AR_LL_NTHREADS - 1) / GGML_CUDA_P2P_AR_LL_NTHREADS * sizeof(unsigned int);

    for (size_t i = 0; i < n_devices; ++i) {
        ar->devices[i] = devices[i];

        if (cudaSetDevice(devices[i]) != cudaSuccess) {
            (void) cudaGetLastError();
            ggml_cuda_p2p_ar_free(ar);
            return nullptr;
        }

        for (size_t j = 0; j < n_devices; ++j) {
            if (i == j) {
                continue;
            }
            const cudaError_t err = cudaDeviceEnablePeerAccess(devices[j], 0);
            if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                (void) cudaGetLastError();
                GGML_LOG_INFO("%s: could not enable peer access %d -> %d, keeping NCCL for AllReduce\n",
                    __func__, devices[i], devices[j]);
                ggml_cuda_p2p_ar_free(ar);
                return nullptr;
            }
            (void) cudaGetLastError(); // clear a sticky AlreadyEnabled
        }

        if (cudaMalloc((void **) &ar->stage[i], stage_bytes) != cudaSuccess ||
            cudaMalloc((void **) &ar->flag [i], flag_bytes)  != cudaSuccess ||
            cudaMemset(ar->flag[i], 0, flag_bytes)           != cudaSuccess ||
            (ll_bytes > 0 && cudaMalloc((void **) &ar->ll[i], ll_bytes) != cudaSuccess) ||
            (ll_bytes > 0 && cudaMemset(ar->ll[i], 0, ll_bytes)         != cudaSuccess) ||
            (ll_bytes > 0 && cudaMalloc((void **) &ar->ll_seq[i], ll_seq_bytes) != cudaSuccess) ||
            (ll_bytes > 0 && cudaMemset(ar->ll_seq[i], 0, ll_seq_bytes)         != cudaSuccess)) {
            (void) cudaGetLastError();
            GGML_LOG_INFO("%s: staging allocation failed, keeping NCCL for AllReduce\n", __func__);
            ggml_cuda_p2p_ar_free(ar);
            return nullptr;
        }
    }

    GGML_LOG_INFO("%s: one-shot P2P AllReduce enabled for %zu devices, up to %zu KiB per rank "
                  "(LL up to %zu KiB, %zu MiB staging per device), %d blocks\n",
        __func__, n_devices, max_bytes >> 10, ll_max_bytes >> 10, (stage_bytes + flag_bytes + ll_bytes) >> 20, ar->nblocks);

    return ar;
}

bool ggml_cuda_p2p_ar_allreduce(ggml_cuda_p2p_ar * ar, ggml_backend_t * backends, ggml_tensor ** tensors) {
    if (!ggml_cuda_p2p_ar_prepare(ar, backends, tensors)) {
        return false;
    }
    for (size_t i = 0; i < ar->n_devices; ++i) {
        ggml_cuda_p2p_ar_launch(ar, backends, tensors, i);
    }
    return true;
}

// the values on the LL lines, GGML_CUDA_P2P_AR_LL_PAYLOAD: f32 (0, default), f16 (1) or bf16 (2). A 16-bit payload
//   halves the bytes on the links and rounds what each rank adds (opt-in, it changes the results)
static int ggml_cuda_p2p_ar_ll_payload() {
    static const int payload = [] {
        const char * env = getenv("GGML_CUDA_P2P_AR_LL_PAYLOAD");
        return env == nullptr ? 0 : strcmp(env, "f16") == 0 ? 1 : strcmp(env, "bf16") == 0 ? 2 : 0;
    }();
    return payload;
}

// the LL kernel of 2 to 4 devices
static void ggml_cuda_p2p_ar_launch_ll_n(const ggml_cuda_p2p_ar_args & args, const int nblocks_ll, cudaStream_t stream) {
    const int payload = ggml_cuda_p2p_ar_ll_payload();
    const auto launch = [&](auto kernel) {
        kernel<<<nblocks_ll, GGML_CUDA_P2P_AR_LL_NTHREADS, 0, stream>>>(args);
    };
    switch (args.n_devs * 3 + payload) {
        case 2*3 + 0: launch(ggml_cuda_p2p_ar_ll_kernel_n<2, 0>); break;
        case 2*3 + 1: launch(ggml_cuda_p2p_ar_ll_kernel_n<2, 1>); break;
        case 2*3 + 2: launch(ggml_cuda_p2p_ar_ll_kernel_n<2, 2>); break;
        case 3*3 + 0: launch(ggml_cuda_p2p_ar_ll_kernel_n<3, 0>); break;
        case 3*3 + 1: launch(ggml_cuda_p2p_ar_ll_kernel_n<3, 1>); break;
        case 3*3 + 2: launch(ggml_cuda_p2p_ar_ll_kernel_n<3, 2>); break;
        case 4*3 + 0: launch(ggml_cuda_p2p_ar_ll_kernel_n<4, 0>); break;
        case 4*3 + 1: launch(ggml_cuda_p2p_ar_ll_kernel_n<4, 1>); break;
        case 4*3 + 2: launch(ggml_cuda_p2p_ar_ll_kernel_n<4, 2>); break;
        default: GGML_ABORT("fatal error");
    }
}

void ggml_cuda_p2p_ar_launch(ggml_cuda_p2p_ar * ar, ggml_backend_t * backends, ggml_tensor ** tensors, size_t rank) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backends[rank]->context;

    ggml_cuda_p2p_ar_args args = ar->args;
    args.rank   = (int) rank;
    args.src    = (const float4 *) tensors[rank]->data;
    args.dst    = (float4 *)       tensors[rank]->data;
    args.ll_seq = ar->ll_seq[rank];

    ggml_cuda_set_device(cuda_ctx->device);

    if ((size_t) args.nvec * sizeof(float4) <= ar->ll_max_bytes) {
        const int nblocks_ll = (int) ((args.nvec + GGML_CUDA_P2P_AR_LL_NTHREADS - 1) / GGML_CUDA_P2P_AR_LL_NTHREADS);
        if (args.n_devs >= 2 && args.n_devs <= 4) {
            ggml_cuda_p2p_ar_launch_ll_n(args, nblocks_ll, cuda_ctx->stream());
        } else {
            ggml_cuda_p2p_ar_ll_kernel<<<nblocks_ll, GGML_CUDA_P2P_AR_LL_NTHREADS, 0, cuda_ctx->stream()>>>(args);
        }
    } else {
        ggml_cuda_p2p_ar_kernel<<<ar->nblocks, GGML_CUDA_P2P_AR_NTHREADS, 0, cuda_ctx->stream()>>>(args);
    }
    CUDA_CHECK(cudaGetLastError());
}

// whether the call can take this path, and which ranks contribute
static bool ggml_cuda_p2p_ar_check(const ggml_cuda_p2p_ar * ar, ggml_backend_t * backends, ggml_tensor ** tensors,
        unsigned int & contrib_mask) {
    const size_t  n_devices = ar->n_devices;
    const int64_t ne        = ggml_nelements(tensors[0]);

    // Empty reductions and the large, bandwidth-bound ones stay on the NCCL path.
    if (ne == 0 || tensors[0]->type != GGML_TYPE_F32) {
        return false;
    }
    if ((size_t) ne * sizeof(float) > ar->max_bytes) {
        return false;
    }
    if (ne % 4 != 0) { // one float4 per lane, there is no scalar tail path
        return false;
    }

    contrib_mask = 0;
    for (size_t i = 0; i < n_devices; ++i) {
        const ggml_tensor * t = tensors[i];
        if (t == nullptr || t->type != GGML_TYPE_F32 || ggml_nelements(t) != ne) {
            return false;
        }
        if (!ggml_is_contiguously_allocated(t) || ((uintptr_t) t->data % sizeof(float4)) != 0) {
            return false;
        }
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backends[i]->context;
        if (cuda_ctx->device != ar->devices[i]) {
            return false;
        }
        if (t->flags & GGML_TENSOR_FLAG_COMPUTE) {
            contrib_mask |= 1u << i;
        }
    }
    return true;
}

bool ggml_cuda_p2p_ar_ll_capturable(const ggml_cuda_p2p_ar * ar, ggml_backend_t * backends, ggml_tensor ** tensors) {
    unsigned int contrib_mask = 0;
    return ar != nullptr && ar->n_devices >= 2 && ar->n_devices <= 4 && ar->ll_max_bytes > 0 &&
        ggml_cuda_p2p_ar_check(ar, backends, tensors, contrib_mask) &&
        (size_t) ggml_nelements(tensors[0]) * sizeof(float) <= ar->ll_max_bytes;
}

void ggml_cuda_p2p_ar_launch_ll(const ggml_cuda_p2p_ar * ar, ggml_backend_t * backends, ggml_tensor ** tensors, size_t rank) {
    unsigned int contrib_mask = 0;
    GGML_ASSERT(ggml_cuda_p2p_ar_ll_capturable(ar, backends, tensors) && ggml_cuda_p2p_ar_check(ar, backends, tensors, contrib_mask));

    ggml_cuda_p2p_ar_args args = {};
    for (size_t j = 0; j < ar->n_devices; ++j) {
        args.ll[j] = ar->ll[j];
    }
    args.nvec         = ggml_nelements(tensors[0]) / 4;
    args.ll_stride    = ar->ll_stride;
    args.n_devs       = (int) ar->n_devices;
    args.contrib_mask = contrib_mask;
    args.rank         = (int) rank;
    args.src          = (const float4 *) tensors[rank]->data;
    args.dst          = (float4 *)       tensors[rank]->data;
    args.ll_seq       = ar->ll_seq[rank];

    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backends[rank]->context;
    ggml_cuda_set_device(cuda_ctx->device);
    const int nblocks_ll = (int) ((args.nvec + GGML_CUDA_P2P_AR_LL_NTHREADS - 1) / GGML_CUDA_P2P_AR_LL_NTHREADS);
    ggml_cuda_p2p_ar_launch_ll_n(args, nblocks_ll, cuda_ctx->stream());
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_p2p_ar_launch_ll_recv(const ggml_cuda_p2p_ar * ar, ggml_backend_t * backends, ggml_tensor ** tensors, size_t rank) {
    unsigned int contrib_mask = 0;
    GGML_ASSERT(ggml_cuda_p2p_ar_ll_capturable(ar, backends, tensors) && ggml_cuda_p2p_ar_check(ar, backends, tensors, contrib_mask));

    ggml_cuda_p2p_ar_args args = {};
    for (size_t j = 0; j < ar->n_devices; ++j) {
        args.ll[j] = ar->ll[j];
    }
    args.nvec         = ggml_nelements(tensors[0]) / 4;
    args.ll_stride    = ar->ll_stride;
    args.n_devs       = (int) ar->n_devices;
    args.contrib_mask = contrib_mask;
    args.rank         = (int) rank;
    args.src          = nullptr;
    args.dst          = (float4 *) tensors[rank]->data;
    args.ll_seq       = ar->ll_seq[rank];

    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backends[rank]->context;
    ggml_cuda_set_device(cuda_ctx->device);
    const int nblocks_ll = (int) ((args.nvec + GGML_CUDA_P2P_AR_LL_NTHREADS - 1) / GGML_CUDA_P2P_AR_LL_NTHREADS);
    const auto launch = [&](auto kernel) {
        kernel<<<nblocks_ll, GGML_CUDA_P2P_AR_LL_NTHREADS, 0, cuda_ctx->stream()>>>(args);
    };
    switch (args.n_devs) {
        case 2: launch(ggml_cuda_p2p_ar_ll_kernel_n<2, 0, false>); break;
        case 3: launch(ggml_cuda_p2p_ar_ll_kernel_n<3, 0, false>); break;
        case 4: launch(ggml_cuda_p2p_ar_ll_kernel_n<4, 0, false>); break;
        default: GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_p2p_ar_fold_args(const ggml_cuda_p2p_ar * ar, ggml_backend_t * backends, ggml_tensor ** tensors, size_t rank,
        ggml_cuda_ar_fold_args & args) {
    unsigned int contrib_mask = 0;
    // the lines of a fold carry f32 values, see ggml_cuda_ar_fold_push
    if (!ggml_cuda_p2p_ar_ll_capturable(ar, backends, tensors) || !ggml_cuda_p2p_ar_check(ar, backends, tensors, contrib_mask) ||
            ar->n_devices > 4 || contrib_mask != (1u << ar->n_devices) - 1 || ggml_cuda_p2p_ar_ll_payload() != 0) {
        return false;
    }
    args = {};
    for (size_t j = 0; j < ar->n_devices; ++j) {
        args.ll[j] = ar->ll[j];
    }
    args.own       = ar->ll[rank];
    args.ll_seq    = ar->ll_seq[rank];
    args.ll_stride = ar->ll_stride;
    args.rank      = (int) rank;
    args.n_devs    = (int) ar->n_devices;
    return true;
}

void ggml_cuda_p2p_ar_launch_ll_moe_reduce(const ggml_cuda_p2p_ar * ar, ggml_backend_t * backends, ggml_tensor ** tensors,
        size_t rank, const float * experts, const float * weights, int n_used, const float * add_x, const float * add_g) {
    unsigned int contrib_mask = 0;
    GGML_ASSERT(ggml_cuda_p2p_ar_ll_capturable(ar, backends, tensors) && ggml_cuda_p2p_ar_check(ar, backends, tensors, contrib_mask));

    ggml_cuda_p2p_ar_args args = {};
    for (size_t j = 0; j < ar->n_devices; ++j) {
        args.ll[j] = ar->ll[j];
    }
    args.nvec         = ggml_nelements(tensors[0]) / 4;
    args.ll_stride    = ar->ll_stride;
    args.n_devs       = (int) ar->n_devices;
    args.contrib_mask = contrib_mask;
    args.rank         = (int) rank;
    args.src          = nullptr;
    args.dst          = (float4 *) tensors[rank]->data;
    args.ll_seq       = ar->ll_seq[rank];
    const int64_t n_embd = ggml_nelements(tensors[rank]);

    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backends[rank]->context;
    ggml_cuda_set_device(cuda_ctx->device);
    const int nblocks_ll = (int) ((args.nvec + GGML_CUDA_P2P_AR_LL_NTHREADS - 1) / GGML_CUDA_P2P_AR_LL_NTHREADS);
    const auto launch = [&](auto kernel) {
        kernel<<<nblocks_ll, GGML_CUDA_P2P_AR_LL_NTHREADS, 0, cuda_ctx->stream()>>>(
            args, experts, weights, n_used, n_embd, add_x, add_g);
    };
    switch ((args.n_devs < 4 ? args.n_devs : 4) * 3 + ggml_cuda_p2p_ar_ll_payload()) {
        case 2*3 + 0: launch(ggml_cuda_p2p_ar_ll_moe_reduce_kernel<2, 0>); break;
        case 2*3 + 1: launch(ggml_cuda_p2p_ar_ll_moe_reduce_kernel<2, 1>); break;
        case 2*3 + 2: launch(ggml_cuda_p2p_ar_ll_moe_reduce_kernel<2, 2>); break;
        case 3*3 + 0: launch(ggml_cuda_p2p_ar_ll_moe_reduce_kernel<3, 0>); break;
        case 3*3 + 1: launch(ggml_cuda_p2p_ar_ll_moe_reduce_kernel<3, 1>); break;
        case 3*3 + 2: launch(ggml_cuda_p2p_ar_ll_moe_reduce_kernel<3, 2>); break;
        case 4*3 + 0: launch(ggml_cuda_p2p_ar_ll_moe_reduce_kernel<4, 0>); break;
        case 4*3 + 1: launch(ggml_cuda_p2p_ar_ll_moe_reduce_kernel<4, 1>); break;
        case 4*3 + 2: launch(ggml_cuda_p2p_ar_ll_moe_reduce_kernel<4, 2>); break;
        default: GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_p2p_ar_allgather(ggml_cuda_p2p_ar * ar, ggml_backend_t * backends, ggml_tensor ** srcs, ggml_tensor ** dsts) {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_P2P_ALLGATHER");
        return env == nullptr || atoi(env) != 0;
    }();
    if (!enabled || ar == nullptr) {
        return false;
    }
    const size_t  n_devices = ar->n_devices;
    const int64_t ne        = ggml_nelements(srcs[0]);
    // one row of equal 4-byte slices, in float4s, each slice fitting a rank's staging region
    if (ne == 0 || ne % 4 != 0 || ggml_type_size(srcs[0]->type) != 4 || ggml_blck_size(srcs[0]->type) != 1 ||
            (size_t) ne * 4 > ar->max_bytes) {
        return false;
    }
    for (size_t i = 0; i < n_devices; ++i) {
        const ggml_tensor * s = srcs[i];
        const ggml_tensor * d = dsts[i];
        if (s == nullptr || d == nullptr || s->type != srcs[0]->type || d->type != srcs[0]->type ||
                ggml_nelements(s) != ne || ggml_nelements(d) != ne * (int64_t) n_devices ||
                ggml_nrows(s) != 1 || ggml_nrows(d) != 1) {
            return false;
        }
        if (!ggml_is_contiguously_allocated(s) || !ggml_is_contiguously_allocated(d) ||
                ((uintptr_t) s->data % sizeof(float4)) != 0 || ((uintptr_t) d->data % sizeof(float4)) != 0) {
            return false;
        }
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backends[i]->context;
        if (cuda_ctx->device != ar->devices[i]) {
            return false;
        }
    }

    ar->seq++;

    ggml_cuda_p2p_ar_args args = {};
    for (size_t j = 0; j < n_devices; ++j) {
        args.stage[j] = ar->stage[j];
        args.flag [j] = ar->flag [j];
    }
    args.nvec         = ne / 4;
    args.stage_stride = ar->stage_stride;
    args.n_devs       = (int) n_devices;
    args.slot         = (int) (ar->seq % GGML_CUDA_P2P_AR_NSLOTS);
    args.seq          = ar->seq;

    for (size_t i = 0; i < n_devices; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backends[i]->context;
        args.rank = (int) i;
        args.src  = (const float4 *) srcs[i]->data;
        args.dst  = (float4 *)       dsts[i]->data;
        ggml_cuda_set_device(cuda_ctx->device);
        ggml_cuda_p2p_ag_kernel<<<ar->nblocks, GGML_CUDA_P2P_AR_NTHREADS, 0, cuda_ctx->stream()>>>(args);
        CUDA_CHECK(cudaGetLastError());
    }
    return true;
}

bool ggml_cuda_p2p_ar_prepare(ggml_cuda_p2p_ar * ar, ggml_backend_t * backends, ggml_tensor ** tensors) {
    if (ar == nullptr) {
        return false;
    }

    const size_t  n_devices = ar->n_devices;
    const int64_t ne        = ggml_nelements(tensors[0]);

    unsigned int contrib_mask = 0;
    if (!ggml_cuda_p2p_ar_check(ar, backends, tensors, contrib_mask)) {
        return false;
    }

    ar->seq++;

    ggml_cuda_p2p_ar_args & args = ar->args;
    args = {};
    for (size_t j = 0; j < n_devices; ++j) {
        args.stage[j] = ar->stage[j];
        args.flag [j] = ar->flag [j];
        args.ll   [j] = ar->ll   [j];
    }
    args.nvec         = ne / 4;
    args.stage_stride = ar->stage_stride;
    args.ll_stride    = ar->ll_stride;
    args.n_devs       = (int) n_devices;
    args.slot         = (int) (ar->seq % GGML_CUDA_P2P_AR_NSLOTS);
    args.contrib_mask = contrib_mask;
    args.seq          = ar->seq;

    return true;
}
