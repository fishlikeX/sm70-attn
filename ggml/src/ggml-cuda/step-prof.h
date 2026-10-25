#pragma once
// LLAMA_SM70_STEP_PROF=1 — CUDA half of the decode-round budget instrument:
// per-device stream-event rings around each backend graph compute call.
// Each drained pair yields the GPU busy span of that call; the host half
// (ggml-step-prof.h) classifies and prints them. Capture-safe: records sit
// outside every capture region, drains only touch pairs from earlier calls,
// age/completion-gated so the probe never stalls ahead of the GPU (a full
// ring force-syncs the oldest pair instead of dropping samples).

#include "../ggml-step-prof.h"

#include <cuda_runtime.h>

struct ggml_sm70_step_prof_ring {
    static constexpr int RING = 256;
    static constexpr int SMALL_GRAPH_NODES = 600;
    static constexpr float VERIFY_SPAN_MS = 200.0f;

    cudaEvent_t ev_a[RING] = {};
    cudaEvent_t ev_b[RING] = {};
    long long   ev_t0[RING] = {};   // host ms when the pair was recorded
    int         ev_nodes[RING] = {};
    int         ev_head = 0;        // oldest undrained slot
    int         ev_slot = 0;        // next slot to record into
    bool        ev_init = false;

    static long long now_ms() {
        return ggml_sm70_step_prof::now_ms();
    }

    void drain(bool force) {
        const long long cutoff = now_ms() - 100;
        while (ev_head != ev_slot) {
            if (!force && ev_t0[ev_head] >= cutoff) {
                break;
            }
            cudaError_t q = cudaEventQuery(ev_b[ev_head]);
            if (q == cudaErrorNotReady) {
                if (!force) {
                    break;
                }
                cudaEventSynchronize(ev_b[ev_head]);
            } else if (q != cudaSuccess) {
                ev_head = (ev_head + 1) % RING; // poisoned pair, drop it
                continue;
            }
            float ms = 0;
            if (cudaEventElapsedTime(&ms, ev_a[ev_head], ev_b[ev_head]) == cudaSuccess) {
                pool_add(ev_nodes[ev_head] < SMALL_GRAPH_NODES, ms < VERIFY_SPAN_MS, ms);
            }
            ev_head = (ev_head + 1) % RING;
        }
    }

    double ms_draft = 0, ms_verify = 0, ms_prefill = 0;
    uint64_t n_draft = 0, n_verify = 0, n_prefill = 0;

    void pool_add(bool small, bool short_span, double ms) {
        if (small) {
            ms_draft += ms; n_draft++;
        } else if (short_span) {
            ms_verify += ms; n_verify++;
        } else {
            ms_prefill += ms; n_prefill++;
        }
    }

    void enter(cudaStream_t stream, int n_nodes) {
        if (!ev_init) {
            for (int i = 0; i < RING; i++) {
                if (cudaEventCreateWithFlags(&ev_a[i], cudaEventDefault) != cudaSuccess ||
                    cudaEventCreateWithFlags(&ev_b[i], cudaEventDefault) != cudaSuccess) {
                    return; // degraded: this ring stays inert
                }
            }
            ev_init = true;
        }
        if ((ev_slot + 1) % RING == ev_head) {
            drain(true); // ring full: bounded sync on the oldest pair, no sample dropped
        }
        cudaEventRecord(ev_a[ev_slot], stream);
        ev_nodes[ev_slot] = n_nodes;
    }

    void exit(cudaStream_t stream) {
        if (!ev_init) {
            return;
        }
        cudaEventRecord(ev_b[ev_slot], stream);
        ev_t0[ev_slot] = now_ms();
        ev_slot = (ev_slot + 1) % RING;
        drain(false);
    }
};

struct ggml_sm70_step_prof_cuda {
    static constexpr int MAX_DEV = 16;

    ggml_sm70_step_prof_ring dev[MAX_DEV];

    static ggml_sm70_step_prof_cuda & get() {
        static ggml_sm70_step_prof_cuda p;
        return p;
    }

    void enter(int device, cudaStream_t stream, int n_nodes) {
        ggml_sm70_step_prof & prof = ggml_sm70_step_prof::get();
        if (!prof.on || device < 0 || device >= MAX_DEV) {
            return;
        }
        if (!prof.cuda_collect) {
            prof.cuda_collect = collect;
        }
        dev[device].enter(stream, n_nodes);
    }

    void exit(int device, cudaStream_t stream) {
        if (device >= 0 && device < MAX_DEV && dev[device].ev_init) {
            dev[device].exit(stream);
        }
    }

    static void collect(
            double & d_ms, uint64_t & d_n,
            double & v_ms, uint64_t & v_n,
            double & p_ms, uint64_t & p_n) {
        ggml_sm70_step_prof_cuda & c = get();
        for (int i = 0; i < MAX_DEV; i++) {
            d_ms += c.dev[i].ms_draft;   d_n += c.dev[i].n_draft;
            v_ms += c.dev[i].ms_verify;  v_n += c.dev[i].n_verify;
            p_ms += c.dev[i].ms_prefill; p_n += c.dev[i].n_prefill;
        }
    }
};
