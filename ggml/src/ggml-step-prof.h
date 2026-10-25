#pragma once
// LLAMA_SM70_STEP_PROF=1 — decode-round budget instrument (P0), host half.
//
// Must compile CUDA-free: ggml-backend-meta.cpp is built without GGML_USE_CUDA.
// The CUDA ring probe (ggml-cuda/step-prof.h) registers a collect hook on its
// first use; this header prints host walls + whatever the hook reports.
//
//   - meta probe (ggml-backend-meta.cpp): host wall of ggml_backend_meta_graph_compute
//   - ctx probe (src/llama-context.cpp): host wall of llama_context::decode
//   - cuda collect hook: GPU busy spans per backend compute call, classified:
//       draft   = small graph (<600 nodes)   — MTP draft passes, glue subgraphs
//       verify  = large graph, span < 200ms  — decode rounds of the full model
//       prefill = large graph, span >= 200ms — prefill ubatches
//
// Prints go to stderr as window deltas every 32 meta-graph computes.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

struct ggml_sm70_step_prof {
    typedef void (*cuda_collect_fn)(
        double & d_ms, uint64_t & d_n,
        double & v_ms, uint64_t & v_n,
        double & p_ms, uint64_t & p_n);

    bool  on    = false;
    bool  tried = false;

    cuda_collect_fn cuda_collect = nullptr;

    uint64_t n_meta       = 0;
    double   meta_wall_ms = 0;

    long long t_print = 0;
    bool      primed  = false;
    double    pmeta   = 0;
    uint64_t  pn_meta = 0;
    double    p_d = 0, p_v = 0, p_p = 0;
    uint64_t  pn_d = 0, pn_v = 0, pn_p = 0;

    static long long now_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    static ggml_sm70_step_prof & get() {
        static ggml_sm70_step_prof p;
        if (!p.tried) {
            p.tried = true;
            const char * e = getenv("LLAMA_SM70_STEP_PROF");
            p.on = e && e[0] == '1';
            if (p.on) {
                // MSVC block-buffers stderr when redirected to a file: without this
                // the budget lines would sit in a 4 KB buffer until process exit
                setvbuf(stderr, nullptr, _IONBF, 0);
            }
        }
        return p;
    }

    void meta_exit() {
        if (!on) {
            return;
        }
        n_meta++;
        if (n_meta % 32 == 0) {
            print("meta");
        }
    }

    void print(const char * tag) {
        double   d_ms = 0, v_ms = 0, p_ms = 0;
        uint64_t d_n = 0, v_n = 0, p_n = 0;
        if (cuda_collect) {
            cuda_collect(d_ms, d_n, v_ms, v_n, p_ms, p_n);
        }
        const long long t = now_ms();
        const double dt_s = t_print ? (t - t_print) / 1000.0 : 0.0;
        const uint64_t dv = v_n - pn_v;
        fprintf(stderr,
                "[step-prof/%s] +%.1fs | verify %llu calls %.1fms (%.2f ms/rd) | draft %llu calls %.1fms | "
                "prefill %llu calls %.1fms | meta %llu calls %.1fms | meta %.3f ms/call\n",
                tag, dt_s,
                (unsigned long long) dv, v_ms - p_v,
                dv ? (v_ms - p_v) / (double) dv : 0.0,
                (unsigned long long)(d_n - pn_d), d_ms - p_d,
                (unsigned long long)(p_n - pn_p), p_ms - p_p,
                (unsigned long long)(n_meta - pn_meta), meta_wall_ms - pmeta,
                n_meta > pn_meta ? (meta_wall_ms - pmeta) / (double)(n_meta - pn_meta) : 0.0);
        if (!primed) {
            primed = true;
        }
        t_print = t;
        p_d = d_ms; p_v = v_ms; p_p = p_ms;
        pn_d = d_n; pn_v = v_n; pn_p = p_n;
        pmeta = meta_wall_ms; pn_meta = n_meta;
    }
};
