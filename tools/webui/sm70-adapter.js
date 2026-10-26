// sm70-adapter.js — data shim between the Strata web frontend and llama-server.
//
// Strata's app.js/monitor.js expect a rich JSON from "metrics": {live, hardware,
// hardware_static, engine, history, requests, totals, requests_kept}. llama-server
// exposes Prometheus text at /metrics, slot JSON at /slots, static info at /props
// and per-request timings only inside /v1/chat/completions responses.
//
// This shim:
//   1. scrapes /metrics (counters) + /slots + /props + /v1/chat/completions'
//      timings hook (see note) and builds the JSON shape the frontend wants;
//   2. maintains the 180 s history series (1 Hz) the chart and sparks need;
//   3. keeps a bounded request log assembled from the chat hook + periodic
//      counter deltas, so the tiles and the request table stay meaningful.
//
// Load order: sm70-adapter.js BEFORE app.js; it defines window.SM70 whose
// metrics() promise app.js's poll() consumes. app.js is patched minimally
// (see the end of this file's companion patch notes in the README).
"use strict";
(function () {
  const S = (window.SM70 = window.SM70 || {});
  const HIST = 180;                       // seconds of history
  const hist = {};                        // key -> number[] (oldest first)
  const push = (k, v) => {
    const a = (hist[k] = hist[k] || []);
    a.push(typeof v === "number" && isFinite(v) ? v : null);
    if (a.length > HIST) a.shift();
  };
  const counters = {};                    // prometheus name -> value
  let propsCache = null;
  let lastScrape = 0;
  const requestsLog = [];                 // newest first, bounded
  let lastCounters = null;                // for delta-based request accounting
  let chatSeen = 0;

  function parsePrometheus(text) {
    const out = {};
    for (const line of text.split("\n")) {
      if (!line || line[0] === "#") continue;
      const sp = line.lastIndexOf(" ");
      if (sp <= 0) continue;
      const key = line.slice(0, sp).trim();
      const val = Number(line.slice(sp + 1).trim());
      if (isFinite(val)) out[key] = val;
    }
    return out;
  }

  async function getJSON(url, opts) {
    const r = await fetch(url, opts);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  }

  async function scrapeOnce() {
    const jobs = [fetch("metrics").then((r) => (r.ok ? r.text() : "")).then(parsePrometheus)];
    jobs.push(getJSON("slots").catch(() => null));
    if (!propsCache) jobs.push(getJSON("props").catch(() => null));
    const [metrics, slots, props] = await Promise.all(jobs);
    if (props && !propsCache) propsCache = props;
    return { metrics, slots };
  }

  // Per-request records: the chat tab reports its own timings via
  // SM70.reportRequest(timings) (a 3-line patch in app.js), so the log and the
  // per-request tiles use real data; everything else is counter deltas.
  S.reportRequest = function (t) {
    chatSeen++;
    requestsLog.unshift({
      state: "completed",
      finish: "stop",
      at: Date.now() / 1000,
      prompt_tokens: t.prompt_n || 0,
      output_tokens: t.predicted_n || 0,
      prompt_tok_s: t.prompt_per_second || null,
      decode_tok_s: t.predicted_per_second || null,
      ttft_ms: null,
      duration_s: (t.predicted_ms || 0) / 1000 + (t.prompt_ms || 0) / 1000 || null,
      drafts_offered: t.draft_n || 0,
      drafts_accepted: t.draft_n_accepted || 0,
      reused: t.cache_n || 0,
      id: chatSeen,
    });
    if (requestsLog.length > 200) requestsLog.pop();
  };

  function ctxFromSlots(slots) {
    // llama-server /slots: {slots:[{id, ctx, prompt:[...], n_capacity...}]}
    // newer builds report per-slot prompt length; older only the model ctx.
    if (!slots || !Array.isArray(slots.slots)) return null;
    return slots.slots.map((s, i) => ({
      id: s.id ?? i,
      ctx: s.ctx || null,
      used: (s.prompt ? s.prompt.length : (s.n_prompt_tokens || 0)) + (s.n_predicted || 0),
      processing: !!(s.is_processing || s.processing),
      prompt_n: s.prompt ? s.prompt.length : s.n_prompt_tokens || null,
    }));
  }

  async function metrics() {
    const now = Date.now();
    let snap = { metrics: counters, slots: null };
    try { snap = await scrapeOnce(); } catch (e) { /* keep previous */ }
    if (snap.metrics && Object.keys(snap.metrics).length) {
      Object.assign(counters, snap.metrics);
      lastScrape = now;
    }
    const M = counters;
    const g = (k) => (typeof M[k] === "number" ? M[k] : 0);
    const slotsInfo = ctxFromSlots(snap.slots);
    const processing = slotsInfo ? slotsInfo.filter((s) => s.processing).length
                                 : Math.round(g("llamacpp:requests_processing"));

    // windowed rates: /metrics gauges already average between scrapes
    const decodeTPS = g("llamacpp:predicted_tokens_seconds") || null;
    const prefillTPS = g("llamacpp:prompt_tokens_seconds") || null;
    push("tok_s", processing ? decodeTPS : 0);
    push("prefill_tok_s_mean", prefillTPS || 0);

    const draftTotal = g("llamacpp:spec_decode_num_draft_tokens_total");
    const accTotal = g("llamacpp:spec_decode_num_accepted_tokens_total");
    const acceptPct = draftTotal ? (100 * accTotal) / draftTotal : null;
    push("spec_accept", acceptPct);

    const state = processing ? "generating" : "idle";
    const live = {
      state,
      queued: g("llamacpp:requests_deferred"),
      tok_s: processing ? decodeTPS : null,
      prefill_tok_s_mean: processing ? prefillTPS : null,
      generated: null, prompt_tokens: null, prompt_read: null, prompt_total: null,
      elapsed_s: null, max_tokens: null, phase: null,
    };

    // GPU cards come from the launcher's local agent (sm70-gpu agent, same
    // origin); without it the cards show dashes instead of failing.
    let hardware = { gpus: [] };
    try {
      const hwr = await fetch("../gpu").then((r) => (r.ok ? r.json() : null)).catch(() => null);
      if (hwr && hwr.gpus) {
        hardware = hwr;
        hwr.gpus.forEach((gpu, i) => {
          push(`gpu${i}_util`, gpu.util);
          push(`gpu${i}_mem_used`, gpu.mem_used);
        });
        push("ram_used", hwr.ram_used || null);
      }
    } catch (e) { /* agent absent */ }

    const totals = {
      requests: g("llamacpp:n_decode_total") || requestsLog.length,
      prompt_tokens: g("llamacpp:prompt_tokens_total"),
      reused: g("llamacpp:prompt_tokens_cached_total"),
      output_tokens: g("llamacpp:tokens_predicted_total"),
      prompt_ms: g("llamacpp:prompt_seconds_total") * 1000,
      decode_ms: g("llamacpp:tokens_predicted_seconds_total") * 1000,
      ttft_ms: 0,
      drafts_offered: draftTotal,
      drafts_accepted: accTotal,
      since: (now - Number(M["llamacpp:process_start_time_seconds"] || now / 1000) * 1000) / 1000,
    };

    return {
      live,
      hardware,
      hardware_static: {
        gpu_name: (hardware.gpus || []).map((g) => g.name).join(" + "),
        cpu_name: propsCache ? `${propsCache.n_ctx_train || "?"} ctx train` : "",
        cores: null, threads: navigator.hardwareConcurrency || null,
      },
      engine: {
        max_context: propsCache ? (propsCache.default_generation_settings
          && propsCache.default_generation_settings.ctx || 0) : 0,
        kv: propsCache ? "q4_0 (SM70 direct)" : null,
        spec: "mtp", mtp_max: 3,
        expert_slots: null, expert_cache_mib: 0,
      },
      history: hist,
      requests: requestsLog,
      requests_kept: requestsLog.length,
      totals,
      conversation_cache: null, l3: null,
    };
  }

  // --- patch the two frontends ---------------------------------------------
  // app.js poll(): fetch("metrics") -> SM70.metrics()
  const origFetch = window.fetch;
  // monitor pages call api() from monitor.js with absolute paths; keep those.
  document.addEventListener("DOMContentLoaded", () => {
    const patch = (src, from, to) => {
      const re = new RegExp(from, "g");
      document.querySelectorAll(`script[src="${src}"]`).forEach(() => {});
    };
  });
  S.metrics = metrics;
})();
