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
  let propsAt = 0;
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
    const now = Date.now();
    const jobs = [fetch("metrics").then((r) => (r.ok ? r.text() : "")).then(parsePrometheus)];
    jobs.push(getJSON("slots").catch(() => null));
    // /props was cached once for the life of the page, so a restart (another model,
    // another context, an mmproj) left the About card describing the old server.
    // Re-read it now and then; the page redraws from the fresh copy every tick.
    const wantProps = !propsCache || now - propsAt > 5000;
    if (wantProps) jobs.push(getJSON("props").catch(() => null));
    const [metrics, slots, props] = await Promise.all(jobs);
    if (props) { propsCache = props; propsAt = now; }
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
      time: Date.now() / 1000,
      prompt_tokens: t.prompt_n || 0,
      output_tokens: t.predicted_n || 0,
      prompt_tok_s: t.prompt_per_second || null,
      decode_tok_s: t.predicted_per_second || null,
      ttft_ms: t.ttft_ms || null,
      duration_s: ((t.predicted_ms || 0) + (t.prompt_ms || 0)) / 1000 || null,
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
    const list = Array.isArray(slots) ? slots : (slots && Array.isArray(slots.slots) ? slots.slots : null);
    if (!list) return null;
    // field names are the ones server_slot::to_json actually emits: id, n_ctx,
    // is_processing, n_prompt_tokens, n_prompt_tokens_processed, n_prompt_tokens_cache,
    // next_token[0].n_decoded. `prompt` only exists with LLAMA_SERVER_SLOTS_DEBUG and is
    // a detokenized STRING (its length is characters, not tokens).
    return list.map((s, i) => {
      const dec = s.next_token && s.next_token[0] ? (s.next_token[0].n_decoded || 0) : 0;
      const pn = s.n_prompt_tokens || 0;
      return {
        id: s.id ?? i,
        ctx: s.n_ctx || null,
        used: pn + dec,
        processing: !!(s.is_processing || s.processing),
        prompt_n: pn || null,
        prompt_read: s.n_prompt_tokens_processed || 0,
        n_decoded: dec,
      };
    });
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

    // live rates. The /metrics gauges/counter buckets only settle when the
    // server flushes them, so during processing diff /slots' per-slot
    // n_decoded (live per request) at the scrape interval.
    let decodeTPS = g("llamacpp:predicted_tokens_seconds") || null;
    let prefillTPS = g("llamacpp:prompt_tokens_seconds") || null;
    if (processing && slotsInfo && slotsInfo.length) {
      const decoded = slotsInfo.reduce((a, s2) => a + (s2.n_decoded || 0), 0);
      if (S._lastDecoded != null && decoded >= S._lastDecoded) {
        const dt = Math.max(0.25, (now - (S._lastScrapeTs || now - 1000)) / 1000);
        decodeTPS = (decoded - S._lastDecoded) / dt;
      }
      S._lastDecoded = decoded;
      S._lastScrapeTs = now;
    } else {
      S._lastDecoded = null;
    }
    push("tok_s", processing ? decodeTPS : 0);
    push("prefill_tok_s_mean", prefillTPS || 0);

    const draftTotal = g("llamacpp:spec_decode_num_draft_tokens_total");
    const accTotal = g("llamacpp:spec_decode_num_accepted_tokens_total");
    const acceptPct = draftTotal ? (100 * accTotal) / draftTotal : null;
    push("spec_accept", acceptPct);

    const prefillLeft = slotsInfo && slotsInfo.some((s2) => s2.processing && s2.prompt_read < (s2.prompt_n || 0));
    const state = prefillLeft ? "reading" : processing ? "generating" : "idle";
    const live = {
      state,
      queued: g("llamacpp:requests_deferred"),
      tok_s: processing ? decodeTPS : null,
      prefill_tok_s_mean: processing ? prefillTPS : null,
      generated: null, prompt_tokens: null, prompt_read: null, prompt_total: null,
      elapsed_s: null, max_tokens: null, phase: null,
    };
    if (slotsInfo && slotsInfo.length) {
      const act = slotsInfo.find((s2) => s2.processing);
      if (act) {
        live.prompt_tokens = act.prompt_n || null;
        live.prompt_total = act.prompt_n || null;
        live.prompt_read = act.prompt_read != null ? act.prompt_read : (act.prompt_n || null);
        live.generated = act.n_decoded || null;
      }
    }

    // GPU cards come from the launcher's local agent (sm70-gpu agent, same
    // origin); without it the cards show dashes instead of failing.
    let hardware = { gpus: [] };
    let hostThreads = null;
    let hostCpuName = "";
    try {
      const [hwr, srv, host] = await Promise.all([
        fetch("../gpu").then((r) => (r.ok ? r.json() : null)).catch(() => null),
        fetch("../server").then((r) => (r.ok ? r.json() : null)).catch(() => null),
        fetch("../host").then((r) => (r.ok ? r.json() : null)).catch(() => null),
      ]);
      if (host) {
        hostThreads = host.threads;
        hostCpuName = host.cpu_name || "";
        hardware.cpu = host.cpu;
        hardware.ram_used = host.ram_used; hardware.ram_total = host.ram_total;
        hardware.disk_read_mb = host.disk_read_mb; hardware.disk_write_mb = host.disk_write_mb;
        push("cpu", host.cpu);
        push("ram_used", host.ram_used);
        push("disk_read", host.disk_read_mb ?? null);
        push("disk_write", host.disk_write_mb ?? null);
      }
      if (hwr && hwr.gpus) {
        // show only the GPUs llama-server actually uses: CUDA_VISIBLE_DEVICES
        // from the launcher's recorded command line environment. Without that
        // (server started by hand), fall back to filtering out GPUs with less
        // than 24 GB (gaming cards) — never silently show a card doing nothing.
        let keep = null;
        const cvd = srv && srv.env && srv.env.CUDA_VISIBLE_DEVICES;
        if (cvd != null && cvd !== "") {
          keep = new Set(cvd.split(",").map((x) => parseInt(x.trim(), 10)));
        }
        // keep is in CUDA order; the /gpu payload now carries cuda_index for each card
        // (nvidia-smi's numbering differs on this box, so matching on `index` picked the
        // wrong cards).
        const idxOf = (g) => (g.cuda_index != null ? g.cuda_index : g.index);
        const gpus = hwr.gpus.filter((g) =>
          keep ? keep.has(idxOf(g)) : (g.mem_total || 0) >= 24 * 1073741824)
          .map((g, i) => ({ ...g, index: i }));
        hardware = { ...hardware, gpus, ram_used: hardware.ram_used ?? hwr.ram_used,
                     ram_total: hardware.ram_total ?? hwr.ram_total };
        gpus.forEach((gpu, i) => {
          push(`gpu${i}_util`, gpu.util);
          push(`gpu${i}_mem_used`, gpu.mem_used);
          push(`gpu${i}_temp`, gpu.temp ?? null);
          push(`gpu${i}_power`, gpu.power ?? null);
        });
        push("disk_read", hwr.disk_read_mb ?? null);
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
      decode_steps: g("llamacpp:n_decode_total"),
      drafts_accepted: accTotal,
      since: (now - Number(M["llamacpp:process_start_time_seconds"] || now / 1000) * 1000) / 1000,
    };

    return {
      live,
      hardware,
      hardware_static: {
        gpu_name: (hardware.gpus || []).map((g) => g.name).join(" + "),
        cpu_name: hostCpuName,
        cores: hostThreads, threads: hostThreads,
      },
      engine: {
        // the running server's own name: --alias when set, else the file name
        model: propsCache
          ? (propsCache.model_alias
             || String(propsCache.model_path || "").split(String.fromCharCode(92)).join("/").split("/").pop()
             || null)
          : null,
        max_context: propsCache ? ((propsCache.default_generation_settings || {}).n_ctx || 0) : 0,
        version: propsCache ? (propsCache.build_info || null) : null,
        kv: propsCache ? "q4_0 (SM70 direct)" : null,
        spec: "mtp", mtp_max: 3,
        images: !!(propsCache && propsCache.modalities && propsCache.modalities.vision),
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
