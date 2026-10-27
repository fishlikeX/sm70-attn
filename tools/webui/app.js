// serve/web/app.js - the Strata web app: Chat, Monitor, About. No framework, no network beyond this server.
// The Monitor tab rebuilds PR #22's dashboard idea (code-martin) on the server's own /metrics.
"use strict";

const $ = (id) => document.getElementById(id);
const SPRITE = "web/sprite.svg";
const icon = (name, cls = "st-icon") => `<svg class="${cls}" aria-hidden="true"><use href="${SPRITE}#i-${name}"/></svg>`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const fmt = (n, d = 0) => (n == null || Number.isNaN(n) ? "–" : Number(n).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}));
const kfmt = (n) => (n == null ? "–" : n >= 1000 ? `${fmt(n / 1000, n >= 10000 ? 0 : 1)}k` : fmt(n));
// a context size: 32768 -> "32K" (powers of two), else like kfmt
const ctxfmt = (n) => (n && n % 1024 === 0 ? `${fmt(n / 1024)}K` : kfmt(n));
const gb = (b, d = 1) => (b == null ? "–" : fmt(b / 1073741824, d));   // memory: binary GB, as Windows shows it
// a card's energy used: Wh, in kWh once it is worth it (a V100 at 45 W reaches 1 kWh in a day)
const wh = (v) => (v == null ? "–" : v >= 1000 ? `${fmt(v / 1000, 2)} kWh` : `${fmt(v, v < 10 ? 1 : 0)} Wh`);

// The launcher is the single source of truth for the chat, the sampler, the key,
// the theme and the language (sm70gui-ui.json beside its settings), so this window,
// Edge and any other client on the PC share one copy. localStorage is only the
// local cache that makes the first paint instant; syncUiState() at the bottom of
// this file adopts the launcher's copy and set() mirrors changes back to it.
const store = {
  ready: false,            // false until the launcher's copy has been read/written
  timers: {},
  get(k, d) { try { const v = localStorage.getItem("strata." + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
  cache(k, v) { try { localStorage.setItem("strata." + k, JSON.stringify(v)); } catch (e) { /* in memory only */ } },
  set(k, v) { this.cache(k, v); this.push(k, v); },
  push(k, v) {
    if (!this.ready) return;             // before the first sync: never clobber the file
    clearTimeout(this.timers[k]);
    this.timers[k] = setTimeout(async () => {
      try {
        const r = await fetch("ui-state", {method: "POST", headers: {"Content-Type": "application/json"},
                                           body: JSON.stringify({[k]: v})});
        if (r.ok) { const j = await r.json(); if (j.at) uiStateAt = j.at; }
      } catch (e) { /* launcher gone: stay local */ }
    }, 600);
  },
};
let uiStateAt = 0;

// ------------------------------------------------------------------ toasts
function toast(kind, title, text = "", ms = 3500, action = null) {
  const names = {info: "info", success: "check", warn: "warning", error: "error"};
  const el = document.createElement("div");
  el.className = `st-toast st-toast--${kind}`;
  el.innerHTML = `${icon(names[kind] || "info")}<div><div class="st-toast__title"></div><div class="t-text"></div></div>`;
  el.querySelector(".st-toast__title").textContent = title;
  el.querySelector(".t-text").textContent = text;
  if (action) {
    const b = document.createElement("button");
    b.className = "st-btn st-btn--secondary";
    b.style.height = "32px";
    b.style.marginLeft = "auto";
    b.textContent = action.label;
    b.onclick = () => { action.run(); el.remove(); };
    el.appendChild(b);
  }
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), ms);
}

async function copyText(text, btn) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (e) {                                   // http on another host: no async clipboard
    const ta = document.createElement("textarea");
    ta.value = text; document.body.appendChild(ta); ta.select(); document.execCommand("copy"); ta.remove();
  }
  if (btn) {
    const use = btn.querySelector("use");
    use.setAttribute("href", `${SPRITE}#i-check`);
    setTimeout(() => use.setAttribute("href", `${SPRITE}#i-copy`), 1500);
  }
  toast("success", "Copied to clipboard", "", 1800);
}

// ------------------------------------------------------------------ theme and tabs
// the system's theme until the user picks one (only a click is saved)
function setTheme(t, save) {
  document.documentElement.dataset.theme = t;
  if (save) try { localStorage.setItem("strata.theme", t); } catch (e) { /* ignore */ }
  $("theme-icon").setAttribute("href", `${SPRITE}#i-${t === "dark" ? "sun" : "moon"}`);
  $("dark-toggle").setAttribute("aria-checked", String(t === "dark"));
}
const flipTheme = () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark", true);
$("theme-btn").onclick = flipTheme;
$("dark-toggle").onclick = flipTheme;
setTheme(document.documentElement.dataset.theme || "light", false);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
  let saved = null;
  try { saved = localStorage.getItem("strata.theme"); } catch (err) { /* ignore */ }
  if (!saved) setTheme(e.matches ? "dark" : "light", false);
});

let tab = "chat";
function showTab(name) {
  if (tab === "launch" && name !== "launch" && window.SM70Launch && window.SM70Launch.flush) {
    window.SM70Launch.flush();     // a 500 ms debounce must not lag behind the other views
  }
  tab = ["chat", "images", "monitor", "launch", "about"].includes(name) ? name : "chat";
  for (const b of document.querySelectorAll(".st-tab")) b.setAttribute("aria-selected", String(b.dataset.tab === tab));
  for (const v of ["chat", "images", "monitor", "launch", "about"]) $(`view-${v}`).hidden = v !== tab;
  if (tab === "images") $("img-prompt").focus();
  if (location.hash.slice(1) !== tab) history.replaceState(null, "", tab === "chat" ? location.pathname : `#${tab}`);
  if (tab === "chat") $("input").focus();
  if (tab === "monitor") loadMcp();
  if (tab === "launch" && window.SM70Launch) window.SM70Launch.show();
  if (tab === "about") { loadConfig(); loadSelection(); loadServerInfo(); }
  if (lastMetrics) render(lastMetrics);
}
for (const b of document.querySelectorAll(".st-tab")) b.onclick = () => showTab(b.dataset.tab);
window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));

// ------------------------------------------------------------------ server access
function headers(json = false) {
  const h = {};
  const key = store.get("apikey", "");
  if (key) h.Authorization = "Bearer " + key;
  if (json) h["Content-Type"] = "application/json";
  return h;
}
$("api-key").value = store.get("apikey", "");
$("api-key").onchange = () => { store.set("apikey", $("api-key").value.trim()); toast("success", "API key saved", "Shared by every client on this PC."); };

let health = {model: "sm70-attn", images: false, max_context: 0};
async function loadHealth() {
  try {
    const h = await (await fetch("health")).json();
    if (h.status !== "ok") throw new Error("not ok");
    try {
      const props = await (await fetch("props")).json();
      health.model = props.model_path ? baseName(props.model_path) : health.model;
      health.max_context = props.default_generation_settings?.n_ctx || 0;
      // /props.modalities is the only truthful answer to "can this server take a
      // picture": it is on exactly when an mmproj loaded. Before this the flag
      // stayed false forever, so the composer refused images (and Ctrl+V did
      // nothing) even with a vision head running.
      health.images = !!(props.modalities && props.modalities.vision);
    } catch (e) { /* props may be off; keep defaults */ }
    $("attach-btn").title = health.images ? "Attach a text file or a picture (or drop it here)"
                                          : "Attach a text file (or drop it here)";
    $("chat-empty-sub").textContent = health.model && health.model !== "sm70-attn"
      ? `${health.model} · local, nothing leaves this PC`
      : "Local inference · nothing leaves this PC";
    $("tab-btn-images").hidden = !health.image_generation;
    if (location.hash === "#images" && health.image_generation) showTab("images");
  } catch (e) {
    setTimeout(loadHealth, 2000);
  }
}
// A restart through the launcher can add (or drop) the vision head while this page
// stays open, and the form can move --port; re-read the facts while visible instead
// of requiring a reload.
setInterval(() => {
  if (document.hidden || busy) return;
  loadHealth();
  if (tab === "about") { loadServerInfo(); loadSelection(); }
}, 5000);

// ------------------------------------------------------------------ Monitor
// The Monitor tab, rendered from GET /metrics: the engine's own numbers, the hardware sampler's one-second series
// (each card and the total), and the finished requests.  A reading that is absent stays "–"; nothing here may throw.

let lastMetrics = null, metricsFailures = 0, keyWarned = false, mcpTick = 0;
let reqShowAll = false;   // the request log: the last 12, or every one the server keeps (issue #35)

const setText = (id, value) => { const el = $(id); if (el && el.textContent !== String(value)) el.textContent = value; };
const mean = (values) => {
  const v = values.filter((x) => typeof x === "number" && !Number.isNaN(x));
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
};
function median(values) {
  const v = values.filter((x) => typeof x === "number" && !Number.isNaN(x)).sort((a, b) => a - b);
  if (!v.length) return null;
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}
const dur = (v, d = 2) => (v == null ? "–" : v >= 1000 ? `${fmt(v / 1000, d)} s` : `${fmt(v, 0)} ms`);
const rate = (v, d = 1) => (v == null ? "–" : `${fmt(v, d)} t/s`);
const pctText = (v, d = 0) => (v == null ? "–" : `${fmt(v, d)}%`);
// the prompt reading speed of one finished request: the tokens it read, the cached prefix excluded
const reqPrefill = (r) => (r && r.prompt_ms > 0 ? Math.max(0, (r.prompt_tokens || 0) - (r.reused || 0)) / (r.prompt_ms / 1000) : null);

function sparkPaths(values, max) {
  const v = (values || []).map((x) => (x == null ? 0 : x));
  if (v.length < 2) return {line: "", area: ""};
  const top = Math.max(max || 0, ...v, 1);
  const pts = v.map((x, i) => [(i / (v.length - 1)) * 100, 30 - (x / top) * 26]);
  const line = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join("");
  return {line, area: `${line}L100,32L0,32Z`};
}
function sparkSvg(cls, path, tone) {
  return `<svg class="${cls}" viewBox="0 0 100 32" preserveAspectRatio="none" aria-hidden="true"${tone ? ` data-tone="${tone}"` : ""}>` +
    `<path class="area" fill="currentColor" opacity=".12" d="${path.area}"/>` +
    `<path class="line" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"` +
    ` vector-effect="non-scaling-stroke" d="${path.line}"/></svg>`;
}
function spark(id, values, max) {
  const svg = $(id);
  if (!svg) return;
  const p = sparkPaths(values, max);
  svg.querySelector(".line").setAttribute("d", p.line);
  svg.querySelector(".area").setAttribute("d", p.area);
}

// ---- the throughput chart: a line chart sized to its element, so the axis text stays crisp
function axisAge(seconds) {
  if (seconds < 1) return "now";
  if (seconds < 90) return `-${Math.round(seconds)}s`;
  return `-${fmt(seconds / 60, seconds < 300 ? 1 : 0)}m`;
}
function chart(svg, series, max) {
  const W = Math.max(260, Math.round(svg.clientWidth || 720));
  const H = Math.max(90, Math.round(svg.clientHeight || 190));
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  // a series may name the axis it scales to: "left" (default) or "right", so two
  // series on different orders of magnitude each keep a readable curve
  const hasRight = series.some((s) => s.axis === "right");
  const padL = 46, padR = hasRight ? 46 : 10, padT = 8, padB = 18;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const n = Math.max(0, ...series.map((s) => s.values.length));
  const axisTop = (side) => {
    const flat = series.filter((s) => (s.axis || "left") === side)
                       .flatMap((s) => s.values).filter((v) => typeof v === "number");
    return Math.max(side === "left" ? max || 0 : 0, ...flat, 1e-9) * 1.06;
  };
  const topL = axisTop("left"), topR = axisTop("right");
  const px = (i) => padL + (n <= 1 ? plotW : (i / (n - 1)) * plotW);
  const pyOn = (top) => (v) => padT + plotH - (Math.max(0, Math.min(v, top)) / top) * plotH;
  const py = pyOn(topL);
  const base = (padT + plotH).toFixed(1);
  let out = "";
  const axisColor = (side) => {
    const s = series.find((x) => (x.axis || "left") === side && x.color);
    return s ? ` style="fill:${s.color}"` : "";
  };
  for (let t = 0; t <= 4; t++) {
    const yy = py((topL / 4) * t);
    out += `<line class="grid" x1="${padL}" y1="${yy.toFixed(1)}" x2="${(W - padR).toFixed(1)}" y2="${yy.toFixed(1)}"/>` +
           `<text class="axis"${axisColor("left")} x="${padL - 7}" y="${(yy + 4).toFixed(1)}" text-anchor="end">${fmt((topL / 4) * t, topL >= 100 ? 0 : 1)}</text>`;
    if (hasRight) {
      const yr = pyOn(topR)((topR / 4) * t);
      out += `<text class="axis"${axisColor("right")} x="${(W - padR + 7).toFixed(1)}" y="${(yr + 4).toFixed(1)}" text-anchor="start">${fmt((topR / 4) * t, topR >= 100 ? 0 : 1)}</text>`;
    }
  }
  series.forEach((s, si) => {
    const pyS = pyOn((s.axis || "left") === "right" ? topR : topL);
    const pts = s.values.map((v, i) => (typeof v === "number" ? [px(i), pyS(v)] : null)).filter(Boolean);
    if (pts.length < 2) return;
    const d = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("");
    if (si === 0) {
      out += `<path class="chart-area" d="${d}L${pts[pts.length - 1][0].toFixed(1)},${base}L${pts[0][0].toFixed(1)},${base}Z"/>`;
    }
    out += `<path class="chart-line" style="stroke:${s.color}" d="${d}"/>`;
  });
  for (const f of [0, 0.25, 0.5, 0.75, 1]) {
    const i = f * (n - 1);
    out += `<text class="axis" x="${px(i).toFixed(1)}" y="${H - 4}" text-anchor="${f === 0 ? "start" : f === 1 ? "end" : "middle"}">` +
           `${axisAge(n - 1 - i)}</text>`;
  }
  svg.innerHTML = out;
}

// ---- the KPI tiles: session averages, with the recent window beside them
const KPIS = [
  {key: "decode", label: "Decode", icon: "gauge", spark: true},
  {key: "prefill", label: "Prefill", icon: "download", spark: true},
  {key: "ttft", label: "Time to first token", icon: "clock", spark: true},
  {key: "duration", label: "Request time", icon: "activity"},
  {key: "requests", label: "Requests", icon: "layers"},
  {key: "tokens", label: "Tokens written", icon: "memory"},
  {key: "hit", label: "Prompt cache hit", icon: "experts"},
  {key: "spec", label: "Draft acceptance", icon: "bolt"},
];
$("kpis").innerHTML = KPIS.map((k) => `
  <div class="kpi">
    <span class="kpi__label">${icon(k.icon, "st-icon st-icon--sm")}${esc(k.label)}</span>
    <span class="kpi__value" id="kpi-${k.key}">–</span>
    <span class="kpi__sub" id="kpisub-${k.key}"></span>
    ${k.spark ? `<svg class="kpi__spark" id="kpispark-${k.key}" viewBox="0 0 100 32" preserveAspectRatio="none" aria-hidden="true">
      <path class="area" fill="currentColor" opacity=".12"/><path class="line" fill="none" stroke="currentColor"
      stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/></svg>` : ""}
  </div>`).join("");
function setKpi(key, value, sub) {
  const el = $(`kpi-${key}`);
  if (el.innerHTML !== value) el.innerHTML = value;
  setText(`kpisub-${key}`, sub || "");
}
const kpiValue = (text, unit) => (text === "–" ? "–" : `${esc(text)}${unit ? `<small>${esc(unit)}</small>` : ""}`);

// ---- the per-GPU breakout: one card per card of the run, each with its own series
function gpuCards(hw, h, st) {
  const cards = hw.gpus || [];
  const names = String(st.gpu_name || "").split(" + ");
  const cardName = (i) => names[i] || (names.length === 1 ? names[0] : "");
  const block = (label, value, note, path, bar, tone) => `<div class="gpu-block">
    <div class="gpu-block__top"><span>${esc(label)}</span><span><b>${value}</b>${note ? ` <em>${esc(note)}</em>` : ""}</span></div>
    ${bar == null ? "" : `<div class="st-progress"${tone ? ` data-tone="${tone}"` : ""}><div class="st-progress__bar" style="width:${Math.max(0, Math.min(100, bar)).toFixed(1)}%"></div></div>`}
    ${path ? sparkSvg("gpu-spark", path, tone) : ""}
  </div>`;
  return cards.map((g) => {
    const i = g.index;
    const memPct = g.mem_used != null && g.mem_total ? (100 * g.mem_used) / g.mem_total : null;
    const pwPct = g.power != null && g.power_limit ? (100 * g.power) / g.power_limit : null;

    return `<div class="st-card gpu-card">
      <div class="gpu-card__head"><span class="gpu-card__index">GPU ${esc(i)}</span>
        <span class="gpu-card__name" title="${esc(cardName(i))}">${esc(cardName(i))}</span></div>
      ${block("Load", pctText(g.util), "", sparkPaths(h[`gpu${i}_util`], 100), g.util, "warn")}
      ${block("VRAM", g.mem_used == null ? "–" : `${gb(g.mem_used)} GB`, g.mem_total ? `of ${gb(g.mem_total, 0)} GB` : "",
              sparkPaths(h[`gpu${i}_mem_used`], g.mem_total), memPct, null)}
      ${block("Temperature", g.temp == null ? "–" : `${fmt(g.temp)} °C`, "",
              sparkPaths(h[`gpu${i}_temp`], 90), g.temp == null ? null : Math.min(100, g.temp), "warn")}
      ${block("Power", g.power == null ? "–" : `${fmt(g.power)} W`, g.power_limit ? `of ${fmt(g.power_limit)} W` : "",
              sparkPaths(h[`gpu${i}_power`], g.power_limit), pwPct, null)}
    </div>`;
  }).join("");
}

// ---- the machine card and the L3 card share one tile: label, value, a sub-line, an optional bar and series
const lastOf = (series) => {
  const v = (series || []).filter((x) => typeof x === "number");
  return v.length ? v[v.length - 1] : null;
};
const mbRate = (v) => (v == null ? "–" : v >= 1000 ? `${fmt(v / 1024, 2)}<small>GB/s</small>`
                                                  : `${fmt(v, v < 1 ? 2 : v < 10 ? 1 : 0)}<small>MB/s</small>`);
const plural = (n, word) => `${fmt(n)} ${word}${n === 1 ? "" : "s"}`;
function metricTile(label, iconName, value, sub, bar, tone, path) {
  return `<div class="sys-metric">
    <span class="sys-metric__label">${icon(iconName, "st-icon st-icon--sm")}${esc(label)}</span>
    <span class="sys-metric__value">${value}</span>
    <span class="sys-metric__sub">${esc(sub || "")}</span>
    ${bar == null ? "" : `<div class="st-progress"${tone ? ` data-tone="${tone}"` : ""}><div class="st-progress__bar" style="width:${Math.max(0, Math.min(100, bar)).toFixed(1)}%"></div></div>`}
    ${path ? sparkSvg("sys-metric__spark", path) : ""}
  </div>`;
}

function sysMetrics(hw, st, h) {
  const ramPct = hw.ram_total ? (100 * hw.ram_used) / hw.ram_total : null;
  return [
    metricTile("CPU load", "cpu", hw.cpu == null ? "–" : `${fmt(hw.cpu)}<small>%</small>`,
               st.threads ? `${fmt(st.threads)} threads` : "",
               hw.cpu, hw.cpu != null && hw.cpu > 90 ? "danger" : null, sparkPaths(h.cpu, 100)),
    metricTile("System RAM", "memory", hw.ram_used == null ? "–" : `${gb(hw.ram_used)}<small>GB</small>`,
               hw.ram_total ? `of ${gb(hw.ram_total, 0)} GB` : "", ramPct,
               ramPct != null && ramPct > 92 ? "danger" : null, sparkPaths(h.ram_used, hw.ram_total)),
    metricTile("Disk read", "disk", mbRate(hw.disk_read_mb), "", null, null, sparkPaths(h.disk_read_mb)),
    metricTile("Disk write", "disk", mbRate(hw.disk_write_mb), "", null, null, sparkPaths(h.disk_write_mb)),
  ].join("");
}

// ---- the L3 conversation disk cache: parked conversations kept as records on disk, resumed by a later request
function l3Metrics(l3, h) {
  const hits = l3.hits || 0, misses = l3.misses || 0;
  const hitPct = hits + misses ? (100 * hits) / (hits + misses) : null;
  const used = l3.budget_bytes ? (100 * (l3.bytes || 0)) / l3.budget_bytes : null;
  const avg = (total, n) => (n ? (total || 0) / n : null);
  return [
    metricTile("Store", "layers", l3.bytes == null ? "–" : `${gb(l3.bytes)}<small>GB</small>`,
               `${plural(l3.records || 0, "record")}${l3.budget_bytes ? ` of ${gb(l3.budget_bytes, 0)} GB` : ""}`,
               used, used != null && used > 90 ? "warn" : null, null),
    metricTile("Reuse", "experts", pctText(hitPct, 1), `${plural(hits, "hit")} · ${plural(misses, "miss")}`,
               null, null, null),
    metricTile("Parked", "download", `${gb(l3.parked_bytes)}<small>GB</small>`,
               `${plural(l3.parks, "park")} · ${dur(avg(l3.park_ms, l3.parks))} each`, null, null, null),
    metricTile("Restored", "refresh", kfmt(l3.restored_tokens || 0),
               `${plural(l3.restores, "restore")} · ${dur(avg(l3.restore_ms, l3.restores))} each`, null, null, null),
    metricTile("Disk read", "disk", mbRate(lastOf(h.l3_read_mb)),
               `last record read in ${dur(l3.last_read_ms)}`, null, null, sparkPaths(h.l3_read_mb)),
    metricTile("Disk write", "disk", mbRate(lastOf(h.l3_write_mb)),
               `last park in ${dur(l3.last_park_ms)}`, null, null, sparkPaths(h.l3_write_mb)),
  ].join("");
}

async function poll() {
  try {
    const r = {ok: true, status: 200, json: async () => await (window.SM70 ? SM70.metrics() : {})};
    if (r.status === 401) {
      setPill("error", "API key needed");
      if (!keyWarned) { keyWarned = true; toast("warn", "API key needed", "This server needs a key: add it under About > Settings.", 6000); }
    } else if (r.ok) {
      lastMetrics = await r.json();
      metricsFailures = 0;
      render(lastMetrics);
    } else {
      throw new Error(`HTTP ${r.status}`);
    }
  } catch (e) {
    if (++metricsFailures === 3) setPill("error", "Server not reachable");
  }
  if (tab === "monitor" && ++mcpTick % 10 === 0) loadMcp();       // server states change rarely: every 10 s
  if (tab === "about") loadSelection();   // /selection is a prefs read (~1 ms); keeps About live
  setTimeout(poll, 1000);
}

function setPill(state, text) {
  $("pill").dataset.state = state === "error" ? "queued" : state;
  $("pill-text").textContent = text;
}

function render(m) {
  const live = m.live || {}, hw = m.hardware || {}, st = m.hardware_static || {}, eng = m.engine || {};
  // the header pill
  if (live.state === "reading") {
    const pct = live.prompt_total ? Math.round((100 * live.prompt_read) / live.prompt_total) : null;
    setPill("reading", pct != null ? `Reading prompt · ${pct}%` : "Reading prompt");
  } else if (live.state === "generating") {
    setPill("generating", `Generating · ${fmt(live.tok_s, 1)} tok/s`);
  } else {
    setPill("idle", "Idle");
  }
  if (live.queued > 0) setPill("queued", `${live.queued} queued`);
  if (tab === "monitor") renderMonitor(m);
  if (tab === "monitor") renderConvCache(m.conversation_cache);
  if (tab === "about") renderAbout(eng, hw, st);
}
// the chart is sized from its element: a window resize redraws it before the next poll
window.addEventListener("resize", () => { if (lastMetrics && tab === "monitor") render(lastMetrics); });

// #596: the conversation cache - the prompt's state the engine keeps between requests (always), and the whole
// conversations it parks in RAM when "--conversation-cache-mib N" is in the run config's args (opt-in)
function since(t) {
  if (!t) return "";
  const s = Math.max(0, Date.now() / 1000 - t);
  return s < 60 ? "just now" : s < 3600 ? `${fmt(s / 60)} min ago` : `${fmt(s / 3600, 1)} h ago`;
}
function renderConvCache(c) {
  $("cc-card").hidden = !c;
  if (!c) return;                                  // an older server
  const pct = (a, b) => (b ? `${Math.min(100, (100 * a) / b)}%` : "0%");
  $("cc-bars").hidden = !c.enabled;
  if (c.enabled) {
    $("cc-slots-text").textContent = `${fmt(c.parked)} / ${fmt(c.slots)}`;
    $("cc-slots-bar").style.width = pct(c.parked, c.slots);
    const budget = c.budget_mib * 1048576;
    $("cc-mem-text").textContent = `${gb(c.bytes)} / ${gb(budget)} GB`;
    $("cc-mem-bar").style.width = pct(c.bytes, budget);
  }
  $("cc-sum").textContent = c.requests ? `${fmt(c.requests_reused)} of ${fmt(c.requests)} requests reused part of their prompt` : "";
  const share = c.prompt_tokens ? ` (${fmt((100 * c.reused_tokens) / c.prompt_tokens)}% of all prompt tokens)` : "";
  const event = c.last_event ? `${c.last_event === "parked" ? "Parked" : "Restored"} ${fmt(c.last_tokens)} tokens, ${since(c.last_at)}` : null;
  facts($("cc-facts"), [
    ["Last request", c.last_prompt != null ? `${fmt(c.last_reused || 0)} of ${fmt(c.last_prompt)} prompt tokens reused` : null],
    ["Reused since start", c.requests ? `${fmt(c.reused_tokens)} tokens${share}` : null],
    ["Parked / restored", c.enabled ? `${fmt(c.parks)} / ${fmt(c.restores)}${c.evictions ? ` · ${fmt(c.evictions)} evicted` : ""}` : null],
    ["Last switch", c.enabled ? event : null],
  ]);
  $("cc-note").textContent = c.enabled
    ? "The L2 RAM tier keeps whole parked conversations in host memory: a request that continues one gets its state back instead of reading it again; the oldest goes when the slots or the memory are full."
    : "The engine keeps the last conversation's state, so a follow-up reads only what is new. To keep several conversations (agents taking turns) in the L2 RAM tier, add \"--conversation-cache-mib\", \"8192\" to the run config's args (docs/DETAILS.md).";
}

function renderTotals(t) {
  if (!t || !t.requests) return "";
  const since = new Date(t.since * 1000).toLocaleString([], {weekday: "short", hour: "2-digit", minute: "2-digit"});
  const read = Math.max(0, (t.prompt_tokens || 0) - (t.reused || 0));
  const pSpeed = t.prompt_ms > 0 && read > 0 ? ` at ${fmt(read / (t.prompt_ms / 1000))} tok/s` : "";
  const oSpeed = t.decode_ms > 0 && t.output_tokens > 0 ? ` at ${fmt(t.output_tokens / (t.decode_ms / 1000), 1)} tok/s` : "";
  const ttft = t.ttft_ms > 0 ? ` · first token after ${dur(t.ttft_ms / t.requests)} on average` : "";
  return `Since ${since}: ${fmt(t.requests)} requests · ${fmt(read)} prompt tokens read${pSpeed} (${fmt(t.reused)} reused) · ` +
         `${fmt(t.output_tokens)} written${oSpeed}${ttft}`;
}

function renderMonitor(m) {
  const live = m.live || {}, hw = m.hardware || {}, st = m.hardware_static || {}, eng = m.engine || {};
  const h = m.history || {}, requests = m.requests || [], totals = m.totals || {};
  const keptCount = m.requests_kept == null ? requests.length : m.requests_kept;
  const last = requests[0];
  const win = requests.slice(0, 20);        // the recent window the tiles and the TTFT outliers are measured over

  // model state
  const on = live.queued > 0 ? "queued" : live.state;
  for (const b of document.querySelectorAll("#state-badges .st-badge")) b.classList.toggle("on", b.dataset.s === on || b.dataset.s === live.state);
  const prog = $("state-progress");
  let label = "Waiting for a request", detail = "", pct = 0;
  if (live.state === "reading") {
    label = "Reading prompt";
    prog.dataset.tone = "info";
    if (live.prompt_total) {
      pct = (100 * live.prompt_read) / live.prompt_total;
      detail = `${fmt(live.prompt_read)} / ${fmt(live.prompt_total)} tokens · ${fmt(pct)}%`;
    } else {
      detail = `${fmt(live.prompt_tokens)} tokens`;
    }
  } else if (live.state === "generating") {
    label = live.phase ? live.phase[0].toUpperCase() + live.phase.slice(1) : "Generating";
    delete prog.dataset.tone;
    pct = live.max_tokens ? Math.min(100, (100 * live.generated) / live.max_tokens) : 0;
    detail = `${fmt(live.generated)} tokens · ${fmt(live.tok_s, 1)} tok/s`;
  } else if (last) {
    delete prog.dataset.tone;
    detail = `last: ${fmt(last.output_tokens)} tokens${last.decode_tok_s ? ` at ${fmt(last.decode_tok_s, 1)} tok/s` : ""}`;
  }
  setText("state-label", label);
  setText("state-detail", detail);
  $("state-bar").style.width = `${pct}%`;

  // live chips: only what the running request actually has
  const chips = [];
  if (live.state !== "idle") {
    if (live.tok_s != null) chips.push(["Decode", rate(live.tok_s)]);
    if (live.prefill_tok_s_mean != null) chips.push(["Prefill", rate(live.prefill_tok_s_mean)]);
    if (live.generated != null) chips.push(["Written", `${fmt(live.generated)}${live.max_tokens ? ` / ${fmt(live.max_tokens)}` : ""}`]);
    if (live.elapsed_s != null) chips.push(["Elapsed", `${fmt(live.elapsed_s, 1)} s`]);
  }
  if (live.queued) chips.push(["Queued", fmt(live.queued)]);
  $("state-live").innerHTML = chips.map(([k, v]) => `<span class="live-chip">${esc(k)} <b>${esc(v)}</b></span>`).join("");

  // the tiles: the session's averages, the recent window beside them
  const readTokens = totals.prompt_tokens || 0;   // the counter already excludes cached tokens
  const sessionDecode = totals.decode_ms > 0 && totals.output_tokens ? totals.output_tokens / (totals.decode_ms / 1000) : null;
  const sessionPrefill = totals.prompt_ms > 0 && readTokens ? readTokens / (totals.prompt_ms / 1000) : null;
  const sessionTtft = totals.ttft_ms > 0 && totals.requests ? totals.ttft_ms / totals.requests : null;
  const winDecode = mean(win.map((r) => r.decode_tok_s));
  const winPrefill = mean(win.map(reqPrefill));
  const winTtft = mean(win.map((r) => r.ttft_ms));
  const winDuration = mean(win.map((r) => r.duration_s));
  const winHit = mean(win.map((r) => (r.hit_rate == null ? null : r.hit_rate * 100)));
  const winSpec = mean(win.map((r) => (r.drafts_offered ? (100 * (r.drafts_accepted || 0)) / r.drafts_offered : null)));
  const errors = requests.filter((r) => r.finish === "error").length;
  setKpi("decode", kpiValue(fmt(sessionDecode ?? winDecode, 1), "t/s"),
         `recent ${rate(winDecode)}`);
  setKpi("prefill", kpiValue(fmt(sessionPrefill ?? winPrefill, 1), "t/s"),
         `now ${rate(lastOf(h.prefill_tok_s_mean))} · recent ${rate(winPrefill)}`);
  setKpi("ttft", kpiValue(dur(winTtft), ""),
         `recent ${dur(winTtft)} · last ${dur(last ? last.ttft_ms : null)}`);
  setKpi("duration", kpiValue(winDuration == null ? "–" : fmt(winDuration, 1), "s"), `${fmt(win.length)} recent requests`);
  setKpi("requests", kpiValue(fmt(requests.length), ""),
         `${fmt(totals.decode_steps ?? 0)} decode steps · ${fmt(errors)} error${errors === 1 ? "" : "s"}`);
  setKpi("tokens", kpiValue(kfmt(totals.output_tokens || 0), ""),
         `${kfmt(readTokens)} prompt read · ${kfmt(totals.reused || 0)} reused`);
  {
    const pt = totals.prompt_tokens || 0, rt = totals.reused || 0;
    const sessionHit = pt + rt ? (100 * rt) / (pt + rt) : null;
    setKpi("hit", kpiValue(pctText(sessionHit, 1), ""),
           `${fmt(rt)} reused of ${fmt(pt + rt)} prompt tokens seen`);
  }
  const sessionSpec = totals.drafts_offered
    ? (100 * (totals.drafts_accepted || 0)) / totals.drafts_offered : null;
  setKpi("spec", kpiValue(pctText(sessionSpec ?? winSpec, 1), ""),
         `${fmt(totals.drafts_accepted || 0)} of ${fmt(totals.drafts_offered || 0)} drafts accepted (session)`);
  spark("kpispark-decode", h.tok_s);
  spark("kpispark-prefill", h.prefill_tok_s_mean);
  spark("kpispark-ttft", win.map((r) => r.ttft_ms).reverse());

  // throughput over the sampler's window
  const samples = (h.tok_s || []).length;
  chart($("chart-speed"), [{values: h.tok_s || [], color: "var(--st-accent)", axis: "left"},
                           {values: h.prefill_tok_s_mean || [], color: "var(--st-info)", axis: "right"}]);
  $("legend-speed").innerHTML =
    `<span style="color:var(--st-accent)"><i></i>Decode <b>${rate(live.state === "generating" ? live.tok_s : last ? last.decode_tok_s : null)}</b></span>` +
    `<span style="color:var(--st-info)"><i></i>Prefill <b>${rate(live.state === "idle" ? null : live.prefill_tok_s_mean)}</b></span>`;
  setText("chart-note", samples ? `${fmt(samples)} s of history, one sample a second` : "waiting for the sampler");

  // each card of the run on its own
  $("gpu-grid").innerHTML = gpuCards(hw, h, st);

  // the L3 store: only a run started with --conversation-cache-disk has one, and only once its store opened
  const l3 = m.l3;
  $("l3-card").hidden = !(l3 && l3.on);
  if (l3 && l3.on) {
    setText("l3-sub", l3.path || "");
    $("l3-metrics").innerHTML = l3Metrics(l3, h);
    const skipped = l3.skips ? ` ${fmt(l3.skips)} parks were skipped.` : "";
    setText("l3-note", (l3.evictions || l3.corruptions || l3.write_failures)
      ? `${fmt(l3.evictions || 0)} evictions · ${fmt(l3.corruptions || 0)} corrupt records · ` +
        `${fmt(l3.write_failures || 0)} failed writes.${skipped}`
      : `Parked conversations are records in this folder. A request resumes one when its prompt starts with that ` +
        `record's tokens; the engine keeps the longest matching prefix. Disk read and write are the engine's own ` +
        `traffic, so a record the operating system still has cached counts as almost nothing.${skipped}`);
  }

  // context fill: the running request, else the last one
  const ctx = eng.max_context || 0;
  let used = 0;
  if (live.state !== "idle") used = (live.prompt_tokens || 0) + (live.generated || 0);
  else if (last) used = (last.prompt_tokens || 0) + (last.output_tokens || 0);
  const frac = ctx ? Math.min(1, used / ctx) : 0;
  $("ctx-fill").setAttribute("stroke-dasharray", `${(235.6 * frac).toFixed(1)} 314.2`);
  $("ctx-fill").style.opacity = 235.6 * frac >= 3 ? "1" : "0";         // a near-zero arc would draw just its round cap
  setText("ctx-pct", `${Math.round(frac * 100)}%`);
  setText("ctx-sub", ctx ? `${kfmt(used)} / ${ctxfmt(ctx)}` : "–");
  const cacheBytes = (eng.expert_cache_mib || 0) * 1048576;
  setText("slots-text", `${fmt(requests.length)} requests · cache ${kfmt(totals.reused || 0)} tok reused`);
  $("slots-bar").style.width = hw.gpu_mem_total ? `${Math.min(100, (100 * (hw.gpus?.[1]?.mem_used || 0)) / hw.gpu_mem_total)}%` : "0%";
  const kv = {q4_0: "q4_0 · SM70 direct", q8_0: "q8_0", f16: "f16"}[eng.kv] || eng.kv || "q4_0";
  setText("kv-text", kv);
  setText("spec-text", "MTP, n_max 3");
  setText("kept-text", `${fmt(requests.length)} of ${fmt(keptCount)}`);

  // this machine
  setText("sys-sub", st.cpu_name || "");
  $("sys-metrics").innerHTML = sysMetrics(hw, st, h);

  // the request log
  const body = $("req-body");
  const shown = requests.slice(0, reqShowAll ? requests.length : 12);
  const medTtft = median(win.map((r) => r.ttft_ms));
  const ttftCls = (v) => (v == null || !medTtft ? "" : v > 3 * medTtft ? " slower" : v > 1.5 * medTtft ? " slow" : "");
  if (!shown.length) {
    body.innerHTML = `<tr><td colspan="10" class="muted">No requests yet</td></tr>`;
  } else {
    const badge = {stop: ["", "Done"], length: ["", "Max tokens"], cancel: ["st-badge--queued", "Stopped"],
                   disconnect: ["st-badge--queued", "Closed"], error: ["st-badge--error", "Error"]};
    body.innerHTML = shown.map((r) => {
      const [cls, text] = badge[r.finish] || ["", r.finish || "–"];
      const t = new Date(r.time * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
      const proj = r.projection == null ? "" : ` <span class="st-badge${r.projection ? " st-badge--reading" : ""}" title="experimental speed projection ${r.projection ? "on" : "off"}">${r.projection ? "ESP" : "stock"}</span>`;
      // #588: the VRAM share; the PCIe share (--pcie-frac) beside it when there is one
      const hit = r.hit_rate == null ? "–" : `${(r.hit_rate * 100).toFixed(1)}%` +
        (r.pcie_share ? ` <span class="muted" title="routed experts the GPU read over PCIe (--pcie-frac) or another GPU computed">+${(r.pcie_share * 100).toFixed(1)}% PCIe</span>` : "");
      const drafts = r.drafts_offered ? `${fmt(r.drafts_accepted || 0)} of ${fmt(r.drafts_offered)} drafts accepted` : "";
      return `<tr><td>${esc(t)}</td><td><span class="st-badge ${cls}">${esc(text)}</span>${proj}</td>
        <td class="num${ttftCls(r.ttft_ms)}" title="Time to first token${medTtft ? `; recent median ${dur(medTtft)}` : ""}">${dur(r.ttft_ms)}</td>
        <td class="num">${fmt(r.prompt_tokens)}</td>
        <td class="num">${fmt(r.reused)}</td>
        <td class="num">${fmt(reqPrefill(r), 0)}</td>
        <td class="num">${fmt(r.output_tokens)}</td>
        <td class="num">${fmt(r.decode_tok_s, 1)}</td>
        <td class="num" title="Prompt tokens reused from cache">${r.reused != null ? fmt(r.reused) : "–"}</td>
        <td class="num">${fmt(r.duration_s, 1)} s</td></tr>`;
    }).join("");
  }
  const all = $("req-all");
  all.hidden = keptCount <= 12;
  all.textContent = reqShowAll ? "Show fewer" : `Show all (${keptCount})`;
  $("req-wrap").classList.toggle("all", reqShowAll);
  setText("req-sum", `${fmt(requests.length)} of ${fmt(keptCount)} kept`);
  $("req-totals").textContent = renderTotals(totals);

  // #465: the engine's batch slots - sent only when the run has more than one (live.parallel / live.slots)
  const slots = live.slots || [];
  $("slots-card").hidden = !slots.length;
  if (slots.length) {
    const running = slots.filter((s) => s.state !== "idle").length;
    setText("slots-sum", `${fmt(running)} of ${fmt(slots.length)} in use` +
      (live.waiting ? ` · ${fmt(live.waiting)} waiting` : ""));
    $("slots-grid").innerHTML = slots.map((s) => {
      const busy = s.state !== "idle";
      const detail = !busy ? (s.held_tokens ? `${fmt(s.held_tokens)} tokens held for a next turn` : "free")
        : `${s.state === "reading" ? "reading the prompt" : "decoding"} · ${fmt(s.generated)} tokens · ` +
          `${fmt(s.elapsed_s, 1)} s${s.tok_s != null ? ` · ${fmt(s.tok_s, 1)} tok/s` : ""}`;
      return `<div class="slot-row" data-state="${busy ? s.state : "idle"}">
        <span class="slot-row__n">Slot ${s.slot + 1}</span><span class="slot-row__state">${esc(s.state)}</span>
        <span class="muted small">${esc(detail)}</span></div>`;
    }).join("");
  }
}

function facts(el, rows) {
  el.innerHTML = rows.filter((r) => r[1] != null && r[1] !== "").map(([k, v, copy]) =>
    `<dt>${esc(k)}</dt><dd>${copy ? `<code>${esc(v)}</code><button class="st-btn st-btn--icon" data-copy="${esc(v)}" aria-label="Copy">${icon("copy")}</button>` : esc(v)}</dd>`).join("");
}
// INFO cvec=project:4-44[:singleL] | add:A-B | 0
function projectionText(c) {
  if (!c || c === "0" || c === 0) return null;
  const [mode, range, single] = String(c).split(":");
  const [a, b] = (range || "").split("-");
  return `${mode === "project" ? "Projection" : "Additive"} control vector on layers ${a}–${b}` +
         `${single ? ` (layer ${single.replace("single", "")}'s direction)` : ""}. Per chat in Sampling. Its package ` +
         "describes the vector as a refusal-direction projection; measure the speed yourself";
}
// The launcher owns what will be started (model path or -hf repo, alias). /selection is
// cheap (no folder scan), so About can re-read it while visible instead of showing a
// snapshot from the moment the tab was opened.
let sel = null;
let selTick = 0;
async function loadSelection() {
  try { sel = await (await fetch("selection")).json(); } catch (e) { sel = null; }
  if (sel) {
    setSwitch("autostart-toggle", sel.autostart);
    setSwitch("minimize-toggle", sel.minimize_tray);
  }
  if (lastMetrics) render(lastMetrics);
}

// The launcher's view of the server it runs: the live port and the argv it was
// started with. About's "Connect your tools" used location.origin, i.e. the GUI
// port, which stops being the API port as soon as --port is changed in the form.
let srvInfo = null;
async function loadServerInfo() {
  try { srvInfo = await (await fetch("server")).json(); } catch (e) { srvInfo = null; }
  if (lastMetrics) render(lastMetrics);
}

// Launcher-owned switches (stored in the launcher config, not this browser):
// - autostart: load the saved model when the exe starts (off by default)
// - minimize_tray: the X button hides to the tray instead of quitting (on by default)
function setSwitch(id, on) {
  const t = $(id);
  if (t) t.setAttribute("aria-checked", String(!!on));
}
async function flipSwitch(id, key) {
  const before = $(id).getAttribute("aria-checked") === "true";
  const next = !before;
  setSwitch(id, next);                      // optimistic: the switch must feel immediate
  try {
    const r = await fetch("save", {method: "POST", headers: {"Content-Type": "application/json"},
                                   body: JSON.stringify({[key]: next})});
    const j = await r.json().catch(() => ({}));
    if (!r.ok || !j.ok) throw new Error(j.message || `HTTP ${r.status}`);
  } catch (e) {
    setSwitch(id, before);
    const t = (k, d) => (window.SM70I18n ? SM70I18n.t(k) : d);
    toast("error", t("about.autostartFailed", "Could not save the setting"),
          String(e.message || e), 3000);
  }
}
$("autostart-toggle").onclick = () => flipSwitch("autostart-toggle", "autostart");
$("minimize-toggle").onclick = () => flipSwitch("minimize-toggle", "minimize_tray");
const SEP = String.fromCharCode(92) + "/";  // both separators, no escape games
const baseName = (p) => {
  const s = String(p || "");
  let i = -1;
  for (const c of SEP) i = Math.max(i, s.lastIndexOf(c));
  return i >= 0 ? s.slice(i + 1) : s;
};

// What the running server was actually told. About used to print fixed values -
// q4_0 KV and "MTP drafts up to 2" - even with speculation off, a draft model or
// another KV type. The launcher reports the argv it spawned (GET /server); an
// external server has none, and then we say nothing instead of guessing.
function liveEngineFacts() {
  const cmd = srvInfo && Array.isArray(srvInfo.cmdline) ? srvInfo.cmdline : null;
  if (!cmd || !cmd.length) return { kv: null, spec: null };
  const arg = (flag) => {
    const i = cmd.indexOf(flag);
    return i >= 0 && i + 1 < cmd.length ? cmd[i + 1] : "";
  };
  const name = (t) => ({q4_0: "4-bit (Hadamard-rotated)", q8_0: "8-bit", f16: "16-bit"}[t] || t);
  const k = arg("-ctk"), v = arg("-ctv");
  const kv = !k ? null : (!v || v === k ? name(k) : `K ${name(k)} / V ${name(v)}`);
  const type = arg("--spec-type") || (arg("--spec-draft-model") ? "draft-simple" : "");
  const n = Number(arg("--spec-draft-n-max") || 0);
  let spec = null;
  if (type && type !== "none") {
    if (type === "draft-mtp") spec = `MTP drafts up to ${Math.max(0, n - 1)} tokens`;
    else {
      const d = arg("--spec-draft-model");
      spec = `${type}${d ? ` · ${baseName(d)}` : ""}${n ? ` (n=${n})` : ""}`;
    }
  }
  return { kv, spec };
}

function renderAbout(eng, hw, st) {
  const live = liveEngineFacts();
  facts($("facts-engine"), [
    ["Model", eng.model],
    ["Engine", eng.version || null],
    ["Context", eng.max_context ? `${fmt(eng.max_context)} tokens` : null],
    ["KV cache", live.kv ? `${live.kv}, all in VRAM` : null],
    ["Experts in VRAM", eng.expert_slots ? `${fmt(eng.expert_slots)} (${gb((eng.expert_cache_mib || 0) * 1048576)} GB)` : null],
    ["Speculation", live.spec],
    ["Images", eng.images ? "on" : "off"],
    ["Experimental speed projection", projectionText(eng.cvec)],
    [window.SM70I18n ? SM70I18n.t("about.configured") : "Configured model (launcher)",
     sel && sel.model ? (/\.gguf$/i.test(sel.model) ? baseName(sel.model) : sel.model) : null],
    [window.SM70I18n ? SM70I18n.t("about.alias") : "Model name (alias)",
     sel && sel.args ? sel.args.alias : null],
    [window.SM70I18n ? SM70I18n.t("about.draft") : "Speculative draft model",
     sel && sel.args ? (sel.args.draft_model ? baseName(sel.args.draft_model) : null) : null],
  ]);
  facts($("facts-hw"), [
    ["GPU", st.gpu_name ? `${st.gpu_name}${hw.gpu_mem_total ? `, ${gb(hw.gpu_mem_total, 0)} GB` : ""}` : (st.gpu_note || "not readable (NVML)")],
    ["CPU", st.cpu_name ? `${st.cpu_name}${st.threads ? `, ${st.threads} threads` : ""}` : null],
    ["RAM", hw.ram_total ? `${gb(hw.ram_total, 0)} GB` : null],
  ]);
  // the API lives on the llama-server port, which is not the page's port once the
  // form sets --port; /server reports the one that is actually in use
  const apiPort = (srvInfo && srvInfo.port) || location.port;
  const base = `${location.protocol}//${location.hostname}:${apiPort}`;
  facts($("facts-api"), [
    ["OpenAI base URL", `${base}/v1`, true],
    ["Anthropic base URL", base, true],
    ["Model name", eng.model, true],
  ]);
}
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-copy]");
  if (b) copyText(b.dataset.copy, b);
});
$("req-all").addEventListener("click", () => { reqShowAll = !reqShowAll; if (lastMetrics) render(lastMetrics); });

// ------------------------------------------------------------------ MCP servers (GET /mcp)
// Tools from the MCP servers in the run config: the chat offers them to the model (opt-in per request,
// "strata_mcp": true, which only this page sends); the Monitor lists the servers and what they offer.
let mcpInfo = {servers: [], tools: 0}, mcpRetry = null;
async function loadMcp() { /* sm70: no MCP endpoint; panel stays hidden */ }
function renderMcp() {
  const servers = mcpInfo.servers || [];
  $("mcp-card").hidden = !servers.length;
  $("mcp-row").hidden = !servers.length;
  const ready = servers.filter((s) => s.status === "ready" || s.status === "stopped");
  $("mcp-sum").textContent = servers.length ? `${fmt(mcpInfo.tools)} tools · ${ready.length} of ${servers.length} servers connected` : "";
  $("mcp-row-sub").textContent = mcpInfo.tools ? `${fmt(mcpInfo.tools)} tools from ${ready.map((s) => s.name).join(", ")}; the model calls them when it decides to`
                                               : "no server is connected yet (see the Monitor)";
  $("mcp-list").innerHTML = servers.map((s) => {
    const [cls, text] = MCP_STATE[s.status] || ["", s.status];
    const info = s.info && s.info.name ? ` · ${s.info.name}${s.info.version ? ` ${s.info.version}` : ""}` : "";
    return `<div class="mcp-server"><div class="mcp-server__head"><span class="st-badge ${cls}">${esc(text)}</span>` +
      `<strong>${esc(s.name)}</strong><span class="muted small">${esc(s.transport)} · ${fmt(s.tools.length)} tools${esc(info)}</span></div>` +
      (s.error ? `<div class="msg-error">${esc(s.error)}</div>` : "") +
      (s.tools.length ? `<div class="mcp-server__tools">${s.tools.map((t) => `<span class="chip" title="${esc(t.description || "")}">${esc(t.tool)}</span>`).join("")}</div>` : "") +
      `</div>`;
  }).join("");
}

// ------------------------------------------------------------------ Model settings (GET / POST /config, #564)
// A few documented keys of the run config (strata-<model>.json), for every client, from the next start on.  The
// server lists them, checks every value and keeps every other key of the file as it is.
let cfgKeys = [];
async function loadConfig() {
  let r;
  return; // sm70: no run-config endpoint; the launcher owns startup parameters
  if (!r.ok) { $("cfg-card").hidden = true; return; }       // no run config, an older server, or no key yet
  const c = await r.json();
  cfgKeys = c.keys || [];
  $("cfg-file").textContent = c.file || "";
  $("cfg-form").innerHTML = cfgKeys.map((k, i) => {
    const id = `cfg-${i}`, v = k.value;
    let input;
    if (k.kind === "bool" || k.kind === "enum") {
      const opts = k.kind === "bool" ? [["true", "on"], ["false", "off"]] : k.choices.map((x) => [x, x]);
      const cur = v == null ? "" : String(v);
      input = `<select class="st-input" id="${id}"><option value=""${cur === "" ? " selected" : ""}>default</option>` +
        opts.map(([val, text]) => `<option value="${esc(val)}"${cur === val ? " selected" : ""}>${esc(text)}</option>`).join("") + `</select>`;
    } else {
      const text = v == null ? "" : Array.isArray(v) ? v.join(", ") : String(v);
      input = `<input class="st-input" id="${id}" ${k.kind === "number" ? 'type="number" step="any" min="0"' : 'type="text"'} ` +
        `value="${esc(text)}" placeholder="default" autocomplete="off">`;
    }
    return `<label for="${id}" title="${esc(k.help)}">${esc(k.help)}<code>${esc(k.key)}</code></label>${input}`;
  }).join("");
  $("cfg-card").hidden = false;
}
function configValue(k, el) {
  const s = el.value.trim();
  if (s === "") return null;
  if (k.kind === "bool") return s === "true";
  if (k.kind === "number") return Number(s);
  return s;                                         // enum, or names (the server splits them at commas)
}
$("cfg-save").addEventListener("click", async () => {
  const set = {};
  cfgKeys.forEach((k, i) => {
    const v = configValue(k, $(`cfg-${i}`));
    const old = Array.isArray(k.value) ? k.value.join(", ") : k.value;
    if (JSON.stringify(v) !== JSON.stringify(old ?? null)) set[k.key] = v;
  });
  if (!Object.keys(set).length) { $("cfg-msg").textContent = "Nothing changed."; return; }
  try {
    return; // sm70: read-only settings here; startup params live in the launcher
    const b = await r.json();
    if (!r.ok) throw new Error((b.error || {}).message || `HTTP ${r.status}`);
    $("cfg-msg").textContent = b.changed.length
      ? `Saved (${b.changed.join(", ")}); the earlier file is ${b.file}.bak. Start the model again to use it.` : "Nothing changed.";
    loadConfig();
  } catch (e) {
    $("cfg-msg").textContent = "";
    toast("error", "Not saved", String(e.message || e), 6000);
  }
});

// ------------------------------------------------------------------ Markdown (escaped first, then formatted)
function inline(s) {
  const codes = [];
  s = s.replace(/`([^`\n]+)`/g, (_, c) => { codes.push(c); return `\u0000${codes.length - 1}\u0000`; });
  s = esc(s)
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?![*\w])/g, "$1<em>$2</em>")
    .replace(/\[([^\]\n]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code class="inline">${esc(codes[+i])}</code>`);
}
function codeBlock(lang, code) {
  return `<div class="st-code"><div class="st-code__head"><span>${esc(lang || "code")}</span>` +
    `<button class="st-btn st-btn--icon" data-code-copy aria-label="Copy code">${icon("copy")}</button></div>` +
    `<pre><code>${esc(code)}</code></pre></div>`;
}
function blocks(text) {
  const out = [], lines = text.split("\n");
  let para = [], list = null;
  const flushPara = () => { if (para.length) out.push(`<p>${para.map(inline).join("<br>")}</p>`); para = []; };
  const flushList = () => { if (list) out.push(`<${list.tag}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.tag}>`); list = null; };
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    let m;
    if (!l.trim()) { flushPara(); flushList(); continue; }
    if ((m = l.match(/^(#{1,6})\s+(.*)$/))) { flushPara(); flushList(); out.push(`<${m[1].length <= 2 ? "h3" : "h4"}>${inline(m[2])}</${m[1].length <= 2 ? "h3" : "h4"}>`); continue; }
    if (/^\s*([-*_])\s*\1\s*\1[\s\1]*$/.test(l)) { flushPara(); flushList(); out.push("<hr>"); continue; }
    if ((m = l.match(/^>\s?(.*)$/))) { flushPara(); flushList(); out.push(`<blockquote>${inline(m[1])}</blockquote>`); continue; }
    if (/^\s*\|.*\|\s*$/.test(l) && i + 1 < lines.length && /^\s*\|?[\s:-]+\|[\s|:-]*$/.test(lines[i + 1])) {
      flushPara(); flushList();
      const cells = (row) => row.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));
      let html = `<table><thead><tr>${cells(l).map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>`;
      i += 2;
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) html += `<tr>${cells(lines[i++]).map((c) => `<td>${c}</td>`).join("")}</tr>`;
      i--;
      out.push(html + "</tbody></table>");
      continue;
    }
    if ((m = l.match(/^\s*(?:[-*+]|(\d+)[.)])\s+(.*)$/))) {
      flushPara();
      const tag = m[1] ? "ol" : "ul";
      if (!list || list.tag !== tag) { flushList(); list = {tag, items: []}; }
      list.items.push(m[2]);
      continue;
    }
    if (list && /^\s{2,}\S/.test(l)) { list.items[list.items.length - 1] += " " + l.trim(); continue; }
    flushList();
    para.push(l);
  }
  flushPara(); flushList();
  return out.join("");
}
function markdown(text) {
  let html = "", rest = text;
  for (;;) {
    const m = rest.match(/(^|\n)```([^\n`]*)\n/);
    if (!m) { html += blocks(rest); break; }
    html += blocks(rest.slice(0, m.index));
    rest = rest.slice(m.index + m[0].length);
    const end = rest.match(/(^|\n)```[ \t]*(\n|$)/);
    if (!end) { html += codeBlock(m[2].trim(), rest); break; }         // still streaming
    html += codeBlock(m[2].trim(), rest.slice(0, end.index));
    rest = rest.slice(end.index + end[0].length);
  }
  return html;
}

// ------------------------------------------------------------------ Chat
// The sampling shown until the user changes it is the model card's thinking set. Untouched, the page sends NO sampling
// fields: the server picks the card's set for the request's thinking state (thinking 1.0 / 0.95 / 20, not thinking
// 0.7 / 0.8 / 20 with a presence penalty). Before 0.1.45 the page sent 0.6 itself; a stored 0.6 with the other two
// values untouched counts as "not set" too.
const DEFAULTS = {thinking: "medium", temperature: 1.0, top_p: 0.95, top_k: 20, max: "", seed: "", show: true, esp: true, mcp: true};
const OLD_DEFAULT_TEMPERATURE = 0.6;
function samplingUntouched(s) {
  const t = +s.temperature;
  return (t === DEFAULTS.temperature || t === OLD_DEFAULT_TEMPERATURE) && +s.top_p === DEFAULTS.top_p && +s.top_k === DEFAULTS.top_k;
}
// one-time hygiene: purge assistant turns with no answer text from the STORED
// chat (the -n 320 era left corpses that derailed later prompts even after the
// apiMessages fix, because they kept coming back from localStorage)
{
  const stored = store.get("chat", []);
  if (Array.isArray(stored)) {
    const cleaned = stored.filter((m) => !(m.role === "assistant" && !m.text && !m.error));
    if (cleaned.length !== stored.length) store.set("chat", cleaned);
  } else {
    store.set("chat", []);   // a hand-edited/older value must not break the module
  }
}
let settings = {...DEFAULTS, ...store.get("sampling", {})};
// llama-server's qwen chat template accepts xhigh | high? no: xhigh/medium/low.
// "high" came from Strata's defaults and 400s the template — remap it.
if (settings.thinking === "high") settings.thinking = "xhigh";
if (samplingUntouched(settings)) settings.temperature = DEFAULTS.temperature;
let messages = store.get("chat", []);
let attachments = [];                 // {name, url}
let busy = null;                      // {controller, msg}

function saveChat() {
  // keep the last file's text so a reload does not silently drop attachments
  // (bounded at 512 KB per file, matching MAX_TEXT_FILE)
  store.set("chat", messages.map((m) => ({...m, images: (m.images || []).map((i) => ({name: i.name})),
                                           files: (m.files || []).map((f) => ({name: f.name, text: (f.text || "").slice(0, 524288)}))})));
}
function timeStr(t) { return new Date(t).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"}); }

function msgEl(m, i) {
  const el = document.createElement("div");
  el.className = `st-msg st-msg--${m.role}`;
  el.dataset.i = i;
  if (m.role === "user") {
    if (m.files && m.files.length) {
      const wrap = document.createElement("div");
      wrap.className = "msg-images";
      for (const f of m.files) {
        const c = document.createElement("span");
        c.className = "chip";
        c.innerHTML = icon("attach", "st-icon st-icon--sm");
        c.append(f.name);
        wrap.appendChild(c);
      }
      el.appendChild(wrap);
    }
    if (m.images && m.images.length) {
      const wrap = document.createElement("div");
      wrap.className = "msg-images";
      for (const im of m.images) {
        if (im.url) { const img = document.createElement("img"); img.src = im.url; img.alt = im.name || "image"; wrap.appendChild(img); }
        else { const c = document.createElement("span"); c.className = "chip"; c.innerHTML = icon("image", "st-icon st-icon--sm"); c.append(im.name || "image"); wrap.appendChild(c); }
      }
      el.appendChild(wrap);
    }
    const b = document.createElement("div");
    b.className = "st-bubble";
    b.textContent = m.text;
    el.appendChild(b);
    const meta = document.createElement("div");
    meta.className = "st-msg__meta";
    meta.textContent = `You · ${timeStr(m.time)}`;
    el.appendChild(meta);
  } else {
    el.innerHTML = `<details class="st-collapse think" hidden><summary>${icon("thinking", "st-icon st-icon--sm")}<span class="think-title"></span>` +
      `${icon("chevron", "st-icon st-icon--sm st-chev")}</summary><div class="st-collapse__body thinking"></div></details>` +
      `<div class="st-bubble"></div><div class="st-msg__meta"><span class="meta-text"></span>` +
      `<button class="st-btn st-btn--icon" data-msg-copy aria-label="Copy the answer" title="Copy">${icon("copy")}</button></div>`;
    updateAssistant(el, m, false);
  }
  return el;
}
// One MCP tool call in the answer: a compact block (name, state, a one-line preview) that opens to the arguments and
// the result as the model read it.  Its body is built only while open: a result can be 20,000 characters.
const TOOL_STATE = {writing: ["st-badge--reading", "Writing"], running: ["st-badge--generating", "Running"], done: ["", "Done"],
                    error: ["st-badge--error", "Error"], skipped: ["st-badge--queued", "Not run"]};
function toolHtml(t, k) {
  const [cls, label] = TOOL_STATE[t.state] || ["", t.state];
  const args = t.arguments == null ? "" : JSON.stringify(t.arguments, null, 2);
  const preview = t.result != null ? t.result : args.replace(/\s+/g, " ");
  let body = "";
  if (t.open) {
    body = `<div class="tool-call__label">Arguments</div><pre class="tool-call__pre">${esc(args || "(being written)")}</pre>`;
    if (t.result != null) {
      body += `<div class="tool-call__label">${t.ok ? "Result" : "Error"}${t.chars ? ` · ${fmt(t.chars)} characters` : ""}` +
              `${t.truncated ? ", cut for the model" : ""}</div><pre class="tool-call__pre">${esc(t.result)}</pre>`;
    }
  }
  return `<details class="st-collapse tool-call" data-tool="${k}" data-state="${esc(t.state)}"${t.open ? " open" : ""}>` +
    `<summary>${icon("tool", "st-icon st-icon--sm")}<span class="tool-call__name" title="${esc(t.name || "")}">${esc(t.tool || t.name || "tool")}</span>` +
    (t.server ? `<span class="muted small">${esc(t.server)}</span>` : "") +
    `<span class="tool-call__preview muted">${esc(preview.slice(0, 200))}</span>` +
    `<span class="st-badge ${cls}">${esc(label)}</span>${t.ms != null && t.state !== "skipped" ? `<span class="muted small">${fmt(t.ms / 1000, 1)} s</span>` : ""}` +
    `${icon("chevron", "st-icon st-icon--sm st-chev")}</summary><div class="st-collapse__body">${body}</div></details>`;
}
// the answer's text with the tool blocks where the model called them
function answerHtml(m) {
  if (!m.tools || !m.tools.length) return markdown(m.text || "");
  let html = "", pos = 0;
  m.tools.forEach((t, k) => {
    const at = Math.min(Math.max(t.at || 0, pos), m.text.length);
    if (at > pos) html += markdown(m.text.slice(pos, at));
    pos = at;
    html += toolHtml(t, k);
  });
  return html + markdown(m.text.slice(pos));
}
// a tool event from the stream (the `strata_mcp` field of a chunk)
function onTool(m, x) {
  if (x.event === "limit") { m.limit = x.max_rounds; return; }
  m.tools = m.tools || [];
  let t = m.tools.find((y) => y.id === x.id);
  if (!t) { t = {id: x.id, name: x.name, at: m.text.length, rat: m.reasoning.length, state: "writing"}; m.tools.push(t); }
  if (x.event === "call") {
    Object.assign(t, {name: x.name, server: x.server, tool: x.tool, arguments: x.arguments, round: x.round, state: "running"});
  } else if (x.event === "result") {
    Object.assign(t, {result: x.text, ok: x.ok, chars: x.chars, truncated: x.truncated, ms: x.ms,
                      state: x.skipped ? "skipped" : x.ok ? "done" : "error"});
  }
}
function updateAssistant(el, m, streaming) {
  const det = el.querySelector("details.think");
  if (m.reasoning) {
    det.hidden = false;
    const thinkingNow = streaming && !m.text;
    el.querySelector(".think-title").textContent = thinkingNow ? "Thinking…" :
      m.thinkSecs != null ? `Thought for ${fmt(m.thinkSecs, 1)} s` : "Thoughts";
    const body = el.querySelector(".thinking");
    if (det.open || thinkingNow) body.textContent = m.reasoning;
    else body.dataset.pending = "1";
    // open while it streams (if wanted), closed once the answer starts - unless the user toggled it themselves
    if (thinkingNow && settings.show && !det.dataset.touched && !det.open) { det._auto = true; det.open = true; }
    if (!thinkingNow && det.open && !det.dataset.touched) { det._auto = true; det.open = false; }
  }
  const bubble = el.querySelector(".st-bubble");
  if (m.error) {
    bubble.innerHTML = `<div class="msg-error"></div>`;
    bubble.firstChild.textContent = m.error;
  } else if (!m.text && streaming && !(m.tools && m.tools.length)) {
    bubble.innerHTML = m.reasoning ? `<span class="muted cursor">Writing</span>` : `<span class="cursor"></span>`;
  } else {
    bubble.innerHTML = answerHtml(m);
    if (streaming) bubble.classList.add("cursor"); else bubble.classList.remove("cursor");
  }
  el.querySelector(".meta-text").textContent = m.meta || (streaming ? "" : m.stopped ? "Stopped" : "");
  el.querySelector("[data-msg-copy]").hidden = streaming || !m.text;
}
function renderChat() {
  const chat = $("chat");
  chat.querySelectorAll(".st-msg").forEach((e) => e.remove());
  $("chat-empty").hidden = messages.length > 0;
  messages.forEach((m, i) => chat.appendChild(msgEl(m, i)));
  scrollDown(true);
}
function nearBottom() { const s = $("chat-scroll"); return s.scrollHeight - s.scrollTop - s.clientHeight < 120; }
function scrollDown(force) { const s = $("chat-scroll"); if (force || nearBottom()) s.scrollTop = s.scrollHeight; }

$("chat").addEventListener("click", (e) => {
  const cc = e.target.closest("[data-code-copy]");
  if (cc) { copyText(cc.closest(".st-code").querySelector("pre").textContent, cc); return; }
  const mc = e.target.closest("[data-msg-copy]");
  if (mc) { const i = +mc.closest(".st-msg").dataset.i; copyText(messages[i].text, mc); return; }
  // a tool block: its open state lives in the message (the answer is rebuilt while it streams), so the click sets it
  const sum = e.target.closest(".tool-call > summary");
  if (sum) {
    e.preventDefault();
    const el = sum.closest(".st-msg"), m = messages[+el.dataset.i], t = m && m.tools && m.tools[+sum.parentElement.dataset.tool];
    if (!t) return;
    t.open = !t.open;
    updateAssistant(el, m, !!busy && busy.msg === m);
  }
});
$("chat").addEventListener("toggle", (e) => {
  const d = e.target;
  if (d.tagName !== "DETAILS" || !d.classList.contains("think")) return;
  if (d._auto) { d._auto = false; return; }          // our own open/close, not the user's
  d.dataset.touched = "1";
  const body = d.querySelector(".thinking");
  if (d.open && body.dataset.pending) { body.textContent = messages[+d.closest(".st-msg").dataset.i].reasoning; delete body.dataset.pending; }
}, true);

function apiMessages() {
  const out = [];
  // bound the API history to the last MAX_HISTORY messages (the full chat
  // stays on screen): a long session with big attachments otherwise pushes
  // the current question out of focus and slows every request
  const MAX_HISTORY = 24;
  const source = messages.length > MAX_HISTORY ? messages.slice(-MAX_HISTORY) : messages;
  for (const m of source) {
    if (m.role === "user") {
      const imgs = (m.images || []).filter((i) => i.url);
      const text = userText(m);
      out.push({role: "user", content: imgs.length ? [{type: "text", text},
        ...imgs.map((i) => ({type: "image_url", image_url: {url: i.url}}))] : text});
    } else if (!(busy && busy.msg === m)) {            // the answer being asked for now is not history yet
      out.push(...assistantMessages(m));
    }
  }
  return out;
}
// An answer that used MCP tools goes back as the model wrote it: per round the text before the calls, the calls and
// their results (as the model read them), then the rest - so the next question can build on what the tools found.
function assistantMessages(m) {
  const ran = (m.tools || []).filter((t) => t.round != null && t.result != null && t.state !== "skipped");
  // #1392: a turn with no answer text (only reasoning, a stop before the first content token, or an error) still goes
  // back as an assistant turn, so the history keeps alternating and a reasoning model sees that it already answered.
  // The server leaves such a turn out of the prompt itself (serve/frontend.py, #843); reasoning_content rides along.
  if (!ran.length) {
    // sm70: a turn that produced no answer text (error, or the old -n limit)
    // would go back as content:"" + a reasoning blob; qwen3's template derails
    // on those. Skip empty turns entirely — the visible chat keeps them.
    if (!m.text) return [];
    const msg = {role: "assistant", content: m.text};
    return [msg];
  }
  const out = [];
  let pos = 0;
  for (const r of [...new Set(ran.map((t) => t.round))]) {
    const calls = ran.filter((t) => t.round === r);
    const at = Math.min(Math.max(pos, calls[0].at || 0), m.text.length);
    out.push({role: "assistant", content: m.text.slice(pos, at).trim(),
              tool_calls: calls.map((t) => ({id: t.id, type: "function", function: {name: t.name, arguments: JSON.stringify(t.arguments || {})}}))});
    for (const t of calls) out.push({role: "tool", tool_call_id: t.id, content: t.result});
    pos = at;
  }
  const rest = m.text.slice(pos).trim();
  if (rest) out.push({role: "assistant", content: rest});
  return out;
}

function setBusy(on) {
  $("stop-btn").hidden = !on;
  $("send-btn").disabled = on;
  $("composer-hint").textContent = on ? "" : "Shift+Enter: new line";
}

async function send() {
  const text = $("input").value.trim();
  if ((!text && !attachments.length) || busy) return;
  messages.push({role: "user", text, images: attachments.filter((a) => a.kind !== "file"),
                 files: attachments.filter((a) => a.kind === "file"), time: Date.now()});
  attachments = [];
  renderAttachments();
  $("input").value = "";
  autosize();
  const m = {role: "assistant", text: "", reasoning: "", time: Date.now()};
  messages.push(m);
  renderChat();
  const el = $("chat").lastElementChild;
  const controller = new AbortController();
  busy = {controller, msg: m};
  setBusy(true);

  const body = {model: health.model, messages: apiMessages(), stream: true,
                reasoning_effort: settings.thinking};
  if (!(settings.temperature > 0)) {
    body.temperature = 0;
  } else if (!samplingUntouched(settings)) {
    Object.assign(body, {temperature: +settings.temperature, top_p: +settings.top_p, top_k: +settings.top_k});
  }
  if (settings.seed) body.seed = +settings.seed;
  if (settings.max) body.max_tokens = +settings.max;
  if (projectionLoaded()) body.experimental_speed_projection = !!settings.esp;
  // sm70: no MCP endpoint on llama-server; strata_mcp is never sent

  let firstAt = null, thinkStart = null, usage = null, frame = 0;
  const tSubmit = performance.now();
  const paint = () => { frame = 0; updateAssistant(el, m, true); scrollDown(); };
  try {
    const r = await fetch("v1/chat/completions", {method: "POST", headers: headers(true), body: JSON.stringify(body),
                                                   signal: controller.signal});
    if (!r.ok) {
      let msg = `HTTP ${r.status}`;
      try { msg = (await r.json()).error.message || msg; } catch (e) { /* not json */ }
      if (r.status === 401) msg = "This server needs an API key: add it under About > Settings.";
      throw new Error(msg);
    }
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line.startsWith("data:")) continue;              // ": keep-alive" comments while a long prompt is read
        const data = line.slice(5).trim();
        if (data === "[DONE]") continue;
        let j;
        try { j = JSON.parse(data); } catch (e) { continue; }
        if (j.error) throw new Error(j.error.message || "the engine reported an error");
        if (j.usage) usage = j.usage;
    const fr = j.choices && j.choices[0] && j.choices[0].finish_reason;
    if (fr === "length") m.hitLimit = true;
        if (j.timings) usage = { ...(usage || {}), ...(j.timings.prompt_n != null ? {
          prompt_n: j.timings.prompt_n, predicted_n: j.timings.predicted_n,
          prompt_per_second: j.timings.prompt_per_second,
          predicted_per_second: j.timings.predicted_per_second,
          prompt_ms: j.timings.prompt_ms, predicted_ms: j.timings.predicted_ms,
          draft_n: j.timings.draft_n, draft_n_accepted: j.timings.draft_n_accepted,
          cache_n: j.timings.cache_n,
        } : j.timings) };
        if (j.strata_mcp) onTool(m, j.strata_mcp);
        const d = (j.choices && j.choices[0] && j.choices[0].delta) || {};
        const lastTool = m.tools && m.tools.length ? m.tools[m.tools.length - 1] : null;   // a new round after a tool
        if (d.reasoning_content) {
          if (!firstAt) firstAt = performance.now();
          if (!thinkStart) thinkStart = performance.now();
          if (lastTool && m.reasoning && lastTool.rat === m.reasoning.length) m.reasoning += "\n\n";
          m.reasoning += d.reasoning_content;
        }
        if (d.content) {
          if (!firstAt) firstAt = performance.now();
          if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000;
          if (lastTool && m.text && lastTool.at === m.text.length) m.text += "\n\n";
          m.text += d.content;
        }
        if (!frame) frame = requestAnimationFrame(paint);
      }
    }
  } catch (e) {
    if (e.name === "AbortError") m.stopped = true;
    else { m.error = e.message || String(e); toast("error", "The request failed", m.error, 6000); }
  }
  if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000;
  // llama-server answers a streaming request with `timings` (merged into usage above) and
  // no `completion_tokens`; predicted_n is the generated-token count there.
  const n = usage ? (usage.completion_tokens ?? usage.predicted_n) : null;
  if (n && firstAt) {
    const secs = (performance.now() - firstAt) / 1000;
    m.meta = `${fmt(n)} tokens${secs > 0.25 ? ` · ${fmt(n / secs, 1)} tok/s` : ""}` +
             (m.stopped ? " · stopped" : "") +
             (m.hitLimit ? " · hit the server's max-token limit (-n); raise it or set Max tokens in the sampling drawer" : "") +
             (projectionLoaded() ? (settings.esp ? " · projection on" : " · projection off") : "");
  } else if (m.stopped) {
    m.meta = "Stopped";
  }
  for (const t of m.tools || []) if (t.state === "writing" || t.state === "running") { t.state = "skipped"; t.ms = null; }
  const ran = (m.tools || []).filter((t) => t.state === "done" || t.state === "error").length;
  if (ran) m.meta = `${m.meta ? `${m.meta} · ` : ""}${ran} tool call${ran > 1 ? "s" : ""}`;
  if (m.limit) m.meta = `${m.meta || ""} · stopped at the limit of ${m.limit} tool rounds (mcp.max_rounds)`;
  busy = null;
  setBusy(false);
  if (frame) cancelAnimationFrame(frame);
  updateAssistant(el, m, false);
  saveChat();
  scrollDown();
  if (window.SM70) SM70.reportRequest({
    prompt_n: (usage && (usage.prompt_n ?? usage.prompt_tokens)) || 0,
    predicted_n: (usage && (usage.predicted_n ?? usage.completion_tokens)) || 0,
    prompt_per_second: usage ? usage.prompt_per_second : null,
    predicted_per_second: usage ? usage.predicted_per_second : null,
    prompt_ms: usage ? usage.prompt_ms : null,
    predicted_ms: usage ? usage.predicted_ms : null,
    ttft_ms: firstAt ? Math.round(firstAt - tSubmit) : null,
    draft_n: (usage && usage.draft_n) || 0,
    draft_n_accepted: (usage && usage.draft_n_accepted) || 0,
    cache_n: (usage && (usage.cache_n ?? usage.prompt_tokens_cached)) || 0,
  });
}

$("composer").onsubmit = (e) => { e.preventDefault(); send(); };
$("stop-btn").onclick = () => { if (busy) busy.controller.abort(); };
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
});
function autosize() { const t = $("input"); t.style.height = "auto"; t.style.height = `${Math.min(t.scrollHeight, innerHeight * 0.4)}px`; }
$("input").addEventListener("input", autosize);

$("new-btn").onclick = () => {
  if (busy) { toast("warn", "Still writing", "Stop the answer first."); return; }
  if (!messages.length) return;
  const backup = messages;
  messages = [];
  saveChat();
  renderChat();
  toast("info", "New chat", "The last one was cleared.", 6000, {label: "Undo", run: () => { messages = backup; saveChat(); renderChat(); }});
};
$("export-btn").onclick = () => {
  if (!messages.length) { toast("info", "Nothing to save yet"); return; }
  const tools = (m) => (m.tools || []).filter((t) => t.result != null).map((t) =>
    `<details><summary>Tool ${t.server ? `${t.server} / ` : ""}${t.tool || t.name}${t.ok ? "" : " (error)"}</summary>\n\n` +
    `\`\`\`json\n${JSON.stringify(t.arguments || {}, null, 2)}\n\`\`\`\n\n\`\`\`\n${t.result}\n\`\`\`\n\n</details>\n\n`).join("");
  const md = messages.map((m) => m.role === "user" ? `## You\n\n${m.text}\n` :
    `## ${health.model}\n\n${m.reasoning ? `<details><summary>Thinking</summary>\n\n${m.reasoning}\n\n</details>\n\n` : ""}${tools(m)}${m.text || m.error || ""}\n`).join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([md], {type: "text/markdown"}));
  a.download = `sm70-chat-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-")}.md`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
};

// pictures and text files: the attach button, dropping them on the chat, or pasting a picture (issue #30)
const TEXT_EXT = /\.(txt|md|markdown|rst|tex|py|pyi|ipynb|js|mjs|cjs|ts|tsx|jsx|vue|svelte|json|jsonl|csv|tsv|log|ya?ml|toml|ini|cfg|conf|env|xml|html?|css|scss|less|c|cc|cpp|cxx|h|hh|hpp|cu|cuh|rs|go|java|kt|kts|swift|rb|php|pl|lua|r|jl|scala|sql|sh|bash|zsh|fish|ps1|psm1|bat|cmd|diff|patch|gradle|cmake|mk|dockerfile|gitignore|proto|graphql)$/i;
const MAX_TEXT_FILE = 512 * 1024;
function isTextFile(f) {
  return f.type.startsWith("text/") || /json|xml|javascript|yaml|toml|x-sh|x-python/.test(f.type) ||
         TEXT_EXT.test(f.name) || /(^|[\\/])(makefile|dockerfile|readme|license)$/i.test(f.name);
}
function addFiles(files) {
  for (const f of files) {
    if (f.type.startsWith("image/")) {
      if (!health.images) { toast("warn", "Pictures are off", "This model was set up for text only."); continue; }
      if (f.size > 20e6) { toast("warn", "Picture too large", `${f.name} is over 20 MB.`); continue; }
      const r = new FileReader();
      r.onload = () => { attachments.push({kind: "image", name: f.name || "pasted image", url: r.result}); renderAttachments(); };
      r.readAsDataURL(f);
      continue;
    }
    if (!isTextFile(f)) { toast("warn", "Not a text file", `${f.name}: attach text files (code, notes, logs, data)${health.images ? " or pictures" : ""}.`); continue; }
    if (f.size > MAX_TEXT_FILE) { toast("warn", "File too large", `${f.name} is over 512 KB.`); continue; }
    const r = new FileReader();
    r.onload = () => {
      const text = String(r.result);
      if (text.includes("\u0000")) { toast("warn", "Not a text file", `${f.name} looks like a binary file.`); return; }
      attachments.push({kind: "file", name: f.name, text});
      renderAttachments();
    };
    r.readAsText(f);
  }
}
// a file's text in the message, fenced with more backticks than it contains itself
function fileBlock(f) {
  const longest = Math.max(2, ...(f.text.match(/`+/g) || []).map((s) => s.length));
  const fence = "`".repeat(longest + 1);
  return `File: ${f.name}\n${fence}\n${f.text}\n${fence}`;
}
function userText(m) {
  const files = (m.files || []).filter((f) => f.text != null);
  return [m.text, ...files.map(fileBlock)].filter((s) => s).join("\n\n");
}
function renderAttachments() {
  const box = $("attachments");
  box.hidden = !attachments.length;
  box.innerHTML = "";
  attachments.forEach((a, i) => {
    const c = document.createElement("span");
    c.className = "chip";
    c.innerHTML = icon(a.kind === "file" ? "attach" : "image", "st-icon st-icon--sm");
    c.append(a.name);
    const x = document.createElement("button");
    x.type = "button"; x.className = "st-btn st-btn--icon"; x.setAttribute("aria-label", "Remove");
    x.innerHTML = icon("trash");
    x.onclick = () => { attachments.splice(i, 1); renderAttachments(); };
    c.appendChild(x);
    box.appendChild(c);
  });
}
$("attach-btn").onclick = () => $("file").click();
// drop files on the chat or the message box
for (const id of ["chat", "composer"]) {
  const el = $(id);
  el.addEventListener("dragover", (e) => {
    if (![...(e.dataTransfer || {}).types || []].includes("Files")) return;
    e.preventDefault();
    $("composer").classList.add("dragging");
  });
  el.addEventListener("dragleave", () => $("composer").classList.remove("dragging"));
  el.addEventListener("drop", (e) => {
    $("composer").classList.remove("dragging");
    if (!e.dataTransfer || !e.dataTransfer.files.length) return;
    e.preventDefault();
    addFiles(e.dataTransfer.files);
    $("input").focus();
  });
}
$("file").onchange = () => { addFiles($("file").files); $("file").value = ""; };
$("input").addEventListener("paste", (e) => {
  const dt = e.clipboardData;
  if (!dt) return;
  // copied pictures are not reliably in clipboardData.files (it depends on the
  // app that copied them); the items list is the dependable place
  let files = [...(dt.files || [])].filter((f) => f.type.startsWith("image/"));
  if (!files.length) {
    for (const it of dt.items || []) {
      if (it.kind === "file" && it.type.startsWith("image/")) {
        const f = it.getAsFile();
        if (f) files.push(f);
      }
    }
  }
  if (!files.length) return;                  // a plain text paste: keep the default
  e.preventDefault();
  addFiles(files);        // it explains "Pictures are off" when no vision head is loaded
});

// sm70: the control-vector "speed projection" is a Strata engine feature; this
// fork has no such input, so the toggle row stays hidden and the field never
// reaches the request body.
function projectionLoaded() {
  return false;
}

// ------------------------------------------------------------------ the sampling drawer
function openDrawer(open) {
  $("drawer").dataset.open = String(open);
  $("drawer").setAttribute("aria-hidden", String(!open));
  $("scrim").hidden = !open;
  if (open) { loadDrawer(); loadShared(); loadMcp(); }
}
function loadDrawer(s = settings) {
  for (const b of $("s-thinking").children) b.setAttribute("aria-checked", String(b.dataset.v === s.thinking));
  $("s-temp").value = s.temperature; $("s-topp").value = s.top_p; $("s-topk").value = s.top_k;
  $("s-max").value = s.max; $("s-seed").value = s.seed;
  $("s-show").setAttribute("aria-checked", String(!!s.show));
  $("s-esp").setAttribute("aria-checked", String(s.esp !== false));
  $("esp-row").hidden = !projectionLoaded();
  $("s-mcp").setAttribute("aria-checked", String(s.mcp !== false));
  $("s-share").setAttribute("aria-checked", String(sharedOn));
  outputs();
}
function outputs() {
  $("o-thinking").textContent = settings.thinking === "none" ? "off"
    : settings.thinking === "xhigh" ? "high" : settings.thinking;
  $("o-temp").textContent = fmt(settings.temperature, 2);
  $("o-topp").textContent = fmt(settings.top_p, 2);
  $("o-topk").textContent = fmt(settings.top_k, 0);
}
$("s-thinking").addEventListener("click", (e) => {
  const b = e.target.closest("[data-v]");
  if (!b) return;
  settings.thinking = b.dataset.v;
  loadDrawer(); saveChat();
});
$("s-temp").oninput = (e) => { settings.temperature = +e.target.value; outputs(); };
$("s-topp").oninput = (e) => { settings.top_p = +e.target.value; outputs(); };
$("s-topk").oninput = (e) => { settings.top_k = +e.target.value; outputs(); };
$("s-max").onchange = (e) => { settings.max = e.target.value; saveChat(); };
$("s-seed").onchange = (e) => { settings.seed = e.target.value; saveChat(); };
$("s-show").onclick = () => { settings.show = !settings.show; $("s-show").setAttribute("aria-checked", String(!!settings.show)); saveChat(); };
$("s-apply").onclick = () => {
  const sel = [...document.getElementById("s-thinking").children].find((b) => b.getAttribute("aria-checked") === "true");
  settings = {thinking: sel ? sel.dataset.v : DEFAULTS.thinking, temperature: +document.getElementById("s-temp").value,
              top_p: +document.getElementById("s-topp").value, top_k: +document.getElementById("s-topk").value,
              max: document.getElementById("s-max").value.trim(), seed: document.getElementById("s-seed").value.trim(),
              show: document.getElementById("s-show").getAttribute("aria-checked") === "true",
              esp: false, mcp: false};
  store.set("sampling", settings);
  openDrawer(false);
  const hint = document.getElementById("composer-hint");
  hint.textContent = "Sampling updated · applies to the next message";
  setTimeout(() => { hint.textContent = "Shift+Enter: new line"; }, 4000);
};
$("s-reset").onclick = () => { settings = {...DEFAULTS}; loadDrawer(); saveChat(); };
$("sampling-btn").onclick = () => openDrawer(true);
$("drawer-close").onclick = () => openDrawer(false);
$("scrim").onclick = () => openDrawer(false);
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && $("drawer").dataset.open === "true") openDrawer(false); });

// "Use for other apps too": the server keeps these settings as every client's defaults (GET/POST /settings)
let sharedOn = false;
async function loadShared() {
  /* sm70: settings are local to this browser; the switch stays off */
}
function sharedDefaults(s) {
  const d = {reasoning_effort: s.thinking};
  if (!(+s.temperature > 0)) d.temperature = 0;
  else if (!samplingUntouched(s)) Object.assign(d, {temperature: +s.temperature, top_p: +s.top_p, top_k: +s.top_k});
  if (s.seed) d.seed = +s.seed;
  if (s.max) d.max_tokens = +s.max;
  if (projectionLoaded()) d.experimental_speed_projection = s.esp !== false;
  return d;
}
async function saveShared(on, s) {
  /* sm70: settings stay in this browser */
}

// ------------------------------------------------------------------ start
setBusy(false);
renderChat();
const startQuestion = new URLSearchParams(location.search).get("q");   // /?q=... starts a chat (a shortcut)
if (startQuestion) history.replaceState(null, "", location.pathname + location.hash);
loadHealth().then(loadMcp).then(() => { if (startQuestion) { $("input").value = startQuestion; send(); } });
showTab(location.hash.slice(1) || "chat");
poll();

// ------------------------------------------------------------------ images (POST /v1/images/generations)
$("img-form").onsubmit = async (ev) => {
  ev.preventDefault();
  const prompt = $("img-prompt").value.trim();
  if (!prompt) return;
  const body = {prompt, size: $("img-size").value, response_format: "b64_json"};
  if ($("img-seed").value !== "") body.seed = Number($("img-seed").value);
  $("img-go").disabled = true;
  const t0 = performance.now();
  $("img-hint").textContent = "Generating\u2026 (a minute or more on a small GPU)";
  try {
    const r = await fetch("v1/images/generations", {method: "POST", headers: headers(true), body: JSON.stringify(body)});
    const j = await r.json();
    if (!r.ok) throw new Error((j.error && j.error.message) || r.statusText);
    for (const d of j.data) {
      const fig = document.createElement("figure");
      const img = document.createElement("img");
      img.src = "data:image/png;base64," + d.b64_json;
      img.alt = prompt;
      const cap = document.createElement("figcaption");
      cap.textContent = `${j.size}, seed ${d.seed}, ${((performance.now() - t0) / 1000).toFixed(0)} s: ${prompt}`;
      fig.append(img, cap);
      $("img-results").prepend(fig);
    }
    $("img-hint").textContent = "Done. Right-click an image to save it.";
  } catch (e) {
    $("img-hint").textContent = "";
    toast("error", "No image", String(e.message || e), 8000);
  } finally {
    $("img-go").disabled = false;
  }
};

// ------------------------------------------------------------------ shared UI state
// Everything above reads its first value from localStorage. This brings the
// launcher's copy in: if the shared file does not exist yet it is seeded from this
// client (migration for an existing browser profile), otherwise the launcher wins
// and the page repaints. Writes from store.set() go back to the file, and a focus
// after another client changed something adopts that.
function adoptUiState(s, keepChat) {
  if (!s || typeof s !== "object") return;
  if (s.theme === "light" || s.theme === "dark") {
    setTheme(s.theme, false);
    try { localStorage.setItem("strata.theme", s.theme); } catch (e) { /* ignore */ }
  }
  if (s.lang && window.SM70I18n) {
    SM70I18n.setPref(s.lang, false);
    try { localStorage.setItem("strata.lang", s.lang); } catch (e) { /* ignore */ }
    const picker = $("lang-select");
    if (picker) picker.value = s.lang;
  }
  if (typeof s.apikey === "string") {
    store.cache("apikey", s.apikey);
    const input = $("api-key");
    if (input) input.value = s.apikey;
  }
  if (s.sampling && typeof s.sampling === "object") {
    settings = {...settings, ...s.sampling};
    store.cache("sampling", settings);
    loadDrawer();
  }
  if (Array.isArray(s.chat) && !busy && !keepChat) {      // never rewrite an active conversation
    messages = s.chat;
    store.cache("chat", messages);
    renderChat();
  }
}
async function syncUiState(first) {
  const localLen = messages.length;           // before the round trip: did the user type?
  try {
    const r = await fetch("ui-state");
    if (!r.ok) return;                        // page served without the launcher
    const j = await r.json();
    const s = j.state || {};
    if (!Object.keys(s).length) {
      // no shared file yet: seed it from this client's cache when there is something
      // to keep. A client with nothing to say leaves the file absent, so a client
      // that does have history is still the one that seeds it.
      const seed = {chat: store.get("chat", []), sampling: store.get("sampling", {}),
                    apikey: store.get("apikey", "")};
      let theme = ""; try { theme = localStorage.getItem("strata.theme") || ""; } catch (e) { /* ignore */ }
      if (theme === "light" || theme === "dark") seed.theme = theme;
      seed.lang = window.SM70I18n ? SM70I18n.pref() : "system";
      const hasData = seed.chat.length || seed.apikey || Object.keys(seed.sampling).length;
      if (hasData) {
        const w = await fetch("ui-state", {method: "POST", headers: {"Content-Type": "application/json"},
                                           body: JSON.stringify(seed)});
        if (w.ok) { const wj = await w.json(); if (wj.at) uiStateAt = wj.at; }
      }
      store.ready = true;
      return;
    }
    if (first || (j.at || 0) > uiStateAt) {   // another client wrote since we looked
      let typedWhileFetching = false;
      if (first && Array.isArray(s.chat) && messages.length > localLen) {
        // the user sent a message before the fetch returned: theirs is newer than
        // the file, so publish it instead of replacing their screen
        typedWhileFetching = true;
      }
      adoptUiState(s, typedWhileFetching);
      uiStateAt = j.at || Date.now();
      store.ready = true;
      if (typedWhileFetching) store.set("chat", messages);   // now allowed to POST
      return;
    }
    store.ready = true;
  } catch (e) { /* no launcher: localStorage-only behaviour, exactly as before */ }
}
syncUiState(true);
window.addEventListener("focus", () => { if (!busy) syncUiState(false); });
