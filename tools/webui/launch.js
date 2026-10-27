// Launch tab: model picker, device switches, basic server arguments, a free-text
// field for the rest, the environment catalog and a live command preview.
// All strings/values come from GET /config, built server-side from
// tools/sm70gui/sm70_config.py — this file only renders and posts the selection
// back (/preview, /start, /save).
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const S = {
    cfg: null, env: {}, args: {}, extra: "", devices: [],
    busy: false, timer: 0, logTimer: 0, wired: false,
  };

  const T = (k, f) => (window.SM70I18n ? window.SM70I18n.t(k, f) : (f || k));
  const lang = () => (window.SM70I18n ? window.SM70I18n.lang() : "en");
  const descOf = (e) => (lang() === "zh" ? (e.desc_zh || e.desc) : e.desc);
  const labelOf = (f) => (lang() === "zh" ? (f.label_zh || f.label) : f.label);
  const groupTitle = (gid) => {
    const g = S.cfg && S.cfg.env_groups ? S.cfg.env_groups[gid] : null;
    return (g && (g[lang()] || g.en)) || gid;
  };

  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  };
  const debounce = (fn, ms) => {
    let t = 0;
    return (...args) => {
      clearTimeout(t);
      t = setTimeout(() => fn(...args), ms);
    };
  };
  const api = async (path, opt) => {
    const r = await fetch(path, opt);
    if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
    return r.json();
  };
  const post = (path, body) => api(path, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });

  const defaults = () => (S.cfg ? S.cfg.env_defaults : {});
  const valueOf = (name) => (name in S.env ? S.env[name] : defaults()[name]);

  // ------------------------------------------------------------------ render

  // The model is free text: a local .gguf (-> -m) or user/repo[:quant] (-> -hf).
  // The datalist only suggests what the folder scan found.
  function renderModel() {
    const inp = $("launch-model");
    const list = $("launch-model-list");
    list.textContent = "";
    for (const m of S.cfg.models) {
      const o = el("option");
      o.value = m.path;
      o.label = `${m.name} · ${(m.size / 1e9).toFixed(1)} GB`;
      list.append(o);
    }
    if (!inp.value) inp.value = S.cfg.selected_model || "";
    updateModelFacts();
  }

  function updateModelFacts() {
    const v = $("launch-model").value.trim();
    const m = S.cfg && S.cfg.models.find((x) => x.path === v);
    $("launch-model-facts").textContent = m ? modelFacts(m)
      : (v ? T("launch.notLocal", "not a scanned .gguf — launched as -hf <repo>") : "");
  }

  function modelFacts(m) {
    if (m.error) return `${T("launch.headerFailed", "header read failed")}: ${m.error}`;
    const types = Object.entries(m.types).sort((a, b) => b[1] - a[1])
      .map(([t, n]) => `${t}×${n}`).join("  ");
    const bits = [`arch ${m.arch || "?"}`, `${m.block_count || "?"} layers`];
    if (m.has_mtp) bits.push("MTP layers");
    if (m.size < 2e9) bits.push(T("launch.smallFile", "small file — check it is complete"));
    return `${bits.join(" · ")}\n${types}`;
  }

  // One row per CUDA device. Only SM70 gets a switch: the kernels are Volta-only,
  // so offering the desktop card would just produce a launch that dies.
  // ---- config files -------------------------------------------------------
  const profSel = () => $("launch-prof-name").value.trim();

  function renderProfiles(list) {
    const box = $("launch-profiles");
    box.textContent = "";
    $("launch-prof-dir").textContent = S.cfg && S.cfg.prof_dir ? S.cfg.prof_dir : "";
    if (!list || !list.length) {
      box.append(el("div", "muted small", T("launch.profEmpty", "no saved files yet")));
      return;
    }
    for (const p of list) {
      const row = el("div", "recipe");
      row.dataset.active = String(p.name === profSel());
      const head = el("div", "recipe__head");
      head.append(el("span", "recipe__title", p.name));
      const meta = el("span", "recipe__badges");
      meta.append(el("span", "st-badge", T("launch.profLoad", "load")));
      head.append(meta);
      row.append(head);
      const bits = [];
      if (p.model) bits.push(p.model);
      if (p.ctx) bits.push(`-c ${p.ctx}`);
      if (p.np) bits.push(`-np ${p.np}`);
      if (p.alias) bits.push(`alias ${p.alias}`);
      row.append(el("div", "recipe__meta", bits.join(" · ") || "—"));
      row.onclick = async () => {
        $("launch-prof-name").value = p.name;
        await loadProfile(p.name);
      };
      box.append(row);
    }
  }

  async function refreshProfiles() {
    try {
      const r = await api("/profiles");
      S.profiles = r.profiles;
      renderProfiles(r.profiles);
    } catch (e) { /* launcher restarting */ }
  }

  async function profOp(op, body) {
    const msg = $("launch-prof-msg");
    msg.textContent = T("launch.profWorking", "working…");
    try {
      const r = await post(`/profiles/${op}`, body);
      if (!r.ok) {
        msg.textContent = `${T("launch.failed", "failed")}: ${r.message}`;
      } else {
        msg.textContent = T("launch.profDone", "done");
        setTimeout(() => { if (msg.textContent === T("launch.profDone", "done")) msg.textContent = ""; }, 1500);
      }
      if (r.profiles) renderProfiles(r.profiles);
      return r;
    } catch (e) {
      msg.textContent = `${T("launch.failed", "failed")}: ${e}`;
      return null;
    }
  }

  async function loadProfile(name) {
    const r = await profOp("load", { name });
    if (r && r.ok) { await load(true); }
  }

  function wireProfiles() {
    $("launch-prof-save").onclick = async () => {
      const name = profSel();
      if (!name) { $("launch-prof-msg").textContent = T("launch.profNeedName", "type a name first"); return; }
      await savePrefs();                    // snapshot what the form shows
      await profOp("save", { name });
      await refreshProfiles();
    };
    $("launch-prof-rename").onclick = async () => {
      const from = profSel();
      const to = (prompt(T("launch.profNewName", "new name"), from) || "").trim();
      if (!from || !to || to === from) return;
      await profOp("rename", { name: from, to });
      $("launch-prof-name").value = to;
    };
    $("launch-prof-delete").onclick = async () => {
      const name = profSel();
      if (!name || !confirm(T("launch.profDeleteAsk", `Delete ${name}?`))) return;
      await profOp("delete", { name });
      $("launch-prof-name").value = "";
    };
    $("launch-prof-export").onclick = async () => {
      const r = await profOp("export", {});
      if (r && r.ok) $("launch-prof-msg").textContent = T("launch.profExported", "written");
    };
    $("launch-prof-import").onclick = async () => {
      const r = await profOp("import", {});
      if (r && r.ok) await load(true);
    };
  }

  function renderDevices() {
    const box = $("launch-devices");
    box.textContent = "";
    if (!S.cfg || !S.cfg.gpus || !S.cfg.gpus.length) {
      box.append(el("div", "muted small", T("launch.noDevices", "no CUDA devices detected")));
      return;
    }
    if (S.cfg.cuda_warn) box.append(el("div", "launch-warn muted small", `⚠ ${S.cfg.cuda_warn}`));
    for (const d of S.cfg.gpus) {
      const row = el("div", "launch-dev");
      row.append(el("span", "launch-dev__idx", `${T("launch.cuda", "CUDA")} ${d.cuda}`));
      row.append(el("span", "launch-dev__name", d.name));
      const badge = d.sm70 ? `cc ${d.cc} · SM70`
                           : `cc ${d.cc} · ${T("launch.unsupported", "unsupported")}`;
      row.append(el("span", `st-badge ${d.sm70 ? "st-badge--generating" : "st-badge--error"}`, badge));
      if (d.vram_mib) row.append(el("span", "muted small", `${Math.round(d.vram_mib / 1024)} GB`));
      row.append(el("span", "launch-dev__spacer"));
      if (d.sm70) {
        const sw = el("button", "st-toggle");
        sw.type = "button";
        sw.setAttribute("role", "switch");
        sw.setAttribute("aria-label", `${T("launch.cuda", "CUDA")} ${d.cuda}`);
        sw.setAttribute("aria-checked", String(S.devices.includes(d.cuda)));
        sw.onclick = () => {
          const on = sw.getAttribute("aria-checked") !== "true";
          sw.setAttribute("aria-checked", String(on));
          S.devices = on ? [...S.devices, d.cuda].sort((a, b) => a - b)
                         : S.devices.filter((i) => i !== d.cuda);
          renderCudaVisible();
          refreshPreview();
          savePrefs();
        };
        row.append(sw);
      } else {
        row.append(el("span", "muted small", T("launch.notSelectable", "not selectable")));
      }
      box.append(row);
    }
    renderCudaVisible();
  }

  function renderCudaVisible() {
    const v = S.devices.join(",");
    $("launch-cuda-visible").textContent = v
      ? `CUDA_VISIBLE_DEVICES=${v}` : T("launch.noCard", "no device selected");
  }

  function renderArgs() {
    const box = $("launch-args");
    box.textContent = "";
    if (!S.cfg) return;
    for (const f of S.cfg.arg_fields) {
      if (f.card) continue;                 // rendered inside the Model card
      const field = el("div", "launch-arg");
      field.append(el("label", "st-label", labelOf(f)));
      let ctl;
      if (f.kind === "enum") {
        ctl = el("select", "st-input");
        for (const c of f.choices) {
          const o = el("option", null, c);
          o.value = c;
          ctl.append(o);
        }
      } else {
        ctl = el("input", "st-input");
        ctl.type = f.kind === "int" ? "number" : "text";
        ctl.spellcheck = false;
      }
      const base = (f.key in S.args) ? S.args[f.key] : S.cfg.args[f.key];
      ctl.value = base != null ? base : "";
      ctl.dataset.key = f.key;
      ctl.oninput = () => {
        S.args = { ...S.args, [f.key]: ctl.value };
        schedulePreview();
        scheduleSave();
      };
      ctl.onchange = () => {
        S.args = { ...S.args, [f.key]: ctl.value };
        refreshPreview();
        savePrefs();
      };
      field.append(ctl);
      box.append(field);
    }
  }

  function renderModelCardFields() {
    const box = $("launch-model-extra");
    box.textContent = "";
    if (!S.cfg) return;
    for (const f of S.cfg.arg_fields) {
      if (f.card !== "model") continue;
      const field = el("div", "st-field");
      field.append(el("label", "st-label", labelOf(f)));
      const ctl = el("input", "st-input");
      ctl.type = "text";
      ctl.spellcheck = false;
      ctl.value = (f.key in S.args ? S.args[f.key] : S.cfg.args[f.key]) != null
        ? (f.key in S.args ? S.args[f.key] : S.cfg.args[f.key]) : "";
      if (f.key === "draft_model") {
        ctl.setAttribute("list", "launch-model-list");
        ctl.placeholder = T("launch.draftPh", "optional, a second .gguf");
      }
      if (f.key === "mmproj") {
        ctl.setAttribute("list", "launch-model-list");
        ctl.placeholder = T("launch.mmprojPh", "optional, the vision head .gguf (mmproj)");
      }
      ctl.oninput = () => {
        S.args = { ...S.args, [f.key]: ctl.value };
        schedulePreview();
        scheduleSave();
      };
      ctl.onchange = () => {
        S.args = { ...S.args, [f.key]: ctl.value };
        refreshPreview();
        savePrefs();
      };
      const pick = el("button", "st-btn st-btn--secondary", T("launch.browse", "Browse…"));
      pick.type = "button";
      pick.onclick = () => browse("file", null, null, ctl);
      const row = el("div", "launch-dir");
      row.append(ctl, pick);
      field.append(row);
      box.append(field);
    }
  }

  function envRow(e) {
    const row = el("div", "env-row");
    const cur = valueOf(e.name);
    row.dataset.changed = String(e.name in S.env && S.env[e.name] !== e.default);

    const name = el("div", "env-row__name");
    name.append(el("span", null, e.name));
    if (e.risk === "debug") name.append(el("span", "env-row__tag", T("launch.debug", "debug")));

    const ctl = el("div", "env-row__ctl");
    let input;
    if (e.kind === "bool") {
      input = el("button", "st-toggle");
      input.type = "button";
      input.setAttribute("role", "switch");
      input.setAttribute("aria-checked", String(cur === "1"));
      input.setAttribute("aria-label", e.name);
      input.onclick = () => {
        const on = input.getAttribute("aria-checked") !== "true";
        input.setAttribute("aria-checked", String(on));
        setEnv(e.name, on ? "1" : "0");
      };
    } else if (e.kind === "enum") {
      input = el("select", "st-input");
      for (const c of e.choices) {
        const o = el("option", null, c);
        o.value = c;
        input.append(o);
      }
      input.value = cur;
      input.onchange = () => setEnv(e.name, input.value);
    } else {
      input = el("input", "st-input st-input--compact");
      input.type = e.kind === "int" ? "number" : "text";
      input.spellcheck = false;
      input.value = cur;
      input.oninput = () => setEnvLive(e.name, input.value, row);
      input.onchange = () => setEnvLive(e.name, input.value, row);
    }
    ctl.append(input);

    const desc = el("div", "env-row__desc");
    desc.append(el("span", null, descOf(e)));
    desc.append(el("span", "env-row__default",
                   `${T("launch.default", "default")} ${e.default || "(empty)"}`));

    row.append(name, ctl, desc);
    return row;
  }

  function renderEnv() {
    if (!S.cfg) return;
    const onlyChanged = $("launch-only-rec").checked;
    const q = $("launch-filter").value.trim().toLowerCase();
    const defs = defaults();
    const pass = (e) => {
      if (e.derived) return false;
      if (onlyChanged) {
        const cur = e.name in S.env ? S.env[e.name] : defs[e.name];
        if (cur === e.default) return false;
      }
      if (q && !`${e.name} ${e.desc} ${e.desc_zh || ""}`.toLowerCase().includes(q)) return false;
      return true;
    };
    const fill = (box, tier) => {
      box.textContent = "";
      let shown = 0;
      for (const gid of Object.keys(S.cfg.env_groups)) {
        const vars = S.cfg.env_vars.filter((e) => e.group === gid && e.tier === tier && pass(e));
        if (!vars.length) continue;
        box.append(el("div", "env-group", groupTitle(gid)));
        for (const e of vars) box.append(envRow(e));
        shown += vars.length;
      }
      if (!shown) box.append(el("div", "muted small", T("launch.nothing", "nothing matches")));
      return shown;
    };
    fill($("launch-env"), "common");
    fill($("launch-env-adv"), "advanced");
    const count = (tier) => S.cfg.env_vars.filter((e) => !e.derived && e.tier === tier).length;
    $("launch-env-count").textContent = String(count("common"));
    $("launch-adv-count").textContent = `(${count("advanced")})`;
  }

  // typing in a text/number cell: keep the node (a re-render would steal focus),
  // refresh the preview and save shortly after the last keystroke
  function setEnvLive(name, value, row) {
    if (String(value) === String(defaults()[name])) delete S.env[name];
    else S.env[name] = String(value);
    S.env = { ...S.env };
    if (row) row.dataset.changed = String(name in S.env);
    schedulePreview();
    scheduleSave();
  }

  function setEnv(name, value) {
    if (String(value) === String(defaults()[name])) delete S.env[name];
    else S.env[name] = String(value);
    S.env = { ...S.env };
    renderEnv();
    refreshPreview();
    savePrefs();
  }

  function renderWarns(warns) {
    const box = $("launch-warns");
    box.textContent = "";
    for (const w of warns || []) box.append(el("div", "launch-warn muted small", `⚠ ${w}`));
  }

  function renderAll() {
    renderModel();
    renderModelCardFields();
    renderDevices();
    renderArgs();
    $("launch-extra").value = S.extra;
    renderEnv();
    refreshPreview();
  }

  // ------------------------------------------------------------- preview/state

  const selection = () => ({
    model: $("launch-model").value, args: S.args, env: S.env,
    extra: $("launch-extra").value, devices: S.devices,
  });

  let previewSeq = 0;

  async function refreshPreview() {
    if (!S.cfg) return;
    clearTimeout(S.timer);
    S.timer = setTimeout(async () => {
      const mine = ++previewSeq;
      try {
        const p = await post("/preview", selection());
        if (mine !== previewSeq) return;    // a newer keystroke already answered
        $("launch-cmd").textContent = p.preview;
        renderWarns(p.warns);
      } catch (e) {
        if (mine !== previewSeq) return;
        $("launch-cmd").textContent = String(e);
      }
    }, 120);
  }

  async function refreshState() {
    try {
      const s = await api("/server");
      const pill = $("launch-state");
      pill.textContent = s.running
        ? (s.owned ? `${T("launch.running", "running")} · port ${s.port}`
                   : `${T("launch.running", "running")} · port ${s.port} · ${T("launch.external", "started outside this launcher")}`)
        : T("launch.stopped", "stopped");
      pill.className = `st-badge ${s.owned ? "st-badge--generating" : s.running ? "st-badge--queued" : ""}`;
      $("launch-start").disabled = s.running || S.busy;
      // Stop only kills what this process spawned; an external server is not ours
      $("launch-stop").disabled = !s.owned || S.busy;
      $("launch-stop").title = s.running && !s.owned
        ? T("launch.external", "started outside this launcher") : "";
      $("launch-restart").disabled = S.busy;
    } catch (e) { /* launcher restarting */ }
  }

  async function refreshLog() {
    try {
      const l = await api("/serverlog");
      const box = $("launch-log");
      const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 24;
      box.textContent = (l.lines || []).slice(-80).join("\n");
      if (atBottom) box.scrollTop = box.scrollHeight;
    } catch (e) { /* ignore */ }
  }

  async function load(force = false) {
    try {
      const cfg = await api("/config");
      S.cfg = cfg;
      // force: take the saved state verbatim (profile load / import / reset). Without
      // it the form keeps whatever the operator typed, which is what a plain refresh
      // of /config (tab switch, language change) should do.
      if (force || !$("launch-dir").value) $("launch-dir").value = cfg.saved.models_dir || "";
      if (force || !Object.keys(S.env).length) S.env = { ...cfg.env };
      delete S.env.CUDA_VISIBLE_DEVICES;   // derived from the switches
      if (force || !Object.keys(S.args).length) S.args = { ...(cfg.args || {}) };
      if (force || S.extra === "") S.extra = cfg.extra || "";
      S.devices = (cfg.devices && cfg.devices.length) ? cfg.devices.slice() : (cfg.sm70 || []);
      renderAll();
      refreshProfiles();
      refreshState();
      refreshLog();
      const sp = cfg.saved && cfg.saved.prefs_path;
      $("launch-saved").textContent = cfg.saved && cfg.saved.prefs_exists
        ? T("launch.restored", "restored from the saved file")
        : T("launch.fresh", "built-in defaults (nothing saved yet)");
      $("launch-saved").title = sp ? `${T("launch.savedTo", "saved to")} ${sp}` : "";
    } catch (e) {
      $("launch-cmd").textContent = `${T("launch.notReachable", "launcher not reachable")}: ${e}`;
    }
  }

  function showSaved(at, path) {
    const box = $("launch-saved");
    if (!box) return;
    box.textContent = `${T("launch.saved", "saved")} ${at || ""}`;
    box.title = path ? `${T("launch.savedTo", "saved to")} ${path}` : "";
  }

  async function savePrefs() {
    try {
      const defs = defaults();
      const changed = Object.fromEntries(
        Object.entries(S.env).filter(([k, v]) => v !== defs[k] && k !== "CUDA_VISIBLE_DEVICES"));
      const r = await post("/save", {
        args: S.args, env: changed, extra: $("launch-extra").value,
        devices: S.devices, model: $("launch-model").value,
        models_dir: $("launch-dir").value,
      });
      showSaved(r.at, r.path);
    } catch (e) { /* preferences are best-effort */ }
  }

  const schedulePreview = debounce(() => refreshPreview(), 120);
  const scheduleSave = debounce(() => savePrefs(), 500);

  async function resetAll() {
    if (!confirm(T("launch.resetConfirm",
      "Reset every launch setting (arguments, environment, devices, extra arguments) to the built-in defaults?"))) {
      return;
    }
    try { await post("/reset", {}); } catch (e) { /* ignore */ }
    S.env = {}; S.args = {}; S.extra = ""; S.devices = [];
    $("launch-model").value = "";
    await load(true);
    $("launch-saved").textContent = T("launch.resetDone", "reset to the built-in defaults");
  }

  // ------------------------------------------------------------------ actions

  async function start(restart) {
    if (S.busy) return;
    const model = $("launch-model").value;
    if (!model) { $("launch-msg").textContent = T("launch.pickModel", "pick a model first"); return; }
    if (!S.devices.length) { $("launch-msg").textContent = T("launch.pickCard", "switch on at least one SM70 card"); return; }
    S.busy = true;
    $("launch-msg").textContent = restart ? T("launch.restarting", "restarting…")
                                          : T("launch.starting", "starting…");
    refreshState();
    try {
      if (restart) await post("/stop", {}).catch(() => {});
      const r = await post("/start", selection());
      $("launch-msg").textContent = r.ok ? T("launch.up", "server is up")
                                         : `${T("launch.failed", "failed")}: ${r.message}`;
    } catch (e) {
      $("launch-msg").textContent = `${T("launch.failed", "failed")}: ${e}`;
    }
    S.busy = false;
    refreshState();
    refreshLog();
  }

  let browsing = false;

  async function browse(kind, target, after, ctl) {
    if (browsing) return;          // one dialog at a time
    browsing = true;
    const msg = $("launch-model-msg") || $("launch-msg");
    msg.textContent = T("launch.browsing", "waiting for the dialog…");
    try {
      const start = ctl ? ctl.value
                        : (target === "launch-dir" ? $("launch-dir").value : $("launch-model").value);
      const r = await post("/browse", { kind, start });
      if (r.path) {
        if (ctl) {
          // the model-card inputs keep their value in S.args. Setting .value alone
          // showed the picked file in the form while /preview and /start still ran
          // the old arguments - a browsed mmproj never reached --mmproj.
          ctl.value = r.path;
          ctl.dispatchEvent(new Event("input", {bubbles: true}));
          ctl.dispatchEvent(new Event("change", {bubbles: true}));
        } else {
          $(target).value = r.path;
          if (after) after();
          refreshPreview();
          savePrefs();
        }
        msg.textContent = "";
      } else if (r.error) {
        msg.textContent = `${T("launch.browseFailed", "dialog failed")}: ${r.error}`;
      } else {
        msg.textContent = "";      // cancelled
      }
    } catch (e) {
      msg.textContent = `${T("launch.browseFailed", "dialog failed")}: ${e}`;
    } finally {
      browsing = false;            // a cancelled or failed dialog must not wedge the buttons
    }
  }

  function wire() {
    if (S.wired) return;
    S.wired = true;
    $("launch-browse-model").onclick = () => browse("file", "launch-model", updateModelFacts);
    const rescan = async () => {
      const dir = $("launch-dir").value.trim();
      try {
        const r = await api(`/models?dir=${encodeURIComponent(dir)}`);
        S.cfg.models = r.models;
        renderModel();          // keeps whatever the operator chose; never auto-picks
        savePrefs();
        refreshPreview();
      } catch (e) {
        $("launch-msg").textContent = String(e);
      }
    };
    $("launch-dir").onchange = rescan;
    $("launch-browse").onclick = () => browse("dir", "launch-dir", rescan);
    $("launch-model").oninput = () => { updateModelFacts(); schedulePreview(); scheduleSave(); };
    $("launch-model").onchange = () => { updateModelFacts(); savePrefs(); refreshPreview(); };
    $("launch-only-rec").onchange = renderEnv;
    $("launch-filter").oninput = renderEnv;
    $("launch-extra").oninput = () => { S.extra = $("launch-extra").value; schedulePreview(); scheduleSave(); };
    $("launch-extra").onchange = () => { refreshPreview(); savePrefs(); };
    $("launch-env-reset").onclick = () => {
      S.env = {};
      renderEnv();
      refreshPreview();
      savePrefs();
    };
    $("launch-start").onclick = () => start(false);
    $("launch-restart").onclick = () => start(true);
    $("launch-stop").onclick = async () => {
      S.busy = true;
      try { await post("/stop", {}); } catch (e) { /* ignore */ }
      S.busy = false;
      refreshState();
    };
    $("launch-copy").onclick = async () => {
      try {
        await navigator.clipboard.writeText($("launch-cmd").textContent || "");
        $("launch-msg").textContent = T("launch.copied", "copied");
        setTimeout(() => { $("launch-msg").textContent = ""; }, 1500);
      } catch (e) { $("launch-msg").textContent = T("launch.clipboard", "clipboard blocked"); }
    };
    $("launch-reset-all").onclick = resetAll;
    wireProfiles();
    $("launch-log-refresh").onclick = refreshLog;
    document.addEventListener("langchange", () => {
      if (!S.cfg) return;
      renderAll();
      renderProfiles(S.profiles || []);
    });
  }

  window.SM70Launch = {
    flush() {
      clearTimeout(S.timer);
      return savePrefs();
    },
    show() {
      wire();
      load();
      clearInterval(S.logTimer);
      S.logTimer = setInterval(() => {
        if (document.getElementById("view-launch").hidden) return;
        refreshLog();
        refreshState();
      }, 2000);
    },
  };
})();
