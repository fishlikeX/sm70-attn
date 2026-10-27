// i18n: English and 简体中文, default follows the browser/system language.
// Strings live here; markup opts in with data-i18n / data-i18n-ph / data-i18n-title.
// The launch tab renders through t() so a language switch repaints it without a
// reload (it listens for the "langchange" event).
(() => {
  "use strict";

  const DICT = {
    en: {
      "tab.chat": "Chat", "tab.images": "Images", "tab.monitor": "Monitor", "tab.launch": "Launch", "tab.about": "About",
      "lang.label": "Language", "lang.system": "Follow system", "lang.en": "English",
      "lang.zh": "简体中文",
      "about.model": "Model and engine", "about.pc": "This PC", "about.tools": "Connect your tools",
      "about.settings": "Settings",
      "about.configured": "Configured model (launcher)", "about.alias": "Model name (alias)",
      "about.draft": "Speculative draft model",
      "about.autostart": "Load the saved model when the launcher starts",
      "about.autostartNote": "Off by default. Stored in the launcher config; the next sm70gui start loads the model the form last saved.",
      "about.minimizeTray": "Closing the window keeps the launcher in the tray",
      "about.minimizeTrayNote": "On by default. The X button only hides the window; Quit from the tray icon ends the process. Turn this off to make X quit.",
      "about.autostartFailed": "Could not save the setting",
      "about.settingsNote": "Chats, the sampler and the key are stored by the launcher (sm70gui-ui.json) and shared by every client on this PC.",

      "launch.title": "Launch", "launch.model": "Model", "launch.modelField": "Model file or HF repo",
      "launch.draftPh": "optional: standalone draft model (draft-simple / eagle3 / dflash); draft-mtp stays empty and uses the model's own head",
      "launch.mmprojPh": "optional: the vision head .gguf (mmproj)",
      "launch.modelPh": "D:/models/x.gguf  or  user/repo[:quant]",
      "launch.notLocal": "not a scanned .gguf — it will be launched as -hf <repo>",
      "launch.folder": "Model folder",
      "launch.browse": "Browse…",
      "launch.browsing": "waiting for the dialog…", "launch.browseFailed": "dialog failed", "launch.modelFile": "Model file",
      "launch.noModels": "No .gguf in this folder", "launch.choose": "Choose a model…",
      "launch.devices": "Devices",
      "launch.devicesNote": "Only SM70 (compute capability 7.0) cards can run these kernels. Switch on the cards you want; CUDA_VISIBLE_DEVICES is composed from them in CUDA's order, not nvidia-smi's.",
      "launch.notSelectable": "not selectable", "launch.unsupported": "unsupported",
      "launch.noDevices": "no CUDA devices detected",
      "launch.args": "Server arguments", "launch.extra": "Extra arguments",
      "launch.extraNote": "Appended verbatim to the server command line, for anything the fields above do not cover.",
      "launch.extraPh": "--key value --flag",
      "launch.cmd": "Command", "launch.env": "Environment",
      "launch.envNote": "Advanced variables are folded away below. If you are not sure what a switch does, leave it alone — most of them change numerics or can hang the run.",
      "launch.advanced": "Advanced environment variables",
      "launch.advancedNote": "Only touch these if you know what you are adjusting; most are not needed to run the server.",
      "launch.default": "default", "launch.debug": "debug",
      "launch.filterPh": "filter name or description",
      "launch.onlyChanged": "only non-default values",
      "launch.reset": "Reset env to defaults",
      "launch.vars": "variables",
      "launch.start": "Start", "launch.stop": "Stop", "launch.restart": "Restart",
      "launch.copy": "Copy command", "launch.log": "Server log", "launch.refresh": "Refresh",
      "launch.pickModel": "pick a model first", "launch.pickCard": "switch on at least one SM70 card",
      "launch.starting": "starting…", "launch.restarting": "restarting…", "launch.up": "server is up",
      "launch.failed": "failed", "launch.copied": "copied", "launch.clipboard": "clipboard blocked",
      "launch.running": "running", "launch.stopped": "stopped",
      "launch.external": "started outside this launcher",
      "launch.cuda": "CUDA", "launch.notReachable": "launcher not reachable",
      "launch.smallFile": "small file — check it is complete",
      "launch.headerFailed": "header read failed",
      "launch.nothing": "nothing matches", "launch.noCard": "no device selected",
      "launch.profiles": "Config files",
      "launch.profilesNote": "Save the form under a name, load it later, or keep a copy anywhere as JSON.",
      "launch.profileName": "Name", "launch.profileNamePh": "e.g. 229k-tp2",
      "launch.profSave": "Save", "launch.profRename": "Rename", "launch.profDelete": "Delete",
      "launch.profExport": "Export…", "launch.profImport": "Import…",
      "launch.profEmpty": "no saved files yet", "launch.profLoad": "load",
      "launch.profWorking": "working…", "launch.profDone": "done",
      "launch.profNeedName": "type a name first", "launch.profNewName": "new name",
      "launch.profDeleteAsk": "Delete this config file?", "launch.profExported": "written",
      "launch.saved": "saved", "launch.savedTo": "saved to",
      "launch.restored": "restored from the saved file",
      "launch.fresh": "built-in defaults (nothing saved yet)",
      "launch.resetAll": "Reset everything",
      "launch.resetConfirm": "Reset every launch setting (arguments, environment, devices, extra arguments) to the built-in defaults?",
      "launch.resetDone": "reset to the built-in defaults",
    },
    zh: {
      "tab.chat": "对话", "tab.images": "图像", "tab.monitor": "监控", "tab.launch": "启动", "tab.about": "关于",
      "lang.label": "语言", "lang.system": "跟随系统", "lang.en": "English",
      "lang.zh": "简体中文",
      "about.model": "模型与引擎", "about.pc": "本机", "about.tools": "接入你的工具",
      "about.settings": "设置",
      "about.configured": "配置的模型（launcher）", "about.alias": "模型名称（别名）",
      "about.draft": "投机解码文件",
      "about.autostart": "启动时自动加载已保存的模型",
      "about.autostartNote": "默认关闭。保存在 launcher 配置文件里；下次启动 sm70gui 时加载表单里保存的模型。",
      "about.minimizeTray": "关闭窗口时最小化到托盘",
      "about.minimizeTrayNote": "默认开启。点 × 只是隐藏窗口，从托盘图标退出才会结束进程；关掉这项后点 × 直接退出。",
      "about.autostartFailed": "保存设置失败",
      "about.settingsNote": "对话、采样设置和密钥由 launcher 统一保存（sm70gui-ui.json），本机所有窗口和浏览器共用。",

      "launch.title": "启动", "launch.model": "模型", "launch.modelField": "模型文件或 HF 仓库",
      "launch.draftPh": "可选：独立草稿模型（draft-simple / eagle3 / dflash）；draft-mtp 留空，用主模型自带的头",
      "launch.mmprojPh": "可选：视觉投影文件（mmproj）",
      "launch.modelPh": "D:/models/x.gguf  或  user/repo[:quant]",
      "launch.notLocal": "不是扫描到的 .gguf——将以 -hf <repo> 启动",
      "launch.folder": "模型目录",
      "launch.browse": "浏览…",
      "launch.browsing": "等待选择…", "launch.browseFailed": "选择框打不开", "launch.modelFile": "模型文件",
      "launch.noModels": "该目录下没有 .gguf", "launch.choose": "选择一个模型…",
      "launch.devices": "设备",
      "launch.devicesNote": "只有 SM70（计算能力 7.0）的卡能跑这些 kernel。打开要用的卡；CUDA_VISIBLE_DEVICES 由它们拼出，顺序是 CUDA 的序号，不是 nvidia-smi 的。",
      "launch.notSelectable": "不可选", "launch.unsupported": "不支持",
      "launch.noDevices": "没有检测到 CUDA 设备",
      "launch.args": "服务器参数", "launch.extra": "额外参数",
      "launch.extraNote": "原样追加到服务器命令行，用来补上面字段没有覆盖的参数。",
      "launch.extraPh": "--key value --flag",
      "launch.cmd": "命令行", "launch.env": "环境变量",
      "launch.envNote": "高级变量折叠在下面。不确定某个开关的作用就别动——多数会改变数值结果，甚至让运行卡住。",
      "launch.advanced": "高级环境变量",
      "launch.advancedNote": "除非你知道自己在调什么，否则非必要不调整这些开关。",
      "launch.default": "默认", "launch.debug": "调试",
      "launch.filterPh": "过滤名称或说明",
      "launch.onlyChanged": "只看非默认值",
      "launch.reset": "环境变量恢复默认",
      "launch.vars": "个变量",
      "launch.start": "启动", "launch.stop": "停止", "launch.restart": "重启",
      "launch.copy": "复制命令", "launch.log": "服务器日志", "launch.refresh": "刷新",
      "launch.pickModel": "先选一个模型", "launch.pickCard": "至少打开一张 SM70 卡",
      "launch.starting": "启动中…", "launch.restarting": "重启中…", "launch.up": "服务已启动",
      "launch.failed": "失败", "launch.copied": "已复制", "launch.clipboard": "剪贴板被拒绝",
      "launch.running": "运行中", "launch.stopped": "已停止",
      "launch.external": "由其他进程启动",
      "launch.cuda": "CUDA", "launch.notReachable": "launcher 未响应",
      "launch.smallFile": "文件偏小——确认是否完整",
      "launch.headerFailed": "读取头部失败",
      "launch.nothing": "没有匹配项", "launch.noCard": "未选择设备",
      "launch.profiles": "配置文件",
      "launch.profilesNote": "把当前表单存成一个名字，随时读回来；也可以导出成 JSON 放到任意位置。",
      "launch.profileName": "名称", "launch.profileNamePh": "例如 229k-tp2",
      "launch.profSave": "保存", "launch.profRename": "重命名", "launch.profDelete": "删除",
      "launch.profExport": "导出…", "launch.profImport": "导入…",
      "launch.profEmpty": "还没有保存过", "launch.profLoad": "读取",
      "launch.profWorking": "处理中…", "launch.profDone": "完成",
      "launch.profNeedName": "先起个名字", "launch.profNewName": "新名字",
      "launch.profDeleteAsk": "删除这个配置文件？", "launch.profExported": "已写出",
      "launch.saved": "已保存", "launch.savedTo": "保存到",
      "launch.restored": "已从保存的配置文件恢复",
      "launch.fresh": "内置默认值（尚未保存过）",
      "launch.resetAll": "全部恢复默认",
      "launch.resetConfirm": "把启动设置（参数、环境变量、设备、额外参数）全部恢复成内置默认值？",
      "launch.resetDone": "已恢复为内置默认值",
    },
  };

  const KEY = "strata.lang";
  let pref = "system";

  const stored = () => {
    try { return localStorage.getItem(KEY) || "system"; } catch (e) { return "system"; }
  };
  const system = () => {
    const l = (navigator.language || "en").toLowerCase();
    return l.startsWith("zh") ? "zh" : "en";
  };
  const resolve = () => (pref === "system" ? system() : (DICT[pref] ? pref : "en"));

  function t(key, fallback) {
    const lang = resolve();
    return (DICT[lang] && DICT[lang][key]) || (DICT.en[key]) || fallback || key;
  }

  function apply(root) {
    const scope = root || document;
    for (const n of scope.querySelectorAll("[data-i18n]")) n.textContent = t(n.dataset.i18n, n.textContent);
    for (const n of scope.querySelectorAll("[data-i18n-ph]")) n.placeholder = t(n.dataset.i18nPh, n.placeholder);
    for (const n of scope.querySelectorAll("[data-i18n-title]")) n.title = t(n.dataset.i18nTitle, n.title);
    document.documentElement.lang = resolve() === "zh" ? "zh-CN" : "en";
  }

  function setPref(v, save) {
    pref = v === "en" || v === "zh" ? v : "system";
    if (save) try { localStorage.setItem(KEY, pref); } catch (e) { /* ignore */ }
    apply();
    document.dispatchEvent(new CustomEvent("langchange", { detail: { lang: resolve() } }));
  }

  window.SM70I18n = { t, apply, setPref, lang: resolve, pref: () => pref };

  pref = stored();
  apply();

  // the picker lives in About; wire it here so app.js stays unaware of i18n
  const picker = document.getElementById("lang-select");
  if (picker) {
    picker.value = pref;
    picker.onchange = () => setPref(picker.value, true);
  }
  // a language change in the OS while "follow system" is selected
  if (window.matchMedia) {
    window.matchMedia("(prefers-language: zh)").addEventListener?.("change", () => {
      if (pref === "system") setPref("system", false);
    });
  }
})();
