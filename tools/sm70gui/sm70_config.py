"""Config catalog for the sm70gui launcher.

Three rules, because the launcher must not drift from the code it starts:

  * every environment variable here is one that the tree actually reads
    (`ggml/src/ggml-cuda`, `common`, `tools/server`); the default column is the
    default *in the code*, with the file:line noted where the value matters;
  * there are no preset "profiles": the form exposes the basic server
    arguments, a free-text field for anything else, and the full environment
    table. The launcher only *derives* what it can measure;
  * model facts (quant composition, MTP layers) are read from the GGUF header,
    so "IQ_SKINNY does nothing for this model" is derived, not asserted.

The UI renders ENV_VARS as a table and RECIPES as cards; sm70gui.py turns the
selection into argv + env. Nothing here contains a machine path.
"""
from __future__ import annotations

import json
import os
import shlex
import struct
import subprocess
import sys


def hidden_kwargs() -> dict:
    """subprocess kwargs that keep a console window from appearing.

    The launcher is packaged --noconsole, so it owns no console; every
    console-subsystem child it spawns (nvidia-smi, powershell) then allocates a
    fresh console *window*, and Windows draws it for the lifetime of the call.
    Polled once a second that is a cmd window flashing nonstop. CREATE_NO_WINDOW
    plus a hidden STARTUPINFO suppresses it; empty on POSIX."""
    if os.name != "nt":
        return {}
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return {"creationflags": 0x08000000, "startupinfo": si}   # CREATE_NO_WINDOW


# --------------------------------------------------------------------------- env

# The catalog is data (sm70_env.json), not code: it is edited far more often than
# the launcher, and every entry carries both languages so the UI can switch
# without a round trip. kind: bool | int | enum | text (enum carries `choices`);
# tier: common | advanced (the UI folds advanced away); risk: safe | careful | debug.
_HERE = os.path.dirname(os.path.abspath(__file__))


def _catalog() -> dict:
    for base in (_HERE, getattr(sys, "_MEIPASS", _HERE)):
        path = os.path.join(base, "sm70_env.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError("sm70_env.json not found next to sm70_config.py")


_CATALOG = _catalog()
ENV_GROUPS = _CATALOG["groups"]          # id -> {"en": ..., "zh": ...}
ENV_VARS = _CATALOG["vars"]

ENV = {e["name"]: e for e in ENV_VARS}


def env_defaults() -> dict:
    """Every catalog variable at its code default (strings, as the UI edits them)."""
    return {e["name"]: e["default"] for e in ENV_VARS}


# ------------------------------------------------------------------ server args

# the basic server arguments the form exposes as fields
ARG_FIELDS = [
    dict(key="ctx", flag="-c", kind="int", label="Context (tokens)", label_zh="上下文（token）", group="core"),
    dict(key="np", flag="-np", kind="int", label="Parallel slots", label_zh="并行槽", group="core"),
    dict(key="batch", flag="-b", kind="int", label="Batch (-b)", label_zh="批大小 (-b)", group="core"),
    dict(key="ubatch", flag="-ub", kind="int", label="Micro-batch (-ub)", label_zh="微批 (-ub)", group="core"),
    dict(key="threads", flag="-t", kind="int", label="CPU threads", label_zh="CPU 线程", group="core"),
    dict(key="threads_batch", flag="-tb", kind="int", label="Batch threads", label_zh="批处理线程", group="core"),
    dict(key="ngl", flag="-ngl", kind="int", label="GPU layers", label_zh="GPU 层数", group="core"),
    dict(key="split_mode", flag="-sm", kind="enum", choices=["none", "layer", "row", "tensor"],
         label="Split mode", label_zh="切分模式", group="core"),
    dict(key="fa", flag="-fa", kind="enum", choices=["on", "off", "auto"], label="Flash attention", label_zh="Flash attention",
         group="kv"),
    dict(key="ctk", flag="-ctk", kind="enum",
         choices=["f16", "q8_0", "q4_0", "q5_1"], label="K cache type", label_zh="K 缓存类型", group="kv"),
    dict(key="ctv", flag="-ctv", kind="enum",
         choices=["f16", "q8_0", "q4_0", "q5_1"], label="V cache type", label_zh="V 缓存类型", group="kv"),
    dict(key="port", flag="--port", kind="int", label="Server port", label_zh="服务器端口", group="server"),
    # rendered inside the Model card, in this order: folder, file, draft file, name
    # rendered inside the Model card, after the model file: vision head, draft, name
    dict(key="mmproj", flag="--mmproj", kind="text", card="model", group="model",
         label="Vision projection file (mmproj, optional)", label_zh="视觉投影文件（mmproj，可选）"),
    dict(key="draft_model", flag="--spec-draft-model", kind="text", card="model", group="decode",
         label="Speculative draft model (optional)", label_zh="投机解码文件（可选）"),
    dict(key="alias", flag="--alias", kind="text", card="model",
         label="Model name (alias, optional)", label_zh="模型名称（别名，可选）", group="server"),
    dict(key="cache_ram", flag="--cache-ram", kind="int",
         label="Prompt cache RAM (MiB, 0 = off)", label_zh="提示缓存内存（MiB，0 = 关）", group="kv"),
    dict(key="spec_type", flag="--spec-type", kind="enum",
         choices=["none", "draft-simple", "draft-eagle3", "draft-mtp", "draft-dflash",
                  "ngram-simple", "ngram-cache"],
         label="Speculation", label_zh="投机解码", group="decode"),
    dict(key="spec_n", flag="--spec-draft-n-max", kind="int", label="Max draft tokens", label_zh="最大草稿数", group="decode"),
    dict(key="temp", flag="--temp", kind="text", label="Temperature", label_zh="温度", group="decode"),
    dict(key="top_p", flag="--top-p", kind="text", label="top-p", label_zh="top-p", group="decode"),
    dict(key="top_k", flag="--top-k", kind="int", label="top-k", label_zh="top-k", group="decode"),
    dict(key="min_p", flag="--min-p", kind="text", label="min-p", label_zh="min-p", group="decode"),
    dict(key="n_predict", flag="-n", kind="int", label="Max tokens (-n, -1 = unbounded)", label_zh="最大输出 token（-n，-1 = 不限）", group="decode"),
]
ARGS = {a["key"]: a for a in ARG_FIELDS}

# `needs` values: gpus2 (>=2 cards with >=24 GB), any
# Initial form values. These are the README's production parameters, exposed as
# plain editable fields — not a profile: nothing applies unless the form shows it,
# and the command preview is the single source of truth.
DEFAULT_ARGS = dict(ctx=32768, np=1, batch=4096, ubatch=8192, threads=12, threads_batch=16,
                    ngl=99, split_mode="tensor", fa="on", ctk="q4_0", ctv="q4_0",
                    cache_ram=0, spec_type="draft-mtp", spec_n=3, temp="0", n_predict=-1)


_TRUE = ("1", "true", "on", "yes")


def _clean_arg(field: dict, value) -> str | None:
    """Coerce one form value to a string the server will accept, or None to drop it."""
    if value is None:
        return None
    txt = str(value).strip()
    if txt == "":
        return None
    if field["kind"] == "int":
        try:
            return str(int(float(txt)))
        except (TypeError, ValueError):
            return None            # a half-typed number must not reach argv
    if field["kind"] == "enum":
        return txt if txt in field["choices"] else None
    return txt


def args_for(model: dict | None, overrides: dict | None) -> dict:
    a = dict(DEFAULT_ARGS)
    if model and not model.get("has_mtp"):
        a["spec_type"] = "none"   # no MTP head: a draft would never be used
    for k, v in (overrides or {}).items():
        field = ARGS.get(k)
        if field is None:
            continue
        clean = _clean_arg(field, v)
        if clean is not None:
            a[k] = clean
    return a


# ------------------------------------------------------------------ gguf helpers

_GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
               9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
               15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
               20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
               26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0",
               35: "TQ2_0", 36: "MXFP4"}

# the types the skinny repack can touch (ggml_cuda_q4k_skinny_side_supported)
SKINNY_IQ_TYPES = ("IQ3_S", "IQ3_XXS", "IQ2_XS", "IQ2_S")
SKINNY_KQ_TYPES = ("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K")


def read_gguf(path: str) -> dict:
    """Header-only read: metadata KVs, per-type tensor counts, MTP layer presence."""
    path = os.path.normpath(path)
    info = dict(path=path, name=os.path.basename(path), size=0, arch="", block_count=0,
                types={}, has_mtp=False, error="")
    try:
        info["size"] = os.path.getsize(path)
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                info["error"] = "not a GGUF"
                return info
            _ver, = struct.unpack("<I", f.read(4))
            n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

            def rstr() -> str:
                n, = struct.unpack("<Q", f.read(8))
                return f.read(n).decode("utf-8", "replace")

            def rval(t: int):
                if t == 8:
                    return rstr()
                if t == 9:
                    et, = struct.unpack("<I", f.read(4))
                    n, = struct.unpack("<Q", f.read(8))
                    return [rval(et) for _ in range(n)]
                fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
                       7: "<?", 10: "<Q", 11: "<q", 12: "<d"}.get(t)
                if fmt is None:
                    raise ValueError(f"kv type {t}")
                return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

            meta = {}
            for _ in range(n_kv):
                k = rstr()
                t, = struct.unpack("<I", f.read(4))
                v = rval(t)
                if len(meta) < 4096:
                    meta[k] = v
            info["arch"] = str(meta.get("general.architecture", ""))
            if info["arch"]:
                info["block_count"] = int(meta.get(f"{info['arch']}.block_count", 0) or 0)
            for _ in range(n_tensors):
                name = rstr()
                nd, = struct.unpack("<I", f.read(4))
                f.read(8 * nd)
                tt, = struct.unpack("<I", f.read(4))
                f.read(8)
                tn = _GGML_TYPES.get(tt, str(tt))
                info["types"][tn] = info["types"].get(tn, 0) + 1
                if ".nextn." in name or name.endswith("eh_proj.weight"):
                    info["has_mtp"] = True
    except Exception as e:  # noqa: BLE001 - a bad file must not break the UI
        info["error"] = str(e)[:120]
    return info


def _same_file(a: str, b: str) -> bool:
    """Same GGUF, however the path was typed (case, separators)."""
    try:
        return bool(a) and bool(b) and os.path.normcase(os.path.abspath(a)) == \
            os.path.normcase(os.path.abspath(b))
    except (OSError, ValueError):
        return False


def scan_models(dir_path: str) -> list[dict]:
    out = []
    if not dir_path or not os.path.isdir(dir_path):
        return out
    for name in sorted(os.listdir(dir_path)):
        if not name.lower().endswith(".gguf"):
            continue
        p = os.path.join(dir_path, name)
        if os.path.isfile(p):
            out.append(read_gguf(p))
    return out


def probe_cuda() -> list[dict]:
    """CUDA's own enumeration order plus the compute capability.

    CUDA_VISIBLE_DEVICES indexes *this* order, which is not nvidia-smi's: on this
    box the two V100s are CUDA 0/1 and the desktop 3080 is CUDA 2. Reading the
    capability is what makes "the SM70 cards" a measurement instead of a guess."""
    import ctypes
    names = ("nvcuda.dll", "libcuda.so.1", "libcuda.so", "libcuda.dylib")
    lib = None
    for n in names:
        try:
            lib = ctypes.CDLL(n)
            break
        except OSError:
            continue
    if lib is None:
        return []
    try:
        lib.cuInit(0)
        count = ctypes.c_int()
        if lib.cuDeviceGetCount(ctypes.byref(count)) != 0:
            return []
        out = []
        for i in range(count.value):
            dev = ctypes.c_int()
            if lib.cuDeviceGet(ctypes.byref(dev), i) != 0:
                continue
            nm = ctypes.create_string_buffer(256)
            lib.cuDeviceGetName(nm, 256, dev)
            major, minor = ctypes.c_int(), ctypes.c_int()
            lib.cuDeviceGetAttribute(ctypes.byref(major), 75, dev)  # CC major
            lib.cuDeviceGetAttribute(ctypes.byref(minor), 76, dev)  # CC minor
            pci = ctypes.create_string_buffer(32)
            lib.cuDeviceGetPCIBusId(pci, 32, dev)
            cc = (major.value, minor.value)
            out.append(dict(cuda=i, name=nm.value.decode("utf-8", "replace"),
                            cc=f"{cc[0]}.{cc[1]}", sm70=(cc == (7, 0)),
                            pci=norm_pci(pci.value.decode("utf-8", "replace"))))
        return out
    except Exception:
        return []


def norm_pci(s: str) -> str:
    """CUDA says 0000:03:00.0, nvidia-smi says 00000000:03:00.0: keep bus:dev.fn."""
    parts = (s or "").lower().strip().split(":")
    return ":".join(parts[-2:]) if len(parts) >= 2 else (s or "").lower()


_GPU_CACHE: dict = {"t": 0.0, "v": []}


def probe_gpus(fresh: bool = False) -> list[dict]:
    """Merged view: CUDA order (authoritative for CUDA_VISIBLE_DEVICES) plus the
    per-card VRAM nvidia-smi reports, matched by PCI bus id."""
    import time as _t
    if not fresh and _t.time() - _GPU_CACHE["t"] < 5.0:
        return _GPU_CACHE["v"]
    cuda = probe_cuda()
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,pci.bus_id",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL,
            **hidden_kwargs()).stdout
    except Exception:
        out = ""
    smi = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 4:
            smi.append(dict(index=int(p[0]), name=p[1], vram_mib=float(p[2]),
                            pci=norm_pci(p[3])))
    by_pci = {g["pci"]: g for g in smi}
    if cuda:
        for d in cuda:
            hit = by_pci.get(d["pci"])
            if hit:
                d["vram_mib"] = hit["vram_mib"]
                d["smi_index"] = hit["index"]
        if not any("vram_mib" in d for d in cuda):
            for d in cuda:
                d["vram_mib"] = 0.0
        _GPU_CACHE.update(t=_t.time(), v=cuda)
        return cuda
    _GPU_CACHE.update(t=_t.time(), v=smi)
    return smi


def sm70_devices(devs: list[dict]) -> list[dict]:
    return [d for d in devs if d.get("sm70")]


def device_string(indices: list[int], devs: list[dict]) -> str:
    """Keep the operator's switches, but only for cards that exist and are SM70."""
    ok = {d["cuda"] for d in sm70_devices(devs)}
    picked = [i for i in indices if i in ok]
    return ",".join(str(i) for i in picked)


# --------------------------------------------------------------------- commands

def split_extra(extra: str) -> list[str]:
    """Split the free-text argument line without eating Windows backslashes.

    POSIX shlex collapses the backslashes of a Windows path; posix=False keeps
    them but also keeps the quotes (stripped below).
    backslashes but also the quotes, so strip a matching pair afterwards."""
    try:
        words = shlex.split(extra, posix=False)
    except ValueError:
        words = extra.split()
    out = []
    for w in words:
        if len(w) >= 2 and w[0] == w[-1] and w[0] in ("'", '"'):
            w = w[1:-1]
        out.append(w)
    return out


def build_argv(args: dict, model_path: str, extra: str, port: int) -> list[str]:
    model = (model_path or "").strip()
    if model and not model.startswith("<") and not os.path.isfile(model):
        cmd = ["-hf", model]          # user/repo[:quant] - llama-server resolves it
    else:
        cmd = ["-m", model or "<model.gguf>"]
    for key, field in ARGS.items():
        if key == "port":
            continue                   # appended below, the launcher must agree with it
        if key not in args or str(args[key]) == "":
            continue
        if key == "spec_n" and str(args.get("spec_type", "none")) == "none":
            continue  # a draft cap with no draft is noise
        cmd += [field["flag"], str(args[key])]
    if str(args.get("split_mode")) == "tensor":  # fork-specific, required for TP2
        cmd += ["-fit", "off"]
    try:
        listen = int(args.get("port") or port)
    except (TypeError, ValueError):
        listen = port
    cmd += ["--metrics", "--slots", "--host", "127.0.0.1", "--port", str(listen)]
    if str(args.get("cache_ram", "0")) == "0":
        # --cache-idle-slots is on by default and warns without a RAM prompt cache
        cmd += ["--no-cache-idle-slots"]
    if extra:
        cmd += split_extra(extra)
    return cmd


def check(model: dict | None, args: dict, env: dict) -> list[str]:
    """Warnings only: what the form asks for cannot be applied by this model/box."""
    warns = []
    if model and not model.get("error"):
        types = model.get("types", {})
        has_iq = any(types.get(t) for t in SKINNY_IQ_TYPES)
        has_kq = any(types.get(t) for t in SKINNY_KQ_TYPES)
        if env.get("LLAMA_SM70_IQ_SKINNY") == "1" and not has_iq:
            warns.append("该模型没有 IQ3_S/IQ3_XXS/IQ2_XS/IQ2_S 张量：IQ_SKINNY 不会生效。")
        if env.get("LLAMA_SM70_IQ_SKINNY") == "1" and str(env.get("LLAMA_SM70_SKINNY", "0")) == "0":
            warns.append("IQ_SKINNY 需要 LLAMA_SM70_SKINNY ≠ 0 才生效。")
        if str(env.get("LLAMA_SM70_SKINNY", "0")) != "0" and not has_kq:
            warns.append("该模型没有 k-quant（Q2_K..Q6_K）张量：SKINNY 不会生效。")
        spec = str(args.get("spec_type", "none"))
        draft = str(args.get("draft_model", "") or "")
        dft = read_gguf(draft) if draft and os.path.isfile(draft) else None
        if spec == "none" and draft:
            warns.append("给了投机解码文件，但 Speculation = none：这个文件不会被使用。")
        if spec == "draft-mtp" and dft is not None:
            # -md pointing at the model itself (the usual llama.cpp MTP invocation)
            # is valid: the file has the nextn head and the widths match. Only a
            # draft without that head is the wrong pairing.
            if _same_file(draft, model.get("path", "")):
                if not model.get("has_mtp"):
                    warns.append("该模型没有 MTP 层：draft-mtp 不会生效（可把 Speculation 设为 none）。")
            elif not dft.get("has_mtp"):
                warns.append("draft-mtp 用主模型自带的 MTP 头：这个草稿文件没有 nextn 层，"
                             "被 -md 加载后没有可用的头。普通小模型请改选 draft-simple；"
                             "EAGLE-3 草稿才用 draft-eagle3。")
        if spec == "draft-dflash" and dft is not None and dft.get("arch") != "dflash":
            warns.append("draft-dflash 需要 arch=dflash 的专用草稿模型（-md 就是它）；"
                         "普通小模型请改选 draft-simple。")
        if spec == "draft-dflash" and not draft:
            warns.append("draft-dflash 需要一份 dflash 草稿模型（--spec-draft-model）；"
                         "没有它这个类型不会启用。")
        if spec in ("draft-simple", "draft-eagle3") and not draft:
            warns.append("draft-simple / draft-eagle3 都需要 --spec-draft-model 指定草稿模型。")
        if spec == "draft-mtp" and not draft and not model.get("has_mtp"):
            warns.append("该模型没有 MTP 层：投机不会被使用（可把 Speculation 设为 none）。")
    draft = str(args.get("draft_model", "") or "")
    if draft and not draft.startswith("<") and not os.path.isfile(draft):
        warns.append("投机解码文件不存在（-md 只接受本地路径）。")
    mm = str(args.get("mmproj", "") or "")
    if mm and not mm.startswith("<") and not os.path.isfile(mm):
        warns.append("视觉投影文件不存在（--mmproj 只接受本地路径）。")
    try:
        np_ = int(args.get("np") or 1)
    except (TypeError, ValueError):
        np_ = 1
    if np_ > 1 and str(env.get("LLAMA_SM70_D256", "1")) != "0":
        warns.append("np > 1：D256 内核只实现单序列因果，混批会自动回落 stock"
                     "（单序列批仍走内核；要恒定速度就设 LLAMA_SM70_D256=0）。")
    if str(args.get("split_mode")) == "tensor":
        n = len([x for x in str(env.get("CUDA_VISIBLE_DEVICES", "")).split(",") if x != ""])
        if n < 2:
            warns.append(f"split mode = tensor 需要 ≥2 张卡，当前启用 {n} 张（会退化成单卡）。")
    return warns


def command_preview(argv: list[str], env: dict, defaults: dict, exe: str = "llama-server.exe") -> str:
    changed = [f"{k}={v}" for k, v in sorted(env.items()) if v != defaults.get(k, "")]
    return " ".join(changed + [f'"{exe}"'] + [f'"{x}"' if " " in x else x for x in argv])


def config_payload(gpus: list[dict], models: list[dict], selected_model: str,
                   args_over: dict, env_over: dict, extra: str, devices_on: list[int],
                   server_exe: str, server_port: int, saved: dict) -> dict:
    """Everything the Launch tab renders, computed server-side."""
    model = next((m for m in models if m["path"] == selected_model), None)
    preset = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    cuda_warn = ("本进程的 CUDA_VISIBLE_DEVICES 已设为 "
                 f"{preset!r}：探测结果只覆盖这些设备，自动选择可能不完整。" if preset else "")
    args = args_for(model, args_over)
    args.setdefault("port", server_port)
    env = dict(env_defaults())
    env.update({k: str(v) for k, v in (env_over or {}).items() if k in ENV})
    cvd = device_string(devices_on, gpus)
    env["CUDA_VISIBLE_DEVICES"] = cvd
    argv = build_argv(args, selected_model or "<model.gguf>", extra, server_port)
    return dict(
        gpus=gpus, sm70=[d["cuda"] for d in sm70_devices(gpus)], cuda_warn=cuda_warn,
        models=models, selected_model=selected_model,
        env_vars=ENV_VARS, env_groups=ENV_GROUPS, arg_fields=ARG_FIELDS,
        env_defaults=env_defaults(), args=args, env=env, extra=extra or "",
        devices=devices_on, cuda_visible=cvd,
        argv=argv, preview=command_preview(argv, env, env_defaults(), server_exe),
        warns=check(model, args, env) + ([
            "模型不是本地文件：按 -hf <repo> 启动，GGUF 头读不到（量化构成与 MTP 层未知）。"
        ] if selected_model and not model else []),
        saved=saved, server_exe=server_exe, server_port=server_port,
    )
