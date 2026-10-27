#!/usr/bin/env python3
"""sm70gui — launcher + local server for the sm70-attn web UI.

One process does three jobs:
  1. serve the adapted web UI (tools/webui) on http://127.0.0.1:<gui_port>
  2. expose GET /gpu  — nvidia-smi + system RAM as JSON for the Monitor page
  3. start / stop llama-server with the parameters the GUI form submits
     (POST /start), tail its log (GET /serverlog), report state (GET /server)

Run:  python sm70gui.py [--gui-port 8199] [--server-port 8098]
      [--webui-dir <repo>/tools/webui]
      [--server-exe <repo>/build/bin/llama-server.exe]
Packaged with PyInstaller (--onefile --add-data "webui;webui").
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
# a one-file build runs from a temp extraction dir: __file__ lives there, so
# anything that must sit beside the user's exe uses APP_DIR instead
APP_DIR = os.path.dirname(os.path.abspath(sys.executable)) if getattr(sys, "frozen", False) else HERE

import sm70_config  # noqa: E402  (catalog: env vars, args, gguf + CUDA probes)


def default_server_exe() -> str:
    """beside the launcher, then <repo>/build/bin; '' if absent.

    A packaged one-file exe usually sits next to the llama-server build, so that
    is checked first; the source-tree layout is the second candidate."""
    roots = [APP_DIR, os.path.join(APP_DIR, "bin"),
             os.path.normpath(os.path.join(APP_DIR, "..", "..", "build", "bin")),
             os.path.normpath(os.path.join(HERE, "..", "..", "build", "bin"))]
    for root in roots:
        for name in ("llama-server.exe", "llama-server"):
            p = os.path.join(root, name)
            if os.path.isfile(p):
                return p
    return ""


def resource_dir() -> str:
    bundled = getattr(sys, "_MEIPASS", None)  # PyInstaller one-file
    if bundled and os.path.isdir(os.path.join(bundled, "webui")):
        return os.path.join(bundled, "webui")
    return os.path.join(HERE, "..", "webui")


CONFIG_PATH = ""   # set in main(): --config, else next to llama-server, else beside the launcher

LAUNCH_LOG = os.path.join(os.environ.get("TEMP", "."), "sm70gui-launcher.log")


def _log(msg: str) -> None:
    """stdout for a console run, a file for the windowed exe (which has no console).

    A crash before the window exists would otherwise be invisible: the process
    just disappears. %TEMP% is the same place the server log goes."""
    line = time.strftime("%H:%M:%S ") + str(msg)
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        with open(LAUNCH_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _excepthook(exc_type, exc, tb) -> None:
    import traceback
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    _log("FATAL\n" + text)
    if getattr(sys, "frozen", False) and os.name == "nt":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, text[-1500:], "sm70gui", 0x10)
        except Exception:
            pass
    sys.__excepthook__(exc_type, exc, tb)


sys.excepthook = _excepthook


def _pick_config_path(server_exe: str, explicit: str) -> str:
    """--config wins, then the build directory (next to llama-server.exe), then the
    launcher's own folder, then the per-user profile."""
    if explicit:
        return os.path.abspath(explicit)
    env = os.environ.get("SM70GUI_CONFIG", "")
    if env:
        return os.path.abspath(env)
    candidates = []
    if server_exe:
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(server_exe)),
                                       "sm70gui-config.json"))
    candidates.append(os.path.join(APP_DIR, "sm70gui-config.json"))
    for c in candidates:
        d = os.path.dirname(c) or "."
        try:
            os.makedirs(d, exist_ok=True)
            probe = os.path.join(d, ".sm70gui-write-test")
            with open(probe, "w") as f:
                f.write("")
            os.remove(probe)
            return c
        except OSError:
            continue
    base = os.environ.get("APPDATA") or os.environ.get("XDG_CONFIG_HOME") or HERE
    d = os.path.join(base, "sm70gui")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "config.json")


def pref_path() -> str:
    global CONFIG_PATH
    if not CONFIG_PATH:
        CONFIG_PATH = _pick_config_path("", "")
    return CONFIG_PATH


def clean_settings(d) -> dict:
    """Keep only the keys and types the form knows; a hand-edited file with
    args: [] or devices: "0,1" would otherwise raise inside every request."""
    if not isinstance(d, dict):
        return {}
    out = {}
    for k in ("models_dir", "model", "extra"):
        if isinstance(d.get(k), str):
            out[k] = d[k]
    if isinstance(d.get("args"), dict):
        out["args"] = {str(k): str(v) for k, v in d["args"].items() if v is not None}
    if isinstance(d.get("env"), dict):
        out["env"] = {str(k): str(v) for k, v in d["env"].items() if v is not None}
    if isinstance(d.get("devices"), list):
        devs = []
        for x in d["devices"]:
            try:
                i = int(x)
            except (TypeError, ValueError):
                continue
            if 0 <= i < 64:
                devs.append(i)
        out["devices"] = sorted(set(devs))
    if isinstance(d.get("autostart"), bool):
        out["autostart"] = d["autostart"]
    if isinstance(d.get("minimize_tray"), bool):
        out["minimize_tray"] = d["minimize_tray"]
    return out


def prefs_load() -> dict:
    try:
        with open(pref_path(), encoding="utf-8") as f:
            return clean_settings(json.load(f))
    except Exception:
        return {}


def prefs_save(patch: dict) -> dict | None:
    """Read-modify-write the settings file atomically; None means it did not stick.

    Two debounced saves can overlap and a crash mid-write used to leave invalid
    JSON (which the next load then treated as an empty config); tmp + replace
    keeps the old file intact on failure."""
    cur = prefs_load()
    cur.update(patch)
    tmp = pref_path() + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cur, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, pref_path())
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return None
    return cur


# ------------------------------------------------------------- shared UI state
# One PC, one config. The chat, the sampler, the API key, the theme and the
# language belong to the *launcher*, not to one browser profile: they live in
# sm70gui-ui.json beside the settings file, and every client (the app window, Edge,
# anything else that opens the page) reads and writes the same copy. localStorage
# in the page is only a cache so the first paint does not wait for a round trip.
UI_STATE_MAX = 32 * 1024 * 1024      # a runaway chat must not fill the disk


def ui_state_path() -> str:
    base = os.path.dirname(CONFIG_PATH) if CONFIG_PATH else os.environ.get("TEMP", ".")
    return os.path.join(base or os.environ.get("TEMP", "."), "sm70gui-ui.json")


def clean_ui_state(d) -> dict:
    """Only the keys the page stores, in the shapes it stores them.

    The file is hand-editable and written by the page, so everything is
    re-validated here; a chat is capped to the newest 500 turns."""
    if not isinstance(d, dict):
        return {}
    out = {}
    if isinstance(d.get("chat"), list):
        out["chat"] = [m for m in d["chat"] if isinstance(m, dict)][-500:]
    if isinstance(d.get("sampling"), dict):
        out["sampling"] = {str(k): v for k, v in d["sampling"].items()
                           if isinstance(v, (str, int, float, bool)) or v is None}
    if isinstance(d.get("apikey"), str):
        out["apikey"] = d["apikey"][:4096]
    if d.get("theme") in ("light", "dark"):
        out["theme"] = d["theme"]
    if d.get("lang") in ("system", "en", "zh"):
        out["lang"] = d["lang"]
    return out


def ui_state_load() -> dict:
    try:
        with open(ui_state_path(), encoding="utf-8") as f:
            return clean_ui_state(json.load(f))
    except Exception:
        return {}


def ui_state_save(patch: dict) -> dict | None:
    """Merge one client's keys into the shared file, atomically."""
    cur = ui_state_load()
    cur.update(clean_ui_state(patch))
    try:
        text = json.dumps(cur, ensure_ascii=False)
    except (TypeError, ValueError):
        return None
    if len(text.encode("utf-8")) > UI_STATE_MAX:
        _log(f"ui-state: refusing to write {len(text)} bytes (limit {UI_STATE_MAX})")
        return None
    tmp = ui_state_path() + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, ui_state_path())
    except OSError as e:
        _log(f"ui-state: write failed: {e}")
        try:
            os.remove(tmp)
        except OSError:
            pass
        return None
    return cur


def profiles_dir() -> str:
    d = os.path.join(os.path.dirname(pref_path()), "sm70gui-profiles")
    os.makedirs(d, exist_ok=True)
    return d


def profile_ok(name: str) -> bool:
    """A profile is a file name inside profiles_dir: no separators, no dot-dot."""
    name = (name or "").strip()
    if not name or len(name) > 64 or name.startswith("."):
        return False
    if any(c in name for c in "/\:*?\"<>|"):
        return False
    stem = name.split(".")[0].lower()
    if stem in ("con", "prn", "aux", "nul") or stem.startswith("com") or stem.startswith("lpt"):
        return False
    return name == os.path.basename(name)


def profile_file(name: str) -> str:
    return os.path.join(profiles_dir(), name + ".json")


def list_profiles() -> list[dict]:
    out = []
    d = profiles_dir()
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(d, fn)
        try:
            with open(path, encoding="utf-8") as f:
                d2 = json.load(f)
            st = os.stat(path)
        except (OSError, ValueError):
            continue
        if not isinstance(d2, dict):
            continue
        model = d2.get("model") if isinstance(d2.get("model"), str) else ""
        args = d2.get("args") if isinstance(d2.get("args"), dict) else {}
        out.append({
            "name": fn[:-5],
            "model": os.path.basename(model) if model else "",
            "ctx": (args or {}).get("ctx"),
            "np": (args or {}).get("np"),
            "alias": (args or {}).get("alias") or "",
            "mtime": int(st.st_mtime),
            "size": st.st_size,
        })
    return out


def save_profile(name: str) -> tuple[bool, str]:
    if not profile_ok(name):
        return False, "invalid name"
    cur = prefs_load()
    keep = {k: cur.get(k) for k in ("models_dir", "model", "args", "env", "extra", "devices")}
    with open(profile_file(name), "w", encoding="utf-8") as f:
        json.dump(keep, f, indent=1)
    return True, ""


def load_profile(name: str) -> tuple[bool, str]:
    if not profile_ok(name):
        return False, "invalid name"
    path = profile_file(name)
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError) as e:
        return False, str(e)
    d = clean_settings(d)
    patch = {k: d[k] for k in ("models_dir", "model", "args", "env", "extra", "devices") if k in d}
    return (True, "") if prefs_save(patch) else (False, "could not write the config")


def rename_profile(old: str, new: str) -> tuple[bool, str]:
    if not profile_ok(old) or not profile_ok(new):
        return False, "invalid name"
    src, dst = profile_file(old), profile_file(new)
    if not os.path.isfile(src):
        return False, "no such profile"
    if os.path.isfile(dst):
        return False, "target exists"
    os.replace(src, dst)
    return True, ""


def delete_profile(name: str) -> tuple[bool, str]:
    if not profile_ok(name):
        return False, "invalid name"
    try:
        os.remove(profile_file(name))
    except OSError as e:
        return False, str(e)
    return True, ""


def models_dir(prefs: dict) -> str:
    """Only what the operator chose: the saved folder, or SM70GUI_MODELS.

    A fresh profile starts empty on purpose - adopting the running server's model
    or the folder this launcher happened to see last time would mean the form
    ships with somebody else's paths."""
    return prefs.get("models_dir") or os.environ.get("SM70GUI_MODELS", "")


DEFAULT_PORT = 8098


def cur_port() -> int:
    """The port llama-server listens on right now (or will, for the next start)."""
    return STATE.port or DEFAULT_PORT


_GPU_SMI = {"t": -10.0, "out": ""}
_GPU_SMI_LOCK = threading.Lock()


def nvidia_smi_lines() -> str:
    """Per-card nvidia-smi line, cached for 2 s, never two spawns at once.

    /gpu is polled every second and every poll used to spawn nvidia-smi; with a
    --noconsole launcher each spawn also drew a console window, and on a busy TCC
    card the queries could overlap. Two seconds keeps the Monitor close to live
    while bounding it to one process per two seconds; a failed query keeps the
    previous text instead of blanking every card."""
    with _GPU_SMI_LOCK:
        now = time.time()
        if now - _GPU_SMI["t"] < 2.0:
            return _GPU_SMI["out"]
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,"
                 "memory.used,memory.total,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
                stdin=subprocess.DEVNULL, **sm70_config.hidden_kwargs()).stdout
            if out.strip():
                _GPU_SMI["out"] = out
            _GPU_SMI["t"] = now
        except Exception:
            _GPU_SMI["t"] = now          # back off, keep the last good sample
        return _GPU_SMI["out"]


class ServerState:
    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.log: list[str] = []
        self.lock = threading.Lock()
        self.cmdline: list[str] = []
        self.env_used: dict = {}
        self.log_path = ""
        self.log_pos = 0
        self.port = 0          # 0 = the --server-port default; /start may change it
        self._starting = False

    def start(self, exe: str, args: list[str], port: int, env_over: dict | None = None) -> tuple[bool, str]:
        if not exe or not os.path.isfile(exe):
            return False, f"llama-server not found: {exe or '(empty)'} -- pass --server-exe"
        with self.lock:
            if self._starting:
                return False, "a start is already in progress"
            if self.proc and self.proc.poll() is None:
                return False, "already running"
            self._starting = True
            env = dict(os.environ)
            env.update(env_over or {})
            self.env_used = dict(env_over or {})
            # extra env vars come as K=V entries in args["env"]
            log_path = os.path.join(os.environ.get("TEMP", "."), "sm70gui-server.log")
            self.log = []
            self.log_path = log_path
            self.log_pos = 0
            try:
                # stderr and stdout get SEPARATE handles: with stderr=STDOUT the child's
                # stderr becomes an alias of the (block-buffered) stdout FILE, and every
                # INFO line -- including "model loaded" and "listening on" -- sat in that
                # buffer instead of reaching the log. llama.cpp logs to stderr, so that is
                # the file we tail.
                self.logf = open(log_path, "w", encoding="utf-8", errors="replace")
                self.logout = open(log_path + ".out", "w", encoding="utf-8", errors="replace")
            except OSError as e:
                self._starting = False
                return False, f"log open failed: {e}"
            try:
                # llama-server is a console program: spawned from this --noconsole
                # launcher without CREATE_NO_WINDOW it gets a visible console window,
                # and closing that window (an easy mistake: it looks like a stray cmd)
                # kills the server. NEW_PROCESS_GROUP keeps a Ctrl-Break aimed at the
                # launcher from reaching it.
                flags = (subprocess.CREATE_NEW_PROCESS_GROUP | 0x08000000
                         if os.name == "nt" else 0)   # | CREATE_NO_WINDOW
                self.proc = subprocess.Popen(
                    [exe] + args, env=env, stdout=self.logout, stderr=self.logf,
                    stdin=subprocess.DEVNULL, cwd=os.path.dirname(exe) or None,
                    creationflags=flags,
                )
            except OSError as e:
                self._starting = False
                return False, f"could not start {exe}: {e}"
            self.cmdline = [exe] + args
            self.pump_stop = False

        # The health wait must NOT hold the lock: /serverlog, /server and /stop come
        # from the same page, and a 30 s model load would block all of them (and fill
        # the browser's connection pool with the 2 s log poll).
        threading.Thread(target=self._pump_loop, daemon=True).start()
        import urllib.request
        try:
            for _ in range(300):
                if self.proc.poll() is not None:
                    return False, (f"server exited during startup "
                                   f"(exit code {self.proc.returncode}, see log)")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                        if r.status == 200:
                            return True, "ok"
                except Exception:
                    pass
                time.sleep(1)
            return False, "health timeout (see log)"
        finally:
            self._starting = False

    def _pump_loop(self) -> None:
        proc = self.proc
        while proc is not None and proc.poll() is None:
            self.pump()
            time.sleep(0.3)
        self.pump()

    def stop(self) -> bool:
        with self.lock:
            if not self.proc or self.proc.poll() is not None:
                return False
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            return True

    def tail(self, n: int = 100) -> list[str]:
        with self.lock:
            return list(self.log[-n:])

    def pump(self) -> None:
        """Move new lines from the log file into memory (call from a thread).

        The child's handle is write-only, so tail a second, read-only handle and
        keep an offset: re-reading the whole file on every tick would be O(n^2)."""
        if not self.log_path:
            return
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self.log_pos)
                chunk = f.read()
                self.log_pos = f.tell()
        except OSError:
            return
        if not chunk:
            return
        with self.lock:
            self.log.extend(chunk.splitlines())
            if len(self.log) > 4000:
                self.log = self.log[-2000:]


STATE = ServerState()




class _HostSampler:
    """CPU, RAM and disk rates without spawning a process per request.

    PowerShell takes ~4 s to start here, so the old implementation made every
    Monitor refresh wait for it and the 1 Hz poll loop collapsed to ~5 s. CPU
    and RAM come from ctypes (microseconds), the disk counters -- which need
    PDH -- come from one background PowerShell every 3 s."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.cpu = None
        self.ram_used = None
        self.ram_total = None
        self.cpu_name = ""
        self.disk_read_mb = 0.0
        self.disk_write_mb = 0.0
        self._prev = None
        self._disk_thread: threading.Thread | None = None

    # ---- cheap, in-process ----
    def sample(self) -> None:
        try:
            if os.name == "nt":
                self._sample_windows()
            else:
                self._sample_linux()
        except Exception:
            pass

    def _sample_windows(self) -> None:
        import ctypes

        class FILETIME(ctypes.Structure):
            _fields_ = [("lo", ctypes.c_uint32), ("hi", ctypes.c_uint32)]

        idle, kern, user = FILETIME(), FILETIME(), FILETIME()
        k32 = ctypes.windll.kernel32
        if k32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)):
            def q(ft):
                return (ft.hi << 32) | ft.lo
            cur = (q(idle), q(kern), q(user))
            if self._prev:
                di = cur[0] - self._prev[0]
                dk = cur[1] - self._prev[1] + cur[2] - self._prev[2]
                if dk > 0:
                    self.cpu = round(100.0 * (1.0 - di / dk), 1)
            self._prev = cur

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                        ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                        ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                        ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                        ("ullAvailExtendedVirtual", ctypes.c_uint64)]
        ms = MEMORYSTATUSEX()
        ms.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if k32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            self.ram_total = ms.ullTotalPhys
            self.ram_used = ms.ullTotalPhys - ms.ullAvailPhys

        if not self.cpu_name:
            try:
                import winreg
                sep = chr(92)   # a backslash, built without an escape
                path = sep.join(["HARDWARE", "DESCRIPTION", "System", "CentralProcessor", "0"])
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as k:
                    self.cpu_name = str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
            except OSError:
                pass

    def _sample_linux(self) -> None:
        try:
            with open("/proc/stat") as f:
                vals = [int(x) for x in f.readline().split()[1:]]
            idle, total = vals[3] + (vals[4] if len(vals) > 4 else 0), sum(vals)
            if self._prev:
                di, dt = idle - self._prev[0], total - self._prev[1]
                if dt > 0:
                    self.cpu = round(100.0 * (1.0 - di / dt), 1)
            self._prev = (idle, total)
        except OSError:
            pass
        try:
            info = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, v = line.split(":", 1)
                    info[k] = int(v.split()[0]) * 1024
            self.ram_total = info.get("MemTotal")
            avail = info.get("MemAvailable", info.get("MemFree"))
            if self.ram_total and avail:
                self.ram_used = self.ram_total - avail
        except (OSError, ValueError):
            pass

    # ---- disk counters: PDH needs a process, so cache it in the background ----
    def start_disk_poll(self) -> None:
        if os.name != "nt" or self._disk_thread:
            return
        self._disk_thread = threading.Thread(target=self._disk_loop, daemon=True)
        self._disk_thread.start()

    def _disk_loop(self) -> None:
        script = (
            "$d = Get-CimInstance Win32_PerfFormattedData_PerfDisk_PhysicalDisk |"
            " Where-Object { $_.Name -ne '_Total' };"
            "$rd = ($d | Measure-Object -Property DiskReadBytesPersec -Sum).Sum;"
            "$wr = ($d | Measure-Object -Property DiskWriteBytesPersec -Sum).Sum;"
                        "Write-Output ('$rd|$wr')"
        )
        while True:
            try:
                line = subprocess.check_output(
                    ["powershell", "-NoProfile", "-Command", script], text=True,
                    timeout=20, stdin=subprocess.DEVNULL,
                    **sm70_config.hidden_kwargs()).strip().splitlines()[-1]
                rd, wr = (float(x) for x in line.split("|")[:2])
                with self.lock:
                    self.disk_read_mb = rd / 1048576.0
                    self.disk_write_mb = wr / 1048576.0
            except Exception:
                pass
            time.sleep(3.0)

    def snapshot(self) -> dict:
        self.sample()
        with self.lock:
            return {
                "cpu": self.cpu,
                "cpu_name": self.cpu_name,
                "ram_used": self.ram_used,
                "ram_total": self.ram_total,
                "disk_read_mb": self.disk_read_mb,
                "disk_write_mb": self.disk_write_mb,
                "threads": os.cpu_count(),
            }


HOST = _HostSampler()


def host_stats() -> dict:
    """CPU %, RAM and disk rates for the This-machine card (no process per call)."""
    return HOST.snapshot()


def _emit(line: str) -> None:
    """print() to a pipe; a windowed build has no stdout, so fall back to fd 1."""
    try:
        if sys.stdout is not None:
            print(line, flush=True)
            return
    except Exception:
        pass
    try:
        os.write(1, (line + chr(10)).encode())
    except OSError:
        pass


def pick_path(kind: str, start: str) -> dict:
    """Ask for a folder/file with the platform dialog, in a child process.

    Tk wants its own main loop and the HTTP handler runs in a worker thread; a
    separate process keeps both honest. SM70GUI_PICK_STUB prints a fixed answer
    instead of opening a dialog (used by the tests)."""
    flag = {"file": "--pick-file", "json": "--pick-json",
            "save": "--pick-save"}.get(kind, "--pick-dir")
    cmd = ([sys.executable, flag] if getattr(sys, "frozen", False)
           else [sys.executable, os.path.abspath(__file__), flag])
    if start:
        cmd += ["--start", start]
    # a one-file build runs the app as a child of its own bootloader; that child
    # carries _PYI_* markers, and a grandchild inheriting them hangs waiting for a
    # parent that is not there (seen: the second Browse click never returned).
    # A clean env makes the picker a fresh instance, like it is from a shell.
    env = {k: v for k, v in os.environ.items() if not k.startswith("_PYI_") and k != "_MEIPASS2"}
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env,
                             **sm70_config.hidden_kwargs())
    except subprocess.TimeoutExpired:
        return {"path": "", "error": "dialog timed out"}
    except OSError as e:
        return {"path": "", "error": str(e)}
    path = ""
    for line in (out.stdout or "").splitlines():
        if line.strip():
            path = line.strip()
    return {"path": path, "error": "" if path else (out.stderr or "").strip()[:200]}


def run_picker(kind: str, start: str) -> int:
    """--pick-dir / --pick-file: print the chosen path (or nothing on cancel)."""
    stub = os.environ.get("SM70GUI_PICK_STUB", "")
    if stub:
        _emit(stub)
        return 0
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:  # noqa: BLE001 - no Tk in this build
        print(f"tkinter unavailable: {e}", file=sys.stderr)
        return 1
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()
    initial = ""
    if start:
        cand = start if kind == "dir" else os.path.dirname(start)
        if cand and os.path.isdir(cand):
            initial = cand
    if kind == "dir":
        path = filedialog.askdirectory(title="Select the folder that holds the .gguf files",
                                       initialdir=initial, mustexist=True, parent=root)
    elif kind == "save":
        path = filedialog.asksaveasfilename(title="Save the launcher settings",
                                            initialdir=initial, parent=root,
                                            defaultextension=".json",
                                            filetypes=[("JSON", "*.json")])
    elif kind == "json":
        path = filedialog.askopenfilename(title="Read launcher settings", initialdir=initial,
                                          parent=root,
                                          filetypes=[("JSON", "*.json"), ("All files", "*.*")])
    else:
        path = filedialog.askopenfilename(title="Select a .gguf file", initialdir=initial,
                                          parent=root,
                                          filetypes=[("GGUF model", "*.gguf"), ("All files", "*.*")])
    root.destroy()
    if path:
        _emit(os.path.normpath(path))
    return 0


def selection(req: dict) -> tuple[list[str], dict, dict, list[str], str, list[int]]:
    """(argv, env, args, warns, extra, devices) for a form-shaped request.

    A field the request leaves out keeps the saved value: a minimal
    {"model": ...} must not reset the rest of the configuration (it used to,
    because /start persisted the raw request body)."""
    req = clean_settings(req)     # first: a junk devices/env must not reach the argv
    prefs = prefs_load()
    if "args" not in req:
        req["args"] = prefs.get("args", {})
    if "env" not in req:
        req["env"] = prefs.get("env", {})
    if "extra" not in req:
        req["extra"] = prefs.get("extra", "")
    gpus = sm70_config.probe_gpus()
    all_sm70 = [d["cuda"] for d in sm70_config.sm70_devices(gpus)]
    devices = req.get("devices")
    if devices is None:
        devices = prefs.get("devices") or all_sm70
    model = req.get("model") or ""
    model_info = sm70_config.read_gguf(model) if model and os.path.isfile(model) else None
    args = sm70_config.args_for(model_info, req.get("args") or {})
    defaults = sm70_config.env_defaults()
    env = dict(defaults)
    env.update({k: str(v) for k, v in (req.get("env") or {}).items() if k in sm70_config.ENV})
    env["CUDA_VISIBLE_DEVICES"] = sm70_config.device_string(devices, gpus)
    argv = sm70_config.build_argv(args, model or "<model.gguf>", req.get("extra") or "",
                                  cur_port())
    return (argv, env, args, sm70_config.check(model_info, args, env),
            req.get("extra") or "", devices)


def start_from_request(req: dict, exe: str) -> tuple[bool, str, int]:
    """Build argv/env from a form-shaped request, persist it, spawn llama-server.

    Shared by POST /start and the double-click autostart so both paths go through
    the same derivation and the same saved state. The exe never comes from the
    request: this endpoint starts exactly the binary the launcher was given."""
    if "model" not in req:
        return False, "send {recipe-shaped} fields: model, args, env, extra, devices", 400
    model = (req.get("model") or "").strip()
    if not model:
        return False, "pick a model file first", 400
    is_file = os.path.isfile(model)
    argv, full_env, args, _warns, eff_extra, eff_devices = selection(req)
    defaults = sm70_config.env_defaults()
    env_over = {k: v for k, v in full_env.items() if v != defaults.get(k)}
    want = args.get("port") if isinstance(args, dict) else None
    try:
        want = int(want)
    except (TypeError, ValueError):
        want = 0
    STATE.port = want if 1024 <= want <= 65535 else cur_port()
    args_keep = {k: v for k, v in args.items() if v != sm70_config.DEFAULT_ARGS.get(k)}
    prefs_save({"model": model, "args": args_keep, "env": env_over,
                "extra": eff_extra, "devices": eff_devices,
                "models_dir": (os.path.dirname(model) if is_file
                               else prefs_load().get("models_dir", ""))})
    ok, msg = STATE.start(exe, argv, cur_port(), env_over)
    return ok, msg, 200


class Handler(BaseHTTPRequestHandler):
    gui_port = 8199
    server_port = 8098
    server_exe = ""

    def log_message(self, fmt, *args):  # quiet
        pass

    def handle_one_request(self) -> None:
        # a client that hangs up mid-write (closed tab, curl -m) used to print a
        # full socketserver traceback for every occurrence
        try:
            super().handle_one_request()
        except (ConnectionError, TimeoutError):
            self.close_connection = True

    def _json(self, obj, code: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _client_ok(self) -> bool:
        """Refuse anything a web page could reach by accident.

        Any site in the browser can POST to 127.0.0.1 with a "simple request"
        (text/plain, no preflight); /start spawns processes and /profiles deletes
        files, so the origin has to be pinned: Host must be loopback (kills DNS
        rebinding), Origin -- when present -- must be this same origin, and an
        application/json body forces a preflight that we never allow."""
        host = (self.headers.get("Host") or "").lower()
        ok_host = (host.startswith("127.0.0.1:") or host == "127.0.0.1"
                   or host.startswith("localhost:") or host == "localhost"
                   or host.startswith("[::1]"))
        if not ok_host:
            self._json({"ok": False, "message": f"host not allowed: {host!r}"}, 403)
            return False
        origin = self.headers.get("Origin")
        if origin:
            want = f"http://{host}"
            if origin.rstrip("/") != want:
                self._json({"ok": False, "message": f"origin not allowed: {origin!r}"}, 403)
                return False
        if self.command == "POST":
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype != "application/json":
                self._json({"ok": False, "message": "Content-Type must be application/json"}, 415)
                return False
        return True

    def _read_json(self) -> dict | None:
        """Body as a dict, or None after answering 400 (a malformed escape in a
        Windows path used to kill the request thread and leave curl with no reply)."""
        n = int(self.headers.get("Content-Length", 0))
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, UnicodeDecodeError) as e:
            self._json({"ok": False, "message": f"bad json: {e}"}, 400)
            return None

    def _proxy(self) -> None:
        """Forward API calls to llama-server (the page uses relative URLs).

        Streaming responses (text/event-stream) are relayed chunk by chunk so
        the chat UI renders tokens as they arrive; everything else is proxied
        as one buffer."""
        import urllib.request
        import urllib.error
        url = f"http://127.0.0.1:{cur_port()}{self.path}"
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else None
        req = urllib.request.Request(url, data=body, method=self.command)
        for h in ("Content-Type", "Authorization", "Accept"):
            if self.headers.get(h):
                req.add_header(h, self.headers[h])
        try:
            r = urllib.request.urlopen(req, timeout=7200)
        except urllib.error.HTTPError as e:
            data = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", e.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        except Exception as e:
            msg = json.dumps({"error": {"message": str(e)}}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return

        ctype = r.headers.get("Content-Type", "application/json")
        self.send_response(r.status)
        self.send_header("Content-Type", ctype)
        if "text/event-stream" in ctype:
            # stream: no Content-Length, flush per read
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            while True:
                chunk = r.read1(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        else:
            data = r.read()
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        r.close()

    def _config_payload(self) -> dict:
        prefs = prefs_load()
        mdir = models_dir(prefs)
        models = sm70_config.scan_models(mdir)
        # the running server's -m wins over a stale saved selection
        sel = prefs.get("model", "")
        if sel and os.path.isfile(sel):
            # show the facts for a saved file the scan does not cover (different folder)
            sel = os.path.normpath(sel)
            if not any(m["path"] == sel for m in models):
                models.insert(0, sm70_config.read_gguf(sel))
        gpus = sm70_config.probe_gpus()
        all_sm70 = [d["cuda"] for d in sm70_config.sm70_devices(gpus)]
        devices = [d for d in prefs.get("devices", all_sm70) if d in all_sm70] or all_sm70
        payload = sm70_config.config_payload(
            gpus=gpus, models=models, selected_model=sel,
            args_over=prefs.get("args", {}), env_over=prefs.get("env", {}),
            extra=prefs.get("extra", ""), devices_on=devices,
            server_exe=self.server_exe, server_port=cur_port(),
            saved=dict(models_dir=mdir, running=bool(STATE.proc and STATE.proc.poll() is None),
                       prefs_path=pref_path(), prefs_exists=os.path.isfile(pref_path())))
        payload["prof_dir"] = profiles_dir()
        return payload

    def do_GET(self) -> None:  # noqa: N802
        if not self._client_ok():
            return
        if self.path == "/profiles":
            self._json({"dir": profiles_dir(), "profiles": list_profiles()})
            return
        if self.path == "/ui-state":
            # the shared copy every client on this PC uses (chat, sampler, key, ...)
            try:
                at = os.path.getmtime(ui_state_path())
            except OSError:
                at = 0
            self._json({"state": ui_state_load(), "at": at, "path": ui_state_path()})
            return
        if self.path == "/selection":
            # what the launcher would start, without scanning any model folder
            pre = prefs_load()
            try:
                at = os.path.getmtime(pref_path())
            except OSError:
                at = 0
            self._json({"model": pre.get("model", ""), "args": pre.get("args", {}),
                        "env": pre.get("env", {}), "extra": pre.get("extra", ""),
                        "devices": pre.get("devices", []),
                        "autostart": pre.get("autostart", False),
                        "minimize_tray": pre.get("minimize_tray", True), "at": at})
            return
        if self.path == "/config":
            self._json(self._config_payload())
            return
        if self.path.split("?")[0] == "/models":
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            d = (q.get("dir") or [models_dir(prefs_load())])[0]
            self._json(dict(dir=d, models=sm70_config.scan_models(d)))
            return
        if self.path == "/gpu":
            pass
        elif self.path == "/host":
            self._json(host_stats())
            return
        elif self.path in ("/server", "/serverlog"):
            pass
        elif (self.path.split("?")[0] in ("/", "/index.html") or self.path.startswith("/web/")
              or os.path.isfile(os.path.join(resource_dir(),
                                             self.path.split("?")[0].lstrip("/")))):
            pass
        else:
            self._proxy()
            return
        if self.path == "/gpu":
            gpus = []
            def num(x, scale=1.0):
                try:
                    return float(x) * scale
                except (TypeError, ValueError):
                    return None      # drivers report [N/A] for some fields
            for line in nvidia_smi_lines().strip().splitlines():
                p = [x.strip() for x in line.split(",")]
                if len(p) >= 7:
                    gpus.append({"index": int(float(p[0])), "name": p[1],
                                 "util": num(p[2]), "mem_used": num(p[3], 1048576),
                                 "mem_total": num(p[4], 1048576), "temp": num(p[5]),
                                 "power": num(p[6])})
            # attach the CUDA index (CUDA_VISIBLE_DEVICES order) so the page does not
            # have to guess from nvidia-smi's numbering, which differs on this box
            try:
                cuda = sm70_config.probe_gpus()
                by_pci = {g.get("smi_index"): g.get("cuda") for g in cuda
                          if g.get("smi_index") is not None}
                for g in gpus:
                    g["cuda_index"] = by_pci.get(g["index"])
            except Exception:
                pass
            self._json({"gpus": gpus})
            return
        if self.path == "/show":
            # a second double-click talks to the running instance instead of
            # opening a browser tab next to the window it already owns
            self._json({"ok": _show_window()})
            return
        if self.path == "/server":
            alive = bool(STATE.proc and STATE.proc.poll() is None)
            external = False
            if not alive:
                # a server started outside this launcher still owns the port; report it
                # rather than pretending it is stopped (Start would just fail to bind)
                import urllib.request
                try:
                    with urllib.request.urlopen(
                            f"http://127.0.0.1:{cur_port()}/health", timeout=2) as r:
                        external = r.status == 200
                except Exception:
                    external = False
            self._json({"running": alive or external, "owned": alive, "external": external,
                        "port": cur_port(),
                        "cmdline": STATE.cmdline if alive else [],
                        "env": getattr(STATE, "env_used", {}) if alive else {}})
            return
        if self.path == "/serverlog":
            self._json({"lines": STATE.tail(200)})
            return
        # the static UI (API paths were proxied above)
        path = self.path.split("?")[0]
        if path == "/":
            path = "/index.html"
        if path.startswith("/web/"):
            path = path[len("/web"):]          # the UI references its assets as web/...
        full = os.path.normpath(os.path.join(resource_dir(), path.lstrip("/")))
        root = os.path.normpath(resource_dir())
        if not (full == root or full.startswith(root + os.sep)) or not os.path.isfile(full):
            self.send_error(404)
            return
        ctype = {"html": "text/html", "js": "text/javascript", "css": "text/css",
                 "svg": "image/svg+xml", "woff2": "font/woff2"}.get(full.rsplit(".", 1)[-1],
                                                                   "application/octet-stream")
        with open(full, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _selection(self, req: dict) -> tuple[list[str], dict, dict, list[str], str, list[int]]:
        return selection(req)

    def _preview(self, req: dict) -> dict:
        argv, env, _args, warns, _extra, _devices = self._selection(req)
        return {"preview": sm70_config.command_preview(argv, env, sm70_config.env_defaults(),
                                                       os.path.basename(self.server_exe)),
                "argv": argv, "env": env, "warns": warns}

    def do_POST(self) -> None:  # noqa: N802
        if not self._client_ok():
            return
        if self.path == "/preview":
            n = int(self.headers.get("Content-Length", 0))
            req = self._read_json()
            if req is None:
                return
            self._json(self._preview(req))
            return
        if self.path == "/stop":
            STATE.stop()
            self._json({"ok": True})
            return
        if self.path == "/ui-state":
            req = self._read_json()
            if req is None:
                return
            saved = ui_state_save(req)
            if saved is None:
                self._json({"ok": False, "path": ui_state_path(),
                            "message": "could not write the shared UI state"}, 500)
                return
            try:
                at = os.path.getmtime(ui_state_path())
            except OSError:
                at = 0
            self._json({"ok": True, "state": saved, "at": at, "path": ui_state_path()})
            return
        if self.path == "/save":
            req = self._read_json()
            if req is None:
                return
            req = clean_settings(req)
            patch = {k: req[k] for k in ("models_dir", "model", "args", "env", "extra",
                                          "devices", "autostart", "minimize_tray")
                     if k in req}
            if isinstance(patch.get("env"), dict):
                defaults = sm70_config.env_defaults()
                patch["env"] = {k: str(v) for k, v in patch["env"].items()
                                if k in sm70_config.ENV and str(v) != defaults.get(k)}
            saved = prefs_save(patch)
            self._json({"ok": saved is not None, "saved": saved, "path": pref_path(),
                        "at": time.strftime("%H:%M:%S"),
                        "message": "" if saved is not None else "could not write the config"},
                      200 if saved is not None else 500)
            return
        if self.path.startswith("/profiles/"):
            req = self._read_json()
            if req is None:
                return
            op = self.path[len("/profiles/"):]
            name = str(req.get("name") or "").strip()
            if op == "save":
                ok, msg = save_profile(name)
            elif op == "load":
                ok, msg = load_profile(name)
            elif op == "rename":
                ok, msg = rename_profile(name, str(req.get("to") or "").strip())
            elif op == "delete":
                ok, msg = delete_profile(name)
            elif op in ("export", "import"):
                # a file chosen with the platform dialog, outside profiles_dir
                kind = "save" if op == "export" else "json"
                picked = pick_path(kind, "")
                if not picked.get("path"):
                    self._json({"ok": False, "message": picked.get("error") or "cancelled"})
                    return
                target = picked["path"]
                if op == "export":
                    cur = prefs_load()
                    keep = {k: cur.get(k) for k in ("models_dir", "model", "args", "env",
                                                    "extra", "devices")}
                    with open(target, "w", encoding="utf-8") as f:
                        json.dump(keep, f, indent=1)
                    ok, msg = True, ""
                else:
                    try:
                        with open(target, encoding="utf-8") as f:
                            d = clean_settings(json.load(f))
                        patch = {k: d[k] for k in ("models_dir", "model", "args", "env",
                                                   "extra", "devices") if k in d}
                        prefs_save(patch)
                        ok, msg = True, ""
                    except (OSError, ValueError) as e:
                        ok, msg = False, str(e)
            else:
                self._json({"ok": False, "message": f"unknown profile op {op!r}"}, 404)
                return
            self._json({"ok": ok, "message": msg, "profiles": list_profiles()})
            return
        if self.path == "/browse":
            req = self._read_json()
            if req is None:
                return
            kind = req.get("kind") or "dir"
            start = req.get("start") or ""
            self._json(pick_path(kind, start))
            return
        if self.path == "/reset":
            try:
                os.remove(pref_path())
            except OSError:
                pass
            self._json({"ok": True, "path": pref_path()})
            return
        if self.path == "/start":
            pass
        elif self.path.startswith("/v1/") or self.path in (
                "/props", "/slots", "/tokenize", "/detokenize", "/health"):
            self._proxy()
            return
        else:
            self.send_error(404)
            return
        if self.path == "/start":
            pass
        req = self._read_json()
        if req is None:
            return
        ok, msg, code = start_from_request(req, self.server_exe)
        self._json({"ok": ok, "message": msg}, code)
        if not ok:
            return

        def pump() -> None:
            while STATE.proc and STATE.proc.poll() is None:
                STATE.pump()
                time.sleep(0.3)
            STATE.pump()
        threading.Thread(target=pump, daemon=True).start()


def stop_handler(_sig, _frame) -> None:
    STATE.stop()
    sys.exit(0)


def _fatal(msg: str) -> None:
    """A --noconsole exe has no stdout to show; a failure to start must not look
    like 'double-click did nothing'."""
    print(msg)
    if getattr(sys, "frozen", False) and os.name == "nt":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, msg, "sm70gui", 0x10)   # MB_ICONERROR
        except Exception:
            pass


def _autostart(exe: str) -> None:
    """Load the saved model right after the page comes up (About switch, default off).

    When the operator turns it on, double-clicking the exe is the whole launch
    gesture: it should end with a serving llama-server, not just a web page with a
    Start button. Nothing saved yet (or no model file) means nothing to start --
    the form is the next step."""
    prefs = prefs_load()
    model = prefs.get("model", "")
    if not model:
        _log("autostart: no model saved yet; pick one in the form")
        return
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{cur_port()}/health", timeout=1.5) as r:
            if r.status == 200:
                _log("autostart: a server is already listening; leaving it alone")
                return
    except Exception:
        pass
    req = {"model": model, "args": prefs.get("args", {}), "env": prefs.get("env", {}),
           "extra": prefs.get("extra", ""), "devices": prefs.get("devices", [])}
    ok, msg, _code = start_from_request(req, exe)
    _log(f"autostart: {'started ' + os.path.basename(model) if ok else 'failed: ' + msg}")


# ------------------------------------------------------- desktop window and tray
# The packaged launcher is a desktop app, not a shortcut that opens a browser: it
# hosts the same page in a WebView2 window it owns, and keeps running (optionally)
# in the notification area when the window is closed. Everything the page does
# still goes through the local HTTP server above, so a browser tab on the same
# port keeps working while the window is hidden.
WINDOW = None            # pywebview Window, set by run_window (for /show and the tray)
_TRAY = None             # pystray Icon while it is running
_SRV = None              # the ThreadingHTTPServer, so Quit can stop serving
_quit_requested = False
_tray_hinted = False


def resource_file(name: str) -> str:
    """A file bundled next to the app (PyInstaller one-file: _MEIPASS)."""
    for base in (getattr(sys, "_MEIPASS", ""), HERE, APP_DIR):
        if base and os.path.isfile(os.path.join(base, name)):
            return os.path.join(base, name)
    return ""


def _show_window() -> bool:
    if WINDOW is None:
        return False
    try:
        WINDOW.show()
    except Exception:
        pass
    try:
        WINDOW.restore()      # un-minimize; harmless when already normal
    except Exception:
        pass
    return True


def _quit_app() -> None:
    """Tray Quit (and window close when the setting says so): end the whole app."""
    global _quit_requested
    _quit_requested = True
    try:
        STATE.stop()          # only the server this process spawned
    except Exception:
        pass
    try:
        if _TRAY is not None:
            _TRAY.stop()
    except Exception:
        pass
    try:
        if WINDOW is not None:
            WINDOW.destroy()
    except Exception:
        pass
    try:
        if _SRV is not None:
            _SRV.shutdown()   # unblocks serve_forever when the window never opened
    except Exception:
        pass


def _on_window_closing() -> bool:
    """pywebview closing handler; returning False cancels the close.

    Default: the X button hides the window and leaves the launcher (and the
    llama-server it owns) running in the tray. Turned off in About -> Settings,
    the X button quits, which stops the server this process started."""
    global _quit_requested, _tray_hinted
    if _quit_requested:
        return True
    if WINDOW is not None and _TRAY is not None and prefs_load().get("minimize_tray", True):
        try:
            WINDOW.hide()
        except Exception:
            pass
        if not _tray_hinted:
            _tray_hinted = True
            try:
                _TRAY.notify("Still running in the tray. Quit from the tray icon.",
                             "sm70-attn")
            except Exception:
                pass
        return False
    _quit_requested = True
    return True


def _tray_image():
    from PIL import Image
    path = resource_file("sm70gui.ico")
    if path:
        try:
            return Image.open(path)
        except Exception:
            pass
    return Image.new("RGBA", (64, 64), (63, 185, 80, 255))


def _build_tray(url: str):
    import pystray
    # bilingual labels: the tray is outside the page, so it cannot follow the
    # language the UI uses (that lives in the browser's localStorage)
    return pystray.Icon(
        "sm70gui", _tray_image(), "sm70-attn",
        menu=pystray.Menu(
            pystray.MenuItem("显示 / Show", lambda icon, item: _show_window(), default=True),
            pystray.MenuItem("在浏览器中打开 / Open in browser",
                             lambda icon, item: webbrowser.open(url)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("退出 / Quit", lambda icon, item: _quit_app()),
        ))


def _run_tray() -> None:
    global _TRAY
    try:
        _TRAY.run()
    except Exception as e:
        _log(f"tray stopped: {e}")
        _TRAY = None      # without a tray icon, closing the window must quit


def run_window(url: str) -> None:
    """Host the page in our own window; return after the window is destroyed."""
    global WINDOW, _TRAY
    import webview
    open_at = url if prefs_load().get("model") else url + "#launch"
    WINDOW = webview.create_window("sm70-attn", open_at, width=1280, height=860,
                                   min_size=(960, 620), text_select=True)
    WINDOW.events.closing += _on_window_closing
    try:
        _TRAY = _build_tray(url)
    except Exception as e:
        _log(f"tray unavailable: {e}")
        _TRAY = None
    if _TRAY is not None:
        threading.Thread(target=_run_tray, daemon=True).start()
    # private_mode would drop the page's localStorage (theme, language, chats)
    # on every launch; keep a WebView2 profile under %LOCALAPPDATA% instead.
    storage = os.path.join(os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP", "."),
                           "sm70-attn", "webview")
    try:
        os.makedirs(storage, exist_ok=True)
    except OSError:
        storage = None
    _log(f"window: {open_at}" + ("" if storage else " (no persistent profile)"))
    webview.start(private_mode=storage is None, storage_path=storage,
                  icon=resource_file("sm70gui.ico") or None)
    # the window is gone: either Quit, or the X with minimize-to-tray disabled
    STATE.stop()
    try:
        if _TRAY is not None:
            _TRAY.stop()
    except Exception:
        pass
    _log("window closed; exiting")


def main() -> None:
    argv = sys.argv[1:]
    if any(f in argv for f in ("--pick-dir", "--pick-file", "--pick-json", "--pick-save")):
        kind = ("dir" if "--pick-dir" in argv else "json" if "--pick-json" in argv
                else "save" if "--pick-save" in argv else "file")
        start = ""
        if "--start" in argv:
            i = argv.index("--start")
            if i + 1 < len(argv):
                start = argv[i + 1]
        sys.exit(run_picker(kind, start))
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui-port", type=int, default=8199)
    ap.add_argument("--server-port", type=int, default=8098)
    ap.add_argument("--server-exe", default="",
                    help="llama-server binary; default: <repo>/build/bin/llama-server")
    ap.add_argument("--config", default="",
                    help="where the form state is kept; default: next to llama-server")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--autostart", dest="autostart", action="store_true", default=None,
                    help="load the saved model on launch (the About switch; off by default)")
    ap.add_argument("--no-autostart", dest="autostart", action="store_false",
                    help="serve the form only, overriding the About switch")
    ap.add_argument("--smoke", type=float, default=0.0, help=argparse.SUPPRESS)
    a = ap.parse_args()

    try:                              # a fresh log per run; last run's crash stays on disk
        with open(LAUNCH_LOG, "w", encoding="utf-8"):
            pass
    except OSError:
        pass
    _log(f"args={sys.argv[1:]} exe="
         f"{sys.executable if getattr(sys, 'frozen', False) else os.path.abspath(__file__)}")

    Handler.gui_port = a.gui_port
    Handler.server_port = a.server_port
    # the last form may have moved the listen port; find the server it left behind
    try:
        saved_port = int((prefs_load().get("args") or {}).get("port") or 0)
    except (TypeError, ValueError):
        saved_port = 0
    STATE.port = saved_port if 1024 <= saved_port <= 65535 else a.server_port
    server_exe = a.server_exe or default_server_exe()
    Handler.server_exe = server_exe
    global CONFIG_PATH
    CONFIG_PATH = _pick_config_path(server_exe, a.config)
    _log(f"config file: {CONFIG_PATH}")
    HOST.start_disk_poll()
    if not shutil.which("nvidia-smi"):
        print("warning: nvidia-smi not on PATH; GPU cards will be empty")

    import signal
    signal.signal(signal.SIGINT, stop_handler)
    url = f"http://127.0.0.1:{a.gui_port}/"
    # one launcher per port. Double-clicking again while the first copy is busy
    # (starting a model, or just unseen) used to run a second poller and a second
    # nvidia-smi every second, and the second copy died silently on the bind.
    # If our endpoint answers, the launcher is already there: show it and leave.
    import urllib.request
    try:
        with urllib.request.urlopen(url + "selection", timeout=1.0) as r:
            if r.status == 200:
                # our own window, if the running copy has one; a headless copy (or
                # an older build) answers ok:false. Never open a browser here: the
                # app's window is the UI, and a stray tab on every double-click is
                # exactly what this launcher is not supposed to do.
                shown = False
                try:
                    with urllib.request.urlopen(url + "show", timeout=2.0) as r2:
                        shown = bool(json.load(r2).get("ok"))
                except Exception:
                    pass
                _log("sm70gui is already running on this port"
                     + ("; showing its window" if shown else " (headless)"))
                if not shown:
                    _fatal(f"sm70gui is already running in the background on port {a.gui_port}.\n"
                           f"Open {url} in a browser, or stop that copy and start again.")
                return
    except Exception:
        pass
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", a.gui_port), Handler)
    except OSError as e:
        _fatal(f"Port {a.gui_port} is taken by another program ({e}).\n"
               f"Close it, or start with --gui-port <number>.")
        raise SystemExit(1)
    global _SRV
    _SRV = srv
    _log(f"sm70gui on {url}  (llama-server port {a.server_port})")
    # opt-in from About (off by default): the launcher is a form first, and someone
    # opening it to change a flag should not wait for a 27B model to load
    want_autostart = a.autostart
    if want_autostart is None:
        want_autostart = bool(prefs_load().get("autostart"))
    if want_autostart:
        threading.Thread(target=_autostart, args=(server_exe,), daemon=True).start()

    if a.no_browser:
        # headless: serve only (tests, or running it as a background service)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            stop_handler(None, None)
        return
    try:
        import webview  # noqa: F401  (pip install pywebview; bundled in the exe)
    except Exception as e:
        # the shipped exe bundles WebView2 support; this is the source-run path.
        # Still no browser tab from a double-click: say where the page is.
        _log(f"webview unavailable ({e}); serving only")
        if getattr(sys, "frozen", False):
            _fatal("The desktop window is unavailable (pywebview/WebView2 missing).\n"
                   f"The page is served at {url}")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            stop_handler(None, None)
        return
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    if a.smoke:
        # internal: exercise the real window and the X-button path, then quit like
        # the tray would. Used to check packaging (WebView2, tray, hide/show).
        def _smoke() -> None:
            time.sleep(a.smoke)
            _log("smoke: calling the closing handler (X with minimize-to-tray on)")
            cancel = _on_window_closing()
            _log(f"smoke: closing handler returned {cancel} (False = stayed in tray)")
            time.sleep(1.0)
            _log("smoke: showing the window again")
            _show_window()
            time.sleep(0.5)
            _log("smoke: quitting")
            _quit_app()
        threading.Thread(target=_smoke, daemon=True).start()
    try:
        run_window(url)
    finally:
        STATE.stop()
        srv.shutdown()



if __name__ == "__main__":
    main()
