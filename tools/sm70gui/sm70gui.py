#!/usr/bin/env python3
"""sm70gui — launcher + local server for the sm70-attn web UI.

One process does three jobs:
  1. serve the adapted web UI (tools/webui) on http://127.0.0.1:<gui_port>
  2. expose GET /gpu  — nvidia-smi + system RAM as JSON for the Monitor page
  3. start / stop llama-server with the parameters the GUI form submits
     (POST /start), tail its log (GET /serverlog), report state (GET /server)

Run:  python sm70gui.py [--gui-port 8199] [--server-port 8098]
      [--webui-dir D:/sm70-attn/tools/webui]
      [--server-exe D:/sm70-attn/build/bin/llama-server.exe]
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


def resource_dir() -> str:
    bundled = getattr(sys, "_MEIPASS", None)  # PyInstaller one-file
    if bundled and os.path.isdir(os.path.join(bundled, "webui")):
        return os.path.join(bundled, "webui")
    return os.path.join(HERE, "..", "webui")


class ServerState:
    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.log: list[str] = []
        self.lock = threading.Lock()
        self.cmdline: list[str] = []

    def start(self, exe: str, args: list[str], port: int) -> tuple[bool, str]:
        with self.lock:
            if self.proc and self.proc.poll() is None:
                return False, "already running"
            env = dict(os.environ)
            # extra env vars come as K=V entries in args["env"]
            log_path = os.path.join(os.environ.get("TEMP", "."), "sm70gui-server.log")
            self.log = []
            try:
                self.logf = open(log_path, "w", encoding="utf-8", errors="replace")
            except OSError as e:
                return False, f"log open failed: {e}"
            self.proc = subprocess.Popen(
                [exe] + args, env=env, stdout=self.logf, stderr=subprocess.STDOUT,
                cwd=os.path.dirname(exe) or None,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
            self.cmdline = [exe] + args
            # wait for /health
            import urllib.request
            for _ in range(300):
                if self.proc.poll() is not None:
                    return False, "server exited during startup (see log)"
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                        if r.status == 200:
                            return True, "ok"
                except Exception:
                    pass
                time.sleep(1)
            return False, "health timeout (see log)"

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
        """Move lines from the log file into memory (call from a thread)."""
        with self.lock:
            f = getattr(self, "logf", None)
            if not f:
                return
            while True:
                line = f.readline()
                if not line:
                    break
                self.log.append(line.rstrip())
                if len(self.log) > 4000:
                    self.log = self.log[-2000:]


STATE = ServerState()


class Handler(BaseHTTPRequestHandler):
    gui_port = 8199
    server_port = 8098
    server_exe = ""

    def log_message(self, fmt, *args):  # quiet
        pass

    def _json(self, obj, code: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/gpu":
            import urllib.request
            # reuse the powershell helper from server.go's python twin
            try:
                out = subprocess.check_output([
                    "nvidia-smi", "--query-gpu=index,name,utilization.gpu,"
                    "memory.used,memory.total,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits"], text=True, timeout=5)
                gpus = []
                for line in out.strip().splitlines():
                    p = [x.strip() for x in line.split(",")]
                    if len(p) >= 7:
                        gpus.append({"index": int(float(p[0])), "name": p[1],
                                     "util": float(p[2]), "mem_used": float(p[3]) * 1048576,
                                     "mem_total": float(p[4]) * 1048576, "temp": float(p[5]),
                                     "power": float(p[6])})
            except Exception:
                gpus = []
            self._json({"gpus": gpus})
            return
        if self.path == "/server":
            alive = bool(STATE.proc and STATE.proc.poll() is None)
            self._json({"running": alive, "port": self.server_port,
                        "cmdline": STATE.cmdline if alive else []})
            return
        if self.path == "/serverlog":
            self._json({"lines": STATE.tail(200)})
            return
        # everything else: the static UI
        path = self.path.split("?")[0]
        if path == "/":
            path = "/index.html"
        if path.startswith("/web/"):
            path = path[len("/web"):]          # the UI references its assets as web/...
        full = os.path.normpath(os.path.join(resource_dir(), path.lstrip("/")))
        if not full.startswith(os.path.normpath(resource_dir())) or not os.path.isfile(full):
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

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/start":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        exe = req.get("exe") or self.server_exe
        args = [str(a) for a in req.get("args", [])]
        for k, v in (req.get("env") or {}).items():
            os.environ[k] = str(v)
        ok, msg = STATE.start(exe, args, self.server_port)
        self._json({"ok": ok, "message": msg})

        def pump() -> None:
            while STATE.proc and STATE.proc.poll() is None:
                STATE.pump()
                time.sleep(0.3)
            STATE.pump()
        threading.Thread(target=pump, daemon=True).start()


def stop_handler(_sig, _frame) -> None:
    STATE.stop()
    sys.exit(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui-port", type=int, default=8199)
    ap.add_argument("--server-port", type=int, default=8098)
    ap.add_argument("--server-exe", default=r"D:\sm70-attn\build\bin\llama-server.exe")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    Handler.gui_port = a.gui_port
    Handler.server_port = a.server_port
    Handler.server_exe = a.server_exe
    if not shutil.which("nvidia-smi"):
        print("warning: nvidia-smi not on PATH; GPU cards will be empty")

    import signal
    signal.signal(signal.SIGINT, stop_handler)
    srv = ThreadingHTTPServer(("127.0.0.1", a.gui_port), Handler)
    url = f"http://127.0.0.1:{a.gui_port}/"
    print(f"sm70gui on {url}  (llama-server port {a.server_port})")
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        stop_handler(None, None)


if __name__ == "__main__":
    main()
