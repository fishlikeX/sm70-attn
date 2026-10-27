# sm70gui — the launcher and its web UI

`sm70gui.exe` (this folder, **committed**) is a one-file build: the launcher, the
web UI under `../webui/`, the environment catalog and the WebView2/pystray runtime
are packed inside. No Python needed.

## Use

Double-click `sm70gui.exe`, or:

```
sm70gui.exe --gui-port 8199
sm70gui.exe --no-browser            (serve only, no window: tests / background)
sm70gui.exe --autostart             (load the saved model once, ignoring the switch)
sm70gui.exe --server-exe D:\llama.cpp\build\bin\llama-server.exe
sm70gui.exe --config D:\cfg\sm70gui.json
```

It opens its own window hosting `127.0.0.1:<gui-port>` (default 8199), proxies
llama-server's API, and starts/stops the server from the Launch tab. It looks for
`llama-server.exe` next to itself, then `./bin`, then `<repo>\build\bin`; pass
`--server-exe` for anything else.

The X button minimizes to the tray by default (setting: About -> Settings, stored
as `minimize_tray` in the config); tray menu: 显示 / Show, 在浏览器中打开 / Open in
browser, 退出 / Quit. Quit stops the server this process started. A second launch
raises the existing window instead of starting another instance; a headless copy
shows a message with its URL. Nothing opens a browser tab on its own.

Settings live next to llama-server (`sm70gui-config.json` plus
`sm70gui-profiles/`), or wherever `--config` / `SM70GUI_CONFIG` points. The chat,
the sampler, the API key, the theme and the language are also launcher-owned:
`sm70gui-ui.json` in the same directory, served by `GET/POST /ui-state`, so every
client on the PC shares one copy (the page's localStorage is only a cache for the
first paint; on the first run of this build an existing browser profile seeds the
file). Two logs: `%TEMP%\sm70gui-launcher.log` (startup, window, tray, crashes) and
`%TEMP%\sm70gui-server.log` (llama.cpp).

Running from source needs `pip install pywebview pystray pillow` (Windows uses the
installed WebView2 runtime); without `pywebview` it falls back to the system
browser, and without a tray icon the X button quits.

## Rebuild (maintainers)

`build_exe.bat` runs PyInstaller and refreshes three places: `dist/sm70gui.exe`
(ignored), `sm70gui.exe` here (tracked - commit it after a GUI change) and
`build\bin\sm70gui.exe` when that folder has a llama-server. It passes the icon,
the webui/env/icon data files and the `webview.platforms.winforms` / `clr` hidden
imports that pywebview needs in a frozen build.

Layout: `sm70gui.py` (HTTP server, endpoints, pickers, window/tray), `sm70_config.py`
+ `sm70_env.json` (argument/environment catalog, GGUF and CUDA probes), `sm70gui.ico`,
`../webui/` (the UI), `build_exe.bat`.
