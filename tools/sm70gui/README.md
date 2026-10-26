# sm70gui (planned)

Single-exe launcher: param form -> llama-server -> browser with the adapted
web UI. Toolchain decision: no Go on this box; Python 3.11 + PyInstaller 6.20
are present, so the launcher is Python + pyinstaller (one-file exe, web UI
embedded via --add-data).
