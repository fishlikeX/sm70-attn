@echo off
rem Build the single-file launcher: sm70gui.exe
rem   - the web UI is bundled as "webui/" inside the exe (resource_dir looks in _MEIPASS)
rem   - sm70_env.json is data, sm70_config.py is imported source
rem   - --noconsole: no terminal window; the folder/file pickers re-execute this exe
rem     with --pick-dir/--pick-file/--pick-json/--pick-save and print via fd 1
setlocal
cd /d "%~dp0"
python -m PyInstaller ^
  --noconfirm --clean --onefile --noconsole ^
  --name sm70gui ^
  --icon sm70gui.ico ^
  --add-data "..\webui;webui" ^
  --add-data "sm70_env.json;." ^
  --add-data "sm70gui.ico;." ^
  --hidden-import webview.platforms.winforms ^
  --hidden-import clr ^
  sm70gui.py
if errorlevel 1 (
  echo packaging failed
  exit /b 1
)
echo.
echo built: %~dp0dist\sm70gui.exe
rem the shipped copy (tracked in the repo) and a copy next to llama-server, so a
rem clone runs without a build step and the exe finds its server without flags
copy /Y "%~dp0dist\sm70gui.exe" "%~dp0sm70gui.exe" >nul
if errorlevel 1 echo copy to tools\sm70gui\sm70gui.exe failed
rem keep a copy next to llama-server too: the exe finds it there without flags
set "TARGET=%~dp0..\..\build\bin"
if exist "%TARGET%\llama-server.exe" (
  copy /Y "%~dp0dist\sm70gui.exe" "%TARGET%\sm70gui.exe" >nul
  if errorlevel 1 (
    echo copy to %TARGET% failed
  ) else (
    echo copied: %TARGET%\sm70gui.exe
  )
)
endlocal
