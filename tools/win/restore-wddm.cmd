@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0set-v100-mode.ps1" -Mode WDDM
set "switchResult=%errorlevel%"
pause
exit /b %switchResult%
