@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0set-v100-mode.ps1" -Mode TCC
set "switchResult=%errorlevel%"
pause
exit /b %switchResult%
