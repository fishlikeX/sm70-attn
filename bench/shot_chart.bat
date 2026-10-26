@echo off
rem Rasterise a chart HTML to PNG with a headless browser.
rem   bench\shot_chart.bat <html> <png>
rem Override the browser with EDGE=<path to msedge/chrome>.
setlocal
if "%~1"=="" (
  echo usage: shot_chart.bat ^<html^> ^<png^>
  exit /b 2
)
if "%EDGE%"=="" set "EDGE=C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
for %%I in ("%~1") do set "HTML=%%~fI"
for %%I in ("%~2") do set "PNG=%%~fI"
"%EDGE%" --headless --disable-gpu --hide-scrollbars --force-device-scale-factor=1 ^
  --window-size=1400,940 --screenshot="%PNG%" "file:///%HTML:\=/%" >nul 2>&1
echo wrote %PNG%
