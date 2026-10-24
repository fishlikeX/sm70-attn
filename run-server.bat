@echo off
rem ===========================================================================
rem run-server.bat - launch the native Windows sm70-attn llama-server on the
rem   V100s (works in TCC mode and while the cards are still in MCDM/WSL mode).
rem
rem Requires the CUDA toolkit bin on PATH (cudart64_12.dll / cublas64_12.dll)
rem   - this script adds it. Override SM70_MODEL to point at your .gguf;
rem   a \\wsl.localhost path works but is slow (~140 MB/s over 9p) - copy the
rem   .gguf to a local disk for full load speed.
rem
rem Extra args are appended, e.g.:  run-server.bat -c 32768 --port 8091
rem ===========================================================================
setlocal
set "CUDA=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=%CUDA%\bin;%~dp0build\bin;%PATH%"

rem sm70-attn D256 prefill kernel (the fork's whole point): 1 = on, 0 = stock
set "LLAMA_SM70_D256=1"
rem optional: in-kernel q4_0 KV direct loads (saves ~470MB VRAM per card)
rem set "LLAMA_SM70_D256_Q4_DIRECT=1"

rem Default: the two V100s only (adjust indices to your nvidia-smi listing
rem so any display card is never touched). For a single card use one index.
set "CUDA_VISIBLE_DEVICES=1,2"

if not defined SM70_MODEL (
  echo SM70_MODEL not set - pointing at the default local model path.
  set "SM70_MODEL=D:\models\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
)

"%~dp0build\bin\llama-server.exe" ^
  -m "%SM70_MODEL%" ^
  -c 262144 -b 16384 -ub 8192 ^
  -ngl 99 -fa on -ctk q4_0 -ctv q4_0 ^
  -t 12 -tb 16 --temp 0 ^
  --host 127.0.0.1 --port 8080 %*
