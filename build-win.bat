@echo off
rem ===========================================================================
rem build-win.bat - native Windows build of sm70-attn for V100s (WDDM or TCC).
rem   MSVC 2022 + CUDA 12.8 nvcc, SASS sm_70 + PTX.
rem   Output: <repo>\build\bin  (llama-server.exe, llama-cli.exe, ...)
rem
rem Environment overrides (defaults auto-detect on a typical install):
rem   SM70_VS       - path to VS2022 root containing VC\Auxiliary\Build\vcvars64.bat
rem   SM70_CUDA     - path to CUDA toolkit root (bin\nvcc.exe)
rem   SM70_CMAKE    - path to cmake.exe (defaults to the VS-bundled one)
rem   SM70_JOBS     - build parallelism (defaults to %NUMBER_OF_PROCESSORS%)
rem
rem -Xcompiler /Zc:__cplusplus is REQUIRED: the vendored cutlass
rem   (sm70-vendor/cutlass) decides C++17 via _MSVC_LANG but gates
rem   cutlass::platform::is_unsigned_v on __cplusplus; without this flag
rem   MSVC reports 199711 and fattn-sm70-d256.cu fails with
rem   "namespace cutlass::platform has no member is_unsigned_v".
rem -Xcompiler /utf-8        : sm70-vendor/flash/utils.h has a Chinese
rem   comment whose UTF-8 bytes decode as CP936 with a trailing backslash,
rem   which swallows the next source line (error "expected an expression").
rem -DM_LOG2E=1.4426950408889634f : flash/softmax.h uses the POSIX constant
rem   M_LOG2E, which MSVC's <math.h> does not define; same literal as the
rem   fork's own fallback in fattn-sm70-d256.cu:47.
rem -DBUILD_SHARED_LIBS=OFF : upstream defaults to shared libs on MSVC, and
rem   llama.cpp's WINDOWS_EXPORT_ALL_SYMBOLS auto-generated exports.def for
rem   llama-server-impl picks up a bogus "__" DATA entry from the PCH marker
rem   symbol (__@@_PchSym_...), so linking llama-server-impl.dll fails with
rem   LNK2001/LNK1120. A static build skips the def files entirely and yields
rem   self-contained .exe files (needs cudart64_12/cublas64_12/cublasLt64_12
rem   DLLs from the CUDA toolkit on PATH).
rem ===========================================================================
setlocal
set "SRC=%~dp0"
if "%SRC:~-1%"=="\" set "SRC=%SRC:~0,-1%"

if defined SM70_VS (set "VS=%SM70_VS%") else set "VS=D:\Microsoft Visual Studio\2022\Community"
if defined SM70_CUDA (set "CUDA=%SM70_CUDA%") else set "CUDA=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "VSCM=%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake"
if defined SM70_CMAKE (set "CMAKE=%SM70_CMAKE%") else set "CMAKE=%VSCM%\CMake\bin\cmake.exe"
if defined SM70_JOBS (set "JOBS=%SM70_JOBS%") else set "JOBS=%NUMBER_OF_PROCESSORS%"
set "PATH=%CUDA%\bin;%VSCM%\Ninja;%PATH%"

call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul

"%CMAKE%" --version | findstr /C:"cmake version"
echo === CONFIGURE ===
"%CMAKE%" -S "%SRC%" -B "%SRC%\build" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_COMPILER:FILEPATH="%CUDA%/bin/nvcc.exe" ^
  "-DCMAKE_CUDA_FLAGS=-Xcompiler /Zc:__cplusplus -Xcompiler /utf-8 -DM_LOG2E=1.4426950408889634f" ^
  -DGGML_CUDA=ON ^
  -DBUILD_SHARED_LIBS=OFF ^
  -DCMAKE_CUDA_ARCHITECTURES=70 ^
  -DLLAMA_CURL=OFF ^
  -DLLAMA_BUILD_TESTS=OFF ^
  -DLLAMA_BUILD_EXAMPLES=OFF > "%SRC%\build_configure.log" 2>&1
if errorlevel 1 (
  echo CONFIGURE_FAILED
  powershell -NoProfile -Command "Get-Content '%SRC%\build_configure.log' -Tail 40"
  exit /b 1
)
echo CONFIGURE_OK

echo === BUILD ===
"%CMAKE%" --build "%SRC%\build" -j %JOBS% > "%SRC%\build_compile.log" 2>&1
if errorlevel 1 (
  echo BUILD_FAILED
  powershell -NoProfile -Command "Select-String -Path '%SRC%\build_compile.log' -Pattern 'error' | Select-Object -First 25 | ForEach-Object { $_.Line }; ''; Get-Content '%SRC%\build_compile.log' -Tail 30"
  exit /b 1
)
echo BUILD_OK
dir /b "%SRC%\build\bin\*.exe"
echo === WIN BUILD DONE ===
