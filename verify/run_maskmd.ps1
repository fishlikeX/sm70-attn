# sm70 d256 mask-driven (unified KV pool) — standalone cross-check
#
# Feeds the production kernel with MaskMD = true and compares against a CPU
# reference: softmax over the cells the KQ mask leaves visible, times V. The
# battery covers a single sequence (regression), four slots contiguous, four
# slots with holes, four slots interleaved in runs of 8 cells, a q_len that is
# not a multiple of kBlockM, and one row with no visible cell at all.
#
# It also cross-checks sm70_mask_row_bounds_kernel against the host.
#
# Usage:  .\run_maskmd.ps1
#
# The CUDA binary is compiled ONCE (sm_70). Needs nvcc on PATH (run from a
# Developer Command Prompt, or let this script find the toolkit).

param()

$ErrorActionPreference = "Stop"
$root = Join-Path (Split-Path -Parent $PSScriptRoot) "ggml\src\ggml-cuda"
$inc  = "$root\sm70-vendor"

$md5 = {
    param($p)
    $raw = [IO.File]::ReadAllText($p) -replace "`r`n", "`n"   # CRLF-normalized
    $bytes = [Text.Encoding]::UTF8.GetBytes($raw)
    (Get-FileHash -Algorithm MD5 -InputStream (New-Object IO.MemoryStream(,$bytes))).Hash.ToLower()
}
$kmd5   = & $md5 "$root\fattn-sm70-d256-kernel.cuh"
$expect = "38ffd5fd527f4910d7911194d5462674"   # fattn-sm70-d256-kernel.cuh @ kvu-mask-driven (2026-11-01)
$mmd5   = & $md5 "$root\fattn-sm70-d256-maskmd.cuh"
$mexp   = "1056311e3daff9bf5aaef37bc0bdde25"   # fattn-sm70-d256-maskmd.cuh @ kvu-mask-driven (2026-11-01)
if ($kmd5 -ne $expect) { throw "kernel md5 drift: $kmd5 != $expect - re-audit before trusting results" }
if ($mmd5 -ne $mexp)   { throw "maskmd md5 drift: $mmd5 != $mexp - re-audit before trusting results" }
Write-Host ("kernel md5 ok: " + $kmd5)
Write-Host ("maskmd md5 ok: " + $mmd5)

$bin = Join-Path $PSScriptRoot "sm70_verify_maskmd.exe"

Write-Host "== build =="
# Windows/MSVC quirks the WSL build gets for free: /Zc:__cplusplus is what
# makes __cplusplus report C++17 for the vendored cutlass, and math.h there
# does not define M_LOG2E for the vendored softmax.
& nvcc -O2 -std=c++17 -arch=sm_70 -Xcompiler "/Zc:__cplusplus /utf-8" "-DM_LOG2E=1.4426950408889634f" -I "$inc" -I "$root" "$PSScriptRoot\sm70_verify_maskmd.cu" -o "$bin"
if ($LASTEXITCODE -ne 0) { throw "nvcc build failed" }

Write-Host "== run =="
& $bin
Write-Host ("exit = " + $LASTEXITCODE)
exit $LASTEXITCODE
