# sm70 d256 attention — standalone cross-check (run on the V100 physical machine)
#
# Usage:  .\run.ps1          small battery (all-rows cases + sampled 3000)
#         .\run.ps1 -Full    + 32k sampled case (f64-free but CPU ref slower)
#         .\run.ps1 -Oob     OOB-guard: K/V exact-size, any past-kv_len read is an illegal access
#         .\run.ps1 -Full -Oob
#
# The CUDA binary is compiled ONCE (sm_70). The CPU reference runs on this
# host (f32, row-sampled for big cases).

param([switch]$Full, [switch]$Oob)

$ErrorActionPreference = "Stop"
$root = Join-Path (Split-Path -Parent $PSScriptRoot) "ggml\src\ggml-cuda"   # .../sm70-attn/ggml/src/ggml-cuda
$inc  = "$root\sm70-vendor"

# ---- source fingerprint guard ------------------------------------------------
# The harness is valid ONLY against the exact kernel + vendor bundle that was
# static-audited (md5s below). If the repo drifted, fail loudly instead of
# producing unverifiable numbers.
$md5 = {
    param($p)
    $raw = [IO.File]::ReadAllText($p) -replace "`r`n", "`n"   # CRLF-normalized
    $bytes = [Text.Encoding]::UTF8.GetBytes($raw)
    (Get-FileHash -Algorithm MD5 -InputStream (New-Object IO.MemoryStream(,$bytes))).Hash.ToLower()
}
$kmd5  = $md5 "$root\fattn-sm70-d256-kernel.cuh"
$expect = "0e49bfdd895a27953f4e0ea0bc12bd3f"     # fattn-sm70-d256-kernel.cuh @ main bfc1e99, LF-normalized (2026-08-21)
if ($kmd5 -ne $expect) { throw "kernel md5 drift: $kmd5 != $expect — rebuild the harness audit before trusting results" }
Write-Host ("kernel md5 ok: " + $kmd5)

$bin  = Join-Path $PSScriptRoot "sm70_verify.exe"

Write-Host "== build =="
& nvcc -O2 -std=c++17 -arch=sm_70 -I "$inc" -I "$root" "$PSScriptRoot\sm70_verify.cu" -o "$bin"
if ($LASTEXITCODE -ne 0) { throw "nvcc build failed" }

Write-Host "== device check =="
# quick capability check via the binary's own error path is overkill;
# the kernel simply won't launch on cc!=70 (cudaErrorNoKernelImageForDevice)

$mode = ""
if ($Oob)  { $mode = "o" }
if ($Full) { $mode = $mode + "f" }
if ($mode -eq "") { $mode = "n" }

Write-Host "== run ($mode) =="
if ($Oob)   { $env:SM70_VERIFY_OOB  = "1" } else { Remove-Item Env:SM70_VERIFY_OOB  -ErrorAction SilentlyContinue }
if ($Full)  { $env:SM70_VERIFY_FULL = "1" } else { Remove-Item Env:SM70_VERIFY_FULL -ErrorAction SilentlyContinue }
& $bin
Write-Host ("exit = " + $LASTEXITCODE)

# ---- optional sanitizer pass (OOB only) -------------------------------------
# Re-run the small battery under compute-sanitizer memcheck. SLOWS the kernel
# ~10-50x; keep it to the non-Full battery.
if ($Oob -and $env:RUN_SANITIZER -eq "1") {
    Write-Host "== compute-sanitizer memcheck =="
    & compute-sanitizer --tool memcheck --error-exitcode 9 $bin
    Write-Host ("sanitizer exit = " + $LASTEXITCODE)
}
