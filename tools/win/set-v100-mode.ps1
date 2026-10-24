# Adapted from Havensky-stack/windows-ninfer-V100X2 tools/windows (Apache-2.0 lineage); log dir renamed to sm70-attn. Status check needs no admin; switching does + a Windows restart to apply. NEVER route the display GPU (RTX 3080) to TCC - this script only ever touches cards named "Tesla V100".
# Native Windows dual-V100 driver mode control. No driver installation or reboot.
param([ValidateSet('Status', 'TCC', 'WDDM')][string]$Mode = 'Status')
$ErrorActionPreference = 'Stop'

function Get-V100State {
    $lines = & $smi --query-gpu=uuid,name,driver_model.current,driver_model.pending --format=csv,noheader
    if ($LASTEXITCODE -ne 0) { throw 'Unable to query NVIDIA driver modes.' }
    $rows = @($lines | ConvertFrom-Csv -Header UUID,Name,Current,Pending | ForEach-Object {
        [pscustomobject]@{
            UUID = $_.UUID.Trim()
            Name = $_.Name.Trim()
            Current = $_.Current.Trim()
            Pending = $_.Pending.Trim()
        }
    } | Where-Object { $_.Name -match '^Tesla V100(?:-|\s|$)' })
    if ($rows.Count -ne 2) { throw "Expected exactly two Tesla V100 GPUs; found $($rows.Count). No mode change performed." }
    foreach ($row in $rows) {
        if ($row.Current -notin @('TCC','WDDM') -or $row.Pending -notin @('TCC','WDDM')) {
            throw "Driver mode is unavailable for $($row.UUID). No mode change performed."
        }
    }
    return $rows
}

$transcribing = $false
$result = 0
try {
    $smi = (Get-Command nvidia-smi.exe -ErrorAction Stop).Source
    $cards = @(Get-V100State)
    $cards | Format-Table -AutoSize | Out-Host
    if ($Mode -eq 'Status') { exit 0 }
    if (@($cards | Where-Object { $_.Pending -ne $Mode }).Count -eq 0) {
        if (@($cards | Where-Object { $_.Current -ne $Mode }).Count -gt 0) {
            Write-Host "$Mode is already pending. Save work and restart Windows to apply it."
        } else {
            Write-Host "Both V100 GPUs are already in $Mode mode."
        }
        exit 0
    }

    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Write-Host 'Windows administrator approval is required to change driver modes.'
        $arguments = '-NoProfile -ExecutionPolicy Bypass -File "{0}" -Mode {1}' -f $PSCommandPath, $Mode
        # This interactive window lets the user see the requested mode and driver response.
        $child = Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
            -Verb RunAs -WindowStyle Normal -ArgumentList $arguments -Wait -PassThru
        $result = $child.ExitCode
        Get-V100State | Format-Table -AutoSize | Out-Host
        Write-Host "Mode-switch logs: $env:LOCALAPPDATA\NInferV100X2\driver-mode"
        if ($result -ne 0) { Write-Host 'Switch failed or was incomplete. Inspect both current/pending modes and the log.' }
        else { Write-Host 'Save work and restart Windows if pending differs from current.' }
        exit $result
    }

    $logDirectory = Join-Path $env:LOCALAPPDATA 'NInferV100X2\driver-mode'
    New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
    $log = Join-Path $logDirectory ('switch-{0}-{1}-{2}.log' -f $Mode, (Get-Date -Format 'yyyyMMdd-HHmmss'), $PID)
    Start-Transcript -Path $log | Out-Null
    $transcribing = $true
    Write-Host "Requested mode: $Mode. Only the two detected V100 GPUs will be changed."
    $cards | Format-Table -AutoSize | Out-Host
    $modeValue = if ($Mode -eq 'TCC') { '1' } else { '0' }
    foreach ($card in $cards) {
        if ($card.Pending -eq $Mode) { continue }
        Write-Host "Setting $($card.UUID) to $Mode"
        & $smi -i $card.UUID -dm $modeValue
        if ($LASTEXITCODE -ne 0) {
            throw "Driver rejected the switch for $($card.UUID) (exit $LASTEXITCODE). The other GPU may already have changed pending mode. Inspect the states below; toolswinestore-wddm.cmd can request WDDM for both cards."
        }
    }
    $after = @(Get-V100State)
    if (@($after | Where-Object { $_.Pending -ne $Mode }).Count -gt 0) {
        throw "Driver did not report $Mode pending for both GPUs. Inspect the final states."
    }
    Write-Host "Both GPUs report $Mode pending. Save work and restart Windows if current differs."
} catch {
    Write-Host "Mode switch/query failed: $($_.Exception.Message)" -ForegroundColor Red
    $result = 1
} finally {
    if ($transcribing) {
        & $smi --query-gpu=uuid,name,driver_model.current,driver_model.pending --format=csv
        Stop-Transcript | Out-Null
        Write-Host "Log: $log"
    }
}
exit $result
