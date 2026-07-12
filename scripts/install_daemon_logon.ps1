# Register ollamaq daemon to start at user logon (no admin required for current-user tasks).
# Usage (from repo root):
#   powershell -ExecutionPolicy Bypass -File scripts/install_daemon_logon.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/install_daemon_logon.ps1 -Remove

param(
    [switch]$Remove
)

$ErrorActionPreference = "Stop"
$TaskName = "ollamaq-daemon"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Starter = Join-Path $Root "scripts\start_daemon.ps1"

if ($Remove) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "removed scheduled task: $TaskName"
    exit 0
}

if (-not (Test-Path $Starter)) {
    throw "missing $Starter"
}

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Starter`"" `
    -WorkingDirectory $Root

$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero)

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal -Force | Out-Null

Write-Host "registered scheduled task: $TaskName (AtLogOn for $env:USERNAME)"
Write-Host "also starting daemon now..."
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Starter
Write-Host "verify: ollamaq status"
Write-Host "remove later: powershell -File scripts/install_daemon_logon.ps1 -Remove"
