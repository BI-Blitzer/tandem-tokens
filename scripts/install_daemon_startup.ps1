# Install ollamaq daemon to start with Windows user logon via Startup folder
# (no admin / Task Scheduler rights required).
#
#   powershell -ExecutionPolicy Bypass -File scripts/install_daemon_startup.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/install_daemon_startup.ps1 -Remove

param([switch]$Remove)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Startup = [Environment]::GetFolderPath("Startup")
$LinkPath = Join-Path $Startup "ollamaq-daemon.cmd"
$Starter = Join-Path $Root "scripts\start_daemon.ps1"

if ($Remove) {
    if (Test-Path $LinkPath) { Remove-Item $LinkPath -Force }
    Write-Host "removed $LinkPath"
    exit 0
}

$cmd = @"
@echo off
cd /d "$Root"
powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "$Starter"
"@
Set-Content -Path $LinkPath -Value $cmd -Encoding ASCII
Write-Host "installed startup launcher: $LinkPath"
Write-Host "starting daemon now..."
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Starter
Write-Host "remove later: powershell -File scripts/install_daemon_startup.ps1 -Remove"
