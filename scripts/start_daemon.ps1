# Start ollamaq daemon for Claude / Grok MCP queue_submit (one per machine).
# Usage (from repo root):
#   powershell -ExecutionPolicy Bypass -File scripts/start_daemon.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/start_daemon.ps1 -Foreground

param(
    [switch]$Foreground
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$env:OLLAMAQ_HOME = $Root
$log = Join-Path $Root "daemon.log"
$cfg = Join-Path $Root "config\models.yaml"

$pyCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pyCmd) { $pyCmd = Get-Command py -ErrorAction SilentlyContinue }
if (-not $pyCmd) { throw "python not found on PATH" }
$py = $pyCmd.Source

$existing = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
        $_.CommandLine -and (
            $_.CommandLine -like "*ollamaq.cli*daemon*" -or
            $_.CommandLine -like "*ollamaq* daemon*"
        ) -and ($_.CommandLine -notlike "*mcp*")
    }
if ($existing) {
    Write-Host ("daemon already running (PID {0})" -f ($existing.ProcessId -join ", "))
    exit 0
}

if ($Foreground) {
    Write-Host "starting ollamaq daemon in foreground (Ctrl-C to stop)"
    & $py -m ollamaq.cli daemon --config $cfg
    exit $LASTEXITCODE
}

$pyw = Join-Path (Split-Path $py) "pythonw.exe"
if (-not (Test-Path $pyw)) { $pyw = $py }

$argList = @("-m", "ollamaq.cli", "daemon", "--config", $cfg)
Write-Host ("starting daemon: {0} {1}" -f $pyw, ($argList -join " "))
Write-Host ("log: {0}" -f $log)
$p = Start-Process -FilePath $pyw -ArgumentList $argList -WorkingDirectory $Root `
    -RedirectStandardOutput $log -RedirectStandardError ($log + ".err") -PassThru -WindowStyle Hidden
Write-Host ("daemon PID {0} -- Claude and Grok can queue_submit now" -f $p.Id)
Write-Host "check: ollamaq status"
