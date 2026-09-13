param(
    [ValidateSet("start", "stop", "status", "restart")]
    [string]$Action = "status"
)

# Ultron backend control script (PowerShell).
# Used by the .bat wrappers and the Windows scheduled task.

$root   = Split-Path -Parent $PSScriptRoot
$logs   = Join-Path $root "logs"
$pidFile = Join-Path $logs "ultron.pid"
$outLog = Join-Path $logs "ultron.out.log"
$errLog = Join-Path $logs "ultron.err.log"
$portEnv = [Environment]::GetEnvironmentVariable("FLASK_PORT")
if ($portEnv) { $port = [int]$portEnv } else { $port = 5000 }
New-Item -ItemType Directory -Force -Path $logs | Out-Null

function Get-UltronPids {
    $pids = @()
    if (Test-Path $pidFile) {
        foreach ($p in (Get-Content $pidFile -ErrorAction SilentlyContinue)) {
            $proc = Get-Process -Id $p -ErrorAction SilentlyContinue
            if ($proc -and $proc.ProcessName -like "python*") { $pids += $p }
        }
    }
    return $pids
}

function Test-PortListening {
    return [bool](Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
}

switch ($Action) {
    "start" {
        $existing = Get-UltronPids
        if ($existing) { Write-Host "Already running (PID $($existing -join ', '))"; exit 0 }
        if (Test-PortListening) { Write-Host "Port $port already in use - is Ultron already running?"; exit 1 }

        $proc = Start-Process -FilePath "python" -ArgumentList "serve.py" `
            -WorkingDirectory $root -WindowStyle Hidden `
            -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
        Set-Content -Path $pidFile -Value $proc.Id

        Write-Host "Ultron backend started in the background (PID $($proc.Id))."
        Write-Host "  Web UI : http://127.0.0.1:$port/"
        Write-Host "  Logs   : $logs"
        Write-Host "  To stop: scripts\stop_backend.bat"

        Start-Sleep -Seconds 4
        try { $r = Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$port/"; Write-Host "  Health : HTTP $($r.StatusCode) OK" }
        catch { Write-Host "  Health : still starting - check logs\ultron.err.log" }
    }

    "stop" {
        $existing = Get-UltronPids
        if (-not $existing) { Write-Host "Not running."; Remove-Item $pidFile -ErrorAction SilentlyContinue; exit 0 }
        foreach ($p in $existing) { Stop-Process -Id $p -Force; Write-Host "Stopped PID $p" }
        Remove-Item $pidFile -ErrorAction SilentlyContinue
        Write-Host "Ultron backend stopped."
    }

    "status" {
        $existing = Get-UltronPids
        if (-not $existing) {
            if (Test-PortListening) { Write-Host "A server is on port $port but it was not started by this script." }
            else { Write-Host "Not running." }
            exit 1
        }
        Write-Host "Running (PID $($existing -join ', '))"
        try { $r = Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$port/"; Write-Host "Web UI health: HTTP $($r.StatusCode) OK" }
        catch { Write-Host "Web UI health: no response" }
    }

    "restart" {
        powershell -NoProfile -ExecutionPolicy Bypass -File $PSCommandPath -Action stop
        Start-Sleep -Seconds 2
        powershell -NoProfile -ExecutionPolicy Bypass -File $PSCommandPath -Action start
    }
}
