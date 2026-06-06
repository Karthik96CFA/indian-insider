$ErrorActionPreference = "Stop"
$EngineRoot = Split-Path -Parent $PSScriptRoot
$RepoRoot = Split-Path -Parent $EngineRoot
$AgentsPath = Join-Path $EngineRoot "agents"

Write-Host "==> StockLens engine setup"
Write-Host "    Repo:   $RepoRoot"
Write-Host "    Engine: $EngineRoot"

New-Item -ItemType Directory -Force -Path (Join-Path $EngineRoot ".state\logs") | Out-Null

$EnvFile = Join-Path $EngineRoot ".env"
$EnvExample = Join-Path $EngineRoot "config\.env.example"
if (-not (Test-Path $EnvFile)) {
    if (Test-Path $EnvExample) {
        Copy-Item $EnvExample $EnvFile
        Write-Host "==> Created $EnvFile - add your API keys."
    }
} else {
    Write-Host "==> Keeping existing $EnvFile"
}

$Python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $Python) { $Python = (Get-Command py -ErrorAction SilentlyContinue).Source }
if (-not $Python) { throw "Python not found." }

& $Python -m pip install -r (Join-Path $EngineRoot "requirements.txt")

$initScript = "import sys; sys.path.insert(0, sys.argv[1]); from common import initialize_db; initialize_db(); print('DB ready')"
& $Python -c $initScript $AgentsPath

& (Join-Path $PSScriptRoot "schedule_windows.ps1")

Write-Host ""
Write-Host "Engine setup complete."
Write-Host "  Config:  $EnvFile"
Write-Host "  Logs:    $(Join-Path $EngineRoot '.state\logs')"
Write-Host "  Test:    python $(Join-Path $EngineRoot 'agents\stocklens_bridge.py') --test"
Write-Host "  Tick:    python $(Join-Path $EngineRoot 'agents\orchestrator.py') --phase tick"
Write-Host "  Web app: npm install; npm run dev  (from repo root)"
