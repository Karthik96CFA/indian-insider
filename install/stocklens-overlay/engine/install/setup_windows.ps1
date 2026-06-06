$ErrorActionPreference = "Stop"
$EngineRoot = Split-Path -Parent $PSScriptRoot
$RepoRoot = Split-Path -Parent $EngineRoot

Write-Host "==> StockLens engine setup"
Write-Host "    Repo:   $RepoRoot"
Write-Host "    Engine: $EngineRoot"

New-Item -ItemType Directory -Force -Path "$EngineRoot\.state\logs" | Out-Null

$EnvFile = "$EngineRoot\.env"
$EnvExample = "$EngineRoot\config\.env.example"
if (-not (Test-Path $EnvFile)) {
    if (Test-Path $EnvExample) {
        Copy-Item $EnvExample $EnvFile
        Write-Host "==> Created $EnvFile — add your API keys."
    }
} else {
    Write-Host "==> Keeping existing $EnvFile"
}

$Python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $Python) { $Python = (Get-Command py -ErrorAction SilentlyContinue).Source }
if (-not $Python) { throw "Python not found." }

& $Python -m pip install -r "$EngineRoot\requirements.txt"
& $Python -c "import sys; sys.path.insert(0, r'$EngineRoot\agents'); from common import initialize_db; initialize_db(); print('DB ready')"

& "$PSScriptRoot\schedule_windows.ps1"

Write-Host @"

Engine setup complete.

  Config:  $EngineRoot\.env
  Logs:    $EngineRoot\.state\logs
  Test:    python $EngineRoot\agents\stocklens_bridge.py --test
  Tick:    python $EngineRoot\agents\orchestrator.py --phase tick

Web app: npm install && npm run dev  (from repo root)
"@
