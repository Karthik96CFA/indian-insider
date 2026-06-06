$ErrorActionPreference = "Stop"
$Root = "$env:USERPROFILE\indian-insider"
$RepoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "==> Indian Insider Windows setup"
Write-Host "    Target: $Root"

New-Item -ItemType Directory -Force -Path "$Root\agents", "$Root\config", "$Root\.state\logs", "$Root\install" | Out-Null

Copy-Item -Path "$RepoRoot\agents\*" -Destination "$Root\agents\" -Recurse -Force
Copy-Item -Path "$RepoRoot\config\*" -Destination "$Root\config\" -Recurse -Force
Copy-Item -Path "$RepoRoot\install\*" -Destination "$Root\install\" -Recurse -Force
Copy-Item -Path "$RepoRoot\requirements.txt" -Destination "$Root\requirements.txt" -Force

$EnvFile = "$Root\.env"
if (-not (Test-Path $EnvFile)) {
    Copy-Item "$Root\config\.env.example" $EnvFile
    Write-Host "==> Created $EnvFile — add your API keys before running."
} else {
    Write-Host "==> Keeping existing $EnvFile"
}

$Python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $Python) { $Python = (Get-Command py -ErrorAction SilentlyContinue).Source }
if (-not $Python) { throw "Python not found. Install Python 3.11+ and retry." }

& $Python -m pip install -r "$Root\requirements.txt"
& $Python -c "import sys; sys.path.insert(0, r'$Root\agents'); from common import initialize_db; initialize_db(); print('DB ready')"

& "$Root\install\schedule_windows.ps1"

Write-Host @"

Setup complete.

Next:
  1. Edit $EnvFile
  2. Test:  python $Root\agents\orchestrator.py --phase tick --dry-run
  3. Logs:  $Root\.state\logs\
"@
