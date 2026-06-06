# Merge indian-insider into a local stocklens clone as engine/
# Usage:
#   powershell -ExecutionPolicy Bypass -File C:\Users\karth\indian-insider\install\merge_into_stocklens.ps1
param(
    [string]$StocklensPath = "C:\Users\karth\stocklens",
    [string]$InsiderPath = "",
    [string]$Branch = "cursor/merge-indian-insider-engine-70df"
)

$ErrorActionPreference = "Stop"
$OverlayRoot = Join-Path $PSScriptRoot "stocklens-overlay"
if (-not $InsiderPath) {
    $InsiderPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

if (-not (Test-Path $StocklensPath)) { throw "StockLens not found: $StocklensPath" }
if (-not (Test-Path (Join-Path $StocklensPath ".git"))) { throw "Not a git repo: $StocklensPath" }

Write-Host "==> Indian Insider -> StockLens monorepo merge"
Write-Host "    Insider:   $InsiderPath"
Write-Host "    StockLens: $StocklensPath"
Write-Host "    Branch:    $Branch"

Push-Location $StocklensPath
try {
    git checkout master
    git pull origin master

    $existing = git branch --list $Branch
    if ($existing) { git checkout $Branch }
    else { git checkout -b $Branch }

    $Engine = Join-Path $StocklensPath "engine"
    New-Item -ItemType Directory -Force -Path $Engine | Out-Null

    Write-Host "==> Copying engine files from indian-insider"
    robocopy $InsiderPath $Engine /E /XD .git .state __pycache__ /XF .env /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed with exit $LASTEXITCODE" }

    Write-Host "==> Applying monorepo overlay"
    robocopy (Join-Path $OverlayRoot "engine") $Engine /E /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null

    Copy-Item (Join-Path $OverlayRoot "stocklens-README.md") (Join-Path $StocklensPath "README.md") -Force
    Copy-Item (Join-Path $OverlayRoot "stocklens.gitignore") (Join-Path $StocklensPath ".gitignore") -Force
    Copy-Item (Join-Path $OverlayRoot "route.ts") (Join-Path $StocklensPath "src\app\api\integrations\signals\route.ts") -Force

    $pkg = Join-Path $StocklensPath "package.json"
    $json = Get-Content $pkg -Raw | ConvertFrom-Json
    if (-not $json.scripts.'engine:tick') {
        $json.scripts | Add-Member -NotePropertyName 'engine:tick' -NotePropertyValue 'python engine/agents/orchestrator.py --phase tick --continue-on-error' -Force
        $json.scripts | Add-Member -NotePropertyName 'engine:sync' -NotePropertyValue 'python engine/agents/stocklens_bridge.py --sync-opportunities' -Force
        $json.scripts | Add-Member -NotePropertyName 'engine:test' -NotePropertyValue 'python engine/agents/stocklens_bridge.py --test' -Force
        $json | ConvertTo-Json -Depth 10 | Set-Content $pkg -Encoding UTF8
    }

    $deploy = Join-Path $StocklensPath "DEPLOYMENT.md"
    $deployText = Get-Content $deploy -Raw -Encoding UTF8
    if ($deployText -notmatch 'INTEGRATION_SECRET') {
        $integrationRow = "| ``INTEGRATION_SECRET`` | *(Optional)* Shared secret for the engine HTTP bridge. Set same value as ``STOCKLENS_INTEGRATION_SECRET`` in ``engine/.env``. | ``random-32-char-string`` |`r`n"
        $deployText = [regex]::Replace($deployText, '(\| ``CRON_SECRET`` \|[^\r\n]+\r?\n)', "`${1}$integrationRow", 1)
    }
    if ($deployText -notmatch 'Indian Insider Engine') {
        $engineBlock = Get-Content (Join-Path $OverlayRoot "DEPLOYMENT-engine-section.md") -Raw -Encoding UTF8
        $deployText = [regex]::Replace(
            $deployText,
            '(\r?\n---\r?\n\r?\n## [^\r\n]*GitHub Actions Secrets)',
            "$engineBlock`$1",
            1
        )
    }
    Set-Content $deploy -Value $deployText -Encoding UTF8 -NoNewline

    git add engine/ README.md .gitignore package.json DEPLOYMENT.md src/app/api/integrations/signals/route.ts
    $status = git status --porcelain
    if (-not $status) {
        Write-Host "==> No changes - merge may already be applied."
    } else {
        git commit -m "Merge Indian Insider engine into monorepo"
        Write-Host "==> Committed. Push with:"
        Write-Host "    git push -u origin $Branch"
    }
}
finally {
    Pop-Location
}

Write-Host @"

Next steps:
  1. git push -u origin $Branch
  2. Open PR on GitHub: $Branch -> master
  3. copy $InsiderPath\.env -> $StocklensPath\engine\.env
  4. powershell -ExecutionPolicy Bypass -File $StocklensPath\engine\install\setup_windows.ps1
"@
