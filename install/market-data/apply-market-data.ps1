# Apply market data Phase 2+3 (NSE/BSE providers + corp action sync) to StockLens.
# Phase 1 is already merged on master (PR #39). This script applies only the delta.
# Run: powershell -File C:\Users\karth\indian-insider\install\market-data\apply-market-data.ps1

$ErrorActionPreference = 'Stop'
$StockLens = 'C:\Users\karth\stocklens'
$Branch = 'cursor/market-data-service-70df'

$Patch2 = Get-ChildItem -Path $PSScriptRoot -Filter '0002-*.patch' | Select-Object -First 1
if (-not $Patch2) { throw "Phase 2 patch (0002-*.patch) not found in $PSScriptRoot" }
if (-not (Test-Path $StockLens)) { throw "StockLens not found at $StockLens" }

Set-Location $StockLens
if (Test-Path '.git\rebase-apply') { Remove-Item -Recurse -Force '.git\rebase-apply' }
if (Test-Path '.git\am') { git am --abort 2>$null }

git fetch origin
git checkout master
git pull origin master

# Phase 1 must already be on master
if (-not (Test-Path 'src\services\marketData\marketDataService.ts')) {
    throw 'Phase 1 not found on master. Merge PR #39 first, then re-run this script.'
}

# Already applied?
if (Test-Path 'src\services\marketData\corporateActionSync.ts') {
    Write-Host 'Phase 2+3 already present. Syncing branch to master + existing commits...'
    git checkout -B $Branch
    git push -u origin $Branch --force
    git log master..HEAD --oneline
    Write-Host "https://github.com/Karthik96CFA/stocklens/compare/master...${Branch}?expand=1"
    exit 0
}

git checkout -B $Branch master
git am $Patch2.FullName
if ($LASTEXITCODE -ne 0) {
    git am --abort 2>$null
    git apply --3way $Patch2.FullName
    if ($LASTEXITCODE -ne 0) { throw 'Patch apply failed. Resolve conflicts manually.' }
    git add -A
    git commit -m 'feat: NSE/BSE live providers and corporate action auto-ingest'
}

$ahead = git rev-list --count master..HEAD
if ($ahead -eq '0') { throw 'No commits ahead of master after patch — nothing to push.' }

git log master..HEAD --oneline
git push -u origin $Branch --force
Write-Host "https://github.com/Karthik96CFA/stocklens/compare/master...${Branch}?expand=1"
