# Apply market data service layer to StockLens.
# Run: powershell -File C:\Users\karth\indian-insider\install\market-data\apply-market-data.ps1

$ErrorActionPreference = 'Stop'
$StockLens = 'C:\Users\karth\stocklens'
$Patch = Get-ChildItem -Path $PSScriptRoot -Filter '0001-*.patch' | Select-Object -First 1

if (-not (Test-Path $StockLens)) { throw "StockLens not found at $StockLens" }
if (-not $Patch) { throw "Patch not found in $PSScriptRoot" }

Set-Location $StockLens
if (Test-Path '.git\rebase-apply') { Remove-Item -Recurse -Force '.git\rebase-apply' }

git fetch origin
git checkout master
git pull origin master
git checkout -B cursor/market-data-service-70df master
git am $Patch.FullName
if ($LASTEXITCODE -ne 0) {
    git am --abort 2>$null
    git apply --3way $Patch.FullName
    git add -A
    git commit -m 'feat: market data service layer with NSE/BSE providers and corp action sync'
}

git log master..HEAD --oneline
git push -u origin cursor/market-data-service-70df --force
Write-Host 'https://github.com/Karthik96CFA/stocklens/compare/master...cursor/market-data-service-70df?expand=1'
