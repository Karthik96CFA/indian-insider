# Apply alert QA fixes to StockLens.
# Run: powershell -File C:\Users\karth\indian-insider\install\qa-alerts\apply-qa-alerts.ps1

$ErrorActionPreference = 'Stop'
$StockLens = 'C:\Users\karth\stocklens'
$Patch = Get-ChildItem -Path $PSScriptRoot -Filter '0001-*.patch' | Select-Object -First 1

if (-not (Test-Path $StockLens)) { throw "StockLens not found at $StockLens" }
if (-not $Patch) { throw "Patch not found in $PSScriptRoot" }

Set-Location $StockLens

if (Test-Path '.git\rebase-apply') { Remove-Item -Recurse -Force '.git\rebase-apply' }
if (Test-Path '.git\am') { Remove-Item -Recurse -Force '.git\am' }

git fetch origin
git checkout master
git pull origin master
git checkout -B cursor/qa-alerts-fix-70df master

git am $Patch.FullName
if ($LASTEXITCODE -ne 0) {
    Write-Host ''
    Write-Host 'git am failed. Trying git apply --3way...' -ForegroundColor Yellow
    git am --abort 2>$null
    git apply --3way $Patch.FullName
    if ($LASTEXITCODE -ne 0) { throw 'Patch apply failed. Paste the error in chat.' }
    git add -A
    git commit -m 'fix: alert pipeline QA - secrets, ticker match, Telegram retry'
}

Write-Host ''
Write-Host 'Commits ahead of master:' -ForegroundColor Cyan
git log master..HEAD --oneline
if (-not (git log master..HEAD --oneline)) { throw 'No commits on branch. Do not push yet.' }

git push -u origin cursor/qa-alerts-fix-70df --force
Write-Host ''
Write-Host 'Open PR:' -ForegroundColor Green
Write-Host 'https://github.com/Karthik96CFA/stocklens/compare/master...cursor/qa-alerts-fix-70df?expand=1'
