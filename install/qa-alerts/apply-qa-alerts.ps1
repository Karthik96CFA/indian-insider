# Apply alert QA fixes to StockLens.
# Run: powershell -File C:\Users\karth\indian-insider\install\qa-alerts\apply-qa-alerts.ps1

$ErrorActionPreference = 'Stop'
$StockLens = 'C:\Users\karth\stocklens'
$Patch = Join-Path $PSScriptRoot '0001-fix-alert-pipeline-QA-secrets-ticker-match-Telegram-re.patch'

if (-not (Test-Path $StockLens)) { throw "StockLens not found at $StockLens" }
if (-not (Test-Path $Patch)) { throw "Patch not found at $Patch" }

Set-Location $StockLens
if (Test-Path '.git\rebase-apply') { Remove-Item -Recurse -Force '.git\rebase-apply' }
if (Test-Path '.git\am') { Remove-Item -Recurse -Force '.git\am' }

git fetch origin
git checkout master
git pull origin master
git checkout -B cursor/qa-alerts-fix-70df master
git am $Patch
if ($LASTEXITCODE -ne 0) {
    git am --abort 2>$null
    git apply --3way $Patch
    git add -A
    git commit -m "fix: alert pipeline QA — secrets, ticker match, Telegram retry"
}

git log master..HEAD --oneline
git push -u origin cursor/qa-alerts-fix-70df --force
Write-Host 'https://github.com/Karthik96CFA/stocklens/compare/master...cursor/qa-alerts-fix-70df?expand=1'
