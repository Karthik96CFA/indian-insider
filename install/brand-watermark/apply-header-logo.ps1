# Apply Ex Libris corner-logo fix to StockLens (removes full-page background watermark).
# Run from anywhere:  .\apply-header-logo.ps1
# Or:  powershell -File C:\Users\karth\indian-insider\install\brand-watermark\apply-header-logo.ps1

$ErrorActionPreference = 'Stop'
$StockLens = 'C:\Users\karth\stocklens'
$Patch = Join-Path $PSScriptRoot '0002-fix-remove-full-page-watermark-show-Ex-Libris-logo-t.patch'

if (-not (Test-Path $StockLens)) { throw "StockLens not found at $StockLens" }
if (-not (Test-Path $Patch)) { throw "Patch not found at $Patch" }

Set-Location $StockLens

# Clear failed git am / rebase state from a previous attempt
if (Test-Path '.git\rebase-apply') { Remove-Item -Recurse -Force '.git\rebase-apply' }
if (Test-Path '.git\am') { Remove-Item -Recurse -Force '.git\am' }

git fetch origin
git checkout master
git pull origin master
git checkout -B cursor/header-logo-70df master

git am $Patch
if ($LASTEXITCODE -ne 0) {
    Write-Host ''
    Write-Host 'git am failed. Aborting and trying git apply --3way...' -ForegroundColor Yellow
    git am --abort 2>$null
    git apply --3way $Patch
    if ($LASTEXITCODE -ne 0) { throw 'Patch apply failed — paste the error in chat.' }
    git add -A
    git commit -m "fix: remove full-page watermark, Ex Libris stamp top-right only"
}

Write-Host ''
Write-Host 'Commits ahead of master:' -ForegroundColor Cyan
git log master..HEAD --oneline
if (-not (git log master..HEAD --oneline)) { throw 'No commits on branch — do not push yet.' }

git push -u origin cursor/header-logo-70df --force
Write-Host ''
Write-Host 'Open PR:' -ForegroundColor Green
Write-Host 'https://github.com/Karthik96CFA/stocklens/compare/master...cursor/header-logo-70df?expand=1'
