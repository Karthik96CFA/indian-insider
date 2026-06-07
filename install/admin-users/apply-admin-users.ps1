# Apply admin read-only roles + Users invite page to StockLens.
# Includes Pattern A (view-only mirror) and /admin/users.
# Run: powershell -File C:\Users\karth\indian-insider\install\admin-users\apply-admin-users.ps1

$ErrorActionPreference = 'Stop'
$StockLens = 'C:\Users\karth\stocklens'
$Branch = 'cursor/admin-users-70df'

$Patch = Get-ChildItem -Path $PSScriptRoot -Filter '0001-*.patch' | Select-Object -First 1
if (-not $Patch) { throw "Patch not found in $PSScriptRoot" }
if (-not (Test-Path $StockLens)) { throw "StockLens not found at $StockLens" }

Set-Location $StockLens
if (Test-Path '.git\rebase-apply') { Remove-Item -Recurse -Force '.git\rebase-apply' }
if (Test-Path '.git\am') { git am --abort 2>$null }

git fetch origin
git checkout master
git pull origin master
git checkout -B $Branch master

# Skip if already applied
if (Test-Path 'src\app\(dashboard)\admin\users\page.tsx') {
    Write-Host 'Admin Users page already present.'
    git log master..HEAD --oneline
    git push -u origin $Branch --force 2>$null
    Write-Host "https://github.com/Karthik96CFA/stocklens/compare/master...${Branch}?expand=1"
    exit 0
}

git am $Patch.FullName
if ($LASTEXITCODE -ne 0) {
    git am --abort 2>$null
    git apply --3way $Patch.FullName
    if ($LASTEXITCODE -ne 0) { throw 'Patch apply failed.' }
    git add -A
    git commit -m 'feat: admin read-only roles and Users invite page'
}

$ahead = git rev-list --count master..HEAD
if ($ahead -eq '0') { throw 'No commits ahead of master after patch.' }

git log master..HEAD --oneline
git push -u origin $Branch --force
Write-Host "https://github.com/Karthik96CFA/stocklens/compare/master...${Branch}?expand=1"
Write-Host ''
Write-Host 'After merge: set ADMIN_EMAIL=your@email.com in Railway, redeploy, then use System > Users to invite others.'
