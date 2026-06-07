# Apply admin Users invite page to StockLens.
# Pattern A (read-only roles) must already be on master (PR merged).
# Run: powershell -File C:\Users\karth\indian-insider\install\admin-users\apply-admin-users.ps1

$ErrorActionPreference = 'Stop'
$StockLens = 'C:\Users\karth\stocklens'
$Branch = 'cursor/admin-users-70df'

$Patch2 = Get-ChildItem -Path $PSScriptRoot -Filter '0002-*.patch' | Select-Object -First 1
if (-not $Patch2) { throw "Users patch (0002-*.patch) not found in $PSScriptRoot" }
if (-not (Test-Path $StockLens)) { throw "StockLens not found at $StockLens" }

Set-Location $StockLens
if (Test-Path '.git\rebase-apply') { Remove-Item -Recurse -Force '.git\rebase-apply' }
if (Test-Path '.git\am') { git am --abort 2>$null }

git fetch origin
git checkout master
git pull origin master

if (-not (Test-Path 'src\server\auth\appAccess.ts')) {
    throw 'Admin read-only (appAccess.ts) not on master. Merge that PR first, then re-run.'
}

if (Test-Path 'src\app\(dashboard)\admin\users\page.tsx') {
    Write-Host 'Admin Users page already present.'
    git checkout -B $Branch master 2>$null
    git log master..HEAD --oneline
    git push -u origin $Branch --force 2>$null
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
    git commit -m 'feat: admin Users page — invite and manage view-only accounts'
}

$ahead = git rev-list --count master..HEAD
if ($ahead -eq '0') { throw 'No commits ahead of master after patch — nothing to push.' }

git log master..HEAD --oneline
git push -u origin $Branch --force
Write-Host "https://github.com/Karthik96CFA/stocklens/compare/master...${Branch}?expand=1"
Write-Host ''
Write-Host 'After merge: System > Users in the sidebar to invite view-only accounts.'
