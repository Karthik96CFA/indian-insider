# One-time cleanup after monorepo migration.
# Keeps C:\Users\karth\indian-insider as backup; removes cruft from stocklens\engine.
param(
    [string]$StocklensPath = "C:\Users\karth\stocklens",
    [string]$InsiderPath = "C:\Users\karth\indian-insider"
)

$ErrorActionPreference = "Stop"
$Engine = Join-Path $StocklensPath "engine"

Write-Host "==> Cleaning stocklens engine cruft (backup kept at $InsiderPath)"

$removeDirs = @(
    (Join-Path $Engine "scratch"),
    (Join-Path $Engine "install\stocklens-overlay")
)
foreach ($d in $removeDirs) {
    if (Test-Path $d) {
        Remove-Item -Recurse -Force $d
        Write-Host "  removed $d"
    }
}

$removeFiles = @(
    (Join-Path $Engine "entire_codebase_report.md"),
    (Join-Path $Engine "test_concall.txt"),
    (Join-Path $Engine "install\merge_into_stocklens.ps1")
)
foreach ($f in $removeFiles) {
    if (Test-Path $f) {
        Remove-Item -Force $f
        Write-Host "  removed $f"
    }
}

$fixedSetup = Join-Path $InsiderPath "install\stocklens-overlay\engine\install\setup_windows.ps1"
if (Test-Path $fixedSetup) {
    Copy-Item $fixedSetup (Join-Path $Engine "install\setup_windows.ps1") -Force
    Write-Host "  updated setup_windows.ps1"
}

Write-Host ""
Write-Host "==> Removing legacy per-agent scheduled tasks (pre-orchestrator only)"
$Folder = "\IndianInsider\"
$Legacy = @("doraemon", "shinchan", "nobita", "dekisugi", "suneo", "doraemi", "collector", "detector", "scorer", "gian")
foreach ($n in $Legacy) {
    $name = "Indian-$n"
    $task = Get-ScheduledTask -TaskName $name -TaskPath $Folder -ErrorAction SilentlyContinue
    if ($task) {
        $wd = $task.Actions.WorkingDirectory
        if ($wd -and $wd -like "*indian-insider*") {
            Unregister-ScheduledTask -TaskName $name -TaskPath $Folder -Confirm:$false
            Write-Host "  - removed legacy task $name (was -> $wd)"
        }
    }
}

Write-Host ""
Write-Host "Done. Backup folder untouched: $InsiderPath"
Write-Host "Active engine: $Engine"
Write-Host "Current tasks (tick/morning/...) should point to stocklens\engine"
