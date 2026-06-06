$ErrorActionPreference = "SilentlyContinue"
$Folder = "\IndianInsider\"
$Names = @(
    "tick", "morning", "briefing", "portfolio", "eod", "weekly", "research",
    "doraemon", "shinchan", "nobita", "dekisugi", "suneo", "doraemi",
    "collector", "detector", "scorer", "gian"
)
foreach ($n in $Names) {
    $name = "Indian-$n"
    if (Get-ScheduledTask -TaskName $name -TaskPath $Folder -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $name -TaskPath $Folder -Confirm:$false
        Write-Host "  - removed $Folder$name"
    }
}
Write-Host "All engine tasks unregistered."
