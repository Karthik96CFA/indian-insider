$ErrorActionPreference="Stop"
$Root="$env:USERPROFILE\indian-insider"; $Agents="$Root\agents"
$Logs="$Root\.state\logs"; $Folder="\IndianInsider"
$PythonCmd = Get-Command python -EA SilentlyContinue
if ($PythonCmd) { $Python = $PythonCmd.Source } else { $Python = $null }
if (-not $Python) {
  $PyCmd = Get-Command py -EA SilentlyContinue
  if ($PyCmd) { $Python = $PyCmd.Source }
}
if(-not $Python){Write-Error "Python not found."; exit 1}
New-Item -ItemType Directory -Force -Path $Logs|Out-Null
function RT{param($N,$S,$T)
  $p="$Folder\"; $n="Indian-$N"
  if(Get-ScheduledTask -TaskName $n -TaskPath $p -EA SilentlyContinue){Unregister-ScheduledTask -TaskName $n -TaskPath $p -Confirm:$false}
  $a=New-ScheduledTaskAction -Execute $Python -Argument "`"$Agents\$S`"" -WorkingDirectory $Root
  $st=New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit(New-TimeSpan -Minutes 30)
  Register-ScheduledTask -TaskName $n -TaskPath $p -Action $a -Trigger $T -Settings $st -Description "Indian Insider $N"|Out-Null
  Write-Host "  OK $p$n"
}
Write-Host "Registering pipeline agents..."
RT "collector" "nse_collector.py" (New-ScheduledTaskTrigger -Once -At(Get-Date) -RepetitionInterval(New-TimeSpan -Minutes 30))
RT "detector"  "event_detector.py" (New-ScheduledTaskTrigger -Once -At((Get-Date).AddMinutes(2)) -RepetitionInterval(New-TimeSpan -Minutes 30))
RT "scorer"    "scoring_engine.py" (New-ScheduledTaskTrigger -Once -At((Get-Date).AddMinutes(4)) -RepetitionInterval(New-TimeSpan -Minutes 30))
RT "gian"      "gian.py"           (New-ScheduledTaskTrigger -Once -At((Get-Date).AddMinutes(6)) -RepetitionInterval(New-TimeSpan -Minutes 30))
Write-Host "`nAll pipeline agents registered. Logs -> $Logs"
