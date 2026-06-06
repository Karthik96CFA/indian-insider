$ErrorActionPreference="Stop"
$EngineRoot = Split-Path -Parent $PSScriptRoot
$Agents = "$EngineRoot\agents"
$Logs = "$EngineRoot\.state\logs"
$Folder="\IndianInsider"
$PythonCmd = Get-Command python -EA SilentlyContinue
if ($PythonCmd) { $Python = $PythonCmd.Source } else { $Python = $null }
if (-not $Python) {
  $PyCmd = Get-Command py -EA SilentlyContinue
  if ($PyCmd) { $Python = $PyCmd.Source }
}
if(-not $Python){Write-Error "Python not found."; exit 1}
New-Item -ItemType Directory -Force -Path $Logs|Out-Null

function RT{param($N,$Phase,$T)
  $p="$Folder\"; $n="Indian-$N"
  if(Get-ScheduledTask -TaskName $n -TaskPath $p -EA SilentlyContinue){Unregister-ScheduledTask -TaskName $n -TaskPath $p -Confirm:$false}
  $a=New-ScheduledTaskAction -Execute $Python -Argument "`"$Agents\orchestrator.py`" --phase $Phase --continue-on-error" -WorkingDirectory $EngineRoot
  $st=New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit(New-TimeSpan -Hours 2)
  Register-ScheduledTask -TaskName $n -TaskPath $p -Action $a -Trigger $T -Settings $st -Description "StockLens engine $N ($Phase)"|Out-Null
  Write-Host "  OK $p$n"
}

Write-Host "Registering StockLens engine automation..."
RT "tick"      "tick"      (New-ScheduledTaskTrigger -Once -At(Get-Date) -RepetitionInterval(New-TimeSpan -Minutes 30))
RT "morning"   "morning"   (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "02:00")
RT "briefing"  "briefing"  (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "02:30")
RT "portfolio" "portfolio" (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "10:30")
RT "eod"       "eod"       (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "12:30")
RT "weekly"    "weekly"    (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At "03:30")
RT "research"  "research"  (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At "00:30")
Write-Host "`nAll engine phases registered. Logs -> $Logs"
