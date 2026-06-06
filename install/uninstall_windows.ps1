$ErrorActionPreference="SilentlyContinue"; $Folder="\IndianInsider\"
foreach($n in @("doraemon","shinchan","nobita","dekisugi","suneo","doraemi","collector","detector","scorer","gian")){
  $name="Indian-$n"
  if(Get-ScheduledTask -TaskName $name -TaskPath $Folder -EA SilentlyContinue){
    Unregister-ScheduledTask -TaskName $name -TaskPath $Folder -Confirm:$false
    Write-Host "  - removed $Folder$name"
  }
}
Write-Host "All tasks unregistered."
