# wait for hazard7 big run to finish, then build the Google Drive export (folder + 2 GB zips). ASCII only.
param([string]$Name = "big1")
Set-Location (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
while (-not (Select-String -Path "data\hazard7\$Name\logs\run.log" -Pattern "done" -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep 120 }
& .venv\Scripts\python.exe -u scripts\export_hazard7.py --name $Name --pack 2 *> "data\hazard7\$Name\logs\export.log"
