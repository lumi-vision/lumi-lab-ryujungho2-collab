# judge all South 2021 videos with Partial B2 after the correction evaluation frees the GPU (judging only). ASCII only.
param([string]$TeamDir)
Set-Location (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$py = '.venv\Scripts\python.exe'
$logdir = "data\correction\logs"
while (-not (Select-String -Path "$logdir\eval.log" -Pattern "safe-input done$" -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep 120 }
New-Item -ItemType Directory -Force "data\judge" | Out-Null
for ($try = 1; $try -le 5; $try++) {
  $log = "data\judge\south_partial_try$try.log"
  Start-Process -FilePath $py -ArgumentList @("-u", "scripts\judge_south_partial.py", "--team-dir", $TeamDir) -NoNewWindow -Wait -RedirectStandardOutput $log -RedirectStandardError "data\judge\south_partial_try$try.err"
  if (Select-String -Path $log -Pattern "DONE" -Quiet -ErrorAction SilentlyContinue) { break }
}
"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') south partial judging finished" | Out-File -Append -Encoding ascii "data\judge\south_partial.done.log"
