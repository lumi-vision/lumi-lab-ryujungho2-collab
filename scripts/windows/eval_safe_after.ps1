# A3 model-only safe-input measurement after the main evaluation (pre-registered 2026-11-22). ASCII only.
param([string]$TeamDir, [string]$Data = "big1")
Set-Location (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$py = '.venv\Scripts\python.exe'
$logdir = "data\correction\logs"
while (-not (Select-String -Path "$logdir\eval.log" -Pattern " done$" -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep 120 }
$models = @(Get-ChildItem "data\correction\*.done" | ForEach-Object { $_.BaseName } | Where-Object { $_ -match "^h_s\d_e60$" } | Sort-Object)
$argv = @("-u", "scripts\eval_safe_input.py", "--team-dir", $TeamDir, "--data", $Data, "--models") + $models
Start-Process -FilePath $py -ArgumentList $argv -NoNewWindow -Wait -RedirectStandardOutput "$logdir\eval_safe.log" -RedirectStandardError "$logdir\eval_safe.err"
"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') safe-input done" | Out-File -Append -Encoding ascii "$logdir\eval.log"
