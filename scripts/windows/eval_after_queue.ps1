# after the training queue (which also trains cnn_e0), evaluate val -> test -> Kaya. ASCII only.
param([string]$TeamDir, [string]$Data = "big1")
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$py = '.venv\Scripts\python.exe'
$logdir = "data\correction\logs"
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File -Append -Encoding ascii "$logdir\eval.log" }
Log "waiting for training queue"
# the queue writes exactly "queue finished" as its last line (do not match other lines containing "done")
while (-not (Select-String -Path "$logdir\queue.log" -Pattern "queue finished$" -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep 120 }
# then the 60-epoch extra runs (run_long_runs.ps1) must finish too
while (-not (Select-String -Path "$logdir\long.log" -Pattern "long finished$" -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep 120 }
# and the extra extension (run_extra_runs.ps1, h_s1_e60)
while (-not (Select-String -Path "$logdir\extra.log" -Pattern "extra finished$" -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep 120 }
# every finished model (protocol runs + extensions), in name order
$models = @(Get-ChildItem "data\correction\*.done" | ForEach-Object { $_.BaseName } | Where-Object { $_ -notmatch "crashed" } | Sort-Object)
Log "models: $($models -join ' ')"
foreach ($step in @(@("val", "--split", "val", "--limit", "70"), @("test", "--split", "test"), @("kaya", "--real", "kaya"))) {
  Log "eval $($step[0])"
  $argv = @("-u", "scripts\eval_correction.py", "--team-dir", $TeamDir, "--data", $Data, "--tag", "all", "--models") + $models + $step[1..($step.Count - 1)]
  Start-Process -FilePath $py -ArgumentList $argv -NoNewWindow -Wait -RedirectStandardOutput "$logdir\eval_$($step[0]).log" -RedirectStandardError "$logdir\eval_$($step[0]).err"
}
Log "done"
