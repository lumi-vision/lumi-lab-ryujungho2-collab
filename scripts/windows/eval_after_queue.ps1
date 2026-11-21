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
$models = @("h_s0", "h_s1", "h_s2", "abl_noattn", "abl_nophys", "abl_nolayer", "abl_nosafe", "cnn_e0")
foreach ($step in @(@("val", "--split", "val", "--limit", "70"), @("test", "--split", "test"), @("kaya", "--real", "kaya"))) {
  Log "eval $($step[0])"
  $argv = @("-u", "scripts\eval_correction.py", "--team-dir", $TeamDir, "--data", $Data, "--tag", "all", "--models") + $models + $step[1..($step.Count - 1)]
  Start-Process -FilePath $py -ArgumentList $argv -NoNewWindow -Wait -RedirectStandardOutput "$logdir\eval_$($step[0]).log" -RedirectStandardError "$logdir\eval_$($step[0]).err"
}
Log "done"
