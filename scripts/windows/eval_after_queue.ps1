# after the training queue: train the CNN end-to-end baseline (E0), then evaluate val -> test -> Kaya. ASCII only.
param([string]$TeamDir, [string]$Data = "big1")
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$py = '.venv\Scripts\python.exe'
$logdir = "data\correction\logs"
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File -Append -Encoding ascii "$logdir\eval.log" }
Log "waiting for training queue"
while (-not (Select-String -Path "$logdir\queue.log" -Pattern " done" -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep 120 }
for ($try = 1; $try -le 4; $try++) {
  if (Test-Path "data\correction\cnn_e0.done") { break }
  Log "train cnn_e0 try $try"
  $argv = @("-u", "scripts\train_correction.py", "--run", "cnn_e0", "--data", $Data, "--no-attn", "--no-phys", "--w-layer", "0", "--w-safe", "0")
  if ($try -gt 1) { $argv += "--resume" }
  Start-Process -FilePath $py -ArgumentList $argv -NoNewWindow -Wait -RedirectStandardOutput "$logdir\cnn_e0_try$try.log" -RedirectStandardError "$logdir\cnn_e0_try$try.err"
}
$models = @("h_s0", "h_s1", "h_s2", "abl_noattn", "abl_nophys", "abl_nolayer", "abl_nosafe", "cnn_e0")
foreach ($step in @(@("val", "--split", "val", "--limit", "70"), @("test", "--split", "test"), @("kaya", "--real", "kaya"))) {
  Log "eval $($step[0])"
  $argv = @("-u", "scripts\eval_correction.py", "--team-dir", $TeamDir, "--data", $Data, "--tag", "all", "--models") + $models + $step[1..($step.Count - 1)]
  Start-Process -FilePath $py -ArgumentList $argv -NoNewWindow -Wait -RedirectStandardOutput "$logdir\eval_$($step[0]).log" -RedirectStandardError "$logdir\eval_$($step[0]).err"
}
Log "done"
