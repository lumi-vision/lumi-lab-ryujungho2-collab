# extra experiment (added after seeing results, 2026-11-21): continue H and H-noattn from epoch 30 to 60. ASCII only.
# continuing from the epoch-30 resume state (optimizer, LR schedule, RNG) equals training 60 epochs from scratch with the same seed.
param([string]$Data = "big1")
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$py = '.venv\Scripts\python.exe'
$logdir = "data\correction\logs"
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File -Append -Encoding ascii "$logdir\long.log" }
Log "waiting for training queue"
while (-not (Select-String -Path "$logdir\queue.log" -Pattern "queue finished$" -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep 120 }
$runs = @(@("h_s0", "h_s0_e60"), @("abl_noattn", "abl_noattn_e60", "--no-attn"))
foreach ($r in $runs) {
  $src = $r[0]; $name = $r[1]
  $extra = @(); if ($r.Count -gt 2) { $extra = $r[2..($r.Count - 1)] }
  if (-not (Test-Path "data\correction\$name.last.pt")) {
    Copy-Item "data\correction\$src.last.pt" "data\correction\$name.last.pt"
    Copy-Item "data\correction\$src.pt" "data\correction\$name.pt"
    Log "copied $src (epoch 30 state) -> $name"
  }
  for ($try = 1; $try -le 20; $try++) {
    if (Test-Path "data\correction\$name.done") { break }
    Log "train $name try $try"
    $argv = @("-u", "scripts\train_correction.py", "--run", $name, "--data", $Data, "--max-epochs", "60", "--resume") + $extra
    Start-Process -FilePath $py -ArgumentList $argv -NoNewWindow -Wait -RedirectStandardOutput "$logdir\${name}_try$try.log" -RedirectStandardError "$logdir\${name}_try$try.err"
  }
}
Log "long finished"
