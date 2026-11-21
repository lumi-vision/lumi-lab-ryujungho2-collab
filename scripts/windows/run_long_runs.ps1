# extra training after the protocol runs (added after seeing results, user decisions 2026-11-21). ASCII only.
#  - abl_nophys: plateaued -> exactly +10 epochs from its last state, early stopping off (to confirm it does not improve)
#  - every other finished run whose best epoch is within 2 of its last epoch (still improving) -> continue to 60 epochs
#    (early stopping patience 4 as in the protocol)
# continuing from the saved resume state (optimizer, LR schedule, RNG) equals training longer from scratch with the same seed.
param([string]$Data = "big1")
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$py = '.venv\Scripts\python.exe'
$logdir = "data\correction\logs"
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File -Append -Encoding ascii "$logdir\long.log" }
$flags = @{ "h_s0" = @(); "abl_noattn" = @("--no-attn"); "abl_nophys" = @("--no-phys"); "abl_nolayer" = @("--w-layer", "0");
            "abl_nosafe" = @("--w-safe", "0"); "h_s1" = @("--seed", "1"); "h_s2" = @("--seed", "2");
            "cnn_e0" = @("--no-attn", "--no-phys", "--w-layer", "0", "--w-safe", "0") }
$order = @("h_s0", "abl_noattn", "abl_nophys", "abl_nolayer", "abl_nosafe", "h_s1", "h_s2", "cnn_e0")

function Extend($src, $name, $maxEp, $extra) {
  if (-not (Test-Path "data\correction\$name.last.pt")) {
    Copy-Item "data\correction\$src.last.pt" "data\correction\$name.last.pt"
    Copy-Item "data\correction\$src.pt" "data\correction\$name.pt"
    Log "copied $src state -> $name (max epochs $maxEp)"
  }
  for ($try = 1; $try -le 25; $try++) {
    if (Test-Path "data\correction\$name.done") { break }
    Log "train $name try $try"
    $argv = @("-u", "scripts\train_correction.py", "--run", $name, "--data", $Data, "--max-epochs", "$maxEp", "--resume") + $extra
    Start-Process -FilePath $py -ArgumentList $argv -NoNewWindow -Wait -RedirectStandardOutput "$logdir\${name}_try$try.log" -RedirectStandardError "$logdir\${name}_try$try.err"
  }
}

# one training at a time on the GPU: start only after the protocol queue has finished (two trainings at once ran out of memory)
Log "waiting for training queue"
while (-not (Select-String -Path "$logdir\queue.log" -Pattern "queue finished$" -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep 120 }
foreach ($src in $order) {
  if (-not (Test-Path "data\correction\$src.done")) { Log "$src not done -> skip"; continue }
  $d = Get-Content "data\correction\$src.done" | ConvertFrom-Json
  if ($src -eq "abl_nophys") {
    Extend $src "abl_nophys_plus10" ([int]$d.epochs + 10) ($flags[$src] + @("--patience", "1000"))
  } elseif ([int]$d.best_epoch -ge [int]$d.epochs - 2) {
    Extend $src "${src}_e60" 60 $flags[$src]
  } else {
    Log "$src converged (best $($d.best_epoch) of $($d.epochs)) -> no extension"
  }
}
Log "long finished"
