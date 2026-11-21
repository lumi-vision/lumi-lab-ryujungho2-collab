# extra extension decided after the long-run rule (2026-11-22, see CORRECTION_PROTOCOL.md section 8). ASCII only.
#  h_s1 missed the "best within last 2 epochs" rule only because its noisy val loss peaked at 26/30, while its
#  recoverable PSNR was highest at the last epoch (28.95) and its LR was halved early (ep 11).
#  -> extend h_s1 to 60 like the other two H seeds so the seed comparison stays symmetric.
param([string]$Data = "big1")
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$py = '.venv\Scripts\python.exe'
$logdir = "data\correction\logs"
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File -Append -Encoding ascii "$logdir\extra.log" }

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

# one training at a time: wait for the long runs
Log "waiting for long runs"
while (-not (Select-String -Path "$logdir\long.log" -Pattern "long finished$" -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep 120 }
Extend "h_s1" "h_s1_e60" 60 @("--seed", "1")
Log "extra finished"
