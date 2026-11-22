# every model to exactly 60 epochs, early stopping off (user decision 2026-11-22 15:30). ASCII only.
#  - continues each run from its saved resume state (optimizer, LR schedule incl. its own patience 2, RNG)
#  - --patience 1000 only disables early stopping; the best-val checkpoint (<name>.pt) is still what gets evaluated
#  - runs that had stopped early under the old rule: their files are backed up to data\correction\early_stop_backup first
param([string]$Data = "big1")
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$py = '.venv\Scripts\python.exe'
$logdir = "data\correction\logs"
$bk = "data\correction\early_stop_backup"
New-Item -ItemType Directory -Force $bk | Out-Null
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File -Append -Encoding ascii "$logdir\extra.log" }

function Run60($src, $name, $extra) {
  $c = "data\correction"
  if (Test-Path "$c\$name.done") {
    $d = Get-Content "$c\$name.done" | ConvertFrom-Json
    if ([int]$d.epochs -ge 60) { Log "$name already at 60 epochs"; return }
    foreach ($ext in @(".done", ".pt", ".last.pt")) { Copy-Item "$c\$name$ext" "$bk\$name$ext" -Force }
    Copy-Item "checks\correction\${name}_log.jsonl" "$bk\${name}_log.jsonl" -Force -ErrorAction SilentlyContinue
    Remove-Item "$c\$name.done"
    Log "$name stopped early at $($d.epochs) (best $($d.best_epoch)) -> backed up, continue to 60"
  }
  if (-not (Test-Path "$c\$name.last.pt")) {
    Copy-Item "$c\$src.last.pt" "$c\$name.last.pt"
    Copy-Item "$c\$src.pt" "$c\$name.pt"
    Log "copied $src state -> $name"
  }
  for ($try = 1; $try -le 40; $try++) {
    if (Test-Path "$c\$name.done") { break }
    $tag = "60_try$try"
    Log "train $name try $try"
    $argv = @("-u", "scripts\train_correction.py", "--run", $name, "--data", $Data, "--max-epochs", "60", "--patience", "1000", "--resume") + $extra
    Start-Process -FilePath $py -ArgumentList $argv -NoNewWindow -Wait -RedirectStandardOutput "$logdir\${name}_$tag.log" -RedirectStandardError "$logdir\${name}_$tag.err"
  }
}

Log "all-60 start"
Run60 "abl_nophys_plus10" "abl_nophys_e60" @("--no-phys")
Run60 "h_s1" "h_s1_e60" @("--seed", "1")
Run60 "h_s0" "h_s0_e60" @()
Run60 "h_s2" "h_s2_e60" @("--seed", "2")
Run60 "abl_noattn" "abl_noattn_e60" @("--no-attn")
Run60 "abl_nosafe" "abl_nosafe_e60" @("--w-safe", "0")
Run60 "cnn_e0" "cnn_e0_e60" @("--no-attn", "--no-phys", "--w-layer", "0", "--w-safe", "0")
Run60 "abl_nolayer" "abl_nolayer_e60" @("--w-layer", "0")
Log "extra finished"
