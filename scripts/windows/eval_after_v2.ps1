# evaluation after all 60-epoch training (replaces eval_after_queue.ps1, 2026-11-22). ASCII only.
#  - GPU sharing: $Shards processes in parallel, each a disjoint slice of the samples (eval_correction.py --shard i/k);
#    one sample costs ~20 s of mostly serial judge/metric work, so one process leaves the GPU idle most of the time
#  - order: test (primary, 140) -> Kaya (22 real, corrected videos of H 60-epoch seeds + B2 saved for viewing) -> done
#  - val is not re-evaluated: no model selection uses it (checkpoints were chosen by val loss during training)
#  - models: every finished run except intermediate abl_nophys_plus10 and the crashed h_s0 copy
param([string]$TeamDir, [string]$Data = "big1", [int]$Shards = 3)
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$py = '.venv\Scripts\python.exe'
$logdir = "data\correction\logs"
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File -Append -Encoding ascii "$logdir\eval.log" }
Log "v2 waiting for training (extra finished)"
while (-not (Select-String -Path "$logdir\extra.log" -Pattern "extra finished$" -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep 120 }
$models = @(Get-ChildItem "data\correction\*.done" | ForEach-Object { $_.BaseName } | Where-Object { $_ -notmatch "crashed|plus10" } | Sort-Object)
Log "models: $($models -join ' ')"
$save = @("h_s0_e60", "h_s1_e60", "h_s2_e60", "B2")

function RunSharded($name, $extra) {
  Log "eval $name ($Shards shards)"
  $ps = @()
  for ($i = 0; $i -lt $Shards; $i++) {
    $argv = @("-u", "scripts\eval_correction.py", "--team-dir", $TeamDir, "--data", $Data, "--tag", "all", "--shard", "$i/$Shards", "--models") + $models + $extra
    $ps += Start-Process -FilePath $py -ArgumentList $argv -NoNewWindow -PassThru -RedirectStandardOutput "$logdir\eval_${name}_s$i.log" -RedirectStandardError "$logdir\eval_${name}_s$i.err"
    Start-Sleep 20
  }
  $ps | ForEach-Object { $_.WaitForExit() }
  $argv = @("-u", "scripts\eval_correction.py", "--team-dir", $TeamDir, "--data", $Data, "--tag", "all", "--report-only", "--models") + $models + $extra
  Start-Process -FilePath $py -ArgumentList $argv -NoNewWindow -Wait -RedirectStandardOutput "$logdir\eval_${name}_report.log" -RedirectStandardError "$logdir\eval_${name}_report.err"
  Log "eval $name finished"
}

RunSharded "test" @("--split", "test")
RunSharded "kaya" (@("--real", "kaya", "--save-models") + $save)
Log "done"
