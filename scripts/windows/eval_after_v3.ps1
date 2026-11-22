# unattended evaluation (2026-11-23, replaces v2 while the user is away). ASCII only.
#  - first waits for an already running eval_correction.py (started by v2) to exit
#  - each step: one process (GPU memory), up to 6 tries; every try resumes (finished (sample, method) rows are skipped)
#  - stall guard: if the step's stdout log does not grow for 40 min, kill the process and retry
#  - CPU limited inside eval_correction.py (8 logical cores, below-normal priority)
#  - order: test -> report -> Kaya (+ corrected videos saved) -> report -> Kaya side-by-side review videos -> "done"
param([string]$TeamDir, [string]$Data = "big1")
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$py = '.venv\Scripts\python.exe'
$logdir = "data\correction\logs"
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File -Append -Encoding ascii "$logdir\eval.log" }
function EvalProcs { Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'eval_correction\.py' } }

Log "v3 start"
while (EvalProcs) { Start-Sleep 60 }
$models = @(Get-ChildItem "data\correction\*.done" | ForEach-Object { $_.BaseName } | Where-Object { $_ -notmatch "crashed|plus10" } | Sort-Object)
$save = @("h_s0_e60", "h_s1_e60", "h_s2_e60", "B2")

function RunStep($name, $extra) {
  for ($try = 1; $try -le 6; $try++) {
    Log "eval $name try $try"
    $out = "$logdir\eval_${name}_v3_try$try.log"
    $argv = @("-u", "scripts\eval_correction.py", "--team-dir", $TeamDir, "--data", $Data, "--tag", "all", "--models") + $models + $extra
    $p = Start-Process -FilePath $py -ArgumentList $argv -NoNewWindow -PassThru -RedirectStandardOutput $out -RedirectStandardError "$logdir\eval_${name}_v3_try$try.err"
    while (-not $p.HasExited) {
      Start-Sleep 60
      $age = ((Get-Date) - (Get-Item $out).LastWriteTime).TotalMinutes
      if ($age -gt 40) {
        Log "eval $name stalled $([int]$age) min -> kill"
        EvalProcs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Start-Sleep 30
        break
      }
    }
    if ((Get-Content $out -Encoding utf8 -ErrorAction SilentlyContinue | Select-String -Pattern "\.md" -Quiet)) { break }   # finished normally (printed report path)
  }
  $argv = @("-u", "scripts\eval_correction.py", "--team-dir", $TeamDir, "--data", $Data, "--tag", "all", "--report-only", "--models") + $models + $extra
  Start-Process -FilePath $py -ArgumentList $argv -NoNewWindow -Wait -RedirectStandardOutput "$logdir\eval_${name}_report.log" -RedirectStandardError "$logdir\eval_${name}_report.err"
  Log "eval $name finished"
}

RunStep "test" @("--split", "test")
RunStep "kaya" (@("--real", "kaya", "--save-models") + $save)
Log "kaya review videos"
Start-Process -FilePath $py -ArgumentList @("-u", "scripts\make_kaya_review.py", "--methods", "h_s0_e60", "B2") -NoNewWindow -Wait -RedirectStandardOutput "$logdir\kaya_review.log" -RedirectStandardError "$logdir\kaya_review.err"
Log "done"
