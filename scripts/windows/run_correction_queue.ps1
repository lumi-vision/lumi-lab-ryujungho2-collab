# correction model training queue (detached). ASCII only.
# waits for the hazard7 big run to finish (run.log contains "done"), then trains H, ablations, extra seeds.
param([string]$Data = "big1")
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$py = '.venv\Scripts\python.exe'
$logdir = "data\correction\logs"
New-Item -ItemType Directory -Force $logdir | Out-Null
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File -Append -Encoding ascii "$logdir\queue.log" }
Log "waiting for data\hazard7\$Data\logs\run.log done"
while (-not (Select-String -Path "data\hazard7\$Data\logs\run.log" -Pattern "done" -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep 120 }
$runs = @(
  @("h_s0", "--seed", "0"),
  @("abl_noattn", "--no-attn"),
  @("abl_nophys", "--no-phys"),
  @("abl_nolayer", "--w-layer", "0"),
  @("abl_nosafe", "--w-safe", "0"),
  @("h_s1", "--seed", "1"),
  @("h_s2", "--seed", "2"),
  @("cnn_e0", "--no-attn", "--no-phys", "--w-layer", "0", "--w-safe", "0")
)
foreach ($r in $runs) {
  $name = $r[0]
  $extra = @()
  if ($r.Count -gt 1) { $extra = $r[1..($r.Count - 1)] }
  for ($try = 1; $try -le 4; $try++) {
    if (Test-Path "data\correction\$name.done") { break }
    Log "train $name try $try $($extra -join ' ')"
    $argv = @("-u", "scripts\train_correction.py", "--run", $name, "--data", $Data) + $extra
    if ($try -gt 1) { $argv += "--resume" }
    Start-Process -FilePath $py -ArgumentList $argv -NoNewWindow -Wait -RedirectStandardOutput "$logdir\${name}_try$try.log" -RedirectStandardError "$logdir\${name}_try$try.err"
  }
}
Log "queue finished"
