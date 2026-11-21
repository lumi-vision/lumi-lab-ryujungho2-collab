# watchdog: if the running training's log has not grown for $StaleMin minutes, kill it so the queue resumes it (--resume). ASCII only.
param([int]$StaleMin = 15)
Set-Location (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$logdir = "data\correction\logs"
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File -Append -Encoding ascii "$logdir\watchdog.log" }
Log "start (stale > $StaleMin min)"
while (-not (Select-String -Path "$logdir\queue.log" -Pattern "queue finished$" -Quiet -ErrorAction SilentlyContinue)) {
  Start-Sleep 60
  $procs = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'train_correction\.py' }
  if (-not $procs) { continue }
  $latest = Get-ChildItem "$logdir\*_try*.log" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if (-not $latest) { continue }
  $age = ((Get-Date) - $latest.LastWriteTime).TotalMinutes
  if ($age -gt $StaleMin) {
    Log "stale $([int]$age) min on $($latest.Name) -> kill training (queue will resume)"
    $procs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep 120
  }
}
Log "queue finished -> exit"
