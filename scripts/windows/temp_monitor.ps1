# log GPU / CPU temperature and load every minute to data\correction\logs\temps.csv. ASCII only.
#  GPU: nvidia-smi. CPU temperature: Windows has no built-in sensor access on this board (WMI thermal zone not supported);
#  if LibreHardwareMonitor is running (it publishes WMI namespace root/LibreHardwareMonitor) the CPU package temperature is read from it.
#  Stops after the whole evaluation (eval.log "safe-input done").
param([int]$EverySec = 60, [int]$GpuWarnC = 83)
Set-Location (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$logdir = "data\correction\logs"
$csv = "$logdir\temps.csv"
if (-not (Test-Path $csv)) { "time,gpu_c,gpu_util,gpu_w,fan_pct,sm_mhz,throttle,gpu_mem_mb,cpu_load,cpu_c" | Out-File -Encoding ascii $csv }
function CpuTemp {
  try {
    $s = Get-CimInstance -Namespace root/LibreHardwareMonitor -ClassName Sensor -ErrorAction Stop |
         Where-Object { $_.SensorType -eq 'Temperature' -and $_.Name -match 'Tctl|Tdie|Package|Core \(Tctl' } | Select-Object -First 1
    if ($s) { return [math]::Round($s.Value, 1) }
  } catch {}
  return ""
}
while (-not (Select-String -Path "$logdir\eval.log" -Pattern "safe-input done$" -Quiet -ErrorAction SilentlyContinue)) {
  $g = (& nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw,fan.speed,clocks.sm,clocks_throttle_reasons.active,memory.used --format=csv,noheader,nounits) -replace ' ', ''
  $cpu = [math]::Round((Get-CimInstance Win32_Processor | Measure-Object LoadPercentage -Average).Average, 0)
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'),$g,$cpu,$(CpuTemp)"
  $line | Out-File -Append -Encoding ascii $csv
  $t = [int]($g.Split(',')[0])
  if ($t -ge $GpuWarnC) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') GPU HOT $t C" | Out-File -Append -Encoding ascii "$logdir\temps_warn.log" }
  Start-Sleep $EverySec
}
