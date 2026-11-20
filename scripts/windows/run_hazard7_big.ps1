# hazard7 large dataset run (detached). ASCII only: Windows PowerShell 5.1 misreads UTF-8 without BOM.
# usage: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\run_hazard7_big.ps1 -TeamDir <lumi-lab-rockrovks clone>
param([Parameter(Mandatory=$true)][string]$TeamDir, [string]$Name = "big1", [int]$Shards = 3)
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$py = '.venv\Scripts\python.exe'
$logdir = "data\hazard7\$Name\logs"
New-Item -ItemType Directory -Force $logdir | Out-Null
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File -Append -Encoding ascii "$logdir\run.log" }
$plan = @(@{split='train'; per=200}, @{split='val'; per=20}, @{split='test'; per=10})
foreach ($p in $plan) {
  Log "gen $($p.split) per=$($p.per)"
  Start-Process -FilePath $py -ArgumentList "-u","scripts\make_hazard7_dataset.py","--team-dir",$TeamDir,"--split",$p.split,"--per",$p.per,"--name",$Name,"--workers","6","--phase","gen" -NoNewWindow -Wait -RedirectStandardOutput "$logdir\gen_$($p.split).log" -RedirectStandardError "$logdir\gen_$($p.split).err"
}
foreach ($split in @('val','test','train')) {
  Log "judge $split shards=$Shards"
  $procs = @()
  for ($i = 0; $i -lt $Shards; $i++) {
    $procs += Start-Process -FilePath $py -ArgumentList "-u","scripts\make_hazard7_dataset.py","--team-dir",$TeamDir,"--split",$split,"--per","0","--name",$Name,"--workers","2","--phase","judge","--shard","$i/$Shards" -NoNewWindow -PassThru -RedirectStandardOutput "$logdir\judge_${split}_$i.log" -RedirectStandardError "$logdir\judge_${split}_$i.err"
  }
  $procs | Wait-Process
}
Log "done"
