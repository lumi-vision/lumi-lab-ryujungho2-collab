# after the Kaya review videos are made, zip them for the user to upload (team-only cloud, no public link). ASCII only.
Set-Location (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$logdir = "data\correction\logs"
while (-not (Select-String -Path "$logdir\eval.log" -Pattern " done$" -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep 120 }
$stamp = Get-Date -Format 'yyyyMMdd'
$dst = "data\export\LUMI_kaya_review_$stamp"
New-Item -ItemType Directory -Force $dst | Out-Null
Copy-Item "data\correction\kaya_review\*" $dst -Force
Copy-Item "checks\correction\eval_kaya_all.md" "$dst\eval_kaya_all.md" -Force -ErrorAction SilentlyContinue
Copy-Item "docs\REPORT_PROFESSOR_2026-11-23.md" "$dst\REPORT.md" -Force -ErrorAction SilentlyContinue
Compress-Archive -Path "$dst\*" -DestinationPath "data\export\LUMI_kaya_review_$stamp.zip" -Force
"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') zipped data\export\LUMI_kaya_review_$stamp.zip" | Out-File -Append -Encoding ascii "$logdir\eval.log"
