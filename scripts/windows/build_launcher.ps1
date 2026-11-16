$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
if (-not (Test-Path -LiteralPath $compiler)) { throw '.NET Framework C# compiler not found' }
$source = Join-Path $PSScriptRoot 'LumiLauncher.cs'
$output = Join-Path $projectRoot 'LUMI.exe'
& $compiler /nologo /target:winexe /platform:anycpu /optimize+ /codepage:65001 /r:System.Windows.Forms.dll /r:System.Drawing.dll /r:System.Web.Extensions.dll "/out:$output" $source
if ($LASTEXITCODE -ne 0) { throw 'Launcher build failed' }
Copy-Item -LiteralPath $output -Destination (Join-Path $projectRoot 'LUMI_생성기.exe') -Force
Copy-Item -LiteralPath $output -Destination (Join-Path $projectRoot 'LUMI_플레이어.exe') -Force
Get-FileHash -LiteralPath $output -Algorithm SHA256
