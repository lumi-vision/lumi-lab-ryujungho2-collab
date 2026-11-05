# LUMI 설치 스크립트 — 노트북(CPU)과 집 GPU PC(RTX 50 시리즈) 공용
#
# 사용 (PowerShell, 저장소 폴더에서):
#   .\setup_windows.ps1                 # 자동: NVIDIA GPU가 있으면 cu132, 없으면 CPU 빌드
#   .\setup_windows.ps1 -Torch cpu      # 강제 CPU (노트북)
#   .\setup_windows.ps1 -Torch cu130    # 드라이버가 CUDA 13.2를 지원하지 않을 때
#
# 저장소 폴더에는 git에 올릴 파일만 둔다. 그래서 venv는 저장소 옆의 ..\_work\.venv에 만들고,
# 패키지는 pip install -e 대신 .pth 파일로 src를 등록한다(src에 egg-info가 생기지 않음).
# 같은 .pth에서 바이트코드 캐시(__pycache__) 생성도 끈다.
param([ValidateSet("auto", "cpu", "cu128", "cu130", "cu132")][string]$Torch = "auto")
$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot
$work = Join-Path (Split-Path $repo -Parent) "_work"
$venv = Join-Path $work ".venv"
$py = Join-Path $venv "Scripts\python.exe"

if ($Torch -eq "auto") {
    $Torch = "cpu"
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        nvidia-smi | Out-Null
        if ($LASTEXITCODE -eq 0) { $Torch = "cu132" }
    }
}
Write-Host "venv: $venv · torch: $Torch"

New-Item -ItemType Directory -Force $work | Out-Null
if (-not (Test-Path $py)) { py -3 -m venv $venv; if ($LASTEXITCODE -ne 0) { throw "venv 생성 실패" } }
& $py -m pip install --upgrade pip
# torch를 먼저 지정 인덱스에서 설치해야 이후 의존성 해석에서 다른 빌드로 바뀌지 않는다.
& $py -m pip install torch --index-url "https://download.pytorch.org/whl/$Torch"
if ($LASTEXITCODE -ne 0) { throw "torch 설치 실패" }

# 의존성은 pyproject.toml 한 곳에서만 관리한다(기본 + precise + dev).
$req = Join-Path $work "requirements.lock.txt"
& $py -c "import tomllib,sys; p=tomllib.load(open(sys.argv[1],'rb'))['project']; o=p['optional-dependencies']; print('\n'.join(p['dependencies']+o['precise']+o['dev']))" (Join-Path $repo "pyproject.toml") | Set-Content -Encoding ascii $req
& $py -m pip install -r $req
if ($LASTEXITCODE -ne 0) { throw "의존성 설치 실패" }

$site = & $py -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
@("$repo\src", "import sys; sys.dont_write_bytecode = True") | Set-Content -Encoding ascii (Join-Path $site "lumi-src.pth")

Push-Location $repo
try {
    if ($Torch -eq "cpu") { & $py scripts\check_env.py } else { & $py scripts\check_env.py --require-cuda }
    if ($LASTEXITCODE -ne 0) { throw "환경 진단 실패: 위 [!!] 항목을 확인하세요." }
    & $py -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw "테스트 실패" }
} finally { Pop-Location }
Write-Host "`nLUMI 환경 준비 완료. 실행: $py scripts\... (docs\COLLABORATION.md 참고)"
