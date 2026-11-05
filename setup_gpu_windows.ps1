# LUMI 집 GPU PC(RTX 50 시리즈) 설치 스크립트
# 사용: PowerShell에서  .\setup_gpu_windows.ps1            (기본 cu132)
#       드라이버가 CUDA 13.2를 지원하지 않으면  .\setup_gpu_windows.ps1 -CudaTag cu130
param([string]$CudaTag = "cu132")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

nvidia-smi
if ($LASTEXITCODE -ne 0) { throw "nvidia-smi 실패: NVIDIA 드라이버를 먼저 설치하세요." }

if (-not (Test-Path .venv\Scripts\python.exe)) { py -3 -m venv .venv }
$py = ".venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
# CUDA 빌드 torch를 먼저 설치해야 이후 의존성 해석에서 CPU 빌드로 바뀌지 않는다.
& $py -m pip install torch --index-url "https://download.pytorch.org/whl/$CudaTag"
& $py -m pip install -e ".[precise,dev]"
& $py scripts\check_env.py --require-cuda
if ($LASTEXITCODE -ne 0) { throw "환경 진단 실패: 위 [!!] 항목을 확인하세요." }
& $py -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "테스트 실패" }
Write-Host "`nLUMI GPU 환경 준비 완료. 다음: docs\HOME_GPU_SETUP.md 의 '첫 실행' 절"
