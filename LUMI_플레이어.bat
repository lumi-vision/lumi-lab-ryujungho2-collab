@echo off
chcp 65001 > nul
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

if not exist ".venv\Scripts\python.exe" (
  echo [오류] .venv 가 없습니다. README 의 "처음 설치" 를 먼저 해 주세요.
  pause
  exit /b 1
)

echo ============================================================
echo  LUMI 영상 플레이어 (http://127.0.0.1:8765/)
echo  - 브라우저가 자동으로 열립니다. 폴더 열기 / 파일 열기로 검사하세요.
echo  - 이 창을 닫으면 서버가 꺼집니다.
echo  - 광과민성 위험 자극이 있으니 다른 사람이 보는 화면에서는 재생하지 마세요.
echo ============================================================
echo.

".venv\Scripts\python.exe" scripts\make_playlist.py --serve %*

echo.
pause
