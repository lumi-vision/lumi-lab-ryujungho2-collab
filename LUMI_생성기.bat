@echo off
chcp 65001 > nul
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if not exist ".venv\Scripts\python.exe" (
  echo [오류] .venv 가 없습니다. README 의 설치 방법을 확인하세요.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" scripts\make_playlist.py --serve --open-page generate %*
pause
