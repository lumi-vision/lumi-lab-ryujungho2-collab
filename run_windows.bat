@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  py -3 -m venv .venv || goto :failed
)
.venv\Scripts\python.exe -m pip install -e ".[precise,dev]" || goto :failed
.venv\Scripts\python.exe scripts\generate_demo.py || goto :failed
.venv\Scripts\python.exe -m lumi inspect sample\lumi_demo.avi || goto :failed
.venv\Scripts\python.exe -m lumi scan sample\lumi_demo.avi --output output\frames.jsonl || goto :failed
.venv\Scripts\python.exe -m pytest -q || goto :failed
echo.
echo LUMI 1단계 준비 완료. 미리보기에는 PSE 보정 필터가 없습니다.
echo 위험 요소가 없음을 확인한 데모 영상을 수동으로 열려면:
echo .venv\Scripts\python.exe -m lumi preview sample\lumi_demo.avi --allow-preview
pause
exit /b 0
:failed
echo 설치 또는 테스트에 실패했습니다. 위의 오류 메시지를 확인하세요.
pause
exit /b 1
