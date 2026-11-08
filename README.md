# lumi-lab-ryujungho2-collab

LUMI 프로젝트 개인 실험 저장소입니다. 2026-11-08에 처음부터 다시 설계하기 위해 초기화했습니다.

- 데이터 출처: [DATA_SOURCES.md](DATA_SOURCES.md)
- 초기화 전 작업 전체: 브랜치 [`archive/2026-11-08-before-reset`](https://github.com/lumi-vision/lumi-lab-ryujungho2-collab/tree/archive/2026-11-08-before-reset) (태그 `before-reset-2026-11-08`)

- 작업 설계: [PLAN.md](PLAN.md). 깜빡임 학습 데이터와 정답지를 규칙 기반으로 합성합니다.

## 처음 설치 (PowerShell)

```powershell
cd C:\dev\lumi-vision\lumi-lab-ryujungho2-collab
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`pip install -r requirements.txt` 한 번으로 필요한 패키지와 이 저장소의 `lumi_synth` 코드가 모두 `.venv`에 설치됩니다.

## 단계별 실행

```powershell
python scripts\step0_check.py   # 0단계: 환경·원본 영상 점검
python scripts\step1_cut.py     # 1단계: 원본 5초 클립
pytest                          # 측정 함수 자동 테스트
```

결과는 `checks\stepN\README.md`에서 확인합니다. 계획은 [PLAN.md](PLAN.md)에 있습니다.
