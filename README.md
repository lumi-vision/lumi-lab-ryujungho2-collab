# lumi-lab-ryujungho2-collab

LUMI 프로젝트 개인 실험 저장소입니다. 2026-11-08에 처음부터 다시 설계하기 위해 초기화했습니다.

- 데이터 출처: [DATA_SOURCES.md](DATA_SOURCES.md)
- 초기화 전 작업 전체: 브랜치 [`archive/2026-11-08-before-reset`](https://github.com/lumi-vision/lumi-lab-ryujungho2-collab/tree/archive/2026-11-08-before-reset) (태그 `before-reset-2026-11-08`)

- 작업 설계: [PLAN.md](PLAN.md). 깜빡임 학습 데이터와 정답지를 규칙 기반으로 합성합니다.
- 판정 기준: [docs/CRITERIA.md](docs/CRITERIA.md) · 계산법: [docs/BASICS.md](docs/BASICS.md)

### 2026-11-09 추가: 규칙 명세 · 오라클 · GAN

- **전체 흐름(먼저 읽기)**: [docs/WORKFLOW.md](docs/WORKFLOW.md)
- 판정 규칙 수식 명세와 설계 결정 D1~D4: [docs/RULES_SPEC.md](docs/RULES_SPEC.md)
- **2026-11-09 변경 기록(무엇·어떻게·왜·결과)**: [docs/CHANGELOG_2026-11-09.md](docs/CHANGELOG_2026-11-09.md)
- **집 PC 에서 할 일**: [docs/HOME_NEXT_STEPS.md](docs/HOME_NEXT_STEPS.md)
- 오라클(규칙 기반 합성): [docs/ORACLE.md](docs/ORACLE.md) · GAN: [docs/GAN.md](docs/GAN.md) · 학습 데이터 계획: [docs/DATA_PLAN.md](docs/DATA_PLAN.md)
- 실측 결과: `checks/rules/`, `checks/oracle/`, `checks/gan/`

## 처음 설치 (PowerShell)

```powershell
cd C:\dev\lumi-vision\lumi-lab-ryujungho2-collab
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch   # GAN 학습용 (집 PC 는 CUDA 빌드)
```

`pip install -r requirements.txt` 한 번으로 필요한 패키지와 이 저장소의 `lumi_synth` 코드가 모두 `.venv`에 설치됩니다.

## 단계별 실행

```powershell
python scripts\step0_check.py   # 0단계: 환경·원본 영상 점검
python scripts\step1_cut.py     # 1단계: 원본 5초 클립
pytest                          # 측정 함수 자동 테스트
```

결과는 `checks\stepN\README.md`에서 확인합니다. 계획은 [PLAN.md](PLAN.md)에 있습니다.
