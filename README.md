# lumi-lab-ryujungho2-collab

LUMI 프로젝트 개인 실험 저장소입니다. 2026-11-08에 처음부터 다시 설계하기 위해 초기화했습니다.

- 데이터 출처: [DATA_SOURCES.md](DATA_SOURCES.md)
- 초기화 전 작업 전체: 브랜치 [`archive/2026-11-08-before-reset`](https://github.com/lumi-vision/lumi-lab-ryujungho2-collab/tree/archive/2026-11-08-before-reset) (태그 `before-reset-2026-11-08`)

- **현황 보고 (먼저 읽기)**: [docs/reports/STAGE_hazard_generators_2026-11-12.md](docs/reports/STAGE_hazard_generators_2026-11-12.md) — 위험 3종 생성기 단계 보고. 2026-11-13: 판정기 버그 6건 수정([docs/STANDARDS.md](docs/STANDARDS.md) §8 참고)
- 이전 현황 기록(2026-11-10 시점): [docs/STATUS_2026-11-10.md](docs/STATUS_2026-11-10.md)
- **작업 설계 (지금 진행 중)**: [PLAN.md](PLAN.md). 휘도·적색·줄무늬 위험 영상을 의도대로 합성할 수 있는지, 우리 판정기와 팀원 판별기로 전·후 검사합니다.
- 판정 기준: [docs/CRITERIA.md](docs/CRITERIA.md) · 계산법: [docs/BASICS.md](docs/BASICS.md)
- **판정 기준 근거 보고서 (수식·출처 등급)**: [docs/STANDARDS.md](docs/STANDARDS.md)
- **패턴 정의 문헌 조사 (줄무늬 기준의 근거·계보·도구 비교)**: [docs/PATTERN_LITERATURE.md](docs/PATTERN_LITERATURE.md) · 조사 노트: `docs/research/pattern_notes/`
- **실패와 정정 기록**: [docs/LESSONS.md](docs/LESSONS.md) (논문·발표용, 출처 등급 정의 포함)

### 2026-11-09 추가: 규칙 명세 · 오라클 · GAN

- **전체 흐름(먼저 읽기)**: [docs/WORKFLOW.md](docs/WORKFLOW.md)
- **연구 흐름(논문 순서) 정리**: [docs/RESEARCH_NARRATIVE.md](docs/RESEARCH_NARRATIVE.md)
- **환경·해상도 차이 문제와 판정 해상도 제안**: [docs/RESOLUTION_ENV_REPORT.md](docs/RESOLUTION_ENV_REPORT.md)
- 판정 규칙 수식 명세와 설계 결정 D1~D4: [docs/RULES_SPEC.md](docs/RULES_SPEC.md)
- **2026-11-09 변경 기록(무엇·어떻게·왜·결과)**: [docs/CHANGELOG_2026-11-09.md](docs/CHANGELOG_2026-11-09.md)
- 집 PC 에서 할 일 (⏸ 보류, PLAN 먼저): [docs/HOME_NEXT_STEPS.md](docs/HOME_NEXT_STEPS.md)
- 오라클(규칙 기반 합성): [docs/ORACLE.md](docs/ORACLE.md) · GAN: [docs/GAN.md](docs/GAN.md) · 학습 데이터 계획: [docs/DATA_PLAN.md](docs/DATA_PLAN.md)
- 실측 결과: `checks/rules/`, `checks/oracle/`, `checks/gan/`, `checks/red/`, `checks/pattern/`, `checks/extended/`, `checks/hazard_sweep/`, `checks/emeans/`, `checks/demo_hazards/`, `checks/stripe_freq/`(줄무늬 주파수 탐지기 FFT·DCT·E-Means 대 스캔선, 11-13)

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
python scripts\step1_cut.py     # 1단계: 원본 5초 클립 (초기 작업, 결과는 옛 것)
pytest                          # 측정 함수 자동 테스트
```

결과는 `checks\stepN\README.md`에서 확인합니다. 계획은 [PLAN.md](PLAN.md)에 있습니다.

## 영상 확인 (플레이어)

- `LUMI_플레이어.bat` 을 더블클릭하면 `python scripts\make_playlist.py --serve` 를 실행하고 브라우저를 엽니다. 브라우저에서 파일이나 폴더를 열면 판정기를 돌려 영상 · 정답 · 우리 판정기 세 화면과 3단 그래프(위험 시간선, 기준 대비 %, 1초 전이 수)를 보여 줍니다.
- 한꺼번에 만들 때는 `python scripts\make_playlist.py` 를 씁니다. 결과는 `data/playlist/` 에 생기며 Git 에 올리지 않습니다.
- npz · gif · mp4 · avi · mkv · mov 를 브라우저용 mp4 로 변환합니다.
- 주의: 광과민성 발작을 일으킬 수 있는 영상이 들어 있습니다. 이 PC 에서 로컬로만 쓰고 공유하지 않습니다.
