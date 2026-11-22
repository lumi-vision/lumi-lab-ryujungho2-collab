# lumi-lab-ryujungho2-collab

LUMI 프로젝트 개인 실험 저장소입니다. 2026-11-08에 처음부터 다시 설계하기 위해 초기화했습니다.

- 데이터 통합 현황(로컬 보유·수량·분할): [docs/DATA_INVENTORY.md](docs/DATA_INVENTORY.md)
- Phase 1 데이터셋 관리: [실행 방법](docs/dataset_readme.md) · [스키마](docs/dataset_schema.md) · [기존 데이터 조사](docs/dataset_audit.md)
- 데이터 출처 상세: [DATA_SOURCES.md](DATA_SOURCES.md)
- 초기화 전 작업 전체: 브랜치 [`archive/2026-11-08-before-reset`](https://github.com/lumi-vision/lumi-lab-ryujungho2-collab/tree/archive/2026-11-08-before-reset) (태그 `before-reset-2026-11-08`)

- **현황 보고 (먼저 읽기)**: [docs/reports/STAGE_hazard_generators_2026-11-12.md](docs/reports/STAGE_hazard_generators_2026-11-12.md) — 위험 3종 생성기 단계 보고. 2026-11-13: 판정기 버그 6건 수정([docs/STANDARDS.md](docs/STANDARDS.md) §8 참고)
- 이전 현황 기록(2026-11-10 시점): [docs/STATUS_2026-11-10.md](docs/STATUS_2026-11-10.md)
- **작업 설계 (지금 진행 중)**: [PLAN.md](PLAN.md). 휘도·적색·줄무늬 위험 영상을 의도대로 합성할 수 있는지, 우리 판정기와 팀원 판별기로 전·후 검사합니다.
- **생성기 G 정리 (종류·위치·실행 논리)**: [docs/GENERATORS.md](docs/GENERATORS.md) · 영상만 보기: `LUMI_영상보기.bat`
- **생성 → 탐지 → 보정 단계별 검증 (2026-11-14~)**: [docs/GDR_VERIFY.md](docs/GDR_VERIFY.md) · **학습한 모델 목록(어떤 데이터로 어떻게 학습했나, sha256)**: [docs/MODELS.md](docs/MODELS.md) — `python scripts/model_registry.py` 로 갱신
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

## 두 가지 작업: GAN 생성 / 탐지·검사

Windows 실행 파일: 프로젝트 폴더의 **`LUMI.exe`**를 열어 생성/검사를 선택하거나 **`LUMI_생성기.exe`**, **`LUMI_플레이어.exe`**를 직접 실행한다. 서버가 꺼져 있으면 시작하고, 같은 프로젝트의 서버가 실행 중이면 재사용한다. 실패 사유와 로그는 `data/launcher/`에 남는다.

이 EXE는 **현재 프로젝트의 `.venv`, 코드, 모델을 사용하는 로컬 실행기**다. 다른 PC용 독립 배포판이 아니므로 EXE만 옮기지 않는다. 재빌드: `powershell -ExecutionPolicy Bypass -File scripts/windows/build_launcher.ps1`. 진단: `LUMI.exe --check` (결과 `data/launcher/check-8765.txt`). GPU 여유 메모리가 부족하면 GAN 추론은 CPU로 실행하고, 실제 장치는 레시피의 `inference_device`에 기록한다.

1. **`LUMI_생성기.bat`** → [GAN 생성 화면](http://127.0.0.1:8765/generate)
   - 원본 폴더 경로를 입력하고 **폴더 불러오기**를 누른다. 필요하면 하위 폴더도 포함한다.
   - 목록에서 원본을 하나 또는 여러 개 선택한다. 크기·fps·길이가 표시된다.
   - 체크포인트, 시작 시간, 구간 길이, 영상당 생성 수, 영역, 시드를 고른 뒤 생성한다.
   - 생성 단계에서는 탐지하지 않는다. `data/playlist/generated/`에 무손실 FFV1 **MKV**, 원본·합성본·마스크를 담은 **NPZ**, 완료 기록 `*.result.json`을 저장한다. 결과 목록은 재시작 후에도 남는다.
   - 현재는 지정 구간(최대 10초)을 합성한다. 긴 변 최대 768 축소 후 해상도와 원본 fps가 체크포인트 학습 규격과 일치해야 한다. fps 변경·강제 확대는 하지 않는다.
2. **`LUMI_플레이어.bat`** → [탐지·검사 플레이어](http://127.0.0.1:8765/)
   - 생성 결과의 **플레이어에서 검사** 링크로 바로 넘기거나, 파일·폴더 열기로 기존 결과를 불러온다.
   - 생성 쌍을 넘기면 원본과 합성본을 각각 검사한다. 비교 화면, 위험 종류·영역·시간선, 프레임 이동을 제공한다.
   - 검사 결과는 `data/playlist/cache/`에 저장한다. 생성 의도와 판정 결과는 별개로 표시한다.

두 화면은 같은 로컬 서버를 사용한다. 하나를 켠 상태에서 다른 실행 파일을 열면 해당 화면으로 이동한다. 기존 통합 생성 메뉴는 플레이어의 **고급 합성 실험…**에 남아 있다.

## 플레이어 상세 기능

- `LUMI_플레이어.bat` 을 더블클릭하면 `python scripts\make_playlist.py --serve` 를 실행하고 브라우저를 엽니다. 브라우저에서 파일이나 폴더를 열면 판정기를 돌려 영상 · 정답 · 우리 판정기 · 줄무늬 주파수 탐지기(쿼드트리+가버, 블록 테두리 표시) 네 화면과 3단 그래프(위험 시간선, 기준 대비 %, 1초 전이 수)를 보여 줍니다. '빠른 판정'(기본)은 느린 스캔선 줄무늬 판정을 건너뛰고 줄무늬를 주파수 탐지기로 판정합니다(768×432 2초 영상 약 18초).
- **생성해서 보기 (2026-11-14)**: 플레이어의 '생성…' 버튼을 누르면 조건을 고르는 줄이 열립니다. '생성'을 누르면 서버가 그 자리에서 위험 영상을 만들고 판정해서 목록에 넣습니다(768×432 2초 영상 하나에 5~15초).
  - 종류: GAN(`data/gan/*.pt` 체크포인트) · 규칙 휘도 깜빡임 · 규칙 적색 깜빡임 · 규칙 줄무늬(정지·반전·흐름) · 섞어서. GAN 은 체크포인트마다 종류가 하나입니다(휘도·적색·줄무늬, 목록에 `[종류]` 로 표시 — [docs/GAN.md](docs/GAN.md) §10). '규칙 …' 은 식으로 만드는 생성기입니다.
  - 목표: 위험 / 근처 안전(조건 하나만 기준 아래 — 휘도·적색은 초당 1~2.5회, 줄무늬는 면적 20~35%). **생성 의도는 정답이 아닙니다.** 생성 결과는 '의도 일치/불일치'로 표시하며, 둘째 화면에는 원본 위 실제 변경 영역을 보여 줍니다. 생성 의도로 위험 시간선을 채우지 않습니다.
  - 바탕(val·test·train 클립), 영역 모양·면적, 주파수, cpd, 움직임, 시드를 정할 수 있고 비우면 무작위입니다. 같은 시드는 같은 영상을 만듭니다. GAN 은 'z 만 바꾸기'로 같은 조건에서 얼마나 다르게 만드는지 볼 수 있습니다.
  - '원본 경로'를 지정하면 해당 파일의 시작 초·길이 초 구간에 합성합니다. 비우면 기존 바탕을 사용합니다. 일정 프레임률 SDR 영상 기준이며, 긴 변은 최대 768로 축소하고 fps 는 유지합니다. GAN 은 학습 체크포인트와 해상도·fps 가 일치해야 합니다(현재 체크포인트는 768×432, 24fps). 원본 자체가 위험할 수 있으므로 원본도 별도 판정하세요. Kaya 등 실제 위험 자료를 안전한 복원 정답으로 간주하지 않습니다.
  - 만든 쌍은 `data/playlist/generated/<id>.npz` 에 저장됩니다: `clean`(합성 전 원본), `flicker`(합성본), `mask_allow`(허용 영역), `mask_changed`(프레임별 실제 변경), `fps`, `intent_hazard`, `recipe`, `schema_version=2`. 원본을 포함하므로 파일 크기는 이전 형식보다 커집니다. `hazard_frames`·`hazard_mask`에 의도를 저장하지 않습니다. 코드: `src/lumi_synth/generate.py`.
  - 판정 결과는 `data/playlist/cache/<id>.json` 및 `<id>.judgement.npz`에 별도 저장됩니다. NPZ에는 판정 모드(`fast`), 판정 코드 해시, 위험 여부, 종류별 시간선과 마스크가 있습니다. 빠른 판정과 전체 판정은 줄무늬 알고리즘이 달라 구분해서 사용해야 합니다.
  - `−1 프레임 / +1 프레임`은 재생을 멈춘 상태로 이동합니다. 원본 복원용 쌍을 만드는 단계이며, PLAN의 최소 보정 B 정답 계산은 아직 구현하지 않았습니다. 점검 결과: [2026-11-14 생성 파이프라인](checks/gan/generator_pipeline_2026-11-14.md).
- 줄무늬 시연 영상: `python scripts\make_stripe_demo.py` → `data/stripe_demo/` 를 플레이어의 '폴더 열기'로 엽니다.
- 한꺼번에 만들 때는 `python scripts\make_playlist.py` 를 씁니다. 결과는 `data/playlist/` 에 생기며 Git 에 올리지 않습니다.
- npz · gif · mp4 · avi · mkv · mov 를 브라우저용 mp4 로 변환합니다.
- 주의: 광과민성 발작을 일으킬 수 있는 영상이 들어 있습니다. 이 PC 에서 로컬로만 쓰고 공유하지 않습니다.
