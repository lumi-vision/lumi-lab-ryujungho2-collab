# 협업 안내 — 노트북 · 집 GPU PC · 팀원

이 저장소(`lumi-vision/lumi-lab-ryujungho2-collab`)의 담당은 **GAN·학습 데이터 합성·IRIS 라벨링**이다. 위험 탐지 알고리즘은 팀원이 맡는다. 검증이 끝난 기능은 나중에 팀 통합 저장소 `lumi-vision/lumi`로 한 번에 옮긴다.

## 1. 무엇을 git으로 공유하고, 무엇을 공유하지 않나

| 공유한다 (git) | 공유하지 않는다 (git 밖) | 이유 |
|---|---|---|
| `src/`, `scripts/`, `tests/` | 생성 데이터 `datasets/`, `*.npz`, `*.mkv` | 광과민성 유해 자극이 들어 있을 수 있음, 용량 |
| `docs/` 연구 문서·실험 기록(수치는 텍스트로) | 체크포인트 `*.pt` | 용량. 같은 시드와 명령으로 다시 만든다 |
| `pyproject.toml` (의존성은 여기 한 곳) | venv, 캐시 | 장비마다 다름 |
| `setup_windows.ps1` | IRIS 소스·실행 파일 | 외부 프로젝트(BSD-3). 커밋 해시와 설정 해시만 기록한다 |
| — | IRIS 시험 자극 `iris_checks/` | 유해 자극 |

**규칙: 저장소 폴더에는 커밋할 파일만 둔다.** 무시 대상 파일도 저장소 안에 만들지 않는다. 그래서 작업 폴더를 다음과 같이 나눈다(GitHub 조직 이름과 맞춤).

```
C:\dev\lumi-vision\
├─ lumi-lab-ryujungho2-collab\   ← 이 저장소 (커밋 대상만)
├─ lumi\                         ← 나중에 팀 통합 저장소
├─ _work\                        ← .venv, datasets, sample, iris_checks, checkpoints
└─ _ext\                         ← IRIS 소스·빌드 (GPU PC만)
```

`git status --short --ignored`에 `!!`로 시작하는 줄이 보이면 그 파일을 `_work`로 옮긴다. OneDrive 같은 동기화 폴더 안에는 두지 않는다(venv·빌드 파일 수만 개가 업로드되고 파일 잠금으로 빌드가 깨질 수 있다).

## 2. 장비별 역할

| 장비 | 하는 일 | 못 하는 일 |
|---|---|---|
| 노트북 (CUDA 없음) | 코드·문서 수정, CPU 테스트(IRIS는 가짜 실행기로 시험) | IRIS 라벨링, GPU 학습 |
| 집 GPU PC (RTX 5080 16GB) | 데이터 생성, IRIS 라벨링, GAN 학습 | — |
| GitHub | 코드·문서·실험 기록 동기화 | 데이터·가중치 보관 |

데이터를 다른 장비로 옮겨야 하면 git이 아니라 외장 SSD를 쓴다. 기본은 **같은 커밋·같은 명령·같은 시드로 다시 생성**하는 것이다. 그래서 실험 기록에는 커밋 해시와 명령을 함께 남긴다.

## 3. 설치 (장비마다 1회)

준비물: Python 3.11 이상, Git. GPU PC는 최신 NVIDIA 드라이버도 필요하다.

```powershell
mkdir C:\dev\lumi-vision; cd C:\dev\lumi-vision
git clone https://github.com/lumi-vision/lumi-lab-ryujungho2-collab.git
cd lumi-lab-ryujungho2-collab
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows.ps1              # NVIDIA GPU가 있으면 cu132, 없으면 CPU 빌드 자동 선택
# .\setup_windows.ps1 -Torch cu130   # 드라이버가 CUDA 13.2를 지원하지 않을 때
```

스크립트는 `..\_work\.venv`를 만들고, `pyproject.toml`의 의존성을 설치한다. `src`는 `.pth` 파일로 등록하므로 저장소 안에 egg-info나 `__pycache__`가 생기지 않는다. 끝에서 환경 진단과 테스트를 실행한다. 이후 명령은 모두 저장소 폴더에서 아래처럼 실행한다.

```powershell
$py = "..\_work\.venv\Scripts\python.exe"
& $py -m pytest -q
```

**RTX 50 시리즈 주의:** Blackwell(sm_120) 커널은 CUDA 12.8 이상 빌드의 PyTorch에만 들어 있다. 기본 `pip install torch`는 Windows에서 CPU 빌드를 설치하므로 스크립트가 `--index-url`로 CUDA 빌드를 지정한다. 진단에서 `[OK] 아키텍처 커널`과 `[OK] Conv3d + bf16 autocast`가 나와야 한다. `nvidia-smi`의 "CUDA Version"은 드라이버가 지원하는 최대 버전이며, 휠 태그(cu132 = 13.2)보다 크거나 같아야 한다.

### IRIS (GPU PC만)

[`docs/IRIS_LABELING_PLAN.md`](IRIS_LABELING_PLAN.md) 7절 참고. VS 2026 BuildTools(C++), vcpkg, IRIS 커밋 `d96978a`로 `_ext\IRIS`에서 `cmake --preset windows-release -DBUILD_EXAMPLE_APP=ON`을 실행한다. 실행 파일 경로는 환경 변수로 넘긴다.

```powershell
$env:LUMI_IRIS_EXE = "C:\dev\lumi-vision\_ext\IRIS\bin\build\windows-release\example\IrisApp.exe"
& $py scripts\iris_selftest.py --output ..\_work\iris_checks    # 라벨러 확인 시험 (재생 금지)
```

## 4. 실행 예 (출력은 항상 `..\_work` 아래)

```powershell
& $py scripts\generate_demo.py --output ..\_work\sample\lumi_demo.avi

# 기존 16프레임 데이터 (GAN 학습용)
& $py scripts\build_dataset.py --source ..\_work\sample\lumi_demo.avi --output ..\_work\datasets\oracle --samples 200 --seed 42
& $py scripts\audit_dataset.py --dataset ..\_work\datasets\oracle

# IRIS 라벨용 6초 클립 + 무손실 영상 → IRIS 판정 기록 (GPU PC)
& $py scripts\build_dataset.py --source ..\_work\sample\lumi_demo.avi --output ..\_work\datasets\iris_v1 --samples 40 --iris-clips --seed 42
& $py scripts\label_iris.py --dataset ..\_work\datasets\iris_v1

# GPU 학습: CUDA에서는 bf16 autocast가 자동으로 켜진다
& $py scripts\train_gan.py --dataset ..\_work\datasets\oracle --device cuda --epochs 20 --batch-size 8 --workers 4 --width 32 --checkpoint ..\_work\checkpoints\lumi_cgan.pt
```

`--device cuda`는 CUDA가 없으면 CPU로 넘어가지 않고 바로 실패한다. 배치·폭 값은 출발점이지 측정된 최적값이 아니다.

## 5. 일상 동기화

```powershell
git pull --rebase                 # 작업 시작 전 (어느 장비든)
# ... 작업 ...
git status --short --ignored      # !! 줄이나 .npz/.pt/.mkv가 보이면 커밋하지 않는다
git add <파일>                     # git add -A 대신 파일을 지정한다
git commit -m "feat: ..."
git push
```

학습·라벨링 결과 수치(손실, 라벨 분포, 메모리, 소요 시간)는 커밋 해시·명령과 함께 `docs/`의 해당 문서에 텍스트로 남긴다.

## 6. 팀원에게 넘기는 것 — 데이터 규약

위험 탐지를 맡은 팀원은 이 저장소의 **코드가 아니라 데이터 규약**을 쓴다. 규약은 [`docs/DATASET_PROTOCOL.md`](DATASET_PROTOCOL.md) 3절이 기준이다. 바뀌면 그 문서를 먼저 고친다.

- `type`(normal/luminance/red/pattern)은 **합성 조건**이지 정답이 아니다. 예: 빨강 깜빡임은 IRIS에서 휘도 실패도 함께 받는다.
- 정답 후보는 `standard_result`와 `iris`(프레임별 결과)다. `risk_verified: true`는 "IRIS로 검사했다"는 뜻일 뿐, 위험하다거나 안전하다는 판정이 아니다.
- `standard_result == "iris_error"`인 샘플은 학습과 평가에서 뺀다(통과로 취급하지 않는다).
- IRIS **패턴 판정은 현재 학습 정답으로 쓰지 않는다**. 같은 줄무늬가 몇 픽셀만 움직여도 판정이 뒤집힌다([IRIS 계획 5.1절](IRIS_LABELING_PLAN.md)).
- 데이터 파일 자체는 git으로 주고받지 않는다. 커밋 해시와 생성 명령을 주고 각자 다시 생성하거나, 외장 저장장치로 전달한다.

## 7. 팀 저장소로 옮길 때

`lumi-vision/lumi`로는 기능 브랜치와 Pull Request로만 옮긴다. PR에는 다음을 붙인다: 인터페이스·데이터 규약 버전, 재현 명령과 커밋 해시, 데이터 출처·라이선스 감사, 실패 시 처리, 테스트 결과, 문헌과 구현의 대응표([`IMPLEMENTATION_AUDIT.md`](IMPLEMENTATION_AUDIT.md) 5절).
