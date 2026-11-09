# 집 PC(RTX 5080)에서 할 일 — 2026-11-09 노트북 작업 이후

> 노트북에서 한 일과 결과: [CHANGELOG_2026-11-09.md](CHANGELOG_2026-11-09.md). 전체 흐름: [WORKFLOW.md](WORKFLOW.md).
> 순서대로 한다. 각 단계의 **합격 기준**을 못 넘으면 다음 단계로 가지 않고 원인을 기록한다.

## 0. 준비 (한 번)

```powershell
cd C:\dev\lumi-vision\lumi-lab-ryujungho2-collab
git pull --ff-only
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu132     # 이미 CUDA 빌드면 생략
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

원본 영상 (`DATA_SOURCES.md` 의 SHA256 와 대조):
- `data/raw/BigBuckBunny_640x360.m4v` (zip 을 받아 풀기), `data/raw/elephantsdream-480-h264-st-aac.mov`
- 집 PC 의 기존 `data/` 폴더에 이미 있으면 그대로 쓴다.

**합격**: `torch.cuda.is_available()` 가 `True`, SHA256 일치.

## 1. 테스트

```powershell
pytest
```

**합격**: 34 passed (노트북 결과와 같음). 하나라도 실패하면 멈추고 실패 내용을 기록.

## 2. 판정 규칙 외부 검증 결과 확인

노트북에서 실행한 결과: `checks/rules/README.md` (이 문서 §8 에 요약). 집 PC 에서는 다시 돌릴 필요 없음.
불일치가 있었다면 §8 의 원인별 조치를 먼저 한다.

## 3. GAN 원인 분리 실험 (짧게, 64×36)

노트북 v0 → v1 에서 두 가지(소프트 판정기, λ_div)를 함께 바꿔 원인을 못 나눴다. 노트북에서 'v1 판정기 + λ_div 0.05' 를 돌렸다(결과: `checks/gan/v1_div005.md`, 이 문서 §8). 집 PC 에서는 **시드 반복**으로 신뢰도를 확인한다.

```powershell
foreach ($s in 1,2) {
  foreach ($d in 0.05,0.5) {
    python scripts/train_gan.py --device cuda --steps 500 --batch-size 4 --scale 0.5 --lambda-div $d --seed $s `
      --checkpoint data/gan/abl_div${d}_s${s}.pt --log checks/gan/train_log_abl_div${d}_s${s}.jsonl
    python scripts/eval_gan.py --checkpoint data/gan/abl_div${d}_s${s}.pt --out data/gan/abl_div${d}_s${s} `
      --report checks/gan/abl_div${d}_s${s}.md
  }
}
```

**보는 것**: 'z 만 바꿨을 때 주 주파수 std' 와 '목표 위험 → 실제 위험' 이 시드 0·1·2 에서 같은 방향으로 움직이는지. 판정기 v0 은 코드에서 지웠으므로 v0 과의 비교는 노트북 결과(`checks/gan/README.md`)만 쓴다(필요하면 커밋 `89daf42` 의 `gan.py` 로 재현).

## 4. GAN 본 학습 (128×72, GPU)

λ_div 는 3단계 결과로 정한다(조절력을 잃지 않으면서 다양성이 큰 값).

```powershell
python scripts/train_gan.py --device cuda --steps 4000 --batch-size 8 --width 32 --scale 1.0 --lambda-div <정한 값> `
  --seed 0 --checkpoint data/gan/full_s0.pt --log checks/gan/train_log_full_s0.jsonl
python scripts/eval_gan.py --checkpoint data/gan/full_s0.pt --out data/gan/full_s0 --report checks/gan/full_s0.md
```

시드 0·1·2 세 번. 걸린 시간·GPU 메모리를 보고서에 적는다(배치 8 이 16 GB 를 넘으면 4 로).

**합격(잠정)**: 목표 위험 → 위험 ≥ 95%, 근처 → 안전 & 1~3회/초 ≥ 90%, 마스크 밖 0, z 주파수 std 가 64×36 결과보다 작아지지 않음. 못 넘으면 원인 기록 후 λ 조정.

## 5. 학습용 GAN 샘플 (128×72)

```powershell
python scripts/make_gan_dataset.py --checkpoint data/gan/full_s0.pt --z-per-clip 2
```

노트북에서 만든 `data/gan_train` 은 **64×36** 이라 오라클 샘플(128×72)과 해상도가 다르다 → 집 PC 의 128×72 모델로 다시 만든다.

**합격**: `checks/gan/make_gan_dataset.md` 의 목표별 라벨 분포 기록. split 이 모두 train.

## 6. 바탕 다양화 (코드 수정 필요)

`scripts/make_oracle_dataset.py` 의 `SOURCES` 는 지금 BBB(train/val)·ED(test) 뿐이다. Sintel·Tears of Steel 을 추가하려면 `SOURCES` 를 '분할 → 여러 (파일, 구간)' 목록으로 바꿔야 한다(미구현). 영화 단위 분할 유지: 예) train = BBB + Sintel, val = ToS 일부, test = ED.

**주의**: Sintel 1080p 는 24 fps, ToS 720p 는 24 fps 인지 먼저 확인(`scan_clean_clips` 가 fps 차이 > 1 이면 거부).

## 7. 아직 설계도 없는 것 (집에서 결정할 것)

| 항목 | 필요한 결정 |
|---|---|
| 탐지 모델(H1: 오라클만 vs 오라클+GAN) | 모델 구조(프레임 단위 vs 시공간), 입력 해상도, 출력(위험 여부 / hazard_mask / hazard_frames), 같은 샘플 수 비교 방법, 평가 지표 |
| 실제 영상 시험셋 | 합성하지 않은 시험 영상의 출처(예: South 2021 위험 GIF 163개, 재배포 불가·시험 전용)와 라벨 기준 |
| 1단계(`step1_cut.py`) 재실행 | 규칙 변경으로 `events` 의미가 바뀜(화면 전이). PLAN 의 1080p·30fps 기준으로 다시 할지 |
| 실제 영상과의 분포 거리 | 자연스러움의 코드 지표 추가 여부(예: 프레임 특징 분포 비교) |

## 8. 노트북에서 받은 최신 결과 요약 (채워 넣는 칸)

- 규칙 외부 검증: (작성 예정)
- 원인 분리 실험(v1 판정기 + λ_div 0.05): (작성 예정)

## 9. 기록 규칙

- 결과마다 `checks/...md` 를 남기고, 바꾼 것은 `docs/CHANGELOG_<날짜>.md` 에 '무엇·어떻게·왜·결과' 로 적는다.
- `data/` 는 올리지 않는다. 커밋 전에 `git status` 로 `.npz/.pt/.mkv` 가 없는지 확인한다.
- 합성 결과는 재생·미리보기하지 않는다(광과민성 위험 자극). 평가는 코드 지표로만.
