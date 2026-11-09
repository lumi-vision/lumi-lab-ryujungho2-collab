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

**합격**: 36 passed (노트북 결과와 같음, 2026-11-09 `0e56368` 기준). 하나라도 실패하면 멈추고 실패 내용을 기록.

## 2. 판정 규칙 외부 검증 결과 확인

노트북에서 실행한 결과: `checks/rules/README.md` (이 문서 §8 에 요약, 201/210 일치·놓침 0). 집 PC 에서는 다시 돌릴 필요 없음.
불일치 9개(과잉 검출)는 유형 A 4·B 4·C 1 이다(§8). C 는 기대값 오류 가능성이라 규칙 문제가 아니다. 규칙을 고칠지는 **결정 필요**: 고치면 모든 오라클·GAN 라벨이 바뀌므로, 고친 판정기로 이 검증(`python scripts/validate_rules_pse.py --pse-root <경로> --sets all-luminance`, CPU 약 1.5시간)과 `pytest` 를 다시 돌린 뒤 데이터셋을 다시 만든다.
- 유형 A(계단식 휘도): 픽셀 전이를 '누적 변화'가 아니라 다른 정의로 바꿀지 — ITU 원문·pse-test-media 의 y/l 패턴 설계 의도 확인 먼저.
- 유형 B(간섭): 화면 전이를 '같은 영역의 반대 방향 변화'끼리만 짝짓도록 바꿀지(D1 의 한계 해소).
- 유형 C: pse-test-media 에 기대값(JSON)과 목록 파일이 다른 이유 확인(이슈 등록 고려).

## 3. 두 트랙 원칙 (docs/DATA_PLAN.md §7)

노트북 = 저해상도 트랙 `128x72_24fps`, 집 PC = 고해상도 트랙 `1920x1080_30fps`. 같은 코드를 인자만 바꿔 돌리고, **산출물을 섞지 않는다**(폴더·보고서·체크포인트에 트랙 태그, 다르면 스크립트가 멈춤). 합칠지는 §7 결과로 정한다.

## 4. 저해상도 트랙을 GPU 로 마무리 (`128x72_24fps`)

노트북은 CPU 라 GAN 을 64×36(배율 0.5)으로만 학습했고, 그래서 학습용 GAN 샘플을 만들 수 없었다(해상도 불일치로 삭제). 집 PC 에서 **같은 트랙을 배율 1.0** 으로 학습한다.

```powershell
python scripts/make_oracle_dataset.py                                  # data/oracle_128x72_24fps (노트북과 같은 시드 → 같은 결과여야 함)
# 원인 분리(시드 반복): λ_div 0.05 / 0.5 × 시드 0·1·2
foreach ($s in 0,1,2) { foreach ($d in 0.05,0.5) {
  python scripts/train_gan.py --device cuda --steps 500 --batch-size 4 --scale 1.0 --lambda-div $d --seed $s `
    --checkpoint data/gan/l_div${d}_s${s}.pt --log checks/gan/train_log_l_div${d}_s${s}.jsonl
  python scripts/eval_gan.py --checkpoint data/gan/l_div${d}_s${s}.pt --out data/gan/l_div${d}_s${s} --report checks/gan/l_div${d}_s${s}.md
} }
# 본 학습 (λ_div 는 위 결과로 정함) → 학습용 GAN 샘플
python scripts/train_gan.py --device cuda --steps 4000 --batch-size 8 --width 32 --scale 1.0 --lambda-div <값> --seed 0 `
  --checkpoint data/gan/l_full_s0.pt --log checks/gan/train_log_l_full_s0.jsonl
python scripts/eval_gan.py --checkpoint data/gan/l_full_s0.pt --out data/gan/l_full_s0 --report checks/gan/l_full_s0.md
python scripts/make_gan_dataset.py --checkpoint data/gan/l_full_s0.pt --z-per-clip 2   # → data/gan_train_128x72_24fps
```

**확인**: `checks/oracle/128x72_24fps.md` 가 노트북 결과(의도-판정 351/352)와 같은지. GAN: 목표 위험 → 위험 ≥ 95%, 근처 → 안전 & 1~3회/초 ≥ 90%, 마스크 밖 0, z 주파수 std 가 시드마다 같은 방향인지(잠정 기준). 배치 8 이 16 GB 를 넘으면 4.

## 5. 고해상도 트랙 오라클 (`1920x1080_30fps`, PLAN.md 기준)

```powershell
python scripts/make_oracle_dataset.py --width 1920 --height 1080 --fps 30 --frames 150 `
  --train big_buck_bunny_1080p_h264.mov:30:420 --val big_buck_bunny_1080p_h264.mov:450:560 `
  --test Sintel.2010.1080p.mkv:<시작>:<끝> --raw <영상 폴더>
```

- 시험 분할 원본도 1080p 여야 한다(Elephants Dream 은 480p 라 키우지 않는다). 후보: Sintel 1080p(24 fps → 30 fps 복제). 영화 단위 분할 유지.
- 1080p·150프레임 클립 하나가 압축 전 약 0.9 GB → npz 저장 용량·시간을 먼저 2~3개로 시험. 필요하면 저장을 FFV1(.mkv)+PNG+CSV 로 바꿔야 함(미구현).
- 판정 규칙 계산 메모리: 휘도 float32 150×1920×1080 ≈ 1.2 GB (64 GB RAM 에서 가능, 미측정).

## 6. 고해상도 트랙 GAN (설계 결정 필요)

1080p·150프레임을 3D 합성곱으로 직접 학습하는 것은 16 GB 에서 어려울 것으로 예상(미측정). 둘 중 하나를 고른다(둘 다 미구현):
- (a) 중간 해상도 트랙(예: `480x270_30fps`)을 하나 더 만들어 오라클·GAN 을 같은 방식으로 돌린다 — 코드 수정 없이 인자만으로 가능.
- (b) 작업 해상도에서 만든 빛 Δ 를 1080p 로 키워 1080p 바탕에 더하고, 1080p 에서 판정 — `gan.py` 에 확대·합성 단계 추가 필요.

## 7. 트랙 합치기와 아직 설계도 없는 것 (집에서 결정할 것)

| 항목 | 필요한 결정 |
|---|---|
| 탐지 모델(H1: 오라클만 vs 오라클+GAN) | 모델 구조(프레임 단위 vs 시공간), 입력 해상도, 출력(위험 여부 / hazard_mask / hazard_frames), 같은 샘플 수 비교 방법, 평가 지표 |
| 실제 영상 시험셋 | 합성하지 않은 시험 영상의 출처(예: South 2021 위험 GIF 163개, 재배포 불가·시험 전용)와 라벨 기준 |
| 판정 해상도 통일 | `docs/RESOLUTION_ENV_REPORT.md` §4 제안(긴 변 720 통일, 128×72 는 깜빡임 전용, 고해상도 GAN 은 전체 판정 후 조각 학습) 채택 여부 |
| 감마 가정(A1) | `checks/rules/gamma_sensitivity.md` §3 이 70/210 에서 중단됨(노트북 메모리 부족). `python scripts/gamma_sensitivity.py --pse bt1886 -- --pse-root <경로> --sets all-luminance` 로 210개 완료 후 sRGB 유지 여부 결정 |
| 트랙 합치기 | DATA_PLAN §7 기준: 같은 조건을 1080p 로 합성한 것과 128×72 로 줄인 것의 판정 일치율(해상도 민감도 스크립트 미구현), 두 트랙의 경향 비교 |
| 1단계(`step1_cut.py`) 재실행 | 규칙 변경으로 `events` 의미가 바뀜(화면 전이). PLAN 의 1080p·30fps 기준으로 다시 할지 |
| 실제 영상과의 분포 거리 | 자연스러움의 코드 지표 추가 여부(예: 프레임 특징 분포 비교) |

## 8. 노트북에서 받은 최신 결과 요약 (채워 넣는 칸)

- 규칙 외부 검증(`checks/rules/README.md`, pse-test-media `ef78542`, 1920×1080 원본, 휘도 전용 세트 11개·210개 영상, CPU 5,211초):
  ITU-R BT.1702 기대값과 **201/210 일치**(기본 2세트 53/56). **기대 fail 16개는 모두 잡음(놓침 0)**, 불일치 9개는 모두 '기대 pass → 우리 fail'(과잉 검출).
  - **주의: 201/210 은 과대평가다.** 210개 중 142개는 우리 판정기가 화면 전이를 한 번도 세지 않은 영상(면적이 25%보다 훨씬 작거나 밝기 차이 < 20)이라 '안전' 일치가 쉽다. 기대 fail 이 있는 세트는 3개뿐(alternating_01, broadcast_01, broadcast_inf01).
  - 유형 A (4개: broadcast_01 `f001y033`·`f005y036`·`f006l034`, inf01 `f003y038_f005cs035`): 한 번의 밝기 변화를 20 cd/m² 미만 두 단계로 나눈 패턴(예: 87.9→68.4→49.2, 19.5 + 19.1 = 38.7). pse-test-media 는 '밝기 미달'로 설계했지만 우리 픽셀 전이는 직전 극값 대비 누적(38.7)을 센다. 기본 2세트 불일치 3개가 모두 이 원인.
  - 유형 B (4개: inf01 `f002c015_f001cs012`, inf02 `f002tc03_f004ics03`·`f006tc03_f001ico03`·`f007ty03_f006ics03`): 두 영역이 서로 다른 시점에 깜빡임. 각 영역만 따로 재면 3.0회/초(안전)인데, 화면 상태 기계(D1)가 서로 다른 영역의 변화를 한 화면 깜빡임으로 이어 4~6회/초가 됨.
  - 유형 C (1개: inf02 `f001tf03_f003ico03`): JSON 기대값은 pass 인데 같은 세트의 목록 파일은 FAIL, 이 패턴만 따로 재면 3.5회/초. **기대값 쪽이 틀렸을 가능성** — 우리 판정이 설계 의도와 맞음.
  - 건너뛴 세트 7개: 빨강 깜빡임이 들어 있어 휘도 전용 판정기로는 원리상 판정 불가.
  - `itu_r1702_4` 의 `_4` 가 BT.1702-3 인지 이후 개정판인지는 pse-test-media 에서 확인되지 않음.
- 원인 분리 실험(v1 판정기 + λ_div 0.05, 64×36): 조절력 100%/100%, 그러나 z 주파수 std **0.06 Hz**(v1+λ0.5 는 1.92) → 다양성 회복은 λ_div 덕분, 판정기 수정은 조절력을 올림. 시드 1개라 §4 에서 시드 반복으로 확인(`docs/GAN.md` §9)
- 저해상도 트랙 GAN 학습용 샘플: 64×36 이라 삭제(`checks/gan/make_gan_dataset.md`) → §4 에서 다시 만듦

## 9. 기록 규칙

- 결과마다 `checks/...md` 를 남기고, 바꾼 것은 `docs/CHANGELOG_<날짜>.md` 에 '무엇·어떻게·왜·결과' 로 적는다.
- `data/` 는 올리지 않는다. 커밋 전에 `git status` 로 `.npz/.pt/.mkv` 가 없는지 확인한다.
- 합성 결과는 재생·미리보기하지 않는다(광과민성 위험 자극). 평가는 코드 지표로만.
