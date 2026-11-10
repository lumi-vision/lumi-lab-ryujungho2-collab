> ⏸ **보류 (2026-11-09)**: [PLAN.md](../PLAN.md) 개정 3판(합성 능력 검증: 휘도·적색·줄무늬, 팀원 판별기로 전·후 검사)을 먼저 끝낸 뒤 이 문서의 GAN·데이터셋 단계를 다시 검토합니다.

# 집 PC(RTX 5080)에서 할 일 — 2026-11-09 노트북 작업 이후

## ★ 먼저: 위험 3종 생성 스윕 평가 (2026-11-10 노트북에서 코드만 완성, 실행은 집에서)

노트북에서 한 것: 생성기 확장(`oracle.py` 의 `render_flicker`·`render_flat_flash`·`red_pair`·`render_red`·`render_stripes`),
스윕 생성·평가 스크립트, 생성기 테스트 4개. 회색 바탕 106개는 노트북에서 만들었지만 BBB 1080p 원본이 노트북에 없어
받다가 메모리 부족으로 멈춤 → 평가·그래프는 아직 없음.

```powershell
git pull --ff-only
pip install -r requirements.txt                       # av(PyAV) 포함
python scripts\step0_check.py                         # BBB 1080p 가 없으면 받고 SHA256 확인 (data/sources/)
python scripts\make_hazard_sweep.py                   # 회색·BBB 바탕 × 106개 = 212개 (data/hazard_sweep/, 재생 금지)
python scripts\eval_hazard_sweep.py --only team --team-dir <lumi-lab-rockrovks 경로>   # 팀원 판별기, 수정 없이
python scripts\eval_hazard_sweep.py --only ours       # 우리 판정기 (클립당 수십 초, 중단돼도 이어서 실행됨)
python scripts\eval_hazard_sweep.py --only report --team-dir <lumi-lab-rockrovks 경로>
```

- 결과: `checks/hazard_sweep/README.md`, `results.csv`, 그래프 `fig_sweep_{lum,red,stripe}.png`·`fig_agreement.png`·
  `fig_measured_vs_design.png`·`fig_timeseries.png` (수치 그래프만, 영상 프레임 그림 없음)
- 팀원 판별기 버전은 실행 전에 `git -C <경로> log --oneline -1` 로 기록(노트북 확인 시 `e29ba9b`)
- **합격**: 우리 판정기가 회색 바탕에서 생성 의도와 계열별로 일치. 어긋나면 원인(생성기 / 판정기 / 기대값 계산)을 README 에 기록
- 다음: 결과를 발표자료(07 다음 장 '생성 3종')에 추가
- 다루지 않은 줄무늬 정의: 쌍 수(5 또는 6, 면적과 얽힘), 체커보드·동심원, 흐름 줄무늬의 휘도 제외(결정 A)

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

**합격**: 40 passed (노트북 결과와 같음, 2026-11-09 해상도 통일 커밋 기준). 하나라도 실패하면 멈추고 실패 내용을 기록.

## 2. 판정 규칙 외부 검증 결과 확인

노트북에서 실행한 결과: `checks/rules/README.md` (이 문서 §8 에 요약, 201/210 일치·놓침 0). 집 PC 에서는 다시 돌릴 필요 없음.
불일치 9개(과잉 검출)는 유형 A 4·B 4·C 1 이다(§8). C 는 기대값 오류 가능성이라 규칙 문제가 아니다. 규칙을 고칠지는 **결정 필요**: 고치면 모든 오라클·GAN 라벨이 바뀌므로, 고친 판정기로 이 검증(`python scripts/validate_rules_pse.py --pse-root <경로> --sets all-luminance`, CPU 약 1.5시간)과 `pytest` 를 다시 돌린 뒤 데이터셋을 다시 만든다.
- 유형 A(계단식 휘도): 픽셀 전이를 '누적 변화'가 아니라 다른 정의로 바꿀지 — ITU 원문·pse-test-media 의 y/l 패턴 설계 의도 확인 먼저.
- 유형 B(간섭): 화면 전이를 '같은 영역의 반대 방향 변화'끼리만 짝짓도록 바꿀지(D1 의 한계 해소).
- 유형 C: pse-test-media 에 기대값(JSON)과 목록 파일이 다른 이유 확인(이슈 등록 고려).

## 3. 해상도 원칙: 판정 = 학습 = 768×432 (docs/DATA_PLAN.md §8)

2026-11-09 에 두 트랙(128×72 / 1920×1080) 계획을 접고 한 해상도로 통일했다. 128×72 결과는 깜빡임 전용 예비 실험.

## 4. 시험 분할 원본 정하기 (먼저)

Elephants Dream(480×270)은 768×432 보다 작아 쓸 수 없다(확대 금지 → 스크립트가 멈춤). 학습 영화(BBB)와 다른 영화의 768×432 이상 영상을 하나 받아 `DATA_SOURCES.md` 에 출처·라이선스·SHA256 을 적는다. 후보: Sintel 1080p, Elephants Dream HD.

## 5. 오라클 데이터셋 (768×432)

```powershell
python scripts/step0_check.py                       # BBB 1080p 가 없으면 받음
python scripts/make_oracle_dataset.py --test <data/ 기준 경로>:<시작초>:<끝초>
```

**확인**: `checks/oracle/768x432_24fps.md` 의 의도-판정 일치율(128×72 예비 실험 351/352 와 비교), 클립 하나 용량·생성 시간 기록.

## 6. GAN (768×432, GPU)

```powershell
# 원인 분리(시드 반복): λ_div 0.05 / 0.5 × 시드 0·1·2 — 메모리가 넘치면 --batch-size 2
foreach ($s in 0,1,2) { foreach ($d in 0.05,0.5) {
  python scripts/train_gan.py --device cuda --steps 500 --batch-size 4 --lambda-div $d --seed $s `
    --checkpoint data/gan/u_div${d}_s${s}.pt --log checks/gan/train_log_u_div${d}_s${s}.jsonl
  python scripts/eval_gan.py --checkpoint data/gan/u_div${d}_s${s}.pt --out data/gan/u_div${d}_s${s} --report checks/gan/u_div${d}_s${s}.md
} }
# 본 학습 → 학습용 GAN 샘플
python scripts/train_gan.py --device cuda --steps 4000 --batch-size 8 --width 32 --lambda-div <값> --seed 0 `
  --checkpoint data/gan/u_full_s0.pt --log checks/gan/train_log_u_full_s0.jsonl
python scripts/make_gan_dataset.py --checkpoint data/gan/u_full_s0.pt --z-per-clip 2
```

**확인**: 첫 실행에서 GPU 메모리·스텝당 시간 기록(미측정). 목표 위험 → 위험 ≥ 95%, 근처 → 안전 & 1~3회/초 ≥ 90%, 마스크 밖 0, z 주파수 std 가 시드마다 같은 방향인지.

## 7. 아직 설계도 없는 것 (집에서 결정할 것)

| 항목 | 필요한 결정 |
|---|---|
| 탐지 모델(H1: 오라클만 vs 오라클+GAN) | 모델 구조(프레임 단위 vs 시공간), 입력 해상도, 출력(위험 여부 / hazard_mask / hazard_frames), 같은 샘플 수 비교 방법, 평가 지표 |
| 실제 영상 시험셋 | 합성하지 않은 시험 영상의 출처(예: South 2021 위험 GIF 163개, 재배포 불가·시험 전용)와 라벨 기준 |
| 줄무늬 판정 속도 | 768×432·48프레임 판정 시간 측정, 느리면 구간 표시 부분 최적화 또는 GPU 이전 |
| fps | 24 유지 또는 30(PLAN.md) — 해상도 통일 결정에는 포함되지 않음 |
| 감마 가정(A1) | `checks/rules/gamma_sensitivity.md` §3 이 70/210 에서 중단됨(노트북 메모리 부족). `python scripts/gamma_sensitivity.py --pse bt1886 -- --pse-root <경로> --sets all-luminance` 로 210개 완료 후 sRGB 유지 여부 결정 |
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
