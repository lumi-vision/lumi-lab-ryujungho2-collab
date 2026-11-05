# 소스 코드 감사 및 재현성 점검표

이 문서는 '구현했다'와 '실험적으로 검증했다'를 분리하기 위한 증거 대장이다. 코드 점검 날짜: **2026-11-05**.

## 1. 현재 모듈과 주장 가능한 사실

| 경로 | 현재 동작 | 증거 | 한계/후속 작업 |
|---|---|---|---|
| `src/lumi/video_io.py` | OpenCV/PyAV 영상 입력·프레임 메타데이터 | 영상 입출력 테스트 | OpenCV VFR 타임스탬프는 FPS 추정 |
| `src/lumi/player.py` | 명시적으로 허용 시 수동 미리보기 | `test_preview_requires_opt_in` | PSE 보호 필터가 아니며 기본 재생 금지 |
| `src/lumi/analysis.py` | 프레임 평균/최소/최대 8-bit 회색조 밝기 | JSONL 스캔 테스트 | 위험 검사기나 선형 휘도 분석이 아님 |
| `src/lumi/dataset_synthesis.py` | Oracle 합성·변형 허용 마스크·시간 파형 생성 | 결정성, ROI 외 보존 테스트 | 위험 판정 라벨 미구현; 기본 16 @ 24fps |
| `src/lumi/gan_synthesis.py` | 조건부 3D CNN G/D, Hinge + ROI L1 | shape, 1 step 학습 테스트 | Barbu 위험 perceptual loss 미구현 |
| `src/lumi/dataset_audit.py` | 배열·마스크·원본 해시 기반 누출 확인 | 단일 소스 감사 테스트 | 표준 검사/장면 중복 탐지 아님 |
| `scripts/train_gan.py` | Oracle train에 대해 G/D 학습·체크포인트 저장 | 단일 step 테스트와 정적 점검 | 다중 epoch 수렴/실제 성능 미검증 |
| `scripts/generate_gan.py` | train 분할에서 G로 새 후보 생성 | 정적 코드 점검 | 실제 위험성 독립 검증 필요 |
| `src/lumi/lossless_video.py` | FFV1/MKV 저장 후 다시 읽어 바이트 일치 확인 | `test_lossless_video_roundtrip`, 데이터셋 내보내기 테스트 | IRIS의 OpenCV(vcpkg ffmpeg 7.0.2)와 같은 디코딩인지는 확인 시험 7번 한 사례로만 확인 |
| `src/lumi/iris_labeling.py` | IRIS 실행(`-v -j 1 -p 1`), `framedata.csv` 해석(NUL 제거), 요약·판정 문자열·창 라벨, 실패 시 `iris_error` | 가짜 실행기 테스트 6개, 실제 IRIS 확인 시험 8건 | 패턴 판정 불안정(IRIS 계획 5.1) |
| `scripts/label_iris.py`, `scripts/iris_selftest.py` | 데이터셋 전체 라벨링, 라벨러 확인 시험 | 확인 시험 6/8 기대 일치(불일치 2건은 패턴) | 라벨 분포 확인(계획 5단계) 전 |

## 2. 테스트 결과

집 GPU PC(RTX 5080, torch 2.15.0+cu132)에서 `pytest -q`를 실행해 **25 passed**를 확인했다(2026-11-05). CUDA가 없는 노트북에서는 CUDA bf16 시험 1개가 skip된다. 테스트는 코드 불변식 확인이며, 정확도·성능·의료 안전성에 대한 증거가 아니다.

## 3. 우선 수정할 위험 요소

- **라벨 의미:** `mask`는 `M_allow`다. 실제 변형 픽셀(`M_changed`) 및 위험 기여 마스크(`M_risk`)와 구별해야 한다.
- **시퀀스 길이:** 16프레임/24fps는 1초보다 짧다. 전체 1초 이벤트 검사와 동시에 사용하면 안 된다.
- **원본 표본:** 파일 해시만으로 같은 장면의 다른 인코딩을 식별하지 못한다.
- **샘플 경계:** 짧은 영상에서 프레임을 순환하며 구성한 클립은 자연스러운 시간 연속성이 없을 수 있다.
- **입력 색공간:** OpenCV 회색조 값은 실제 표시 휘도 측정이 아니다.
- **안전성 부재:** IRIS 판정은 인증·임상 판정이 아니다. `risk_verified=true`는 IRIS로 검사했다는 뜻일 뿐이다. 인간 전문가 검수는 없다.
- **GAN 수렴:** 샘플링만 가능한 모델이나 1 step 동작 확인이 '잘 학습된 생성 모델'임을 의미하지 않는다.

## 4. 완료 판단 기준

- [x] 기본 영상 입출력과 프레임 로그
- [x] Oracle의 공간·시간 변형 후보 생성
- [x] 소스 단위 해시 및 train/val/test 분할(기초)
- [x] 조건부 GAN G/D 학습 스크립트
- [x] 외부 ROI 픽셀 보존에 대한 자동 테스트
- [x] 독립 검사기(IRIS) 실행·버전(커밋·설정 해시) 기록 — 휘도·적색만 신뢰, 패턴 미신뢰
- [ ] 공인 표준 적합성 검사와 다중 표준 버전 관리
- [ ] 장면 단위 중복 그룹화 및 PTS 정확도 보존
- [x] IRIS 판정용 6초 클립·무손실 영상 (`--iris-clips`)
- [ ] 학습 창(16프레임) 라벨 규칙 결정
- [ ] GAN 학습 수렴 및 독립 데이터 비교
- [ ] 보정 Teacher 및 보정 전후 검증
- [ ] 스트리밍 종단 지연 측정

## 5. 협업 저장소 이관 기준

개인 실험 저장소에서는 가설과 비교 실험 코드를 자유롭게 수정한다. 팀 저장소 `lumi-vision/lumi`로 병합할 때에는 (i) 인터페이스·데이터 버전, (ii) 재현 절차, (iii) 데이터/라이선스 감사, (iv) 실패 시 처리, (v) 테스트 결과, (vi) 논문 인용과 실제 구현의 대응표를 첨부한 Pull Request를 사용한다.
