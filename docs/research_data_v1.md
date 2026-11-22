# Research Data v1와 비교 실험 준비

2026-11-18. 데이터 기록은 `data/datasets/research_data_v1/sources_with_spring.jsonl`, BBB 전수 보고서는 같은 폴더의 `bbb_audit.json`이다. 과거 manifest는 변경하지 않았다. 원본 자료의 검증 상태는 모두 미확인이다. 작품별 역할은 개발용 임시 구분이며 독립 최종 시험셋을 의미하지 않는다.

| 작품 | 사양·출처 | 라이선스·이력 | 역할 |
|---|---|---|---|
| BBB | 1920×1080, H.264, 24fps, 약 596초, 725MB. [Blender](https://download.blender.org/peach/bigbuckbunny_movies/) | CC BY 3.0, 기존 G train/val | train |
| Tears of Steel | 1280×534, H.264, 24fps, 약 734초, 372MB. [Blender](https://download.blender.org/demo/movies/ToS/) | CC BY 3.0, 기존 초기 test 지정 | train |
| Elephants Dream | 480×270 H.264, 평균 약 23.39fps, 약 658초, 104MB. [Blender](https://orange.blender.org/press/) | CC BY 2.5, 기존 실험 이력 확인 필요 | val |
| Sintel | 1920×818 H.264 24fps, 약 888초, 1.18GB. [Blender](https://download.blender.org/durian/movies/) | CC BY 3.0, 기존 시험 사용 | unassigned |
| Spring | 2048×858 VP9 24fps, 약 464초, 81,781,205 bytes. [Commons](https://commons.wikimedia.org/wiki/File:Spring_-_Blender_Open_Movie.webm) | CC BY 4.0, © Blender Foundation | train |

Spring SHA-256 `d691a199035cc7d295210b286f8f6734893c7d4358d228081af6f0da98a56343`, 디코드 11,138프레임, PTS 단조 검사 통과. Commons에서 확인한 [Cosmos CC BY-SA 3.0](https://commons.wikimedia.org/wiki/File:Cosmos_Laundromat_-_First_Cycle_-_Official_Blender_Foundation_release.webm) 및 [Elephants Dream 1080p24 CC BY 2.5](https://commons.wikimedia.org/wiki/File:Elephants_Dream_(2006)_1080p24.webm) 파일은 HTTP 429로 다운로드 실패하여 등록하지 않았다. 동일 작품의 다른 인코딩은 같은 work_id를 사용한다.

## BBB 전수 대조

`python scripts/audit_bbb_research_v1.py`: 72/72개 바탕이 기록된 원본 구간과 일치하고, 연결된 288/288개 합성 NPZ의 `clean` 배열이 바탕과 일치했다. 원본 해시는 실행 전후 일치. 각 바탕에 연결된 manifest 레코드의 출처·시작 프레임·background_check는 72/72 그룹에서 일치했다. 선별 당시 판정 코드의 정확한 해시는 기록되지 않았다. 이는 파일 대응 확인이며 안전 보증이 아니다. 과거 선별의 `measure.analyze`는 휘도 전이와 밝기·장면 변화 조건에 집중했고 모든 위험 유형의 독립 라벨을 뜻하지 않는다.

추가로 train 바탕 0번을 현재 B0(`measure.analyze_all`, 24fps)로 평가했으며 휘도·적색·패턴 모두 미검출이었다. 결과는 `data/datasets/research_data_v1/bbb_000_b0_observation.json`에 있다. 실행 시 `measure.py` SHA-256은 `ef79d67d7a6cd53f18becd41a8be2aede4b3c546ee032f8cdc512ef41ab54fda`다. 이것은 1개 바탕의 도구 관측이며 안전성 보증이나 72개에 대한 신규 판정이 아니다.

## South

South et al. 2021의 OSF `https://osf.io/5a3dy/` 공개 API를 조회한 결과 프로젝트는 공개이지만 license 필드는 null이다. 영상의 권리 및 학습 조건은 확인하지 못했다. CSV의 위험 163/안전 1,837, flash·red·pattern은 저자 도구 결과이며 임상 라벨이 아니다. South를 학습 또는 평가에 사용할 권한을 자동 승인하지 않는다. 문의를 위한 [초안](south_usage_request.md)을 제공하며 메시지는 보내지 않았다.

## 모델 이력과 공통 평가

| 모델 | 입력 및 출력 | 사용 이력·제약 |
|---|---|---|
| B0 LUMI | BGR uint8 T×H×W, fps → 휘도·적색·패턴 이진 판정 | 규칙 기반. 자체 판정이 만든 라벨과의 일치는 독립 성능이 아니다. 연속 위험 점수는 사용하지 않으므로 AUROC 미산출 |
| B1 FlashGuard-inspired | BGR 영상·fps, 학습한 계수·문턱 → 전체 점수·판정 | 계수 미학습, 위험 유형별 출력 없음. 원 논문의 정확한 재현 아님 |
| B2 HazNet | 24fps, 48프레임 → 90×160 시간 입력과 360×640 두 프레임, 휘도·적색·패턴 점수 | a/b는 합성 자료로 학습. a는 Kaya 포함 실제 val로 모델 선택, a/b는 Kaya val로 문턱 선택. Sintel·Kaya 사용 이력을 독립 시험으로 오인하지 않음 |

`lumi_synth.baselines.comparison.infer`는 동일 인터페이스에서 B0/B1/B2를 호출한다. B2의 기존 구현에 맞춘 크기 변환은 종횡비를 강제 변경하므로, 비교 시 변환에 의한 공정성 영향을 따로 분석해야 한다. B1은 현재 모든 공간·시간 점수의 최대를 클립 점수로 쓰는 추가 가정이다. `evaluate_records`는 사용 조건·라벨 근거·final_test 및 모델 선택 미사용을 요구하며 이 필드는 사용자가 확인한 근거를 기록하는 것이지 자동으로 권리를 판정하지 않는다.

유형별 Precision·Recall·F1·FNR·AUROC(연속 점수와 양쪽 클래스가 있을 때)를 계산한다. 외부 라벨별 결측이나 B1의 유형 미지원은 결과 없음으로 둔다. 처리 시간은 디코딩을 제외한 호출 시간이다. CPU RSS 전후와 Windows의 프로세스 생애 최대 작업 집합, CUDA 할당량 최대값을 기록한다. 프로세스 생애 최대값은 각 모델만의 순수 피크가 아니며, 엄밀한 비교는 독립 프로세스·동일 하드웨어·예열과 반복이 필요하다.

기존 데이터에 독립 최종 시험으로 확정 가능한 실제 위험 정답과 사용 조건이 아직 없으므로 실제 성능을 산출하지 않았다. 정상 테스트 영상과 비영상 배열, 그리고 단색 배열로 기존 HazNet b 체크포인트를 읽는 동작 확인만 수행했다. 추후 승인된 훈련 데이터에서 B1 계수·검증 문턱을 고정한 다음 별도 작품·출처의 독립 시험을 1회 평가한다.

동작 확인 명령: `python scripts/compare_research_v1.py --checkpoint data/hazcls/b.pt`. 시험용 단색 배열과 임의 B1 계수를 쓰므로 이 출력은 위험 성능 평가가 아니다. B2 체크포인트는 신뢰하는 로컬 파일만 사용한다.

동작 확인에서 32×48 단색 입력을 B2 기존 전처리 크기로 확대한 경우 b 체크포인트의 패턴 점수가 약 0.967로 기본 0.5를 넘었다. 이는 한 인공 입력의 관찰이며 일반화 성능을 뜻하지 않는다. 작은 입력의 확대, 전처리와 모델 동작을 추가 조사해야 한다.

테스트: `python -m pytest tests/test_research_compare.py tests/test_flashguard_baseline.py -q` 6 passed(공통 인터페이스 수정 전), 이후 `python -m pytest -q` 201 passed, 1 skipped, 1 warning. 실제 위험 영상 신규 생성·모델 학습은 하지 않았다.
