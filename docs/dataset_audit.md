# Phase 1 데이터셋 인프라 사전 분석 및 구현 계획

작성일: 2026-11-18. 소스와 기존 문서 및 로컬 데이터 목록을 확인했다. 영상은 재생하지 않았고, 새 위험 영상 생성이나 모델 학습은 하지 않았다. 이 문서는 구현 전 계획이다.

## 저장소 및 현재 변경 사항

- 저장소 루트와 상위 디렉터리, 예상 작업 디렉터리의 `AGENTS.md`를 확인했으며 발견하지 못했다. `README.md`, `DATA_SOURCES.md`, `docs/DATA_INVENTORY.md`, 기존 G·데이터·판정·학습 코드를 확인했다.
- 이미 수정되었거나 추적되지 않는 파일이 있다: `DATA_SOURCES.md`, `README.md`, `scripts/train_gan.py`, `src/lumi_synth/gan.py`, `docs/DATA_INVENTORY.md`, `docs/TEAM_PROPOSAL_realism_2026-11-17.md`, `scripts/prepare_real_split.py`, `src/lumi_synth/dynamic.py`, `tests/test_dynamic.py`, `tests/test_gan_moving_mask.py`. 이들은 기존 작업으로 보존한다.

## 기존 흐름과 재사용 판단

| 대상 | 현재 구현 | 재사용·한계 |
|---|---|---|
| `src/lumi_synth/gan.py` | 휘도·적색·패턴별 조건부 G. 원본 BGR 프레임, 영역 마스크, 목표, 잠재벡터를 입력으로 합성 프레임과 필드를 출력 | Phase 1에서 변경하지 않는다. 정지·시간별 마스크 경로가 있으나 G 학습은 범위 밖 |
| `scripts/train_gan.py`, `generate.py` | `backgrounds_<split>.npz`의 `[N,T,H,W,3]` uint8 BGR 및 fps 사용. 체크포인트에서 위험 종류·학습 해상도·fps 등을 읽음 | 장래 별도 내보내기로 기존 NPZ 트랙과 연결. 현재 생성 경로는 체크포인트 fps·해상도 불일치를 거부하므로 범용 원본 저장과 분리 |
| `src/lumi_synth/data.py` | PyAV 기반 디코딩, 색 메타데이터, SHA-256, `ClipSpec`, 클립 추출·판정 유틸리티 | 해시·디코딩 로직을 참고. 기존 클립 API의 고정 fps와 프레임 인덱스 가정은 VFR 타임스탬프 보존의 보증이 아니므로 확인 필요 |
| `scripts/make_oracle_dataset.py` | 정해진 작품 구간을 스캔, 배경 NPZ와 생성 샘플·JSONL을 만듦 | 소스·판정·합성이 결합돼 Phase 1 등록기로 직접 사용하지 않는다. 현재 manifest에는 작품별 분할 위반이 존재 |
| `scripts/make_gdr_dataset.py`, `scripts/gdr_common.py` | 합성 원본 인덱스, 판정기 튜브, 선택적 alpha를 가진 NPZ 및 manifest. 학습기는 기존 트랙의 bg 인덱스에 의존 | 기존 자료는 수정하지 않고 포맷 및 계보를 문서화. 판정 결과를 검증 정답으로 자동 승격하지 않는다 |
| `src/lumi_synth/roi.py` | `judge_tubes(T,H,W,3 uint8 BGR, fps)` → 종류별 불리언 튜브 및 요약. 줄무늬는 표본 프레임 분석 | 결과는 도구 이름·버전·설정과 함께 관측값으로만 저장. 이 단계는 자동 판정 실행 불필요 |
| `scripts/train_roi.py`, `scripts/train_corrector.py` | 기존 GDR manifest/NPZ 학습 및 체크포인트 | 학습 로더·모델·체크포인트 형식은 변경하지 않는다 |
| `scripts/prepare_real_split.py` | South CSV와 Kaya 파일의 SHA-256, 목록 작성. South 전량 train, Kaya external | 목록 형식·해시 로직을 참고. 파일 해시는 재인코딩된 중복을 찾지 못하며 이 스크립트는 새 시스템의 데이터 계약이 아님 |

## 보유 자료 및 발견 사항

- `data/sources/`에 BBB 1080p 1편, `data/raw/`에 Sintel 1편. `data/oracle_768x432_24fps/`는 train 60·val 12·test 16 배경 클립이며 train·val은 BBB, test는 Sintel이다. 기존 train/val은 다른 시간대이지만 **동일 작품**이므로 새로운 작품 단위 분할 조건에 부합하지 않는다.
- `data/real/south2021/`: 원본 파일 2,000개와 CSV. 위험 163·안전 1,837은 기존 도구 라벨이다. 사용자 방침은 South 전체 학습, Kaya 외부 평가다. 이번 단계에서 다운로드하거나 재배포하지 않는다.
- `data/real/kaya2025/`: 원본 22개 및 결과 영상·자료. 기존 `data/hazcls/real/items.jsonl`에 Kaya `val` 기록이 있어, 이전 실험의 독립 외부 평가 여부는 별도 추적이 필요하다.
- `checks/hazcls/a.md`와 `scripts/train_hazcls.py`를 대조하면 실험 a는 Kaya 16개를 포함한 실제 val 손실을 `both` 감시에 포함해 최적 epoch를 선택했다. b는 합성 val로 epoch를 선택했으나 실제 val의 Kaya를 문턱 선택에 사용했다. 따라서 각 실험에서 Kaya 전체를 손대지 않은 최종 시험으로 취급할 수 없다. Kaya 6개의 별도 test 기록은 있지만 독립적인 새 실험 여부는 따로 관리해야 한다.
- `data/gdr/`, `data/gan/`, `data/hazcls/`, `data/playlist/`는 파생 데이터·체크포인트·캐시가 섞여 있다. 파일 개수는 고유 원본 수가 아니다. 기존 파일을 이동·삭제하거나 기존 manifest를 덮어쓰지 않는다.
- 실제 자료의 라이선스와 사용 범위, South/Kaya 간 재인코딩 중복, 동영상별 프레임 시간의 신뢰성은 아직 확인되지 않았다. 바이트 해시 일치는 정확한 파일 중복만 의미한다.

## 제안하는 스키마와 구현

1. 독립 패키지 `src/lumi_synth/dataset/`에 버전이 있는 JSONL manifest 스키마, 등록·검증·분할·통계 모듈을 추가한다. `work_id`는 명시적으로 부여한 원본 작품 ID, `source_id`는 파일 해시 기반 ID로 두고, 클립은 원본과 변환 이력·작품 ID를 참조한다. 동일 바이트 파일이 다른 작품 ID로 등록되면 충돌을 보고하고 분할을 막는다. 서로 다른 인코딩의 동일 작품 여부는 자동으로 보증할 수 없으므로 사용자가 같은 작품 ID를 부여해야 한다.
2. 데이터 유형 `clean`, `paired`, `identity`, `unpaired`, `uncertain`과 판정 관측, 검증 상태, 라벨 출처·증거를 별도 필드로 둔다. 미확인은 `uncertain`, clean은 명시적 검증 근거가 있을 때만 인정한다. paired는 정답 참조와 정렬 관계를 요구하고 identity는 입력과 정답의 동일성을 검사한다. 판정기 결과만으로 검증 정답을 만들지 않는다.
3. PyAV를 우선 사용해 컨테이너·코덱·크기·시간 기준·원본 평균/명목 fps를 추출하고, 디코드 오류와 누락·비단조 PTS를 기록한다. 설정 기반 구간 추출은 원본을 보존하고 원본 PTS/time_base와 출력 프레임의 대응을 기록한다. 인코더·컨테이너에 따라 PTS가 달라질 수 있어 원본-출력 매핑을 테스트한다. VFR을 평균 fps만으로 CFR로 바꾸지 않는다. PTS가 없거나 보존을 검증할 수 없는 경우 실패 또는 미확인으로 기록한다.
4. 작품 ID 기준 seed 고정 분할을 구현하고, 클립·paired 관련 레코드에 원본과 동일한 분할을 강제한다. 파일 해시, 참조 무결성, 작품 분할 충돌, 클립 경계·경로·라벨 근거를 검사한다. 비율은 작은 작품 수에서 근사치이며 빈 분할이나 고정 외부 평가를 명시적으로 보고한다. South 전체 train·Kaya 외부 평가라는 사용자 방침은 구성에 고정 역할로 표현한다.
5. `SouthAdapter`는 로컬 경로·CSV를 받는 인터페이스만 제공한다. 접근 권한·이용 조건의 확인 상태가 기록되기 전 실제 자료 등록을 수행하지 않고, 다운로드·재배포 기능을 만들지 않는다. 실제 위험 라벨은 관측으로만 보존하며 복원 정답을 만들지 않는다.
6. CLI와 설정 예제, 단위·누출 테스트, 통계 생성 및 실행 README를 추가한다. 테스트는 임시로 생성한 단색 정상 영상과 잘못된 파일만 사용하며 자동 재생하지 않는다. 통계는 유형·작품·파일·분할·관측/검증 상태·오류를 집계한다.

### 예정 변경 파일

- 신규: `src/lumi_synth/dataset/`의 스키마·등록·영상·분할·검증·통계·South adapter 모듈, `scripts/dataset_cli.py`, `configs/dataset.example.json`, `tests/test_dataset_*.py`, `docs/dataset_schema.md`, `docs/dataset_readme.md`, 시험 데이터 기반 통계 보고서.
- 기존: `README.md`에 새 문서 링크만 추가. 필요시 기존 함수는 변경하지 않고 호출한다.
- 변경하지 않음: 기존 G·보정 구조, 가중치, 기존 데이터 및 실험 결과. 위험 생성과 G 학습은 하지 않는다.

## 검증 및 예상 문제

- 같은 seed·입력과 설정이면 분할 결과가 같아야 한다. 같은 작품의 서로 다른 클립, 파일 중복, 쌍의 서로 다른 split, 누락 참조를 실패시키는 테스트를 작성한다.
- 정상 테스트 영상의 메타데이터·프레임 PTS·변환 후 크기·원본 해시 불변을 확인한다. 손상 파일과 읽기 실패는 예외 또는 명시적 실패로 기록한다.
- 원본 PTS, VFR 및 색 메타데이터는 컨테이너·코덱에 따라 결측되거나 재인코딩 과정에서 변할 수 있다. 보존 확인이 불가능한 경우 성공으로 보고하지 않는다. G용 24fps 변환은 별도 모델 입력 변환이며 원본 보존과 구분한다.
- 안전 영상이라는 분류는 검증 근거에 의존하며 디코드 성공이나 판정 미실행으로 유추하지 않는다. 자동 재생·미리보기는 하지 않는다.
- 고유 작품 수가 두 편뿐인 현재 로컬 영화로 train/val/test 세 분할을 모두 작품 단위로 채울 수 없다. 추가로 적법하게 사용할 수 있는 작품이 필요하다.
