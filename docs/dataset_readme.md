# Phase 1 사용 방법

설치된 `.venv`에서 `python scripts/dataset_cli.py configs/dataset.example.json`을 실행한다. 예제의 경로는 사용자가 준비한 정상 영상으로 바꿔야 한다. CLI는 GUI나 재생을 하지 않는다. 설정의 `action`은 다음과 같다.

- `register`: `dataset_version`, `sources`의 `path`, `work_id`, `license`, 선택 `role`, 새 `output` JSONL을 지정한다. 원본 해시·전체 디코드·PTS를 검사한다. 위험 라벨과 검증은 미확인으로 기록한다.
- `split`: `input`, 새 `output`, 정수 `seed`, 합이 1인 `ratios` (`train`, `val`, `test`)를 지정한다. 명시적으로 지정된 역할은 유지하고, 미지정 작품은 해시 기반으로 분할한다. 소규모 작품 수에서는 특정 분할이 비어 있을 수 있다.
- `validate`: `input` 및 선택 `check_files`를 지정한다. 오류가 있으면 종료 코드 1이다. 파일 검사는 기본으로 켜져 있다.
- `statistics`: `input` JSONL의 수량을 표준 출력에 JSON으로 쓴다. 필요시 셸 리디렉션으로 새 보고 파일에 저장한다.
- `virtual_clip`: `source`, `start`, `end`, 새 `output` JSON을 지정한다. 원본 해시와 프레임별 PTS·time_base·원본 시간·상대 시간을 기록한다.
- `clip`: 입력 manifest `input`, `parent_id`, 새 `sample_id`, 새 영상 `output`, 새 `manifest_output`, 유리수 또는 십진 문자열 `start`/`end`, 선택 `width`/`height`를 지정한다. FFV1 MKV 출력은 1ms 시간축으로 양자화하며 원본과 출력 시각 및 프레임별 오차를 분리 기록한다. 역디코딩 후 프레임 수·순서·픽셀·시간이 다르면 실패한다. 원본 PTS가 없거나 비단조이거나 양자화로 시간이 겹치면 실패한다.

기존 G용 `export_backgrounds`는 크기와 시간 간격이 이미 일치하는 파일들만 NPZ로 내보낸다. 원본 프레임률 변환이나 위험/안전 인증을 수행하지 않는다. 사용자는 검증된 정상 영상만 지정해야 한다. 이 단계의 호환성은 작은 단색 시험 영상으로 기존 `train_gan.load_backgrounds`를 호출해 확인했다. 실제 G 학습은 수행하지 않았다.

`export_virtual_backgrounds`는 별도 모델 입력 단계다. virtual clip의 원본 해시와 PTS를 재검사하고 지정 fps에 정확히 일치하는 프레임만 NPZ로 내보낸다. 재표본화·크기 조정은 하지 않으며 VFR은 지정 fps와 다르면 거부한다. 마지막 프레임 지속시간은 미확인으로 둔다.

Phase 1.2의 `register_virtual` 동작은 `input` manifest, `parent_id`, `start`, `end`, 새 `output` manifest를 받아 계보 레코드를 추가한다. 같은 원본·시간 구간을 다시 등록해도 레코드를 중복시키지 않는다. `export_model`은 `input`, `clip_id`, 새 `output`, 새 `metadata_output`을 요구한다. 명시적으로 검증된 정상 클립만 한 클립씩 리사이즈·BGR 변환하며 프레임 대응과 색상·변환 이력을 별도로 저장한다. 원본 fps와 정확히 일치하지 않으면 변환하지 않는다. [실제 원본 실행 결과](dataset_phase1_2_report.md).

새 manifest 및 영상은 기존 경로를 덮어쓰지 않는다. South 어댑터는 `terms_verified=True`를 명시해야 로컬 CSV를 읽으며, 그 값은 실제 권한 확인을 대신하지 않는다. 다운로드 기능은 없다.
