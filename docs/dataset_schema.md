# Phase 1 데이터셋 레코드

각 JSONL 행은 스키마 버전 1의 표본이다. 필수 필드는 `schema_version`, `dataset_version`, `sample_id`, `work_id`, `source_id`, `parent_ids`, `root_source_ids`, `origin`, `pairing`, `pair_id`, `risk_label`, `verification_status`, `label_source`, `label_evidence`, `dataset_role`, `transform_history`, `license`, `sha256`이다. `path`, `metadata`, `observations`는 선택 필드다. 경로는 현재 로컬 절대 경로이며 배포용 경로가 아니다.

`work_id`는 사람이 동일 작품을 묶어 부여한다. `source_id`는 원본 파일 식별자, `sample_id`는 이 레코드 식별자다. `parent_ids`는 직접 부모의 sample_id 목록이고 `root_source_ids`는 모든 조상의 원본 source_id 집합이다. 다중 부모가 다른 split이면 오류다. 작품·바이트 동일 파일의 split 충돌도 오류다.

`pairing`: `clean`(검증 근거 있는 정상), `paired`(대응 정답 참조), `identity`(자기 자신), `unpaired`, `uncertain`. `paired`의 `pair_id`는 기존 sample_id이며 같은 split이어야 한다. `risk_label`은 별도 값이며 null은 미확인이다. `verification_status`, `label_source`, `label_evidence`는 판정 및 라벨의 근거를 표현하며, tool 출력이 곧 의학적 안전성이나 검증 정답은 아니다. 현재 `clean`/`identity`는 `verification_status=verified`와 근거를 요구한다.

여러 판정기는 `observations` 배열에 개별 `{tool, version, settings, run, result}`로 보존한다. 후속 판정은 기존 배열을 덮어쓰지 않고 덧붙이는 책임이 호출자에게 있다. `dataset_role`은 `train`, `val`, `test`, `external`, `unassigned`이다. `transform_history`는 변환 이력이며 클립은 원본 PTS와 출력 시간 매핑을 담는다.

현 구현은 스키마 의미 전체의 완전한 형식 검증기나 데이터베이스가 아니다. 파일 해시와 작품 ID는 자동 유사 영상 탐지가 아니며, 검증자는 재인코딩·부분 중복을 별도로 확인해야 한다.

평가 사용 이력은 별도 JSONL에 `experiment_id`, `sample_id`, `usage_role`, `model_selection_used`, `threshold_selection_used`, `first_accessed_at`, `evidence`로 기록한다. 확인되지 않은 시각·사용 여부는 null이다. 이력이 없거나 불명확하면 독립 최종 시험 후보로 자동 분류하지 않는다. [Kaya 이력](kaya_usage_phase1_1.jsonl)의 정확한 최초 접근 시각은 미확인이다.
