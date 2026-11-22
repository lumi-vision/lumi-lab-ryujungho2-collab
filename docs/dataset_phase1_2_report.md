# Phase 1.2 실제 데이터 구축 기록

2026-11-18. 원본을 재생하지 않고 전체 디코드·PTS 및 파일 해시를 검사했다. 새 결과는 `data/datasets/phase1_2_v1/manifest.jsonl`과 `statistics.json`이며 기존 manifest는 건드리지 않았다. 아래 역할은 초기 개발 분할로, 최종 논문의 독립 시험셋 확정이 아니다. 모든 표본의 위험 라벨은 null, 검증 상태는 unverified다.

| 작품 | 출처·이용 조건 | 영상 사양(실제 파일 메타데이터) | 크기 | 신규 역할 |
|---|---|---|---:|---|
| Big Buck Bunny | [공식 다운로드](https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_1080p_h264.mov.zip), 기존 기록 CC BY 3.0, © Blender Foundation | H.264, 1920×1080, 24fps, 14,315프레임, 약 596.46초 | 725,106,140 bytes | train |
| Sintel | [공식 다운로드](https://download.blender.org/durian/movies/Sintel.2010.1080p.mkv), 기존 기록 CC BY 3.0, © Blender Foundation | H.264, 1920×818, 24fps, 21,312프레임, 컨테이너 약 888.03초 | 1,180,090,590 bytes | val |
| Tears of Steel | [공식 파일](https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov), [공식 배포 설명](https://www.youtube.com/watch?v=R6MlUcmOul8), CC BY 3.0, `(CC) Blender Foundation | mango.blender.org` | H.264, 1280×534, 24fps, 17,620프레임, 약 734.17초 | test |

각 작품의 30~32초에 대해 48프레임 가상 클립 1개를 만들었다. `source_id`, `work_id`, `clip_id`, 부모·원본 ID, 프레임 인덱스·PTS·time_base·상대 시각과 분할·변환 이력을 새 manifest에 보존했다. 총 레코드 6개, 작품 3개, 오류 0개. 해시 불변을 각 단계에 검사했으며 동일 원본 구간 재등록 중복 및 누출 오류는 자동 테스트로 검사했다. 과거 BBB train/val은 이전 실험으로 유지하고 새 분할과 혼합하지 않는다.

실제 데이터 구축 명령 `python scripts/build_phase12_inventory.py`: 153.22초, 해당 프로세스의 Windows peak working set 70,418,432바이트(약 67.2MiB). 환경별 편차가 있으며 이는 원본 3편의 검사 및 2초 가상 클립 생성 과정의 수치이지 기존 G 학습 메모리가 아니다. 디코딩은 스트리밍하고 영상 프레임을 누적하지 않는다. 가상 클립은 프레임별 메타데이터만 메모리에 보관하므로 클립이 길어질수록 메타데이터가 증가한다. 디스크 중간 영상은 만들지 않았다. 기존 G 로더는 NPZ 전체를 메모리에 적재한다.

모델 변환은 검증된 정상 라벨과 근거가 있는 가상 클립만 허용한다. PyAV 리사이즈 및 BGR24 변환, 원본 색상 태그, 프레임별 시간 매핑·해시를 기록하며 시간 보간이나 프레임률 변환은 하지 않는다. 알려진 HDR 전이 및 일부 픽셀 포맷은 거부한다. 색상 태그가 미기재된 영상의 HDR 여부는 자동으로 보증할 수 없다. 정상 테스트 영상에서 기존 G 로더와 호환됨을 확인했다. 실제 영화는 정상 판정이 없어 모델 입력으로 내보내지 않았다.

`python -m pytest -q`: 195 passed, 1 skipped, 1 warning. 전체 테스트에는 기존 합성 함수 단위 테스트가 포함되지만 별도 위험 영상 파일 생성이나 모델 학습은 수행하지 않았다. 추가 후보·라이선스 미확인 사항은 [후보 조사](source_candidates_phase1_1.md)를 참조한다.
