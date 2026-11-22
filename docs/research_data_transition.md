# 연구 자료 확보 및 기존 바탕 조사

2026-11-18. 위험 영상 신규 생성·자동 재생 없이 기존 자료를 조사했다.

## 추가 작품

- Elephants Dream: [공식 배포](https://orange.blender.org/download/) 및 [공식 CC BY 2.5 조건](https://orange.blender.org/press/). 공식 Blender 서버의 `elephantsdream-480-h264-st-aac.mov` 103,636,531바이트를 `data/research_sources/elephants_dream/`에 확보하고 신규 `data/datasets/research_a_v1/manifest.jsonl`에 미확인 원본으로 등록했다. SHA-256 `a7dbacae5fa2bc345dbd7b10588575f3e2bba36012e9a93df5346467053b0634`, H.264 yuv420p 480×270, 평균 fps 367056/15691(약 23.39), 명목 24fps, 15,294프레임, PTS 단조 검사 통과. 원본 자동 재생 없음. 768×432 G 입력으로 직접 쓰지 않는다.
- Spring: [공식 CC BY 4.0 및 저작자 표시](https://studio.blender.org/projects/spring/pages/about/), [배포 안내](https://studio.blender.org/blog/spring-spring-open-content-overview/). 영화 파일의 직접 주소·크기·fps 등은 아직 미확인. 확보하지 않음.
- Cosmos Laundromat: [공식 영상 항목](https://studio.blender.org/projects/cosmos-laundromat/gallery/?asset=1537)은 공유·수정을 허용한다고 설명한다. 정확한 파일 라이선스 판본·직접 주소·사양은 미확인. 확보하지 않음.
- South: 이전 자료에 OSF 공개 위치와 원본 2,000개 기록은 있지만 영상별 저작권 및 재사용 조건이 명시적으로 확인되지 않았다. OSF 페이지·API 접근에서 403/접근 실패가 발생해 추가 조건을 확인하지 못했다. 공개 접근 사실을 학습·재배포 권한으로 간주하지 않는다. 문의가 필요하면 사용자가 직접 결정한다.

## BBB 원본 및 판정 이력

`data/oracle_768x432_24fps/manifest.jsonl`에서 BBB train 60개·val 12개 바탕은 동일 작품의 서로 다른 구간이다. `backgrounds_<split>.npz`는 클립 배열과 fps만 담고 각 원본 시점은 oracle manifest의 `source`·`start_frame`에 있다. train 0(start 720), val 0(start 10944)의 manifest에 연결된 NPZ `clean`이 각 배경 배열의 0번과 동일함을 실측했다. 나머지 70개에 대한 전수 일치는 이번에 검사하지 않았다.

생성 스크립트의 `is_clean`은 `measure.analyze`의 화면 전이 수 0, 평균 밝기 20 cd/m² 이상, 연속 프레임 평균 절대 차이 40 미만 조건으로 후보를 선택한다. 이는 주로 휘도 검사이며 적색·공간 패턴 전체의 독립 승인이나 의학적 안전 보증이 아니다. 각각의 `background_check`에는 수치가 보존되지만 개별 판정 코드 해시·정확한 버전은 manifest에 없다. `checks/oracle/768x432_24fps.md`는 2026-11-14 동시성 창 20ms로 합성 라벨을 재판정한 이력을 기록한다. 원래 바탕 선별이 그 새 버전으로 다시 수행됐다는 근거는 확인되지 않았다.

## 종횡비 및 독립 평가

원본 비율을 유지하며 여백을 두면 내용은 잘리지 않지만 검은 면적이 위험 면적 분모에 포함되고 합성 마스크가 여백에 닿을 수 있다. 중앙 크롭은 비율을 유지하지만 가장자리 위험 요소가 사라질 수 있다. 강제 늘리기는 줄무늬 공간 주파수와 물체 모양을 바꾼다. 따라서 콘텐츠 유효 영역과 변환 이력을 저장하고, 판정 도구가 전체 화면과 유효 영역 중 어느 좌표계의 면적을 쓰는지 분명히 해야 한다. 기존 `data.content_box`·`letterbox`는 종횡비 보존과 여백 처리의 참고 구현이다. 실제 선택은 도구 설정과 별도로 비교 검증해야 한다.

기존 Sintel은 GDR 시험 이력이 있고 Kaya 16개는 HazNet 선택·문턱에 쓰였다. 새로운 작품 또는 권한이 확인된 별도 실제 자료를 작품·출처 및 유사 영상 단위로 분리해 최종 시험 후보로 남기고, 라벨 근거를 확인한 뒤 학습·개발·문턱 선택에 사용하지 않는다. 지금은 그러한 최종 독립 시험셋이 확보되었다고 주장할 수 없다.
