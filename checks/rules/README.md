# 규칙 검증: 우리 판정기 vs pse-test-media ITU 기대값

**결과: ITU-R BT.1702 기대값과 201/210 일치** (기본 2세트 53/56)

> ⚠️ 시험 영상은 광과민성 발작 위험 패턴입니다. 프레임은 메모리에서만 만들어 판정기에 넣었고, 화면에 띄우거나 파일로 남기지 않았습니다.

## 무엇을 시험했나

- 판정기: `src/lumi_synth/measure.py` 의 `analyze(frames_bgr, fps)` → `dangerous` (1초 깜빡임 > 3번)
- 정답: pse-test-media `video_creation/<세트>/*.json` 의 `expected_result["itu_r1702_4"]` (pass/fail)
- 프레임: pse-test-media `generate_frames.py` + `generate_video.py` 와 같은 계산을 메모리에서 재현 (PNG 불투명 픽셀 재칠 → 알파 합성 → 배경 → BGR, 앞쪽 padding 10프레임 = `video_config.json`). 원본 도구는 FFV1(무손실)로 저장하므로 픽셀이 같습니다 (`--ffv1-check 3`: 처음 3개 영상은 실제 FFV1 저장→읽기 후 픽셀 100% 같음을 확인).
- 해상도: 원본 1920×1080 (축소 없음). docs/BASICS.md·PLAN.md 에 적힌 대로 640×360 으로 줄이면 안전 패턴 a003 이 25.05%가 되어 위험으로 뒤집히므로 원본에서 잽니다.
- 메모리: `luminance_cd` 를 프레임 4장씩 나눠 부르는 래퍼로 바꿔 끼웠습니다(원소별 계산이라 결과가 비트 단위로 같음, 실행 시작 때 원래 함수와 같은지 assert). measure.py 자체는 고치지 않았습니다.

## 버전

- pse-test-media: `ef78542aad496827f25064f57aaf13d6a741ba75`
- 이 저장소 HEAD: `6847650e8051f6786f1458162fec4745296a422c`
- measure.py SHA256: `79675481a41351d2f04ba9ac8fd25f138a84295fccad5ab2b91f51cee8010382`
- 실행 시간: 5211초 (210개 영상, CPU)
- 참고: 모든 판정 수치는 위 1회 전체 실행(2026-11-09)의 값이다. 불일치 원인 메모·겹침 면적·기대값 메모 칸은 실행 뒤 스크립트에 추가했고, 이 문서는 그 전체 실행의 결과표를 다시 읽어 같은 `write_report()` 로 다시 쓴 것이다(87분 재실행 생략). 영상별 결과표는 처음 생성본과 글자 하나까지 같음을 확인했다.

## 혼동표 — 기본 2세트 (broadcast_30fps_01, 30fps_alternating_01) (행 = 기대값, 열 = 우리 판정)

| | 우리 fail (위험) | 우리 pass (안전) |
|---|---|---|
| **기대 fail** | 12 | 0 |
| **기대 pass** | 3 | 41 |

일치 **53/56**

## 혼동표 — 이번 실행 전체 (행 = 기대값, 열 = 우리 판정)

| | 우리 fail (위험) | 우리 pass (안전) |
|---|---|---|
| **기대 fail** | 16 | 0 |
| **기대 pass** | 9 | 185 |

일치 **201/210**

### 세트별

| 세트 | 영상 수 | 일치 | 기대 fail 수 |
|---|---|---|---|
| 30fps_alternating_01 | 16 | 16/16 | 4 |
| broadcast_30fps_01 | 40 | 37/40 | 8 |
| broadcast_30fps_inf01 | 10 | 8/10 | 4 |
| broadcast_30fps_inf02 | 10 | 6/10 | 0 |
| scene_30fps_0p1rl | 12 | 12/12 | 0 |
| trace24_30fps_01 | 54 | 54/54 | 0 |
| trace24_30fps_area_combo01 | 20 | 20/20 | 0 |
| trace24_30fps_inf01 | 16 | 16/16 | 0 |
| trace24_30fps_sync01 | 8 | 8/8 | 0 |
| wcagc_30fps_area01 | 12 | 12/12 | 0 |
| wcagc_30fps_area02 | 12 | 12/12 | 0 |

### 건너뛴 세트

| 세트 | 이유 |
|---|---|
| broadcast_30fps_combo01 | 회색이 아닌 색(빨강) 프레임이 있음 — 휘도 전용 판정기로는 빨강 깜빡임 조건을 볼 수 없음 |
| broadcast_30fps_red01 | 회색이 아닌 색(빨강) 프레임이 있음 — 휘도 전용 판정기로는 빨강 깜빡임 조건을 볼 수 없음 |
| broadcast_30fps_red02 | 회색이 아닌 색(빨강) 프레임이 있음 — 휘도 전용 판정기로는 빨강 깜빡임 조건을 볼 수 없음 |
| trace24_30fps_combo01 | 회색이 아닌 색(빨강) 프레임이 있음 — 휘도 전용 판정기로는 빨강 깜빡임 조건을 볼 수 없음 |
| trace24_30fps_red01 | 회색이 아닌 색(빨강) 프레임이 있음 — 휘도 전용 판정기로는 빨강 깜빡임 조건을 볼 수 없음 |
| trace24_30fps_red02 | 회색이 아닌 색(빨강) 프레임이 있음 — 휘도 전용 판정기로는 빨강 깜빡임 조건을 볼 수 없음 |
| wcagc_30fps_area03 | 회색이 아닌 색(빨강) 프레임이 있음 — 휘도 전용 판정기로는 빨강 깜빡임 조건을 볼 수 없음 |

### 일치 해석 주의

- 우리 판정기가 화면 전이를 한 번도 세지 않은 영상: 142개 (그중 기대 pass 142개). trace24·WCAG 용 세트처럼 깜빡이는 영역이 화면의 25%보다 훨씬 작거나(416×416 나 341×256 의 25%), 배경과 패턴의 밝기 차이가 20 cd/m² 미만인 경우다. 이런 영상의 일치는 '면적이 작거나 밝기 차이가 작으면 안전'만 확인한다.
- 기대 fail 영상이 있는 세트: 30fps_alternating_01, broadcast_30fps_01, broadcast_30fps_inf01 (나머지 세트의 ITU 기대값은 모두 pass).
- measure.py 모듈 설명(D2)에는 '초기화 전 판정기(같은 정의)가 pse-test-media ITU 기대값 56/56과 일치'라고 적혀 있다. 현재 measure.py 로 같은 기본 2세트를 재면 **53/56** 이다. 그 문장은 지금 판정기로는 재현되지 않는다.

### pse-test-media 기대값 메모

- `itu_r1702_4` 를 정답으로 썼다. 키 이름의 `_4` 가 BT.1702 의 어느 개정판(-0…-3)을 뜻하는지 저장소에 설명이 없다(video_creation/README.md 는 'Recommendation ITU-R BT.1702' 링크만 둠). 확인 못 함.
- 기대값의 'pass - small area' 는 pass 로 셌다.
- 세트 목록 CSV 의 `pass` 열은 그 세트의 **대상 표준** 기준이다(예: trace24 세트는 trace24). 그래서 ITU 기대값과 다른 행이 있는 것은 정상이다. 다만 ITU 가 대상 표준인 세트에서 다른 행은 기대값이 의심스럽다.

| 세트 | 영상 | 목록 CSV pass | dimension | JSON itu_r1702_4 |
|---|---|---|---|---|
| broadcast_30fps_inf02 | f001tf03_f003ico03 | FALSE | FAIL | pass |
| broadcast_30fps_inf02 | f003tf01_f005ico01 | FALSE | FAIL | pass |
| scene_30fps_0p1rl | f01f026 | FALSE | FAIL | pass |
| scene_30fps_0p1rl | f02f013 | FALSE | FAIL | pass |
| scene_30fps_0p1rl | f03f009 | FALSE | FAIL | pass |
| scene_30fps_0p1rl | f05f024 | FALSE | FAIL | pass |
| trace24_30fps_01 | f001f034 | FALSE | FAIL | pass |
| trace24_30fps_01 | f002fr005 | FALSE | FAIL | pass |
| trace24_30fps_01 | f003f024 | FALSE | FAIL | pass |
| trace24_30fps_01 | f004f001 | FALSE | FAIL | pass |
| trace24_30fps_01 | f005fr002 | FALSE | FAIL | pass |
| trace24_30fps_01 | f006f017 | FALSE | FAIL | pass |
| trace24_30fps_01 | f007f003 | FALSE | FAIL | pass |
| trace24_30fps_01 | f008fr016 | FALSE | FAIL | pass |
| trace24_30fps_01 | f009f013 | FALSE | FAIL | pass |
| trace24_30fps_01 | f010f038 | FALSE | FAIL | pass |
| trace24_30fps_01 | f011fr011 | FALSE | FAIL | pass |
| trace24_30fps_01 | f012f025 | FALSE | FAIL | pass |
| trace24_30fps_area_combo01 | 01m70f018_01n30f016 | FALSE | FAIL | pass |
| trace24_30fps_area_combo01 | 02m50f001_02n50f007 | FALSE | FAIL | pass |
| trace24_30fps_area_combo01 | 02m90f024_02n10f022 | FALSE | FAIL | pass |
| trace24_30fps_area_combo01 | 03m70f035_03n30f033 | FALSE | FAIL | pass |
| trace24_30fps_area_combo01 | 03m90f006_03n10f004 | FALSE | FAIL | pass |
| trace24_30fps_area_combo01 | 04m50fr013_04n50fr015 | FALSE | FAIL | pass |
| trace24_30fps_area_combo01 | 04m70f012_04n30f011 | FALSE | FAIL | pass |
| trace24_30fps_area_combo01 | 05m70f027_05n30f028 | FALSE | FAIL | pass |
| trace24_30fps_area_combo01 | 05m90fr007_05n10fr004 | FALSE | FAIL | pass |
| trace24_30fps_inf01 | a004f014a_f002f014 | FALSE | FAIL | pass |
| trace24_30fps_inf01 | f002f014a_f005c014 | FALSE | FAIL | pass |
| trace24_30fps_inf01 | f003f011a_f008y011 | FALSE | FAIL | pass |
| trace24_30fps_inf01 | f003y013a_f004fr013 | FALSE | FAIL | pass |
| trace24_30fps_inf01 | f004cr016a_f010fr016 | FALSE | FAIL | pass |
| trace24_30fps_inf01 | f007f011a_f009f011 | FALSE | FAIL | pass |
| trace24_30fps_inf01 | f007fr016a_a001fr016 | FALSE | FAIL | pass |
| trace24_30fps_sync01 | 01m70f018_01n30f018 | FALSE | FAIL | pass |
| trace24_30fps_sync01 | 02m50f024_02n50f024 | FALSE | FAIL | pass |
| trace24_30fps_sync01 | 03m90f006_03n10f006 | FALSE | FAIL | pass |
| trace24_30fps_sync01 | 05m70f007_05n30ff007 | FALSE | FAIL | pass |
| wcagc_30fps_area01 | f002f038 | FALSE | FAIL | pass |
| wcagc_30fps_area01 | f005f026 | FALSE | FAIL | pass |
| wcagc_30fps_area01 | f007fr001 | FALSE | FAIL | pass |
| wcagc_30fps_area01 | f009fr013 | FALSE | FAIL | pass |
| wcagc_30fps_area01 | f010f005 | FALSE | FAIL | pass |
| wcagc_30fps_area01 | f011f014 | FALSE | FAIL | pass |
| wcagc_30fps_area02 | f001f011 | FALSE | FAIL | pass |
| wcagc_30fps_area02 | f003f027 | FALSE | FAIL | pass |
| wcagc_30fps_area02 | f004fr006 | FALSE | FAIL | pass |
| wcagc_30fps_area02 | f006f038 | FALSE | FAIL | pass |
| wcagc_30fps_area02 | f008f002 | FALSE | FAIL | pass |
| wcagc_30fps_area02 | f012fr014 | FALSE | FAIL | pass |

## 불일치 분석

| 원인 | 개수 | 영상 |
|---|---|---|
| A. 여러 프레임에 나뉜 밝기 변화 (밝기 조건) | 4 | broadcast_30fps_01/f001y033, broadcast_30fps_01/f005y036, broadcast_30fps_01/f006l034, broadcast_30fps_inf01/f003y038_f005cs035 |
| B. 서로 다른 영역의 엇박자 깜빡임을 하나로 합쳐 셈 (영역·동기 조건) | 4 | broadcast_30fps_inf01/f002c015_f001cs012, broadcast_30fps_inf02/f002tc03_f004ics03, broadcast_30fps_inf02/f006tc03_f001ico03, broadcast_30fps_inf02/f007ty03_f006ics03 |
| C. 기대값(JSON) 자체가 의심됨 | 1 | broadcast_30fps_inf02/f001tf03_f003ico03 |

- **A. 여러 프레임에 나뉜 밝기 변화**: 우리 판정기는 픽셀마다 '직전 전이 이후의 극값'에서 변화량을 잰다(measure.pixel_transitions, 의도된 설계). pse-test-media 의 y/l 패턴은 한 깜빡임의 오르막이나 내리막을 20 cd/m² 미만 걸음 두 개로 나눠 '문턱 미달'로 설계했다. 걸음 합계(약 38~40 cd/m²)는 문턱의 두 배라서, 우리 판정은 위험이다. 즉 ITU 의 '20 cd/m² 이상 변화'를 한 프레임 사이로 재느냐(pse 기대값), 여러 프레임 누적으로 재느냐(우리)의 정의 차이다.
- **B. 엇박자 두 영역 합산**: 우리 screen_transitions 는 화면 어디서든 같은 방향 25% 초과 전이가 있으면 하나의 화면 상태 기계에 넣는다. 그래서 각자 3번/s 인 두 영역이 2프레임 이상 엇갈려 깜빡이면 합쳐서 4~6번/s 가 된다. pse 기대값은 영역별(또는 동시에 일어난 것만 면적 합산)로 센다. 겹친 픽셀만 실제로 더 많이 깜빡이지만 그 면적은 25% 미만.
- **C. 기대값 의심**: JSON 과 목록 CSV 가 서로 다르고, 패턴 설계로는 우리 결과가 맞는 경우.

### broadcast_30fps_01/f001y033 — 기대 pass, 우리 fail

- **원인: A. 여러 프레임에 나뉜 밝기 변화 (밝기 조건)** — y033 은 기대 fail 인 f033 과 깜빡임 하나만 다르다: 11→13프레임 하강이 87.9→68.4→49.2 cd/m² (한 프레임 걸음 19.5, 19.2 — 둘 다 20 미만, 합계 38.7). pse-test-media 는 이 깜빡임을 '밝기 문턱 미달'로 설계했다(파일명 y = 깜빡임 하나만 문턱 아래). 우리 pixel_transitions 는 '직전 극값'에서 재므로 두 프레임(67 ms)에 걸친 38.7 cd/m² 하강을 전이 1번으로 센다 → 3.5번/s → 위험. f033 에서는 같은 하강이 90.2→69.3→49.2 (20.9, 20.1).
- 목록 CSV 의 설계 조건(dimension): `luminance1`, 배경 휘도 69.3 cd/m²
- 패턴 `f001_fhd.png` (불투명 면적 25.1852%) × `rel_lum/y033_srgba.csv` 프레임별 휘도(cd/m², 괄호 = 투명이라 배경): 49.2 69.3 90.2 49.2 49.2 49.2 49.2 49.2 49.2 49.2 49.2 87.9 68.4 49.2 49.2 49.2 49.2 49.2 49.2 49.2 49.2 69.3 90.2 69.3 49.2 49.2 49.2 49.2 49.2 49.2 69.3 90.2 49.2 49.2

| 조건 | 측정값 | 기준 | 넘음 |
|---|---|---|---|
| 1. 밝기 변화폭 | 40.9 cd/m² (어두운 쪽 49.2) | ≥ 20 cd/m², 어두운 쪽 < 160 | 예 |
| 2. 깜빡임 횟수 (픽셀 기준) | 1초 최대 3.5번 | > 3번 | 예 |
| 3. 같은 방향 동시 전이 면적 | 최대 25.19% | > 25% | 예 |
| 최종: 1+2+3 모두 (화면 기준 깜빡임) | 1초 최대 3.5번 | > 3번이면 위험 | 예 |

### broadcast_30fps_01/f005y036 — 기대 pass, 우리 fail

- **원인: A. 여러 프레임에 나뉜 밝기 변화 (밝기 조건)** — y033 과 같은 구조: 하강 146.1→127.5→107.9 cd/m² (걸음 18.6, 19.6 / 합계 38.2). 걸음마다 재면 문턱 미달, 극값 기준으로 재면 전이.
- 목록 CSV 의 설계 조건(dimension): `luminance1`, 배경 휘도 128.9 cd/m²
- 패턴 `f005_fhd.png` (불투명 면적 25.1852%) × `rel_lum/y036_srgba.csv` 프레임별 휘도(cd/m², 괄호 = 투명이라 배경): 107.9 128.9 149.1 107.9 107.9 107.9 107.9 107.9 107.9 107.9 107.9 146.1 127.5 107.9 107.9 107.9 107.9 107.9 107.9 107.9 107.9 128.9 149.1 128.9 107.9 107.9 107.9 107.9 107.9 107.9 128.9 149.1 107.9 107.9

| 조건 | 측정값 | 기준 | 넘음 |
|---|---|---|---|
| 1. 밝기 변화폭 | 41.2 cd/m² (어두운 쪽 107.9) | ≥ 20 cd/m², 어두운 쪽 < 160 | 예 |
| 2. 깜빡임 횟수 (픽셀 기준) | 1초 최대 3.5번 | > 3번 | 예 |
| 3. 같은 방향 동시 전이 면적 | 최대 25.19% | > 25% | 예 |
| 최종: 1+2+3 모두 (화면 기준 깜빡임) | 1초 최대 3.5번 | > 3번이면 위험 | 예 |

### broadcast_30fps_01/f006l034 — 기대 pass, 우리 fail

- **원인: A. 여러 프레임에 나뉜 밝기 변화 (밝기 조건)** — l = 모든 깜빡임이 밝기 문턱 미달로 설계됨. 깜빡임마다 한쪽(오르막 또는 내리막)이 두 걸음 68.4↔87.9↔107.9 (19.5, 20.0 미만 — 정확히는 19.97)으로 나뉘어 있다. 합계 39.5 cd/m² 라서 우리 판정기는 매번 전이로 센다.
- 목록 CSV 의 설계 조건(dimension): `luminance`, 배경 휘도 87.9 cd/m²
- 패턴 `f006_fhd.png` (불투명 면적 25.1852%) × `rel_lum/l034_srgba.csv` 프레임별 휘도(cd/m², 괄호 = 투명이라 배경): 68.4 87.9 107.9 68.4 68.4 68.4 68.4 68.4 68.4 68.4 68.4 107.9 87.9 68.4 68.4 68.4 68.4 68.4 68.4 68.4 68.4 87.9 107.9 87.9 68.4 68.4 68.4 68.4 68.4 68.4 87.9 107.9 68.4 68.4

| 조건 | 측정값 | 기준 | 넘음 |
|---|---|---|---|
| 1. 밝기 변화폭 | 39.5 cd/m² (어두운 쪽 68.4) | ≥ 20 cd/m², 어두운 쪽 < 160 | 예 |
| 2. 깜빡임 횟수 (픽셀 기준) | 1초 최대 3.5번 | > 3번 | 예 |
| 3. 같은 방향 동시 전이 면적 | 최대 25.19% | > 25% | 예 |
| 최종: 1+2+3 모두 (화면 기준 깜빡임) | 1초 최대 3.5번 | > 3번이면 위험 | 예 |

### broadcast_30fps_inf01/f002c015_f001cs012 — 기대 pass, 우리 fail

- **원인: B. 서로 다른 영역의 엇박자 깜빡임을 하나로 합쳐 셈 (영역·동기 조건)** — 두 영역(f002·f001, 각 25.19%, 겹침 0%)이 각각 3번/s 로 깜빡인다(따로 판정하면 둘 다 3.0번/s, 안전). co012 의 깜빡임 하나(18~19프레임)만 c015 와 엇박자. 우리 screen_transitions 는 화면 전체에서 25% 넘는 전이를 한 줄의 상태 기계로 세므로, 엇박자 깜빡임 하나가 더해져 4번/s → 위험. pse 기대값은 영역마다 따로 센다.
- 목록 CSV 의 설계 조건(dimension): `count`, 배경 휘도 98.2 cd/m²
- 패턴 `f001_fhd.png` (불투명 면적 25.1852%) × `30fps_broadcast_interference/co012.csv` 프레임별 휘도(cd/m², 괄호 = 투명이라 배경): 30.0 50.8 30.0 30.0 30.0 30.0 30.0 30.0 30.0 30.0 30.0 50.8 30.0 30.0 30.0 30.0 30.0 30.0 50.8 30.0 30.0 30.0 30.0 30.0 30.0 30.0 30.0 30.0 30.0 30.0 30.0 50.8 30.0 30.0
- 패턴 `f002_fhd.png` (불투명 면적 25.1852%) × `rel_lum/c015_srgba.csv` 프레임별 휘도(cd/m², 괄호 = 투명이라 배경): 87.9 109.1 87.9 87.9 87.9 87.9 87.9 87.9 87.9 87.9 87.9 109.1 87.9 87.9 87.9 87.9 87.9 87.9 87.9 87.9 87.9 109.1 87.9 87.9 87.9 87.9 87.9 87.9 87.9 87.9 87.9 109.1 87.9 87.9
- 두 영역의 겹침 면적: 0.0000%

| 조건 | 측정값 | 기준 | 넘음 |
|---|---|---|---|
| 1. 밝기 변화폭 | 21.2 cd/m² (어두운 쪽 87.9) | ≥ 20 cd/m², 어두운 쪽 < 160 | 예 |
| 2. 깜빡임 횟수 (픽셀 기준) | 1초 최대 3번 | > 3번 | 아니오 |
| 3. 같은 방향 동시 전이 면적 | 최대 50.37% | > 25% | 예 |
| 최종: 1+2+3 모두 (화면 기준 깜빡임) | 1초 최대 4번 | > 3번이면 위험 | 예 |

### broadcast_30fps_inf01/f003y038_f005cs035 — 기대 pass, 우리 fail

- **원인: A. 여러 프레임에 나뉜 밝기 변화 (밝기 조건)** — 목표 패턴 y038 (f003 영역)의 하강 184.3→164.6→146.1 (걸음 19.7, 18.5 / 합계 38.2, 어두운 쪽 146.1 < 160). 픽셀 기준 횟수가 이미 3.5번이라 두 영역을 합친 효과가 아니라 A 원인.
- 목록 CSV 의 설계 조건(dimension): `luminance`, 배경 휘도 166.2 cd/m²
- 패턴 `f005_fhd.png` (불투명 면적 25.1852%) × `30fps_broadcast_interference/cs035.csv` 프레임별 휘도(cd/m², 괄호 = 투명이라 배경): 87.9 87.9 87.9 87.9 109.1 130.3 87.9 87.9 87.9 87.9 87.9 87.9 87.9 87.9 130.3 109.1 87.9 87.9 87.9 87.9 87.9 87.9 87.9 87.9 109.1 130.3 109.1 87.9 87.9 87.9 87.9 87.9 87.9 87.9
- 패턴 `f003_fhd.png` (불투명 면적 25.1852%) × `rel_lum/y038_srgba.csv` 프레임별 휘도(cd/m², 괄호 = 투명이라 배경): 146.1 166.2 187.7 146.1 146.1 146.1 146.1 146.1 146.1 146.1 146.1 184.3 164.6 146.1 146.1 146.1 146.1 146.1 146.1 146.1 146.1 166.2 187.7 166.2 146.1 146.1 146.1 146.1 146.1 146.1 166.2 187.7 146.1 146.1
- 두 영역의 겹침 면적: 7.5926%

| 조건 | 측정값 | 기준 | 넘음 |
|---|---|---|---|
| 1. 밝기 변화폭 | 42.3 cd/m² (어두운 쪽 87.9) | ≥ 20 cd/m², 어두운 쪽 < 160 | 예 |
| 2. 깜빡임 횟수 (픽셀 기준) | 1초 최대 3.5번 | > 3번 | 예 |
| 3. 같은 방향 동시 전이 면적 | 최대 25.19% | > 25% | 예 |
| 최종: 1+2+3 모두 (화면 기준 깜빡임) | 1초 최대 3.5번 | > 3번이면 위험 | 예 |

### broadcast_30fps_inf02/f001tf03_f003ico03 — 기대 pass, 우리 fail

- **원인: C. 기대값(JSON) 자체가 의심됨** — JSON 은 모든 표준 pass 인데, 같은 세트의 목록 CSV(broadcast_30fps_inf02.csv)는 pass=FALSE, dimension=FAIL 이다. tf03 = '실패하는 목표 패턴'(lum_flash_patterns/30fps_broadcast_interference/README.md). tf03 만 f001 영역에 따로 판정해도 3.5번/s (25.19%) → 위험. 즉 우리 결과는 패턴 설계·목록 CSV 와 일치하고 JSON 과만 다르다. (우리 판정의 4.5번/s 에는 B 원인으로 ico03 이 더해짐.) 이 불일치는 판정기 오류로 보기 어렵지만 숨기지 않고 센다.
- 목록 CSV 의 설계 조건(dimension): `FAIL`, 배경 휘도 20.4 cd/m²
- 패턴 `f003_fhd.png` (불투명 면적 25.1852%) × `30fps_broadcast_interference/ico03.csv` 프레임별 휘도(cd/m², 괄호 = 투명이라 배경): (20.4) 20.4 41.0 (20.4) (20.4) (20.4) (20.4) 41.0 20.4 (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) 20.4 41.0 20.4 (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) 20.4 41.0 (20.4)
- 패턴 `f001_fhd.png` (불투명 면적 25.1852%) × `30fps_broadcast_interference/tf03.csv` 프레임별 휘도(cd/m², 괄호 = 투명이라 배경): (20.4) 20.4 41.0 (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) 41.0 20.4 (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) 20.4 41.0 20.4 (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) 20.4 41.0 (20.4) (20.4)
- 두 영역의 겹침 면적: 6.8519%

| 조건 | 측정값 | 기준 | 넘음 |
|---|---|---|---|
| 1. 밝기 변화폭 | 20.6 cd/m² (어두운 쪽 20.4) | ≥ 20 cd/m², 어두운 쪽 < 160 | 예 |
| 2. 깜빡임 횟수 (픽셀 기준) | 1초 최대 4.5번 | > 3번 | 예 |
| 3. 같은 방향 동시 전이 면적 | 최대 43.52% | > 25% | 예 |
| 최종: 1+2+3 모두 (화면 기준 깜빡임) | 1초 최대 4.5번 | > 3번이면 위험 | 예 |

### broadcast_30fps_inf02/f002tc03_f004ics03 — 기대 pass, 우리 fail

- **원인: B. 서로 다른 영역의 엇박자 깜빡임을 하나로 합쳐 셈 (영역·동기 조건)** — tc03(f002)·ics03(f004) 각각 3번/s, 2프레임 이상 엇박자(따로 판정: 3.0, 3.0). 우리는 두 줄을 합쳐 6번/s. 두 영역이 겹치는 5.74% 픽셀은 실제로 6번/s 를 겪지만(픽셀 기준 6번) 면적이 25% 미만이다.
- 목록 CSV 의 설계 조건(dimension): `count`, 배경 휘도 20.4 cd/m²
- 패턴 `f004_fhd.png` (불투명 면적 25.1852%) × `30fps_broadcast_interference/ics03.csv` 프레임별 휘도(cd/m², 괄호 = 투명이라 배경): (20.4) (20.4) (20.4) (20.4) 20.4 41.0 (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) 41.0 20.4 (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) 20.4 41.0 20.4 (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4)
- 패턴 `f002_fhd.png` (불투명 면적 25.1852%) × `30fps_broadcast_interference/tc03.csv` 프레임별 휘도(cd/m², 괄호 = 투명이라 배경): (20.4) 20.4 41.0 (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) 41.0 20.4 (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) 20.4 41.0 20.4 (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) 20.4 41.0 (20.4)
- 두 영역의 겹침 면적: 5.7407%

| 조건 | 측정값 | 기준 | 넘음 |
|---|---|---|---|
| 1. 밝기 변화폭 | 20.6 cd/m² (어두운 쪽 20.4) | ≥ 20 cd/m², 어두운 쪽 < 160 | 예 |
| 2. 깜빡임 횟수 (픽셀 기준) | 1초 최대 6번 | > 3번 | 예 |
| 3. 같은 방향 동시 전이 면적 | 최대 25.19% | > 25% | 예 |
| 최종: 1+2+3 모두 (화면 기준 깜빡임) | 1초 최대 6번 | > 3번이면 위험 | 예 |

### broadcast_30fps_inf02/f006tc03_f001ico03 — 기대 pass, 우리 fail

- **원인: B. 서로 다른 영역의 엇박자 깜빡임을 하나로 합쳐 셈 (영역·동기 조건)** — tc03(f006)·ico03(f001) — ico03 은 깜빡임 하나만 엇박자. 합쳐 세서 4번/s. 겹침 9.26% (< 25%).
- 목록 CSV 의 설계 조건(dimension): `count`, 배경 휘도 20.4 cd/m²
- 패턴 `f001_fhd.png` (불투명 면적 25.1852%) × `30fps_broadcast_interference/ico03.csv` 프레임별 휘도(cd/m², 괄호 = 투명이라 배경): (20.4) 20.4 41.0 (20.4) (20.4) (20.4) (20.4) 41.0 20.4 (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) 20.4 41.0 20.4 (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) 20.4 41.0 (20.4)
- 패턴 `f006_fhd.png` (불투명 면적 25.1852%) × `30fps_broadcast_interference/tc03.csv` 프레임별 휘도(cd/m², 괄호 = 투명이라 배경): (20.4) 20.4 41.0 (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) 41.0 20.4 (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) 20.4 41.0 20.4 (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) 20.4 41.0 (20.4)
- 두 영역의 겹침 면적: 9.2593%

| 조건 | 측정값 | 기준 | 넘음 |
|---|---|---|---|
| 1. 밝기 변화폭 | 20.6 cd/m² (어두운 쪽 20.4) | ≥ 20 cd/m², 어두운 쪽 < 160 | 예 |
| 2. 깜빡임 횟수 (픽셀 기준) | 1초 최대 4번 | > 3번 | 예 |
| 3. 같은 방향 동시 전이 면적 | 최대 41.11% | > 25% | 예 |
| 최종: 1+2+3 모두 (화면 기준 깜빡임) | 1초 최대 4번 | > 3번이면 위험 | 예 |

### broadcast_30fps_inf02/f007ty03_f006ics03 — 기대 pass, 우리 fail

- **원인: B. 서로 다른 영역의 엇박자 깜빡임을 하나로 합쳐 셈 (영역·동기 조건)** — ty03(f007)은 깜빡임 하나가 문턱 미달(20.4→38.9→20.0: 18.5, 18.9)이라 혼자서는 3번/s 이하, ics03(f006)도 3번/s. 엇박자 두 영역을 합쳐 5.5번/s. 겹침 10.19% (< 25%).
- 목록 CSV 의 설계 조건(dimension): `luminance`, 배경 휘도 20.4 cd/m²
- 패턴 `f006_fhd.png` (불투명 면적 25.1852%) × `30fps_broadcast_interference/ics03.csv` 프레임별 휘도(cd/m², 괄호 = 투명이라 배경): (20.4) (20.4) (20.4) (20.4) 20.4 41.0 (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) 41.0 20.4 (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) 20.4 41.0 20.4 (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4)
- 패턴 `f007_fhd.png` (불투명 면적 25.1852%) × `30fps_broadcast_interference/ty03.csv` 프레임별 휘도(cd/m², 괄호 = 투명이라 배경): (20.4) 20.4 41.0 (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) 38.9 20.0 (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) 20.4 41.0 20.4 (20.4) (20.4) (20.4) (20.4) (20.4) (20.4) 20.4 41.0 (20.4) (20.4)
- 두 영역의 겹침 면적: 10.1852%

| 조건 | 측정값 | 기준 | 넘음 |
|---|---|---|---|
| 1. 밝기 변화폭 | 21.0 cd/m² (어두운 쪽 20.0) | ≥ 20 cd/m², 어두운 쪽 < 160 | 예 |
| 2. 깜빡임 횟수 (픽셀 기준) | 1초 최대 5.5번 | > 3번 | 예 |
| 3. 같은 방향 동시 전이 면적 | 최대 25.19% | > 25% | 예 |
| 최종: 1+2+3 모두 (화면 기준 깜빡임) | 1초 최대 5.5번 | > 3번이면 위험 | 예 |

## 영상별 결과

| 세트 | 영상 | 설계 조건 | 기대(ITU) | 우리 | 1초 최대 깜빡임 | 최대 면적 | 화면 전이 수 | 일치 |
|---|---|---|---|---|---|---|---|---|
| 30fps_alternating_01 | f001c037 | count | pass | pass | 3 | 25.1852% | 8 | ✅ |
| 30fps_alternating_01 | f001f037 | FAIL | fail | fail | 3.5 | 25.1852% | 8 | ✅ |
| 30fps_alternating_01 | f002c033 | count | pass | pass | 3 | 25.1852% | 8 | ✅ |
| 30fps_alternating_01 | f002c037 | count | pass | pass | 3 | 25.1852% | 8 | ✅ |
| 30fps_alternating_01 | f003c038 | count | pass | pass | 3 | 25.1852% | 8 | ✅ |
| 30fps_alternating_01 | f003f033 | FAIL | fail | fail | 3.5 | 25.1852% | 8 | ✅ |
| 30fps_alternating_01 | f004c032 | count | pass | pass | 3 | 25.1852% | 8 | ✅ |
| 30fps_alternating_01 | f004c034 | count | pass | pass | 3 | 25.1852% | 8 | ✅ |
| 30fps_alternating_01 | f005c031 | count | pass | pass | 3 | 25.1852% | 8 | ✅ |
| 30fps_alternating_01 | f005f034 | FAIL | fail | fail | 3.5 | 25.1852% | 8 | ✅ |
| 30fps_alternating_01 | f006c035 | count | pass | pass | 3 | 25.1852% | 8 | ✅ |
| 30fps_alternating_01 | f006f031 | FAIL | fail | fail | 3.5 | 25.1852% | 8 | ✅ |
| 30fps_alternating_01 | f007c032 | count | pass | pass | 3 | 25.1852% | 8 | ✅ |
| 30fps_alternating_01 | f007c038 | count | pass | pass | 3 | 25.1852% | 8 | ✅ |
| 30fps_alternating_01 | f008c035 | count | pass | pass | 3 | 25.1852% | 8 | ✅ |
| 30fps_alternating_01 | f008c036 | count | pass | pass | 3 | 25.1852% | 8 | ✅ |
| broadcast_30fps_01 | a001f008 | area | pass | pass | 0 | 24.8148% | 0 | ✅ |
| broadcast_30fps_01 | a002f002 | area | pass | pass | 0 | 24.8148% | 0 | ✅ |
| broadcast_30fps_01 | a003f031 | area | pass | pass | 0 | 24.8148% | 0 | ✅ |
| broadcast_30fps_01 | a004f026 | area | pass | pass | 0 | 24.8148% | 0 | ✅ |
| broadcast_30fps_01 | a005f015 | area | pass | pass | 0 | 24.8148% | 0 | ✅ |
| broadcast_30fps_01 | a006fr007 | area | pass | pass | 0 | 24.8148% | 0 | ✅ |
| broadcast_30fps_01 | a007fr014 | area | pass | pass | 0 | 24.8148% | 0 | ✅ |
| broadcast_30fps_01 | a008f013 | area | pass | pass | 0 | 24.8148% | 0 | ✅ |
| broadcast_30fps_01 | f001c001 | count | pass | pass | 3 | 25.1852% | 6 | ✅ |
| broadcast_30fps_01 | f001f005 | FAIL | fail | fail | 3.5 | 25.1852% | 7 | ✅ |
| broadcast_30fps_01 | f001l012 | luminance | pass | pass | 0 | 0.0000% | 0 | ✅ |
| broadcast_30fps_01 | f001y033 | luminance1 | pass | fail | 3.5 | 25.1852% | 8 | ❌ |
| broadcast_30fps_01 | f002cr015 | count | pass | pass | 3 | 25.1852% | 8 | ✅ |
| broadcast_30fps_01 | f002f038 | FAIL | fail | fail | 3.5 | 25.1852% | 8 | ✅ |
| broadcast_30fps_01 | f002l008 | luminance | pass | pass | 0 | 0.0000% | 0 | ✅ |
| broadcast_30fps_01 | f002y017 | luminance1 | pass | pass | 2.5 | 25.1852% | 6 | ✅ |
| broadcast_30fps_01 | f003c028 | count | pass | pass | 3 | 25.1852% | 7 | ✅ |
| broadcast_30fps_01 | f003f014 | FAIL | fail | fail | 3.5 | 25.1852% | 8 | ✅ |
| broadcast_30fps_01 | f003lr013 | luminance | pass | pass | 0 | 0.0000% | 0 | ✅ |
| broadcast_30fps_01 | f003y012 | luminance1 | pass | pass | 2.5 | 25.1852% | 6 | ✅ |
| broadcast_30fps_01 | f004c012 | count | pass | pass | 3 | 25.1852% | 8 | ✅ |
| broadcast_30fps_01 | f004f013 | FAIL | fail | fail | 3.5 | 25.1852% | 8 | ✅ |
| broadcast_30fps_01 | f004l025 | luminance | pass | pass | 0 | 0.0000% | 0 | ✅ |
| broadcast_30fps_01 | f004y024 | luminance1 | pass | pass | 2.5 | 25.1852% | 5 | ✅ |
| broadcast_30fps_01 | f005cr006 | count | pass | pass | 3 | 25.1852% | 6 | ✅ |
| broadcast_30fps_01 | f005f021 | FAIL | fail | fail | 3.5 | 25.1852% | 7 | ✅ |
| broadcast_30fps_01 | f005l006 | luminance | pass | pass | 0 | 0.0000% | 0 | ✅ |
| broadcast_30fps_01 | f005y036 | luminance1 | pass | fail | 3.5 | 25.1852% | 8 | ❌ |
| broadcast_30fps_01 | f006c004 | count | pass | pass | 3 | 25.1852% | 6 | ✅ |
| broadcast_30fps_01 | f006fr002 | FAIL | fail | fail | 3.5 | 25.1852% | 7 | ✅ |
| broadcast_30fps_01 | f006l034 | luminance | pass | fail | 3.5 | 25.1852% | 8 | ❌ |
| broadcast_30fps_01 | f006y001 | luminance1 | pass | pass | 2.5 | 25.1852% | 5 | ✅ |
| broadcast_30fps_01 | f007c033 | count | pass | pass | 3 | 25.1852% | 8 | ✅ |
| broadcast_30fps_01 | f007f007 | FAIL | fail | fail | 3.5 | 25.1852% | 7 | ✅ |
| broadcast_30fps_01 | f007lr007 | luminance | pass | pass | 0 | 0.0000% | 0 | ✅ |
| broadcast_30fps_01 | f007y028 | luminance1 | pass | pass | 2.5 | 25.1852% | 5 | ✅ |
| broadcast_30fps_01 | f008c017 | count | pass | pass | 3 | 25.1852% | 8 | ✅ |
| broadcast_30fps_01 | f008fr016 | FAIL | fail | fail | 3.5 | 25.1852% | 8 | ✅ |
| broadcast_30fps_01 | f008l011 | luminance | pass | pass | 0 | 0.0000% | 0 | ✅ |
| broadcast_30fps_01 | f008y005 | luminance1 | pass | pass | 2.5 | 25.1852% | 5 | ✅ |
| broadcast_30fps_inf01 | a006f026_f003co023 | area | pass | pass | 2 | 43.7037% | 5 | ✅ |
| broadcast_30fps_inf01 | a008f014_f005cs015 | area | pass | pass | 0 | 24.8148% | 0 | ✅ |
| broadcast_30fps_inf01 | f001c024_f002cs027 | count | pass | pass | 3 | 25.1852% | 7 | ✅ |
| broadcast_30fps_inf01 | f002c015_f001cs012 | count | pass | fail | 4 | 50.3704% | 10 | ❌ |
| broadcast_30fps_inf01 | f003y038_f005cs035 | luminance | pass | fail | 3.5 | 25.1852% | 8 | ❌ |
| broadcast_30fps_inf01 | f004f011_f003cs018 | FAIL | fail | fail | 3.5 | 25.1852% | 8 | ✅ |
| broadcast_30fps_inf01 | f005f027_f007co026 | FAIL | fail | fail | 3.5 | 41.2963% | 7 | ✅ |
| broadcast_30fps_inf01 | f006f029_f004co021 | FAIL | fail | fail | 3.5 | 44.0741% | 7 | ✅ |
| broadcast_30fps_inf01 | f007y012_f006co013 | luminance | pass | pass | 2.5 | 40.1852% | 6 | ✅ |
| broadcast_30fps_inf01 | f008f036_f004cs034 | FAIL | fail | fail | 3.5 | 25.1852% | 8 | ✅ |
| broadcast_30fps_inf02 | a002tf03_f006ics03 | area | pass | pass | 3 | 42.9630% | 8 | ✅ |
| broadcast_30fps_inf02 | a003tf01_f008ico01 | area | pass | pass | 3 | 46.2963% | 8 | ✅ |
| broadcast_30fps_inf02 | f001tf03_f003ico03 | FAIL | pass | fail | 4.5 | 43.5185% | 10 | ❌ |
| broadcast_30fps_inf02 | f002tc03_f004ics03 | count | pass | fail | 6 | 25.1852% | 14 | ❌ |
| broadcast_30fps_inf02 | f003tf01_f005ico01 | FAIL | pass | pass | 0 | 0.0000% | 0 | ✅ |
| broadcast_30fps_inf02 | f004tc01_f002ics01 | count | pass | pass | 0 | 0.0000% | 0 | ✅ |
| broadcast_30fps_inf02 | f005ty01_f007ico01 | luminance | pass | pass | 0 | 0.0000% | 0 | ✅ |
| broadcast_30fps_inf02 | f006tc03_f001ico03 | count | pass | fail | 4 | 41.1111% | 10 | ❌ |
| broadcast_30fps_inf02 | f007ty03_f006ics03 | luminance | pass | fail | 5.5 | 25.1852% | 12 | ❌ |
| broadcast_30fps_inf02 | f008tc01_f001ics01 | count | pass | pass | 0 | 0.0000% | 0 | ✅ |
| scene_30fps_0p1rl | a02f008 | area | pass | pass | 2 | 79.8611% | 4 | ✅ |
| scene_30fps_0p1rl | a04f021 | area | pass | pass | 2 | 79.8611% | 4 | ✅ |
| scene_30fps_0p1rl | f01c022 | count | pass | pass | 1.5 | 80.0347% | 4 | ✅ |
| scene_30fps_0p1rl | f01f026 | FAIL | pass | pass | 2 | 80.0347% | 4 | ✅ |
| scene_30fps_0p1rl | f02f013 | FAIL | pass | pass | 2 | 80.0347% | 4 | ✅ |
| scene_30fps_0p1rl | f03f009 | FAIL | pass | pass | 2 | 80.0347% | 4 | ✅ |
| scene_30fps_0p1rl | f03f012_a03add012 | area | pass | pass | 2 | 80.0347% | 4 | ✅ |
| scene_30fps_0p1rl | f04f018_a04add018 | area | pass | pass | 2 | 80.0347% | 4 | ✅ |
| scene_30fps_0p1rl | f04y014 | luminance | pass | pass | 1 | 80.0347% | 2 | ✅ |
| scene_30fps_0p1rl | f05c005 | count | pass | pass | 1.5 | 80.0347% | 4 | ✅ |
| scene_30fps_0p1rl | f05f024 | FAIL | pass | pass | 2 | 80.0347% | 4 | ✅ |
| scene_30fps_0p1rl | f05y027 | luminance | pass | pass | 2 | 80.0347% | 4 | ✅ |
| trace24_30fps_01 | a001fr003 | area | pass | pass | 0 | 2.0370% | 0 | ✅ |
| trace24_30fps_01 | a002f036 | area | pass | pass | 0 | 2.0370% | 0 | ✅ |
| trace24_30fps_01 | a003f012 | area | pass | pass | 0 | 2.0370% | 0 | ✅ |
| trace24_30fps_01 | a004f002 | area | pass | pass | 0 | 2.0370% | 0 | ✅ |
| trace24_30fps_01 | a005fr008 | area | pass | pass | 0 | 2.0370% | 0 | ✅ |
| trace24_30fps_01 | a006fr011 | area | pass | pass | 0 | 2.0370% | 0 | ✅ |
| trace24_30fps_01 | a007f004 | area | pass | pass | 0 | 2.0370% | 0 | ✅ |
| trace24_30fps_01 | a008f021 | area | pass | pass | 0 | 2.0370% | 0 | ✅ |
| trace24_30fps_01 | a009fr015 | area | pass | pass | 0 | 2.0370% | 0 | ✅ |
| trace24_30fps_01 | a010f037 | area | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | a011f023 | area | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | a012f014 | area | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | f001cr007 | count | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | f001f034 | FAIL | pass - small area | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | f001l007 | luminance | pass | pass | 0 | 0.0000% | 0 | ✅ |
| trace24_30fps_01 | f002fr005 | FAIL | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | f002lr001 | luminance | pass | pass | 0 | 0.0000% | 0 | ✅ |
| trace24_30fps_01 | f003cr011 | count | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | f003f024 | FAIL | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | f003l032 | luminance | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | f003lr007 | luminance | pass | pass | 0 | 0.0000% | 0 | ✅ |
| trace24_30fps_01 | f003y017 | luminance1 | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | f004c027 | count | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | f004f001 | FAIL | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | f004y022 | luminance1 | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | f005fr002 | FAIL | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | f005lr003 | luminance | pass | pass | 0 | 0.0000% | 0 | ✅ |
| trace24_30fps_01 | f005y031 | luminance1 | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | f006c035 | count | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | f006f017 | FAIL | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | f006lr018 | luminance | pass | pass | 0 | 0.0000% | 0 | ✅ |
| trace24_30fps_01 | f006y023 | luminance1 | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | f007c023 | count | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | f007cr014 | count | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | f007f003 | FAIL | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | f008cr003 | count | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | f008fr016 | FAIL | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | f008l024 | luminance | pass | pass | 0 | 0.0000% | 0 | ✅ |
| trace24_30fps_01 | f008lr014 | luminance | pass | pass | 0 | 0.0000% | 0 | ✅ |
| trace24_30fps_01 | f009c011 | count | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | f009cr005 | count | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | f009f013 | FAIL | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | f009lr016 | luminance | pass | pass | 0 | 0.0000% | 0 | ✅ |
| trace24_30fps_01 | f010c032 | count | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | f010f038 | FAIL | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | f010l021 | luminance | pass | pass | 0 | 0.0000% | 0 | ✅ |
| trace24_30fps_01 | f010y004 | luminance1 | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | f011cr016 | count | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | f011fr011 | FAIL | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | f011l038 | luminance | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | f012c004 | count | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | f012f025 | FAIL | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_01 | f012l013 | luminance | pass | pass | 0 | 0.0000% | 0 | ✅ |
| trace24_30fps_01 | f012y035 | luminance1 | pass | pass | 0 | 2.2222% | 0 | ✅ |
| trace24_30fps_area_combo01 | 01m50c004_01n50f003 | count | pass | pass | 0 | 2.1605% | 0 | ✅ |
| trace24_30fps_area_combo01 | 01m70f018_01n30f016 | FAIL | pass | pass | 0 | 2.1605% | 0 | ✅ |
| trace24_30fps_area_combo01 | 01m70f024_01n25f025 | area | pass | pass | 0 | 2.0062% | 0 | ✅ |
| trace24_30fps_area_combo01 | 01m90f008_01n10y001 | luminance | pass | pass | 0 | 2.1605% | 0 | ✅ |
| trace24_30fps_area_combo01 | 02m45f002_02n50f006 | area | pass | pass | 0 | 2.0062% | 0 | ✅ |
| trace24_30fps_area_combo01 | 02m50f001_02n50f007 | FAIL | pass | pass | 0 | 2.1605% | 0 | ✅ |
| trace24_30fps_area_combo01 | 02m70f021_02n30c028 | count | pass | pass | 0 | 2.1605% | 0 | ✅ |
| trace24_30fps_area_combo01 | 02m90f024_02n10f022 | FAIL | pass | pass | 0 | 2.1605% | 0 | ✅ |
| trace24_30fps_area_combo01 | 03m50f024_03n50y022 | luminance | pass | pass | 0 | 2.1605% | 0 | ✅ |
| trace24_30fps_area_combo01 | 03m70f035_03n30f033 | FAIL | pass | pass | 0 | 2.1605% | 0 | ✅ |
| trace24_30fps_area_combo01 | 03m85f038_03n10f031 | area | pass | pass | 0 | 2.0062% | 0 | ✅ |
| trace24_30fps_area_combo01 | 03m90f006_03n10f004 | FAIL | pass | pass | 0 | 2.1605% | 0 | ✅ |
| trace24_30fps_area_combo01 | 04m50fr013_04n50fr015 | FAIL | pass | pass | 0 | 2.1605% | 0 | ✅ |
| trace24_30fps_area_combo01 | 04m65fr003_04n30fr007 | area | pass | pass | 0 | 2.0062% | 0 | ✅ |
| trace24_30fps_area_combo01 | 04m70f012_04n30f011 | FAIL | pass | pass | 0 | 2.1605% | 0 | ✅ |
| trace24_30fps_area_combo01 | 04m90c032_04n10f037 | count | pass | pass | 0 | 2.1605% | 0 | ✅ |
| trace24_30fps_area_combo01 | 05m50y016_05n50f013 | luminance | pass | pass | 0 | 2.1605% | 0 | ✅ |
| trace24_30fps_area_combo01 | 05m70f027_05n30f028 | FAIL | pass | pass | 0 | 2.1605% | 0 | ✅ |
| trace24_30fps_area_combo01 | 05m90f013_05n05f016 | area | pass | pass | 0 | 2.0062% | 0 | ✅ |
| trace24_30fps_area_combo01 | 05m90fr007_05n10fr004 | FAIL | pass | pass | 0 | 2.1605% | 0 | ✅ |
| trace24_30fps_inf01 | a003f017a_a008f017 | area_area | pass | pass | 0 | 4.0741% | 0 | ✅ |
| trace24_30fps_inf01 | a004f014a_f002f014 | FAIL | pass | pass | 0 | 4.2593% | 0 | ✅ |
| trace24_30fps_inf01 | a004f017a_f006c017 | area_count | pass | pass | 0 | 4.2593% | 0 | ✅ |
| trace24_30fps_inf01 | a008f014a_f012y014 | area_lum | pass | pass | 0 | 4.2593% | 0 | ✅ |
| trace24_30fps_inf01 | f002f014a_f005c014 | FAIL | pass | pass | 0 | 4.4444% | 0 | ✅ |
| trace24_30fps_inf01 | f002y016a_f011cr016 | lum_count | pass | pass | 0 | 4.4444% | 0 | ✅ |
| trace24_30fps_inf01 | f003cr013a_a004fr013 | count_area | pass | pass | 0 | 4.2593% | 0 | ✅ |
| trace24_30fps_inf01 | f003f011a_f008y011 | FAIL | pass | pass | 0 | 4.4444% | 0 | ✅ |
| trace24_30fps_inf01 | f003y013a_f004fr013 | FAIL | pass | pass | 0 | 4.4444% | 0 | ✅ |
| trace24_30fps_inf01 | f004cr016a_f010fr016 | FAIL | pass | pass | 0 | 4.4444% | 0 | ✅ |
| trace24_30fps_inf01 | f007f011a_f009f011 | FAIL | pass | pass | 0 | 4.4444% | 0 | ✅ |
| trace24_30fps_inf01 | f007fr016a_a001fr016 | FAIL | pass | pass | 0 | 4.2593% | 0 | ✅ |
| trace24_30fps_inf01 | f007y011a_f001y011 | lum_lum | pass | pass | 0 | 4.4444% | 0 | ✅ |
| trace24_30fps_inf01 | f008c017a_f003y017 | count_lum | pass | pass | 0 | 4.4444% | 0 | ✅ |
| trace24_30fps_inf01 | f008y011a_a012f011 | lum_area | pass | pass | 0 | 4.4444% | 0 | ✅ |
| trace24_30fps_inf01 | f011cr013a_f007cr013 | count_count | pass | pass | 0 | 4.4444% | 0 | ✅ |
| trace24_30fps_sync01 | 01m50ff032_01n50f032 | area | pass | pass | 0 | 2.1605% | 0 | ✅ |
| trace24_30fps_sync01 | 01m70f018_01n30f018 | FAIL | pass | pass | 0 | 2.1605% | 0 | ✅ |
| trace24_30fps_sync01 | 02m50f024_02n50f024 | FAIL | pass | pass | 0 | 2.1605% | 0 | ✅ |
| trace24_30fps_sync01 | 02m90ff015_02n10f015 | area | pass | pass | 0 | 2.1605% | 0 | ✅ |
| trace24_30fps_sync01 | 03m90f006_03n10f006 | FAIL | pass | pass | 0 | 2.1605% | 0 | ✅ |
| trace24_30fps_sync01 | 04m70f021_04n30ff021 | area | pass | pass | 0 | 2.1605% | 0 | ✅ |
| trace24_30fps_sync01 | 05m70f007_05n30ff007 | FAIL | pass | pass | 0 | 2.1605% | 0 | ✅ |
| trace24_30fps_sync01 | 05m90f033_05n10f033 | area | pass | pass | 0 | 2.1605% | 0 | ✅ |
| wcagc_30fps_area01 | a002f038 | area | pass | pass | 0 | 1.0031% | 0 | ✅ |
| wcagc_30fps_area01 | a005f026 | area | pass | pass | 0 | 1.0283% | 0 | ✅ |
| wcagc_30fps_area01 | a007fr001 | area | pass | pass | 0 | 1.0380% | 0 | ✅ |
| wcagc_30fps_area01 | a009fr013 | area | pass | pass | 0 | 1.0368% | 0 | ✅ |
| wcagc_30fps_area01 | a010f005 | area | pass | pass | 0 | 1.1112% | 0 | ✅ |
| wcagc_30fps_area01 | a011f014 | area | pass | pass | 0 | 1.0263% | 0 | ✅ |
| wcagc_30fps_area01 | f002f038 | FAIL | pass | pass | 0 | 1.0802% | 0 | ✅ |
| wcagc_30fps_area01 | f005f026 | FAIL | pass | pass | 0 | 1.1073% | 0 | ✅ |
| wcagc_30fps_area01 | f007fr001 | FAIL | pass | pass | 0 | 1.1171% | 0 | ✅ |
| wcagc_30fps_area01 | f009fr013 | FAIL | pass | pass | 0 | 1.1167% | 0 | ✅ |
| wcagc_30fps_area01 | f010f005 | FAIL | pass | pass | 0 | 1.1112% | 0 | ✅ |
| wcagc_30fps_area01 | f011f014 | FAIL | pass | pass | 0 | 1.0263% | 0 | ✅ |
| wcagc_30fps_area02 | a001f011 | area | pass | pass | 0 | 1.0031% | 0 | ✅ |
| wcagc_30fps_area02 | a003f027 | area | pass | pass | 0 | 1.0301% | 0 | ✅ |
| wcagc_30fps_area02 | a004fr006 | area | pass | pass | 0 | 1.0321% | 0 | ✅ |
| wcagc_30fps_area02 | a006f038 | area | pass | pass | 0 | 1.0499% | 0 | ✅ |
| wcagc_30fps_area02 | a008f002 | area | pass | pass | 0 | 1.0361% | 0 | ✅ |
| wcagc_30fps_area02 | a012fr014 | area | pass | pass | 0 | 1.0321% | 0 | ✅ |
| wcagc_30fps_area02 | f001f011 | FAIL | pass | pass | 0 | 1.0802% | 0 | ✅ |
| wcagc_30fps_area02 | f003f027 | FAIL | pass | pass | 0 | 1.1073% | 0 | ✅ |
| wcagc_30fps_area02 | f004fr006 | FAIL | pass | pass | 0 | 1.1112% | 0 | ✅ |
| wcagc_30fps_area02 | f006f038 | FAIL | pass | pass | 0 | 1.1290% | 0 | ✅ |
| wcagc_30fps_area02 | f008f002 | FAIL | pass | pass | 0 | 1.1152% | 0 | ✅ |
| wcagc_30fps_area02 | f012fr014 | FAIL | pass | pass | 0 | 1.0321% | 0 | ✅ |

## 다시 만들기

```
.venv\Scripts\python.exe scripts\validate_rules_pse.py --pse-root <pse-test-media 경로> --sets all-luminance --ffv1-check 3
```
