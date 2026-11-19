# 7종 위험 라벨 합성 — d1 / val

> 2026-11-20 07:31 · `scripts/make_hazard7_dataset.py` · 8 s · 24 fps · 768×432 · mp4(crf 12) 로 저장 후 다시 디코드해 판별 · 정답 = 팀원 판별기(partial B2) · 오류 0개

## 목표 라벨 적중 (의도 hazard → 그 라벨이 나옴 / 의도 near → 그 라벨이 안 나옴)

| 종류 → 목표 라벨 | hazard: 적중 | near: 그 라벨 없음 | hazard 에 함께 나온 다른 라벨 | 우리 판정기 (hazard 중 위험) |
|---|---|---|---|---|
| lum_flash → 휘도 플래시 | 4/6 | 3/6 | 정적 공간 패턴 5, 교집합 2 | 6/6 (lum) |
| intersection → 교집합 | 6/6 | 5/6 | 정적 공간 패턴 2, 휘도 플래시 2 | 6/6 (lum) |
| scene → 빠른 장면 전환 | 6/6 | 3/6 | 정적 공간 패턴 1 | 0/6 (lum) |
| red_flash → 적색 플래시 | 3/6 | 6/6 | 정적 공간 패턴 3 | 3/6 (red) |
| lum_ext → 휘도 연장 | 6/6 | 6/6 | 정적 공간 패턴 1 | 6/6 (ext_lum) |
| red_ext → 적색 연장 | 1/6 | 6/6 | 정적 공간 패턴 5 | 1/6 (ext_red) |
| pattern → 정적 공간 패턴 | 6/6 | 0/6 | – | – (줄무늬 판정은 느려 생략) |

## near 의 방식별 (판별기가 붙인 라벨)

| 종류 | near 방식 | 개수 | 나온 라벨 |
|---|---|---|---|
| lum_flash | area | 1 | 정적 공간 패턴 1 |
| lum_flash | weak | 5 | 휘도 플래시 3, 정적 공간 패턴 2, 교집합 1 |
| intersection | area | 5 | 휘도 플래시 4, 정적 공간 패턴 1, 교집합 1 |
| intersection | slow | 1 | 빠른 장면 전환 1 |
| scene | area | 4 | 빠른 장면 전환 3, 정적 공간 패턴 1 |
| scene | slow | 2 | 정적 공간 패턴 1 |
| red_flash | area | 2 | 없음(안전) |
| red_flash | slow | 3 | 정적 공간 패턴 2 |
| red_flash | weak | 1 | 정적 공간 패턴 1 |
| lum_ext | short | 3 | 정적 공간 패턴 1, 빠른 장면 전환 1 |
| lum_ext | slow | 3 | 정적 공간 패턴 3 |
| red_ext | short | 3 | 정적 공간 패턴 1 |
| red_ext | slow | 3 | 정적 공간 패턴 3 |
| pattern | area | 1 | 정적 공간 패턴 1 |
| pattern | cpd | 4 | 정적 공간 패턴 4 |
| pattern | short | 1 | 정적 공간 패턴 1 |

## 무늬 종류별 (pattern, hazard)

| 무늬 | 개수 | 정적 공간 패턴 적중 |
|---|---|---|
| straight | 2 | 2/2 |
| concentric | 1 | 1/1 |
| chirp | 1 | 1/1 |
| wavy | 2 | 2/2 |

## 놓친 hazard 표본 (레시피 요약)

- `val_lum_flash_hazard_001` 나온 라벨 ['교집합', '정적 공간 패턴'] · mode=add, area=0.71, halves=[1, 2, 3], dur=2.351, delta_cd=53.337, sigma=0.0
- `val_lum_flash_hazard_002` 나온 라벨 ['교집합', '정적 공간 패턴'] · mode=gain, area=0.658, halves=[1, 2, 3], dur=2.506, delta_cd=120.639, sigma=0.0
- `val_red_flash_hazard_002` 나온 라벨 ['정적 공간 패턴'] · mode=tint, area=0.641, halves=[1, 2, 3], dur=3.575, sat=0.878, sigma=0.4
- `val_red_flash_hazard_004` 나온 라벨 없음 · mode=tint, area=0.385, halves=[1, 2, 3], dur=2.746, sat=0.877, sigma=0.4
- `val_red_flash_hazard_005` 나온 라벨 ['정적 공간 패턴'] · mode=tint, area=0.368, halves=[1, 2, 3], dur=1.634, sat=0.945, sigma=0.7
- `val_red_ext_hazard_000` 나온 라벨 ['정적 공간 패턴'] · mode=tint, area=0.438, halves=[4, 5, 6], dur=7.401, sat=0.919, sigma=0.7
- `val_red_ext_hazard_001` 나온 라벨 ['정적 공간 패턴'] · mode=tint, area=0.349, halves=[4, 5, 6], dur=7.399, sat=0.884, sigma=0.0
- `val_red_ext_hazard_003` 나온 라벨 ['정적 공간 패턴'] · mode=tint, area=0.35, halves=[4, 5, 6], dur=6.166, sat=0.923, sigma=0.7
- `val_red_ext_hazard_004` 나온 라벨 ['정적 공간 패턴'] · mode=tint, area=0.564, halves=[4, 5, 6], dur=6.259, sat=0.879, sigma=0.0
- `val_red_ext_hazard_005` 나온 라벨 ['정적 공간 패턴'] · mode=tint, area=0.685, halves=[4, 5, 6], dur=6.519, sat=0.946, sigma=0.7
