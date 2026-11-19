# 7종 위험 라벨 합성 — d3 / val

> 2026-11-20 08:37 · `scripts/make_hazard7_dataset.py` · 8 s · 24 fps · 768×432 · mp4(crf 12) 로 저장 후 다시 디코드해 판별 · 정답 = 팀원 판별기(partial B2) · 오류 0개

## 목표 라벨 적중 (의도 hazard → 그 라벨이 나옴 / 의도 near → 그 라벨이 안 나옴)

**적중 = 합성으로 새로 생긴 라벨**(같은 바탕만 판별한 결과를 프레임 단위로 뺌, 무늬는 0.5 s 이상). 괄호 = 바탕 빼기 전.

바탕만 판별했을 때 라벨이 나온 표본: 81/112 (정적 공간 패턴 81)

| 종류 → 목표 라벨 | hazard: 적중 | near: 그 라벨 새로 안 생김 | hazard 에서 새로 생긴 다른 라벨 | 우리 판정기 (hazard 중 위험) |
|---|---|---|---|---|
| lum_flash → 휘도 플래시 | 8/8 (8) | 8/8 (8) | 정적 공간 패턴 1 | 8/8 (lum) |
| intersection → 교집합 | 7/8 (7) | 8/8 (8) | 휘도 플래시 3, 빠른 장면 전환 1 | 8/8 (lum) |
| scene → 빠른 장면 전환 | 8/8 (8) | 8/8 (8) | – | 0/8 (lum) |
| red_flash → 적색 플래시 | 5/8 (5) | 8/8 (8) | – | 5/8 (red) |
| lum_ext → 휘도 연장 | 7/8 (7) | 8/8 (8) | – | 7/8 (ext_lum) |
| red_ext → 적색 연장 | 3/8 (3) | 8/8 (8) | – | 2/8 (ext_red) |
| pattern → 정적 공간 패턴 | 7/8 (8) | 5/8 (0) | – | – (줄무늬 판정은 느려 생략) |

## 마스크 방식별 (hazard, 적중)

| 마스크 | 개수 | 적중 |
|---|---|---|
| anchored | 6 | 5/6 |
| drift | 9 | 6/9 |
| full | 16 | 15/16 |
| multi | 1 | 1/1 |
| perspective | 4 | 3/4 |
| spot | 9 | 6/9 |
| static | 11 | 9/11 |

## near 의 방식별 (새로 생긴 라벨)

| 종류 | near 방식 | 개수 | 나온 라벨 |
|---|---|---|---|
| lum_flash | area | 2 | 정적 공간 패턴 1 |
| lum_flash | weak | 6 | 정적 공간 패턴 1 |
| intersection | area | 6 | 휘도 플래시 6 |
| intersection | slow | 2 | 빠른 장면 전환 2 |
| scene | area | 5 | 새로 생긴 라벨 없음 |
| scene | slow | 3 | 새로 생긴 라벨 없음 |
| red_flash | area | 2 | 새로 생긴 라벨 없음 |
| red_flash | slow | 4 | 새로 생긴 라벨 없음 |
| red_flash | weak | 2 | 새로 생긴 라벨 없음 |
| lum_ext | short | 5 | 정적 공간 패턴 1 |
| lum_ext | slow | 3 | 새로 생긴 라벨 없음 |
| red_ext | short | 4 | 정적 공간 패턴 1 |
| red_ext | slow | 4 | 새로 생긴 라벨 없음 |
| pattern | area | 2 | 정적 공간 패턴 2 |
| pattern | cpd | 5 | 정적 공간 패턴 1 |
| pattern | short | 1 | 새로 생긴 라벨 없음 |

## 무늬 종류별 (pattern, hazard)

| 무늬 | 개수 | 정적 공간 패턴 적중 |
|---|---|---|
| straight | 3 | 3/3 |
| concentric | 1 | 1/1 |
| spiral | 1 | 1/1 |
| chirp | 1 | 0/1 |
| wavy | 2 | 2/2 |

## 놓친 hazard 표본 (레시피 요약)

- `val_intersection_hazard_006` 새로 생긴 라벨 ['빠른 장면 전환'] · mode=gain, mask_mode=full, area=1.0, halves=[1, 2, 3], dur=1.642, sigma=0.7
- `val_lum_ext_hazard_005` 새로 생긴 라벨 없음 · mode=gain, mask_mode=spot, area=0.311, halves=[4, 5, 6], dur=6.484, delta_cd=95.798, sigma=0.7
- `val_pattern_hazard_005` 새로 생긴 라벨 없음 · mask_mode=perspective, pattern=chirp, area=0.526, dur=3.677, cpd=1.654, lo=9.757, hi=81.915, sigma=0.0
- `val_red_ext_hazard_000` 새로 생긴 라벨 없음 · mode=tint, mask_mode=drift, area=0.438, halves=[4, 5, 6], dur=7.401, sat=0.919, sigma=0.7
- `val_red_ext_hazard_002` 새로 생긴 라벨 없음 · mode=tint, mask_mode=static, area=0.502, halves=[4, 5, 6], dur=6.584, sat=0.894, sigma=0.7
- `val_red_ext_hazard_003` 새로 생긴 라벨 없음 · mode=tint, mask_mode=drift, area=0.35, halves=[4, 5, 6], dur=6.166, sat=0.923, sigma=0.7
- `val_red_ext_hazard_004` 새로 생긴 라벨 없음 · mode=tint, mask_mode=anchored, area=0.564, halves=[4, 5, 6], dur=6.259, sat=0.879, sigma=0.0
- `val_red_ext_hazard_005` 새로 생긴 라벨 없음 · mode=tint, mask_mode=static, area=0.685, halves=[4, 5, 6], dur=6.519, sat=0.946, sigma=0.7
- `val_red_flash_hazard_004` 새로 생긴 라벨 없음 · mode=tint, mask_mode=spot, area=0.365, halves=[1, 2, 3], dur=2.746, sat=0.877, sigma=0.4
- `val_red_flash_hazard_006` 새로 생긴 라벨 없음 · mode=tint, mask_mode=drift, area=0.433, halves=[1, 2, 3], dur=3.986, sat=0.936, sigma=0.7
- `val_red_flash_hazard_007` 새로 생긴 라벨 없음 · mode=tint, mask_mode=spot, area=0.573, halves=[1, 2, 3], dur=3.978, sat=0.887, sigma=0.7
