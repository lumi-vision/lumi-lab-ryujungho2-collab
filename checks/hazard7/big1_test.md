# 7종 위험 라벨 합성 — big1 / test

> 2026-11-20 17:41 · `scripts/make_hazard7_dataset.py` · 8 s · 24 fps · 768×432 · mp4(crf 12) 로 저장 후 다시 디코드해 판별 · 정답 = 팀원 판별기(partial B2) · 오류 0개

## 목표 라벨 적중 (의도 hazard → 그 라벨이 나옴 / 의도 near → 그 라벨이 안 나옴)

**적중 = 합성으로 새로 생긴 라벨**(같은 바탕만 판별한 결과를 프레임 단위로 뺌, 무늬는 0.5 s 이상). 괄호 = 바탕 빼기 전.

바탕만 판별했을 때 라벨이 나온 표본: 44/140 (정적 공간 패턴 44)

| 종류 → 목표 라벨 | hazard: 적중 | near: 그 라벨 새로 안 생김 | hazard 에서 새로 생긴 다른 라벨 | 우리 판정기 (hazard 중 위험) |
|---|---|---|---|---|
| lum_flash → 휘도 플래시 | 9/10 (9) | 10/10 (10) | – | 10/10 (lum) |
| intersection → 교집합 | 10/10 (10) | 10/10 (10) | – | 10/10 (lum) |
| scene → 빠른 장면 전환 | 10/10 (10) | 10/10 (10) | – | 0/10 (lum) |
| red_flash → 적색 플래시 | 8/10 (8) | 9/10 (9) | 휘도 플래시 3 | 8/10 (red) |
| lum_ext → 휘도 연장 | 8/10 (8) | 10/10 (10) | 빠른 장면 전환 1 | 8/10 (ext_lum) |
| red_ext → 적색 연장 | 6/10 (6) | 10/10 (10) | 휘도 연장 2 | 6/10 (ext_red) |
| pattern → 정적 공간 패턴 | 8/10 (10) | 5/10 (3) | – | – (줄무늬 판정은 느려 생략) |

## 마스크 방식별 (hazard, 적중)

| 마스크 | 개수 | 적중 |
|---|---|---|
| anchored | 22 | 16/22 |
| drift | 6 | 6/6 |
| full | 20 | 20/20 |
| multi | 4 | 3/4 |
| perspective | 1 | 1/1 |
| spot | 8 | 5/8 |
| static | 9 | 8/9 |

## near 의 방식별 (새로 생긴 라벨)

| 종류 | near 방식 | 개수 | 나온 라벨 |
|---|---|---|---|
| lum_flash | area | 2 | 정적 공간 패턴 1 |
| lum_flash | slow | 6 | 새로 생긴 라벨 없음 |
| lum_flash | weak | 2 | 새로 생긴 라벨 없음 |
| intersection | area | 3 | 휘도 플래시 3 |
| intersection | slow | 7 | 빠른 장면 전환 7 |
| scene | area | 5 | 새로 생긴 라벨 없음 |
| scene | slow | 5 | 새로 생긴 라벨 없음 |
| red_flash | area | 2 | 적색 플래시 1 |
| red_flash | slow | 5 | 새로 생긴 라벨 없음 |
| red_flash | weak | 3 | 새로 생긴 라벨 없음 |
| lum_ext | short | 5 | 새로 생긴 라벨 없음 |
| lum_ext | slow | 5 | 새로 생긴 라벨 없음 |
| red_ext | short | 4 | 새로 생긴 라벨 없음 |
| red_ext | slow | 6 | 새로 생긴 라벨 없음 |
| pattern | area | 3 | 정적 공간 패턴 3 |
| pattern | contrast | 1 | 새로 생긴 라벨 없음 |
| pattern | cpd | 6 | 정적 공간 패턴 2 |

## 무늬 종류별 (pattern, hazard)

| 무늬 | 개수 | 정적 공간 패턴 적중 |
|---|---|---|
| straight | 1 | 1/1 |
| concentric | 1 | 1/1 |
| radial | 1 | 1/1 |
| spiral | 2 | 2/2 |
| chirp | 2 | 2/2 |
| checker | 1 | 0/1 |
| wavy | 2 | 1/2 |

## 놓친 hazard 표본 (레시피 요약)

- `test_lum_ext_hazard_004` 새로 생긴 라벨 없음 · mode=phys, mask_mode=anchored, area=0.501, halves=[4, 5, 6], dur=6.138, delta_cd=103.76, sigma=0.0
- `test_pattern_hazard_009` 새로 생긴 라벨 없음 · mask_mode=static, pattern=checker, area=0.769, dur=2.377, cpd=6.867, lo=12.246, hi=63.494, sigma=0.0
- `test_red_ext_hazard_004` 새로 생긴 라벨 없음 · mode=tint, mask_mode=spot, area=0.379, halves=[4, 5, 6], dur=6.571, sat=0.925, sigma=0.0
- `test_red_ext_hazard_007` 새로 생긴 라벨 없음 · mode=phys, mask_mode=anchored, area=0.502, halves=[4, 5, 6], dur=6.949, sat=0.868, sigma=0.0
- `test_red_flash_hazard_008` 새로 생긴 라벨 없음 · mode=tint, mask_mode=spot, area=0.345, halves=[1, 2, 3], dur=1.884, sat=0.926, sigma=0.0
- `test_lum_flash_hazard_003` 새로 생긴 라벨 없음 · mode=phys, mask_mode=multi, area=0.36, halves=[1, 2, 3], dur=2.681, delta_cd=105.238, sigma=0.0
- `test_pattern_hazard_007` 새로 생긴 라벨 없음 · mask_mode=anchored, pattern=wavy, area=0.618, dur=3.337, cpd=6.829, lo=4.342, hi=104.432, sigma=0.7
- `test_red_ext_hazard_005` 새로 생긴 라벨 없음 · mode=pair, mask_mode=anchored, area=0.392, halves=[4, 5, 6], dur=7.151, sat=0.942, sigma=0.0
- `test_red_flash_hazard_009` 새로 생긴 라벨 없음 · mode=tint, mask_mode=spot, area=0.36, halves=[1, 2, 3], dur=2.153, sat=0.862, sigma=0.0
- `test_lum_ext_hazard_006` 새로 생긴 라벨 없음 · mode=flat, mask_mode=anchored, area=0.419, halves=[4, 5, 6], dur=5.94, delta_cd=70.793, sigma=0.0
- `test_red_ext_hazard_000` 새로 생긴 라벨 없음 · mode=phys, mask_mode=anchored, area=0.671, halves=[4, 5, 6], dur=7.048, sat=0.887, sigma=0.0
