# 7종 위험 라벨 합성 — big1 / val

> 2026-11-20 17:19 · `scripts/make_hazard7_dataset.py` · 8 s · 24 fps · 768×432 · mp4(crf 12) 로 저장 후 다시 디코드해 판별 · 정답 = 팀원 판별기(partial B2) · 오류 0개

## 목표 라벨 적중 (의도 hazard → 그 라벨이 나옴 / 의도 near → 그 라벨이 안 나옴)

**적중 = 합성으로 새로 생긴 라벨**(같은 바탕만 판별한 결과를 프레임 단위로 뺌, 무늬는 0.5 s 이상). 괄호 = 바탕 빼기 전.

바탕만 판별했을 때 라벨이 나온 표본: 98/280 (정적 공간 패턴 98, 휘도 플래시 1)

| 종류 → 목표 라벨 | hazard: 적중 | near: 그 라벨 새로 안 생김 | hazard 에서 새로 생긴 다른 라벨 | 우리 판정기 (hazard 중 위험) |
|---|---|---|---|---|
| lum_flash → 휘도 플래시 | 20/20 (20) | 20/20 (20) | 정적 공간 패턴 1 | 20/20 (lum) |
| intersection → 교집합 | 20/20 (20) | 20/20 (20) | 빠른 장면 전환 4, 휘도 플래시 3 | 20/20 (lum) |
| scene → 빠른 장면 전환 | 20/20 (20) | 20/20 (20) | – | 1/20 (lum) |
| red_flash → 적색 플래시 | 14/20 (14) | 20/20 (20) | 휘도 플래시 7 | 14/20 (red) |
| lum_ext → 휘도 연장 | 18/20 (18) | 20/20 (20) | 정적 공간 패턴 1 | 17/20 (ext_lum) |
| red_ext → 적색 연장 | 18/20 (18) | 20/20 (20) | 휘도 연장 6, 정적 공간 패턴 1 | 18/20 (ext_red) |
| pattern → 정적 공간 패턴 | 17/20 (20) | 12/20 (7) | 휘도 플래시 1 | – (줄무늬 판정은 느려 생략) |

## 마스크 방식별 (hazard, 적중)

| 마스크 | 개수 | 적중 |
|---|---|---|
| anchored | 31 | 27/31 |
| drift | 18 | 15/18 |
| full | 40 | 40/40 |
| multi | 5 | 5/5 |
| perspective | 9 | 8/9 |
| spot | 14 | 9/14 |
| static | 23 | 23/23 |

## near 의 방식별 (새로 생긴 라벨)

| 종류 | near 방식 | 개수 | 나온 라벨 |
|---|---|---|---|
| lum_flash | area | 6 | 새로 생긴 라벨 없음 |
| lum_flash | slow | 2 | 새로 생긴 라벨 없음 |
| lum_flash | weak | 12 | 새로 생긴 라벨 없음 |
| intersection | area | 17 | 휘도 플래시 13 |
| intersection | slow | 3 | 빠른 장면 전환 3 |
| scene | area | 11 | 새로 생긴 라벨 없음 |
| scene | slow | 9 | 새로 생긴 라벨 없음 |
| red_flash | area | 7 | 새로 생긴 라벨 없음 |
| red_flash | slow | 6 | 새로 생긴 라벨 없음 |
| red_flash | weak | 7 | 휘도 플래시 2, 정적 공간 패턴 1 |
| lum_ext | short | 11 | 정적 공간 패턴 1 |
| lum_ext | slow | 9 | 새로 생긴 라벨 없음 |
| red_ext | short | 7 | 새로 생긴 라벨 없음 |
| red_ext | slow | 13 | 정적 공간 패턴 1 |
| pattern | area | 8 | 정적 공간 패턴 6 |
| pattern | contrast | 1 | 새로 생긴 라벨 없음 |
| pattern | cpd | 7 | 정적 공간 패턴 1 |
| pattern | short | 4 | 정적 공간 패턴 1 |

## 무늬 종류별 (pattern, hazard)

| 무늬 | 개수 | 정적 공간 패턴 적중 |
|---|---|---|
| straight | 5 | 5/5 |
| concentric | 3 | 3/3 |
| radial | 2 | 1/2 |
| spiral | 1 | 1/1 |
| chirp | 4 | 3/4 |
| checker | 2 | 2/2 |
| wavy | 3 | 2/3 |

## 놓친 hazard 표본 (레시피 요약)

- `val_lum_ext_hazard_005` 새로 생긴 라벨 없음 · mode=gain, mask_mode=spot, area=0.311, halves=[4, 5, 6], dur=6.484, delta_cd=95.798, sigma=0.7
- `val_pattern_hazard_012` 새로 생긴 라벨 없음 · mask_mode=perspective, pattern=chirp, area=0.635, dur=3.135, cpd=5.594, lo=16.443, hi=61.594, sigma=0.4
- `val_pattern_hazard_015` 새로 생긴 라벨 없음 · mask_mode=anchored, pattern=radial, area=0.732, dur=1.235, cpd=2.495, lo=5.766, hi=79.067, sigma=0.4
- `val_red_flash_hazard_004` 새로 생긴 라벨 없음 · mode=phys, mask_mode=spot, area=0.365, halves=[1, 2, 3], dur=2.746, sat=0.877, sigma=0.0
- `val_red_flash_hazard_007` 새로 생긴 라벨 ['휘도 플래시'] · mode=phys, mask_mode=spot, area=0.573, halves=[1, 2, 3], dur=3.978, sat=0.887, sigma=0.0
- `val_red_flash_hazard_016` 새로 생긴 라벨 ['휘도 플래시'] · mode=phys, mask_mode=spot, area=0.478, halves=[1, 2, 3], dur=1.793, sat=0.886, sigma=0.0
- `val_red_ext_hazard_012` 새로 생긴 라벨 없음 · mode=phys, mask_mode=drift, area=0.304, halves=[4, 5, 6], dur=6.236, sat=0.922, sigma=0.0
- `val_red_flash_hazard_002` 새로 생긴 라벨 없음 · mode=pair, mask_mode=spot, area=0.56, halves=[1, 2, 3], dur=3.575, sat=0.878, sigma=0.0
- `val_lum_ext_hazard_004` 새로 생긴 라벨 없음 · mode=flat, mask_mode=drift, area=0.462, halves=[4, 5, 6], dur=5.81, delta_cd=105.17, sigma=0.7
- `val_pattern_hazard_017` 새로 생긴 라벨 없음 · mask_mode=anchored, pattern=wavy, area=0.482, dur=3.52, cpd=6.57, lo=16.702, hi=133.921, sigma=0.0
- `val_red_ext_hazard_001` 새로 생긴 라벨 없음 · mode=phys, mask_mode=anchored, area=0.349, halves=[4, 5, 6], dur=7.399, sat=0.884, sigma=0.0
- `val_red_flash_hazard_000` 새로 생긴 라벨 ['휘도 플래시'] · mode=phys, mask_mode=drift, area=0.445, halves=[1, 2, 3], dur=1.513, sat=0.876, sigma=0.0
- `val_red_flash_hazard_018` 새로 생긴 라벨 ['휘도 플래시'] · mode=phys, mask_mode=anchored, area=0.595, halves=[1, 2, 3], dur=3.139, sat=0.946, sigma=0.0
