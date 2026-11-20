# 7종 위험 라벨 합성 — d5 / val

> 2026-11-20 14:23 · `scripts/make_hazard7_dataset.py` · 8 s · 24 fps · 768×432 · mp4(crf 12) 로 저장 후 다시 디코드해 판별 · 정답 = 팀원 판별기(partial B2) · 오류 0개

## 목표 라벨 적중 (의도 hazard → 그 라벨이 나옴 / 의도 near → 그 라벨이 안 나옴)

**적중 = 합성으로 새로 생긴 라벨**(같은 바탕만 판별한 결과를 프레임 단위로 뺌, 무늬는 0.5 s 이상). 괄호 = 바탕 빼기 전.

바탕만 판별했을 때 라벨이 나온 표본: 47/140 (정적 공간 패턴 47, 휘도 플래시 1)

| 종류 → 목표 라벨 | hazard: 적중 | near: 그 라벨 새로 안 생김 | hazard 에서 새로 생긴 다른 라벨 | 우리 판정기 (hazard 중 위험) |
|---|---|---|---|---|
| lum_flash → 휘도 플래시 | 10/10 (10) | 10/10 (10) | – | 10/10 (lum) |
| intersection → 교집합 | 10/10 (10) | 10/10 (10) | 빠른 장면 전환 2 | 10/10 (lum) |
| scene → 빠른 장면 전환 | 10/10 (10) | 10/10 (10) | – | 1/10 (lum) |
| red_flash → 적색 플래시 | 5/10 (5) | 10/10 (10) | 휘도 플래시 3 | 5/10 (red) |
| lum_ext → 휘도 연장 | 9/10 (9) | 10/10 (10) | – | 9/10 (ext_lum) |
| red_ext → 적색 연장 | 6/10 (6) | 10/10 (10) | 휘도 연장 2 | 6/10 (ext_red) |
| pattern → 정적 공간 패턴 | 10/10 (10) | 4/10 (0) | – | – (줄무늬 판정은 느려 생략) |

## 마스크 방식별 (hazard, 적중)

| 마스크 | 개수 | 적중 |
|---|---|---|
| anchored | 10 | 8/10 |
| drift | 11 | 8/11 |
| full | 20 | 20/20 |
| multi | 1 | 1/1 |
| perspective | 4 | 4/4 |
| spot | 10 | 7/10 |
| static | 14 | 12/14 |

## near 의 방식별 (새로 생긴 라벨)

| 종류 | near 방식 | 개수 | 나온 라벨 |
|---|---|---|---|
| lum_flash | area | 3 | 새로 생긴 라벨 없음 |
| lum_flash | weak | 7 | 새로 생긴 라벨 없음 |
| intersection | area | 8 | 휘도 플래시 8 |
| intersection | slow | 2 | 빠른 장면 전환 2 |
| scene | area | 7 | 새로 생긴 라벨 없음 |
| scene | slow | 3 | 새로 생긴 라벨 없음 |
| red_flash | area | 3 | 새로 생긴 라벨 없음 |
| red_flash | slow | 4 | 새로 생긴 라벨 없음 |
| red_flash | weak | 3 | 휘도 플래시 1 |
| lum_ext | short | 7 | 새로 생긴 라벨 없음 |
| lum_ext | slow | 3 | 새로 생긴 라벨 없음 |
| red_ext | short | 4 | 새로 생긴 라벨 없음 |
| red_ext | slow | 6 | 새로 생긴 라벨 없음 |
| pattern | area | 3 | 정적 공간 패턴 1 |
| pattern | contrast | 1 | 새로 생긴 라벨 없음 |
| pattern | cpd | 5 | 정적 공간 패턴 4 |
| pattern | short | 1 | 정적 공간 패턴 1 |

## 무늬 종류별 (pattern, hazard)

| 무늬 | 개수 | 정적 공간 패턴 적중 |
|---|---|---|
| straight | 3 | 3/3 |
| concentric | 2 | 2/2 |
| spiral | 1 | 1/1 |
| chirp | 2 | 2/2 |
| wavy | 2 | 2/2 |

## 놓친 hazard 표본 (레시피 요약)

- `val_lum_ext_hazard_005` 새로 생긴 라벨 없음 · mode=gain, mask_mode=spot, area=0.311, halves=[4, 5, 6], dur=6.484, delta_cd=95.798, sigma=0.7
- `val_red_ext_hazard_001` 새로 생긴 라벨 없음 · mode=phys, mask_mode=anchored, area=0.349, halves=[4, 5, 6], dur=7.399, sat=0.884, sigma=0.0
- `val_red_ext_hazard_003` 새로 생긴 라벨 ['휘도 연장'] · mode=phys, mask_mode=drift, area=0.35, halves=[4, 5, 6], dur=6.166, sat=0.923, sigma=0.0
- `val_red_ext_hazard_006` 새로 생긴 라벨 ['휘도 연장'] · mode=phys, mask_mode=drift, area=0.569, halves=[4, 5, 6], dur=6.752, sat=0.913, sigma=0.0
- `val_red_ext_hazard_008` 새로 생긴 라벨 없음 · mode=phys, mask_mode=static, area=0.424, halves=[4, 5, 6], dur=6.285, sat=0.913, sigma=0.0
- `val_red_flash_hazard_000` 새로 생긴 라벨 ['휘도 플래시'] · mode=phys, mask_mode=drift, area=0.445, halves=[1, 2, 3], dur=1.513, sat=0.876, sigma=0.0
- `val_red_flash_hazard_002` 새로 생긴 라벨 없음 · mode=pair, mask_mode=spot, area=0.56, halves=[1, 2, 3], dur=3.575, sat=0.878, sigma=0.0
- `val_red_flash_hazard_004` 새로 생긴 라벨 ['휘도 플래시'] · mode=phys, mask_mode=spot, area=0.365, halves=[1, 2, 3], dur=2.746, sat=0.877, sigma=0.0
- `val_red_flash_hazard_008` 새로 생긴 라벨 없음 · mode=phys, mask_mode=anchored, area=0.511, halves=[1, 2, 3], dur=2.102, sat=0.862, sigma=0.0
- `val_red_flash_hazard_009` 새로 생긴 라벨 없음 · mode=phys, mask_mode=static, area=0.397, halves=[1, 2, 3], dur=1.943, sat=0.915, sigma=0.0
