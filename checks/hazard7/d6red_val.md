# 7종 위험 라벨 합성 — d6red / val

> 2026-11-20 14:36 · `scripts/make_hazard7_dataset.py` · 8 s · 24 fps · 768×432 · mp4(crf 12) 로 저장 후 다시 디코드해 판별 · 정답 = 팀원 판별기(partial B2) · 오류 0개

## 목표 라벨 적중 (의도 hazard → 그 라벨이 나옴 / 의도 near → 그 라벨이 안 나옴)

**적중 = 합성으로 새로 생긴 라벨**(같은 바탕만 판별한 결과를 프레임 단위로 뺌, 무늬는 0.5 s 이상). 괄호 = 바탕 빼기 전.

바탕만 판별했을 때 라벨이 나온 표본: 11/40 (정적 공간 패턴 11)

| 종류 → 목표 라벨 | hazard: 적중 | near: 그 라벨 새로 안 생김 | hazard 에서 새로 생긴 다른 라벨 | 우리 판정기 (hazard 중 위험) |
|---|---|---|---|---|
| lum_flash → 휘도 플래시 | 0/0 (0) | 0/0 (0) | – | 0/0 (lum) |
| intersection → 교집합 | 0/0 (0) | 0/0 (0) | – | 0/0 (lum) |
| scene → 빠른 장면 전환 | 0/0 (0) | 0/0 (0) | – | 0/0 (lum) |
| red_flash → 적색 플래시 | 8/10 (8) | 10/10 (10) | 휘도 플래시 4 | 7/10 (red) |
| lum_ext → 휘도 연장 | 0/0 (0) | 0/0 (0) | – | 0/0 (ext_lum) |
| red_ext → 적색 연장 | 8/10 (8) | 10/10 (10) | 휘도 연장 3 | 8/10 (ext_red) |
| pattern → 정적 공간 패턴 | 0/0 (0) | 0/0 (0) | – | – (줄무늬 판정은 느려 생략) |

## 마스크 방식별 (hazard, 적중)

| 마스크 | 개수 | 적중 |
|---|---|---|
| anchored | 3 | 2/3 |
| drift | 6 | 5/6 |
| spot | 5 | 3/5 |
| static | 6 | 6/6 |

## near 의 방식별 (새로 생긴 라벨)

| 종류 | near 방식 | 개수 | 나온 라벨 |
|---|---|---|---|
| red_flash | area | 3 | 새로 생긴 라벨 없음 |
| red_flash | slow | 4 | 새로 생긴 라벨 없음 |
| red_flash | weak | 3 | 휘도 플래시 1 |
| red_ext | short | 4 | 새로 생긴 라벨 없음 |
| red_ext | slow | 6 | 새로 생긴 라벨 없음 |

## 무늬 종류별 (pattern, hazard)

| 무늬 | 개수 | 정적 공간 패턴 적중 |
|---|---|---|

## 놓친 hazard 표본 (레시피 요약)

- `val_red_ext_hazard_001` 새로 생긴 라벨 없음 · mode=phys, mask_mode=anchored, area=0.349, halves=[4, 5, 6], dur=7.399, sat=0.884, sigma=0.0
- `val_red_ext_hazard_006` 새로 생긴 라벨 ['휘도 연장'] · mode=phys, mask_mode=drift, area=0.569, halves=[4, 5, 6], dur=6.752, sat=0.913, sigma=0.0
- `val_red_flash_hazard_002` 새로 생긴 라벨 없음 · mode=pair, mask_mode=spot, area=0.56, halves=[1, 2, 3], dur=3.575, sat=0.878, sigma=0.0
- `val_red_flash_hazard_004` 새로 생긴 라벨 ['휘도 플래시'] · mode=phys, mask_mode=spot, area=0.365, halves=[1, 2, 3], dur=2.746, sat=0.877, sigma=0.0
