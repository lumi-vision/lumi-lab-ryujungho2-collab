# 7종 위험 라벨 합성 — big1 / train

> 2026-11-21 01:13 · `scripts/make_hazard7_dataset.py` · 8 s · 24 fps · 768×432 · mp4(crf 12) 로 저장 후 다시 디코드해 판별 · 정답 = 팀원 판별기(partial B2) · 오류 0개

## 목표 라벨 적중 (의도 hazard → 그 라벨이 나옴 / 의도 near → 그 라벨이 안 나옴)

**적중 = 합성으로 새로 생긴 라벨**(같은 바탕만 판별한 결과를 프레임 단위로 뺌, 무늬는 0.5 s 이상). 괄호 = 바탕 빼기 전.

바탕만 판별했을 때 라벨이 나온 표본: 1020/2800 (정적 공간 패턴 1020, 휘도 플래시 20)

| 종류 → 목표 라벨 | hazard: 적중 | near: 그 라벨 새로 안 생김 | hazard 에서 새로 생긴 다른 라벨 | 우리 판정기 (hazard 중 위험) |
|---|---|---|---|---|
| lum_flash → 휘도 플래시 | 191/200 (191) | 200/200 (200) | 정적 공간 패턴 6 | 195/200 (lum) |
| intersection → 교집합 | 196/200 (196) | 198/200 (198) | 휘도 플래시 51, 빠른 장면 전환 22, 정적 공간 패턴 1 | 200/200 (lum) |
| scene → 빠른 장면 전환 | 200/200 (200) | 200/200 (200) | – | 3/200 (lum) |
| red_flash → 적색 플래시 | 153/200 (153) | 200/200 (200) | 휘도 플래시 70, 정적 공간 패턴 3 | 151/200 (red) |
| lum_ext → 휘도 연장 | 180/200 (180) | 200/200 (200) | 정적 공간 패턴 7, 빠른 장면 전환 2 | 174/200 (ext_lum) |
| red_ext → 적색 연장 | 148/200 (148) | 200/200 (200) | 휘도 연장 62, 정적 공간 패턴 5 | 142/200 (ext_red) |
| pattern → 정적 공간 패턴 | 163/200 (200) | 142/200 (92) | 휘도 플래시 8 | – (줄무늬 판정은 느려 생략) |

## 마스크 방식별 (hazard, 적중)

| 마스크 | 개수 | 적중 |
|---|---|---|
| anchored | 316 | 265/316 |
| drift | 185 | 167/185 |
| full | 400 | 396/400 |
| multi | 86 | 76/86 |
| perspective | 73 | 61/73 |
| spot | 163 | 108/163 |
| static | 177 | 158/177 |

## near 의 방식별 (새로 생긴 라벨)

| 종류 | near 방식 | 개수 | 나온 라벨 |
|---|---|---|---|
| lum_flash | area | 65 | 정적 공간 패턴 1 |
| lum_flash | slow | 69 | 정적 공간 패턴 1 |
| lum_flash | weak | 66 | 정적 공간 패턴 4 |
| intersection | area | 105 | 휘도 플래시 100, 교집합 2, 정적 공간 패턴 1 |
| intersection | slow | 95 | 빠른 장면 전환 95, 정적 공간 패턴 3 |
| scene | area | 99 | 정적 공간 패턴 1 |
| scene | slow | 101 | 정적 공간 패턴 11 |
| red_flash | area | 69 | 정적 공간 패턴 3 |
| red_flash | slow | 60 | 정적 공간 패턴 2 |
| red_flash | weak | 71 | 휘도 플래시 27, 정적 공간 패턴 1 |
| lum_ext | short | 101 | 정적 공간 패턴 5 |
| lum_ext | slow | 99 | 정적 공간 패턴 3, 휘도 플래시 1 |
| red_ext | short | 107 | 정적 공간 패턴 2 |
| red_ext | slow | 93 | 정적 공간 패턴 2 |
| pattern | area | 48 | 정적 공간 패턴 34 |
| pattern | contrast | 44 | 새로 생긴 라벨 없음 |
| pattern | cpd | 59 | 정적 공간 패턴 23 |
| pattern | short | 49 | 정적 공간 패턴 1 |

## 무늬 종류별 (pattern, hazard)

| 무늬 | 개수 | 정적 공간 패턴 적중 |
|---|---|---|
| straight | 32 | 23/32 |
| concentric | 24 | 18/24 |
| radial | 32 | 27/32 |
| spiral | 25 | 19/25 |
| chirp | 25 | 22/25 |
| checker | 31 | 26/31 |
| wavy | 31 | 28/31 |

## 놓친 hazard 표본 (레시피 요약)

- `train_lum_ext_hazard_029` 새로 생긴 라벨 없음 · mode=flat, mask_mode=drift, area=0.312, halves=[4, 5, 6], dur=6.542, delta_cd=91.48, sigma=0.0
- `train_lum_ext_hazard_032` 새로 생긴 라벨 없음 · mode=flat, mask_mode=anchored, area=0.416, halves=[4, 5, 6], dur=6.202, delta_cd=54.677, sigma=0.4
- `train_lum_ext_hazard_068` 새로 생긴 라벨 없음 · mode=phys, mask_mode=multi, area=0.411, halves=[4, 5, 6], dur=7.02, delta_cd=106.706, sigma=0.0
- `train_lum_ext_hazard_077` 새로 생긴 라벨 없음 · mode=phys, mask_mode=spot, area=0.347, halves=[4, 5, 6], dur=5.878, delta_cd=73.211, sigma=0.0
- `train_lum_ext_hazard_095` 새로 생긴 라벨 없음 · mode=flat, mask_mode=anchored, area=0.339, halves=[4, 5, 6], dur=6.824, delta_cd=102.067, sigma=0.0
- `train_lum_ext_hazard_113` 새로 생긴 라벨 없음 · mode=phys, mask_mode=drift, area=0.313, halves=[4, 5, 6], dur=6.296, delta_cd=122.747, sigma=0.0
- `train_lum_ext_hazard_125` 새로 생긴 라벨 없음 · mode=add, mask_mode=drift, area=0.649, halves=[4, 5, 6], dur=6.632, delta_cd=58.229, sigma=0.0
- `train_lum_ext_hazard_179` 새로 생긴 라벨 없음 · mode=add, mask_mode=anchored, area=0.391, halves=[4, 5, 6], dur=6.748, delta_cd=69.827, sigma=0.0
- `train_lum_flash_hazard_013` 새로 생긴 라벨 없음 · mode=flat, mask_mode=static, area=0.574, halves=[1, 2, 3], dur=3.662, delta_cd=122.385, sigma=0.7
- `train_lum_flash_hazard_079` 새로 생긴 라벨 없음 · mode=flat, mask_mode=static, area=0.323, halves=[1, 2, 3], dur=3.799, delta_cd=53.781, sigma=0.7
- `train_lum_flash_hazard_130` 새로 생긴 라벨 없음 · mode=flat, mask_mode=spot, area=0.325, halves=[1, 2, 3], dur=2.633, delta_cd=105.78, sigma=0.0
- `train_pattern_hazard_000` 새로 생긴 라벨 없음 · mask_mode=static, pattern=chirp, area=0.72, dur=2.238, cpd=3.029, lo=11.666, hi=128.547, sigma=0.7
- `train_pattern_hazard_009` 새로 생긴 라벨 없음 · mask_mode=perspective, pattern=chirp, area=0.607, dur=2.01, cpd=6.457, lo=13.258, hi=100.532, sigma=0.7
- `train_pattern_hazard_021` 새로 생긴 라벨 없음 · mask_mode=perspective, pattern=spiral, area=0.783, dur=1.778, cpd=3.702, lo=15.916, hi=65.504, sigma=0.4
- `train_pattern_hazard_063` 새로 생긴 라벨 없음 · mask_mode=anchored, pattern=spiral, area=0.688, dur=2.702, cpd=5.356, lo=13.174, hi=64.67, sigma=0.0
- `train_pattern_hazard_075` 새로 생긴 라벨 없음 · mask_mode=static, pattern=straight, area=0.569, dur=2.027, cpd=3.321, lo=19.063, hi=71.532, sigma=0.0
- `train_pattern_hazard_087` 새로 생긴 라벨 없음 · mask_mode=anchored, pattern=spiral, area=0.62, dur=2.858, cpd=1.99, lo=18.488, hi=94.196, sigma=0.0
- `train_pattern_hazard_090` 새로 생긴 라벨 없음 · mask_mode=anchored, pattern=straight, area=0.528, dur=2.621, cpd=1.952, lo=4.158, hi=118.933, sigma=0.7
- `train_pattern_hazard_105` 새로 생긴 라벨 없음 · mask_mode=perspective, pattern=concentric, area=0.542, dur=3.146, cpd=2.316, lo=6.282, hi=62.018, sigma=0.0
- `train_pattern_hazard_111` 새로 생긴 라벨 없음 · mask_mode=perspective, pattern=chirp, area=0.683, dur=2.518, cpd=2.264, lo=14.105, hi=129.245, sigma=0.0
- `train_pattern_hazard_117` 새로 생긴 라벨 없음 · mask_mode=anchored, pattern=checker, area=0.803, dur=3.298, cpd=2.935, lo=14.971, hi=97.13, sigma=0.4
- `train_pattern_hazard_126` 새로 생긴 라벨 없음 · mask_mode=static, pattern=wavy, area=0.569, dur=1.83, cpd=2.228, lo=11.058, hi=93.375, sigma=0.4
- `train_pattern_hazard_186` 새로 생긴 라벨 없음 · mask_mode=perspective, pattern=concentric, area=0.524, dur=3.559, cpd=5.769, lo=9.731, hi=68.197, sigma=0.7
- `train_pattern_hazard_195` 새로 생긴 라벨 없음 · mask_mode=static, pattern=concentric, area=0.6, dur=3.662, cpd=4.51, lo=8.027, hi=104.774, sigma=0.4
- `train_red_ext_hazard_008` 새로 생긴 라벨 없음 · mode=tint, mask_mode=anchored, area=0.32, halves=[4, 5, 6], dur=6.754, sat=0.961, sigma=0.0
- `train_red_ext_hazard_026` 새로 생긴 라벨 없음 · mode=phys, mask_mode=spot, area=0.496, halves=[4, 5, 6], dur=6.56, sat=0.899, sigma=0.0
- `train_red_ext_hazard_035` 새로 생긴 라벨 없음 · mode=tint, mask_mode=spot, area=0.433, halves=[4, 5, 6], dur=6.465, sat=0.948, sigma=0.0
- `train_red_ext_hazard_053` 새로 생긴 라벨 ['휘도 연장'] · mode=phys, mask_mode=spot, area=0.451, halves=[4, 5, 6], dur=6.26, sat=0.89, sigma=0.0
- `train_red_ext_hazard_065` 새로 생긴 라벨 없음 · mode=phys, mask_mode=anchored, area=0.39, halves=[4, 5, 6], dur=6.604, sat=0.871, sigma=0.0
- `train_red_ext_hazard_089` 새로 생긴 라벨 없음 · mode=phys, mask_mode=anchored, area=0.329, halves=[4, 5, 6], dur=6.392, sat=0.914, sigma=0.0
- `train_red_ext_hazard_095` 새로 생긴 라벨 없음 · mode=phys, mask_mode=anchored, area=0.686, halves=[4, 5, 6], dur=7.324, sat=0.924, sigma=0.0
- `train_red_ext_hazard_101` 새로 생긴 라벨 ['휘도 연장'] · mode=phys, mask_mode=spot, area=0.331, halves=[4, 5, 6], dur=6.393, sat=0.934, sigma=0.0
- `train_red_ext_hazard_122` 새로 생긴 라벨 없음 · mode=phys, mask_mode=multi, area=0.528, halves=[4, 5, 6], dur=6.082, sat=0.878, sigma=0.0
- `train_red_ext_hazard_131` 새로 생긴 라벨 없음 · mode=pair, mask_mode=spot, area=0.546, halves=[4, 5, 6], dur=7.343, sat=0.955, sigma=0.0
- `train_red_ext_hazard_155` 새로 생긴 라벨 ['휘도 연장'] · mode=phys, mask_mode=static, area=0.52, halves=[4, 5, 6], dur=6.976, sat=0.876, sigma=0.0
- `train_red_ext_hazard_158` 새로 생긴 라벨 없음 · mode=phys, mask_mode=drift, area=0.535, halves=[4, 5, 6], dur=6.605, sat=0.89, sigma=0.0
- `train_red_ext_hazard_176` 새로 생긴 라벨 없음 · mode=tint, mask_mode=spot, area=0.398, halves=[4, 5, 6], dur=5.996, sat=0.943, sigma=0.0
- `train_red_ext_hazard_191` 새로 생긴 라벨 없음 · mode=tint, mask_mode=spot, area=0.315, halves=[4, 5, 6], dur=6.281, sat=0.962, sigma=0.0
- `train_red_flash_hazard_007` 새로 생긴 라벨 없음 · mode=pair, mask_mode=multi, area=0.315, halves=[1, 2, 3], dur=3.098, sat=0.909, sigma=0.0
- `train_red_flash_hazard_013` 새로 생긴 라벨 없음 · mode=phys, mask_mode=drift, area=0.502, halves=[1, 2, 3], dur=1.636, sat=0.878, sigma=0.0
- `train_red_flash_hazard_043` 새로 생긴 라벨 없음 · mode=phys, mask_mode=anchored, area=0.411, halves=[1, 2, 3], dur=1.988, sat=0.94, sigma=0.0
- `train_red_flash_hazard_049` 새로 생긴 라벨 없음 · mode=phys, mask_mode=spot, area=0.376, halves=[1, 2, 3], dur=3.618, sat=0.897, sigma=0.0
- `train_red_flash_hazard_076` 새로 생긴 라벨 ['휘도 플래시'] · mode=phys, mask_mode=anchored, area=0.426, halves=[1, 2, 3], dur=3.732, sat=0.918, sigma=0.0
- `train_red_flash_hazard_079` 새로 생긴 라벨 없음 · mode=phys, mask_mode=spot, area=0.392, halves=[1, 2, 3], dur=2.065, sat=0.961, sigma=0.0
- `train_red_flash_hazard_085` 새로 생긴 라벨 없음 · mode=tint, mask_mode=spot, area=0.428, halves=[1, 2, 3], dur=2.61, sat=0.903, sigma=0.0
- `train_red_flash_hazard_091` 새로 생긴 라벨 없음 · mode=tint, mask_mode=spot, area=0.398, halves=[1, 2, 3], dur=2.677, sat=0.903, sigma=0.0
- `train_red_flash_hazard_094` 새로 생긴 라벨 ['휘도 플래시'] · mode=pair, mask_mode=spot, area=0.484, halves=[1, 2, 3], dur=3.368, sat=0.912, sigma=0.0
- `train_red_flash_hazard_100` 새로 생긴 라벨 없음 · mode=phys, mask_mode=spot, area=0.369, halves=[1, 2, 3], dur=1.736, sat=0.915, sigma=0.0
- `train_red_flash_hazard_112` 새로 생긴 라벨 없음 · mode=pair, mask_mode=spot, area=0.471, halves=[1, 2, 3], dur=2.942, sat=0.86, sigma=0.0
- `train_red_flash_hazard_130` 새로 생긴 라벨 없음 · mode=phys, mask_mode=anchored, area=0.574, halves=[1, 2, 3], dur=1.838, sat=0.956, sigma=0.0
- `train_red_flash_hazard_154` 새로 생긴 라벨 ['휘도 플래시'] · mode=phys, mask_mode=drift, area=0.514, halves=[1, 2, 3], dur=2.957, sat=0.92, sigma=0.0
- `train_red_flash_hazard_163` 새로 생긴 라벨 없음 · mode=phys, mask_mode=spot, area=0.302, halves=[1, 2, 3], dur=1.809, sat=0.931, sigma=0.0
- `train_red_flash_hazard_166` 새로 생긴 라벨 없음 · mode=phys, mask_mode=spot, area=0.472, halves=[1, 2, 3], dur=3.152, sat=0.885, sigma=0.0
- `train_red_flash_hazard_181` 새로 생긴 라벨 ['휘도 플래시'] · mode=phys, mask_mode=drift, area=0.313, halves=[1, 2, 3], dur=3.334, sat=0.959, sigma=0.0
- `train_red_flash_hazard_190` 새로 생긴 라벨 ['휘도 플래시'] · mode=phys, mask_mode=static, area=0.53, halves=[1, 2, 3], dur=2.496, sat=0.924, sigma=0.0
- `train_intersection_hazard_070` 새로 생긴 라벨 ['휘도 플래시'] · mode=gain, mask_mode=full, area=1.0, halves=[1, 2, 3], dur=1.739, sigma=0.7
- `train_intersection_hazard_133` 새로 생긴 라벨 ['빠른 장면 전환'] · mode=gain, mask_mode=full, area=1.0, halves=[1, 2, 3], dur=3.141, sigma=0.7
- `train_intersection_hazard_163` 새로 생긴 라벨 ['빠른 장면 전환'] · mode=gain, mask_mode=full, area=1.0, halves=[1, 2, 3], dur=2.582, sigma=0.7
- `train_lum_ext_hazard_021` 새로 생긴 라벨 없음 · mode=phys, mask_mode=spot, area=0.38, halves=[4, 5, 6], dur=6.258, delta_cd=80.064, sigma=0.0
- `train_lum_ext_hazard_033` 새로 생긴 라벨 없음 · mode=add, mask_mode=anchored, area=0.325, halves=[4, 5, 6], dur=6.995, delta_cd=86.43, sigma=0.7
- `train_lum_ext_hazard_042` 새로 생긴 라벨 없음 · mode=add, mask_mode=spot, area=0.328, halves=[4, 5, 6], dur=7.405, delta_cd=54.396, sigma=0.0
- `train_lum_ext_hazard_084` 새로 생긴 라벨 ['정적 공간 패턴'] · mode=add, mask_mode=anchored, area=0.408, halves=[4, 5, 6], dur=6.4, delta_cd=75.504, sigma=0.0
- `train_lum_ext_hazard_150` 새로 생긴 라벨 없음 · mode=flat, mask_mode=static, area=0.394, halves=[4, 5, 6], dur=7.231, delta_cd=88.775, sigma=0.7
- `train_lum_ext_hazard_195` 새로 생긴 라벨 없음 · mode=gain, mask_mode=drift, area=0.516, halves=[4, 5, 6], dur=7.283, delta_cd=50.773, sigma=0.7
- `train_lum_flash_hazard_020` 새로 생긴 라벨 없음 · mode=phys, mask_mode=anchored, area=0.448, halves=[1, 2, 3], dur=2.2, delta_cd=103.996, sigma=0.0
- `train_lum_flash_hazard_056` 새로 생긴 라벨 없음 · mode=add, mask_mode=anchored, area=0.305, halves=[1, 2, 3], dur=3.315, delta_cd=92.348, sigma=0.7
- `train_lum_flash_hazard_119` 새로 생긴 라벨 없음 · mode=add, mask_mode=spot, area=0.534, halves=[1, 2, 3], dur=3.891, delta_cd=62.17, sigma=0.7
- `train_lum_flash_hazard_176` 새로 생긴 라벨 없음 · mode=flat, mask_mode=spot, area=0.347, halves=[1, 2, 3], dur=3.772, delta_cd=63.073, sigma=0.7
- `train_pattern_hazard_007` 새로 생긴 라벨 없음 · mask_mode=anchored, pattern=straight, area=0.602, dur=1.897, cpd=3.639, lo=15.701, hi=118.285, sigma=0.4
- `train_pattern_hazard_016` 새로 생긴 라벨 없음 · mask_mode=perspective, pattern=wavy, area=0.713, dur=1.671, cpd=1.741, lo=13.849, hi=68.632, sigma=0.4
- `train_pattern_hazard_064` 새로 생긴 라벨 없음 · mask_mode=anchored, pattern=checker, area=0.727, dur=2.153, cpd=5.491, lo=4.361, hi=100.559, sigma=0.4
- `train_pattern_hazard_103` 새로 생긴 라벨 없음 · mask_mode=static, pattern=spiral, area=0.71, dur=3.641, cpd=5.195, lo=5.984, hi=105.116, sigma=0.0
- `train_pattern_hazard_130` 새로 생긴 라벨 없음 · mask_mode=static, pattern=radial, area=0.591, dur=3.278, cpd=1.858, lo=8.316, hi=76.047, sigma=0.7
- `train_pattern_hazard_139` 새로 생긴 라벨 없음 · mask_mode=perspective, pattern=wavy, area=0.482, dur=1.983, cpd=1.985, lo=5.58, hi=58.996, sigma=0.7
- `train_pattern_hazard_199` 새로 생긴 라벨 없음 · mask_mode=anchored, pattern=concentric, area=0.556, dur=2.838, cpd=1.988, lo=13.876, hi=57.92, sigma=0.4
- `train_red_ext_hazard_000` 새로 생긴 라벨 없음 · mode=tint, mask_mode=multi, area=0.311, halves=[4, 5, 6], dur=6.172, sat=0.965, sigma=0.0
- `train_red_ext_hazard_009` 새로 생긴 라벨 ['휘도 연장'] · mode=phys, mask_mode=spot, area=0.476, halves=[4, 5, 6], dur=6.202, sat=0.915, sigma=0.0
- `train_red_ext_hazard_030` 새로 생긴 라벨 없음 · mode=tint, mask_mode=spot, area=0.414, halves=[4, 5, 6], dur=7.014, sat=0.969, sigma=0.0
- `train_red_ext_hazard_039` 새로 생긴 라벨 없음 · mode=tint, mask_mode=spot, area=0.388, halves=[4, 5, 6], dur=6.518, sat=0.879, sigma=0.0
- `train_red_ext_hazard_045` 새로 생긴 라벨 ['휘도 연장'] · mode=phys, mask_mode=anchored, area=0.359, halves=[4, 5, 6], dur=7.153, sat=0.908, sigma=0.0
- `train_red_ext_hazard_051` 새로 생긴 라벨 없음 · mode=tint, mask_mode=anchored, area=0.402, halves=[4, 5, 6], dur=6.408, sat=0.884, sigma=0.0
- `train_red_ext_hazard_054` 새로 생긴 라벨 없음 · mode=phys, mask_mode=anchored, area=0.532, halves=[4, 5, 6], dur=6.132, sat=0.941, sigma=0.0
- `train_red_ext_hazard_099` 새로 생긴 라벨 ['휘도 연장'] · mode=phys, mask_mode=spot, area=0.447, halves=[4, 5, 6], dur=6.668, sat=0.934, sigma=0.0
- `train_red_ext_hazard_102` 새로 생긴 라벨 없음 · mode=phys, mask_mode=static, area=0.406, halves=[4, 5, 6], dur=7.382, sat=0.885, sigma=0.0
- `train_red_ext_hazard_108` 새로 생긴 라벨 없음 · mode=phys, mask_mode=drift, area=0.461, halves=[4, 5, 6], dur=6.646, sat=0.909, sigma=0.0
- `train_red_ext_hazard_114` 새로 생긴 라벨 없음 · mode=tint, mask_mode=anchored, area=0.331, halves=[4, 5, 6], dur=5.996, sat=0.93, sigma=0.0
- `train_red_ext_hazard_150` 새로 생긴 라벨 없음 · mode=phys, mask_mode=anchored, area=0.383, halves=[4, 5, 6], dur=6.33, sat=0.944, sigma=0.0
- `train_red_ext_hazard_156` 새로 생긴 라벨 없음 · mode=phys, mask_mode=drift, area=0.467, halves=[4, 5, 6], dur=6.007, sat=0.925, sigma=0.0
- `train_red_ext_hazard_171` 새로 생긴 라벨 ['휘도 연장'] · mode=phys, mask_mode=drift, area=0.515, halves=[4, 5, 6], dur=7.036, sat=0.954, sigma=0.0
- `train_red_ext_hazard_174` 새로 생긴 라벨 없음 · mode=phys, mask_mode=multi, area=0.353, halves=[4, 5, 6], dur=7.208, sat=0.866, sigma=0.0
- `train_red_ext_hazard_177` 새로 생긴 라벨 없음 · mode=phys, mask_mode=anchored, area=0.466, halves=[4, 5, 6], dur=6.813, sat=0.863, sigma=0.0
- `train_red_ext_hazard_198` 새로 생긴 라벨 없음 · mode=phys, mask_mode=anchored, area=0.599, halves=[4, 5, 6], dur=6.897, sat=0.915, sigma=0.0
- `train_red_flash_hazard_026` 새로 생긴 라벨 ['휘도 플래시'] · mode=tint, mask_mode=spot, area=0.438, halves=[1, 2, 3], dur=1.776, sat=0.861, sigma=0.0
- `train_red_flash_hazard_029` 새로 생긴 라벨 없음 · mode=phys, mask_mode=anchored, area=0.521, halves=[1, 2, 3], dur=2.751, sat=0.955, sigma=0.0
- `train_red_flash_hazard_032` 새로 생긴 라벨 ['휘도 플래시'] · mode=phys, mask_mode=anchored, area=0.426, halves=[1, 2, 3], dur=1.803, sat=0.875, sigma=0.0
- `train_red_flash_hazard_035` 새로 생긴 라벨 없음 · mode=phys, mask_mode=drift, area=0.556, halves=[1, 2, 3], dur=2.463, sat=0.927, sigma=0.0
- `train_red_flash_hazard_080` 새로 생긴 라벨 ['휘도 플래시'] · mode=phys, mask_mode=anchored, area=0.333, halves=[1, 2, 3], dur=2.06, sat=0.895, sigma=0.0
- `train_red_flash_hazard_098` 새로 생긴 라벨 없음 · mode=pair, mask_mode=spot, area=0.374, halves=[1, 2, 3], dur=1.551, sat=0.958, sigma=0.0
- `train_red_flash_hazard_116` 새로 생긴 라벨 없음 · mode=phys, mask_mode=anchored, area=0.324, halves=[1, 2, 3], dur=2.954, sat=0.954, sigma=0.0
- `train_red_flash_hazard_125` 새로 생긴 라벨 없음 · mode=phys, mask_mode=spot, area=0.444, halves=[1, 2, 3], dur=3.058, sat=0.921, sigma=0.0
- `train_red_flash_hazard_146` 새로 생긴 라벨 ['휘도 플래시'] · mode=phys, mask_mode=anchored, area=0.415, halves=[1, 2, 3], dur=2.357, sat=0.96, sigma=0.0
- `train_red_flash_hazard_149` 새로 생긴 라벨 없음 · mode=tint, mask_mode=spot, area=0.312, halves=[1, 2, 3], dur=3.891, sat=0.92, sigma=0.0
- `train_red_flash_hazard_161` 새로 생긴 라벨 없음 · mode=tint, mask_mode=spot, area=0.334, halves=[1, 2, 3], dur=2.374, sat=0.9, sigma=0.0
- `train_red_flash_hazard_164` 새로 생긴 라벨 없음 · mode=tint, mask_mode=spot, area=0.411, halves=[1, 2, 3], dur=3.157, sat=0.894, sigma=0.0
- `train_red_flash_hazard_194` 새로 생긴 라벨 없음 · mode=phys, mask_mode=anchored, area=0.586, halves=[1, 2, 3], dur=3.08, sat=0.965, sigma=0.0
- `train_intersection_hazard_080` 새로 생긴 라벨 ['빠른 장면 전환'] · mode=phys, mask_mode=full, area=1.0, halves=[1, 2, 3], dur=1.969, sigma=0.0
- `train_lum_ext_hazard_010` 새로 생긴 라벨 없음 · mode=phys, mask_mode=spot, area=0.303, halves=[4, 5, 6], dur=6.279, delta_cd=76.562, sigma=0.0
- `train_lum_ext_hazard_067` 새로 생긴 라벨 없음 · mode=phys, mask_mode=spot, area=0.317, halves=[4, 5, 6], dur=7.321, delta_cd=114.627, sigma=0.0
- `train_lum_ext_hazard_082` 새로 생긴 라벨 없음 · mode=phys, mask_mode=spot, area=0.385, halves=[4, 5, 6], dur=6.56, delta_cd=56.558, sigma=0.0
- `train_lum_ext_hazard_127` 새로 생긴 라벨 없음 · mode=add, mask_mode=spot, area=0.696, halves=[4, 5, 6], dur=7.109, delta_cd=143.271, sigma=0.0
- `train_lum_ext_hazard_136` 새로 생긴 라벨 없음 · mode=phys, mask_mode=anchored, area=0.342, halves=[4, 5, 6], dur=7.085, delta_cd=105.186, sigma=0.0
- `train_lum_ext_hazard_196` 새로 생긴 라벨 없음 · mode=gain, mask_mode=anchored, area=0.451, halves=[4, 5, 6], dur=5.817, delta_cd=74.382, sigma=0.7
- `train_lum_flash_hazard_096` 새로 생긴 라벨 없음 · mode=add, mask_mode=anchored, area=0.535, halves=[1, 2, 3], dur=3.323, delta_cd=101.538, sigma=0.4
- `train_lum_flash_hazard_156` 새로 생긴 라벨 없음 · mode=phys, mask_mode=drift, area=0.301, halves=[1, 2, 3], dur=1.847, delta_cd=63.564, sigma=0.0
- `train_pattern_hazard_005` 새로 생긴 라벨 없음 · mask_mode=anchored, pattern=straight, area=0.737, dur=2.688, cpd=2.762, lo=8.539, hi=81.079, sigma=0.7
- `train_pattern_hazard_035` 새로 생긴 라벨 없음 · mask_mode=perspective, pattern=spiral, area=0.766, dur=3.512, cpd=3.539, lo=16.594, hi=117.743, sigma=0.0
- `train_pattern_hazard_044` 새로 생긴 라벨 없음 · mask_mode=static, pattern=straight, area=0.483, dur=2.014, cpd=1.873, lo=4.842, hi=47.212, sigma=0.4
- `train_pattern_hazard_053` 새로 생긴 라벨 없음 · mask_mode=anchored, pattern=concentric, area=0.502, dur=2.309, cpd=1.939, lo=14.656, hi=81.096, sigma=0.7
- `train_pattern_hazard_065` 새로 생긴 라벨 없음 · mask_mode=static, pattern=radial, area=0.499, dur=2.951, cpd=3.327, lo=10.747, hi=102.096, sigma=0.0
- `train_pattern_hazard_068` 새로 생긴 라벨 없음 · mask_mode=static, pattern=checker, area=0.732, dur=1.393, cpd=2.939, lo=10.953, hi=111.902, sigma=0.7
- `train_pattern_hazard_077` 새로 생긴 라벨 없음 · mask_mode=anchored, pattern=radial, area=0.502, dur=3.012, cpd=5.817, lo=13.814, hi=82.954, sigma=0.4
- `train_pattern_hazard_083` 새로 생긴 라벨 없음 · mask_mode=static, pattern=checker, area=0.751, dur=3.12, cpd=4.442, lo=13.537, hi=60.804, sigma=0.0
- `train_pattern_hazard_101` 새로 생긴 라벨 없음 · mask_mode=perspective, pattern=radial, area=0.719, dur=3.745, cpd=5.991, lo=16.097, hi=93.97, sigma=0.0
- `train_pattern_hazard_104` 새로 생긴 라벨 ['휘도 플래시'] · mask_mode=static, pattern=straight, area=0.808, dur=1.47, cpd=4.158, lo=12.932, hi=65.379, sigma=0.4
- `train_pattern_hazard_128` 새로 생긴 라벨 없음 · mask_mode=anchored, pattern=straight, area=0.556, dur=1.34, cpd=4.092, lo=13.21, hi=104.393, sigma=0.7
- `train_pattern_hazard_134` 새로 생긴 라벨 없음 · mask_mode=perspective, pattern=checker, area=0.511, dur=2.128, cpd=2.436, lo=8.048, hi=106.066, sigma=0.0
- `train_pattern_hazard_140` 새로 생긴 라벨 ['휘도 플래시'] · mask_mode=perspective, pattern=straight, area=0.742, dur=1.206, cpd=5.257, lo=8.768, hi=103.065, sigma=0.0
- `train_pattern_hazard_155` 새로 생긴 라벨 없음 · mask_mode=perspective, pattern=spiral, area=0.569, dur=1.558, cpd=1.636, lo=17.213, hi=98.399, sigma=0.4
- `train_pattern_hazard_158` 새로 생긴 라벨 없음 · mask_mode=anchored, pattern=concentric, area=0.506, dur=3.029, cpd=2.902, lo=15.648, hi=123.817, sigma=0.7
- `train_pattern_hazard_194` 새로 생긴 라벨 없음 · mask_mode=anchored, pattern=radial, area=0.501, dur=3.465, cpd=3.48, lo=8.148, hi=57.81, sigma=0.0
- `train_pattern_hazard_197` 새로 생긴 라벨 없음 · mask_mode=anchored, pattern=straight, area=0.497, dur=2.777, cpd=2.793, lo=16.233, hi=77.996, sigma=0.7
- `train_red_ext_hazard_010` 새로 생긴 라벨 ['휘도 연장'] · mode=phys, mask_mode=drift, area=0.648, halves=[4, 5, 6], dur=6.75, sat=0.889, sigma=0.0
- `train_red_ext_hazard_013` 새로 생긴 라벨 ['휘도 연장'] · mode=phys, mask_mode=anchored, area=0.382, halves=[4, 5, 6], dur=7.262, sat=0.95, sigma=0.0
- `train_red_ext_hazard_016` 새로 생긴 라벨 ['휘도 연장'] · mode=phys, mask_mode=multi, area=0.427, halves=[4, 5, 6], dur=7.087, sat=0.864, sigma=0.0
- `train_red_ext_hazard_019` 새로 생긴 라벨 없음 · mode=tint, mask_mode=spot, area=0.323, halves=[4, 5, 6], dur=6.565, sat=0.91, sigma=0.0
- `train_red_ext_hazard_025` 새로 생긴 라벨 없음 · mode=phys, mask_mode=spot, area=0.443, halves=[4, 5, 6], dur=6.38, sat=0.906, sigma=0.0
- `train_red_ext_hazard_034` 새로 생긴 라벨 ['휘도 연장'] · mode=phys, mask_mode=anchored, area=0.5, halves=[4, 5, 6], dur=7.343, sat=0.954, sigma=0.0
- `train_red_ext_hazard_043` 새로 생긴 라벨 ['휘도 연장'] · mode=phys, mask_mode=anchored, area=0.412, halves=[4, 5, 6], dur=6.263, sat=0.872, sigma=0.0
- `train_red_ext_hazard_064` 새로 생긴 라벨 ['휘도 연장'] · mode=phys, mask_mode=drift, area=0.585, halves=[4, 5, 6], dur=7.328, sat=0.955, sigma=0.0
- `train_red_ext_hazard_076` 새로 생긴 라벨 ['휘도 연장', '정적 공간 패턴'] · mode=phys, mask_mode=multi, area=0.408, halves=[4, 5, 6], dur=6.599, sat=0.946, sigma=0.0
- `train_red_ext_hazard_085` 새로 생긴 라벨 없음 · mode=phys, mask_mode=spot, area=0.663, halves=[4, 5, 6], dur=6.089, sat=0.874, sigma=0.0
- `train_red_ext_hazard_115` 새로 생긴 라벨 ['휘도 연장'] · mode=phys, mask_mode=multi, area=0.344, halves=[4, 5, 6], dur=6.212, sat=0.897, sigma=0.0
- `train_red_ext_hazard_121` 새로 생긴 라벨 없음 · mode=tint, mask_mode=spot, area=0.485, halves=[4, 5, 6], dur=6.604, sat=0.897, sigma=0.0
- `train_red_ext_hazard_127` 새로 생긴 라벨 없음 · mode=pair, mask_mode=spot, area=0.35, halves=[4, 5, 6], dur=7.36, sat=0.872, sigma=0.0
- `train_red_ext_hazard_133` 새로 생긴 라벨 없음 · mode=tint, mask_mode=spot, area=0.463, halves=[4, 5, 6], dur=6.224, sat=0.899, sigma=0.0
- `train_red_ext_hazard_160` 새로 생긴 라벨 ['휘도 연장'] · mode=phys, mask_mode=anchored, area=0.613, halves=[4, 5, 6], dur=7.354, sat=0.96, sigma=0.0
- `train_red_ext_hazard_163` 새로 생긴 라벨 없음 · mode=tint, mask_mode=multi, area=0.346, halves=[4, 5, 6], dur=6.654, sat=0.896, sigma=0.0
- `train_red_ext_hazard_175` 새로 생긴 라벨 없음 · mode=phys, mask_mode=anchored, area=0.316, halves=[4, 5, 6], dur=6.59, sat=0.902, sigma=0.0
- `train_red_ext_hazard_181` 새로 생긴 라벨 ['휘도 연장'] · mode=phys, mask_mode=spot, area=0.41, halves=[4, 5, 6], dur=6.337, sat=0.938, sigma=0.0
- `train_red_ext_hazard_184` 새로 생긴 라벨 ['휘도 연장'] · mode=phys, mask_mode=spot, area=0.347, halves=[4, 5, 6], dur=5.916, sat=0.955, sigma=0.0
- `train_red_ext_hazard_187` 새로 생긴 라벨 ['휘도 연장'] · mode=phys, mask_mode=spot, area=0.397, halves=[4, 5, 6], dur=6.429, sat=0.959, sigma=0.0
- `train_red_ext_hazard_199` 새로 생긴 라벨 없음 · mode=pair, mask_mode=spot, area=0.396, halves=[4, 5, 6], dur=7.224, sat=0.916, sigma=0.0
- `train_red_flash_hazard_015` 새로 생긴 라벨 없음 · mode=phys, mask_mode=spot, area=0.481, halves=[1, 2, 3], dur=2.911, sat=0.917, sigma=0.0
- `train_red_flash_hazard_018` 새로 생긴 라벨 ['휘도 플래시'] · mode=phys, mask_mode=anchored, area=0.385, halves=[1, 2, 3], dur=3.676, sat=0.955, sigma=0.0
- `train_red_flash_hazard_033` 새로 생긴 라벨 ['휘도 플래시'] · mode=phys, mask_mode=drift, area=0.37, halves=[1, 2, 3], dur=2.521, sat=0.899, sigma=0.0
- `train_red_flash_hazard_036` 새로 생긴 라벨 없음 · mode=tint, mask_mode=spot, area=0.383, halves=[1, 2, 3], dur=2.603, sat=0.867, sigma=0.0
- `train_red_flash_hazard_042` 새로 생긴 라벨 ['휘도 플래시'] · mode=phys, mask_mode=drift, area=0.533, halves=[1, 2, 3], dur=2.602, sat=0.929, sigma=0.0
- `train_red_flash_hazard_045` 새로 생긴 라벨 ['휘도 플래시'] · mode=phys, mask_mode=multi, area=0.347, halves=[1, 2, 3], dur=1.907, sat=0.875, sigma=0.0
- `train_red_flash_hazard_054` 새로 생긴 라벨 없음 · mode=phys, mask_mode=static, area=0.589, halves=[1, 2, 3], dur=3.555, sat=0.899, sigma=0.0
- `train_red_flash_hazard_057` 새로 생긴 라벨 ['휘도 플래시'] · mode=phys, mask_mode=static, area=0.308, halves=[1, 2, 3], dur=2.735, sat=0.884, sigma=0.0
- `train_red_flash_hazard_078` 새로 생긴 라벨 ['휘도 플래시'] · mode=tint, mask_mode=spot, area=0.44, halves=[1, 2, 3], dur=2.157, sat=0.968, sigma=0.0
- `train_red_flash_hazard_081` 새로 생긴 라벨 ['휘도 플래시'] · mode=phys, mask_mode=anchored, area=0.518, halves=[1, 2, 3], dur=1.602, sat=0.964, sigma=0.0
- `train_red_flash_hazard_096` 새로 생긴 라벨 없음 · mode=phys, mask_mode=spot, area=0.345, halves=[1, 2, 3], dur=3.586, sat=0.966, sigma=0.0
- `train_red_flash_hazard_105` 새로 생긴 라벨 없음 · mode=pair, mask_mode=spot, area=0.404, halves=[1, 2, 3], dur=3.113, sat=0.901, sigma=0.0
- `train_red_flash_hazard_132` 새로 생긴 라벨 ['휘도 플래시'] · mode=phys, mask_mode=spot, area=0.322, halves=[1, 2, 3], dur=1.544, sat=0.93, sigma=0.0
- `train_red_flash_hazard_153` 새로 생긴 라벨 ['휘도 플래시'] · mode=phys, mask_mode=spot, area=0.442, halves=[1, 2, 3], dur=3.098, sat=0.963, sigma=0.0
- `train_red_flash_hazard_162` 새로 생긴 라벨 ['휘도 플래시'] · mode=phys, mask_mode=drift, area=0.411, halves=[1, 2, 3], dur=2.994, sat=0.926, sigma=0.0
- `train_red_flash_hazard_189` 새로 생긴 라벨 없음 · mode=phys, mask_mode=spot, area=0.32, halves=[1, 2, 3], dur=3.715, sat=0.867, sigma=0.0
- `train_red_flash_hazard_198` 새로 생긴 라벨 없음 · mode=tint, mask_mode=spot, area=0.351, halves=[1, 2, 3], dur=2.788, sat=0.89, sigma=0.0
