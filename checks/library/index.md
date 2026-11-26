# 위험 영상 분류 모음 (scripts/library_build.py)

> 실제 영상 186 개 (Kaya 22 · South 위험 163 · 사용자 1; 패턴영상2 = Kaya 6). 세 근거(출처 라벨 · G · 팀원 판 v1) 중 둘 이상 = 확정, 하나 = 보류.

## 폴더별 개수

| 폴더 | 개수 |
|---|---|
| 01_휘도깜빡임 | 20 |
| 05_정지패턴 | 51 |
| 07_복합 | 8 |
| 99_분류보류 | 107 |

## 종류별 (확정 / 보류)

| 종류 | 확정 | 보류 |
|---|---|---|
| 휘도 깜빡임 | 28 | 91 |
| 적색 깜빡임 | 5 | 14 |
| 연장 깜빡임 | 0 | 2 |
| 빠른 장면 전환 | 0 | 2 |
| 정지 패턴 | 54 | 66 |
| 변화 패턴 | 1 | 2 |

## 항목

| id | 폴더 | 확정 | 보류(근거 하나) | 출처 라벨 | G | 팀원 판 v1 |
|---|---|---|---|---|---|---|
| kaya_1 | 01_휘도깜빡임 | 휘도 깜빡임 | - | 휘도 깜빡임 (Kaya 저자 보정 도구 general) | 휘도 플래시 | 휘도 플래시 |
| kaya_2 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (Kaya 저자 보정 도구 general) | - | 정적 공간 패턴 |
| kaya_3 | 01_휘도깜빡임 | 휘도 깜빡임 | 정지 패턴 | 휘도 깜빡임 (Kaya 저자 보정 도구 general) | 휘도 플래시 | 정적 공간 패턴 |
| kaya_4 | 01_휘도깜빡임 | 휘도 깜빡임 | 정지 패턴 | 휘도 깜빡임 (Kaya 저자 보정 도구 general) | 휘도 플래시 | 교집합, 정적 공간 패턴, 휘도 플래시 |
| kaya_5 | 01_휘도깜빡임 | 휘도 깜빡임 | 정지 패턴 | 휘도 깜빡임 (Kaya 저자 보정 도구 general) | 휘도 플래시 | 정적 공간 패턴, 휘도 플래시 |
| kaya_6 | 01_휘도깜빡임 | 휘도 깜빡임 | 정지 패턴, 변화 패턴 | 휘도 깜빡임 (Kaya 저자 보정 도구 general) | 변화 공간 패턴, 휘도 플래시 | 정적 공간 패턴, 휘도 플래시 |
| kaya_7 | 01_휘도깜빡임 | 휘도 깜빡임 | 정지 패턴 | 휘도 깜빡임 (Kaya 저자 보정 도구 general) | 휘도 플래시 | 정적 공간 패턴, 휘도 플래시 |
| kaya_8 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (Kaya 저자 보정 도구 general) | - | - |
| kaya_9 | 07_복합 | 휘도 깜빡임, 적색 깜빡임 | 정지 패턴 | 적색 깜빡임, 휘도 깜빡임 (Kaya 저자 보정 도구 both, general, red) | 적색 플래시, 휘도 플래시 | 교집합, 적색 플래시, 정적 공간 패턴, 휘도 플래시 |
| kaya_10 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (Kaya 저자 보정 도구 general) | - | - |
| kaya_11 | 07_복합 | 휘도 깜빡임, 적색 깜빡임 | - | 적색 깜빡임, 휘도 깜빡임 (Kaya 저자 보정 도구 both, general, red) | 적색 플래시, 휘도 플래시 | 교집합, 적색 플래시, 휘도 플래시 |
| kaya_12 | 01_휘도깜빡임 | 휘도 깜빡임 | - | 휘도 깜빡임 (Kaya 저자 보정 도구 general) | 휘도 플래시 | 교집합, 휘도 플래시 |
| kaya_13 | 01_휘도깜빡임 | 휘도 깜빡임 | - | 휘도 깜빡임 (Kaya 저자 보정 도구 general) | 휘도 플래시 | 교집합 |
| kaya_14 | 01_휘도깜빡임 | 휘도 깜빡임 | - | 휘도 깜빡임 (Kaya 저자 보정 도구 general) | 휘도 플래시 | 휘도 플래시 |
| kaya_15 | 99_분류보류 | - | 휘도 깜빡임, 적색 깜빡임 | 적색 깜빡임, 휘도 깜빡임 (Kaya 저자 보정 도구 both, general) | - | - |
| kaya_16 | 01_휘도깜빡임 | 휘도 깜빡임 | - | 휘도 깜빡임 (Kaya 저자 보정 도구 general) | 휘도 플래시 | 휘도 플래시 |
| kaya_17 | 01_휘도깜빡임 | 휘도 깜빡임 | 적색 깜빡임 | 적색 깜빡임, 휘도 깜빡임 (Kaya 저자 보정 도구 both, general, red) | 휘도 플래시 | 휘도 플래시 |
| kaya_18 | 01_휘도깜빡임 | 휘도 깜빡임 | - | 휘도 깜빡임 (Kaya 저자 보정 도구 general) | 휘도 플래시 | 교집합, 휘도 플래시 |
| kaya_19 | 07_복합 | 휘도 깜빡임, 적색 깜빡임 | 정지 패턴 | 적색 깜빡임, 휘도 깜빡임 (Kaya 저자 보정 도구 both, general, red) | 휘도 플래시 | 교집합, 적색 플래시, 정적 공간 패턴, 휘도 플래시 |
| kaya_20 | 07_복합 | 휘도 깜빡임, 적색 깜빡임 | 연장 깜빡임, 정지 패턴 | 적색 깜빡임, 휘도 깜빡임 (Kaya 저자 보정 도구 both, general) | 적색 플래시, 휘도 플래시 | 교집합, 적색 연장, 적색 플래시, 정적 공간 패턴, 휘도 연장, 휘도 플래시 |
| kaya_21 | 07_복합 | 휘도 깜빡임, 적색 깜빡임 | 연장 깜빡임, 빠른 장면 전환, 정지 패턴, 변화 패턴 | 적색 깜빡임, 휘도 깜빡임 (Kaya 저자 보정 도구 both, general) | 변화 공간 패턴, 적색 플래시, 휘도 플래시 | 교집합, 빠른 장면 전환, 적색 플래시, 정적 공간 패턴, 휘도 연장, 휘도 플래시 |
| kaya_22 | 01_휘도깜빡임 | 휘도 깜빡임 | 정지 패턴 | 휘도 깜빡임 (Kaya 저자 보정 도구 general) | 휘도 플래시 | 정적 공간 패턴, 휘도 플래시 |
| south_giphy-127 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_giphy-155 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_giphy-16 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_giphy-160 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_giphy-194 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_giphy-199 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_giphy-200 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_giphy-206 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_giphy-217 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_giphy-223 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_giphy-229 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_giphy-231 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_giphy-240 | 99_분류보류 | - | 적색 깜빡임, 정지 패턴 | 적색 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_giphy-273 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_giphy-274 | 99_분류보류 | - | 정지 패턴 | 정지 패턴 (PhotosensitivityPal) | - | - |
| south_giphy-289 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_giphy-314 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_giphy-315 | 05_정지패턴 | 정지 패턴 | 휘도 깜빡임 | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴, 휘도 플래시 |
| south_giphy-349 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_giphy-356 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_giphy-358 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_giphy-371 | 05_정지패턴 | 정지 패턴 | 휘도 깜빡임 | 정지 패턴, 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_giphy-373 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_giphy-376 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_giphy-379 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_giphy-383 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_giphy-387 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_giphy-4 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_giphy-444 | 07_복합 | 휘도 깜빡임, 정지 패턴 | - | 정지 패턴, 휘도 깜빡임 (PhotosensitivityPal) | 휘도 플래시 | 교집합, 정적 공간 패턴, 휘도 플래시 |
| south_giphy-465 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_giphy-89 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_giphy-91 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_giphy-93 | 99_분류보류 | - | 정지 패턴 | 정지 패턴 (PhotosensitivityPal) | - | - |
| south_tenor-126 | 99_분류보류 | - | 적색 깜빡임, 정지 패턴 | 적색 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tenor-132 | 99_분류보류 | - | 휘도 깜빡임, 적색 깜빡임 | 적색 깜빡임, 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_tenor-15 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tenor-153 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tenor-19 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tenor-196 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_tenor-199 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tenor-203 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 정지 패턴, 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_tenor-222 | 99_분류보류 | - | 정지 패턴 | 정지 패턴 (PhotosensitivityPal) | - | - |
| south_tenor-23 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tenor-233 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tenor-237 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_tenor-238 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tenor-255 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_tenor-257 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tenor-271 | 99_분류보류 | - | 휘도 깜빡임, 빠른 장면 전환 | 휘도 깜빡임 (PhotosensitivityPal) | - | 빠른 장면 전환 |
| south_tenor-274 | 01_휘도깜빡임 | 휘도 깜빡임 | - | 휘도 깜빡임 (PhotosensitivityPal) | - | 휘도 플래시 |
| south_tenor-284 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_tenor-288 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tenor-290 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tenor-292 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_tenor-3 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tenor-31 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_tenor-312 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tenor-344 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tenor-360 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tenor-374 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tenor-380 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tenor-394 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tenor-403 | 05_정지패턴 | 정지 패턴 | 휘도 깜빡임 | 정지 패턴, 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tenor-412 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_tenor-431 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tenor-433 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tenor-436 | 01_휘도깜빡임 | 휘도 깜빡임 | - | 휘도 깜빡임 (PhotosensitivityPal) | 휘도 플래시 | 휘도 플래시 |
| south_tenor-441 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_tenor-446 | 99_분류보류 | - | 정지 패턴 | 정지 패턴 (PhotosensitivityPal) | - | - |
| south_tenor-453 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_tenor-460 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_tenor-471 | 01_휘도깜빡임 | 휘도 깜빡임 | - | 휘도 깜빡임 (PhotosensitivityPal) | 휘도 플래시 | - |
| south_tenor-473 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_tenor-480 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_tenor-491 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tenor-50 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tenor-56 | 99_분류보류 | - | 적색 깜빡임 | 적색 깜빡임 (PhotosensitivityPal) | - | - |
| south_tenor-64 | 99_분류보류 | - | 정지 패턴 | 정지 패턴 (PhotosensitivityPal) | - | - |
| south_tenor-65 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tenor-76 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tenor-8 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_tenor-81 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tenor-86 | 99_분류보류 | - | 정지 패턴 | 정지 패턴 (PhotosensitivityPal) | - | - |
| south_tumblr-128 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tumblr-143 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tumblr-153 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tumblr-16 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_tumblr-171 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tumblr-178 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_tumblr-194 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tumblr-200 | 99_분류보류 | - | 정지 패턴 | 정지 패턴 (PhotosensitivityPal) | - | - |
| south_tumblr-204 | 01_휘도깜빡임 | 휘도 깜빡임 | - | 휘도 깜빡임 (PhotosensitivityPal) | 휘도 플래시 | 휘도 플래시 |
| south_tumblr-234 | 05_정지패턴 | 정지 패턴 | 휘도 깜빡임 | 정지 패턴, 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tumblr-241 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tumblr-260 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tumblr-261 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tumblr-262 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tumblr-265 | 99_분류보류 | - | 적색 깜빡임, 정지 패턴 | 적색 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tumblr-267 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tumblr-269 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tumblr-30 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tumblr-308 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_tumblr-31 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tumblr-310 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_tumblr-318 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tumblr-321 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_tumblr-329 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tumblr-345 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tumblr-346 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tumblr-361 | 99_분류보류 | - | 적색 깜빡임 | 적색 깜빡임 (PhotosensitivityPal) | - | - |
| south_tumblr-385 | 99_분류보류 | - | 정지 패턴 | 정지 패턴 (PhotosensitivityPal) | - | - |
| south_tumblr-403 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tumblr-423 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_tumblr-431 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_tumblr-447 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tumblr-479 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_tumblr-480 | 99_분류보류 | - | 정지 패턴 | 정지 패턴 (PhotosensitivityPal) | - | - |
| south_tumblr-486 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_tumblr-488 | 05_정지패턴 | 정지 패턴 | 휘도 깜빡임 | 정지 패턴, 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tumblr-490 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_tumblr-59 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_tumblr-62 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_tumblr-7 | 01_휘도깜빡임 | 휘도 깜빡임 | - | 휘도 깜빡임 (PhotosensitivityPal) | - | 휘도 플래시 |
| south_tumblr-70 | 07_복합 | 휘도 깜빡임, 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | 휘도 플래시 | 정적 공간 패턴, 휘도 플래시 |
| south_tumblr-79 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_twitter-102 | 01_휘도깜빡임 | 휘도 깜빡임 | - | 휘도 깜빡임 (PhotosensitivityPal) | 휘도 플래시 | - |
| south_twitter-110 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_twitter-131 | 99_분류보류 | - | 정지 패턴 | 정지 패턴 (PhotosensitivityPal) | - | - |
| south_twitter-135 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_twitter-140 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_twitter-152 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_twitter-167 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_twitter-176 | 01_휘도깜빡임 | 휘도 깜빡임 | - | 휘도 깜빡임 (PhotosensitivityPal) | 휘도 플래시 | 휘도 플래시 |
| south_twitter-177 | 99_분류보류 | - | 휘도 깜빡임, 적색 깜빡임 | 적색 깜빡임, 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_twitter-203 | 99_분류보류 | - | 정지 패턴 | 정지 패턴 (PhotosensitivityPal) | - | - |
| south_twitter-211 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_twitter-218 | 99_분류보류 | - | 정지 패턴 | 정지 패턴 (PhotosensitivityPal) | - | - |
| south_twitter-22 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_twitter-245 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_twitter-264 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_twitter-270 | 99_분류보류 | - | 휘도 깜빡임, 적색 깜빡임 | 적색 깜빡임, 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_twitter-271 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_twitter-277 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_twitter-338 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_twitter-339 | 99_분류보류 | - | 휘도 깜빡임, 적색 깜빡임 | 적색 깜빡임, 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_twitter-369 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_twitter-371 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_twitter-372 | 99_분류보류 | - | 적색 깜빡임, 정지 패턴 | 적색 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_twitter-387 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_twitter-400 | 99_분류보류 | - | 적색 깜빡임 | 적색 깜빡임 (PhotosensitivityPal) | - | - |
| south_twitter-422 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_twitter-447 | 99_분류보류 | - | 정지 패턴 | 정지 패턴 (PhotosensitivityPal) | - | - |
| south_twitter-479 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_twitter-494 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_twitter-498 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_twitter-5 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_twitter-50 | 05_정지패턴 | 정지 패턴 | - | 정지 패턴 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_twitter-51 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_twitter-62 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_twitter-64 | 99_분류보류 | - | 적색 깜빡임 | 적색 깜빡임 (PhotosensitivityPal) | - | - |
| south_twitter-71 | 99_분류보류 | - | 휘도 깜빡임, 정지 패턴 | 휘도 깜빡임 (PhotosensitivityPal) | - | 정적 공간 패턴 |
| south_twitter-81 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| south_twitter-98 | 99_분류보류 | - | 휘도 깜빡임 | 휘도 깜빡임 (PhotosensitivityPal) | - | - |
| user_pattern1 | 07_복합 | 휘도 깜빡임, 정지 패턴, 변화 패턴 | - | 정지 패턴|변화 패턴 (사용자 지정 패턴 영상) | 변화 공간 패턴, 정적 공간 패턴, 휘도 플래시 | 정적 공간 패턴, 휘도 플래시 |
