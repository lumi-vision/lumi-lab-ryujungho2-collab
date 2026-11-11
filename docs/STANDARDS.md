# 판정 기준 근거 보고서: 휘도 깜빡임 · 적색 깜빡임 · 줄무늬 패턴

> 작성 2026-11-09 · 갱신 2026-11-10: 패턴 문헌 전수 조사 결과는 [PATTERN_LITERATURE.md](PATTERN_LITERATURE.md). 논문과 주간 발표에 쓸 근거 문서입니다. 모든 기준값에 **출처 등급**을 붙였습니다.
> - **원문**: 표준이나 논문 원문에서 직접 확인함 (인용 포함)
> - **문헌 간접**: 관련 문헌 근거는 있지만 그 숫자를 그대로 쓰지는 않음
> - **구현 선택**: 문헌에 숫자가 없어서 우리(또는 팀원)가 정함
> - **미확인**: 출처를 아직 찾지 못함
>
> 원문 확인에 쓴 자료: ITU-R BT.1702 -0·-1·-2·-3 PDF, WCAG 2.0·2.1·2.2와 Understanding 2.3.1, Ofcom Guidance Notes(2017 Annex 1, Wayback 사본), Jordan & Vanderheiden 2024(PMC11872230), Jordan 2025(PMC12249941), pse-test-media README, Wilkins 1979·2005와 EFA 2005 초록(PubMed).
> 원문을 보지 못한 자료: ISO 9241-391(유료), Fisher 2005 전문, Harding & Jeavons 1994.

## 현재 상태 한눈에

| 위험 종류 | 근거 표준 | 우리 판정기 | 검증 | 우리 합성 |
|---|---|---|---|---|
| 휘도 깜빡임 | ITU-R BT.1702-3 Guideline 1 | ✅ 구현 | ✅ pse-test-media 210개 중 201개 일치, 위험 놓침 0 (A7 과대평가 주의) | ✅ 오라클 모양 4종 |
| 적색 깜빡임 | WCAG 2.2 / ISO 9241-391 (수식), ITU는 정의 없음 | ✅ 구현 (2026-11-10, `analyze_red`) | ✅ pse 적색 패턴 208/208 (`checks/red/README.md`), 1080p 세트 96개 중 화면 25% 세트 48/48 (세트 목록 CSV 기준, JSON 모순 16개 — `checks/red/sets.md`) | ✅ 시연 영상 (`checks/demo_hazards`) |
| 줄무늬 패턴 | ITU-R BT.1702-3 Guideline 2 / Ofcom | △ 구현됨, 검증 전 | ❌ | ❌ 없음 |

---

## 0. 공통: 픽셀 값 → 밝기

```
① 0~1로:        c = 8비트값 / 255
② 선형화:        c ≤ 0.04045 → c / 12.92
                 c > 0.04045 → ((c + 0.055) / 1.055)^2.4
③ 상대 휘도:     Y = 0.2126·R + 0.7152·G + 0.0722·B          (R, G, B = 선형값, Y는 0~1)
④ 물리 휘도:     L = 200 · Y   [cd/m²]
```

| 항목 | 값 | 등급 | 근거 |
|---|---|---|---|
| ② 선형화 식 | 0.04045, 12.92, 2.4 | **원문** | WCAG relative luminance 정의(IEC 61966-2-1 sRGB). 2021년 5월 이전 WCAG는 0.03928이었지만 실질 영향 없음이 명시됨 |
| ③ 가중치 | 0.2126 / 0.7152 / 0.0722 | **원문** | WCAG relative luminance, ITU-R BT.709 |
| ④ 흰색 = 200 cd/m² | 200 | **문헌 간접** | 표준 원문 수치가 아니라 Binnie 2002의 BT.500 시청 가정. WCAG의 상대 휘도 0.1·0.8이 200을 곱하면 ITU의 20·160 cd/m²와 같아짐. 노트북 감마 민감도 시험: BT.1886으로 바꾸면 pse 경계 패턴 70개 중 7개가 뒤집힘(`checks/rules/gamma_sensitivity.md`) |

---

## 1. 휘도 깜빡임

### 1-1. 표준 원문 (ITU-R BT.1702-3, 2023, Guideline 1)

> "When screen luminance of the darker image is below 160 cd/m², a potentially harmful sequence of flashes occurs where there is a difference of 20 cd/m² or more between the screen luminance of the darker and the brighter images"
>
> "a sequence of flashes is not permitted when both the following occur: the combined area of flashes occurring concurrently occupies more than 25% of the displayed screen area; and there are more than three flashes (i.e. six changes in luminance as described above) within any one-second period. For clarification, successive flashes for which the leading edges are separated by 360 ms or more are acceptable in a 50 Hz environment or separated by 334 ms or more are acceptable in a 60 Hz environment"

### 1-2. 수식

```
픽셀 전이:   |L[t,p] − ref[p]| ≥ 20   그리고   min(L[t,p], ref[p]) < 160
             (ref = 직전 전이 이후의 극값, 방향이 바뀔 때만 1번)
동시 면적:   A[t] = 같은 방향으로 전이한 픽셀 수 / 전체 픽셀 수   (최근 c 프레임, c = round(fps/15))
화면 전이:   A[t] > 0.25 인 프레임
위험:        아무 1초 창(round(fps) 프레임)에 화면 전이 ≥ 7번   (= 깜빡임 > 3번)
```

### 1-3. 기준값과 출처 등급

| 기준 | 값 | 경계 | 등급 | 근거 |
|---|---|---|---|---|
| 밝기 변화 | 20 cd/m² | 이상(≥) | **원문** | ITU-3 "20 cd/m² or more" |
| 어두운 쪽 | 160 cd/m² | 미만(<) | **원문** | ITU-3 "below 160 cd/m²" |
| 동시 면적 | 25% | 초과(>) | **원문** | ITU-3 "more than 25%" |
| 횟수 | 3번 | 초과(>) | **원문** | ITU-3 "more than three flashes (i.e. six changes)" |
| 334 ms 간격 허용 | 334 ms | 이상이면 허용 | **원문** | ITU-3 (60 Hz). **rockrovks 판별기 v2의 334 ms가 이것** |
| 160 이상에서 Michelson 1/17 | 1/17 | 초과 | **원문** | ITU-3, **HDR 전용**. 우리는 SDR만 다룸 → 미적용 |
| "동시" = 최근 c 프레임 (D3) | 30fps에서 2프레임 | | **구현 선택** | 원문은 "concurrently"만 있고 정량 정의 없음. 근거: [RULES_SPEC.md](RULES_SPEC.md) D3 |
| 같은 방향 변화는 1번 (D1), 방향별 면적 (D2), 두 방향 동시 1번 (D4) | | | **구현 선택** | 근거와 시험: RULES_SPEC D1·D2·D4 |
| 극값 기준 전이 (히스테리시스) | | | **구현 선택** | 20 미만 두 단계로 나눈 변화를 1번으로 셈 → pse 불일치 유형 A(4건)의 원인 |
| 면적 = 화면 비율 | | | **원문**(ITU) | WCAG·trace24는 시야각 기준(341×256 px, 416×416 CSS px)이라 다름 |

### 1-4. 숫자의 의학적 의미 (논문 서론용)

| 숫자 | 유래 | 등급 |
|---|---|---|
| 초당 3회 | Jeavons & Harding 170명 간헐 광자극: 3 Hz에서 3%, 10 Hz 약 65%, 정점 16 Hz. 판정 규칙의 절단점은 방송 녹화 약 3시간의 **오경보 수**로 결정(Binnie 2002) | 문헌 간접 (보관 브랜치 `docs/PSE_EVIDENCE.md`) |
| 20 / 160 cd/m² | 줄무늬 실험의 휘도·대비 함수를 깜빡임에 빌려 옴. **깜빡임 실험값이 아님**(Binnie 2002) | 문헌 간접 |
| 25% | 시각피질 모델 + 오경보 분석(EFA 2005, Wilkins 2005) | 문헌 간접 |

---

## 2. 적색 깜빡임

### 2-1. 표준별 정의

| 표준 | 포화 빨강 | 색 차이 | 등급 |
|---|---|---|---|
| ITU-R BT.1702 (-0~-3) | **정의 없음** | 정의 없음 | 원문: "Irrespective of luminance, a transition to or from a saturated red is also potentially harmful." 부속서 3은 "more explicit definitions… will also be required"라고 함 |
| Ofcom | 정의 없음 | 정의 없음 | 원문 (ITU와 같은 문장) |
| WCAG 2.0 / 2.1 초판 (옛 공식) | R/(R+G+B) ≥ 0.8 | (R−G−B)×320 변화 > 20 | 원문 |
| **WCAG 2.2** (2023) / WCAG 2.1 재발행판 | R/(R+G+B) **≥ 0.8** | CIE 1976 UCS 거리 **> 0.2** | 원문: "one transition is either to or from a state with a value R/(R + G + B) that is greater than or equal to 0.8, and the difference between states is more than 0.2 (unitless) in the CIE 1976 UCS chromaticity diagram" |
| ISO 9241-391:2016 | ≥ 0.8 | > 0.2 | **미확인** (WCAG가 출처로 인용한 2차 정보) |
| trace24 (Jordan & Vanderheiden 2024) | ≥ 0.8 | **≥ 0.2** | 원문. WCAG와 경계 연산자가 다름 |
| rockrovks 판별기 | 선형 R 비율 ≥ 0.8 | Δu′v′ > 0.2 | 코드 확인 (`vdcoreops_v2.py`) |
| EA IRIS | | **옛 공식** (R−G−B)×320 > 20 | 코드 확인 (`appsettings.json`의 `FlashThreshold: 20`) |

### 2-2. 수식 (WCAG 2.2 Understanding 2.3.1)

```
① 선형화:      §0의 ②와 같음 → R, G, B (선형값)
② 포화 빨강:   s = R / (R + G + B) ≥ 0.8          ← 반드시 선형값으로
③ XYZ:         X = 0.4124564·R + 0.3575761·G + 0.1804375·B
               Y = 0.2126729·R + 0.7151522·G + 0.0721750·B
               Z = 0.0193339·R + 0.1191920·G + 0.9503041·B
④ u′v′:        D = X + 15·Y + 3·Z
               u′ = 4·X / D,   v′ = 9·Y / D
⑤ 색 차이:     Δu′v′ = √( (u′₁ − u′₂)² + (v′₁ − v′₂)² )
⑥ 적색 전이:   Δu′v′ > 0.2  그리고  두 상태 중 한쪽이 포화 빨강
⑦ 적색 깜빡임: 반대 방향 적색 전이 한 쌍 = 1번
               면적 > 25%, 1초에 > 3번이면 위험 (휘도와 같은 규칙)
```

### 2-3. 우리 검산: pse-test-media 색표 12색 (2026-11-09)

포화 빨강 비율을 **선형값**으로 계산해야 pse-test-media 표와 맞습니다. 감마 값(0~255 그대로)으로 계산하면 맞지 않습니다.

| 색 (R,G,B) | 감마 값으로 계산 | **선형값으로 계산** | pse 표 |
|---|---|---|---|
| (205, 78, 78) | 0.5679 | **0.8003** | 0.8003 |
| (220, 55, 104) | 0.5805 | **0.8021** | 0.8021 |
| (233, 13, 122) | 0.6332 | **0.8040** | 0.8040 |

pse-test-media의 색 쌍은 **Δu′v′ = 0.2의 바로 위·아래로 설계**되어 있고, 휘도 Y는 모두 약 0.19로 같습니다. 그래서 휘도 깜빡임이 섞이지 않은 순수 적색 시험입니다.

| 빨강 | 상대 색 | Δu′v′ | 설계 의도 |
|---|---|---|---|
| (205,78,78) | (31,134,134) | **0.2046** | 위험 (f) |
| (205,78,78) | (60,132,132) | **0.1948** | 색 차이 미달 (d) |
| (220,55,104) | (119,109,197) / (129,107,193) | 0.2045 / 0.1941 | 위험 / 미달 |
| (233,13,122) | (190,37,248) / (193,36,241) | 0.2047 / 0.1949 | 위험 / 미달 |
| (196,74,74) | (29,127,127) / (57,125,125) | 0.2050 / 0.1949 | 위험 / 미달 |

### 2-4. 구현할 때 정해야 할 것

| 결정 | 선택지 | 제안 | 등급 |
|---|---|---|---|
| 색 차이 경계 | > 0.2 (WCAG 2.2) / ≥ 0.2 (trace24) | **> 0.2**. pse 시험 패턴은 0.195 / 0.205라 둘 다 같은 결과 | 원문 둘 중 선택 |
| 검정 (R+G+B = 0) | 포화 빨강 아님 + 색도를 D65 백색(u′ 0.1978, v′ 0.4683)으로 둠 | 그렇게 두면 빨강↔검정 Δ ≈ 0.26 → 적색 깜빡임으로 잡힘 | **구현 선택** (W3C 이슈 #4285 미해결) |
| 거의 검정인 빨강 | #080101도 포화 빨강이 됨 | 휘도 하한을 둘지 결정 필요 | **구현 선택** (J&V 2024가 지적) |
| 판정 단위 | 픽셀 / 블록 평균 | 픽셀 (휘도와 같게). ITU 부속서 3은 블록 평균 언급 | 구현 선택 |
| 옛 공식 | IRIS가 옛 공식 사용 | 비교용으로 함께 계산해 기록 | 구현 선택 |

### 2-5. 구현 (2026-11-10, `measure.analyze_red`)

```
ⓡ① 상태:     포화 빨강 = 선형 R/(R+G+B) ≥ 0.8,  색도 u′v′ (검정은 D65 백색점)
ⓡ② 변화:     바로 앞 프레임과 비교. 적색 전이 = Δu′v′ > 0.2 이고 '빨강 아님 → 포화 빨강'(+1) 또는 반대(−1)
ⓡ③ 깜빡임 쌍: 픽셀마다 색 변화(Δu′v′ ≥ 0.02)를 시간 순서로 (1,2), (3,4), … 짝짓고,
              두 변화가 모두 적색 전이이고 방향이 반대인 쌍만 셈. 짝이 없는 마지막 변화는 적색 전이면 셈
ⓡ④ 이후:     휘도와 같은 방향별 동시 면적 > 25%, 1초에 화면 전이 ≥ 7번이면 위험 (공통 함수 flash_from_transitions)
```

| 선택 | 등급 | 근거 |
|---|---|---|
| 바로 앞 상태와 비교 | 원문 해석 | WCAG 2.2 "the difference between states". 처음에 '마지막 전이 때의 상태'와 비교했더니 s014_multi 과잉 검출 → 바꿈 |
| 깜빡임 쌍 규칙 (한쪽이라도 미달이면 그 쌍 무효) | 원문 해석 + pse 설계로 검증 | ITU "a flash is a pair of opposing changes". 쌍 규칙이 없으면 d015·dr011·dr015·s013 과잉 검출(204/208), 넣으면 208/208 |
| 변화로 볼 최소 색도 차 0.02 | **구현 선택** | 쌍을 짓기 위한 잡음 문턱. 민감도 분석 대상 |
| 검정의 색도 = D65 | **구현 선택** | W3C 이슈 #4285 미해결 |
| 서서히 바뀌는 색은 세지 않음 | 한계 | 바로 앞 상태와 비교하기 때문. pse 설계도 같은 해석 |

---

## 3. 줄무늬 패턴

### 3-1. 표준 원문 (ITU-R BT.1702-3 Guideline 2, 정보성 부속서. Ofcom과 같은 문구)

> "Potentially harmful regular patterning may occur if an image contains more than five light and dark pairs of clearly discernible stripes in any orientation when either of the following two conditions apply:
> – the stripes are stationary and the pattern occupies more than 40% of the displayed screen area; or
> – the stripes change direction, oscillate, flash, or reverse in contrast and the pattern occupies more than 25% of the displayed screen area.
> For a pattern to be considered potentially harmful, the difference between the luminance of the darker and lighter bars of the pattern will be the same as … a potentially harmful flash described in Guideline 1.
> If the patterns obviously flow smoothly across, into, or out of the screen in one direction they are [exempt]."

ITU-R BT.1702 -0·-1·-2판에는 패턴 수치가 없고, -3판(2023)에서 처음 들어왔습니다. 내용은 구 ITC(2001) 규칙을 옮긴 것입니다. WCAG와 trace24에는 패턴 기준이 없습니다.

### 3-2. 수식 (노트북에서 구현한 `measure.analyze_pattern`)

```
① 스캔:      12방향(0°, 15°, …, 165°)으로 화면 전체를 1픽셀 간격 직선으로 읽음
② 줄 경계:   휘도 프로파일에서 극값을 찾음 (반대 방향으로 20 cd/m² 이상 물러나면 극값 확정)
③ 유효한 줄: 이웃 극값 사이(반 주기 h 픽셀)가
             min(두 극값) < 160 cd/m²   그리고   ppd / (2h) ≥ 1  [cpd]
             (상한 8 cpd 는 픽셀마다 모든 방향 cpd 의 최댓값으로 판정 — 2026-11-10)
             (ppd = 시야각 1°당 픽셀 수, 시청 조건으로 계산)
④ 줄무늬:    유효한 줄이 끊김 없이 9개 이상 (= 극값 10개 = 5쌍) → 그 구간 픽셀 표시
⑤ 움직임:    줄의 법선 방향 '위치×시간' 휘도 단면(±0.5초)을 2차원 FFT
             시간 에너지 비율 < 10% → 정지 / |DI| ≥ 0.5 → 흐름 / 그 밖 → 반전(진동·명암 반전·점멸)
             DI = (E(ω·k>0) − E(ω·k<0)) / (E(ω·k>0) + E(ω·k<0))   (Adelson & Bergen 1985 운동 에너지)
⑥ 면적:      정지 > 40%, 반전 > 25%, 흐름은 제외 → 후보 프레임
⑦ 위험:      아무 1초 창에서 후보 프레임이 0.5초 이상
```

### 3-3. 기준값과 출처 등급

| 기준 | 우리 값 | 원문 값 | 등급 | 근거 / 차이 |
|---|---|---|---|---|
| 쌍 수 | **5쌍 이상** (≥5) | **5쌍 초과** (>5, 즉 6쌍 이상) | 원문 + **구현 선택** | ITU-3·Ofcom 원문은 "more than five". 팀원 판별기를 따라 1쌍 엄격하게 둠. EFA 2005는 변하는 패턴 ≤5쌍, 정지·표류 8**쌍**(본문 "Eight light-dark pairs of stripes"로 확인, 2026-11-10) |
| 줄 사이 휘도 차 | 20 cd/m² | 20 cd/m² | **원문** | ITU-3 "the same as … Guideline 1" |
| 어두운 줄 | < 160 cd/m² | < 160 cd/m² | **원문** | 같음 |
| 면적 (정지) | > 40% | > 40% | **원문** | ITU-3·Ofcom. 단 Wilkins 2005는 "불필요하게 제한적, 유도 근거 없음"이라고 비판 |
| 면적 (변화) | > 25% | > 25% | **원문** | ITU-3·Ofcom |
| 공간 주파수 | **1~8 cpd** (2026-11-10 변경, 이전 1~10) | 없음 | **합의 문헌** | 이전 값 1~10은 실험·모델·표준·도구 어디에도 없었음(팀원 코드에서 출처 없이 들어옴). 지금 근거: **EF 2026 합의문 "~1–8 cycles per degree"** (Fisher 외 2026, Epilepsia 67:565, doi:10.1111/epi.18702, Wilkins 1979 인용). 실험 최적 대역은 1~4 cpd(Wilkins 1979), 8 cpd 초과 환자 자료는 없음. 자세히: [PATTERN_LITERATURE.md](PATTERN_LITERATURE.md) |
| 시청 거리 | 56 cm | | **문헌 간접** | Jordan & Vanderheiden 2024 표 2: "15-in. computer display **(4:3)** viewed at 56 cm" (WCAG의 "15–17 inch at 22–26 inches"에서 유래). **우리와 팀원 코드는 15인치 16:9로 계산** → 화면 비율이 원문과 다름 |
| 시청 거리 (데스크톱, 팀원) | 68 cm | | **문헌 간접** | 같은 표: "19-in. (4:3) viewed at 68.0 cm" (Shin & Hegde 2010, Human Factors 52:574) |
| 움직임 판정 | 정지/반전/흐름 3분류 (시간 에너지 10%, \|DI\| 0.5) | 없음 | **구현 선택** (분류 필요성은 실험 근거: Binnie 1985) | 어떤 표준에도 정량 기준이 없음. 시험 결과: [checks/pattern/motion_classes.md](../checks/pattern/motion_classes.md). J&V 2024 원문: "there is no specific, actionable guidance available on motion compensation thresholds" |
| 지속 시간 | 0.5초 | ITU-3에는 없음 | **문헌 간접** | EFA 2005·Wilkins 2005의 "≥ 0.5 s". Wilkins 2005도 "패턴 지속 시간을 공식 조사한 연구는 없다"고 씀 |
| 극값 찾는 기준 | 20 cd/m² | | **구현 선택** | 숫자는 원문(20)이지만 줄 경계 찾기에 쓴 것은 우리 방식. 팀원은 5로 찾고 20으로 거름 |
| 한 방향 흐름 면제 | **구현** (`PATTERN_DRIFT_EXEMPT=True`, 2026-11-10) | 면제 | **원문** | 흐름 판정 자체(\|DI\| ≥ 0.5)는 구현 선택 |

### 3-4. 패턴 종류

| 종류 | 표준 | 의학 근거 |
|---|---|---|
| 줄무늬 (평행, 방사, 곡선, 직선) | **원문**: Ofcom "parallel or radial, curved or straight" | 단일 방향 줄무늬가 가장 위험 (Wilkins 1979) |
| 물방울 같은 반복 요소의 행 | **원문**: Ofcom "rows of repetitive elements such as polka dots" | |
| 동심원·나선 | "curved"·"radial"에 해당한다고 해석 (**추정**) | |
| 체커보드·격자 | 명시 없음 | 단일 방향 줄무늬보다 덜 위험 (Wilkins 1979) |
| 등휘도 적/녹 줄무늬 | | 발작파 반응 0 (Wilkins 1979) → 패턴 위험은 **휘도 대비**가 좌우 |

### 3-5. 흐름 줄무늬를 휘도 판정에서도 제외 (결정 A, 2026-11-10 · 재검토 예정)

`analyze_all`은 줄무늬 판정이 '흐름(drifting)'으로 분류한 프레임의 줄무늬 픽셀에서 생긴 휘도 전이를 휘도 깜빡임으로 세지 않는다. `analyze`(휘도만)에는 이 제외가 없다.

| 근거 | 내용 | 등급 |
|---|---|---|
| 표준 문구 | "If the patterns obviously flow smoothly across, into, or out of the screen in one direction they are [exempt]" — ITC 2001 → Ofcom 2017 Annex 1 → ITU-R BT.1702-3 Attachment 1 | 원문 (단, **줄무늬 규칙의 예외**) |
| 환자 실험 | Binnie, Findlay & Wilkins 1985 (10명): 한 방향 흐름 2.5~10°/s PPR 0~0.04, 정지 0.38, 진동 15 Hz 0.97 | 실험 (원문) |
| 검사 도구 | Harding FPA 매뉴얼: 흐름과 카메라 이동(팬·줌)은 분석에서 제외 | 벤더 문서 (원문) |

**한계 (재검토할 때 볼 것)**
1. 표준은 **줄무늬 규칙**에서만 흐름을 제외한다. 휘도 깜빡임 규칙(ITU 본문, 규범)에는 움직임 예외가 없다 → 휘도 판정에서 빼는 것은 **우리의 확장 해석**이다.
2. 실험이 지지하는 흐름 속도는 2.5~10°/s뿐이다. 그보다 빠른 흐름은 근거가 없는데 지금은 속도와 상관없이 뺀다.
3. 흐름 판정(|DI| ≥ 0.5)은 구현 선택이다. 진동을 흐름으로 잘못 분류하면 위험을 놓친다(시험 영상의 흐름 |DI| = 0.71로 문턱과 여유가 크지 않음).
4. 움직임 보정 기준은 어떤 표준에도 없다 (Jordan & Vanderheiden 2024: "no specific, actionable guidance available on motion compensation thresholds").

**재검토 후보**: ① 흐름이 확실할 때(|DI| ≥ 0.7 등)와 실험 범위 속도(≤ 10°/s)일 때만 제외 ② 데이터 라벨에 `flash_raw`(제외 없음)와 `flash_drift_exempt`(제외)를 둘 다 저장.

시험: `tests/test_pattern_motion.py` — 흐름 줄무늬 55%는 `analyze`로는 위험, `analyze_all`로는 휘도·줄무늬 모두 안전 / 반전 줄무늬는 그대로 위험 / 흐름 영역 밖의 휘도 깜빡임은 그대로 위험.

---

## 4. 판정기 세 개의 차이 (5단계 비교용)

| | 우리 `measure.py` | rockrovks `vdcoreops_v2.py` | IRIS |
|---|---|---|---|
| 휘도 "바뀜" | 같은 픽셀의 시간 변화 ≥ 20 cd/m² (ITU 원문) | 같은 프레임에서 평균 × 1.3 이상 밝은 픽셀 (**출처 미확인**) | 상대 휘도 0.1 변화 (= 20 cd/m²) |
| 깜빡임 간격 | 1초 창에 7번 이상 | 334 ms 미만 간격만 연속으로 봄 (ITU 원문) | 1초에 전이 6번 초과 |
| 적색 | WCAG 2.2 방식, **시간 변화**(바로 앞 상태)와 깜빡임 쌍 규칙 | WCAG 2.2 숫자(0.8, 0.2)지만 Δu′v′를 **같은 프레임 안** 빨강 대 주변으로 계산 | 옛 WCAG 공식 |
| 줄무늬 | 12방향 전체 스캔, 5쌍, 1~10 cpd | 중앙 60개 직선, 5쌍, 1~10 cpd | FFT, 줄 6개 이상, 0.5초, 25% |
| 면적 | 픽셀 단위 | 픽셀 단위 | 화면 평균 → 국소 깜빡임 놓침 |

---

## 5. 다음 조사 목록 (판정 추가 전에)

| 항목 | 왜 필요한가 | 방법 |
|---|---|---|
| ISO 9241-391 원문 | 적색 기준의 1차 출처 (WCAG가 인용) | 학교 도서관 표준 열람 |
| ~~EFA 2005 "eight stripes"의 뜻~~ | **해결: 8쌍** (2026-11-10) | [PATTERN_LITERATURE.md](PATTERN_LITERATURE.md) |
| 1~10 cpd 범위의 출처 | 문헌 전수 조사로 출처 없음 확인 (2026-11-10) | 팀원에게 마지막 확인 → 없으면 EF 2026의 1~8 cpd로 바꾸거나 구현 선택으로 명시 |
| rockrovks "평균 × 1.3"의 출처 | 팀원 판별기와 결과가 다를 때 설명 | 팀원에게 문의 |
| 15인치 4:3 → 16:9 차이의 영향 | ppd가 바뀌어 cpd 판정이 달라짐 | 두 조건으로 계산해 비교 |
| 움직임 정량 기준 | 25% / 40% 중 어느 것을 쓸지 결정 | 표준에 없음. 구현 선택으로 두고 민감도(5·10·20%)를 측정해 보고 |

## 6. 결정 기록 (2026-11-12)

| 항목 | 결정 | 근거 | 등급 |
|---|---|---|---|
| 면적 25% 의 기준 | **화면 전체 면적의 25%** (시야각 25% 아님) | ITU-R BT.1702-3 "more than 25% of the displayed screen area". 시야 기준(WCAG 0.006 sr, Fisher 2026 '시야 25%')도 근거는 있으나 시야 크기 정의가 없어 수치로 옮기기 어려움 (docs/research/viewing_geometry.md) | 원문 + 결정 |
| 지속(연장) 플래시 | **ITU-R 방식**: 진폭·면적은 기준 이상, 빈도는 1초 기준 아래(2~3 Hz)인 깜빡임이 오래 지속되면 위험. 숫자는 EA IRIS 구현(1초 전이 4~6, 최근 5초 중 80%)을 따름 | ITU-R BT.1702-3 Annex 1: "a sequence of flashing images lasting more than five seconds might constitute a risk even when it complies with the guidelines above"(주의 문구, 빈도·창 없음) · IRIS d96978a TransitionTracker | ITU 원문(정성) + IRIS 구현 선택 |
| 지속 플래시 검증 | 우리 `measure.extended_flash`(iris) 와 IRIS 실행 파일이 시험 영상 24개에서 24/24 일치, 실패 시각 동일 | checks/extended/README.md | 실측 |
| 66 ms 단일 전이 허용 시간 | **보류**: 프레임 규약(pse 205/210) 대 연속 시간(201/210) — 둘 다 옵션 | checks/rules/jv24.md | 제안(JV24 2024) |
