# 픽셀 값 → 화면 밝기(EOTF) 변환의 근거: sRGB냐 BT.1886이냐 (gamma / EOTF)

작성 2026-11-10 (노트북). 측정 결과 [checks/rules/gamma_sensitivity.md](../../checks/rules/gamma_sensitivity.md)를 해석하고, 남은 BT.1886 재판정 140개를 돌릴지 정하기 위한 문서다.
표기: **원문**은 본문·PDF·소스 코드를 직접 읽은 것, **2차**는 위키 등 2차 출처, **미확인**은 확인하지 못한 것이다.
조사는 웹 조사 에이전트가 했다. 아래 '직접 재확인' 표시 항목은 노트북에서 다시 계산하거나 원문 추출본에서 다시 확인했다.

---

## 결론 먼저

1. **BT.1702-3의 SDR 기준은 BT.1886(감마 2.4)에 최대 밝기 Lw = 200 cd/m², 최저 밝기 Lb = 0, 10비트 narrow range(64–940)를 넣은 것이다.**
   - NOTE 값(400 → 20.1, 863 → 160.4)이 이 식으로 정확히 재현된다(직접 재확인: 20.06, 160.37).
   - (Lw, Lb)를 격자 탐색으로 맞추면 Lw = 200.0, Lb = 0.0이 나온다. full range(D/1023)로 해석하면 두 점이 맞지 않는다.
2. **지금 쓰는 sRGB × 200도 근거가 있다.**
   - BT.1702-0(2005)과 BT.1702-1(2018)의 SDR 가정이 '감마 2.2, 공칭 최대 200 cd/m²'였다.
   - BT.1702-1 원문 표 제목이 "Display gamma 2.2 – Nominal peak 200 cd/m2"이고, 372 → 20.06, 855 → 159.78이다(직접 재확인).
   - 코드 400에서 sRGB × 200 = 24.33, 감마 2.2 × 200 = 24.29로 사실상 같다.
   - BT.1886으로 바뀐 것은 -2(2019)부터다.
3. **비교 기준선들은 모두 sRGB 상대 휘도 체계다.** WCAG 2.2, IRIS 코드, pse-test-media가 그렇다. 이들은 절대값 20 cd/m² 대신 상대값 0.1, 어두운 쪽 0.8을 쓴다.
   - pse-test-media의 경계 쌍은 **sRGB 상대 휘도 0.1에서 1코드 차이**로 설계돼 있다. 그래서 EOTF를 바꾸면 구조적으로 뒤집힌다.
   - 우리가 측정한 7/70 뒤집힘은 BT.1886 구현 버그라기보다 이 설계와 맞지 않아서 생긴 것으로 **추정**한다.
4. **EOTF뿐 아니라 range 해석(full/limited)도 결과를 바꾼다.**
   - 예: pse-test-media f001(코드 64↔109)
     - sRGB × 200: ΔL 20.33 → 위험
     - BT.1886 full range: 18.76 → 안전
     - BT.1886 video range(16–235): 20.37 → 위험
5. **입력 종류에 따라 변환을 고르는 방식이 가장 방어하기 쉽다.** Trace24 가이드라인(Jordan & Vanderheiden 2024)이 정확히 이렇게 나눈다.
   - sRGB 콘텐츠(웹·모바일 RGB)는 IEC 61966-2-1(sRGB)
   - SDR 방송 신호(BT.709/2020 Y'CbCr)는 BT.1886, Lw = 200, Lb = 0
   - PQ/HLG는 BT.2100

---

## 1. BT.1702 판본별 SDR 변환 (원문 PDF 텍스트 추출본으로 확인)

| 판 | SDR 변환 | 최대 밝기 | ΔL 규칙 | NOTE의 기준점 |
|---|---|---|---|---|
| -0 (2005/02) | "gamma of 2.2 may be assumed", 아날로그 mV 기준 | 200 cd/m² | 절대값 20, 어두운 쪽 < 160 | 234 mV → 20.1, 631 mV → 160 |
| -1 (2018/07) | 표 제목 "Display gamma 2.2 – Nominal peak 200 cd/m2", V = (D−64)/876 | 200 | 같음 | **372 → 20.1, 855 → 160** (직접 재확인) |
| -2 (2019/10) | SDR이 **BT.1886(감마 2.4)**으로 바뀜. PQ/HLG 열 추가 | SDR 200 / HLG 1000 / PQ 10000 | 어두운 쪽 < 160이면 20 cd/m². ≥ 160이면 Michelson 1/17 초과("HDR programmes only") | 400 → 20.1, 863 → 160.4 |
| **-3 (2023/11, 현행)** | -2와 같음 (Annex 2 Table 1) | 같음 | 같음 | 같음 |

- -3 NOTE 2: "Video waveform luminance is not a direct measure of display screen brightness… may be assumed as described in Annex 2"
- 즉 원문도 이 변환을 측정 편의를 위한 **가정**으로 둔다. 실제 측정은 스폿 광도계로 한다.
- HDR 전용 부속서는 없다. Annex 2 표의 PQ/HLG 열과 Michelson 규칙이 전부다. PQ(377 → 20.07)와 HLG(Lw 1000, 시스템 감마 1.2, 362 → 20.12) 값도 식으로 재현된다.

### 수치 검증

V = (D−64)/876

| 10비트 코드 | 원문 값 | BT.1886 (200/0) | BT.1886 (200/0.1) | sRGB × 200 | 감마 2.2 × 200 |
|---|---|---|---|---|---|
| 400 | 20.1 | **20.06** | 23.47 | 24.33 | 24.29 |
| 863 | 160.4 | **160.37** | 161.94 | 162.31 | 163.35 |

- 8비트 full range에서 sRGB × 200과 BT.1886(200/0)의 최대 차이는 **+5.03 cd/m²(코드 146)**다. 우리 측정값(gamma_sensitivity.md의 최대 5.03)과 일치한다.
- sRGB의 구간 경계값 0.03928(1996 초안)과 0.04045(1999 정식판)의 차이는 8비트에서 걸리는 코드가 없다. WCAG 2.2 Note 2도 "no practical effect"라고 쓴다.

## 2. 도구·지침별 변환

| 출처 | EOTF | 최대 밝기 가정 | ΔL 정의 | 등급 | 주소 |
|---|---|---|---|---|---|
| ITU-R BT.1702-3 | BT.1886 감마 2.4, Lb = 0 / PQ / HLG | 200 / 10000 / 1000 | 절대값 20 cd/m² (어두운 쪽 < 160). HDR 밝은 쪽은 Michelson 1/17 | 원문 | https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.1702-3-202311-I!!PDF-E.pdf |
| BT.1702-0 / -1 / -2 | 감마 2.2 / 감마 2.2 / BT.1886 | 200 | 절대값 20 | 원문 | 같은 경로의 -0-200502-I, -1-201807-S, -2-201910-S |
| **IRIS (EA)** | sRGB 구간식, 경계 0.04045, Y = 0.2126R + 0.7152G + 0.0722B | 없음(상대값) | **상대값 0.1**, 어두운 쪽 < 0.8 | 원문(코드) | https://github.com/electronicarts/IRIS (`src/RelativeLuminance.h`, `src/Flash.cpp`, `config/appsettings.json`) |
| WCAG 2.2 | sRGB (IEC 61966-2-1) | 명시 없음 | 상대 휘도 10%, 어두운 쪽 < 0.80 | 원문 | https://www.w3.org/TR/WCAG22/ |
| WCAG 2.2 Understanding 2.3.1 | sRGB가 아닌 영상에는 BT.1702의 20/160 cd/m²와 HDR Michelson 1/17 적용 | — | 절대값 | 원문 | https://www.w3.org/WAI/WCAG22/Understanding/three-flashes-or-below-threshold.html |
| **pse-test-media** | sRGB 8비트 전용("sRGB is the only color space supported today") | 없음 | 방송 세트(iso, itu_r1702, ofcom 태그)도 **"0.1 relative luminance"**로 설계. 통과/실패 쌍이 1코드 차이 | 원문(README, CSV) | https://github.com/traceRERC/pse-test-media |
| Jordan 2025 (pse-test-media 논문) | — | — | "in practice, the luminance thresholds match between ISO, ITU-R, Ofcom, and WCAG" | 원문 | https://pmc.ncbi.nlm.nih.gov/articles/PMC12249941/ |
| **Trace24 가이드라인** | sRGB 콘텐츠는 IEC 61966-2-1, SDR TV는 BT.1886(200/0), PQ/HLG는 BT.2100 | 200 (극장 48) | 20 cd/m² = 0.1 × 기준 휘도 | 원문 | https://github.com/traceRERC/pseGuidelines (`pseGuidelines.md`) |
| PEAT | 공개 안 됨(Cambridge Research Systems 엔진, WCAG 2.0 기준) | 공개 안 됨 | WCAG | 소개 페이지 원문, EOTF 미확인 | https://trace.umd.edu/peat/ |
| Harding FPA | 공개 안 됨. 사이트에 BT.1702-3 / HDR10 / HLG 모드가 있다 | 미확인 | 미확인 | 사이트 원문, EOTF 미확인 | https://www.hardingtest.com/ |
| ITC/Ofcom legacy (2001/2002) | 감마 2.2, mV 표 | 200 | 절대값 20 | 원문(web.archive 사본) | https://www.ofcom.org.uk/__data/assets/pdf_file/0021/16248/gn_flash.pdf |
| 영국 DPP012 납품 요건 | BT.1702-2 또는 -3을 준수하는 도구 필수 | — | — | 원문 | https://cdn-2.4sales.com/staging/2025-11/dpp012-production-and-post-production-commercial-guidance-v2.0[43].pdf |
| NHK·민방련 | EOTF 명시 없음 | SDR 200, HLG 1000, PQ 10000 | SDR은 "輝度変化 10パーセント以上"(상대값). 200 cd/m²를 넘는 HDR은 20 cd/m² 또는 어두운 쪽의 1/8 | 원문 | https://j-ba.or.jp/stance/jba103852.html |
| sRGB (IEC 61966-2-1) | 구간식, 기준 디스플레이 80 cd/m² | 80 | — | 2차(Wikipedia) | https://en.wikipedia.org/wiki/SRGB |
| ITU-R BT.2408-8 (2024) | flash, photosensitive, epilep, 1702, flicker 모두 0건. HDR Reference White 203 cd/m²로, SDR 200 가정과 거의 같다 | — | — | 원문 | ITU-R BT.2408 |

**공통점:** 최대 밝기 가정은 일관되게 200 cd/m²다(BT.1702 전 판본, 민방련, Trace24). IRIS와 WCAG의 상대값 0.1/0.8도 200 기준으로 보면 20/160 cd/m²에 대응한다.

## 3. 우리 판정기에 대한 함의와 권고 (결정은 팀 합의 후)

1. **주 판정은 sRGB × 200을 유지한다**(현재 상태 그대로).
   - 비교 기준선(pse-test-media, IRIS, WCAG)과 같은 체계다.
   - BT.1702-1까지의 공식 가정(감마 2.2, 200)과 사실상 같으므로, '근거 없는 선택'이 아니라 '-1 판까지의 가정 + WCAG 체계'라고 쓸 수 있다.
2. **"BT.1702-3을 구현했다"고 쓰려면 문언상 BT.1886(200/0, narrow range)이 정답이다.**
   - 논문에는 "SDR 변환은 sRGB(WCAG·pse-test-media 체계)를 쓰고, BT.1702-3의 BT.1886 변환은 민감도 분석으로 보고한다"고 명시한다.
3. **입력별 변환 선택은 미구현이다.** sRGB는 sRGB, BT.709 신호는 BT.1886 + narrow range, PQ/HLG는 Annex 2 식으로 고르는 구조다.
   - 우리 데이터는 지금 전부 8비트 RGB로 디코드한 영상이라 sRGB 경로 하나로 충분하다.
   - 방송 신호를 직접 다룰 때 다시 검토한다.
4. **남은 BT.1886 재판정 140개(집 PC)는 sRGB와 BT.1886이 엇갈리는 '판정 불확실 구간'을 찾는 민감도 측정으로 돌리면 의미가 있다.** 단, 결론(주 판정 유지)을 바꾸기 위한 실험은 아니다.
5. **range 해석은 sRGB와 별개로 결과를 바꾼다.** OpenCV가 디코드한 RGB가 full range로 펼쳐진 값인지 기록해 두어야 한다. → 확인 필요.

## 4. 확인하지 못한 것

- Harding FPA와 PEAT의 실제 휘도 변환식(독점 엔진)
- ISO 9241-391 원문(유료)
- 현행 Ofcom Section 2 Guidance Notes(403 차단, legacy ITC 문서만 확인)
- EBU의 PSE 전용 문서(존재 여부 미확인)
- BT.2100, IEC 61966-2-1 원문, 소비자 디스플레이 휘도 실측 문헌
- pse-test-media가 방송 세트를 sRGB 0.1로 정의한 근거에 대한 저자의 명시적 설명
- 위 §3-5: OpenCV 디코드 경로의 range 처리(우리 코드에서 확인 필요)
