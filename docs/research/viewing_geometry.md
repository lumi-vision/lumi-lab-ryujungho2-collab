# 시청 조건·시야각 가정의 근거 (viewing geometry)

작성 2026-11-10 (노트북). `measure.pixels_per_degree`의 56 cm·15인치 가정, 768×432 통일, 세로 쇼츠 처리의 근거다.
표기: **원문**은 본문이나 PDF를 직접 읽은 것, **초록**은 초록만 읽은 것, **2차**는 리뷰나 검색 요약, **미확인**은 찾지 못하거나 열지 못한 것이다.
조사는 웹 조사 에이전트가 했다. 15인치 16:9 @56 cm의 시야각(33.0°)은 노트북에서 직접 다시 계산했다.

---

## 결론 먼저

1. **BT.1702-3와 Ofcom(ITC) 규칙에는 시청거리도, cpd도 없다.**
   - 섬광 판정은 "표시된 화면 면적의 25%"다.
   - 패턴 판정은 "명암 쌍이 5쌍 초과"이고, 면적은 정지 패턴 40%, 움직이는 패턴 25%다.
   - 따라서 화면 크기와 무관하다.
2. **각도로 정의된 기준은 WCAG(10° 시야 안 0.006 sr)와 Fisher 2026(1–8 cpd, 시야 25%)뿐이다.**
   - 1도당 픽셀(ppd) 가정이 필요한 곳은 이 두 가지다.
   - 우리 줄무늬 판정의 cpd 대역은 이쪽 계열이다.
3. **WCAG의 "10° 창"은 옛 TV 화면 하나 크기를 데스크톱 위로 옮겨 놓은 것이다**(추론, S2·S9·S11이 뒷받침).
   - 1024 px를 15–17인치(4:3)에 띄우면 75–85 ppi다. 22–26인치 거리에서 341 px는 8.8–11.8°가 된다.
   - 10° 원의 입체각은 0.0239 sr이므로 0.006 sr은 그 25.1%다.
4. **Fisher 2026은 "화면의 25%"가 아니라 "양안 시야(visual field)의 25%"라고 쓰는데, 시야 크기를 정의하지 않는다.** 그래서 이 숫자를 그대로 수치 기준으로 옮길 수 없다.
5. **세로 폰 화면(6.1인치, 30–36 cm)은 시야각으로 보면 옛 TV와 비슷하고, ppd는 데스크톱 가정의 1.5–1.85배다.**
   - 폰 세로 영상에 데스크톱 ppd를 쓰면 실제 cpd를 과소추정한다.
   - 모바일, 세로 영상, VR 전용 규정은 BT.1702, Ofcom, WCAG 어디에도 없다.

---

## 1. 출처

| # | 내용 | 확인한 주소 | 등급 |
|---|---|---|---|
| S1 | WCAG 2.2 Understanding 2.3.1. "The 341 x 256 pixel block represents a 10 degree viewport at a typical viewing distance." 기준 화면은 1024×768, 예시는 "15-17 inch screen at 22-26 inches", "75 - 85 ppi". CSS px(0.0213°)로 환산해 87,296 CSS px²로 정의한다. 모바일은 "the mobile screen itself takes up a smaller portion of the user's visual field."라고만 쓴다. | https://www.w3.org/WAI/WCAG22/Understanding/three-flashes-or-below-threshold.html | 원문 |
| S2 | WCAG 2.0 Understanding. 영국 TV 방송 사양을 "adapted for computer screens … closer distance"했다고 쓴다. | https://www.w3.org/TR/UNDERSTANDING-WCAG20/seizure-does-not-violate.html | 원문 |
| S3 | w3c/wcag #553, #585. 고밀도 화면 문제와, 10°를 CSS px로 환산하면 341이 아니라 469라는 지적. 노트만 수정하고 종료됐다. | https://github.com/w3c/wcag/issues/553 , /585 | 원문(논의) |
| S4 | w3c/wcag #4943 (2026-02, 진행 중). 0.006 sr = 43,400 CSS px²(지름 235 CSS px 원), 10° = 470 CSS px로 바꾸자는 제안. 관련 이슈로 #4358, w3c/wcag3 #631이 있다. | https://github.com/w3c/wcag/issues/4943 · https://github.com/w3c/wcag3/issues/631 | 원문(미확정 논의) |
| S5 | WCAG 3 초안 "No flashing over threshold" (2026-08-28). 0.006 sr과 341×256을 그대로 두고, "size and frequency metrics … likely to change"라고 쓴다. 모바일과 VR은 언급하지 않는다. | https://www.w3.org/WAI/WCAG3/informative/animation-and-movement/avoid-physical-harm/no-flashing-over-threshold/ | 원문(초안) |
| S6 | ITU-R BT.1702-3 PDF. "more than 25% of the displayed screen area", 패턴 ">5 light-dark pairs" + 40%/25%. 휘도 측정은 BT.500 가정 시청 환경 기준이다. Annex 5의 "2 m 이상"은 시청자에게 하는 권고다. 시청거리, 도(°), sr은 본문에 없다. | https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.1702-3-202311-I!!PDF-E.pdf | 원문 |
| S7 | Ofcom이 호스팅하는 ITC Guidance Note (2001, 2002 수정, legacy). "one quarter of the displayed screen area". 시청거리 정의가 없다. | http://web.archive.org/web/20231001095538/https://www.ofcom.org.uk/__data/assets/pdf_file/0021/16248/gn_flash.pdf | 원문 |
| S8 | ITU-R BT.2022 (2012, 2020 폐지). 1080 라인 16:9는 약 31°, 3.2H. SD 행은 PDF 표가 깨져 정렬이 불확실하다. | https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2022-0-201208-W!!PDF-E.pdf | 원문(일부 불확실) |
| S9 | Harding et al. 2005 EFA consensus. "≥0.006 steradians (approximately 10% of the central visual field or 25% of screen area at typical viewing distances)" | Europe PMC, PMID 16146438 | 초록 |
| S10 | **Fisher et al. 2026, *Epilepsia* 67(2):565–581, doi:10.1111/epi.18702.** Table 4: "1-8 cycles that subtend 1 degree at the eye (the width of one finger held at arm's length…)", "more than about 25% of the visual field of both eyes for more than about 0.5 s". 그러나 기존 지침을 요약하는 대목에서는 "0.006 steradians or about 25% of the visual field"라고 써서 두 개념을 섞는다. 그 밖에 "smartphones seem to cause fewer issues than larger screens", VR은 "Evidence does not support specific increased risks"라고 쓴다. | https://dspace.library.uu.nl/server/api/core/bitstreams/511a8fa4-deff-4845-b9af-c3d0d64e969c/content | 원문 |
| S11 | Jordan & Vanderheiden 2024 (TACCESS) Table 2. 기기별 시야각과 0.006 sr이 화면에서 차지하는 비율. 6.1인치 폰 @32 cm: 24.8°×11.6° (10.5%), 15인치 4:3 @56 cm: 30.5°×23.1° (2.7%), VR: 0.3%. "smart phones are not much riskier than television viewing". VR에는 절대 면적(0.006 sr) 기준을 권고한다. | https://pmc.ncbi.nlm.nih.gov/articles/PMC11872230/ | 2차(리뷰) |
| S12 | Bababekova et al. 2011, *Optom Vis Sci* 88(7). 문자 읽기 36.2 cm (n=129), 웹 보기 32.2 cm (n=100) | Europe PMC, PMID 21499163 | 초록 |
| S13 | Boccardo 2021, *J Optom* 14(2). n=233, 문자 읽기 36.8±6.6 cm | https://pmc.ncbi.nlm.nih.gov/articles/PMC8093538 | 원문 |
| S15 | Apple iPhone 16 사양. 6.1인치, 2556×1179, 460 ppi → 화면 폭 65.1 mm | https://www.apple.com/iphone-16/specs/ | 원문 |

## 2. 계산 (중심부 ppd, Nyquist = ppd/2)

- 분석 폭은 가로 영상 768, 세로 영상 432다.
- 폰은 9:16 영상이 화면 폭(65.1 mm)을 꽉 채운다고 가정했다.
- 우리 코드(`measure.pixels_per_degree`)는 화면 폭 ÷ 시야각으로 평균을 내므로 (a)가 **23.3**이다. 아래 표는 화면 중심 값이라 22.6이다. 차이는 정의 차이이고, 결론에는 영향이 없다.

| 경우 | 화면 가로×세로 (cm) | 시청거리/화면높이 | 수평×수직 시야각 | ppd | 최고 cpd | 0.006 sr이 화면에서 차지하는 비율 |
|---|---|---|---|---|---|---|
| (a) 15인치 16:9 @56 cm, 768 (현재 가정) | 33.2×18.7 | 3.0 | 33.0°×18.9° | 22.6 | 11.3 | 3.2% |
| (b) 15인치 4:3 @56 cm, 768 | 30.5×22.9 | 2.45 | 30.4°×23.1° | 24.6 | 12.3 | 2.9% |
| (c) 6.1인치 폰 9:16 세로 @30 cm, 432 | 6.5×11.6 | 2.6 | 12.4°×21.8° | 34.7 | 17.4 | 7.3% |
| (c') 같은 폰 @36 cm | 〃 | 3.1 | 10.3°×18.3° | 41.7 | 20.8 | 10.5% |
| (d) 55인치 TV 16:9 @2.5 m, 768 | 121.8×68.5 | 3.65 | 27.4°×15.6° | 27.5 | 13.8 | 4.7% |

**읽는 법**
- 모든 경우 최고 cpd가 8보다 높아서 1–8 cpd 대역을 표현할 수 있다.
- (a)에서 8 cpd는 한 주기가 2.8 px라서, 축소할 때 앨리어싱에 주의해야 한다(INTER_AREA 사용 중).
- 폰 세로 영상에서는 화면의 25%가 0.006 sr의 2.4–3.4배다. 데스크톱(a)에서는 약 8배다. 즉 폰에서는 화면 비율 기준과 각도 기준이 크게 어긋나지 않는다.

## 3. 우리 코드에 대한 함의와 권고 (결정은 사용자)

1. **데스크톱 가정(15인치 16:9 @56 cm)은 유지해도 무난하다.**
   - 56 cm는 WCAG 거리 하한 22인치에서 왔고, 시청거리가 화면 높이의 약 3배라 HD 설계 시청거리(3.2H, S8)와 비슷하다.
   - 다만 코드 주석과 문서에는 이 값이 "BT.1702 원문 값이 아니라 WCAG·Fisher 계열의 각도 기준을 쓰기 위한 가정"이라고 적어야 한다.
2. **BT.1702 판정(휘도 25%, 패턴 쌍 수와 40%/25%)은 화면 비율만 쓴다.** 현재 구현이 이미 그렇게 되어 있다. ppd는 cpd 대역 판정에만 쓰인다.
3. **세로 쇼츠는 현재 letterbox(16:9 화면 안 필러박스)로 처리한다.** 이는 "데스크톱 화면에서 세로 영상을 보는 경우"에 해당한다. 폰 전체화면 시청을 따로 판정하려면 다음 값을 쓸 수 있다.
   - 6.1인치 @30 cm, 432 폭 기준 ppd 약 35를 기본으로 한다.
   - 민감도 범위는 19 cm(ppd 약 22)부터 36 cm(ppd 약 42)까지로 한다.
   - **이 값들은 어느 PSE 규격에도 근거가 없는 자체 설정값이다.** 미구현.
4. **VR은 범위 밖이라고 명시한다**(S11은 절대 면적 기준을 권고).

## 4. 확인하지 못한 것

- 현행 Ofcom Broadcasting Code 본문(확인한 것은 ITC 2001 legacy 문서뿐)
- Harding FPA가 내부적으로 쓰는 화면과 거리 가정
- EFA 2005 전문(초록만 확인)
- BT.2022 SD 행 정렬, 현행 BT.500 본문
- 스마트폰 화면 크기의 시장 분포(6.1인치는 Apple 사양 하나만 근거)
- **세로 숏폼 영상을 볼 때의 시청거리를 측정한 연구**(찾지 못함. 기존 연구는 문자·웹 읽기 과제)
