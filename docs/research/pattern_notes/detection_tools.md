# 광과민성 '규칙 패턴(줄무늬)' 자동 검출 도구·알고리즘 비교 (detection tools)

> ⚠️ 이 문서는 2026-11-10 시점 기록입니다. 이후 기준 변경(11-12: 6쌍·움직임 무관 40%·흐름 제외 끔·1~10 cpd·규칙성, 11-13: 판정기 버그 6건 수정)은 ../../STANDARDS.md 를 보세요.

작성 2026-11-10. 범위는 검출 도구와 알고리즘만 다룬다. 표준과 의학 문헌은 다른 조사자가 맡았다.
표기: **원문**은 소스 코드나 전문을 직접 읽은 것, **초록**은 초록만 읽은 것, **2차**는 검색 요약이나 2차 출처, **미확인**은 확인하지 못한 것이다.
다운로드한 코드는 실행하지 않고 읽기만 했다.

---

## Q1. EA IRIS: 패턴 검출을 실제로 어떻게 하는가

### Takeaway
IRIS는 **줄무늬 '쌍'을 세지 않는다.** 처리 순서는 이렇다.
1. 휘도 프레임을 FFT로 변환하고, 스펙트럼에서 Otsu로 고른 피크를 지운 뒤 역변환한다.
2. 원본과의 차이가 큰 픽셀을 이진화하고 윤곽선(contour)을 만든다.
3. **모양이 비슷한 윤곽선 개수**(`nComponents`)를 '줄 수'로 쓴다.

판정 조건은 세 가지다. 면적(회전 외접사각형) ≥ 25%, `nComponents` ≥ 6(`MinStripes`), 밝은 성분 평균 상대휘도 ≥ 0.25. 이 상태가 0.5초 지속되면 Fail이다.
cpd, 시청 거리, 정지/움직임 구분, 절대 명암(cd/m²) 검사는 **없다**. 설정 파일에 있는 `RelativeDarkLuminanceThreshold` 0.8은 읽기만 하고 판정에 쓰지 않는다.

### Cited Findings
(모두 원문. github.com/electronicarts/IRIS 커밋 d96978a(2025-01-14) 기준, `src/PatternDetection.cpp`, `config/appsettings.json`, `src/Configuration.cpp`, `src/ConfigurationParams.h`, README를 읽음)

**설정값**
- `appsettings.json` "PatternDetection" 블록: `"MinStripes": 6, //min stripes for harmful patterns`, `"TimeThreshold": 0.5, //max seconds for harmful patterns until failure`, `"RelativeDarkLuminanceThreshold": 0.8`, `"AreaProportion": 0.25`. 주석 외에 수치의 출처(표준·논문) 표기는 없다 — [IRIS appsettings.json](https://github.com/electronicarts/IRIS/blob/main/config/appsettings.json)

**전처리**
- 프레임 폭이 480px를 넘으면 50%로 줄인다(`scale_percent = 50`). 면적 기준도 이 축소 크기로 잡는다. `m_safeArea = area*areaProportion`, `m_diffThreshold = area*0.1`, `m_contourThreshArea = area*0.00155`, `m_thresholdArea = area*0.20`("20% of the screen display"). `m_thresholdArea`는 대입된 것만 보였고 판정에 쓰이는 곳은 PatternDetection.cpp에서 찾지 못했다 — [PatternDetection.cpp](https://github.com/electronicarts/IRIS/blob/main/src/PatternDetection.cpp)
- `detectPattern`: 휘도를 축소한 뒤 `cv::normalize(..., 0, 255, NORM_MINMAX)`로 **프레임마다 min-max 정규화**해 8bit로 만든다. 주석은 "ensures proper contrast if existing pattern"이다. 정규화 때문에 이 단계 이후로는 절대 명암 정보가 없다 — [PatternDetection.cpp](https://github.com/electronicarts/IRIS/blob/main/src/PatternDetection.cpp)

**FFT 단계 (`hasPattern`, `FourierTransform`)**
- DFT 크기 스펙트럼을 [-1,1]로 정규화하고 `abs`를 취한다. 이어 `1 - x`, 제곱, `log(1+x)`를 거쳐 0~255로 정규화한 값을 'PSD'라 부른다(`getPSD`).
- `getPeaks`에서 이 PSD를 Otsu로 이진화한다(인자 7은 Otsu 플래그 때문에 무시된다).
- `filterMagnitude`는 DC 주변 반경 5px 원을 마스크에서 빼고, 마스크된 주파수의 크기를 0으로 만든다.
- 역변환(IFT)한 영상과 원본의 `absdiff > 50`(8bit)인 픽셀이 축소 프레임 면적의 10% 미만이면 "패턴 없음"이다 — [PatternDetection.cpp](https://github.com/electronicarts/IRIS/blob/main/src/PatternDetection.cpp)

**줄 수 세기 (`getPatternRegion`, `getPatternContour`, `getSimilarContours`)**
- 차이 이진영상에서 외곽 윤곽선을 찾고, 면적이 0.155% 이하인 윤곽선은 버린다. 남은 윤곽선을 convex fill한 뒤 윤곽선을 다시 찾는다.
- 윤곽선이 **5개 미만**이면 `nComponents = 0`이다(가장 큰 윤곽선만 영역으로 씀).
- 5개 이상이면 `cv::matchShapes(..., CONTOURS_MATCH_I1)` 값이 0.7 미만인 윤곽선끼리 묶는다. 가장 큰 묶음의 크기가 `nComponents`가 되고, CSV에는 "Pattern Detected Lines"로 출력된다 — [PatternDetection.cpp](https://github.com/electronicarts/IRIS/blob/main/src/PatternDetection.cpp), [README](https://github.com/electronicarts/IRIS/blob/main/README.md)

**면적**
- 가장 큰 묶음의 점들을 합쳐 `cv::minAreaRect`(회전 외접사각형)를 그린다. 그 사각형을 채운 마스크의 픽셀 수가 `pattern.area`다. README도 "Pattern Area: area of the bounding rect. of the pattern"이라고 적는다 — [README](https://github.com/electronicarts/IRIS/blob/main/README.md)

**밝기 조건**
- `setPatternLuminance`가 패턴 영역 안을 Otsu로 밝은/어두운 성분으로 나누고, 각 성분의 **원래 상대휘도** 평균을 구한다.
- `checkFrame`의 harmful 조건은 `pattern.area >= m_safeArea && pattern.nComponents >= minStripes && pattern.avgLightLuminance >= 0.25`다. 0.25는 하드코딩이다.
- `avgDarkLuminance`는 계산만 하고 판정에 쓰지 않는다. `darkLuminanceThreshold`("max luminance of the darker part of a flash/pattern")는 설정에서 읽어 파라미터 구조체에 넣지만 PatternDetection.cpp에서 쓰는 곳이 없다 — [PatternDetection.cpp](https://github.com/electronicarts/IRIS/blob/main/src/PatternDetection.cpp), [Configuration.cpp](https://github.com/electronicarts/IRIS/blob/main/src/Configuration.cpp)

**지속 시간**
- `m_frameTimeThresh = fps * timeThreshold`다. 프레임 관리자가 0.5초 창 안의 harmful 프레임 수(`current`)를 센다. `current >= framesInHalfSecond`이면 그 프레임이 Fail이다. README 표현은 "whenever there is a harmful pattern present for half a second or more, it is marked as a pattern failure" — [PatternDetection.cpp](https://github.com/electronicarts/IRIS/blob/main/src/PatternDetection.cpp), [README](https://github.com/electronicarts/IRIS/blob/main/README.md)

**검증 자료**
- 테스트 이미지는 `test/Iris.Tests/data/TestImages/Patterns`에 있다: 20stripes.png, stripes.png, HorizontalStripes.jpg, ChessBoard.png, Diagonals.png, circle.png 등.
- 패턴 테스트는 4개다: NoPattern_Pass, Straight_Lines_Fail, RealTime_NoPattern_Pass, RealTime_Straight_Lines_Fail. 오검출률이나 비교 데이터는 저장소에 없다 — [IRIS tests](https://github.com/electronicarts/IRIS/tree/main/test/Iris.Tests)
- README와 코드 어디에도 ITU, Ofcom, Harding, Wilkins 같은 출처 인용이 없다. README는 "IRIS' photosensitive epilepsy criteria"라고만 쓴다 — [README](https://github.com/electronicarts/IRIS/blob/main/README.md)

### Inferences
- IRIS의 "stripes"는 **이진 차이영상에서 비슷한 모양의 블롭 개수**다. 밝은 줄과 어두운 줄이 따로 세어지는지는 차이영상의 극성에 따라 달라진다. `MinStripes = 6`은 ITU/Ofcom의 "more than five pairs"(=6쌍 이상)나 Harding의 "6 or more pairs"를 옮기다가 단위가 '쌍'에서 '블롭'으로 바뀐 것으로 보인다. 그렇다면 쌍 기준으로는 ITU보다 덜 엄격할 수 있다(약 3쌍만으로 통과 가능). 추론이고 문서 근거는 없다.
- 면적을 외접 회전사각형으로 잡으므로 원형이나 불규칙한 패턴에서는 면적이 과대 추정된다.
- min-max 정규화와 상대휘도 0.25 조건 때문에 ΔL ≥ 20 cd/m²나 어두운 쪽 < 160 cd/m² 같은 ITU 명암 조건은 구현돼 있지 않다.
- 정지와 움직임의 구분(25%/40%)이 없다. 모든 패턴에 25%를 쓰므로 정지 패턴에는 표준보다 보수적이다.

### Gaps
- `FrameManager`가 0.5초 창을 정확히 어떻게 관리하는지(창 내 전부인지 누적인지)는 `IFrameManager` 구현을 읽지 않아 미확인이다.
- `MinStripes`, `0.25`, `0.00155`, `matchShapes < 0.7` 같은 수치를 왜 골랐는지에 대한 설계 문서나 이슈 논의는 찾지 못했다(미확인).

---

## Q2. Harding FPA (Cambridge Research Systems): 패턴 테스트

### Takeaway
HardingFPA는 **"20 cd/m² 이상 명암의 명암 교대 막대 쌍(pairs of alternating light-dark bars)"을 세고 덮은 면적을 잰다.**
- **정지** 패턴은 6쌍 이상, 0.5초 초과 지속, 면적 > 40%이면 Fail이다. 일부 조건만 넘으면 경고다.
- 방향이 바뀌거나 진동·점멸·명암 반전하는 패턴은 **점멸 자극으로도 분석**하여 25% 면적 기준을 적용한다.
- 체커보드는 무시한다. 카메라 팬/줌 같은 움직임은 Ofcom 규칙상 면제다.
- 알고리즘 내부(cpd 처리, 쌍 계산법)는 공개되지 않았다.

### Cited Findings
- 패턴 검출 설명: "examines material for pattern structures that contain pairs of alternating light-dark bars of at least 20cd/m2 contrast". 체커보드는 무시하고 반복 막대 배열은 검출하며, 막대 쌍 수와 면적을 센다(웹 FAQ Q13, 2차 요약 도구 경유로 읽음) — [How To Interpret HardingFPA Results](https://www.hardingfpa.com/technical-support/how-to-interpret-hardingfpa-results/)
- Fail 조건: "6 or more stationary pairs which persist for more than 0.5 seconds and the area occupied is greater than 40%". 일부 조건만 충족하면 "Pass with Caution" 경고다(FAQ Q13) — [같은 페이지](https://www.hardingfpa.com/technical-support/how-to-interpret-hardingfpa-results/)
- 움직이는 패턴: 방향 변화, 진동, 점멸, 명암 반전 패턴은 "both as a pattern and as a flashing stimulus"로 분석한다. 화면의 25%를 넘으면 flash 위험으로 Fail이 될 수 있다. "A pattern must remain stationary for at least 0.5 seconds in order to fail."(FAQ Q13·Q14·Q17) — [같은 페이지](https://www.hardingfpa.com/technical-support/how-to-interpret-hardingfpa-results/)
- 응답 평활: 분석값은 최근 5프레임 평균이다(FAQ Q15) — [같은 페이지](https://www.hardingfpa.com/technical-support/how-to-interpret-hardingfpa-results/)
- 원문, HardingFPA-X User's Manual 4.0 PDF를 pdftotext로 추출해 읽음:
  - SPAT trace의 범위는 0.0~3.4다. "A value of 0.3 or 0.4 indicates a warning while a value of 0.5 and above indicates failure."
  - Advanced Information에는 "Spatial Pattern Area"(가이드라인을 넘은 면적 %)와 "Spatial Pattern Contrast"(그 영역의 평균 명암)가 나온다(p.38 부근).
  - [HardingFPA-X Users Manual 4.0](https://www.hardingfpa.com/assets/Software-releases/Manuals/HardingFPA-X-Users-Manual-4.0.pdf)
- 원문, 같은 매뉴얼 p.40 부근: "Spatial mask data only appears for stationary, regular patterns which lead to failure. Any spatial patterns which drift, or are not regular, or do not persist in the video long enough to trigger a failure are excluded."
  - 마스크는 타일 단위다. 지속시간 색상은 green 0~1/6 s, yellow 1/6~1/3 s, orange 1/3~1/2 s, red "more than 1/2 second = FAILURE", purple "outside of pattern regularity limit"다.
  - "When running under Ofcom guidelines, the HardingFPA will only tolerate illegal spatial patterns to persist for up to half a second".
  - [Manual 4.0](https://www.hardingfpa.com/assets/Software-releases/Manuals/HardingFPA-X-Users-Manual-4.0.pdf)
- 원문, 매뉴얼 FAQ 6(p.48 부근): "a spatial pattern has been detected with 50 cd/m2 contrast (limit 20 cd/m2) and covers 46% screen area (limit 40%). However … motion, caused by the camera pan and zoom, makes the detected spatial pattern exempt from failure under Ofcom rules." — [Manual 4.0](https://www.hardingfpa.com/assets/Software-releases/Manuals/HardingFPA-X-Users-Manual-4.0.pdf)
- 원문, 매뉴얼 FAQ 8(p.49~50 부근):
  - "patterns that are clearly bar-like in one direction may also possess some local structure in the orthogonal direction … may cause the system to see fewer than 6 light-dark bars or may separate a provocative pattern into two or more regions."
  - 텍스트나 전경 물체가 패턴을 작은 불규칙 영역으로 나누면 Pass가 된다.
  - [Manual 4.0](https://www.hardingfpa.com/assets/Software-releases/Manuals/HardingFPA-X-Users-Manual-4.0.pdf)
- 원문, 매뉴얼 FAQ 9: 원근이 있는 파도, 풍경, 망사 커튼, 종이 더미처럼 "faint, repeating structure"가 있는 장면은 spatial 경고를 낼 수 있다. 다만 "highly unlikely" 실제 Fail로 이어지지는 않는다 — [Manual 4.0](https://www.hardingfpa.com/assets/Software-releases/Manuals/HardingFPA-X-Users-Manual-4.0.pdf)
- 원문(grep): 매뉴얼 4.0 전문에 "cycles", "degree", "viewing distance"가 패턴과 관련해 등장하지 않는다. 공간주파수 기준은 공개 문서에 없다 — [Manual 4.0](https://www.hardingfpa.com/assets/Software-releases/Manuals/HardingFPA-X-Users-Manual-4.0.pdf)
- 2차: Version 3 알고리즘은 HD 표준용이고, SD 레거시 모드는 2.5 알고리즘을 쓴다 — [hardingfpa.com 검색 요약](https://www.hardingfpa.com/)

### Inferences
- 같은 회사 문서 안에서도 단위가 엇갈린다. 웹 FAQ는 "6 or more … pairs", 매뉴얼은 "fewer than 6 light-dark bars"라고 쓴다. 쌍(pair)과 막대(bar)가 혼용되므로 Harding의 실제 임계를 '6쌍'으로 단정하면 안 된다.
- '정지'와 '드리프트'는 **타일 단위로 0.5초 동안 같은 패턴이 유지되는지**로 가르는 것으로 보인다(마스크 설명 근거). 정량적 이동 임계(px/s 등)는 공개되지 않았다.
- Harding은 정지 패턴에 40%, 움직이는(진동·점멸) 패턴에 '점멸 25%'를 적용한다. 이 구조는 ITU/Ofcom의 정지 40%/움직임 25%와 대응한다.

### Gaps
- 쌍을 어떻게 세는지(방향 스캔, 필터뱅크 등), 공간주파수 처리, 'regularity limit'의 정의, 타일 크기는 비공개다(미확인).
- Harding FPA의 패턴 검출 성능을 독립적으로 검증한 논문은 찾지 못했다(미확인).

---

## Q3. PEAT (Trace Center)

### Takeaway
PEAT 1.6 사용자 가이드는 **일반 플래시, 적색 플래시, 연장 플래시 경고만 결과 유형으로 나열**한다. 패턴 분석 기준이나 결과는 문서에 없다. Harding·CRS와 공동 개발한 엔진이지만 공개 문서상으로는 패턴 검출 기능을 확인할 수 없다.

### Cited Findings
- 원문: PEAT User Guide v1.6은 "general flash failures, red flash failures, and extended flash warnings"를 검출한다고 적는다. 패턴은 "bright and rapid flashes, especially red flash, and some spatial patterns"라는 일반 주의 문구로만 나온다 — [PEAT User Guide](https://trace.umd.edu/?p=719)
- 2차: Trace Center(UW–Madison)가 Cambridge Research Systems, Graham Harding과 6년간 협력해 엔진을 만들었다. PEAT 분석은 WCAG 2.0 flash 기준과 맞춘 것이다 — [news.wisc.edu](https://news.wisc.edu/software-tool-helps-web-developers-identify-seizure-causing-content/)
- 2차: 2022년 Trace 게시물은 오픈소스 신판 PEAT를 개발 중이라고 했다(출시 일정 없음) — [검색 요약, trace.umd.edu](https://www.trace.umd.edu/?p=1669)

### Inferences
- WCAG 2.x의 flash 기준에는 패턴 항목이 없다. PEAT가 WCAG에 맞춰졌다는 점을 보면 PEAT는 사실상 플래시 전용으로 보인다(추론).

### Gaps
- PEAT 엔진 내부에 패턴 코드가 있는지(비활성 포함)는 미확인이다.

---

## Q4. 기타 도구와 학술 알고리즘

### Takeaway
공개 문헌에서 줄무늬 검출 방식은 세 계열로 나뉜다.
1. **FFT 피크 제거 + 윤곽선 묶기**: IRIS, 그리고 USPTO 출원 "System and methods for automated photosensitivity detection"(2차).
2. **선분 검출(LSD) + 연결성분 + Wilkins 2005 위험곡선**: Rodrigues 2015, IST Lisboa.
3. **비공개 상용 엔진**: Harding, Vidcheck, Baton, Cel-Soft.

학술 쪽에는 cpd를 명시적으로 계산하는 사례가 하나 있다(Rodrigues 2015). 다만 고정 대역 대신 Wilkins 2005의 연속 위험곡선을 쓴다.

### Cited Findings
**Rodrigues 2015 (IST Lisboa 확장초록, "Automatic Detection of Epileptogenic Video Content", Nelson Rodrigues)** — 원문(pdftotext로 읽음), [PDF](https://fenix.tecnico.ulisboa.pt/downloadFile/844820067124507/Resumo%20Algargado%20Automatic%20Detection%20of%20Epileptogenic%20Video%20Content_NR.pdf)
- 단계: (i) OpenCV Line Segment Detector(LSD)로 선분을 찾는다. (ii) 최소 길이 미만 선분을 버린다. (iii) 방향각 `arctan((y2-y1)/(x2-x1))`으로 각도 빈에 묶고, `binSIZE`보다 적은 빈을 버린다.
- `binSIZE`의 근거: "Concerning the information inferred from the literature [2], a five-stripe pair pattern represents a substantial increase of risk. Therefore, BinSIZE should be not less than 9 line segments." [2]는 Wilkins, Emmett, Harding, Epilepsia 46:1212–1218, 2005다.
- 원형 패턴은 윤곽 다각형 근사와 볼록 껍질(convexHull)로 영역을 분리한다.
- Otsu 임계로 밝은(positive)/어두운(negative) 연결성분을 나누고 십자형 3×3 커널로 침식한다. 면적 `th_area` 미만 성분과 평균 휘도 차가 `Y_maxCCdiff`를 넘는 성분은 버린다.
- `minComp`의 근거: "According to the ITU-R recommendations [11], a potentially harmful regular pattern contains clearly discernible stripes when there are more than 5 light-dark pairs of stripes, 10 stripes, in any orientation. Therefore, it is advisable a value of minComp ≥ 5 for both positive, lightstripes, and negative, darkstripes, components."
- 패턴 면적은 밝은 성분 면적과 어두운 성분 면적의 합이다.
- 위험 추정 입력은 시청 거리 `distanceVIS`, 화면비 A, 시야각 θ = 2·arctan(A·H/2 / distanceVIS)다. 공간주파수는 spatialFreq(cpd) = N/θ로 계산한다(N은 stripe 수 기반).
- 위험은 Wilkins 2005 회귀식을 곱해 낸다: cpd별 환자 비율(식 14, log 3차식), 면적(식 15), Michelson 명암(식 18), 평균 휘도(식 16). RiskFINAL = riskAREA × riskLumaContrast.
- 정지/움직임 구분과 지속 시간 조건은 이 초록에서 찾지 못했다.
- 검증은 합성 2개(줄무늬 5개 + 원형 2개, 잡음 배경 버전)와 실제 영상 1개(22초)다. 저자도 "notorious constraints in the detection", 실시간 미지원이라고 적었다(2차 검색 요약).

**USPTO 출원 12705750 "System and methods for automated photosensitivity detection"** — 2차(검색 스니펫만; PDF는 스캔 이미지라 본문을 읽지 못함), [USPTO PDF](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12705750)
- 스니펫 인용: "two key features of epileptogenic patterns are regularity and periodicity, performing a Fourier analysis on the grayscale video frame allows detecting patterns without the need of executing line or contour detection operations"; "if a pattern is present in the video frame, its spatial frequencies will be present in the magnitude spectrum."
- 출원인과 발명자는 미확인이다. 문구와 방식이 IRIS의 FFT 단계와 비슷하지만 EA 출원인지는 확인하지 못했다.

**Parallel scheme for real-time detection of photosensitive seizures** — 초록/2차, [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0010482516000123)
- Computers in Biology and Medicine, 2016. 스트리밍 영상용 시공간 패턴 검출을 멀티코어로 실시간 처리한다.
- 테스트 영상 생성 도구 "Pattern Inducer"를 포함한다. Jordan 2025에 따르면 원저자에게 요청하면 받을 수 있다.
- 세부 임계값은 미확인이다.

**South, Saffo & Borkin, CHI 2021 "Detecting and Defending Against Seizure-Inducing GIFs in Social Media"** — 초록, [Khoury Vis Lab](https://vis.khoury.northeastern.edu/pubs/South2021DetectingDefendingSeizure), DOI 10.1145/3411764.3445510
- 인터뷰와 기존 시스템 검토로 설계 요구를 만들고, 위험 콘텐츠를 탐지·무력화하는 브라우저 확장 프로토타입을 만들었다. 시뮬레이션 GIF와 SNS GIF 데이터셋으로 검증했다(초록).
- 패턴(줄무늬) 검출 포함 여부와 방법은 전문을 읽지 못해 미확인이다.
- "PhotosensitivityPal"이라는 이름과 이 논문의 연결은 확인하지 못했다.

**도구 목록 (Jordan 2025가 언급한 것)** — 원문(PMC 전문, 요약 도구 경유), [PMC12249941](https://pmc.ncbi.nlm.nih.gov/articles/PMC12249941/)
- HardingFPA, PEAT, Vidcheck, Baton, Cel-Soft, Carreira et al. 도구, Alzubaidi et al. 시스템, PhotosensitivityPal, Flikcer, Tripathi et al. 확장, FFmpeg PSE 필터, EA IRIS, Apple Video Flashing Reduction.
- 이 논문은 이들 도구의 패턴 검출 성능을 보고하지 않는다.

### Inferences
- 공개된 '쌍 수' 근거는 결국 두 가지로 모인다. ITU/Ofcom의 "more than five pairs"(Rodrigues는 각 극성 ≥5, 즉 ITU를 ≥5로 해석)와 Wilkins 2005의 "five-stripe pair"다. 두 근거 모두 5~6 경계에 있다.
- 오픈소스 중 정지/움직임 면적 이원화(25/40%)를 구현한 공개 코드는 찾지 못했다. 상용으로는 Harding만 문서화돼 있다.

### Gaps
- Barbu, Banda & Katz 2020: 찾지 못했다(미확인). 서지 정보를 다시 확인할 필요가 있다.
- Telestream/Tektronix, Vidcheck, Baton, Cel-Soft(ClearView 계열)의 패턴 알고리즘과 임계값은 이번 조사에서 읽지 못했다(미확인).
- FFmpeg `photosensitivity` 필터, Flikcer, Tripathi et al.의 패턴 처리 여부는 미확인이다.
- 딥러닝 기반 줄무늬 검출기는 찾지 못했다(미확인).

---

## Q5. Jordan 2025 / pse-test-media: 패턴 테스트 세트와 패턴 관련 결과

### Takeaway
Jordan(2025)은 **패턴(줄무늬) 검출 성능을 평가하지 않았다.** pse-test-media의 `area_patterns`는 **점멸 면적 임계 시험용 도형**(단일 도형, 평행 막대, 확산 패턴 등)이다. 규칙 줄무늬(쌍 수, cpd)를 시험하는 세트는 저장소 디렉터리에 없다.

### Cited Findings
- 원문(PMC 전문, 요약 도구 경유): J. Bern Jordan, "Evaluating Conformance of Video Safety Tools for Photosensitive Epilepsy", UAHCI 2025, LNCS 15780, pp.85–98, DOI 10.1007/978-3-031-93848-1_7.
  - 이 논문에는 stripe pairs, 40% 면적, cpd, 명암 기반 패턴 정의가 등장하지 않는다.
  - 면적은 25%(ISO 9241-391, ITU-R BT.1702, Ofcom, NAB-J)와 WCAG의 "25% of any 10° visual field"(컴퓨터 시청 근사로 416×416 CSS px)를 쓴다.
  - 인용: "Tests must also include more complex spatial patterns that just exceed (or fall short of) the threshold area."
  - [PMC12249941](https://pmc.ncbi.nlm.nih.gov/articles/PMC12249941/)
- 원문(GitHub API로 디렉터리 트리 확인): traceRERC/pse-test-media의 최상위 폴더는 `area_patterns/{25pct_341x256, 25pct_416x416, 25pct_screen, 80pct_screen, combine_25pct_416x416}`, `lum_flash_patterns`, `red_flash_patterns`, `combo_flash_patterns`, `video_creation`, `utils`다 — [pse-test-media](https://github.com/traceRERC/pse-test-media)
- 원문: `area_patterns/25pct_341x256/README.md`
  - 시험 대상은 "Fail where 25% or more of the pixels are involved from any 341 x 256 px subarea"("WCAG Classic")다.
  - 파일명이 f로 시작하면 면적 초과(Fail), a로 시작하면 통과다.
  - 12종 도형: 단일 도형(중앙/모서리), 속이 빈 도형, "Parallel bars within a 416x416 frame", 확산 패턴, 대각 패턴, S자 등.
  - 인용: "For any failing spatial pattern incorporated into a video, all of the other thresholds also need to fail for a video sequence to be considered potentially hazardous."
  - [README](https://github.com/traceRERC/pse-test-media/tree/main/area_patterns/25pct_341x256)

### Inferences
- pse-test-media의 "spatial pattern"은 '점멸 영역의 공간 배치'라는 뜻이다. 규칙 패턴 검출기를 검증하는 데이터로는 쓸 수 없다. 우리 줄무늬 검출기의 시험 세트는 직접 만들어야 한다. 쌍 수 4/5/6/7, cpd 대역 경계, 면적 24/26%와 39/41%, 정지/드리프트/진동을 경계값 시험(boundary test)으로 넣는 방식은 Jordan의 '임계 바로 위/아래' 설계 원칙을 그대로 가져올 수 있다.

### Gaps
- Jordan의 후속 연구에 패턴 평가가 있는지는 미확인이다.

---

## Q6. 공간주파수(cpd) 대역(예: 1–10 cpd)과 '움직임/진동'의 정량 정의를 쓰는 도구가 있는가

### Takeaway
조사한 도구 중 **고정 cpd 대역(1–10 cpd 등)을 쓰는 것은 없다.**
- IRIS에는 cpd 개념이 없다(원문).
- Harding 매뉴얼에는 cpd 언급이 없다(원문).
- Rodrigues 2015는 시청 조건으로 cpd를 계산하지만 Wilkins 2005 연속 위험곡선에 넣는다(원문).

**1–10 cpd는 팀원 판정기와 우리 measure.py에만 있고, 코드 안에 출처 표기가 없다.**
'움직임'을 도구 수준에서 정량 정의한 공개 사례도 찾지 못했다. Harding은 "stationary for at least 0.5 s"와 "drift → 제외"를 쓰지만 이동량 임계는 비공개다.

### Cited Findings
- IRIS: PatternDetection.cpp와 설정에 시청 거리, 화면 크기, cpd 파라미터가 없다(원문) — [PatternDetection.cpp](https://github.com/electronicarts/IRIS/blob/main/src/PatternDetection.cpp)
- Harding: 매뉴얼 4.0 전문을 "cycles", "degree", "viewing distance"로 grep하면 패턴 관련 결과가 없다(원문). 움직임은 "drift"와 카메라 "pan and zoom" 면제로만 기술된다 — [Manual 4.0](https://www.hardingfpa.com/assets/Software-releases/Manuals/HardingFPA-X-Users-Manual-4.0.pdf)
- Rodrigues 2015: spatialFreq(cpd) = N/θ, θ = 2·arctan(A·H/2/distanceVIS). Wilkins 2005 곡선으로 환자 비율을 추정한다(원문) — [PDF](https://fenix.tecnico.ulisboa.pt/downloadFile/844820067124507/Resumo%20Algargado%20Automatic%20Detection%20of%20Epileptogenic%20Video%20Content_NR.pdf)
- 임상 참고(2차): "two to four cycles per degree is the most epileptogenic", "optimum spatial frequency lies between 1 and 4 cycles/degree". 의학 문헌 담당 조사자가 원문을 확인해야 한다 — [ILAE photic stimulation guideline](https://www.ilae.org/files/ilaeGuideline/PhoticStimulation-2012-.1528-1167.2011.03319.pdf), [Wilkins et al. 2005 (ResearchGate)](https://www.researchgate.net/publication/7686482_Characterizing_the_Patterned_Images_That_Precipitate_Seizures_and_Optimizing_Guidelines_To_Prevent_Them)

### Inferences
- 1–10 cpd라는 대역은 도구 문헌에서 온 것이 아니다. 근거를 대려면 의학 문헌(Wilkins 계열 곡선)에서 직접 끌어와야 하고, 상한 10 cpd의 근거는 특히 따로 확인해야 한다.

### Gaps
- 1–10 cpd의 원출처: 팀원 코드 주석에 없다(미확인).

---

## Q7. 팀원 판정기(vdcoreops_v2.py)와 우리 measure.py `analyze_pattern` 비교, 임계 출처

### Takeaway
두 판정기 모두 '방향 스캔선 1D 프로파일 → 극값 → 명암·cpd 조건을 만족하는 연속 줄 수 → 쌍 수 ≥ 5' 구조다. 이는 IRIS의 FFT/윤곽선 방식이나 Harding의 문서화된 방식과 다르며, Rodrigues 2015의 선분 방향 묶기에 가깝다.
임계 출처는 다음과 같다.
- **ITU에서 온 것**: 명암 ΔL ≥ 20 / 어두운 쪽 < 160 cd/m², 면적 25%(+우리의 40%).
- **코드 주석에 출처가 없는 것**: 5쌍, 1–10 cpd, 56 cm/68 cm, 15인치, 60 스캔선, 0.5초, 확률식.

### Cited Findings
(모두 원문, 로컬 파일을 직접 읽음)

**팀원 판정기** (`team_repos\lumi-lab-rockrovks\Fullscreen\A1\vdcoreops_v2.py`)
- `compute_viewing_angle_and_cpd_spec`
  - device 'laptop'은 `viewing_dist_cm = 56.0`, 'desktop'은 `68.0`이다. `display_area` 기본값은 15인치, 16:9이고 `cpd_min=1.0`, `cpd_max=10.0`이다.
  - 시야각 식에만 출처 주석이 있다: "2*theta = 2 * arctan(x / (2 * d)) (3694790.pdf Appendix A)".
  - 56/68 cm, 15인치, 1–10 cpd에는 출처 주석이 **없다**.
- `count_opposing_stripe_pairs_1d_cpd`
  - 극값 추적 히스테리시스는 **5.0**(cd/m²)이다.
  - 이웃 극값 사이의 명암 조건은 docstring에서 "ITU-R BT.1702-3 명암 대비 기준"이라 한다: `(l_low < 160 and Δ ≥ 20)` 또는 `(l_low ≥ 160 and Michelson > 1/17)`.
  - 반주기로 `local_cpd = px_per_deg / (2·half_cycle_px)`를 계산해 1~10이면 유효로 본다.
  - 연속 유효 전이 수의 최댓값을 `// 2`해서 쌍 수로 쓴다.
- `precompute_stripe_scan_lines`
  - 0°~165°를 15° 간격으로 12방향 스캔한다. 방향마다 법선 방향 offset을 `linspace(-0.5r, 0.5r, 5)`로 5개 두어 **총 60개 직선**을 만든다. r = 0.45·min(h,w)이다.
  - 방향별 px/deg는 `1/sqrt((cosθ/ppd_w)²+(sinθ/ppd_h)²)`다.
- 프레임 판정(`evaluate_single_frame_all_metrics`)
  - 면적은 줄무늬 픽셀이 아니라 **국소 고대비 마스크**(짧은 변 3% 커널)로 잰다. 이를 `avg_pool`(커널 ≥15, 짧은 변의 6%)한 밀도가 ≥ 0.15인 영역을 `pattern_area`로 둔다.
  - `pattern_area > 0.25`일 때만 60선 스캔을 하고, `max_pairs >= 5`이면 `stripe_danger`다.
  - 정지/움직임 구분 없이 25% 하나만 쓴다.
  - 확률 `eta = (area/0.25)·min(1,pairs/5)·contrast; prob = eta²/(1+eta²)`에는 출처 주석이 없다.
- 시간 판정: `stripe_danger` 이력을 플래시와 같은 구간 검출 함수에 `min_duration_sec=0.5, min_pairs=4`로 넘긴다(588~599행 부근).
- 관련 모듈 `fullscreen_coreops_1108.py` 머리말은 근거로 "Fisher, Harding, Wilkins et al., 2005; ITU-R BT.1702-3; Jordan & Vanderheiden, 2024"를 든다. 이 근거는 휘도 보정(20 cd/m²)과 관련된 것이고 cpd 대역의 근거는 아니다.

**우리 measure.py** (`C:\dev\lumi-vision\lumi-lab-ryujungho2-collab\src\lumi_synth\measure.py`, 219~249행 주석과 상수)
- 출처 구분 주석:
  - "ITU 원문: 면적 25%(움직임)/40%(멈춤), 명암은 깜빡임과 같은 기준(ΔL ≥ 20, 어두운 쪽 < 160), 모든 방향"
  - "팀원 판정기(...) 기준을 따름 — 사용자 결정 2026-11-09: 5쌍 이상(ITU 원문은 'more than five' = 6쌍 이상, 우리가 1쌍 보수적), 1~10 cpd, 시청 조건(노트북 56 cm, 15인치 16:9), 12방향 15° 간격, 지속 0.5초"
  - "LUMI 결정: 극값 히스테리시스 20 cd/m²(팀원 5), 쌍 수 = (극값 수)//2, …, 화면 전체를 1픽셀 간격으로 스캔(팀원은 중앙 60개 직선), 움직임 판정 10%"
  - "미구현: '한 방향으로 매끄럽게 흘러가는 패턴은 제외'(ITU)"
- 상수: `PATTERN_MIN_PAIRS = 5`, `PATTERN_AREA_DYNAMIC = 0.25`, `PATTERN_AREA_STATIC = 0.40`, `PATTERN_CPD = (1.0, 10.0)`, `PATTERN_DYNAMIC_FRAC = 0.10`, `PATTERN_MIN_SEC = 0.5`, `PATTERN_ANGLES = range(0,180,15)`, `VIEW_DISTANCE_CM = 56.0`, `DISPLAY_INCH = 15.0`.
- `analyze_pattern` 동작:
  - 움직임: 줄무늬 픽셀 중 ±0.5초 안에 픽셀 전이(ΔL ≥ 20)가 있었던 비율이 ≥ 10%이면 dynamic이다.
  - 면적 한계는 dynamic이면 0.25, 아니면 0.40이다(>).
  - 지속: 아무 1초 창 안에서 후보 프레임이 ≥ ceil(0.5·W)이면 위험이다.

### 비교 요약표

| 항목 | EA IRIS (원문) | HardingFPA (원문·웹 FAQ) | Rodrigues 2015 (원문) | 팀원 vdcoreops_v2 (원문) | 우리 measure.py (원문) |
|---|---|---|---|---|---|
| 검출 원리 | FFT 피크 제거 → absdiff>50 → 윤곽선 → matchShapes 묶기 | 비공개. "pairs of alternating light-dark bars" | LSD 선분 + 각도 빈 + Otsu 연결성분 | 60 스캔선 1D 극값 | 12방향 × 1px 간격 전체 스캔 1D 극값 |
| 세는 단위 | 비슷한 윤곽선 수 ≥ 6 (`MinStripes`) | 6쌍 이상(웹) / 6 bars(매뉴얼) | 선분 ≥ 9 / 각 극성 성분 ≥ 5 | 쌍 ≥ 5 | 쌍 ≥ 5 |
| 명암 | 프레임 min-max 정규화, 밝은 성분 상대휘도 ≥ 0.25 | ≥ 20 cd/m² | Michelson → Wilkins 곡선 | ΔL ≥ 20 & 어두운 쪽 < 160, 또는 Michelson > 1/17 | ΔL ≥ 20 & 어두운 쪽 < 160 |
| 공간주파수 | 없음 | 문서에 없음 | cpd = N/θ → Wilkins 곡선 | 1–10 cpd (출처 없음) | 1–10 cpd (팀원 따름) |
| 시청 조건 | 없음 | 문서에 없음 | 사용자 입력 | 56/68 cm, 15인치 | 56 cm, 15인치 |
| 면적 | 회전 외접사각형 ≥ 25% | 정지 > 40%, 움직임은 점멸 25% | 성분 면적 합 → 곡선 | 고대비 밀도 영역 > 25% | 줄무늬 픽셀 > 25% / 40% |
| 정지/움직임 | 구분 없음 | 정지만 패턴 Fail, drift 제외, pan·zoom 면제 | 없음 | 구분 없음 | 전이 비율 ≥ 10% |
| 지속 | 0.5초 창 harmful 프레임 수 ≥ 창 크기 | > 0.5초(5프레임 평균) | 없음 | 0.5초 (run 검출 함수) | 1초 창 중 ≥ 0.5초 |

### Inferences
- '5쌍'은 ITU("more than five")보다 1쌍 보수적이다. 이것은 우리 코드 주석이 명시한 결정이다. Rodrigues 2015도 ITU를 "minComp ≥ 5"로 해석했고 Wilkins 2005의 "five-stripe pair"를 근거로 들었다. 따라서 '5쌍 이상' 선택은 Rodrigues 2015와 Wilkins 2005를 인용해 정당화할 여지가 있다. 반면 Harding과 IRIS는 6이다.
- 정지/움직임을 정량화(전이 비율 10%)한 것은 조사한 도구 중 우리 measure.py뿐이다. 이 10%는 LUMI 결정이며 외부 근거가 없다. Harding처럼 0.5초 동안의 정지성(타일 지속)으로 정의하는 대안을 검토할 수 있다.
- 면적을 외접사각형(IRIS)이나 고대비 밀도(팀원)로 재면 줄무늬 픽셀로 재는 것(우리)보다 커진다. 같은 영상에서 판정이 갈릴 수 있으니 논문에 측정 정의를 명시해야 한다.

### Gaps
- 팀원 코드의 "3694790.pdf"가 어떤 문헌인지는 미확인이다. 이름으로 보아 ACM DOI 10.1145/3694790일 가능성이 있다(추론, 미확인). 팀원 다른 파일의 "Jordan & Vanderheiden, 2024"와 같은 문헌인지 확인이 필요하다.
- 1–10 cpd, 56/68 cm, 15인치, Michelson 1/17, 히스테리시스 5의 근거는 팀원 코드에 없다(미확인). 팀원에게 직접 물어야 한다.
