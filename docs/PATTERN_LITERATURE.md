# 줄무늬 위험 기준은 실험보다 합의가 정했다

> ⚠️ 이 문서는 2026-11-10 시점 기록입니다. 이후 기준 변경(11-12: 6쌍·움직임 무관 40%·흐름 제외 끔·1~10 cpd·규칙성, 11-13: 판정기 버그 6건 수정)은 STANDARDS.md 를 보세요. 표의 '≥ 5쌍' 행은 지금 ≥ 6쌍입니다.

광과민성 발작(PSE)을 일으키는 규칙 패턴의 정량 기준 가운데 **환자 실험으로 직접 측정된 것은 공간 주파수 최적 대역(1–4 c/deg), 대비 문턱(Michelson 약 0.2–0.4에서 급상승), 크기 법칙(반지름 2배에 확률 0→1), 움직임 종류의 순위(진동 15 Hz 0.97 > 정지 0.38 ≫ 한 방향 흐름 0–0.04) 정도뿐이다.** 이 측정의 근거는 Wilkins·Binnie·Darby 그룹이 1975–1985년에 항경련제를 복용 중인 환자 1–19명을 대상으로 한 실험이다 ([Wilkins 1979](https://visualstress.info/1979-15.pdf); [Binnie 1985](https://visualstress.info/1985-40.pdf)). 우리 판정기가 쓰는 네 숫자의 출처는 이렇다. **'1~10 cpd'는 어떤 실험·모델·표준·도구 문헌에서도 출처를 찾지 못했다.** 팀원 코드에서 출처 표기 없이 들어온 값이다. **'5쌍'**은 ITC 2001이 ad hoc으로 정한 "more than five pairs"(=6쌍 이상)보다 한 쌍 엄격하다. **'정지 40%'**는 ITC가 유도 근거 없이 정한 값이고, Wilkins 2005가 "unnecessarily restrictive"라고 비판했다. **'0.5초'**는 Wilkins와 Harding의 임상 경험값이며, 저자 스스로 "no studies have formally investigated"라고 밝혔다 ([Wilkins, Emmett & Harding 2005](https://visualstress.info/2005-169.pdf)). 정지, 진동, 매끄러운 흐름을 가르는 **정량 기준은 ITC, Ofcom, ITU-R BT.1702-3, NHK, WCAG, trace24, EFA 2005·2026 어디에도 없다.** 표준 계보는 ITC 2001 문구 하나가 Ofcom(2005·2017), ITU-R BT.1702-3 informative 부속서(2023), Harding FPA로 거의 그대로 복사된 구조다. 2026년 EF 합의문은 쌍 수 규칙을 버리고 "**~1–8 cycles per degree**, 시야 25%, 0.5 s"로 옮겨 갔다 ([Fisher et al. 2026](https://dspace.library.uu.nl/server/api/core/bitstreams/511a8fa4-deff-4845-b9af-c3d0d64e969c/content)). 따라서 LUMI 논문은 판정기의 각 파라미터를 '표준 원문', '실험 근거', '모델 유도값', '경험값', 'LUMI 구현 선택' 중 하나로 정직하게 분류해야 한다. 1~10 cpd는 Fisher 2026의 1–8 c/deg로 바꾸거나 구현 선택으로 명시하고, 움직임 판정에는 Binnie 1985의 진동 대 흐름 대비를 재현하는 민감도 분석을 붙이는 것이 가장 방어하기 쉬운 설계다.

> **검증 등급 표기.** **원문**은 조사자가 전문, 스캔 또는 저자 원고를 직접 읽은 것이다. **초록**은 초록만 확인한 것이다. **2차**는 다른 문헌의 인용이나 검색 요약으로만 확인한 것이다. 이전 조사 문서 `PSE_EVIDENCE.md`(archive/2026-11-08-before-reset)에만 원문 판독값이 있고 이번 세션에서 다시 확인하지 못한 항목은 **2차(아카이브)**로 적었다. **미확인**은 출처를 찾지 못한 것이다. **근거 유형**은 네 가지다. 실험은 환자 PPR 측정, 모델은 실험 곡선을 조합해 유도한 값, ad hoc은 유도 근거 없이 정한 값이나 경험값, 없음은 근거 문헌이 없는 값이다.

## 실험이 실제로 측정한 것: 1–4 c/deg, 대비 0.3, 반지름 두 배

고전 실험의 종속변수는 발작이 아니다. **"최대 10초 제시 중 PPR(photoparoxysmal response)이 나올 확률"**이다. 피험자도 소수였고 전원 항경련제를 복용 중이었다 ([Wilkins, Darby & Binnie 1979, Brain 102:1–25, doi:10.1093/brain/102.1.1](https://visualstress.info/1979-15.pdf), 원문). 그래서 모든 수치는 '집단 평균 곡선'이라기보다 '개인 간 분산이 큰 소표본 관찰'로 읽어야 한다.

**공간 주파수.** Wilkins 1979 Study 1은 환자 8명에게 지름 24°의 사각파 줄무늬를 0.5/1/2/4/8 c/deg로 보여 주었다(300 cd/m², 대비 0.75). 결론은 "**the optimum spatial frequency lying between 1 and 4 cycles/degree in all cases**"였다 ([Wilkins 1979](https://visualstress.info/1979-15.pdf), 원문). 같은 그룹의 1980년 리뷰는 이를 "each stripe … must subtend between **7.5 and 30 minutes of arc**"로 다시 썼다 ([Wilkins, Binnie & Darby 1980, Prog Neurobiol 15:85–117](https://visualstress.info/1980-22.pdf), 원문 p.91). 집단 평균은 0.5 c/deg ≈ 0.05, 1 ≈ 0.5, 2 ≈ 0.78, 4 ≈ 0.8, 8 ≈ 0.25이다. 이 판독값은 『Visual Stress』 그림 2.10에서 읽은 것으로 2차(아카이브)이며, Brain 1979 그림은 OCR로 판독하지 못했다. 진동 줄무늬 10명 실험에서도 최적은 2–4 c/deg였고 진동 주파수와 상호작용이 없었다 ([Wilkins 1980 p.97–98](https://visualstress.info/1980-22.pdf), 원문). Soso 1980의 환자 1명은 **5 c/deg에서 정점, 50% 대역폭 1옥타브**였다. 같은 환자의 대비 검출 최적은 약 2 c/deg여서, **유발성 곡선은 대비 감도 함수와 다르다** ([Soso 1980 I, doi:10.1111/j.1528-1157.1980.tb04075.x](https://doi.org/10.1111/j.1528-1157.1980.tb04075.x), 초록). 측정된 최고 주파수는 Wilkins 8 c/deg, Fylan & Harding 1997의 7 c/deg([doi:10.1111/j.1528-1157.1997.tb01202.x](https://doi.org/10.1111/j.1528-1157.1997.tb01202.x), 초록)이고, Soso의 15 c/deg는 억제 상호작용 검사용 단일 조건이었다. **8 c/deg를 넘는 구간의 환자 확률 곡선은 존재하지 않는다.** Wilkins 2005 모델도 고주파 쪽이 CRT 래스터와 간섭한다는 이유로 "Only the lower frequency limit will be considered"라고 하여 이 구간을 의도적으로 뺐다 ([Wilkins 2005](https://visualstress.info/2005-169.pdf), 원문).

**크기·면적.** 한 개인 안에서는 "an increase in the angular subtense of the pattern **by a factor of two**"만으로 확률이 거의 0에서 거의 1로 오른다. 반면 문턱 크기는 사람마다 크게 다르다 ([Wilkins 1979 Study 9](https://visualstress.info/1979-15.pdf), 원문). 원판과 고리가 같은 확률을 낸 조건을 비교하면, 확률을 더 잘 설명하는 것은 단순 시각 면적이 아니라 Drasdo 식 **Q = 100(1 − e^(−0.0574θ))**로 환산한 자극 피질 비율이다(r_s = 0.64 대 0.32, 11명) ([Wilkins 1980 p.104–105](https://visualstress.info/1980-22.pdf), 원문). Wilkins 2005는 19명 자료로 **y = 0.021Q − 0.184**를 제시하며 "no patient is affected when less than 8% of the cortex is stimulated"라고 썼다 ([Wilkins 2005](https://visualstress.info/2005-169.pdf), 원문). Smedley 2010은 이것을 반각 **1.6° 미만이면 위험 0**으로 다시 썼다 ([Smedley, Webb & Wilkins 2010 저자 원고](https://visualstress.info/2009-186.pdf), 원문). **최소 '주기 수'를 독립변수로 조작한 실험은 없다.** Wilkins 1979에는 "반지름 2°에서 2 c/deg는 4주기, 4 c/deg는 8주기"라는 저자 주석만 있다. 쌍 수는 공간 주파수와 크기에서 파생되는 양이다.

**대비와 휘도.** 평균 휘도를 고정하고 대비만 바꾼 개인 자료에서는 대비 0.08/0.28/0.41/0.52에 대해 0/5, 2/16, 9/13, 13/14가, 다른 환자에서는 0.06/0.23/0.43/0.58에 대해 0/10, 0/11, 10/11, 11/11이 나왔다. 저자는 "a very steep ascent for values near 0.3"이라고 요약했다 ([Wilkins 1979 Study 8](https://visualstress.info/1979-15.pdf), 원문). 8명 집단에서도 0–0.3에서 급상승하고 그 뒤 고원에 이르렀다 ([Wilkins 1980 그림 21](https://visualstress.info/1980-22.pdf), 원문). 평균 휘도는 "changes in excess of one log unit"가 있어야 확률이 뚜렷이 바뀐다 ([Wilkins 1980 p.91](https://visualstress.info/1980-22.pdf), 원문). Wilkins 2005는 9명 자료로 10–200 cd/m² 구간의 **y = 0.336 ln(L) − 0.745**를 제시했다 ([Wilkins 2005](https://visualstress.info/2005-169.pdf), 원문).

**방향, 직교 성분, 파형, 색.** 집단 수준에서 방향 효과는 없었다(Kendall W = 0.036). 방향 선택성을 보인 2명은 난시였다 ([Wilkins 1979 Study 2](https://visualstress.info/1979-15.pdf), 원문). 반대로 **직교 성분을 더하면 위험이 줄었다.** 두 격자를 겹친 체크형은 8/41(19%)로 단일 격자 42–54%보다 낮았고, 사선 체크(9′)는 0/9였다 ([Wilkins 1979 Study 5](https://visualstress.info/1979-15.pdf), 원문). 줄무늬가 체크무늬보다 더 유발적이라는 Darby 1977 결과는 Wilkins 1979를 통한 2차 확인이다. Soso는 체크무늬가 가장자리가 사선일 때만 반응을 낸다는 점을 들어, 유발 특징은 **Fourier 기본파 성분**이라고 해석했다 ([Soso 1980 II](https://doi.org/10.1111/j.1528-1157.1980.tb04076.x), 초록). Funatsuka 2001에서는 가장자리 선명도가 영향이 없었다 ([doi:10.1046/j.1528-1157.2001.26000.x](https://doi.org/10.1046/j.1528-1157.2001.26000.x), 초록). 등휘도 적/녹 줄무늬는 7명 모두 PPR 0이었다 ([Wilkins 1979 Study 3](https://visualstress.info/1979-15.pdf), 원문). 그러나 Harding & Fylan 1999는 "non-colour-opponent isoluminant stimuli"로도 PPR이 나온다고 보고했다 ([doi:10.1111/j.1528-1157.1999.tb02018.x](https://doi.org/10.1111/j.1528-1157.1999.tb02018.x), 초록). Fisher 2026은 색 줄무늬의 위험이 "CIE UCS chromaticity" 차이와 함께 오른다고 썼다 ([Fisher 2026](https://dspace.library.uu.nl/server/api/core/bitstreams/511a8fa4-deff-4845-b9af-c3d0d64e969c/content), 원문). 색 항을 0으로 고정할 근거는 약하다.

**2008년 이후의 새 측정은 없다.** cpd × 대비 × 크기 → PPR 확률을 환자에서 새로 측정한 연구는 찾지 못했다. 최근 문헌의 "2–4 cpd"는 고전 문헌을 다시 인용한 것이다(Hermes 2017은 이 문장에 원 자료 인용을 붙이지 않았다) ([Hermes, Kasteleijn-Nolst Trenité & Winawer 2017, PMC5438467](https://pmc.ncbi.nlm.nih.gov/articles/PMC5438467), 원문). Gamma 진동과 pattern glare 연구는 3 cpd 부근에 같은 피크를 보고하지만, 둘 다 발작 예측력으로 검증된 지표가 아니다. PSE 환자의 MEG gamma 군간 차이도 유의하지 않았다 ([Perry et al. 2014, doi:10.1016/j.eplepsyres.2014.04.012](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1016/j.eplepsyres.2014.04.012&resultType=core&format=json), 초록).

### 파라미터별 근거 요약표

| 파라미터 | 값 | 출처 | 근거 유형 | 검증 등급 |
|---|---|---|---|---|
| 공간 주파수: 최적 대역 | 1–4 c/deg (모든 환자) | Wilkins 1979 Study 1 (8명) | 실험 | 원문 |
| 공간 주파수: 집단 곡선 | 0.5≈0.05, 1≈0.5, 2≈0.78, 4≈0.8, 8≈0.25 | Wilkins 1995 그림 2.10 | 실험 | 2차(아카이브) |
| 공간 주파수: 개인 정점 | 5 c/deg, 대역폭 1옥타브 | Soso 1980 I (1명) | 실험 | 초록 |
| 공간 주파수: 진동 패턴 | 2–4 c/deg 최적, 시간 주파수와 독립 | Wilkins 1980 그림 10 (10명) | 실험 | 원문 |
| 공간 주파수: 연속 함수 | y=0.382(log x)³−2.020(log x)²+1.285 log x+0.839 (저주파만) | Wilkins 2005 패널 1 | 모델 | 원문 (로그 밑 미표기, log10 추정) |
| 공간 주파수: 합의 범위 | ~1–8 c/deg ("About 50% of patients") | Fisher 2026 (Wilkins 1979 인용) | 합의(ad hoc) | 원문 |
| 공간 주파수: 1–10 cpd | 1–10 | 팀원 vdcoreops_v2.py → measure.py | **없음** | 미확인 |
| 공간 주파수: 리뷰 표현 | 2–4, 2–3, "4–40′ → 약 3" | Fisher 2005 | 2차 요약 | 원문 (ITC 귀속은 오귀속 의심) |
| 쌍 수 | >5쌍 (=6쌍 이상) | ITC 2001 → Ofcom → BT.1702-3 Att.1 | ad hoc | 원문 |
| 쌍 수 | 변화 5 / 정지 8 / 흐름 12쌍 | Wilkins 2005 | 모델 + 인용 없는 "약 2배" 계수 | 원문 |
| 쌍 수 | 변화 5 / 정지·흐름 8쌍 | EFA 2005 합의 | 합의(흐름 8은 "TV 경험") | 원문 |
| 쌍 수 | ≥6 쌍(웹) / 6 bars(매뉴얼) | Harding FPA | 벤더 운영값 | 원문(벤더) |
| 쌍 수 | ≥5쌍 | 팀원·measure.py, Rodrigues 2015(minComp ≥5) | 구현 선택 | 원문 |
| 명암(대비) | 어두운 막대 <160 cd/m² & 차이 ≥20 cd/m² | ITC 2001 (깜빡임 규칙 차용) | ad hoc | 원문 |
| 명암(HDR) | 어두운 쪽 ≥160이면 Michelson >1/17 | BT.1702-2(깜빡임) → -3 Att.1(패턴) | ad hoc(연속성 맞춤) | 원문 |
| 명암: 실험 문턱 | Michelson 약 0.3 부근 급상승 | Wilkins 1979 Study 8, 1980 그림 21 | 실험 | 원문 |
| 명암: 최소 줄 정의 | >3 cd/m² 차이 | Wilkins 2005 | 모델 | 원문 |
| 평균 휘도 | 밝은 줄 >50 cd/m² | Wilkins 2005 패널 8 → EFA 2005 | 모델 | 원문 |
| 평균 휘도 | 1 log unit 이상 변해야 효과, 10–200 cd/m² log 선형 | Wilkins 1980, 2005 | 실험 / 모델 | 원문 |
| 면적: 정지 | >40% 화면 | ITC 2001 → Ofcom → BT.1702-3 Att.1 | ad hoc | 원문 |
| 면적: 변화 | >25% 화면 | ITC 2001 (깜빡임 1/4 공유) | ad hoc → Wilkins 2005가 모델로 사후 정당화 | 원문 |
| 면적: 일반화 | 0.006 sr (25% ≈ 중심 10°의 1/4) | Wilkins 2005, EFA 2005 | 모델(기하 환산) | 원문 |
| 면적: 피질 | Q<8%(반각 <1.6°)면 위험 0 | Wilkins 2005, Smedley 2010 | 모델(19명) | 원문 |
| 면적: 합의 | 시야 25%, 전체 패턴 <1°면 위험 적음 | Fisher 2026 | 합의 | 원문 |
| 지속 시간 | >0.5 s | Wilkins 2005, EFA 2005 | ad hoc(경험값, 잠복기 1–2 s) | 원문 |
| 지속 시간 | 정지 ≥0.5 s 지속해야 Fail | Harding FPA | 벤더 운영값 | 원문(벤더) |
| 움직임: 종류 순위 | 진동 15 Hz 0.97, 정지 0.38, 흐름 0–0.04 | Binnie 1985 표 II (10명) | 실험 | 원문 |
| 움직임: 유병 | 정지 약 30%, 진동 60–70% | Wilkins 1980 p.90 | 실험 요약 | 원문 |
| 움직임: 진동 주파수 | 10–20 Hz 최적 | Wilkins 1980, Binnie 1985 | 실험 | 원문 |
| 움직임: 진동 주파수 | 3–60 Hz (최대 10–20 Hz) | Fisher 2026 | 합의(깜빡임 범위 차용 추정) | 원문 |
| 움직임: 정량 판정 기준 | 없음 | 모든 표준·합의 | 없음 | 원문(부재 확인) |
| Duty cycle | 밝은 부분 30–50%에서 최대(≈0.62), 20%·80%에서 ≈0.22–0.25 | Wilkins 1995 그림 2.9 (4명) | 실험 | 2차(아카이브) |
| 방향 | 집단 효과 없음; 표준은 "any orientation" | Wilkins 1979 Study 2; ITC/ITU | 실험 / ad hoc | 원문 |
| 직교 성분 | 추가 시 위험 감소(체크 < 줄무늬) | Wilkins 1979 Study 5, Wilkins 2005 | 실험 | 원문 |
| 패턴 종류 | 평행·방사, 곡선·직선, 물방울 행 | ITC 2001 → BT.1702-3 Guideline 2 | ad hoc(정성) | 원문 |
| 패턴 종류 | 縞模様·渦巻き模様·同心円模様, "画面の大部分" | NHK·민방련 1998/2006 | ad hoc(무정량) | 원문 |
| 패턴 종류: 나선·방사형 정량 | 없음 | — | 없음 | 미확인 |
| 색 | 등휘도 적/녹 0/7 | Wilkins 1979 Study 3 | 실험 | 원문 |
| 색 | 색도 차가 클수록 위험 증가 | Fisher 2026 | 합의 | 원문 |
| 시청 조건 | 56 cm, 15인치 | 팀원 코드(출처 미표기); WCAG Note 1 "15-17 inch at 22-26 inches"와 일치 | 구현 선택 | 원문 |

## 네 숫자의 족보: 1~10 cpd·5쌍·40%·0.5초

**'1~10 cpd'는 고아 숫자다.** 표준 조사에서는 cycles/degree를 정의한 공개 표준이 하나도 없었다. ITC, Ofcom, ITU는 쌍 수와 화면 면적 %만 쓰고, WCAG는 0.1° 미세 패턴 예외만 둔다 ([ITC GN 2001](https://www.ofcom.org.uk/__data/assets/pdf_file/0021/16248/gn_flash.pdf); [WCAG 2.2](https://www.w3.org/TR/WCAG22/#dfn-general-flash-and-red-flash-thresholds), 원문). 도구 쪽에서도 IRIS 코드에는 cpd 개념이 없고, Harding FPA 매뉴얼 4.0에는 "cycles", "degree", "viewing distance"가 패턴과 관련해 한 번도 나오지 않는다 ([IRIS PatternDetection.cpp](https://github.com/electronicarts/IRIS/blob/main/src/PatternDetection.cpp); [HardingFPA-X Manual 4.0](https://www.hardingfpa.com/assets/Software-releases/Manuals/HardingFPA-X-Users-Manual-4.0.pdf), 원문). cpd를 계산하는 유일한 학술 구현인 Rodrigues 2015도 고정 대역을 쓰지 않는다. cpd를 Wilkins 2005의 연속 위험곡선에 넣는다 ([Rodrigues 2015](https://fenix.tecnico.ulisboa.pt/downloadFile/844820067124507/Resumo%20Algargado%20Automatic%20Detection%20of%20Epileptogenic%20Video%20Content_NR.pdf), 원문). 리뷰 문헌에 나오는 범위 표현은 1–4, 2–4, 2–3, "4–40′ → 약 3", ~1–8이며, **1–10은 어디에도 없었다** (reviews_models 조사, 원문). 팀원 판정기 `vdcoreops_v2.py`의 `cpd_min=1.0, cpd_max=10.0`에는 출처 주석이 없다. 우리 `measure.py`는 주석에 "팀원 판정기 기준을 따름 — 사용자 결정 2026-11-09"라고 적어 이를 이어받았다 (`C:\dev\lumi-vision\lumi-lab-ryujungho2-collab\src\lumi_synth\measure.py` 219–249행, 원문). 하한 1은 Wilkins 1979의 1–4 c/deg 최적 구간 하단과 맞는다. 그러나 집단 곡선에서 0.5 c/deg 값은 약 0.05이고, log10으로 계산한 Wilkins 2005 모델은 약 0.26을 준다[계산]. 어느 쪽을 쓰든 1 미만의 위험이 0은 아니다. 상한 10은 측정 범위(최대 8)를 넘으므로 실험 근거가 없다. **가장 가까운 인용 가능한 대체값은 Fisher 2026의 "~1–8 cycles per visual degree"**다. 이것은 전문가 합의이고, 본문이 근거로 든 것은 Wilkins 1979다. 다만 8이라는 상한도 "8 이상은 안전"이라는 증거가 아니라 **측정 격자의 끝**이다 ([Fisher 2026](https://dspace.library.uu.nl/server/api/core/bitstreams/511a8fa4-deff-4845-b9af-c3d0d64e969c/content), 원문).

**'5쌍'은 표준보다 한 쌍 엄격하다.** ITC 2001 §5 원문은 "more than five light-dark pairs of stripes in any orientation"이고, 이 문구가 Ofcom Annex 1과 BT.1702-3 Attachment 1로 그대로 넘어갔다 ([ITC GN 2001](https://www.ofcom.org.uk/__data/assets/pdf_file/0021/16248/gn_flash.pdf); [BT.1702-3](https://www.itu.int/rec/R-REC-BT.1702), 원문). Wilkins 2005는 이 규칙이 "**evolved on an ad hoc basis and were not developed from a model**"라고 했다. 이어 20인치 TV를 7H에서 보는 기하에서 "5쌍이면 전체 화면과 1/4 화면의 위험이 비슷하고, 5→8쌍이면 약 두 배"라는 계산으로 사후 정당화했다 ([Wilkins 2005](https://visualstress.info/2005-169.pdf), 원문). EFA 2005 합의문 본문 목록은 "Five light-dark pairs … if the stripes change direction, oscillate, flash or reverse in contrast"와 "**Eight light-dark pairs** of stripes, if the pattern is unchanging or continuously and smoothly drifting"이다. 초록과 근거 절의 "eight stripes"는 쌍의 약칭으로 읽어야 한다. 본문 목록, 출처 모델(Wilkins 2005), 동반 리뷰(Fisher 2005)가 모두 "pairs"라고 쓰기 때문이다 ([Harding et al. 2005, Epilepsia 46:1423–1425 출판본 사본](https://puc.sd.gov/commission/dockets/electric/2019/el19-027/prefiledexhibits/CRWII/A178.pdf), 원문). 이로써 `STANDARDS.md`에 '미확인'으로 남은 항목이 해소된다. 주의할 점이 있다. "no more than five pairs"는 5쌍까지 허용하고 6쌍부터 위험이라는 뜻이다. 따라서 **ITC, EFA, Wilkins 2005, Harding FPA, IRIS 모두 경계를 5와 6 사이에 둔다.** 5쌍 자체를 위험으로 보는 우리 ≥5 규칙의 선례는 Rodrigues 2015의 "minComp ≥ 5"뿐이다. 이 값은 ITU의 "more than 5 … pairs, 10 stripes"를 극성별 5개로 해석한 결과다 ([Rodrigues 2015](https://fenix.tecnico.ulisboa.pt/downloadFile/844820067124507/Resumo%20Algargado%20Automatic%20Detection%20of%20Epileptogenic%20Video%20Content_NR.pdf), 원문).

**'정지 40%'는 ITC에서만 나온 숫자다.** ITC §5.1(i)는 "the stripes are stationary and the pattern occupies more than 40% of the displayed screen area"다. 이를 유도하거나 실험으로 뒷받침한 문헌은 없다. Wilkins 2005는 40%를 따로 반박하지 않았다. 대신 "proportion of patients affected by any pattern that occupies less than 25% of the screen is small regardless of the number of lines"라는 계산으로 **모든 패턴에 25%를 적용**하자고 했다. EFA 2005와 Fisher 2026도 25%(또는 0.006 sr, 시야 25%) 하나만 쓴다 ([Wilkins 2005](https://visualstress.info/2005-169.pdf); [EFA 2005](https://puc.sd.gov/commission/dockets/electric/2019/el19-027/prefiledexhibits/CRWII/A178.pdf); [Fisher 2026](https://dspace.library.uu.nl/server/api/core/bitstreams/511a8fa4-deff-4845-b9af-c3d0d64e969c/content), 원문). 따라서 40%는 'ITU-R BT.1702-3 informative 부속서를 구현한다'는 범위 안에서만 인용 가능하다. 의학 근거로 내세울 수는 없다.

**'0.5초'는 경험값이다.** Wilkins 2005 원문은 "**There are no studies that have formally investigated the effects of pattern duration**, but … it has been rare indeed for a paroxysmal response to pattern to have occurred in less than 0.5s. The latency is typically in the order of 1-2s"라고 쓴다. EFA 2005 근거 절도 "based on experiences of both Wilkins and Harding"이라고 밝힌다 ([Wilkins 2005](https://visualstress.info/2005-169.pdf); [EFA 2005](https://puc.sd.gov/commission/dockets/electric/2019/el19-027/prefiledexhibits/CRWII/A178.pdf), 원문). 1980년 리뷰도 IPS 지속 시간을 체계적으로 조사한 연구가 없다고 했다 ([Wilkins 1980 p.90](https://visualstress.info/1980-22.pdf), 원문). **ITC, Ofcom, ITU-R BT.1702-3의 패턴 조항에는 지속 시간 조건이 아예 없다** (표준 비교표, 원문). 0.5초는 Wilkins 2005와 EFA 2005에서 의학 리뷰로, 그리고 Harding FPA("A pattern must remain stationary for at least 0.5 seconds in order to fail")와 IRIS(`TimeThreshold: 0.5`)라는 도구로 퍼졌다. Fisher 2022는 이 제안을 "Guidelines that are employed in Britain and Japan"이라고 서술했다. ITC 원문에 50 cd/m²와 0.5 s가 없으므로 이는 오귀속이다 ([Fisher 2022](https://www.epilepsy.com/sites/default/files/2022-10/Epilepsia_2022_fisher_visually_sensitive_seizures.pdf), 원문).

## 움직임 판정에는 어떤 표준도 숫자를 주지 않는다

정지, 변화, 흐름을 가르는 문구는 ITC 2001부터 BT.1702-3까지 세 개의 정성어뿐이다. "stationary", "change direction, oscillate, flash, or reverse in contrast", "obviously flow smoothly across, into, or out of the screen in one direction"이다 ([Ofcom Section 2 Annex 1](https://www.ofcom.org.uk/siteassets/resources/documents/tv-radio-and-on-demand/broadcast-guidance/programme-guidance/broadcast-code-guidance/section-2-guidance-notes.pdf); [BT.1702-3](https://www.itu.int/rec/R-REC-BT.1702), 원문). Jordan & Vanderheiden 2024는 "there is **no specific, actionable guidance available on motion compensation thresholds**"라고 명시하고 "When is motion so fast that it is perceived as a flicker rather than motion?"이라고 묻는다 ([JV24, PMC11872230](https://pmc.ncbi.nlm.nih.gov/articles/PMC11872230/), 원문). Wilkins 2005도 "There is difficulty in separating continuous drift from vibratory movement, particularly on refreshed screens"라고 인정했다. 자동 분석에 대해서는 개정 지침이 "insufficient … because they fail to define a stripe"라고 썼다 ([Wilkins 2005](https://visualstress.info/2005-169.pdf), 원문). 공개된 유일한 도구 수준 정의는 Harding FPA의 타일 단위 지속 판정이다. 마스크 색상은 0–1/6 s, 1/6–1/3 s, 1/3–1/2 s, ">1/2 second = FAILURE"로 나뉘고, "spatial patterns which drift … are excluded"라고 되어 있다. 그러나 이동량 임계(px/s)는 공개되지 않았다 ([HardingFPA-X Manual 4.0](https://www.hardingfpa.com/assets/Software-releases/Manuals/HardingFPA-X-Users-Manual-4.0.pdf), 원문).

그렇다면 숫자는 1차 실험 조건에서 가져와야 한다. 이때 그것은 '가이드라인 정의'가 아니라 '근거 문헌의 실험 조건'으로 표기해야 한다. Binnie, Findlay & Wilkins 1985 실험 I의 조건은 환자 10명, 2 c/deg, 대비 0.81, 93 cd/m², 진폭 반주기(15′)의 삼각파 진동이다. 결과는 진동 5/10/15/20 Hz에서 **0/0.4/0.97/0.5**, 정지 **0.38**, 같은 각속도(2.5–10 °/s)의 흐름 **0–0.04**였다 ([Binnie 1985, doi:10.1016/0013-4694(85)91065-x](https://visualstress.info/1985-40.pdf), 원문 표 II). 이 표가 판정기 설계에 주는 함의는 강하다. 2 c/deg에서 15 Hz로 진동하는 줄무늬(평균 속도 7.5 °/s)와 7.5 °/s로 흐르는 줄무늬는 **국소 휘도 변조율이 같다(15 Hz)**. 그런데 위험은 0.97 대 0.04로 정반대다[계산 + 추론]. **픽셀 휘도 변화량이나 평균 속도만 보는 판정기는 진동과 흐름을 원리적으로 구분할 수 없다.** 운동 방향의 반전 빈도나 위상 반전을 따로 재야 한다. 잔상 실험도 있다. 망막에 고정된 줄무늬 잔상에서는 7명 모두 PPR이 없었으므로, 정지 패턴의 유발성은 눈 움직임에 따른 망막 영상 움직임에 의존한다 ([Wilkins 1980 p.96](https://visualstress.info/1980-22.pdf), 원문). 정지 패턴도 위험(0.38)하므로 정지를 면제해서는 안 된다.

Gamma 연구는 이 그림을 복잡하게 만든다. 건강인에서 drifting 격자는 정지 격자보다 gamma power가 크고, 속도가 더 빨라지면 다시 억제된다 (Muthukumaraswamy & Singh 2013; Orekhova 2018, 2차). 반면 Fisher 2022는 "Drifting gratings that induce both discomfort and a large hemodynamic response do not provoke seizures, presumably because they fail to synchronize the activity they evoke"라고 정리한다 ([Fisher 2022](https://www.epilepsy.com/sites/default/files/2022-10/Epilepsia_2022_fisher_visually_sensitive_seizures.pdf), 원문). **gamma 대리지표와 환자 PPR 자료가 흐름 패턴에서 반대 방향을 가리키므로**, 판정기의 움직임 규칙은 PPR 자료(Binnie 1985)를 따르는 것이 맞다. Fisher 2026의 진동 범위 "3-60 Hz [Maximum risk is from 10 to 20Hz]"는 깜빡임 주파수 범위를 패턴 진동에 옮긴 것으로 보인다. 패턴 진동의 3 Hz 하한을 측정한 근거는 확인하지 못했다 ([Fisher 2026](https://dspace.library.uu.nl/server/api/core/bitstreams/511a8fa4-deff-4845-b9af-c3d0d64e969c/content), 원문; 추론).

## ITC 2001 한 장이 표준 계보 전체를 복사했다

현존 원문으로 확인할 수 있는 가장 오래된 패턴 정의는 **ITC Guidance Note 2001(2002 편집 수정)**이다. 1994년 초판과 1999년 개정판은 원문을 찾지 못했다. 1993년 Golden Wonder Pot Noodle 광고 사건이 계기였다 ([ITC GN 2001](https://www.ofcom.org.uk/__data/assets/pdf_file/0021/16248/gn_flash.pdf), 원문; [JV24](https://pmc.ncbi.nlm.nih.gov/articles/PMC11872230/), 2차). ITC §5의 문구는 지금도 거의 그대로 쓰인다. "parallel or radial, curved or straight, and may be formed by rows of repetitive elements such as **polka dots**", >5쌍, 정지 40%, 변화 25%, 어두운 막대 <160 cd/m² & 차이 ≥20 cd/m², smooth-flow 면제가 그 내용이다. 명암 조건은 감마 2.2와 BT.500의 피크화이트 200 cd/m²를 가정한 깜빡임 규칙에서 빌려 왔다. 참고문헌은 Harding & Jeavons 1994, Harding 1998, Wilkins 1995다. Ofcom은 2005년 7월 25일 이 문서를 문구 그대로 재발행했고, 2017년 Issue 12 Annex 1까지 바꾸지 않았다 ([Ofcom](https://www.ofcom.org.uk/siteassets/resources/documents/tv-radio-and-on-demand/broadcast-guidance/programme-guidance/broadcast-code-guidance/section-2-guidance-notes.pdf), 원문).

**ITU-R BT.1702는 18년 동안 패턴 수치를 넣지 않았다.** -0(2005)은 플래시 기준만 ITC에서 옮겼다. 패턴은 측정 프레임워크 그림에서 "Appropriate block averaging reduces spurious detections due to noise and fine patterns", "Motion compensation reduces spurious detections due to panning and object motions"처럼 **오검출 요인**으로만 다뤘다. -1(07/2018)에도 없다. -2(10/2019)는 HDR용 1/17 Michelson 규칙을 플래시에만 도입했다. 160 cd/m²에서 끊김이 없도록 1/17 = ((160+20)−160)/((160+20)+160)으로 맞춘 값이다. **-3(11/2023)**에서 처음으로 Guideline 2(정성 정의, Annex 1 본문)와 **informative Attachment 1**("used by some administrations")이 생겼다. 후자에 >5쌍, 40%/25%, "Guideline 1과 동일한" 명암, smooth-flow 면제가 들어갔다 ([BT.1702 각 판](https://www.itu.int/rec/R-REC-BT.1702), 원문). Annex 2 Table 1 NOTE는 "If the darker image of a flash **or pattern** is above this level…"이라고 써서 1/17 규칙을 패턴에 명시적으로 적용한다. 정리하면 ITU는 패턴의 정량 기준을 비구속 권고 안에서도 다시 '참고'로 한 단계 낮춘 셈이다. -3의 승인 월은 PDF 표지 11/2023과 URL 관례 "202302"가 엇갈려 확인이 필요하다.

**ISO 9241-391:2016**(2021 재확인)은 목차상 5.3.1 "Potentially harmful regular patterns" 조항을 두지만 본문은 유료라 확인하지 못했다. JV24에 따르면 ISO와 Ofcom 모두 "across, into, or out of the screen in one direction" 면제를 갖고 있다. DIS 서론은 IWA 3:2005를 경유한 계보와 BT.1702·WCAG 2.0 참조를 밝힌다 ([SIST EN ISO 9241-391 미리보기](https://cdn.standards.iteh.ai/samples/37651/9af06303a47b4841ae380ed118cff532/SIST-EN-ISO-9241-391-2016.pdf), 원문: 목차·서론; [JV24](https://pmc.ncbi.nlm.nih.gov/articles/PMC11872230/), 2차). **일본 NHK·민방련 지침**은 독자 계보다. 포켓몬 사건 뒤 1998-04-08에 만들어졌고 2006-04-01에 BT.1702-0을 참고해 개정했다. 패턴 조항은 "規則的なパターン模様（縞模様、渦巻き模様、同心円模様など）が、画面の大部分を占めることも避ける" 한 줄이며 **정량 수치가 없다**. 2006년에 참고한 BT.1702-0 자체에 패턴 수치가 없었으므로 정량화될 계기도 없었다 ([NHK·민방련 지침 별지](https://www.tcnet.ne.jp/wp-content/uploads/community_ch/broadcasting-plan-standard/standard_Exhibit.pdf), 원문). JV24가 보고한 2020 개정판은 원문 링크가 404여서 패턴 문구 변경 여부를 확인하지 못했다.

**WCAG 2.0/2.1/2.2에는 패턴 성공기준이 없다.** "a fine, balanced, pattern such as white noise or an alternating checkerboard pattern with 'squares' smaller than **0.1 degree**"를 플래시 임계에서 빼는 예외만 있다 ([WCAG 2.2](https://www.w3.org/TR/WCAG22/#dfn-general-flash-and-red-flash-thresholds), 원문). 0.1° 사각형은 주기 0.2°, 축 방향 기본파 약 5 c/deg에 해당한다[계산]. 이 예외는 위험 기준이 아니라 오검출 방지 규정이다. 2004-02-17 Vanderheiden 제안에는 ITC 2001을 옮긴 정지/변화 줄무늬 기준(1024×768에서 355×268 px의 40%/25%)이 Level 3로 들어 있었으나 2008 최종판에서 사라졌다. 삭제 사유를 남긴 기록은 찾지 못했다 ([w3c-wai-gl 2004JanMar/0306](https://lists.w3.org/Archives/Public/w3c-wai-gl/2004JanMar/0306.html), 2차). **trace24**(Jordan & Vanderheiden 2024, pseGuidelines)의 제안 provision은 플래시 전용이다. 패턴 조항은 없고, 시청 거리별 시야각 표(Table 2)와 23인치 FHD를 71.1 cm에서 보는 기준 화면을 준다 ([pseGuidelines](https://github.com/traceRERC/pseGuidelines); [JV24](https://pmc.ncbi.nlm.nih.gov/articles/PMC11872230/), 원문).

의학 쪽 계보는 따로 흐른다. **Wilkins 2005**는 Binnie 2002의 깜빡임 곱셈 모델을 패턴으로 확장했다. "The assumption that the various parameters have effects that are independent is **almost certainly false**"를 인정하면서 5/8/12쌍, 25%(≈0.006 sr), 밝은 줄 50 cd/m², 0.5 s를 제안했다. 또 ITC의 160 cd/m² 문턱을 "unfortunate choice"라고 비판했다. 어두운 막대 161/밝은 막대 200은 통과하는데(위험 0.66), 159/179는 걸리는(위험 0.33) 불연속 때문이다 ([Wilkins 2005](https://visualstress.info/2005-169.pdf), 원문). **EFA 2005 합의**는 이를 "consensus rather than guidelines"로 채택하면서 흐름을 12쌍이 아닌 8쌍으로 묶었다. 근거는 "experience indicates that drifting patterns have risks similar to those that are stationary"뿐이다. 이해관계도 공개됐다. Harding은 검사 시스템 제조사 Cambridge Research Systems와 재정 관계가 있다 ([EFA 2005](https://puc.sd.gov/commission/dockets/electric/2019/el19-027/prefiledexhibits/CRWII/A178.pdf), 원문). **Fisher 2005** 결론 문단은 정지와 변화를 나누지 않고 "five stripes" 단일 기준으로 줄였다. **Fisher 2022**는 "five … oscillating stripes (eight if drifting smoothly)"로 바꿔 쓰며 pairs를 뺐다 ([Fisher 2005 저자 원고](https://visualstress.info/2005-168.pdf); [Fisher 2022](https://www.epilepsy.com/sites/default/files/2022-10/Epilepsia_2022_fisher_visually_sensitive_seizures.pdf), 원문). **Fisher 2026 EF Working Group 합의**(Epilepsia 67(2):565–581, doi:10.1111/epi.18702)는 체계적 문헌고찰이 아니라 전문가 합의다. 5쌍/8쌍, 50 cd/m², 0.006 sr의 패턴 규칙을 본문에서 없앴고, "stripes or zig-zags having 1-8 cycles that subtend 1 degree", 시야 25%, 0.5 s, 진동 3–60 Hz로 바꿨다 ([Fisher 2026](https://dspace.library.uu.nl/server/api/core/bitstreams/511a8fa4-deff-4845-b9af-c3d0d64e969c/content), 원문). 이 합의는 "signal analysis"로 유발 패턴을 식별할 수 있다고도 했다 ([Stanford Health Care 요약](https://stanfordhealthcare.org/publications/963/963844.html), 2차). 계산형 판정기를 만드는 방향을 지지하는 문장이다. 조사 노트 사이에 엇갈림이 하나 있다. 최근 연구 조사는 Fisher 2026 본문 수치를 '미확인'으로 남겼지만, 리뷰 조사는 Utrecht 저장소 출판본을 직접 읽었다. 이 보고서는 후자를 따른다.

| 문서 / 판 | 쌍 수 | 명암 | 면적 정지 / 변화 | 지속 | cpd·시거리 | 움직임 | 규범성 | 등급 |
|---|---|---|---|---|---|---|---|---|
| ITC GN 1994 / 1999 | ? | ? | ? | ? | ? | ? | 면허 지침 | 미확인 |
| ITC GN 2001(+2002) | >5 | 어두운 <160 & ≥20 cd/m² | >40% / >25% | 없음 | 없음 | 흐름 면제(정성) | 지침 | 원문 |
| Ofcom Annex 1 (2005, 2017 Issue 12) | ITC와 동일 | 동일 | 동일 | 없음 | 없음 | 동일 | Code 2.12 해석 지침 | 원문 |
| ITU-R BT.1702-0/-1/-2 (2005/2018/2019) | 없음 | (-2: HDR 1/17, 플래시만) | 없음 | – | – | 패턴은 오검출 요인 | 권고 | 원문 |
| ITU-R BT.1702-3 (11/2023) | Att.1: >5 | Guideline 1과 동일(1/17 포함) | Att.1: >40% / >25% | 없음 | 없음; 시청자 조언 ≥2 m | Att.1에 흐름 면제 | 본문 정성 + informative 수치 | 원문 |
| ISO 9241-391:2016 | ? | ? | ? | ? | ? | 흐름 면제(JV24) | 국제표준 | 목차 원문 / 본문 미확인 |
| NHK·민방련 1998/2006 | 없음 | 없음 | "画面の大部分" | 없음 | 없음 | 없음 | 업계 자율 | 원문 |
| NHK·민방련 2020 | ? | ? | ? | ? | ? | ? | – | 미확인 |
| WCAG 2.0/2.1/2.2 | 패턴 SC 없음 | – | – | – | 0.1° 미세 패턴 예외(플래시) | – | 규범 | 원문 |
| trace24 (2024) | 패턴 조항 없음 | – | – | – | 시거리 표 | "no actionable guidance" | 제안 | 원문 |
| Wilkins 2005 | 변화 5 / 정지 8 / 흐름 12 | 밝은 줄 >50 cd/m² | 25% (0.006 sr) | >0.5 s | 연속 cpd 곡선 | 진동≈정지×2 | 연구 제안 | 원문 |
| EFA 2005 합의 | 변화 5 / 정지·흐름 8 | >50 cd/m² | 25% / 0.006 sr | >0.5 s | 중심 10° | 정성 | 합의 | 원문 |
| Fisher 2026 EF 합의 | 쌍 규칙 폐기 | (색도 차 언급) | 시야 25% | >0.5 s | ~1–8 c/deg | 진동 3–60 Hz | 합의 | 원문 |
| Xbox XAG 118 (2022) | 없음 | 대비 >10% | ~20% (구분 없음) | – | – | – | 플랫폼 지침 | 원문 |

## 판정기 다섯 개가 같은 '줄'을 서로 다르게 센다

공개 검출 방식은 세 계열로 나뉜다. FFT 피크와 윤곽선을 쓰는 IRIS, 선분 검출과 연속 위험곡선을 쓰는 Rodrigues 2015, 그리고 비공개 상용 엔진(Harding FPA, Vidcheck, Baton, Cel-Soft)이다. 우리와 팀원의 판정기는 네 번째 방식인 **방향 스캔선 1D 극값 계수**를 쓴다.

**EA IRIS**는 '쌍'을 세지 않는다. 처리 순서는 이렇다. 휘도 프레임을 프레임마다 min-max 정규화한다. FFT 스펙트럼에서 Otsu로 고른 피크를 지우고 역변환한다. 원본과의 차이(absdiff > 50, 8bit)를 이진화해 윤곽선을 만들고, `matchShapes < 0.7`로 묶인 가장 큰 무리의 크기를 `nComponents`로 쓴다. 판정 조건은 회전 외접사각형 면적 ≥25%, `nComponents ≥ 6`(`MinStripes`), 밝은 성분 상대휘도 ≥0.25이고, 이 상태가 0.5 s 이어지면 Fail이다. 코드와 README에는 ITU, Ofcom, Harding, Wilkins 인용이 하나도 없다. 설정의 `RelativeDarkLuminanceThreshold` 0.8은 읽기만 하고 판정에 쓰지 않는다. 정지와 움직임도 구분하지 않는다 ([IRIS PatternDetection.cpp, 커밋 d96978a](https://github.com/electronicarts/IRIS/blob/main/src/PatternDetection.cpp); [appsettings.json](https://github.com/electronicarts/IRIS/blob/main/config/appsettings.json), 원문). 정규화 때문에 절대 명암(cd/m²) 조건이 구조적으로 빠진다. 단위도 '쌍'에서 '블롭'으로 바뀌어, 극성에 따라서는 약 3쌍만으로 6을 채울 수 있다(추론).

**Harding FPA**는 "pairs of alternating light-dark bars of at least 20cd/m2 contrast"를 센다. 정지 패턴이 "6 or more stationary pairs which persist for more than 0.5 seconds"이고 면적 >40%이면 Fail이다. 진동, 점멸, 명암 반전 패턴은 "both as a pattern and as a flashing stimulus"로 분석해 깜빡임의 25% 기준을 적용한다. 체커보드는 무시하고, 카메라 팬과 줌은 Ofcom 규칙상 면제한다 ([How to interpret HardingFPA results](https://www.hardingfpa.com/technical-support/how-to-interpret-hardingfpa-results/), 원문·벤더). 매뉴얼 FAQ 8은 "fewer than 6 light-dark bars"라고 써서 웹의 '쌍'과 단위가 엇갈린다. 직교 방향 국소 구조나 텍스트가 패턴을 쪼개면 Pass가 된다는 한계도 스스로 밝힌다 ([Manual 4.0](https://www.hardingfpa.com/assets/Software-releases/Manuals/HardingFPA-X-Users-Manual-4.0.pdf), 원문). **PEAT** 1.6 사용자 가이드는 일반 플래시, 적색 플래시, 연장 플래시 결과만 나열한다. 패턴 분석은 문서상 확인되지 않는다 ([PEAT User Guide](https://trace.umd.edu/?p=719), 원문). **Rodrigues 2015**(IST Lisboa 확장초록)의 처리 순서는 이렇다. LSD 선분을 각도 빈으로 묶는다(빈 크기 ≥9, Wilkins 2005 인용). Otsu로 극성별 연결성분을 나눈다(각 ≥5, ITU 인용). θ = 2·arctan(A·H/2/distanceVIS)로 시야각을 구하고 cpd = N/θ를 계산한다. 마지막으로 Wilkins 2005의 공간 주파수, 면적, Michelson, 평균 휘도 회귀식을 곱해 위험을 낸다. 정지/움직임 구분과 지속 조건은 없고, 검증은 합성 영상 2개와 실제 영상 1개뿐이다 ([Rodrigues 2015](https://fenix.tecnico.ulisboa.pt/downloadFile/844820067124507/Resumo%20Algargado%20Automatic%20Detection%20of%20Epileptogenic%20Video%20Content_NR.pdf), 원문). **검증용 공개 데이터도 없다.** Jordan 2025는 패턴 검출 성능을 평가하지 않았다. pse-test-media의 `area_patterns`는 점멸 면적 시험용 도형이지 규칙 줄무늬 세트가 아니다 ([Jordan 2025, doi:10.1007/978-3-031-93848-1_7, PMC12249941](https://pmc.ncbi.nlm.nih.gov/articles/PMC12249941/); [pse-test-media](https://github.com/traceRERC/pse-test-media), 원문).

**팀원 `vdcoreops_v2.py`**의 처리 순서는 다음과 같다. 0–165°를 15° 간격으로 나눈 12방향마다 중앙 5개 offset을 두어 총 60개 직선을 긋는다. 히스테리시스 5 cd/m²로 극값을 찾고, 이웃 극값 사이가 "ITU-R BT.1702-3 명암 대비 기준"(<160 & ≥20, 또는 ≥160 & Michelson >1/17)과 1–10 cpd를 만족하면 유효한 줄로 센다. 연속 유효 줄 수를 2로 나눈 값이 5쌍 이상이면 줄무늬다. 면적은 줄무늬 픽셀이 아니라 국소 고대비 밀도 영역으로 재며 25% 하나만 쓴다. 0.5 s run 검출도 있다. 시야각 식 "2*theta = 2 * arctan(x / (2 * d))"에만 "3694790.pdf Appendix A"라는 출처가 붙어 있고, 56/68 cm, 15인치, 1–10 cpd, 확률식 `eta²/(1+eta²)`에는 출처가 없다 (팀원 코드, 원문). **우리 `measure.py analyze_pattern`**은 같은 계열이지만 차이가 있다. 화면 전체를 12방향 × 1픽셀 간격으로 스캔한다. 극값 히스테리시스 20 cd/m²가 곧 ΔL ≥20 조건이다. 유효한 줄 조건은 `min(두 극값) < 160 cd/m²`와 `1 ≤ ppd/(2h) ≤ 10`이다. 연속 유효 줄 9개(극값 10개) 이상이면 5쌍이고, 그 구간 픽셀을 줄무늬 픽셀로 표시한다. 움직임은 줄무늬 픽셀 중 ±0.5초 안에 픽셀 전이(ΔL ≥20)가 있었던 비율이 ≥10%인지로 판정한다. 면적 한계는 움직임이면 >25%, 정지면 >40%다. 지속은 아무 1초 창 안의 후보 프레임이 ≥ceil(0.5·W)인지로 본다 (`measure.py` 241–249, 306–345, 408–460행, 원문). 팀원 코드와 달리 **HDR용 1/17 Michelson 분기는 구현돼 있지 않다**. 코드 주석은 smooth-flow 면제를 미구현으로 명시하고, 흐름도 움직임으로 보아 25%를 적용한다.

| 항목 | EA IRIS | Harding FPA | Rodrigues 2015 | 팀원 vdcoreops_v2 | 우리 measure.py |
|---|---|---|---|---|---|
| 원리 | FFT 피크 제거 → 차이 윤곽선 → 모양 묶기 | 비공개 ("light-dark bars") | LSD 선분 + 각도 빈 + Otsu 성분 | 60 스캔선 1D 극값 | 12방향 × 1px 전체 스캔 1D 극값 |
| 세는 단위 | 비슷한 윤곽선 ≥6 | 6쌍(웹) / 6 bars(매뉴얼) | 선분 ≥9, 극성별 성분 ≥5 | 쌍 ≥5 | 쌍 ≥5 |
| 명암 | 프레임 정규화, 밝은 성분 상대휘도 ≥0.25 | ≥20 cd/m² | Michelson → Wilkins 곡선 | <160&≥20, 또는 ≥160&>1/17 | ≥20 & 어두운 쪽 <160 |
| cpd | 없음 | 문서에 없음 | N/θ → Wilkins 연속 곡선 | 1–10 (출처 없음) | 1–10 (팀원 따름) |
| 시청 조건 | 없음 | 문서에 없음 | 사용자 입력 | 56/68 cm, 15인치 | 56 cm, 15인치 16:9 |
| 면적 측정 | 회전 외접사각형 ≥25% | 타일 마스크, 정지 >40% | 성분 면적 합 → 곡선 | 고대비 밀도 영역 >25% | 줄무늬 픽셀 >25% / >40% |
| 정지/움직임 | 없음 | 정지만 패턴 Fail, drift 제외, pan/zoom 면제, 진동은 깜빡임 25% | 없음 | 없음 | 전이 비율 ≥10% → 25%, 흐름 면제 미구현 |
| 지속 | 0.5 s 창 | >0.5 s 정지 지속 (5프레임 평균) | 없음 | 0.5 s run | 1 s 창 중 ≥0.5 s |
| 출처 표기 | 없음 | Ofcom | Wilkins 2005, ITU | 일부(시야각만) | 출처 구분 주석 있음 |
| 등급 | 원문(코드) | 원문(벤더) | 원문 | 원문(코드) | 원문(코드) |

이 비교에서 두 가지 위험이 보인다. 첫째, **면적의 측정 정의가 도구마다 달라서** 같은 영상에서도 판정이 갈릴 수 있다. 외접사각형(IRIS)과 고대비 밀도(팀원)는 줄무늬 픽셀(우리)보다 면적을 크게 잡는다. 둘째, 우리 스캔선 방식은 체커보드를 행 방향으로 지나도 명암이 교대하므로 줄무늬로 셀 가능성이 높다(추론). 이것은 "rows of repetitive elements such as polka dots"를 포함하는 ITU 정의에는 맞는다. 그러나 체커보드를 무시하는 Harding, 그리고 직교 성분이 위험을 줄인다는 Wilkins 1979와는 반대 방향의 보수성이다.

## 우리 판정기의 파라미터별 권고

원칙은 하나다. 각 파라미터에 대해 **'무엇을 구현하는가'(표준 준수)와 '왜 위험한가'(의학 근거)를 분리해서 인용**하고, 둘이 갈리는 곳은 민감도 분석으로 보고한다. Wilkins 2005가 지적했듯 표준 문구는 "fail to define a stripe"이므로, 줄의 정의 자체가 LUMI의 구현 선택이라는 점을 논문 방법 절 첫머리에 밝혀야 한다.

**공간 주파수(현재 1–10 cpd).** 주 분석값은 **1–8 c/deg**로 바꾸고 Fisher 2026(합의, "About 50% of patients are sensitive in the range of 1-8 cycles per degree")과 Wilkins 1979(실험, 최적 1–4 c/deg, 8명)를 함께 인용할 것을 권한다. 1–10을 유지하려면 "Fisher 2026의 상한 8을 측정이 없는 구간(8–10)까지 보수적으로 넓힌 LUMI 구현 선택이며, 팀원 판정기와의 호환을 위한 값"이라고 적어야 한다. 출처가 있는 것처럼 쓰면 안 된다. 하한 1에 대해서는 Wilkins 1995 집단값(0.5 c/deg ≈ 0.05, 2차)과 Wilkins 2005 모델값(약 0.26, 계산)이 엇갈린다는 점을 함께 보고한다. 민감도 분석은 1–4, 1–8, 1–10, 0.5–10 네 대역으로 같은 영상을 판정하고, Wilkins 2005 3차 log 다항식을 가중치로 쓰는 연속 버전을 추가한다(Rodrigues 2015 방식). 주변 시야의 고리 패턴은 최적 주파수가 낮아진다는 점(Wilkins 1980 그림 19)과, 128×72 같은 저해상도 트랙은 약 1.9 cpd까지만 표현할 수 있다는 점(`measure.py` `pixels_per_degree` docstring)도 한계로 적는다.

**쌍 수(현재 ≥5).** 표준 준수 판정은 **>5(=6쌍 이상)**로 두고 ITC 2001 / BT.1702-3 Attachment 1을 인용하는 것이 정직하다. 현재의 ≥5는 "모든 표준·합의·상용 도구보다 1쌍 엄격한 보수적 구현 선택"으로 보고하고, 선례로 Rodrigues 2015를 든다. 주의할 점이 있다. Wilkins 2005와 EFA 2005의 "five pairs"는 '5쌍까지 허용'이라는 뜻이므로 ≥5의 근거가 될 수 없다. 민감도 분석은 ≥5, ≥6, 그리고 Wilkins 2005·EFA 2005 계열의 움직임 의존 쌍 수(변화 >5, 정지 >8, 흐름 >12 또는 >8)의 세 변형으로 한다. 쌍 수가 실험의 독립변수가 아니었고 cpd × 크기에서 유도된 양이라는 점(Wilkins 1979, 2005)을 논의에 넣는다.

**명암(ΔL ≥20 cd/m², 어두운 쪽 <160).** BT.1702-3 Attachment 1 → Guideline 1을 인용한다. 이 값이 깜빡임 규칙에서 ad hoc으로 차용된 것이라는 점, 그리고 Wilkins 2005가 160 cd/m² 불연속을 "unfortunate choice"라고 비판했다는 점을 함께 쓴다. 의학 근거로는 Michelson 약 0.3 부근 급상승(Wilkins 1979 Study 8)을 인용한다. 현재 구현에는 어두운 쪽 ≥160일 때의 1/17 Michelson 분기가 없다. SDR(피크화이트 200 cd/m² 가정)만 다룬다면 범위 한정으로 명시하고, 팀원 판정기와 비교할 때는 이 차이를 표로 적는다. 극값 히스테리시스 20은 ΔL 조건과 겸용하는 LUMI 구현 선택이고 팀원은 5다. 민감도 분석은 히스테리시스 5/10/20, Michelson 0.2/0.3 문턱 변형, Wilkins 2005의 "밝은 줄 >50 cd/m²" 변형으로 한다. 피크화이트 200 cd/m²와 감마 2.2 가정은 ITC 각주(BT.500)를 인용한다.

**평균 휘도.** 어떤 표준도 평균 휘도를 패턴 조건으로 쓰지 않는다. 현재 판정기에 별도 조건이 없는 것은 표준 준수 관점에서 문제가 없다. 논의에서는 평균 휘도 변화가 1 log unit을 넘어야 위험이 뚜렷이 바뀐다는 점(Wilkins 1980)과, Wilkins 2005·EFA 2005의 50 cd/m²가 모델값이라는 점을 함께 보고한다. 선택적 민감도 분석으로 밝은 줄 >50 cd/m² 조건을 켜고 끈 결과를 비교한다.

**면적(현재 정지 >40%, 움직임 >25%).** BT.1702-3 Attachment 1을 인용해 유지한다. 단 40%는 ITC의 ad hoc 값이고 의학 문헌(Wilkins 2005, EFA 2005, Fisher 2026)은 모든 패턴에 25%를 쓴다고 함께 적는다. 민감도 분석은 '모든 패턴 25%'(IRIS와 의학 문헌 기준) 변형이다. 면적을 '줄무늬 픽셀'로 잰다는 측정 정의를 명시하고, 외접사각형(IRIS)과 밀도 영역(팀원) 정의로 다시 잰 결과를 비교한다. 시각도 기반으로 일반화하려면 0.006 sr(Wilkins 2005, EFA 2005)이나 Drasdo Q 환산(Wilkins 1980; Smedley 2010의 1.6° 하한)을 대안 지표로 제시할 수 있다.

**시청 조건(56 cm, 15인치 16:9).** 출처가 없는 팀원 값이지만 인용 경로가 있다. WCAG Note 1의 "15-17 inch screen at 22-26 inches"에서 22 in이 약 56 cm이고, JV24 Table 2는 "15-in(4:3) 56 cm [WCAG]"를 행으로 싣는다. 우리 판정기는 16:9를 쓰므로 "WCAG 기준 거리에 16:9 화면을 적용한 구현 선택"으로 적는다. 민감도 분석은 JV24 Table 2의 다른 행으로 한다. 23인치 FHD를 71.1 cm에서 보는 trace24 기준 화면, 68.0 cm 19인치(팀원 desktop 값과 일치), 20인치 TV 7H, 6.1인치 폰 32 cm가 후보다. 쌍 수와 화면 % 정의는 시거리에 따라 cpd가 바뀐다는 점(표준 조사의 계산 예: 같은 6쌍이 TV 7H에서 약 0.7 cpd, 56 cm 모니터에서 약 0.26 cpd)도 논의한다.

**지속 시간(현재 1 s 창 중 ≥0.5 s).** Wilkins 2005와 EFA 2005를 "임상 경험값(잠복기 1–2 s, 공식 연구 없음)"으로 인용하고, ITU 패턴 조항에는 지속 조건이 없다는 점을 밝힌다. '1초 창 안 누적 0.5초'는 Harding의 "persist for more than 0.5 seconds"(연속)와 다른 LUMI 구현 선택이다. 민감도 분석은 연속 0.5 s 대 창 누적 0.5 s, 그리고 문턱 0.25/0.5/1.0/2.0 s로 한다.

**움직임·변화(현재 전이 비율 ≥10%, 흐름 면제 미구현).** 이 규칙은 외부 근거가 없는 LUMI 결정이다. Binnie 1985 표 II에 비추면 진동과 흐름을 구분하지 못한다. 권고는 세 단계다. 첫째, 논문에서 이를 "표준이 정량 기준을 주지 않으므로(JV24) 만든 작업값"이라고 명시한다. 둘째, 판정을 '정지 / 방향 반전(진동·위상 반전) / 한 방향 흐름' 세 범주로 나눈다. 패턴 위상이나 광류(optical flow)의 부호 변화로 반전 빈도를 재고, 그 기준값을 Binnie 1985 실험 조건(삼각파 진동 5–20 Hz, 진폭 반주기, 흐름 2.5–10 °/s)에서 가져와 "근거 문헌의 실험 조건"으로 표기한다. 셋째, ITU의 흐름 면제를 켠 판정과 끈 판정을 모두 보고한다. 흐름 면제에 대해서는 근거들이 서로 다르다. Wilkins 2005는 의심 사례 2건 때문에 흐름에도 12쌍 제한을 두었고, EFA 2005는 8쌍으로 묶었으며, Wilkins 2005는 "consider moving patterns in the same way as those that are stationary"라는 선택지도 열어 두었다. 따라서 현재처럼 면제하지 않는 것도 인용 가능한 보수적 선택이다. 민감도 분석은 `PATTERN_DYNAMIC_FRAC` 5/10/20%다. 합성 시험 영상으로 Binnie 1985의 세 조건(2 c/deg 15 Hz 진동, 정지, 7.5 °/s 흐름)을 재현해 판정기가 0.97/0.38/0.04의 순서를 따르는지 확인하는 **'Binnie 재현 시험'**을 핵심 검증으로 제안한다. 움직임 때문에 생기는 국소 점멸은 깜빡임 규칙으로 넘긴다는 Wilkins 2005의 문장("If the pattern is such as to produce a 'flash' by virtue of its movement … the restrictions on flashes shall apply")과 Harding FPA의 이중 분석은 `analyze_all()`에서 깜빡임 판정과 결합하는 현재 구조의 근거로 인용할 수 있다.

**Duty cycle, 방향, 패턴 종류, 색.** Duty cycle은 표준에 없고 1차 근거도 모노그래프 그림(2차)뿐이다. 판정에서 거르지 말고, 시험 세트에 20/50/80% 줄무늬를 넣어 판정이 duty와 무관하게 나오는지만 보고한다. 방향은 "any orientation"(ITU)과 집단 방향 효과 없음(Wilkins 1979)을 인용해 12방향 15° 스캔을 정당화한다. 15° 간격이 사이 각도의 cpd를 cos 오차만큼 늘린다는 점은 구현 한계로 적는다. 패턴 종류는 ITU Guideline 2의 "parallel or radial, curved or straight … polka dots"와 NHK의 "縞模様、渦巻き模様、同心円模様"을 인용한다. 나선과 방사형에는 정량 실험이 없으므로 '모양 분류' 없이 국소 공간 주파수와 면적으로 일반화한다는 근거로 Wilkins 1980의 동심 고리 부채꼴 실험(위험은 면적 함수)을 든다. 체커보드는 Wilkins 1979 Study 5와 Harding의 체커보드 무시 정책에 비추어 과검출 가능성으로 보고하고, 방향 집중도(2D 스펙트럼에서 한 방향 에너지 비율)를 보조 지표로 검토한다. 색은 휘도(Y) 채널만 쓴다는 점을 Wilkins 1979 Study 3로 정당화한다. 동시에 Harding & Fylan 1999와 Fisher 2026이 색 줄무늬 위험을 시사하므로, 등휘도 색 줄무늬를 미탐지 범위로 명시한다.

**시험 세트.** 공개된 줄무늬 검증 세트가 없으므로 Jordan 2025의 '임계 바로 위/아래' 설계를 빌려 경계값 시험을 만든다. 축은 쌍 수 4/5/6/7/8/9, cpd 0.9/1.0/4/8/8.5/10/10.5, 면적 24/26%와 39/41%, 명암 ΔL 19/21과 어두운 쪽 159/161 cd/m², 정지/흐름/진동(5, 10, 15, 20 Hz), 그리고 줄무늬·체커보드·동심원·나선·물방울 행·방사형이다.

| 파라미터 | 권고 주값 | 인용 | 구현 선택으로 밝힐 것 | 민감도 분석 |
|---|---|---|---|---|
| cpd | 1–8 (또는 1–10을 구현 선택으로) | Fisher 2026; Wilkins 1979 | 8–10 확장, 하한 1, 저해상도 한계 | 1–4 / 1–8 / 1–10 / 0.5–10, Wilkins 2005 가중 |
| 쌍 수 | >5 (표준), ≥5 (보수 변형) | ITC 2001, BT.1702-3 Att.1; Rodrigues 2015 | ≥5, 극값//2 계산 | ≥5 / ≥6 / 변화5·정지8·흐름12 |
| 명암 | ΔL ≥20 & 어두운 <160 | BT.1702-3 Guideline 1; Wilkins 1979 Study 8 | 히스테리시스 20, 1/17 미구현 | 히스테리시스 5/10/20, Michelson 0.2/0.3, 밝은 줄 >50 |
| 평균 휘도 | 조건 없음 | Wilkins 1980; Wilkins 2005 | 피크화이트 200 cd/m², 감마 2.2 | 밝은 줄 >50 cd/m² 켜기/끄기 |
| 면적 | 정지 >40%, 변화 >25% | BT.1702-3 Att.1; Wilkins 2005 비판 | 줄무늬 픽셀 면적 정의 | 전부 25%, 외접사각형·밀도 정의 |
| 시청 조건 | 56 cm, 15인치 16:9 | WCAG Note 1; JV24 Table 2 | 16:9 적용 | 71.1 cm 23인치, 68 cm, TV 7H, 폰 32 cm |
| 지속 | 0.5 s | Wilkins 2005; EFA 2005 (경험값) | 1 s 창 누적 | 연속 대 누적, 0.25/0.5/1/2 s |
| 움직임 | 정지/반전/흐름 3범주 | Binnie 1985; JV24(기준 부재) | 10% 전이 비율, 흐름 면제 여부 | 5/10/20%, 면제 켜기/끄기, Binnie 재현 시험 |
| 방향 | 12방향 15° | BT.1702-3 "any orientation"; Wilkins 1979 Study 2 | 15° 간격 | 6/12/24방향 |
| 종류·색·duty | 휘도 채널, 모양 무관 | ITU Guideline 2; NHK; Wilkins 1979·1980 | 체커보드 과검출, 색 미탐지 | 체커보드·나선·동심원·duty 20/50/80% |

## 아직 확인되지 않은 것과 확보 경로

| 미확인 항목 | 왜 중요한가 | 확보 경로 |
|---|---|---|
| 1–10 cpd의 원출처 | 판정기 핵심 대역 | 팀원(rockrovks)에게 직접 문의, 없으면 1–8로 교체 |
| 팀원 코드 "3694790.pdf"의 정체 | 시야각 식 인용 | ACM DOI 10.1145/3694790(JV24 TACCESS로 추정)인지 팀원에게 확인 |
| ISO 9241-391:2016 5.3.1 본문 | cpd·운동 정량을 가진 유일한 후보 표준 | 학교 도서관 표준 구독(KSA/ISO), ISO/CD 94287 개정 상태 확인 |
| NHK·민방련 2020 개정판 | 패턴 문구 변경 여부 | j-ba.or.jp 아카이브(Wayback), NHK 방송 가이드라인 페이지 |
| ITC GN 1994·1999 원문 | 5쌍·40%의 최초 문구 | UK Government Web Archive의 ITC 사이트 |
| BT.1702-3 승인 월(11/2023 vs 202302) | 인용 연도 표기 | ITU-R 공식 레코드 페이지 재확인 |
| Wilkins 1979 그림 1B 집단값, Wilkins 1995 그림 2.9·2.10 | 곡선 수치(현재 2차) | Visual Stress 스캔본(visualstress.info) 직접 판독 |
| Wilkins 2005 출판본과 저자 원고의 차이, 패널 6 대비 식 | 모델 인용 정확성 | Wiley 기관 접속 |
| Fisher 2026 "1–8 c/deg, 50%"의 계산 근거 | 대체 대역의 근거 강도 | 저자(Wilkins) 문의 또는 보충자료 |
| "변화 패턴은 정지의 약 2배" 계수 출처 | 5/8쌍 유도의 핵심 | Stefánsson 1977 전문, Visual Stress 대조 |
| Funatsuka 2001 패턴 종류·반전 주파수별 유발률 | 패턴 종류 비교의 유일한 표 | Wiley 기관 접속 |
| Harding & Fylan 1999 등휘도 자극 정의 | 색 항 처리 | Wiley 기관 접속 |
| 8 c/deg 초과 위험, 나선·방사형 정량, 노출 시간 함수, 움직임 정량 정의 | 판정기의 근거 공백 | 문헌상 **존재하지 않음**으로 보고(추가 검색은 Pattern Inducer, CBM 2016 저자 문의) |
| Harding FPA 쌍 계산법·regularity limit·타일 크기 | 상용 도구와의 비교 | Cambridge Research Systems 문의 |
| IRIS 수치(MinStripes 6, 0.25, matchShapes 0.7)의 근거 | 오픈소스 비교 | GitHub 이슈로 질의 |
| Binnie 2002 깜빡임 모델 세부 | 패턴·깜빡임 공통 근거 | visualstress 스캔본 판독(현재 2차(아카이브)) |

## Conclusion

이번 조사로 바뀐 것은 판정기 수치의 성격 인식이다. 우리는 '1~10 cpd, 5쌍, 정지 40%, 0.5초'를 표준과 문헌에서 가져온 값으로 다뤄 왔다. 실제로는 이 중 어떤 것도 환자 실험에서 직접 측정되지 않았다. 표준이 정한 것은 40%와 >5쌍뿐이고, 둘 다 ITC의 ad hoc 값이다. 0.5초는 두 연구자의 경험이고, 1~10은 출처가 없다. 반대로 실험이 확실히 말하는 두 가지, 즉 **진동과 흐름이 정반대의 위험을 갖는다는 것**과 **직교 성분이 위험을 줄인다는 것**은 어떤 표준 수치에도, 어떤 공개 판정기에도 정량으로 구현되어 있지 않다. 따라서 LUMI가 보탤 수 있는 가장 큰 기여는 표준을 한 번 더 구현하는 데 있지 않다. 판정을 운동 범주(정지/반전/흐름)로 나누고, Binnie 1985 재현 시험으로 그 구분이 1차 자료의 순서를 따르는지 보이는 데 있다.

또 하나의 함의는 표준의 무게중심이 이동하고 있다는 점이다. 2026 EF 합의는 화면에 묶인 '쌍 수와 화면 %'를 버리고 시야 기준 '1–8 c/deg와 시야 25%'로 옮겨 갔다. cpd를 쓰는 우리 판정기 구조는 ITU 문구보다 이 최신 합의에 더 가깝다. 그러니 논문에서는 "BT.1702-3 Attachment 1 준수 판정"과 "Fisher 2026 합의 기반 판정"을 두 축으로 나란히 보고하자. 두 판정이 갈리는 영상이 곧 기존 표준의 공백을 보여 주는 사례가 된다.
