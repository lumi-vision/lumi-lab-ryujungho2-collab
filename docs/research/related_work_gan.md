# 관련 연구: PSE 검출 · 위험 자극 합성 · 규칙의 미분 가능 근사 · GAN 평균 붕괴 (related work)

작성 2026-11-10 (노트북). 논문 '관련 연구' 장과 리뷰 방어용.
표기: **원문**은 본문이나 PDF를 직접 읽은 것, **초록**은 초록이나 출판사 메타데이터(Crossref)만 읽은 것, **2차**는 검색 요약이나 2차 서술, **미확인**은 찾지 못하거나 열지 못한 것이다.
조사는 웹 조사 에이전트가 했고, 가장 중요한 항목(A-4 Barbu 2020 §3.1–3.2)은 PDF 본문에서 직접 다시 확인했다.

---

## 결론 먼저

1. **가장 가까운 선행연구는 Barbu, Banda & Katz (2020)이다.** 이 논문은 이미 우리 출발점이다([RESEARCH_NARRATIVE.md](../RESEARCH_NARRATIVE.md) §2).
   - **이번에 새로 확인한 것(주의):**
     - 기존 문서는 Banda 2018 학위논문 §4.2.8을 근거로 "Barbu의 Forward cGAN은 오라클 결과를 목표로 학습했다"고 정리했다. 그리고 미분 가능 판정기를 목표로 바꾼 E2를 우리 쪽 전환으로 서술했다.
     - 그런데 **논문 본문 §3.1.2는 "미분 가능한 연산으로 만든 광과민 탐지기를 GAN 손실에 선형 결합한다"고 쓴다.**
     - 따라서 "판정기를 손실로 쓰는 GAN" 자체를 LUMI의 새로움으로 주장하면 안 된다.
     - **Banda 2018 원문 확인 결과(아래 §F):** 학위논문의 Forward cGAN은 수작업 생성기 출력을 목표로 학습했다(기존 문서가 맞음). 미분 가능 판정기는 **Inverse(보정) 모델의 손실로만** 쓰였다. 논문 §3.1.2의 서술은 학위논문의 구현과 다르다.
   - 같은 점: 지식 기반 생성기가 깨끗한 영상에 위험 기준 **위와 아래** 자극을 넣는다. 미분 가능한 연산으로 만든 광과민 탐지기를 GAN의 손실(perceptual loss)로 쓴다.
   - 다른 점:
     - **목적:** Barbu는 위험 영상을 안전하게 바꾸는 완화(inverse) 변환을 학습한다. LUMI는 검출기를 학습·평가할 데이터를 만든다.
     - **판정 규칙:** Barbu의 탐지기는 "Harding과 비슷한 탐지기를 미분 가능한 연산으로 만든 CNN"이라고만 적혀 있고, 어떤 표준의 어떤 조항을 구현했는지 나오지 않는다. LUMI는 BT.1702-3을 조항별로 구현했다.
     - **생성 공간:** LUMI는 linear RGB에서 빛만 더하는 1채널 Δ로 생성 공간을 제한한다.
2. **제대로 된 공개 위험 섬광 영상 데이터셋은 사실상 없다.** FlashGuard (2025)도 "공개 저장소가 없다"고 명시한다. 공개 후보는 둘이다.
   - South et al. (2021)의 OSF GIF 폴더
   - Kaya et al. (2025)의 `PSE_triggering_dataset.rar`: 공개 URL이지만 라이선스가 명시되지 않았고, 약 18~22개 규모에, 라벨은 PEAT 판정으로 추정된다. 자세한 내용은 [kaya_dataset.md](kaya_dataset.md).
3. **동료심사를 거친 딥러닝 PSE 검출기와 공개 벤치마크는 찾지 못했다.** 검출 쪽은 규칙 기반 도구(Harding FPA, PEAT, IRIS, Carreira 2015)가 중심이다.
4. **GAN이 오라클의 평균으로 붕괴한 현상(LESSONS A1)은 새로운 발견이 아니다.** pix2pix와 Mathieu 2016이 보고한 평균화 및 noise 무시 현상의 사례로 설명하고, DSGAN·MSGAN과 비교 실험으로 뒷받침하는 편이 안전하다.
5. **soft judge의 방법론적 선례는 STL robustness를 미분 가능하게 만든 연구들이다**(STLCG, Smooth Operator, 규칙으로 유도하는 diffusion). "1초에 3회 초과"는 시간 논리 형태의 조건이다.

---

## A. PSE·광과민 위험 검출 (딥러닝과 규칙 기반 기준선)

| # | 서지 | 확인한 주소 | 요약 | LUMI와의 관계 | 등급 |
|---|---|---|---|---|---|
| A-1 | J. B. Jordan, G. C. Vanderheiden (2024). International Guidelines for Photosensitive Epilepsy: Gap Analysis and Recommendations. *ACM TACCESS* 17(3):1–35. doi:10.1145/3694790 | https://pmc.ncbi.nlm.nih.gov/articles/PMC11872230/ (ACM DL은 403) | 영국, 일본, ISO, ITU, W3C의 5개 지침을 비교한다. 공백으로 전이 지속시간(66 ms 제안), 동시성 창(20 ms), 고프레임레이트에서의 프레임 비교, 움직임 보상을 든다. "IRIS·Apple 도구는 어느 국제 표준도 엄밀히 따르지 않는다"고 쓴다. **팀원 코드의 "3694790.pdf"가 이 논문이다.** | 규칙 정의의 근거이고, 규칙 자체가 모호하므로 경계 근처 샘플이 필요하다는 근거 | 원문(PMC 저자 원고) |
| A-2 | ITU-R BT.1702-3 (11/2023) | https://www.itu.int/rec/R-REC-BT.1702 | 버전 이력: -0 2005, -1 2018, -2 2019, -3 2023 | 오라클 판정 기준 | 버전 정보 원문 |
| A-3 | Carreira, Rodrigues, Roque, Queluz (2015). Automatic detection of flashing video content. *QoMEX 2015*. doi:10.1109/QoMEX.2015.7148104 | Crossref | BT.1702 기반 자동 섬광 검출(규칙 기반) | 기준선 후보 | 서지 초록, 내용 2차 |
| **A-4** | **A. Barbu, D. Banda, B. Katz (2020). Deep video-to-video transformations for accessibility with an application to photosensitivity. *Pattern Recognition Letters* 137:99–107. doi:10.1016/j.patrec.2019.01.019** | https://groups.csail.mit.edu/infolab/publications/Barbu-Deep-video-to-video-transformations.pdf | §3.1.1: 지침을 담은 생성기가 RGB·밝기·HSV·파장 분포 채널에 섬광과 패턴을 넣는다. "we insert stimuli both above the threshold … and below that threshold." §3.1.2: "we construct this detector from differentiable operations … a CNN operating over videos which provides a perceptual loss for photosensitivity", 판별기 손실과 선형 결합한다. §3.2: 목표는 inverse pass(위험 → 안전 변환). 사람 평가에서 GAN 결과가 수작업 생성기보다 "subtler flashing"이었다. | **가장 가까운 선행연구.** 차이는 위 '결론 먼저' 1번 참고 | **원문(직접 재확인)** |
| A-5 | South, Saffo, Borkin (2021). Detecting and defending against seizure-inducing GIFs in social media. *CHI '21*. doi:10.1145/3411764.3445510 | https://vis.khoury.northeastern.edu/pubs/South2021DetectingDefendingSeizure | 인터뷰를 거쳐 브라우저 확장 프로토타입을 만들고, 시뮬레이션 GIF와 소셜미디어 GIF로 검증했다. | 웹 GIF 선별 기준선 | 초록 |
| A-6 | South, Yildirim, Pavel, Borkin (2024). Design considerations for photosensitivity warnings in visual media. *ASSETS '24*. doi:10.1145/3663548.3675643 | Crossref | 경고 UI 설계. 사용자는 일괄 차단보다 스스로 통제하기를 원한다. | 검출 결과를 어떻게 쓸지에 대한 맥락(주변 연구) | 서지 초록, 내용 2차 |
| A-7 | Kothari, Srivastava (2021). Flikcer. arXiv:2108.09491 | https://arxiv.org/abs/2108.09491 | Chrome 확장. 트리거 시점을 알리고 안전한 버전을 제공한다. 동료심사 없음. | 기준선·대조군 | 초록 |
| A-8 | Kothari, Srivastava. FlikcerAI (Society for Science 경진대회 초록, 연도 미표기) | https://abstracts.societyforscience.org/Home/PrintPdf?projectId=21661 | CNN-RNN, "합성 영상 24만 개 이상", 정확도 97% 이상이라고 주장한다. 동료심사 없음. | 합성 데이터로 딥러닝 검출기를 학습한 사례(근거 약함) | 초록 |
| A-9 | Pendyala (2025). FlashGuard. arXiv:2507.19692 | https://arxiv.org/html/2507.19692v1 | **"공개된 발작 유발 영상 저장소가 없다"고 명시.** 합성 영상 1000개를 PEAT로 라벨링하고, CIELAB 변화량으로 logistic regression을 학습했다(정확도 80%, AUC 0.90). | '규칙 도구 라벨 + 합성 데이터'가 오라클과 같은 방식. LUMI의 동기를 뒷받침 | 원문 |
| A-10 | Kaya, Kilic, Genc, Kural (2025). Detection and correction of the flashing parts that trigger PSE in videos. *SIVP* 19(12):967. doi:10.1007/s11760-025-04608-4 | https://research.hacettepe.edu.tr/en/publications/detection-and-correction-of-the-flashing-parts-that-trigger-photo/ | Ofcom·ITU·W3C 기준으로 검출하고 보정한다. | 최신 규칙 기반 기준선, 완화 쪽 대조군 | 초록 |
| A-11 | Alzubaidi, Otoom, Al-Tamimi (2016). Parallel scheme for real-time detection of photosensitive seizures. *Comput. Biol. Med.* 70:139–147 | Crossref | 시공간 패턴을 실시간으로 검출하고 마스킹한다. | 기준선 | 서지 초록, 내용 2차 |
| A-12 | Electronic Arts, IRIS (BSD-3) | https://github.com/electronicarts/IRIS | 휘도·적색 섬광과 패턴을 검출한다. "인증·준수 보증 아님"이라고 명시한다. 패턴 구현은 [pattern_notes/detection_tools.md](pattern_notes/detection_tools.md) 참고. | 오픈소스 기준선, 오라클과 교차 검증 | 원문 |
| A-13 | Trace Center, PEAT User Guide | https://trace.umd.edu/?p=719 | WCAG 2.0 기반. 엔진은 Cambridge Research Systems와 Harding이 함께 만들었다. 상용 콘텐츠에는 Harding FPA를 쓰라고 안내한다. | 기준선(FlashGuard의 정답 라벨) | 원문 |
| A-14 | Apple (2023). Video Flashing Reduction: Background and Algorithm v1.0 | https://developer.apple.com/accessibility/downloads/Video-Flashing-Reduction-12.pdf | 플리커 위험 지표와 감쇠 알고리즘 | 산업 기준선(완화) | 원문. GitHub 저장소는 2차 |
| A-15 | Fisher, Harding, Erba, Barkley, Wilkins (2005). Photic- and pattern-induced seizures: EFA Working Group review. *Epilepsia* 46(9):1426–1441 | Crossref | 임상 근거 | 서론용 | 서지만 |

## B. 데이터셋

- **B-1. South et al. (2021) OSF** (https://osf.io/5a3dy/)
  - 파일 목록: `social-media-study-gifs/`, `social-media-study.csv`, `interview-questions.pdf` (OSF API로 확인).
  - 라이선스와 라벨 형식은 **미확인**이다. 실제 위험 GIF로 평가할 때 유일한 공개 후보다. 위험 자극이므로 재생하지 않고 코드 지표로만 다룬다.
- **B-2.** 합성 데이터를 쓴 선례는 Barbu(A-4), FlikcerAI(A-8), FlashGuard(A-9)다. 셋 다 데이터를 공개하지 않았다(FlashGuard는 '요청 시 제공').

## C. 위험 시나리오 생성 · 규칙의 미분 가능 근사

| # | 서지 | 확인한 주소 | LUMI와의 관계 | 등급 |
|---|---|---|---|---|
| C-1 | Ding, Chen, Xu, Zhao (2020). Learning to Collide. *IROS 2020* | https://arxiv.org/abs/2003.01197 | 생성 모델로 대상 시스템을 위험하게 만드는 시나리오를 찾는다. | 초록 |
| C-2 | Ding et al. (2023). A Survey on Safety-Critical Driving Scenario Generation. *IEEE T-ITS* | https://arxiv.org/abs/2202.02215 | 생성 방법을 데이터 기반, 적대적, 지식 기반으로 분류한다. LUMI는 지식 기반(오라클)과 적대적(GAN)의 결합으로 위치시킬 수 있다. | 초록 |
| C-3 | Rempe et al. (2022). STRIVE. *CVPR 2022* | https://arxiv.org/abs/2112.05077 | 학습된 prior 위에서 현실적이면서 위험한 시나리오를 만든다. | 초록 |
| C-4 | Zhong et al. (2022). Guided Conditional Diffusion for Controllable Traffic Simulation. arXiv:2210.17366 | https://arxiv.org/abs/2210.17366 | **미분 가능한 STL 규칙으로 diffusion 생성을 유도한다. soft judge와 가장 가까운 방식.** 학회(ICRA 2023) 표기는 미확인 | 초록 |
| C-5 | Leung, Aréchiga, Pavone (2020). Backpropagation through STL Specifications (STLCG). *WAFR 2020* | https://arxiv.org/abs/2008.00097 | STL robustness를 계산 그래프로 바꿔 역전파한다. soft judge의 방법론적 선례 | 초록 |
| C-6 | Pant, Abbas, Mangharam (2017). Smooth Operator. *IEEE CCTA 2017*. doi:10.1109/CCTA.2017.8062628 | (UPenn 저장소 403) | max/min을 매끄럽게 근사하고 오차 상한을 준다. soft judge 근사 오차 분석에 쓸 수 있다. | 2차 |
| C-7 | Xu et al. (2018). A Semantic Loss Function for Deep Learning with Symbolic Knowledge. *ICML 2018* | https://arxiv.org/abs/1711.11157 | 논리 제약을 손실로 바꾼다. | 초록 |
| C-8 | Fischer et al. (2019). DL2. *ICML 2019* | https://proceedings.mlr.press/v97/fischer19a.html | 논리를 손실로 바꾸고, 제약을 만족하는 입력도 찾는다. | 초록 |
| C-9 | Patel, Hodaň, Matas (2020). Learning Surrogates via Deep Embedding. *ECCV 2020* | https://arxiv.org/abs/2007.00799 | 미분할 수 없는 지표를 학습형 surrogate로 대체한다. 손으로 설계한 soft judge와 비교할 대안 | 초록 |
| C-10 | Corso et al. (2021). A Survey of Algorithms for Black-Box Safety Validation of CPS. *JAIR* 72 | https://arxiv.org/abs/2005.02979 | falsification이라는 용어 틀. LUMI 생성기를 검출기 falsification 도구로 설명할 수 있다. | 초록 |
| C-11 | Yaghoubi, Fainekos (2019). Gray-box Adversarial Testing. *HSCC 2019* | https://arxiv.org/abs/1812.11958 | 요구사항을 위반하는 입력을 기울기로 찾는다. | 초록 |
| C-12 | Zhang et al. (2018). DeepRoad. arXiv:1802.02295 | https://arxiv.org/abs/1802.02295 | GAN으로 테스트 장면을 생성한다. | 초록 |
| C-13 | Ruiz, Schulter, Chandraker (2019). Learning To Simulate. *ICLR 2019* | https://arxiv.org/abs/1810.02513 | 시뮬레이터 파라미터를 다운스트림 성능 기준으로 최적화한다. 오라클 샘플링 분포를 최적화하는 대안 | 초록 |

## D. 평균 붕괴와 해결책

| # | 서지 | 확인한 주소 | LUMI와의 관계 | 등급 |
|---|---|---|---|---|
| D-1 | Isola et al. (2017). pix2pix. *CVPR 2017* | https://arxiv.org/abs/1611.07004 | 본문: "Euclidean distance is minimized by averaging all plausible outputs", "the generator simply learned to ignore the noise". **LESSONS A1(오라클 평균으로 붕괴)을 설명하는 핵심 근거** | 원문 |
| D-2 | Mathieu, Couprie, LeCun (2016). Deep multi-scale video prediction beyond MSE. *ICLR 2016* | https://arxiv.org/abs/1511.05440 | 영상에서 MSE를 쓰면 흐릿한 평균 예측이 나온다. | 초록 |
| D-3 | Yang et al. (2019). Diversity-Sensitive Conditional GANs. *ICLR 2019* | https://arxiv.org/abs/1901.09024 | λ_div(다양성 손실)의 직접적인 선례 | 초록 |
| D-4 | Mao et al. (2019). Mode Seeking GANs (MSGAN). *CVPR 2019* | https://arxiv.org/abs/1903.05628 | 잠재코드 거리 대비 출력 거리 비율을 정규화한다. 비교 후보 | 초록 |

## E. 영상을 안전하게 만드는 연구 (대조군)

- Barbu(A-4), FlashGuard(A-9), Kaya(A-10), Apple(A-14), Flikcer(A-7)가 이 범주에 해당한다.
- Nomura (1999). A comfortable brain-interface to video displays. *Neural Networks* 12(2):347–354. 10 Hz 이상 성분을 줄이는 적응형 필터다. 서지는 Crossref로 확인했고, 내용은 2차다.
- Nomura et al. (2000). *Psychiatry Clin. Neurosci.* 54:685–690. 환자 11명 대상 필터 효과를 보고한다. 2차.

## F. Banda 2018 학위논문 (Barbu 2020의 구현) — 원문 확인

- 서지: D. Banda (2018), MIT M.Eng. thesis, DSpace@MIT 1721.1/121622. 79쪽 PDF를 받아 텍스트로 추출했다.
- 아래 인용 중 ★ 표시는 노트북에서 추출본으로 직접 다시 확인한 것이다.

| 질문 | 원문 | 위치 |
|---|---|---|
| Forward GAN 학습 목표 | ★ "The input to the generator is the video V_i, and the parameters used to generate its synthesized counterpart **(which is the target of the model)** are fed into Z." 입력에 노이즈를 직접 넣지 않고, Z에서 Z̃를 샘플링하며 KL 정규화를 한다. **손실 구성(L1, 적대적 손실, 판정기 손실)은 적혀 있지 않다.** | §4.2.8 p.45 |
| "very deterministic" | ★ "We observed that for video with low intensity flashing the network would amplify the flashing intensity and frequency. Like in [77], our trained model was **very deterministic**." 원인 설명은 없고 pix2pix([77])와 비교만 한다. | §4.2.8 p.45 |
| 미분 가능 판정기 | ★ "we engineered parts of the hand detector to use **differentiable tensor ops** … one to detect luminance flashes and another to detect saturated reds." 휘도는 0.299/0.587/0.114(BT.601 루마, W3C 계수 아님)를 쓰고, ReLU(\|∇x_l\| − 20)를 30프레임 창에서 센다. **25% 면적 조건은 없다.** "20"은 화소값 차이이고 cd/m²가 아니다. | §4.2.7 p.42–44 |
| 판정기가 쓰인 곳 | **Inverse 모델의 손실에만** 쓰였다: L_total = α·L_mse + β·L_d + γ·L_vgg + ζ·L_lum (식 4.7). ★ "including this loss led to networks quickly learning to **blur out flashing regions**." 표 6.1에는 ζ > 0인 모델이 없다. | p.44, p.57 |
| 판정기만으로 학습할 때의 문제 | ★ "the generator would simply learn to **fool the oracle with low luminance images or garbage**." | p.42 |
| 학습형 탐지기 | CNN-LSTM, 합성 데이터에 BCE로 학습, 라벨은 "t = 1 if the input video contains a flash". Inverse 모델에서 사전학습된 고정 판별기로 쓰였다. | §4.1.2–4.1.3 p.31, p.42 |
| 수작업 생성기 | 지속 1–3초. 저주파 1–3 Hz(위험 아님 샘플 역할)와 고주파 15–25 Hz. 유형은 적·청, 흑·백, HSV, 파장 이동 4가지. **면적 파라미터는 없고 프레임 전체에 섞는다.** | §3.2.1–3.2.2 p.25–27 |
| Forward GAN 평가 | 사람 평가만 있다(수정본 선택 비율: 수작업 91%, GAN 83%). 판정기로 한 정량 평가는 없다. | 그림 6-2 p.60 |
| 내부 불일치 | 그림 4-8 캡션과 본문의 Z 정의가 다르다. 3장(20,000 클립, 200×200 이상)과 5장(수백만 링크, 64×64)의 데이터 규모가 다르다. | p.50, p.25, p.53 |

**LUMI에 대한 함의**
1. `RESEARCH_NARRATIVE.md` §2·§3의 서술("Forward cGAN은 오라클 결과를 목표로 학습, 매우 결정적")은 **학위논문 기준으로 맞다.**
   - 다만 "오라클"이라는 말을 학위논문은 두 가지 뜻으로 쓴다. 서론에서는 수작업 생성기, §4.2.7에서는 미분 가능 판정기다. 그러니 '수작업 생성기 출력을 목표로'라고 써야 오해가 없다.
2. **"판정기를 생성기 손실로 쓴다"는 아이디어는 Barbu 2020 논문 §3.1.2에 이미 적혀 있다.** 하지만 학위논문에서는 Forward GAN에 구현되지 않았다. Inverse에서는 흐리게 만드는 지름길이 생겨서 거의 쓰이지 않았다.
   - 그러므로 LUMI E2의 의미는 이렇게 쓸 수 있다. "구상으로만 제시된 방식을 **BT.1702-3 조항별 판정(25% 면적 포함, cd/m² 단위)**으로 실제 구현하고, 붕괴와 지름길(흐림, 어둡게 만들기)을 측정했다."
   - "처음 제안했다"고 쓰면 안 된다.
3. Banda의 "판정기만 쓰면 어둡거나 엉망인 영상으로 속인다"는 관찰은 우리 LESSONS A6(규칙 빈틈 공략)과 같은 현상이다. 그대로 인용할 수 있다.

---

## 찾지 못했거나 확인하지 못한 것

- 동료심사를 거친 딥러닝 PSE 영상 검출기와 공개 벤치마크
- 포켓몬 사례 장면을 자동 검출 실험에 쓴 학술 연구(위키와 블로그 수준만 있음)
- South et al. 외의 공개 위험 섬광·패턴 데이터셋
- GAN이 규칙 기반 오라클을 모방하다 평균으로 붕괴하는 현상을 직접 다룬 논문(일반 문헌 D-1~D-4로만 설명 가능)
- Williams et al. (2024?) "'Not Only Annoying but Dangerous'": venue와 DOI 미확인(2차)
- 열지 못한 원문: Harding FPA 제품 페이지, Apple GitHub 저장소, Alzubaidi 초록 원문, Nomura 2000, Pant 2017

## 논문에 쓸 때의 주장 범위 (조심스럽게)

- **주장할 것:** "BT.1702-3 조항별 오라클과 미분 가능한 soft judge로, linear RGB에서 빛을 더하는 1채널 생성 공간 안에 검출기 시험용 경계 샘플을 만든다." 즉 조합과 응용 대상(검출기 학습·평가 데이터)의 새로움이다.
- **주장하지 말 것:**
  - "규칙 기반 생성기 + 미분 가능 판정기 GAN"이 처음이라는 주장. Barbu 2020이 이미 했다.
  - "평균 붕괴의 발견". pix2pix와 Mathieu 2016이 이미 보고했다.
- **다음에 할 것:**
  - ~~Banda 2018 확인~~ → 완료(§F). RESEARCH_NARRATIVE §2에 "판정기 손실은 Barbu 논문에 구상으로 있으나 학위논문 Forward에는 미구현"이라는 한 줄을 추가할지 결정한다.
  - Barbu §3.1.2의 탐지기 구조(섬광 판정을 어떤 연산으로 미분 가능하게 했는지)와 LUMI soft judge를 표로 비교한다. 현재 PDF 본문에 세부가 부족해서 보충자료를 확인해야 한다.
  - DSGAN·MSGAN과 비교 실험을 설계한다.
