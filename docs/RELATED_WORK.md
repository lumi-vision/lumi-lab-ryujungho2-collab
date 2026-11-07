# 관련 연구 — GAN·합성 데이터를 쓴 사례 (조사 2026-11-07)

목적: LUMI의 G 설계 전에, Barbu 외에 GAN 또는 합성 데이터로 "열화를 만들고 → 복원기를 학습"한 사례가 무엇을 했고 무엇이 통했는지 정리한다. 각 항목의 근거 수준을 표시한다. **[읽음]** 원문 확인, **[초록]** 초록·목록·검색 요약만 확인, **[2차]** 제3자 요약만 확인.

## 1. 광과민성(PSE) 분야

| 연구 | 생성 방식 | 라벨 | 근거 | 메모 |
|---|---|---|---|---|
| Barbu·Banda·Katz, *Pattern Recognition Letters* 2020 | 수작업 Oracle + cGAN | 사람 평가 | [읽음] | [BARBU_GAN_DEEP_DIVE.md](BARBU_GAN_DEEP_DIVE.md) |
| Banda, MIT M.Eng. 학위논문 2018 | 같은 연구의 구현 상세 | PEAT 대조, 사람 평가 | [읽음] | 같은 문서 6A절. Forward cGAN은 Oracle 결과를 목표로 학습, "매우 결정적" |
| Pendyala, FlashGuard, arXiv 2507.19692 (2025) | 무작위 파라미터 합성 영상(학습 1,000·시험 200). GAN 아님 | PEAT | [2차] | 라벨링·필터 조정·평가를 모두 PEAT로 해 순환적이라는 비판 |
| South·Saffo·Borkin, CHI 2021 | 시뮬레이션 GIF + 실제 SNS GIF | 미확인 | [초록] | 브라우저 확장. 자료 https://osf.io/5a3dy/ |
| SIViP 2025, 10.1007/s11760-025-04608-4 | 영상처리 기반 검출·보정(학습 아님) | 미확인 | [초록] | 데이터·코드 공개라고 하나 원문은 로그인 필요 |
| 국제 가이드라인 격차 분석, PMC11872230 | — | — | [초록] | 최근 보정 기법은 광과민성 당사자로 정식 검증된 적이 없다고 지적 |

Barbu 이후 **GAN으로 광과민성 자극을 생성한 후속 연구는 찾지 못했다.** 인용 목록에는 경고 디자인(South 2024), 검사 도구 적합성(Jordan 2025) 같은 비생성 연구가 주로 보인다.

## 2. 같은 구조: "열화를 먼저 학습 → 복원기 학습"

| 연구 | 무엇을 생성하나 | G가 배우는 신호 | 근거 |
|---|---|---|---|
| Bulat 외, ECCV 2018 (초해상도) | 고해상도→저해상도 열화 | **실제 저화질 이미지와 짝 없는(unpaired) GAN.** 실제 열화 분포가 있음 | [초록] |
| Chen 외 GCBD, CVPR 2018 (잡음 제거) | 실제 잡음 패치 | 실제 잡음 이미지의 분포 | [초록] |
| Tran 외, ACCV 2020 | raw 영역 잡음 | 실제 잡음 | [초록] |
| Jang 외 C2N, ICCV 2021 | 실제 잡음 | 짝 없는 실제 잡음 이미지 | [초록] |

**공통점:** GAN이 유용했던 사례에는 **흉내 낼 실제 열화 데이터**가 있다. GAN은 사람이 규칙으로 못 만드는 실제 분포를 배운다. Barbu와 LUMI는 실제 위험 영상을 학습 목표로 쓰지 않고(구하기 어렵고 위험함) **Oracle 결과를 목표**로 하므로 이 이점이 없다.

## 3. 반대 방향: 무작위화한 수작업 열화가 주류가 된 사례

| 연구 | 내용 | 근거 |
|---|---|---|
| Zhang 외 BSRGAN, ICCV 2021 | 흐림·축소·잡음 단계를 무작위 순서로 섞는 수작업 열화 모델 | [초록] |
| Wang 외 Real-ESRGAN, ICCVW 2021 | 고전 열화를 2번 반복하는 "고차 열화" + sinc 필터. 제목이 "**순수 합성 데이터**만으로 학습" | [초록] |

초해상도에서는 학습된 GAN 열화(Bulat 2018) 이후, 충분히 무작위화한 수작업 열화로 학습하는 쪽이 널리 쓰였다. 다만 Real-ESRGAN 저자들도 이 방식이 실제 열화 공간 전체를 덮지 못한다고 적었다. **LUMI의 H0(Oracle만으로 충분한가)에 대한 강한 선행 사례다.**

## 4. 영상 깜빡임 제거(deflicker)

- 합성 깜빡임(밝기 변조)으로 학습하는 것이 흔하다. [초록]
- BurstDeflicker (NeurIPS 2025): Retinex 기반 합성 + 실측 데이터. 합성만으로 학습하면 **움직임을 깜빡임으로 오인**하는 문제가 있어 실측을 섞었다. [초록]

## 5. LUMI G 설계에 주는 시사점

1. **Oracle을 목표로 한 cGAN은 Oracle보다 다양해지기 어렵다.** Barbu 구현(학위논문)도 "매우 결정적"이었고, LUMI E1도 같은 결과다. pix2pix 계열에서 잘 알려진 현상이다.
2. GAN이 의미가 있으려면 **Oracle에 없는 정보 신호**가 필요하다. 후보는 다음과 같다.
   - (a) 실제 열화 데이터. 광과민성에서는 확보가 어렵고 위험하다.
   - (b) 미분 가능한 위험 대리 R. Barbu 학위논문의 Algorithm 2·3처럼 규칙을 미분 가능 연산으로 옮긴다. 저자는 R만으로는 "어둡게 만들거나 쓰레기로 속인다"고 경고했다.
3. **H0를 먼저 진지하게 시험할 가치가 있다.** Oracle을 더 무작위화하는 것(Real-ESRGAN식)이 GAN보다 싸고 통제 가능하다. GAN은 H0가 실패하는 지점에서만 정당화된다.
4. 다양성 자체가 목표라면 cVAE-GAN/BicycleGAN류처럼 전역 잠재 변수와 잠재 복원 손실을 쓰는 방법이 표준이다. Barbu도 64차원 잠재를 썼지만 잠재 복원 손실 없이 KL만 걸어 결정적이 됐다. [이 항목은 일반 문헌 지식에 근거하며, 이번 조사에서 해당 논문 원문은 다시 확인하지 않았다]

## 출처

- Barbu 외 2020: https://groups.csail.mit.edu/infolab/publications/Barbu-Deep-video-to-video-transformations.pdf
- Banda 2018: https://dspace.mit.edu/handle/1721.1/121622
- FlashGuard (3자 요약): https://pith.science/paper/2507.19692
- South 외 CHI 2021: https://vis.khoury.northeastern.edu/pubs/South2021DetectingDefendingSeizure
- SIViP 2025: https://link.springer.com/article/10.1007/s11760-025-04608-4
- 가이드라인 격차 분석: https://pmc.ncbi.nlm.nih.gov/articles/PMC11872230/
- Bulat 외 2018: https://openaccess.thecvf.com/content_ECCV_2018/html/Adrian_Bulat_To_learn_image_ECCV_2018_paper.html
- Chen 외 2018: https://openaccess.thecvf.com/content_cvpr_2018/html/Chen_Image_Blind_Denoising_CVPR_2018_paper.html
- Tran 외 2020: https://openaccess.thecvf.com/content/ACCV2020/papers/Tran_GAN-based_Noise_Model_for_Denoising_Real_Images_ACCV_2020_paper.pdf
- Jang 외 2021: https://openaccess.thecvf.com/content/ICCV2021/papers/Jang_C2N_Practical_Generative_Noise_Modeling_for_Real-World_Denoising_ICCV_2021_paper.pdf
- BSRGAN: https://sotaverified.org/papers/designing-a-practical-degradation-model-for
- Real-ESRGAN: https://arxiv.org/pdf/2107.10833
- BurstDeflicker: https://arxiv.org/html/2510.09996v1
