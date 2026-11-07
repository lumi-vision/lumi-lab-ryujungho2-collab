# Barbu(2020)의 GAN과 LUMI 조건부 GAN: 원문 근거·재현 수준·채택 이유

## 1. 연구 질문

**질문:** Barbu가 사용한 GAN을 LUMI에 그대로 가져올 수 있는가?  
**답:** 그대로 가져왔다고 보기 어렵다. Barbu가 제시한 **Forward/Inverse 데이터 생성 논리**는 채택할 수 있지만, 현재 코드는 LUMI가 독립적으로 설계한 조건부 3D PatchGAN 프로토타입이다. 동일한 G/D의 레이어 구성, 위험 평가 CNN, 학습 하이퍼파라미터를 재현한 결과가 아니다.

## 2. Barbu 논문의 실제 기법 [1, §3.1–3.2]

논문은 Forward에서 영상의 변형 전후 쌍을 확보하고, Inverse에서 변형된 영상을 원래의 영상에 가깝게 복원하는 비디오 변환을 학습한다.

- **수작업 Oracle:** RGB/휘도/HSV 공간과 반복 시각 패턴에 변형을 가한다. 임계값 **위·아래 사례를 모두** 생성하여 수정할 필요 없는 영상도 보존하도록 학습시킨다 [1, §3.1.1].
- **GAN:** G와 D는 비디오 볼륨을 받는 합성곱 신경망이다. 비디오에 대한 *미분 가능한* 광과민성 관련 CNN 평가기 `R`을 만들고, GAN의 판별 손실과 `R`의 평가 손실을 결합한다 [1, §3.1.2].
- **보정기(Inverse):** 원본·변형본·변형 관련 지표의 학습 자료로 양방향 LSTM, Stacked LSTM U-Net, 시공간 Autoencoder 등을 비교한다 [1, §3.2].
- **실험:** 논문은 64×64 영상 클립을 사용했으며 방법 설명에는 101프레임 단위가 명시된다. 실험 데이터 구성에서는 100프레임의 크롭 수를 설명하므로 전처리/모델별 길이 해석에 주의해야 한다 [1, §3–4].

**중요:** 'Barbu = pix2pix' 또는 'Barbu = PatchGAN'이라는 표현은 원문 근거가 없다. 특정 코드 구현과 완전히 동일하다는 주장도 해서는 안 된다.

## 3. 현재 LUMI 구현의 수학적 구조

기준 클립 `Y ∈ [0,1]^{B×3×T×H×W}`, 허용 영역 `M`, 시간 활성화 `A`, 변형 유형 원-핫 `c`, 잡음 `z`에 대해:

$$
\Delta = \delta_{max}\tanh\bigl(G_{CNN}([Y,M,A,c,z])\bigr),
\quad X_G = \operatorname{clip}(Y+M\odot A\odot \Delta,0,1).
$$

- `G`: 3D convolution + GroupNorm + LeakyReLU 블록과 RGB 잔차 헤드.
- `D`: `(Y, X, M, A, c)`의 조건부 3D 영상 패치에 대한 판별 점수를 출력한다.
- `G`와 `D`는 GAN 학습 때 함께 사용한다. **학습을 마친 뒤 합성 추론 시에는 G만 필요하다.**
- `mask`는 위험 정답이 아니라 **허용된 변형 공간**, `activation`은 시간적 조건이다.

현재 구현된 손실은 Hinge GAN과 국소 L1 재구성이다.

$$
L_D=\mathbb E[\max(0,1-D(Y,X_O,c))]+\mathbb E[\max(0,1+D(Y,X_G,c))]
$$

$$
L_G=-\mathbb E[D(Y,X_G,c)] + \lambda_{rec}\;\frac{\sum M\lvert X_G-X_O\rvert}{3\sum M+\epsilon}.
$$

여기서 `X_O`는 Oracle 변형본이다. 식에서는 D의 `M`, `A` 조건을 간결하게 `c`에 포함해 표기했다. 코드는 `lambda_reconstruction=20.0`을 기본값으로 사용하지만 **검증된 최적값이 아니다.**

### 왜 이 손실을 사용하나?

1. **Hinge 대립 손실:** Oracle 변형 데이터와 생성된 변형의 시공간 분포를 D를 통해 구분하도록 학습한다.
2. **ROI 내 L1:** 생성 결과가 Oracle이 정의한 기본 변형 형태에서 무한정 벗어나지 않도록 제한한다.
3. **`M×A×Δ` 잔차:** 허용 영역 밖의 픽셀과 활성화가 0인 프레임을 가능한 보존한다.
4. **독립 확인 필요:** 이 손실들은 광과민성 표준의 임계값 준수를 보장하지 않으며, 비슷한 영상의 생성과 실제 일반화도 별도 문제다.

## 4. 원문과 LUMI 구현의 차이

| 분석 대상 | Barbu 논문에 기술 | 현 LUMI 코드 | 결과적 해석 |
|---|---|---|---|
| Forward의 Oracle | 있음 | 있음 | 방법론 차용 |
| 비디오 GAN G/D | 있음 | 조건부 3D CNN G/D | **별도 구현** |
| 미분 가능한 위험 평가 CNN `R` | 있음 | 없음 | **원문 핵심 일부 미구현** |
| 판별 손실 + 위험 퍼셉추얼 손실 | 있음 | Hinge + ROI L1 | **학습 목표 다름** |
| Oracle 결과를 목표로 학습 | **예** — 저널에는 없고 구현자 학위논문 §4.2.8에 있음 ("수작업 합성 예제 위에서 cGAN 학습") | ROI L1, λ=20 | 둘 다 Oracle 복사 쪽으로 수렴. 학위논문은 "매우 결정적", LUMI E1은 세기 표준편차 ≤ 0.004 |
| G의 잡음·조건 입력 | 학위논문: Oracle 파라미터(유형·지속시간·주파수) + 64차원 잠재(KL 정규화) | 유형·ROI·파형 + 복셀 단위 잡음 | 조건 방식 다름 |
| R의 정체 | 학위논문: 검출 규칙을 미분 가능한 텐서 연산으로 옮긴 **규칙 기반** 검출기(학습 파라미터 없음) | 없음 | IRIS 휘도 규칙을 같은 방식으로 옮길 수 있음 |
| 명시적 변형 영역 주석 | 원문에 LUMI와 동일한 스키마 없음 | `mask` 구현 | LUMI의 확장 |
| 표준 기반 독립 검사 후 위험 라벨 | 최종 연구 대상 | IRIS 판정(휘도·적색), 패턴은 불안정 | `IRIS_LABELING_PLAN.md` 5.1 |
| Inverse 보정 네트워크 | 여러 모델 실험 | 없음 | 데이터 생성 프로토타입 단계 |
| 60FPS 종단 파이프라인 | LUMI의 목표 | 없음 | 학습/실시간 성능과 무관 |

## 5. 주요 위험: GAN은 '새로운 위험 분포'를 보장하지 않는다

현재 GAN의 판별기는 Oracle의 변형 영상을 '실제' 데이터로 받는다. 따라서 **Oracle이 표현하지 못하는 현상을 자동으로 발견하도록 최적화됐다고 주장할 수 없다.** Oracle에 내재한 합성 흔적을 재현할 위험도 있다. 원논문의 미분 가능한 위험 평가기를 생략했으므로 단지 `G`를 붙였다는 사실만으로 Barbu의 효과를 구현했다고 볼 수도 없다.

이 때문에 최초의 연구 질문은 **'GAN이 필요한가?'**가 되어야 한다. 다음 조건에서만 추가한다.

- `Oracle-only` 대비 `Oracle+GAN`으로 학습한 후속 모델이 **별도 원본으로 구성한 비합성 평가셋**에서 성능을 개선하는가?
- 위험 이벤트 재현율 증가가 정상 영역 오탐과 과잉 보정 면적 증가를 초래하지 않는가?
- 생성 결과가 다양하며 특정 영상 또는 랜덤 시드에 붕괴하지 않는가?
- 판정은 D의 점수가 아니라 독립적 검사기의 결과에 근거하는가?

## 6. 결론: 채택·유보·제외

**채택:** Forward/Inverse 구분, 정상·변형 페어, 규칙 합성, 생성기 G/D 경쟁 학습의 가능성, 변형 전후 데이터 추적.  
**유보:** 위험 평가 CNN의 손실 결합, GAN 생성 데이터의 실질적 기여, 긴 영상 및 고해상도 일반화.  
**제외:** 'GAN 자체로 안전성 검증', '원문 완전 재현', '생성 영상의 자동 미리보기'.

## 참고문헌

[1] Barbu, A., Banda, D., & Katz, B. (2020). *Deep video-to-video transformations for accessibility with an application to photosensitivity.* Pattern Recognition Letters, 137, 99–107. DOI: https://doi.org/10.1016/j.patrec.2019.01.019  
원문 공개판: https://groups.csail.mit.edu/infolab/publications/Barbu-Deep-video-to-video-transformations.pdf  
[2] Isola, P. et al. (2017). *Image-to-Image Translation with Conditional Adversarial Networks.* CVPR. https://openaccess.thecvf.com/content_cvpr_2017/html/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.html  
[3] Wang, T.-C. et al. (2018). *Video-to-Video Synthesis.* NeurIPS. https://research.nvidia.com/publication/2018-12_video-video-synthesis
