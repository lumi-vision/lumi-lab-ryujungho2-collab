# Kaya et al. 2025 데이터셋의 근거 확인 (팀원이 받은 데이터셋)

작성 2026-11-10 (노트북). 목적은 팀원이 받았다는 데이터셋이 **무엇인지 근거만** 확인하는 것이다. 데이터 자체는 받지도 열지도 않았다.
표기: **원문**은 본문·코드·원 자료를 직접 읽은 것, **초록**은 초록·메타데이터만 읽은 것, **2차**는 다른 자료를 통한 것, **미확인**은 확인하지 못한 것이다. ★는 노트북에서 추출본으로 직접 다시 확인한 것이다.

> ⚠ 이 데이터셋에는 **실제 위험 영상**(방송·인터넷 아카이브·사용자 영상에서 모은 섬광 장면)이 들어 있을 가능성이 높다. 재생하거나 미리보기하지 않는다. 코드 지표로만 다룬다.

## 결론 먼저

1. **논문이 공식으로 지정한 데이터셋은 `PSE_triggering_dataset.rar`다.** Aydın Kaya 교수의 Hacettepe 웹페이지에서 배포하며 약 2.43 GB다. 팀원이 받은 것이 이 파일인지는 크기(2,429,272,890 bytes)나 파일 목록으로 대조해야 한다.
2. **라벨은 사람이 아니라 PEAT 판정으로 대신한 것으로 보인다.**
   - 같은 지도교수의 석사논문(Akyıldız 2025)이 "PEAT was used as a practical proxy for labeling"이라고 명시한다(★).
   - 다만 이 석사논문과 SIViP 논문의 데이터셋이 같은 것인지는 **미확인**이다.
3. **영상 수는 약 18~22개로 추정된다**(보충자료의 보정 그래프 22개, 설문 대상 18개). 출처, 해상도, fps, 라이선스는 **미확인**이다.
4. **탐지기는 규칙 기반이고 머신러닝이 아니다**(공식 코드 확인). W3C 상대휘도 0.1/0.8과 Ofcom 20/160 cd/m² 두 모드가 있다.
5. **LUMI에 대한 함의**
   - 실제 장면 위험 영상으로 우리 판정기와 팀원 판별기를 외부 검증할 수 있는 드문 자료다. 단, 정답이 PEAT 판정이라는 점을 함께 적어야 한다.
   - 석사논문은 합성 섬광 클립을 "너무 인위적('jumpscare')"이라며 주 데이터셋에서 뺐다. 이는 우리 오라클과 GAN의 **현실성**에 대해 리뷰어가 던질 질문과 같다(아래 §4).

## 1. 논문

- **서지:** Kaya, Kilic, Genc, Kural (2025). Detection and correction of the flashing parts that trigger photosensitive epilepsy in videos. *Signal, Image and Video Processing* 19(12), Art. 967. doi:10.1007/s11760-025-04608-4
  - 투고 2025-06-02, 게재 2025-09-05.
  - 구독 논문이라 본문은 **미확인**이다. OpenAlex와 Semantic Scholar 모두 closed로 표시한다.
  - 초록 [초록]: https://research.hacettepe.edu.tr/en/publications/detection-and-correction-of-the-flashing-parts-that-trigger-photo/
  - Springer 페이지 [초록]: https://link.springer.com/article/10.1007/s11760-025-04608-4
- **초록 요지:** 영상처리 기법으로 PSE 유발 구간을 자동 탐지하고 보정한다. Ofcom, ITU, W3C 지침을 따르며 특히 W3C를 강조한다. "a range of video sequences"로 평가했다.
- **저자 기여 (Springer 페이지):** "A.Y.K, M.C.G and Ali K., conducted the experiments, prepared the dataset and wrote the codebase."
- **그림 캡션:** Fig.1 "Sample frame sequences from the dataset", Fig.2 "GUI of the PEAT", Fig.3 "Output of our application".
- **보충자료 DOCX** [원문]: https://static-content.springer.com/esm/art%3A10.1007%2Fs11760-025-04608-4/MediaObjects/11760_2025_4608_MOESM1_ESM.docx
  - 부록 A: 보정 결과 그래프 VID:1~VID:22
  - 부록 B: 응답자 42명이 영상 18개를 1~5점으로 평가한 설문 원자료

## 2. 데이터셋

| 항목 | 내용 | 등급 |
|---|---|---|
| 데이터 가용성 진술 | "Dataset is available at: http://web.cs.hacettepe.edu.tr/~aydinkaya/pse/dataset/PSE_triggering_dataset.rar Code is available at: https://github.com/samfatu/pse-detection-correction." | 초록(Springer 페이지 공개 부분) |
| 파일 | HTTP HEAD 응답만 확인했다(다운로드 안 함). `Content-Length: 2429272890`, `application/vnd.rar`, `Last-Modified: Mon, 21 Oct 2024`. 상위 디렉터리는 403 | 원문(헤더) |
| 라이선스 | 데이터셋 라이선스는 명시되어 있지 않다. 코드는 BSD-3-Clause(2026-06-01 추가, "Per email correspondence with Prof. Aydın Kaya") | 데이터 미확인 / 코드 원문 |
| 영상 수 | 약 18~22개로 추정(보충자료 근거) | 추정 |
| 출처·해상도·fps | — | 미확인 |
| 라벨 | PEAT 판정(석사논문 근거) 또는 합성 구간 txt(코드 근거)일 가능성이 있다. 사람이 라벨링했다는 근거는 없다. | 추정 |
| 다른 저장소 | Zenodo, Figshare, GitHub에는 관련 데이터가 없다. Kaggle, Mendeley, ResearchGate는 확인하지 않았다. | 원문 / 미확인 |

**공식 코드의 단서** [원문, github.com/samfatu/pse-detection-correction]

- `corrupt_video.py`는 정상 영상에 3~6 Hz 사인파로 백색 또는 적색 섬광을 무작위 구간에 합성한다. 구간마다 `{video}_info.txt`에 `red:start-end` / `general:start-end`를 기록하고, `{video}_noised.avi`로 저장한다(프레임 구간 라벨).
- 탐지 조건 (`guidelines/w3c.py`): 상대휘도 변화 ≥ 0.1, 어두운 쪽 < 0.8, 화면 1/3 × 1/3 영역 중 25% 이상, 적색 `(R−G−B)*320` 변화 ≥ 20과 적색 비율 ≥ 0.8. 1초에 3회를 넘으면 실패(`THRESHOLD = 3`). `ofcom.py`는 20/160 cd/m², 25%, 3회 초과다.
- 보정: 앞뒤 프레임 시간 평균, 로그식 밝기 계수, 적색은 R 채널 감쇠.
- 논문 본문이 코드와 같은 임계값을 썼는지는 **미확인**이다.

## 3. 같은 연구실의 관련 연구: Akyıldız 석사논문 (2025-10, 지도교수 Aydın Kaya)

- 전문 [원문]: https://hdl.handle.net/11655/38400 (엠바고 해제 2026-11-01)
- 4장 데이터셋 (★ 원문 확인):
  - "an extensive search for existing datasets was conducted, but no suitable collection could be located."
  - 영상은 "online broadcasts, internet archives, and user-generated content"에서 직접 모았다. 게임 장면은 제외했고, "only publicly available videos containing appropriate warnings were included."
  - 합성 클립(백색, 적색, 적청 교대)은 PEAT 기준으로 대부분 위험이었지만, "appeared highly artificial … resembled an artificial 'jumpscare'"라서 **주 데이터셋에서 빼고 진단용 스트레스 테스트로만 남겼다.**
  - "PEAT was used as a practical proxy for labeling, but it does not perfectly reflect human perceptual responses and may produce both false positives and false negatives."
  - 데이터가 작아서 머신러닝 대신 결정적 영상처리를 택했다.
- 6장: PEAT 입력 규격은 AVI, 640×360, 25 FPS다(★). 영상 18개의 PEAT 실패 프레임 감소율은 1차 알고리즘 61.75%, 2차 79.33%다. 합성 데이터 탐지는 민감도 96.2%, 특이도 92.8%다.
- **주의:** 이 석사논문은 SIViP 논문을 인용하지 않고, 데이터셋 이름도 쓰지 않는다. 영상 18개라는 숫자만 같으므로 같은 데이터라고 단정할 수 없다.

## 4. LUMI에 대한 함의

1. **외부 검증 자료로 쓸 수 있다**(팀원이 가진 파일이 확인되면).
   - 실제 장면 위험 영상으로 우리 `measure.analyze_all`과 팀원 판별기의 판정을 비교할 수 있다.
   - 다만 정답은 PEAT 판정이다. 그래서 '정답 일치율'이 아니라 'PEAT와의 일치율'로 보고해야 한다.
   - PEAT는 WCAG 2.0 기반 구버전 Harding 엔진이다([gamma_eotf.md](gamma_eotf.md) §2).
   - 640×360 / 25 FPS로 변환된 영상이라면 해상도 민감도([RESOLUTION_ENV_REPORT.md](../RESOLUTION_ENV_REPORT.md))를 고려해야 한다. 25 FPS는 우리 24 fps 트랙과도 다르다.
2. **현실성 비판에 대비해야 한다.** Akyıldız는 합성 섬광을 'jumpscare 같다'며 주 데이터에서 뺐다.
   - 우리 오라클은 실제 장면 위에 국소 마스크로 빛을 더하므로 전체 화면 합성보다는 자연스럽다.
   - 하지만 '실제 위험 장면과 분포가 얼마나 비슷한가'는 아직 측정한 적이 없다. 이 데이터셋의 판정 지표(면적, 주파수, ΔL 분포)와 오라클 분포를 비교하는 것이 가능한 검증이다(재생 없이 수치만).
3. **"공개 데이터셋이 없다"는 우리 동기 문장은 이렇게 다듬어야 한다.** 이 데이터셋은 공개 URL로 배포되지만 라이선스가 없고, 규모가 작고, PEAT 라벨이다. 따라서 "공개됐지만 라이선스·규모·라벨이 제한적인 소규모 자료만 있다"고 쓴다.
4. 디코드 주의: 영상이 YUV(AVI/MP4)라면 OpenCV는 range와 행렬 표시를 무시한다. 열기 전에 [opencv_decode.md](opencv_decode.md)의 스크립트로 메타데이터부터 확인한다.

## 5. 팀원에게 확인할 것

1. 받은 파일 이름과 크기가 `PSE_triggering_dataset.rar` / 2,429,272,890 bytes와 같은지
2. 압축 안에 `*_info.txt`(`red:` / `general:` 구간)나 `*_noised.avi`가 있는지. 있으면 합성 파이프라인에서 나온 데이터다.
3. PEAT 보고서가 들어 있는지, 영상 수, 해상도, fps
4. 라이선스나 사용 조건을 저자에게서 받았는지

## 6. 확인하지 못한 것

- SIViP 논문 본문(유료): 데이터셋 설명, 정량 결과, 실제 임계값
- RAR 내용: 위험 영상일 수 있고 2.4 GB라 받지 않음
- Kaggle, Mendeley Data, ResearchGate
