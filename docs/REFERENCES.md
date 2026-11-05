# 연구 근거 및 참조 링크

**[R1]** Barbu, A., Banda, D., & Katz, B. (2020). *Deep video-to-video transformations for accessibility with an application to photosensitivity*. Pattern Recognition Letters 137, 99–107. DOI: https://doi.org/10.1016/j.patrec.2019.01.019. 공개 원문: https://groups.csail.mit.edu/infolab/publications/Barbu-Deep-video-to-video-transformations.pdf. 활용: §3.1 Forward(Oracle/GAN), §3.2 Inverse(복원), §4 실험. 온라인 2019년 공개, 저널 권호 기준 2020년.

**[R2]** Jordan, J. B. (2025). *Evaluating Conformance of Video Safety Tools for Photosensitive Epilepsy*. In HCII 2025 (LNCS 15780), 85–98. DOI: https://doi.org/10.1007/978-3-031-93848-1_7. 활용: 재현 가능한 벤치마크 미디어 및 검사기의 일치도 평가.

**[R3]** traceRERC. *pse-test-media*: https://github.com/traceRERC/pse-test-media. 활용: 시험 미디어 생성의 시간·색상·공간 패턴 구성 참고. 라이선스와 실행 환경은 도입 전 원 저장소에서 확인.

**[R4]** ITU-R (2023). *Recommendation BT.1702-3: Guidance for the reduction of photosensitive epileptic seizures caused by television*. https://www.itu.int/rec/R-REC-BT.1702-3-202311-I/en. 현행판 BT.1702-3(11/2023). 활용: 방송 영상의 점멸·패턴 기준 및 검사 환경 참고.

**[R5]** W3C. *Understanding WCAG 2.2 SC 2.3.1: Three Flashes or Below Threshold*. https://www.w3.org/WAI/WCAG22/Understanding/three-flashes-or-below-threshold.html. 활용: 웹 콘텐츠의 점멸 관련 성공 기준. ITU 방송 가이드라인과 동일 규격이 아님.

**[R6]** Isola, P. et al. (2017). *Image-to-Image Translation with Conditional Adversarial Networks*. CVPR. https://openaccess.thecvf.com/content_cvpr_2017/html/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.html. 활용: 조건부 적대학습이라는 일반 설계의 배경. LUMI가 pix2pix 자체를 재현했다는 뜻이 아님.

**[R7]** Wang, T.-C. et al. (2018). *Video-to-Video Synthesis*. NeurIPS. https://research.nvidia.com/publication/2018-12_video-video-synthesis. 활용: 시공간 GAN·비디오 일관성의 추가 설계 근거.

**인용 규칙:** 선행연구에서 확인한 사실에는 해당 문헌·절을 표시하고, LUMI의 설계 선택은 반드시 `제안`, `구현`, `검증됨`을 구분한다. 표준과 임상 안전성은 서로 다른 주장이다.
