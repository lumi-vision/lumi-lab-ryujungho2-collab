# 현재 개인 실험의 범위와 한계

## 현재 구현

- 영상 입력 및 수동 플레이어(기본 미리보기 차단)
- Oracle의 원본–변형본 쌍과 `mask`(변형 허용 ROI), `activation`(시간 파형) 생성
- 조건부 3D PatchGAN 생성기 G와 판별기 D, ROI 재구성 손실, GAN 학습 스크립트
- 형식 및 소스 해시 기반 분할 정합성 감사

## 아직 확인할 수 없는 주장

- 실제 기준 위반/비위반 진단 및 임상적 안전성
- 합성 GAN 학습의 수렴과 비합성 영상에 대한 일반화
- 픽셀별 PSE 위험 정답 마스크 및 국소 보정 정답
- 60FPS 실시간 위험 탐지·국소 보정 파이프라인

상세 근거는 [연구 설계](RESEARCH_DESIGN.md), [Barbu 원문 비교](BARBU_GAN_REVIEW.md), [실험 계획](EXPERIMENT_PLAN.md), [소스 감사](IMPLEMENTATION_AUDIT.md)에 기록한다.
