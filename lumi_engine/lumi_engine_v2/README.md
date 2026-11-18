# LUMI-7 Real-Time Visual Safety Pipeline (v2)

광과민성 발작(PSE/PPR) 예방을 위한 **FPS 인식형 인과적 시공간 특징 추출(`CausalAuxExtractor`) + 초경량 통합 탐지·국소 보정 모델(`LUMI7`) + FastAPI 서버** 프로젝트입니다.

## 1. 주요 반영 사항 (v2)
- **[P1] 미학습 가중치 추론 차단**: `lumi7_best.pth`가 없으면 추론기 초기화(`FileNotFoundError`) 및 API 요청(`HTTP 503`)을 즉시 차단합니다.
- **[P1] 출력 컨테이너 `.mp4` 고정 및 VideoWriter 검증**: 입력 확장자(.avi, .mov 등)와 무관하게 `.mp4`로 출력하며, `VideoWriter.isOpened()`, 처리 프레임 수, 파일 크기(`> 0`)를 검증합니다.
- **[P1] FPS 기반 시간 상수 반영**: `CausalAuxExtractor`가 영상의 실제 FPS($\Delta t = 1/\text{fps}$)를 반영해 EMA 감쇠율과 차분 스케일을 조정하며, `--target-fps` 리샘플링도 지원합니다.
- **[P2] TBPTT 청크 경계 시간 손실 반영**: 이전 청크의 마지막 보정·정답 프레임을 이어받아 청크 경계 전환 및 `--chunk-size 1`에서도 시간 일관성 손실이 계산됩니다.

## 2. 실행 순서

```bash
pip install -r requirements.txt
python train.py --smoke-test
python train.py --epochs 30 --batch-size 4 --chunk-size 16 --fps 24.0 --save-path lumi7_best.pth
python infer.py --input sample_hazard.mp4 --output-safe outputs/safe.mp4 --output-mask outputs/mask.mp4 --weights lumi7_best.pth
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
