# LUMI-7 Real-Time Visual Safety Pipeline

광과민성 발작(PSE/PPR) 예방을 위한 **인과적 시공간 특징 추출(`CausalAuxExtractor`) + 초경량 통합 탐지·국소 보정 모델(`LUMI7`) + FastAPI 서버** 프로젝트입니다.

## 1. 파일 구조

- `features.py`: 9채널 인과적 시공간·전역 특징 추출기 (`CausalAuxExtractor`)
- `model.py`: ConvGRU + Zero-init Global Context 기반 탐지·보정 통합 모델 (`LUMI7`)
- `loss.py`: 탐지·복원·배경보존·시간일관성·안전규칙 공동 학습 손실 (`lumi_loss`)
- `train.py`: TBPTT + 영상 단위 Gradient Accumulation 학습 스크립트
- `infer.py`: 학습된 가중치(`lumi7_best.pth`)를 이용한 MP4 스트리밍 추론 및 시연 영상 생성기
- `app.py`: Next.js(v0.dev) 대시보드 연동용 FastAPI 백엔드 서버
- `requirements.txt`: 필수 파이썬 패키지 목록

## 2. 실행 순서

### Step 1. 환경 설치 및 스모크 테스트
```bash
pip install -r requirements.txt
python train.py --smoke-test
```

### Step 2. 모델 학습 (`lumi7_best.pth` 생성)
`train.py`의 `DummySyntheticDataset` 부분을 생성기(G) 데이터셋으로 연결한 뒤 학습을 실행합니다.
```bash
python train.py --epochs 30 --batch-size 4 --chunk-size 16 --save-path lumi7_best.pth
```

### Step 3. 단독 시연 영상(MP4) 생성 테스트
```bash
python infer.py --input sample_hazard.mp4 --output-safe outputs/safe.mp4 --output-mask outputs/mask.mp4 --weights lumi7_best.pth
```

### Step 4. FastAPI 백엔드 서버 실행 (웹 대시보드 연동)
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
