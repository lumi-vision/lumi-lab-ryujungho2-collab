# LUMI-7 Real-Time Visual Safety Pipeline (v3)

광과민성 발작(PSE/PPR) 예방을 위한 **FPS 인식형 인과적 시공간 특징 추출(`CausalAuxExtractor`) + 초경량 통합 탐지·국소 보정 모델(`LUMI7`) + 비동기 FastAPI 서버** 프로젝트입니다.

## 1. 주요 반영 사항 (v3)
- **[P1] 더미 학습 가중치 보호 및 실제 데이터셋 강제**: `train.py`는 `--data-dir` 없이 기본 실행 시 오류를 반환하며, `--allow-dummy`로 생성한 체크포인트에는 `trained_on_dummy=True, validated=False`가 기록되어 API에서 거부됩니다.
- **[P1] 체크포인트 추출기 설정 및 버전 호환성 검증**: `CausalAuxExtractor.get_config()`(`extractor_version`, `flash_thresh`, `ema_tau`, `ref_fps`)를 체크포인트에 저장하고 추론 로드 시 엄격히 대조합니다.
- **[P1] 전 프레임 안전 분석 후 출력 리샘플링**: `--target-fps`로 출력 FPS를 낮출 때도 원본 영상의 모든 프레임(`src_fps`)에 대해 탐지·보정을 수행한 뒤 출력 시간 격자로 매핑하여 짧은 교번 섬광 누락을 방지합니다.
- **[P2] 동적 결과 URL 반환**: 하드코딩된 `localhost:8000` 대신 요청 Host 기반 URL(`request.url_for`)과 상대 경로(`result_video_path`)를 반환합니다.
- **[P2] 이벤트 루프 논블로킹 처리**: 업로드 파일 저장과 영상 프레임 추론을 `run_in_threadpool`로 오프로드하여 처리 중에도 `/health` 및 동시 요청이 지연되지 않습니다.

## 2. 실행 순서

```bash
pip install -r requirements.txt

# 1) 스모크 테스트 (동치성, FPS 민감도, 청크 경계 손실, 체크포인트/더미 가드 검증)
python train.py --smoke-test

# 2) 실제 생성기(G) 클립(.npz/.pt) 폴더로 학습
python train.py --data-dir ./dataset_clips --epochs 30 --batch-size 4 --chunk-size 16 --save-path lumi7_best.pth

# 3) 단독 MP4 추론 및 시연 영상 생성
python infer.py --input sample_hazard.mp4 --output-safe outputs/safe.mp4 --output-mask outputs/mask.mp4 --weights lumi7_best.pth

# 4) FastAPI 서버 실행
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
