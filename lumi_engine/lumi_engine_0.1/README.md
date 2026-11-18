# LUMI Engine 0.1

영상의 모든 원본 프레임을 분석하고 보정 MP4, 마스크 MP4, 프레임별 탐지 요약을 생성한다. 출력 FPS를 바꿔도 탐지 시각은 원본 영상 기준이다.

## 실행 점검

이 폴더에서 프로젝트 가상환경을 활성화한 뒤 실행한다.

```powershell
python -m pip install -r requirements.txt
python train.py --smoke-test
python -m unittest -v test_release
```

## 학습과 검증

클립 파일은 .npz 또는 .pt 형식이다. 각 파일에 video(또는 rgb), clean, target_mask를 넣고 fps를 지정한다. RGB와 정답 영상은 [T,3,H,W], 마스크는 [T,1,H,W]이며 값 범위는 0~1이다. 한 배치의 클립은 텐서 크기가 같아야 한다. 클립마다 다른 FPS를 사용할 수 있다. FPS 생략 시 --fps 값(기본 24)을 사용한다.

```powershell
python train.py --data-dir ./dataset/train --save-path candidate.pth
```

검증 없이 저장한 모델과 --allow-dummy로 학습한 모델은 API에서 거부된다. API용 체크포인트를 생성하려면 원본 영상 단위로 학습·검증 데이터를 나누고 같은 학습 명령에 아래 옵션을 제공한다. 프로그램은 양쪽 폴더에 내용이 동일한 파일이 있으면 중단한다. 동일 영상에서 잘라 낸 서로 다른 클립은 데이터 준비 단계에서 분리해야 한다.

| 옵션 | 의미 |
| --- | --- |
| --val-dir | 별도 평가 클립 폴더 |
| --min-recall | 위험 픽셀 재현율의 하한 |
| --max-fpr | 정상 픽셀 오탐률의 상한 |
| --max-hazard-mae | 위험 영역에서 보정 RGB와 정답 RGB 사이 평균 절대 오차의 상한 |

세 기준은 0~1 범위에서 실험 전에 정한다. 매 에폭 평가하고 측정값과 기준을 체크포인트에 함께 저장한다. 검증 통과 모델을 우선 저장하며, 검증 결과가 같을 때는 학습 손실이 낮은 모델을 선택한다. 검증셋에 양성·음성 픽셀이 모두 있고 세 기준을 통과한 경우에만 validated=True가 된다. 이 상태는 지정한 데이터셋에서의 성능을 나타낸다. 임상적 안전성은 별도 평가 대상이다.

## 추론과 API

검증을 통과한 체크포인트로 실행한다. 서버 기본 경로는 이 폴더의 lumi7_best.pth이고 LUMI_WEIGHTS_PATH 환경변수로 바꿀 수 있다.

```powershell
python infer.py --input sample.mp4 --output-safe outputs/safe.mp4 --output-mask outputs/mask.mp4 --weights lumi7_best.pth
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

GET /health에서 모델 준비 상태를 확인하고 POST /api/analyze-and-process에 file 필드로 영상을 업로드한다. 모델이 준비되지 않으면 HTTP 503을 반환한다. 출력 코덱은 환경에 따라 avc1 또는 mp4v를 사용한다.
