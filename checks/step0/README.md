# 0단계 점검표: 환경과 재료 준비

**결과: ✅ 통과** (12/12 항목 통과)

| 항목 | 결과 | 통과 |
|---|---|---|
| 파이썬 | 3.14.7 (C:\dev\lumi-vision\lumi-lab-ryujungho2-collab\.venv\Scripts\python.exe) | ✅ |
| 패키지 numpy | 2.5.4 | ✅ |
| 패키지 opencv-python | 5.0.0.93 | ✅ |
| 패키지 matplotlib | 3.11.2 | ✅ |
| 패키지 pytest | 9.2.0 | ✅ |
| 패키지 lumi-synth | 0.1.0 | ✅ |
| 원본 영상 준비 | 이미 있음 | ✅ |
| 원본 영상 SHA256 | `dc2146a2b1172def…` (기대값 `dc2146a2b1172def…`) | ✅ |
| 원본 영상 열기 | 1920×1080, 24.00 fps, 14315프레임 (약 9.9분) | ✅ |
| 무손실 저장(FFV1) 시험 | 1920×1080 프레임 5장, 최대 픽셀 차이 0 | ✅ |
| 공간 패턴 25pct_screen | PNG 16개 (기대 16개: 모양 8종 × f/a) | ✅ |
| 시간 패턴 rel_lum | CSV 234개 | ✅ |

## 항목 설명

- **원본 영상**: Big Buck Bunny 1080p (CC BY 3.0, © Blender Foundation). 공간 패턴이 1920×1080으로 만들어져 있어서 같은 해상도가 필요합니다.
- **SHA256**: 파일의 지문입니다. 기록해 둔 값과 같으면 공식 원본과 한 바이트도 다르지 않다는 뜻입니다.
- **무손실 저장 시험**: 합성 영상을 FFV1 형식으로 저장합니다. 저장했다가 다시 읽었을 때 픽셀 차이가 0이어야 깜빡임이 압축 때문에 뭉개지지 않습니다.
- **공간·시간 패턴**: `C:\dev\lumi-vision\tools\pse-test-media` (BSD-3-Clause)

만든 명령: `python scripts\step0_check.py`
