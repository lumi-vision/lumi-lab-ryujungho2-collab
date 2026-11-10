# OpenCV 영상 디코드의 range·행렬 처리 확인 (opencv decode)

작성 2026-11-10 (노트북). 재현: `.venv\Scripts\python.exe scripts/decode_colorspace_probe.py` (PyAV 필요).
환경: opencv-python 4.14.0 (FFmpeg 백엔드: avcodec 61.19, swscale 8.3), PyAV 19.0.2.
모든 실험 영상은 정지 화면이다(깜빡임 없음).

## 결론 먼저

1. **우리 코드에는 range나 색 변환 행렬을 처리하는 부분이 없다.** 모든 영상 읽기가 `cv2.VideoCapture` → `read()`의 기본 동작에 맡겨져 있다.
   - 해당 위치: `data.read_clip`, `validate_rules_pse.py`, `make_demo_hazards.py`, `eval_demo_hazards.py`, `step0/step1`, 테스트
2. **OpenCV는 range 메타데이터(color_range)를 보지 않는다.** 디코더의 픽셀 형식 이름으로만 range를 정한다.
   - `yuv420p`이면 항상 limited로 펼친다(16→0, 235→255). **full 표시가 있어도 마찬가지다.**
   - `yuvj420p`(H.264 full range가 이 형식으로 나올 때)일 때만 full로 읽는다.
3. **OpenCV는 색 변환 행렬 표시를 보지 않고 항상 BT.601로 변환한다.** BT.709로 표시된 영상도 BT.601 이론값과 같은 RGB가 나왔다.
4. **우리가 지금 노트북에서 쓰는 원본은 영향을 받지 않는다.**
   - BBB 640×360과 Elephants Dream 480×270은 둘 다 `limited + smpte170m(BT.601)`로 표시되어 있어서, OpenCV 기본 동작이 정답과 일치한다.
5. **HD 원본(BT.709 표시)은 영향을 받는다.**
   - 회색(무채색)은 오차가 0이다.
   - 채도가 높은 색은 휘도가 최대 +39 / −15 cd/m² 틀어진다(RGB 격자 평균 |차| 7.2).
   - 순적색은 42.6이어야 할 휘도가 34.7 cd/m²로 계산된다.
   - 포화 적색 판정이 바뀌는 색은 4913개 중 13개다.
   - 해당 원본: 집 PC의 BBB 1080p(`data/sources/…_1080p_h264.mov`, 노트북에 없어 표시 미확인), 팀원 데이터셋, 앞으로 쓸 HD 영상
   - pse-test-media는 README 기준 무손실 RGB(FFV1/HFYU)라서 YUV 변환을 거치지 않는다. 따라서 영향이 없다. 노트북에 파일이 없어 이번에 직접 확인하지는 못했다.

## 1. 측정 결과

**range** (회색 띠, 색차 U=V=128)

| 영상 | FFmpeg가 본 표시 | Y 0, 16, 64, 128, 235, 255 → cv2 G |
|---|---|---|
| Y4M, 표시 없음 | yuv420p, unspecified | 0, 0, 55, 130, 255, 255 |
| Y4M, LIMITED | yuv420p, tv | 0, 0, 55, 130, 255, 255 |
| **Y4M, FULL** | yuv420p, **pc** | **0, 0, 55, 130, 255, 255** ← full인데 limited로 펼침 |
| H.264, 기본 | yuv420p, unspecified | 0, 0, 55, 130, 255, 255 |
| H.264, full | **yuvj420p**, pc | 0, 16, 64, 128, 235, 255 ← 맞게 읽음 |

limited 해석 이론값은 0, 0, 56, 130, 255, 255다(64만 1 차이, 반올림).

**행렬** (YUV = 120, 90, 180)

| 표시 | cv2 RGB | 이론 BT.601 | 이론 BT.709 |
|---|---|---|---|
| smpte170m (601) | 203, 92, 44 | 204, 94, 44 | 214, 101, 41 |
| **bt709** | **203, 92, 44** | 204, 94, 44 | 214, 101, 41 |

## 2. 왜 중요한가

- 판정기(`measure.luminance_cd`, `red_state`)는 cv2가 돌려준 8비트 BGR을 sRGB로 보고 휘도와 색도를 계산한다.
  - 디코드 단계에서 range가 틀리면 어두운 쪽과 밝은 쪽이 잘리거나 눌린다. 그러면 ΔL 20 cd/m²와 160 cd/m² 경계가 움직인다.
  - 행렬이 틀리면 유채색의 휘도와 적색 비율이 움직인다.
- [gamma_eotf.md](gamma_eotf.md) §1에서 본 것처럼, range 해석만으로도 경계 패턴(pse-test-media f001)의 판정이 바뀐다.
- 오라클 데이터는 바탕 클립을 cv2로 읽은 BGR 그대로 쓴다. 바탕이 HD(BT.709) 원본이면 바탕 색이 BT.601 기준으로 약간 틀어진 채 저장된다.
  - 다만 정답 라벨은 같은 BGR 배열로 판정하므로 데이터 안에서는 일관된다.
  - 문제가 되는 것은 '원본 영상 그대로'를 다른 도구(팀원 판별기, IRIS)와 비교할 때다.

## 3. 대응 방안 (미적용, 결정 필요)

1. **입력 검사(가장 작은 변경):** 영상을 열 때 PyAV로 range와 matrix를 읽는다. `limited + 601`이 아니면 경고하거나 멈춘다. `make_oracle_dataset.py`의 사전 검사(확대 금지)와 같은 자리에 넣을 수 있다.
2. **올바른 디코드:** 디코드를 PyAV로 바꾼다. `frame.to_ndarray(format="bgr24")`에 원본 range와 matrix를 넘긴다(PyAV `reformat`의 `src_colorspace`, `src_color_range`). 의존성이 추가된다(현재 requirements에 av 없음).
3. **기록만:** manifest와 점검표에 원본의 range와 matrix를 기록하고, 판정은 cv2 기본 동작으로 계속한다.

- 우리 팀 기준으로는 1 + 3이 최소 대응이다.
- 1080p BBB를 쓰는 집 PC 트랙에는 2가 필요할 수 있다. 먼저 1080p 원본의 표시를 확인한다(이 스크립트를 집 PC에서 실행).

## 4. 확인하지 못한 것

- 집 PC의 BBB 1080p, pse-test-media, 팀원 데이터셋의 실제 range와 matrix 표시
- 10비트 영상이나 HEVC/VP9/AV1에서의 cv2 동작(이번 실험은 8비트 H.264와 rawvideo만)
- cv2 `CAP_PROP_CONVERT_RGB`를 끄고 YUV를 직접 받는 경로(백엔드마다 동작이 달라 이번에는 쓰지 않음)
