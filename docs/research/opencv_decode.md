# OpenCV 영상 디코드의 range·행렬 처리 확인 (opencv decode)

작성 2026-11-10 (노트북). 재현: `.venv\Scripts\python.exe scripts/decode_colorspace_probe.py` (PyAV 필요).
환경: opencv-python 4.14.0 (FFmpeg 백엔드: avcodec 61.19, swscale 8.3), PyAV 19.0.2.
실험 영상은 정지 화면이다. 단, §0-4의 판정 확인용 영상만 깜빡임 영상이다. 이 영상은 임시 폴더에서 수치로만 판정하고 지웠다.

## 0. 한눈에 보는 정리: OpenCV는 어떻게 RGB를 만들고, 왜 601과 709 차이가 문제인가

### 0-1. 영상 파일 안의 픽셀은 RGB가 아니다
압축 영상(H.264 등)은 픽셀을 **Y(밝기) · Cb · Cr(색차)**로 저장한다. 색차는 가로·세로를 절반으로 줄여(4:2:0) 용량을 아낀다.
RGB에서 Y를 만드는 가중치는 표준마다 다르다.

| 표준 | 쓰는 곳 | Y = |
|---|---|---|
| BT.601 | SD(480/576줄) | 0.299 R + 0.587 G + 0.114 B |
| BT.709 | HD(720/1080줄) | 0.2126 R + 0.7152 G + 0.0722 B |

또 Y·Cb·Cr의 값 범위도 두 가지다. **limited**(Y 16~235, 방송·일반 영상)와 **full**(0~255, 휴대폰·JPEG 계열)이다.
영상 파일에는 어느 행렬과 범위로 저장했는지 **표시(메타데이터)**가 붙는다. 되돌릴 때는 저장할 때와 같은 공식을 써야 원래 RGB가 나온다.

### 0-2. OpenCV가 RGB를 만드는 과정 (소스 코드로 확인)

```
cv2.VideoCapture(path).read()
 ① FFmpeg 디먹서·디코더(avcodec) → 프레임: Y·Cb·Cr 평면 (예: yuv420p), 표시(colorspace, color_range)도 함께 나옴
 ② libswscale 로 BGR24 변환 (SWS_BICUBIC) ← 여기서 행렬·범위를 정함
      - OpenCV 는 행렬·범위를 지정하지 않는다
      - 그래서 swscale 기본값: 행렬 = SWS_CS_DEFAULT(= ITU601), 범위 = limited
        (픽셀 형식 이름이 yuvj* 일 때만 handle_jpeg 가 full 로 바꿈)
 ③ BGR uint8 (0~255, 감마가 걸린 값) 을 numpy 배열로 돌려줌
```

근거 (원문 확인):
- OpenCV `modules/videoio/src/cap_ffmpeg_impl.hpp`(4.x)
  - 변환 대상은 `AV_PIX_FMT_BGR24`이고 `"sws_flags", SWS_BICUBIC`를 쓴다. 변환 호출은 `sws_scale_frame`(이전 버전 경로는 `sws_getCachedContext` + `sws_scale`)이다.
  - 파일 전체에 `sws_setColorspaceDetails`, `AVCOL_`, `SWS_CS_`가 **0건**이다. `colorspace`와 `color_range`는 디버그 로그에서만 쓴다.
  - 같은 파일의 하드웨어 프레임 경로 주석: "sws_scale_frame() applies BT.601 limited range whatever the stream signals"
- FFmpeg `libswscale/swscale.h`: `#define SWS_CS_DEFAULT 5`, `#define SWS_CS_ITU601 5`
- FFmpeg `libswscale/utils.c`
  - 기본 계수는 `sws_setColorspaceDetails(sws, ff_yuv2rgb_coeffs[SWS_CS_DEFAULT], sws->src_range, …)`로 정한다.
  - 범위는 `sws->src_range |= handle_jpeg(&sws->src_format)`로 정하고, `handle_jpeg`는 `AV_PIX_FMT_YUVJ420P` 등일 때만 1(full)을 돌려준다.
- 실험(§1)으로도 같은 동작을 확인했다. BT.709 표시 영상도 BT.601 값이 나오고, full 표시 영상도 yuvj가 아니면 limited로 펼쳐진다.

그다음 우리 판정기가 이 BGR을 받아 밝기를 잰다(`measure.luminance_cd`).
```
BGR/255 → sRGB 감마 풀기(선형화) → 0.2126 R + 0.7152 G + 0.0722 B → × 200 cd/m²
```
여기서도 0.2126/0.7152/0.0722를 쓰지만, 이것은 **선형 RGB로 실제 밝기를 재는 sRGB 표준 계수**다. ②의 '저장된 Y·Cb·Cr을 RGB로 되돌리는 행렬'과는 다른 단계다. ②에서 RGB가 틀어지면 이 단계의 공식이 맞아도 결과가 틀린다.

### 0-3. 그래서 무엇이 문제인가
- **SD 영상(BT.601로 저장):** 저장 공식과 되돌리는 공식이 같아서 문제가 없다. 노트북 원본 두 개(BBB 640, Elephants Dream)가 여기에 해당한다.
- **HD 영상(BT.709로 저장):** OpenCV가 BT.601로 되돌리므로 RGB가 틀어진다.
  - 회색은 오차가 0이다(Cb = Cr = 0이면 행렬과 무관).
  - 색이 진할수록 많이 틀어진다. 순적색은 42.6이어야 할 밝기가 34.7 cd/m²가 된다. RGB 격자 기준 최대 +39 / −15 cd/m²다.
  - 색 좌표 u′v′는 최대 0.023 움직인다.
- **full range 영상(휴대폰 촬영본 등):** 픽셀 형식이 yuvj가 아니면 limited로 펼쳐져 어두운 쪽과 밝은 쪽이 잘린다.

### 0-4. 판정까지 확인 (2026-11-10, `tests/test_decode_color.py::test_bt709_decode_changes_luminance_verdict`)
검정 ↔ 빨강(190, 0, 0)이 초당 4번 바뀌는 화면 전체 깜빡임을 BT.709로 저장하고, 두 방법으로 읽어 같은 판정기로 판정했다.

| 읽는 방법 | 빨강이 읽힌 값 | 밝기 | 휘도 판정 |
|---|---|---|---|
| OpenCV (BT.601로 되돌림) | RGB 172, 0, 1 | 17.5 cd/m² | **안전** (ΔL < 20이라 깜빡임으로 세지 않음) |
| 표시대로 (BT.709, 새 경로) | RGB 188, 0, 0 | 21.4 cd/m² | **위험** (초당 4번, 면적 100%) |

→ OpenCV로 읽으면 **위험한 영상을 안전하다고 판정**한다. 오라클 라벨이라면 위험 샘플이 안전으로 잘못 붙는 것이고, GAN의 판정 목표도 틀어진다.

적색 판정에서도 확인을 시도했다. 계산상 '포화 빨강(R/(R+G+B) ≥ 0.8)' 여부가 행렬에 따라 갈리는 색(R 140~255, G = B 30~110)을 실제 H.264로 만들어 판정했다.
- 판정이 뒤집힌 경우는 **0건**이었다.
- 압축 과정의 반올림 때문에 실제 값이 경계를 벗어났다. 그리고 검정과 비교하면 Δu′v′가 커서 0.2 경계와 멀다.
- 이 범위 밖(두 색 사이의 Δu′v′가 0.2에 가까운 쌍 등)에서는 뒤집힐 수 있다. 확인하지 않았다.

### 0-5. 어떻게 했나 (요약, 자세히는 §3)
- 영상을 열 때 PyAV로 표시를 읽어 **영상 단위로** 경로를 고른다.
  - OpenCV가 맞게 읽는 영상(RGB, 또는 BT.601 + OpenCV와 같은 범위)은 **그대로 OpenCV**를 쓴다. 이미 검증한 결과가 비트 단위로 같다.
  - 나머지는 **PyAV에 행렬과 범위를 명시**해서 변환한다.
- 표시가 없으면 mpv와 libplacebo 규칙을 따른다(가로 ≥ 1280 또는 세로 > 576이면 BT.709). HDR은 거부한다.
- 위 판정 사례를 회귀 테스트로 고정했다. 새 경로에서는 위험, OpenCV 경로에서는 안전으로 나와야 통과한다.

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
   - 같은 색의 u′v′ 위치가 최대 0.023(평균 0.008) 이동한다. 무작위 색 쌍 4000개 중 142개(3.6%)에서 적색 전이 조건 'Δu′v′ > 0.2'의 판정이 바뀐다(쌍 Δu′v′ 차이 최대 0.036). 무작위 쌍 기준이라 실제 영상의 비율은 아니다. 다만 0.2 경계 근처의 색 쌍은 행렬 차이만으로 뒤집힐 수 있다.
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

## 3. 대응 (2026-11-10 적용: '문제가 있는 영상만 PyAV로')

검토한 선택지는 세 가지였다. ① 검사만 하고 멈추기, ② 전부 PyAV로 바꾸기, ③ 기록만 하기. 사용자 결정에 따라 **cv2가 틀리는 영상만 PyAV로 읽는 방식**을 택했다.

- **구현:** `src/lumi_synth/data.py`의 `decode_info`(표시 읽기와 경로 결정)와 `iter_frames_bgr`(경로별 읽기). `read_clip`, `make_demo_hazards.py`, `eval_demo_hazards.py`가 이 함수를 쓴다.
- **경로 규칙 (영상 단위로 정하고, 한 영상은 한 경로로만 읽는다):**
  - **cv2:** RGB 계열 픽셀 형식(pse-test-media 등), 또는 BT.601이면서 range가 cv2가 고를 range(yuvj면 full, 그 밖은 limited)와 같을 때. 이미 검증한 결과가 비트 단위로 그대로다.
  - **PyAV:** 그 밖의 경우(BT.709, BT.2020, yuvj가 아닌데 full range 등). 행렬과 range를 명시해 `frame.reformat(format="bgr24", src_colorspace=…, src_color_range=…)`로 변환한다.
- **표시가 없을 때:** mpv `video/csputils.c`의 `mp_csp_guess_colorspace`와 libplacebo `src/colorspace.c`의 `pl_color_system_guess_ycbcr`가 같은 규칙을 쓴다.
  - 원문: `width >= 1280 || height > 576 ? BT_709 : BT_601`
  - range는 YCbCr이면 limited다(libplacebo `pl_color_levels_guess`, mpv의 unknown → limited).
  - 참고로 PyAV(swscale)는 표시가 없으면 해상도와 무관하게 BT.601로 변환한다(실험 확인). 그래서 규칙은 우리 코드에서 정해서 넘긴다.
- **거부하는 영상:** HDR 전달 특성(PQ, HLG)과 지원하지 않는 행렬(YCgCo, ICtCp 등)은 `ValueError`로 멈춘다. 판정 규칙이 SDR(sRGB, 최대 200 cd/m²) 기준이라 조용히 틀리게 재지 않게 하려는 것이다.
- **프레임 번호(PyAV 경로):** '파일 처음부터 표시 순서로 센 번호'로 정한다.
  - 시각 × fps로 계산하지 않는 이유: 실제 파일은 프레임 간격이 고르지 않다. Elephants Dream은 첫 간격이 11프레임 분량이고, 1초 공백 등 간격이 다른 곳이 35군데다. 평균 fps는 23.39, 공칭 fps는 24다.
  - 방법: 패킷 pts를 정렬한 표('n번째 프레임 = pts')를 만들어(디코드 없이 패킷만 읽음, 파일당 1회, ED 5.6초) 목표 pts를 찾는다. 이동이 목표를 넘으면 더 앞에서 다시 이동하고, 그래도 안 되면 처음부터 센다.
  - cv2 경로는 cv2의 `CAP_PROP_POS_FRAMES`를 그대로 쓴다. 이 값은 시각 × 평균 fps라서 간격이 고르지 않은 파일에서 1프레임 안팎 다를 수 있다. 기존 데이터와 같은 결과를 지키려고 바꾸지 않았다.
  - 첫 구현은 시각 × 평균 fps로 계산해서 ED 10곳 중 8곳에서 1프레임 밀렸다. 코드 리뷰에서 발견했다.
  - 디코더가 내보내지 않는 패킷(앞부분을 잘라 낸 MP4/MOV의 편집 목록, `is_discard`)은 표에서 뺀다. 재리뷰에서 발견했다. 빼지 않으면 잘린 개수만큼 앞 프레임으로 이동했다.
  - 같은 pts가 둘 이상이거나 pts가 없으면 처음부터 센다. 이동한 뒤 첫 프레임의 pts가 목표와 정확히 같지 않을 때도 처음부터 센다.
- **기록:** 오라클 manifest의 각 샘플에 `decode`(pix_fmt, matrix, color_range, 추정 여부, 전달 특성, 경로)를 남긴다. `make_oracle_dataset.py`는 사전 검사 때 원본마다 디코드 정보를 출력한다.
- **확인:**
  - `tests/test_decode_color.py` 14개를 추가했다(마지막 1개는 §0-4의 판정 확인). 코드 리뷰를 두 번 받았고, 지적한 HIGH 1건과 MEDIUM 4건을 반영했다.
    - BT.709 표시 영상이 BT.709 이론값으로 읽히는지, cv2로 읽으면 BT.601 값이 나와 문제가 재현되는지
    - full range 표시가 유지되는지
    - BT.601 SD 영상은 cv2 경로이고 결과가 비트 단위로 같은지
    - 표시가 없을 때 해상도 규칙이 적용되는지
    - 두 경로의 프레임 번호가 같은지(키프레임 간격 4, 시작 0/5/9/14)
    - 실제 파일(ED)에서 이동한 프레임이 차례로 센 프레임과 같은지(10곳, 파일이 없으면 건너뜀)
    - HDR(PQ) 영상을 거부하는지
    - BT.601 full range(yuvj)는 cv2 경로에 남는지
    - 앞 3프레임이 편집 목록으로 잘린 MP4에서 이동이 정확한지. 수정을 빼면 실패하는 것도 확인했다.
  - 전체 테스트: 82 passed, 4 skipped(이전과 같은 pse-test-media 없음).
  - 노트북 원본 두 개는 cv2 경로로 정해지고, `read_clip` 결과가 이전과 비트 단위로 같다.
- **남은 것:**
  - 집 PC에서 BBB 1080p의 표시를 확인하고(`scripts/decode_colorspace_probe.py`), 1080p 트랙 결과(시연 9개 등)를 다시 계산해야 한다. BT.709라면 PyAV 경로로 바뀌어 값이 달라진다.
  - 10비트 영상, HEVC/VP9/AV1은 아직 시험하지 않았다.

## 4. 확인하지 못한 것

- 집 PC의 BBB 1080p, pse-test-media, 팀원 데이터셋의 실제 range와 matrix 표시
- 10비트 영상이나 HEVC/VP9/AV1에서의 cv2 동작(이번 실험은 8비트 H.264와 rawvideo만)
- cv2 `CAP_PROP_CONVERT_RGB`를 끄고 YUV를 직접 받는 경로(백엔드마다 동작이 달라 이번에는 쓰지 않음)
