"""밝기와 깜빡임을 재는 함수들 (ITU-R BT.1702 방식). 모든 확인 단계가 이 파일의 함수로 측정한다.

ITU-R BT.1702의 일반 깜빡임 기준(적색 깜빡임·줄무늬 패턴은 아직 다루지 않음)
1. 밝기는 물리 휘도 cd/m²로 잰다. 일반 화면(SDR)의 흰색 = 200 cd/m².
2. 한 픽셀의 밝기가 20 cd/m² 이상 바뀌고, 어두운 쪽이 160 cd/m² 미만이면 그 픽셀에 "전이" 1번.
   밝아졌다가 다시 어두워지면 전이 2번 = 깜빡임 1번.
3. 같은 프레임에 전이한 픽셀이 화면의 25%를 넘으면 그 프레임은 "화면 전이".
4. 아무 1초 구간에서 깜빡임(화면 전이 2번 = 1번)이 3번을 넘으면 위험.

rockrovks 판별기(vdcoreops_v1.py)와 밝기 공식·단위(cd/m²)·25%·3번 기준은 같다.
차이: rockrovks는 "한 프레임 안에서 평균보다 1.3배 밝은 영역"을 보고, 여기서는 ITU 원문대로
"같은 픽셀이 시간에 따라 얼마나 바뀌었는지"를 본다.
"""
import numpy as np

WHITE_CD = 200.0          # 흰색(상대 휘도 1.0)의 물리 휘도 (cd/m²)
DELTA_CD = 20.0           # 전이로 셀 최소 휘도 변화 (cd/m²)
DARK_LIMIT_CD = 160.0     # 어두운 쪽이 이 값 미만일 때만 전이로 셈 (cd/m²)
AREA_LIMIT = 0.25         # 화면 전이로 셀 최소 면적 (화면 대비 비율)
MAX_FLASHES_PER_SEC = 3   # 1초에 이보다 많이 깜빡이면 위험


def luminance_cd(frames_bgr: np.ndarray) -> np.ndarray:
    """uint8 BGR 프레임(…×H×W×3)을 픽셀별 물리 휘도(cd/m², float32)로 바꾼다."""
    c = frames_bgr.astype(np.float32) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    b, g, r = lin[..., 0], lin[..., 1], lin[..., 2]
    return WHITE_CD * (0.2126 * r + 0.7152 * g + 0.0722 * b)


def pixel_transitions(lum: np.ndarray) -> np.ndarray:
    """픽셀별 휘도(T×H×W)에서, 각 프레임에 전이가 일어난 픽셀을 True로 표시한 배열(T×H×W)을 돌려준다.

    픽셀마다 직전 전이 이후의 최고점·최저점을 기준으로 삼고,
    반대 방향으로 DELTA_CD 이상 움직이면(어두운 쪽 < DARK_LIMIT_CD) 전이 1번.
    """
    ref = lum[0].copy()
    direction = np.zeros(lum.shape[1:], dtype=np.int8)  # +1 밝아지는 중, -1 어두워지는 중, 0 아직 없음
    out = np.zeros(lum.shape, dtype=bool)
    for t in range(1, len(lum)):
        v = lum[t]
        # 같은 방향으로 더 가면 기준점을 끝점으로 옮긴다
        extend = ((direction == 1) & (v > ref)) | ((direction == -1) & (v < ref))
        ref = np.where(extend, v, ref)
        diff = v - ref
        big = (np.abs(diff) >= DELTA_CD) & (np.minimum(v, ref) < DARK_LIMIT_CD) & ~extend
        new_dir = np.sign(diff).astype(np.int8)
        trans = big & (new_dir != direction)
        out[t] = trans
        direction = np.where(trans, new_dir, direction)
        ref = np.where(big, v, ref)
    return out


def max_flashes_per_second(events: np.ndarray, n_frames: int, fps: float) -> float:
    """화면 전이 프레임 번호들로, 아무 1초 구간의 깜빡임(전이 2번 = 1번) 최댓값을 구한다."""
    if len(events) == 0:
        return 0.0
    win = int(round(fps))
    best = max(int(np.sum((events >= s) & (events < s + win))) for s in range(n_frames))
    return best / 2.0


def analyze(frames_bgr: np.ndarray, fps: float) -> dict:
    """영상(T×H×W×3, BGR)을 ITU 방식으로 재서 결과를 돌려준다.

    돌려주는 값
    - mean_cd: 프레임별 화면 평균 휘도 (cd/m²)
    - area: 프레임별 전이 픽셀 면적 비율 (0~1)
    - events: 화면 전이(면적 > 25%)가 일어난 프레임 번호
    - max_flashes: 아무 1초 구간의 최대 깜빡임 수
    - peak_area: 가장 넓었던 전이 면적 비율
    - dangerous: 위험 여부 (max_flashes > 3)
    """
    lum = luminance_cd(frames_bgr)
    area = pixel_transitions(lum).mean(axis=(1, 2))
    events = np.flatnonzero(area > AREA_LIMIT)
    flashes = max_flashes_per_second(events, len(lum), fps)
    return {
        "mean_cd": lum.mean(axis=(1, 2)),
        "area": area,
        "events": events,
        "max_flashes": flashes,
        "peak_area": float(area.max()),
        "dangerous": flashes > MAX_FLASHES_PER_SEC,
    }
