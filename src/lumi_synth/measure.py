"""밝기와 깜빡임을 재는 함수들. 모든 확인 단계가 이 파일의 함수로 측정한다.

용어
- 상대 휘도(relative luminance): 사람 눈이 느끼는 밝기를 0(검정)~1(흰색)로 나타낸 값. WCAG·ITU 기준이 이 값을 쓴다.
- 전이(transition): 밝기가 한 방향으로 0.1 이상 바뀐 것. 밝아졌다가 다시 어두워지면 전이 2번 = 깜빡임 1번.
- 위험 기준(ITU-R BT.1702, WCAG 2.x 단순화): 어두운 쪽 휘도가 0.8 미만이면서 0.1 이상 바뀌는 전이가
  1초 안에 6번 넘게(= 깜빡임 3번 초과) 일어나면 위험.
"""
import numpy as np

DELTA = 0.1          # 전이로 셀 최소 휘도 변화
DARK_LIMIT = 0.8     # 어두운 쪽이 이 값 미만일 때만 전이로 셈
MAX_FLASHES_PER_SEC = 3


def relative_luminance_bgr(frames_bgr: np.ndarray) -> np.ndarray:
    """uint8 BGR 프레임(…×3)을 픽셀별 상대 휘도(0~1, float32)로 바꾼다."""
    c = frames_bgr.astype(np.float32) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    b, g, r = lin[..., 0], lin[..., 1], lin[..., 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def count_transitions(luma: np.ndarray) -> np.ndarray:
    """프레임별 밝기 곡선(1차원)에서 전이가 일어난 프레임 번호들을 돌려준다.

    직전 전이 이후의 최고점·최저점을 기준으로, 반대 방향으로 DELTA 이상 움직이면 전이 1번.
    """
    events = []
    ref = float(luma[0])
    direction = 0  # +1 밝아지는 중, -1 어두워지는 중, 0 아직 없음
    for i, v in enumerate(luma[1:], start=1):
        v = float(v)
        # 같은 방향으로 더 가면 기준점을 끝점으로 옮긴다
        if (direction == 1 and v > ref) or (direction == -1 and v < ref):
            ref = v
            continue
        if abs(v - ref) >= DELTA and min(v, ref) < DARK_LIMIT:
            new_dir = 1 if v > ref else -1
            if new_dir != direction:
                events.append(i)
                direction = new_dir
            ref = v
    return np.array(events, dtype=int)


def max_flashes_per_second(luma: np.ndarray, fps: float) -> float:
    """아무 1초 구간에서 깜빡임(전이 2번 = 1번) 수의 최댓값."""
    ev = count_transitions(luma)
    if len(ev) == 0:
        return 0.0
    win = int(round(fps))
    best = max(int(np.sum((ev >= s) & (ev < s + win))) for s in range(0, len(luma)))
    return best / 2.0
