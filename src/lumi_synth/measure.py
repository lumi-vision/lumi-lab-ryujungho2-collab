"""휘도 깜빡임 판정 규칙 (ITU-R BT.1702 일반 깜빡임) — 프로젝트의 유일한 '정답 규칙'.

오라클(데이터 합성), GAN(학습 신호의 기준), 데이터셋 라벨이 모두 이 파일 하나로 판정한다.
규칙의 수식 정의와 설계 결정의 근거는 docs/RULES_SPEC.md 에 있다. 이 파일의 주석은 그 문서의
기호(L, s, A↑, A↓, c, W)를 그대로 쓴다.

판정 절차 (4단계)
  ① 밝기     L[t,p] = 200 · Y(sRGB→선형)        픽셀 p, 프레임 t 의 물리 휘도 (cd/m²)
  ② 픽셀 전이 s[t,p] ∈ {-1, 0, +1}             한 픽셀이 '직전 전이 이후의 최고점/최저점'에서
                                               반대 방향으로 20 cd/m² 이상 움직였고, 어두운 쪽이
                                               160 cd/m² 미만이면 그 프레임에 전이 1번(부호 = 방향)
  ③ 화면 전이 e[t] ∈ {-1, 0, +1}               최근 c 프레임 동안 같은 방향으로 전이한 픽셀이
                                               화면의 25%를 넘으면 그 방향의 '화면 후보'.
                                               화면 상태가 바뀌는 후보만 화면 전이로 센다(상태 기계).
  ④ 위험      아무 1초(W = fps 프레임) 안의 화면 전이가 7번 이상(= 깜빡임 3번 초과)이면 위험

설계 결정 (근거는 RULES_SPEC.md §4)
  D1 같은 방향 연속은 1번: 화면 상태가 이미 '밝아짐'이면 또 밝아져도 세지 않는다.
     → 한 번의 변화가 두 프레임에 나뉘어 그려져도(위 절반 → 다음 프레임 아래 절반) 1번.
       (2026-11-09 이전 구현은 이것을 2번으로 세어 초당 1.88번(안전)을 4번(위험)으로 판정했다.)
  D2 면적은 방향별로 잰다: 밝아진 면적과 어두워진 면적을 더하지 않는다.
     → 초기화 전 판정기(같은 정의)가 pse-test-media ITU 기대값 56/56과 일치했다.
       한계: 15% 밝아짐 + 15% 어두워짐이 동시에 일어나면 위험으로 보지 않는다.
  D3 동시성 창 c = max(1, round(fps/15)) 프레임 (30fps에서 2프레임 ≈ 67ms):
     두 프레임에 걸친 하나의 변화를 '동시'로 본다. (20% + 20%로 나뉜 변화도 40%로 인정)
  D4 한 프레임에 두 방향이 모두 25%를 넘으면(체커보드 반전) 화면 전이 1번으로 센다.

이 판정은 '표준 문턱을 넘느냐'이며 임상적 발작 위험 판단이 아니다.
"""
from __future__ import annotations

import numpy as np

# ---- 규칙 상수 (ITU-R BT.1702 일반 깜빡임) -------------------------------------------
WHITE_CD = 200.0          # SDR 흰색(상대 휘도 1.0)의 물리 휘도 (cd/m²) — 가정 A1
DELTA_CD = 20.0           # 전이로 셀 최소 휘도 변화 (cd/m²), 경계 포함(≥)
DARK_LIMIT_CD = 160.0     # 어두운 쪽이 이 값 '미만'일 때만 전이 (cd/m²)
AREA_LIMIT = 0.25         # 화면 전이 최소 면적, 경계 제외(>): 정확히 25%는 미달
MAX_FLASHES_PER_SEC = 3   # 깜빡임이 이 값을 '초과'하면 위험 (= 1초 화면 전이 7번 이상)


def concurrency_frames(fps: float) -> int:
    """D3: '동시'로 볼 프레임 수 c. 30fps → 2, 24fps → 2, 60fps → 4, 15fps 이하 → 1."""
    return max(1, int(round(fps / 15.0)))


# ---- ① 밝기 ------------------------------------------------------------------------------
def luminance_cd(frames_bgr: np.ndarray) -> np.ndarray:
    """uint8 BGR 프레임(…×H×W×3)을 픽셀별 물리 휘도 L (cd/m², float32)로 바꾼다.

    가정 A1: 화면은 sRGB 전달 함수, 흰색 200 cd/m² (SDR). 영상의 BT.709/BT.1886 차이는 무시한다.
    """
    c = frames_bgr.astype(np.float32) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    b, g, r = lin[..., 0], lin[..., 1], lin[..., 2]
    return WHITE_CD * (0.2126 * r + 0.7152 * g + 0.0722 * b)


# ---- ② 픽셀 전이 -----------------------------------------------------------------------
def pixel_transitions(lum: np.ndarray) -> np.ndarray:
    """픽셀별 휘도 L(T×H×W) → 부호 있는 픽셀 전이 s(T×H×W, int8: +1 밝아짐, -1 어두워짐, 0 없음).

    각 픽셀은 기준점 ref(직전 전이 이후의 극값)와 방향 dir을 기억한다.
      - 같은 방향으로 더 가면 기준점을 끝점으로 옮긴다(극값 추적).
      - ref에서 반대 방향으로 |ΔL| ≥ 20 이고 min(L, ref) < 160 이면 전이 → 방향을 바꾸고 ref = 현재값.
    그래서 여러 프레임에 걸친 느린 변화도 '한 번'으로 센다(프레임 간 차이만 보는 방식과 다름).
    """
    ref = lum[0].copy()
    direction = np.zeros(lum.shape[1:], dtype=np.int8)  # +1 밝아지는 중, -1 어두워지는 중, 0 아직 없음
    out = np.zeros(lum.shape, dtype=np.int8)
    for t in range(1, len(lum)):
        v = lum[t]
        extend = ((direction == 1) & (v > ref)) | ((direction == -1) & (v < ref))
        ref = np.where(extend, v, ref)
        diff = v - ref
        big = (np.abs(diff) >= DELTA_CD) & (np.minimum(v, ref) < DARK_LIMIT_CD) & ~extend
        new_dir = np.sign(diff).astype(np.int8)
        trans = big & (new_dir != direction)
        out[t] = np.where(trans, new_dir, 0)
        direction = np.where(trans, new_dir, direction)
        ref = np.where(big, v, ref)
    return out


# ---- ③ 화면 전이 -----------------------------------------------------------------------
def directional_areas(signed: np.ndarray, c: int) -> tuple[np.ndarray, np.ndarray]:
    """D2·D3: 프레임 t 에서 최근 c 프레임 [t-c+1, t] 동안 밝아진/어두워진 픽셀의 면적 비율 (A↑, A↓).

    한 픽셀이 창 안에서 여러 번 전이해도 '그 방향으로 전이한 픽셀'로 1번만 센다(합집합).
    """
    t_len = signed.shape[0]
    up_px = np.zeros(signed.shape[1:], dtype=np.int32)
    dn_px = np.zeros(signed.shape[1:], dtype=np.int32)
    a_up = np.zeros(t_len, dtype=np.float64)
    a_dn = np.zeros(t_len, dtype=np.float64)
    n_px = float(np.prod(signed.shape[1:]))
    for t in range(t_len):                       # 이동 창 합계: 들어오는 프레임 더하고 나가는 프레임 빼기
        up_px += signed[t] == 1
        dn_px += signed[t] == -1
        if t - c >= 0:
            up_px -= signed[t - c] == 1
            dn_px -= signed[t - c] == -1
        a_up[t] = np.count_nonzero(up_px) / n_px
        a_dn[t] = np.count_nonzero(dn_px) / n_px
    return a_up, a_dn


def screen_transitions(a_up: np.ndarray, a_dn: np.ndarray, signed: np.ndarray) -> np.ndarray:
    """D1·D4: 방향별 면적 → 화면 전이 e[t] (+1/-1/0). 화면 상태가 바뀔 때만 센다.

    후보 조건: 방향 d 의 면적 A_d(t) > 25% 이고, **프레임 t 에 그 방향의 새 픽셀 전이가 실제로 있을 것**.
      (동시성 창 c 때문에 이미 지나간 전이가 다음 프레임에도 면적에 남는다. '새 전이' 조건이 없으면
       체커보드 5 Hz 반전을 10번으로 두 번 세는 것을 2026-11-09 시험에서 확인했다.)
    state = 마지막으로 센 화면 전이의 방향(처음 0).
      - 한 방향만 후보: 그 방향이 state와 다르면 전이. 같으면 '이어지는 같은 변화'라 세지 않음(D1).
      - 두 방향 모두 후보(D4): state의 반대 방향으로 전이 1번(state가 0이면 더 넓은 쪽).
    """
    e = np.zeros(len(a_up), dtype=np.int8)
    state = 0
    for t in range(len(a_up)):
        up = a_up[t] > AREA_LIMIT and bool((signed[t] == 1).any())
        dn = a_dn[t] > AREA_LIMIT and bool((signed[t] == -1).any())
        if up and dn:
            d = -state if state != 0 else (1 if a_up[t] >= a_dn[t] else -1)
        elif up:
            d = 1
        elif dn:
            d = -1
        else:
            continue
        if d != state:
            e[t] = d
            state = d
    return e


# ---- ④ 위험 ------------------------------------------------------------------------------
def window_counts(events: np.ndarray, fps: float) -> np.ndarray:
    """각 프레임 s에서 시작하는 1초 창 [s, s+W) 안의 화면 전이 수 (W = round(fps))."""
    w = max(1, int(round(fps)))
    csum = np.concatenate([[0], np.cumsum(events != 0)])
    t_len = len(events)
    return csum[np.minimum(np.arange(t_len) + w, t_len)] - csum[:t_len]


def max_flashes_per_second(events: np.ndarray, n_frames: int, fps: float) -> float:
    """(호환용) 화면 전이 프레임 번호 목록 → 아무 1초의 최대 깜빡임(전이 2번 = 1번)."""
    e = np.zeros(n_frames, dtype=np.int8)
    e[np.asarray(events, dtype=int)] = 1
    return float(window_counts(e, fps).max()) / 2.0 if n_frames else 0.0


def analyze(frames_bgr: np.ndarray, fps: float) -> dict:
    """영상(T×H×W×3 uint8 BGR)을 규칙대로 판정한다. 오라클·GAN·데이터셋 라벨이 모두 이 함수를 쓴다.

    돌려주는 값
    - mean_cd      프레임별 화면 평균 휘도 (cd/m²)
    - area         프레임별 max(A↑, A↓): 같은 방향으로 동시에 전이한 최대 면적 비율
    - area_up/down 프레임별 A↑, A↓
    - transitions  화면 전이 e[t] (+1/-1/0)
    - events       화면 전이가 일어난 프레임 번호
    - max_flashes  아무 1초의 최대 깜빡임 수 (화면 전이 수 / 2)
    - peak_area    가장 넓은 같은 방향 동시 전이 면적
    - dangerous    위험 여부 (max_flashes > 3)
    - hazard_frames 위험한 1초 창에 속한 프레임 (bool, T) — '언제' 정답
    - hazard_mask  위험한 창 안에서 화면 전이를 만든 픽셀 (bool, H×W) — '어디' 정답
    """
    if frames_bgr.ndim != 4 or frames_bgr.shape[-1] != 3 or frames_bgr.dtype != np.uint8:
        raise ValueError(f"입력은 (T,H,W,3) uint8 BGR이어야 합니다: shape={frames_bgr.shape}, dtype={frames_bgr.dtype}")
    if len(frames_bgr) < 2:
        raise ValueError("프레임이 2장 이상이어야 전이를 셀 수 있습니다.")
    if not (np.isfinite(fps) and fps > 0):
        raise ValueError(f"fps는 양수여야 합니다: {fps}")

    lum = luminance_cd(frames_bgr)
    signed = pixel_transitions(lum)
    c = concurrency_frames(fps)
    a_up, a_dn = directional_areas(signed, c)
    e = screen_transitions(a_up, a_dn, signed)
    counts = window_counts(e, fps)
    w = max(1, int(round(fps)))
    hazardous_starts = np.flatnonzero(counts >= 2 * MAX_FLASHES_PER_SEC + 1)

    hazard_frames = np.zeros(len(e), dtype=bool)
    for s in hazardous_starts:
        hazard_frames[s:s + w] = True
    # '어디': 위험 창 안의 화면 전이 프레임에서, 그 방향으로 (동시성 창 c 안에서) 전이한 픽셀
    hazard_mask = np.zeros(lum.shape[1:], dtype=bool)
    for t in np.flatnonzero((e != 0) & hazard_frames):
        lo = max(0, t - c + 1)
        hazard_mask |= (signed[lo:t + 1] == e[t]).any(axis=0)

    area = np.maximum(a_up, a_dn)
    flashes = float(counts.max()) / 2.0
    return {
        "mean_cd": lum.mean(axis=(1, 2)),
        "area": area,
        "area_up": a_up,
        "area_down": a_dn,
        "transitions": e,
        "events": np.flatnonzero(e),
        "max_flashes": flashes,
        "peak_area": float(area.max()),
        "dangerous": flashes > MAX_FLASHES_PER_SEC,
        "hazard_frames": hazard_frames,
        "hazard_mask": hazard_mask,
    }


def criteria_report(frames_bgr: np.ndarray, fps: float) -> list[dict]:
    """ITU 위험 조건을 하나씩 따로 재서, 어느 조건이 넘었고 어느 조건이 무너졌는지 표로 돌려준다.

    위험 = 1(밝기) AND 2(횟수) AND 3(면적). 마지막 줄은 analyze()의 최종 판정과 같다.
    1. 밝기: 어두운 쪽 < 160인 픽셀 중 가장 큰 밝기 변화폭(최대 − 최소) ≥ 20 cd/m²
    2. 횟수: 픽셀 하나만 놓고 아무 1초에 깜빡임 > 3번 (면적과 무관하게 잼)
    3. 면적: 같은 방향으로 동시에(창 c) 전이한 픽셀이 화면의 > 25%
    """
    lum = luminance_cd(frames_bgr)
    lo, hi = lum.min(axis=0), lum.max(axis=0)
    swing = hi - lo
    eligible = lo < DARK_LIMIT_CD
    if eligible.any():
        k = np.unravel_index(np.argmax(np.where(eligible, swing, -1)), swing.shape)
        max_swing, dark = float(swing[k]), float(lo[k])
    else:
        max_swing, dark = 0.0, float(lo.min())

    signed = pixel_transitions(lum)
    ever = (signed != 0).any(axis=0)
    pixel_flashes = 0.0
    if ever.any():
        sub = (signed[:, ever] != 0).astype(np.uint16)
        csum = np.vstack([np.zeros((1, sub.shape[1]), np.uint16), np.cumsum(sub, axis=0, dtype=np.uint16)])
        win = int(round(fps))
        t = len(lum)
        window = csum[np.minimum(np.arange(t) + win, t)] - csum[:t]
        pixel_flashes = float(window.max()) / 2.0

    r = analyze(frames_bgr, fps)
    lum_ok = max_swing >= DELTA_CD and dark < DARK_LIMIT_CD
    return [
        {"조건": "1. 밝기 변화폭", "측정값": f"{max_swing:.1f} cd/m² (어두운 쪽 {dark:.1f})",
         "기준": f"≥ {DELTA_CD:g} cd/m², 어두운 쪽 < {DARK_LIMIT_CD:g}", "넘음": lum_ok},
        {"조건": "2. 깜빡임 횟수 (픽셀 기준)", "측정값": f"1초 최대 {pixel_flashes:g}번",
         "기준": f"> {MAX_FLASHES_PER_SEC}번", "넘음": pixel_flashes > MAX_FLASHES_PER_SEC},
        {"조건": "3. 같은 방향 동시 전이 면적", "측정값": f"최대 {r['peak_area']:.2%}",
         "기준": f"> {AREA_LIMIT:.0%}", "넘음": r["peak_area"] > AREA_LIMIT},
        {"조건": "최종: 1+2+3 모두 (화면 기준 깜빡임)", "측정값": f"1초 최대 {r['max_flashes']:g}번",
         "기준": f"> {MAX_FLASHES_PER_SEC}번이면 위험", "넘음": r["dangerous"]},
    ]
