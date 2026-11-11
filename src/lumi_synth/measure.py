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
     → 초기화 전 판정기(같은 방향 면적)는 pse-test-media 기본 2세트 56/56 기록이 있다. 현재 판정기는 같은 2세트
       53/56 (checks/rules/README.md, 불일치 3개는 픽셀 전이의 누적 계수 때문 — D2 와 무관).
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


# ---- 선택: Jordan & Vanderheiden 2024 시간창 조항 (기본은 꺼짐) ------------------------------------
# 출처: J. B. Jordan, G. C. Vanderheiden, ACM TACCESS 17(3), 2024, doi:10.1145/3694790 와
#       github.com/traceRERC/pseGuidelines (2024, "Proposed", 값은 "working value") — 원문 확인 2026-11-11
#   qualifying duration   "The time over which a single transition may take place of 66 ms or less."
#   sufficiently synchronized "When transitions are in the same direction and occur within 20 ms of each other."
#   fast flashes          every second transition separated by 15 ms or less → 같은 방향 전이를 합쳐
#                         "the sequence of fast flashes is counted as two transitions total"
# 프레임으로 바꾸는 방법은 구현 선택이다. 66 ms 는 30 fps 에서 2프레임(66.7 ms)과 거의 같아 해석이 갈린다:
#   'strict'  : k/fps ≤ 66 ms 인 프레임 간격 k 만 (30 fps → 1프레임)
#   'lenient' : k = round(0.066·fps) 프레임까지 (30 fps → 2프레임, 66.7 ms)
JV24_QUALIFYING_MS = 66.0
JV24_SYNC_MS = 20.0
JV24_FAST_MS = 15.0


def qualifying_frames(fps: float, mode: str = "strict") -> int:
    """66 ms 안에 들어가는 최대 프레임 간격 K (≥ 1)."""
    x = JV24_QUALIFYING_MS / 1000.0 * fps
    k = int(np.floor(x + 1e-9)) if mode == "strict" else int(round(x))
    return max(1, k)


def sync_frames(fps: float, ms: float = JV24_SYNC_MS) -> int:
    """'동시'(ms 이내)로 볼 프레임 수 c: 간격 d 프레임이 d/fps ≤ ms 이면 동시. 30 fps → 1(같은 프레임만), 60 fps → 2."""
    return max(1, int(np.floor(ms / 1000.0 * fps + 1e-9)) + 1)


def pixel_transitions_windowed(lum: np.ndarray, k: int) -> np.ndarray:
    """JV24 '단일 전이 허용 시간': 최근 k 프레임 안에서 |ΔL| ≥ 20 (어두운 쪽 < 160) 인 변화만 전이로 센다.

    pixel_transitions 와 달리 '직전 극값'에서 재지 않는다 → k 프레임보다 길게 걸친 느린 변화는 전이가 아니다.
    방향 기억(같은 방향 연속 변화는 1번)은 같다.
      위로: L_t − min(L_{t−k..t−1}) ≥ 20 이고 그 최솟값 < 160
      아래로: max(L_{t−k..t−1}) − L_t ≥ 20 이고 L_t < 160
      둘 다면 변화가 큰 쪽.
    """
    t_len = len(lum)
    direction = np.zeros(lum.shape[1:], dtype=np.int8)
    out = np.zeros(lum.shape, dtype=np.int8)
    for t in range(1, t_len):
        past = lum[max(0, t - k):t]
        lo, hi, v = past.min(axis=0), past.max(axis=0), lum[t]
        up_mag, dn_mag = v - lo, hi - v
        up = (up_mag >= DELTA_CD) & (lo < DARK_LIMIT_CD)
        dn = (dn_mag >= DELTA_CD) & (v < DARK_LIMIT_CD)
        d = np.where(up & dn, np.where(up_mag >= dn_mag, 1, -1), np.where(up, 1, np.where(dn, -1, 0))).astype(np.int8)
        trans = (d != 0) & (d != direction)
        out[t] = np.where(trans, d, 0)
        direction = np.where(trans, d, direction)
    return out


def merge_fast_flashes(e: np.ndarray, fps: float, ms: float = JV24_FAST_MS) -> np.ndarray:
    """JV24 '빠른 점멸': 한 칸 건너 전이(같은 방향) 간격이 ms 이하인 연속 구간은 처음과 끝 두 전이로만 센다.

    30 fps 이하에서는 전이 간격이 33 ms 이상이라 바뀌는 것이 없다(고주사율 영상용)."""
    ev = np.flatnonzero(e)
    if len(ev) < 3:
        return e
    out = e.copy()
    gap = ms / 1000.0 * fps
    i = 0
    while i < len(ev) - 2:
        j = i
        while j + 2 < len(ev) and ev[j + 2] - ev[j] <= gap + 1e-9:
            j += 1
        if j > i:                          # ev[i] .. ev[j+1] 이 빠른 점멸 구간
            for m in ev[i + 1:j + 1]:
                out[m] = 0
            i = j + 2
        else:
            i += 1
    return out


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


def _check_input(frames_bgr: np.ndarray, fps: float) -> None:
    """analyze·analyze_pattern 공통 입력 검사."""
    if frames_bgr.ndim != 4 or frames_bgr.shape[-1] != 3 or frames_bgr.dtype != np.uint8:
        raise ValueError(f"입력은 (T,H,W,3) uint8 BGR이어야 합니다: shape={frames_bgr.shape}, dtype={frames_bgr.dtype}")
    if len(frames_bgr) < 2:
        raise ValueError("프레임이 2장 이상이어야 전이를 셀 수 있습니다.")
    if not (np.isfinite(fps) and fps > 0):
        raise ValueError(f"fps는 양수여야 합니다: {fps}")


def analyze_jv24(frames_bgr: np.ndarray, fps: float, qualifying: str | None = "strict", sync: bool = True,
                 fast: bool = True, lum: np.ndarray | None = None) -> dict:
    """(선택) analyze 에 Jordan & Vanderheiden 2024 시간창 조항을 켠 판정. 기본 판정(analyze)은 바뀌지 않는다.

    qualifying: 'strict' | 'lenient' | None(끔) — 단일 전이 허용 시간 66 ms
    sync: 영역 동기화 창 20 ms (끄면 기본 D3 창) · fast: 빠른 점멸 병합 15 ms
    """
    _check_input(frames_bgr, fps)
    if lum is None:
        lum = luminance_cd(frames_bgr)
    signed = (pixel_transitions_windowed(lum, qualifying_frames(fps, qualifying)) if qualifying
              else pixel_transitions(lum))
    out = flash_from_transitions(signed, fps, c=sync_frames(fps) if sync else None,
                                 fast_ms=JV24_FAST_MS if fast else None)
    out["mean_cd"] = lum.mean(axis=(1, 2))
    return out


def analyze(frames_bgr: np.ndarray, fps: float, lum: np.ndarray | None = None,
            signed: np.ndarray | None = None) -> dict:
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
    lum, signed 는 이미 계산한 ①·② 결과를 재사용할 때만 넘긴다(analyze_all).
    """
    _check_input(frames_bgr, fps)
    if lum is None:
        lum = luminance_cd(frames_bgr)
    if signed is None:
        signed = pixel_transitions(lum)
    out = flash_from_transitions(signed, fps)
    out["mean_cd"] = lum.mean(axis=(1, 2))
    return out


def flash_from_transitions(signed: np.ndarray, fps: float, c: int | None = None,
                           fast_ms: float | None = None) -> dict:
    """③·④ 공통: 픽셀 전이 s[t,p](+1/-1/0) → 화면 전이·1초 창·위험·정답.

    휘도 깜빡임(analyze)과 적색 깜빡임(analyze_red)이 같은 면적·횟수 규칙을 쓰므로 이 함수를 함께 쓴다
    (ITU-R BT.1702-3: 적색도 "a transition to or from a saturated red is also potentially harmful",
     면적 > 25%, 1초에 > 3번은 같은 문단의 규칙).
    c: 동시성 창(프레임). None 이면 기본 D3(concurrency_frames). fast_ms: 주면 JV24 빠른 점멸 병합.
    """
    if c is None:
        c = concurrency_frames(fps)
    a_up, a_dn = directional_areas(signed, c)
    e = screen_transitions(a_up, a_dn, signed)
    if fast_ms is not None:
        e = merge_fast_flashes(e, fps, fast_ms)
    counts = window_counts(e, fps)
    w = max(1, int(round(fps)))
    hazardous_starts = np.flatnonzero(counts >= 2 * MAX_FLASHES_PER_SEC + 1)

    hazard_frames = np.zeros(len(e), dtype=bool)
    for s in hazardous_starts:
        hazard_frames[s:s + w] = True
    # '어디': 위험 창 안의 화면 전이 프레임에서, 그 방향으로 (동시성 창 c 안에서) 전이한 픽셀
    hazard_mask = np.zeros(signed.shape[1:], dtype=bool)
    for t in np.flatnonzero((e != 0) & hazard_frames):
        lo = max(0, t - c + 1)
        hazard_mask |= (signed[lo:t + 1] == e[t]).any(axis=0)

    area = np.maximum(a_up, a_dn)
    flashes = float(counts.max()) / 2.0
    return {
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


# =============================================================================================
# 적색 깜빡임 판정 — WCAG 2.2 (ISO 9241-391 인용) 의 적색 정의 + ITU-R BT.1702-3 의 면적·횟수 규칙
# 근거와 출처 등급: docs/STANDARDS.md §2
#
#   ⓡ① 상태      선형 RGB 로 포화 빨강 여부  s = R/(R+G+B) ≥ 0.8              (원문 WCAG 2.2, 선형값 — pse 색표로 검산)
#                 색도  X,Y,Z (sRGB D65 행렬) → u′ = 4X/(X+15Y+3Z), v′ = 9Y/(X+15Y+3Z)  (원문 WCAG 2.2 Understanding)
#                 검정(R+G+B = 0)은 포화 빨강 아님, 색도는 D65 백색점 (0.1978, 0.4683)  (구현 선택: W3C 이슈 #4285 미해결)
#   ⓡ② 적색 전이  픽셀마다 바로 앞 프레임의 상태와 비교해
#                 Δu′v′ > 0.2 이고, 기준이 빨강이 아니고 지금이 포화 빨강이면 +1(빨강으로),
#                 기준이 포화 빨강이고 지금이 아니면 −1(빨강에서). 방향이 바뀔 때만 1번.   (원문: "one transition is either
#                 to or from a state with a value R/(R + G + B) that is greater than or equal to 0.8, and the difference
#                 between states is more than 0.2". 바로 앞 상태와 비교하므로 여러 프레임에 걸쳐 서서히 바뀌는
#                 색은 세지 않음 — pse-test-media 설계와 같은 해석. 2026-11-10 이전에는 '직전 전이 때의 상태'와
#                 비교해 s014_multi 를 과잉 검출했음)
#                 빨강 ↔ 빨강(둘 다 포화)은 세지 않음 (구현 선택: 방향을 정할 수 없음)
#   ⓡ③ 이후      휘도와 같은 ③ 방향별 동시 면적 > 25%, ④ 1초에 화면 전이 ≥ 7번이면 위험  (원문 ITU-R BT.1702-3)
#   경계:         Δu′v′ 는 '초과'(WCAG 2.2 "more than 0.2"). trace24 는 '이상'(≥ 0.2) — pse 시험 패턴(0.195/0.205)은 둘 다 같은 결과
# =============================================================================================
RED_SAT_RATIO = 0.8                 # 포화 빨강: 선형 R/(R+G+B) 이상(≥)
RED_DELTA_UV = 0.2                  # 색 차이: Δu′v′ 초과(>)
D65_UV = (0.1978, 0.4683)           # 검정의 색도로 쓰는 D65 백색점
RED_CHANGE_EPS = 0.02               # 이보다 작은 색도 변화는 '상태 변화'로 보지 않음 (깜빡임 쌍 짓기용) — 구현 선택
_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                 [0.2126729, 0.7151522, 0.0721750],
                 [0.0193339, 0.1191920, 0.9503041]], dtype=np.float64)


def red_state(frames_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ⓡ① (…×3 uint8 BGR) → (포화 빨강 bool, u′, v′)."""
    c = frames_bgr.astype(np.float32) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float64)
    b, g, r = lin[..., 0], lin[..., 1], lin[..., 2]
    total = r + g + b
    sat = (total > 0) & (r >= RED_SAT_RATIO * total)
    x = _XYZ[0, 0] * r + _XYZ[0, 1] * g + _XYZ[0, 2] * b
    y = _XYZ[1, 0] * r + _XYZ[1, 1] * g + _XYZ[1, 2] * b
    z = _XYZ[2, 0] * r + _XYZ[2, 1] * g + _XYZ[2, 2] * b
    d = x + 15.0 * y + 3.0 * z
    ok = d > 0
    u = np.where(ok, 4.0 * x / np.where(ok, d, 1.0), D65_UV[0])
    v = np.where(ok, 9.0 * y / np.where(ok, d, 1.0), D65_UV[1])
    return sat, u.astype(np.float32), v.astype(np.float32)


def red_transitions(frames_bgr: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """ⓡ② 영상(T×H×W×3) → (적색 전이 s[t,p] ∈ {−1,0,+1}, 전이가 일어난 곳의 최대 Δu′v′, 프레임별 포화 빨강 면적).

    비교 기준은 '바로 앞 프레임의 상태'다 (WCAG "the difference between states"). 메모리를 아끼려고 색 상태를
    프레임마다 계산한다(1080p 에서 전체를 한꺼번에 float64 로 바꾸면 수 GB).
    2026-11-10: 처음에는 '마지막 적색 전이 때의 상태'와 비교했는데, pse-test-media s014_multi(청록 → 포화 빨강에
    못 미치는 빨강 → 포화 빨강, 마지막 변화 Δ0.001)를 청록과 비교해 전이로 세는 과잉 검출이 났다. pse 설계와
    WCAG 문구 모두 '상태 사이의 차이'라서 바로 앞 상태와 비교하도록 바꿨다.
    한계: 여러 프레임에 걸쳐 서서히 바뀌는 색(한 프레임 변화 ≤ 0.2)은 세지 않는다.
    """
    t_len = len(frames_bgr)
    prev_s, prev_u, prev_v = red_state(frames_bgr[0])
    sat_area = np.zeros(t_len)
    sat_area[0] = prev_s.mean()
    shape = prev_s.shape
    event = np.zeros((t_len,) + shape, dtype=bool)      # 색 상태가 바뀐 프레임 (Δu′v′ ≥ RED_CHANGE_EPS)
    qual = np.zeros((t_len,) + shape, dtype=np.int8)    # 그 변화가 적색 전이 조건을 넘으면 방향(+1/−1), 아니면 0
    duv_at = np.zeros((t_len,) + shape, dtype=np.float32)
    for t in range(1, t_len):
        s, u, v = red_state(frames_bgr[t])
        sat_area[t] = s.mean()
        duv = np.hypot(u - prev_u, v - prev_v)
        new_dir = np.where(s & ~prev_s, 1, np.where(~s & prev_s, -1, 0)).astype(np.int8)
        event[t] = (duv >= RED_CHANGE_EPS) | (new_dir != 0)
        qual[t] = np.where((duv > RED_DELTA_UV) & (new_dir != 0), new_dir, 0)
        duv_at[t] = duv
        prev_s, prev_u, prev_v = s, u, v

    # 깜빡임 = 반대 방향 변화 한 쌍. 픽셀마다 변화를 시간 순서로 (1,2), (3,4), … 짝짓고,
    # 두 변화가 모두 조건을 넘고 방향이 반대인 쌍만 센다. 짝이 없는 마지막 변화는 조건을 넘으면 센다.
    k = np.cumsum(event, axis=0, dtype=np.int32)          # 변화 번호 (1부터)
    t_idx = np.broadcast_to(np.arange(t_len, dtype=np.int32).reshape((t_len,) + (1,) * len(shape)), (t_len,) + shape)
    ev_t = np.where(event, t_idx, -1)
    prev_ev = np.maximum.accumulate(ev_t, axis=0)          # t 까지(포함) 마지막 변화 프레임
    prev_ev = np.concatenate([np.full((1,) + shape, -1, np.int32), prev_ev[:-1]])   # t 이전 마지막 변화
    ev_t_rev = np.where(event, t_idx, t_len)[::-1]
    next_ev = np.minimum.accumulate(ev_t_rev, axis=0)[::-1]
    next_ev = np.concatenate([next_ev[1:], np.full((1,) + shape, t_len, np.int32)])  # t 이후 첫 변화
    q_pad = np.concatenate([qual, np.zeros((1,) + shape, np.int8)])
    first_of_pair = (k % 2 == 1)
    partner = np.where(first_of_pair, next_ev, prev_ev)
    partner_q = _gather_time(q_pad, partner)
    unpaired_last = first_of_pair & (next_ev >= t_len)
    valid = event & (qual != 0) & (((partner_q == -qual) & (partner >= 0)) | unpaired_last)
    out = np.where(valid, qual, 0).astype(np.int8)
    max_duv = float(duv_at[valid].max()) if valid.any() else 0.0
    return out, max_duv, sat_area


def _gather_time(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """arr[(idx[t,p]), p] — 시간축 인덱스로 값 모으기 (범위 밖 인덱스는 마지막 칸 0)."""
    t_len = arr.shape[0] - 1
    i = np.where((idx >= 0) & (idx <= t_len), idx, t_len).astype(np.int64)
    return np.take_along_axis(arr, i, axis=0)


def analyze_red(frames_bgr: np.ndarray, fps: float) -> dict:
    """적색 깜빡임 판정. 돌려주는 값은 analyze() 와 같은 꼴 + 아래 두 값.

    - sat_area   프레임별 포화 빨강 픽셀 면적 비율
    - max_delta_uv 적색 전이가 일어난 픽셀의 최대 Δu′v′ (전이가 없으면 0)
    """
    _check_input(frames_bgr, fps)
    signed, max_duv, sat_area = red_transitions(frames_bgr)
    out = flash_from_transitions(signed, fps)
    out["sat_area"] = sat_area
    out["max_delta_uv"] = max_duv
    return out


# =============================================================================================
# 줄무늬(규칙 패턴) 판정 — ITU-R BT.1702-3 Annex 1 Guideline 2 + Attachment 1, 일부는 팀원 판정기 기준
# 수식 정의와 근거: docs/RULES_SPEC.md §8. 깜빡임 판정(analyze)과 독립이며, analyze_all() 이 둘을 합친다.
#
#   ⑤ 줄무늬 쌍     화면을 12방향(0°~165°, 15° 간격)으로 1픽셀 간격 직선을 그어 휘도 프로파일을 얻고,
#                  20 cd/m² 히스테리시스로 '밝은 줄·어두운 줄' 극값을 찾는다. 이웃 극값 사이(줄 하나)가
#                  (어두운 쪽 < 160) 이고 (공간 주파수 1~8 cpd) 이면 '유효한 줄'. 유효한 줄이 끊김 없이
#                  10개(밝은 5 + 어두운 5 = 5쌍) 이상 이어진 구간의 픽셀을 그 프레임의 '줄무늬 픽셀'로 표시.
#   ⑥ 움직임 3분류  줄무늬에 수직인 방향의 '위치 × 시간' 휘도 단면을 ±0.5초 창에서 2차원 푸리에 변환해
#                  (가) 시간 변화 에너지 비율 < 10%                → 정지(static)
#                  (나) 방향 지수 |DI| ≥ 0.5 (한 방향 성분에 몰림)  → 흐름(drifting)
#                  (다) 그 밖 (양방향 대칭 = 진동·반전·점멸)       → 반전(reversing)
#                  DI = (E_한방향 − E_반대방향) / (E_한방향 + E_반대방향)  (운동 에너지, Adelson & Bergen 1985)
#   ⑦ 면적          정지 > 40%, 반전 > 25%, 흐름은 제외(ITU·Ofcom 원문) 이면 그 프레임은 '줄무늬 후보'.
#   ⑧ 위험          아무 1초 창 안의 후보 프레임이 0.5초 이상(≥ ceil(W/2) 프레임)이면 위험.
#
# 출처 구분 (논문에 그대로 쓸 것 · 자세히: docs/PATTERN_LITERATURE.md, docs/STANDARDS.md §3)
#   표준 원문:  면적 정지 40% / 변화 25%, 한 방향 흐름 제외("obviously flow smoothly across, into, or out of
#               the screen in one direction"), 명암은 깜빡임과 같은 기준(ΔL ≥ 20, 어두운 쪽 < 160), 모든 방향
#               (ITC 2001 → Ofcom 2005·2017 → ITU-R BT.1702-3 Attachment 1)
#   합의 문헌:  1~8 cpd — EF 개정 합의문 Fisher·Wilkins 외 2026, Epilepsia 67:565, doi:10.1111/epi.18702
#               ("oscillating or static stripes of ~1-8 cycles per visual degree", Wilkins 1979 인용).
#               2026-11-10 이전 값 1~10 cpd 는 어떤 문헌에서도 출처를 찾지 못해 바꿈.
#               지속 0.5초 — EFA 2005·Wilkins 2005 (경험값, 실험 근거 없음을 저자가 명시)
#   실험 근거:  움직임 3분류의 필요성 — Binnie, Findlay & Wilkins 1985(환자 10명): 진동 15 Hz PPR 0.97,
#               정지 0.38, 한 방향 흐름 2.5~10°/s 0~0.04. 진동과 흐름은 한 점의 밝기 변화 주파수가 같아도
#               위험이 정반대라, 픽셀 밝기 변화만으로는 구분할 수 없음 → 방향 에너지로 구분
#   팀원 판정기를 따른 것: 5쌍 이상(원문은 'more than five' = 6쌍 이상, 1쌍 보수적), 시청 조건 56 cm·15인치
#               (원문 출처 J&V 2024 표 2는 15인치 4:3 — 여기서는 16:9로 계산), 12방향 15° 간격
#   LUMI 구현 선택 (근거 없음, 민감도 분석 대상): 정지 판정 시간 에너지 10%, 흐름 판정 |DI| 0.5,
#               극값 히스테리시스 20 cd/m²(팀원 5), 줄무늬 픽셀과 면적을 같은 구간에서 잼, 화면 전체 스캔
#   한계:       30 fps 에서 시간 주파수 15 Hz(나이퀴스트)는 방향을 알 수 없어 '반전'으로 분류됨(보수적).
#               진동 15 Hz 같은 Binnie 조건은 30 fps 영상으로 표현 자체가 안 됨.
# =============================================================================================
PATTERN_MIN_PAIRS = 5             # 밝은·어두운 줄 쌍이 이 수 '이상'(≥)이면 줄무늬 (ITU 는 > 5)
PATTERN_AREA_DYNAMIC = 0.25       # 반전(진동·점멸·명암 반전) 줄무늬 면적 한계, 경계 제외(>) — 원문
PATTERN_AREA_STATIC = 0.40        # 정지 줄무늬 면적 한계, 경계 제외(>) — 원문
PATTERN_DRIFT_EXEMPT = True       # 한 방향으로 매끄럽게 흐르는 줄무늬는 제외 — 원문 (False 면 25% 적용)
PATTERN_CPD = (1.0, 8.0)          # 위험 공간 주파수 (cycles per degree), 경계 포함 — EF 2026 합의
MOTION_STATIC_MAX = 0.10          # 시간 변화 에너지 비율이 이 값 미만이면 정지 — 구현 선택
MOTION_DRIFT_MIN_DI = 0.5         # 방향 지수 |DI| 가 이 값 이상이면 흐름 — 구현 선택
MOTION_ROWS = 64                  # 움직임 판정에 쓰는 스캔 직선 수 (계산량 제한)
STRIPE_EXTREMUM_TOL = 0.25        # 줄 극값을 옮길 최소 차이 (cd/m²) — 구현 선택. 비스듬한 스캔의 보간이 만드는
                                  # 소수점 차이를 따라 극값이 평평한 배경 안으로 밀려가는 것을 막음(2026-11-10).
                                  # 8비트 한 단계(배경 38 cd/m² 근처 약 0.8)보다 작게 둠
PATTERN_DYNAMIC_FRAC = MOTION_STATIC_MAX   # (호환용 옛 이름)
PATTERN_MIN_SEC = 0.5             # 1초 창 안의 줄무늬 후보 프레임이 이 시간 이상이면 위험
PATTERN_ANGLES = tuple(range(0, 180, 15))   # 스캔 방향 (도)
VIEW_DISTANCE_CM = 56.0           # 노트북 표준 시청 거리 (팀원 판정기와 같음)
DISPLAY_INCH = 15.0               # 화면 대각선 (16:9)


def pixels_per_degree(width: int, height: int, angle_deg: float = 0.0) -> float:
    """시청 조건(56 cm, 15인치 16:9)에서 방향 angle_deg 로 1도(시야각)에 들어가는 픽셀 수.

    가로·세로 시야각 = 2·atan(화면 크기 / (2·거리)). 방향별 값은 팀원 판정기와 같은 식
    1 / sqrt((cosθ/ppd_가로)² + (sinθ/ppd_세로)²).
    해상도가 낮으면 같은 화면에서 1도당 픽셀이 적다: 128×72 → 약 3.9 px/도 → 표현 가능한 최고 주파수는
    2픽셀 주기 = 약 1.9 cpd. 즉 저해상도 트랙은 1~1.9 cpd 줄무늬만 만들고 잴 수 있다.
    전제: 프레임 = 16:9 화면 전체. 16:9 가 아닌 영상은 data.letterbox 로 여백을 넣어 16:9 캔버스로 만든 뒤 판정한다
    (비율을 무시하고 늘리면 가로·세로 1도당 픽셀이 달라져 cpd 가 틀린다 — 팀원 판정기도 같은 전제, 2026-11-09 점검).
    """
    diag = DISPLAY_INCH * 2.54
    k = np.hypot(16.0, 9.0)
    fov_w = 2.0 * np.degrees(np.arctan(diag * 16.0 / k / (2.0 * VIEW_DISTANCE_CM)))
    fov_h = 2.0 * np.degrees(np.arctan(diag * 9.0 / k / (2.0 * VIEW_DISTANCE_CM)))
    a = np.radians(angle_deg)
    return float(1.0 / np.hypot(np.cos(a) / (width / fov_w), np.sin(a) / (height / fov_h)))


def _scan_maps(h: int, w: int, angle_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """방향 angle_deg 의 스캔 직선들을 '행'으로 늘어놓은 캔버스 → 원본 좌표 (map_x, map_y, valid).

    캔버스 (i, j): 원본 좌표 = 화면 중심 + (j − 열 중심)·d + (i − 행 중심)·n,
    d = (cosθ, sinθ) 는 스캔 방향(1칸 = 원본 1픽셀), n = (−sinθ, cosθ) 는 이웃 직선 방향(1픽셀 간격).
    0°·90° 는 좌표가 정수라 보간 없이 원본 픽셀을 그대로 읽는다.
    """
    a = np.radians(angle_deg)
    ca, sa = float(np.cos(a)), float(np.sin(a))
    ca, sa = (0.0 if abs(ca) < 1e-12 else ca), (0.0 if abs(sa) < 1e-12 else sa)
    wc = int(np.ceil(abs(w * ca) + abs(h * sa)))
    hc = int(np.ceil(abs(w * sa) + abs(h * ca)))
    jj, ii = np.meshgrid(np.arange(wc, dtype=np.float64) - (wc - 1) / 2.0,
                         np.arange(hc, dtype=np.float64) - (hc - 1) / 2.0)
    mx = (w - 1) / 2.0 + jj * ca - ii * sa
    my = (h - 1) / 2.0 + jj * sa + ii * ca
    for m in (mx, my):                                   # 부동소수 오차로 0°·90° 정수 좌표가 흔들리지 않게
        r = np.round(m)
        m[np.abs(m - r) < 1e-6] = r[np.abs(m - r) < 1e-6]
    valid = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
    return mx, my, valid


def _sample(lum: np.ndarray, mx: np.ndarray, my: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """휘도(T×H×W)를 캔버스 좌표에서 쌍선형 보간으로 읽는다. 화면 밖은 NaN (직선이 끊긴 곳)."""
    h, w = lum.shape[1:]
    x0 = np.clip(np.floor(mx).astype(int), 0, w - 1)
    y0 = np.clip(np.floor(my).astype(int), 0, h - 1)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (mx - x0).astype(np.float32), (my - y0).astype(np.float32)
    out = (lum[:, y0, x0] * (1 - fx) * (1 - fy) + lum[:, y0, x1] * fx * (1 - fy)
           + lum[:, y1, x0] * (1 - fx) * fy + lum[:, y1, x1] * fx * fy)
    out[:, ~valid] = np.nan
    return out


def stripe_runs(profiles: np.ndarray, ppd: float, return_cpd: bool = False):
    """⑤ 휘도 프로파일 여러 줄(N×M, NaN = 끊김) → 줄무늬로 판정된 표본 (N×M bool).

    return_cpd=True 이면 (bool, 구간의 평균 cpd N×M float) 을 돌려준다(줄무늬 아닌 곳은 0).

    극값 찾기 (히스테리시스 20 cd/m², 시간축 픽셀 전이 ②와 같은 방식):
      현재 극값에서 같은 방향으로 STRIPE_EXTREMUM_TOL 넘게 더 가면 극값을 옮기고(평평한 줄은 처음 닿은 곳),
      반대 방향으로 20 이상 물러나면 현재 극값을 '확정'하고 방향을 바꾼다.
    극값이 확정될 때마다 직전 극값과의 사이 = 줄 하나(반 주기 h 표본)를 검사한다:
      유효 ⇔ min(두 극값) < 160 이고 ppd / (2h) ≥ 1 (cpd 하한만).
    상한(8 cpd)은 여기서 자르지 않는다: 비스듬한 스캔 방향에서는 줄 간격이 1/cos(Δθ) 배로 넓어 보여
    실제보다 낮은 cpd 로 재진다. 그래서 stripe_masks 가 모든 방향의 cpd 중 최댓값(= 줄에 수직인 방향의 값)으로
    상한을 판정한다 (2026-11-10: 이전 구현은 방향마다 상한을 잘라서, 10 cpd 줄무늬도 비스듬한 방향에서 통과했음).
    유효한 줄이 k 개 이어지면 극값은 k+1 개, 쌍 수 = (k+1)//2. 5쌍 ⇔ k ≥ 9.
    줄무늬 구간 = 그 연속 구간의 첫 극값부터 마지막으로 확정된 극값까지 [첫 극값, 마지막 극값).
    (2026-11-10: 이전에는 '마지막 극값 + 마지막 줄 폭'까지 표시했다. 마지막 극값은 줄무늬가 끝나고 배경이
     시작되는 곳이라 배경을 줄 폭만큼 더 칠했고, 비스듬한 스캔에서는 1/cos θ 배로 더 넘쳐 면적이 약 +4.5%p 과대 측정됐다.
     남은 오차(2026-11-10 측정, 160×90 세로 줄무늬): 의도 면적보다 +0.6%p 안팎 크게 잰다(비스듬한 스캔이 영역
     경계 밖으로 몇 픽셀 넘침). '첫 극값과 다음 극값 사이를 줄로 세지 않는' 규칙을 시험했더니 경계는 1픽셀 이내로
     맞았지만 화면 가장자리에서 줄을 빠뜨려 −1.6%p 과소 측정(위험을 놓치는 방향)이 생겨 채택하지 않았다.
     화면 끝 줄은 '마지막 극값까지'라 반 주기만큼 덜 칠할 수 있다(100% 화면에서 99.6~99.9%).)
    """
    n, m = profiles.shape
    need = 2 * PATTERN_MIN_PAIRS - 1
    out = np.zeros((n, m + 1), dtype=bool)
    p = np.concatenate([profiles, np.full((n, 1), np.nan, profiles.dtype)], axis=1)  # 끝에서 마지막 극값 확정

    direction = np.zeros(n, np.int8)       # 0: 아직 방향 모름, +1 올라가는 중(현재 극값 = 봉우리), -1 내려가는 중
    lo = np.full(n, np.nan); lo_i = np.zeros(n, np.int64)   # 방향 모를 때의 최소·최대
    hi = np.full(n, np.nan); hi_i = np.zeros(n, np.int64)
    cur = np.full(n, np.nan); cur_i = np.zeros(n, np.int64)  # 추적 중인 극값
    prev = np.full(n, np.nan); prev_i = np.full(n, -1, np.int64)  # 확정된 직전 극값
    k = np.zeros(n, np.int64); run_start = np.zeros(n, np.int64)
    run_hsum = np.zeros(n)                                  # 연속 구간의 반 주기 합 (평균 cpd 계산용)
    segs = []                                               # (줄 번호, 시작, 끝, cpd) — 5쌍 이상 구간

    def confirm(rows: np.ndarray) -> None:
        """rows 의 '현재 극값'을 확정: 직전 극값과의 줄 하나를 검사하고 연속 구간을 갱신."""
        has_prev = rows[prev_i[rows] >= 0]
        if has_prev.size:
            half = (cur_i[has_prev] - prev_i[has_prev]).astype(np.float64)
            cpd = ppd / (2.0 * np.maximum(half, 1e-9))
            ok = (np.minimum(cur[has_prev], prev[has_prev]) < DARK_LIMIT_CD) & \
                 (cpd >= PATTERN_CPD[0])
            start_new = ok & (k[has_prev] == 0)
            run_start[has_prev[start_new]] = prev_i[has_prev[start_new]]
            run_hsum[has_prev] = np.where(ok, run_hsum[has_prev] + half, 0.0)
            k[has_prev] = np.where(ok, k[has_prev] + 1, 0)
            hit = has_prev[ok & (k[has_prev] >= need)]
            if hit.size:
                seg_cpd = ppd / (2.0 * run_hsum[hit] / k[hit])
                segs.append((hit, run_start[hit].copy(), cur_i[hit].copy(), seg_cpd))
        prev[rows], prev_i[rows] = cur[rows], cur_i[rows]

    for j in range(m + 1):
        v = p[:, j]
        nan = np.isnan(v)
        # 끊김: 추적 중인 극값을 확정하고 상태 초기화
        brk = np.flatnonzero(nan & (direction != 0))
        if brk.size:
            confirm(brk)
        reset = np.flatnonzero(nan)
        direction[reset] = 0; k[reset] = 0; prev_i[reset] = -1
        lo[reset] = np.nan; hi[reset] = np.nan

        # 방향 모름: 최소·최대를 추적하다 차이가 20 이상이 되면 먼저 나온 쪽을 첫 극값으로 확정
        z = np.flatnonzero(~nan & (direction == 0))
        if z.size:
            fresh = z[np.isnan(lo[z])]
            lo[fresh], hi[fresh], lo_i[fresh], hi_i[fresh] = v[fresh], v[fresh], j, j
            dn = z[v[z] < lo[z] - STRIPE_EXTREMUM_TOL]; lo[dn], lo_i[dn] = v[dn], j
            up = z[v[z] > hi[z] + STRIPE_EXTREMUM_TOL]; hi[up], hi_i[up] = v[up], j
            go = z[(hi[z] - lo[z]) >= DELTA_CD]
            if go.size:
                rising = lo_i[go] < hi_i[go]
                cur[go] = np.where(rising, lo[go], hi[go]); cur_i[go] = np.where(rising, lo_i[go], hi_i[go])
                confirm(go)                                   # 첫 극값 확정(직전 극값이 없으니 줄 검사는 없음)
                cur[go] = np.where(rising, hi[go], lo[go]); cur_i[go] = np.where(rising, hi_i[go], lo_i[go])
                direction[go] = np.where(rising, 1, -1)

        # 방향 있음: 같은 방향이면 극값 이동, 반대로 20 이상이면 확정 후 방향 전환
        a = np.flatnonzero(~nan & (direction != 0))
        if a.size:
            d = direction[a]
            ext = a[((d == 1) & (v[a] > cur[a] + STRIPE_EXTREMUM_TOL)) | ((d == -1) & (v[a] < cur[a] - STRIPE_EXTREMUM_TOL))]
            cur[ext], cur_i[ext] = v[ext], j
            rev = a[((d == 1) & (cur[a] - v[a] >= DELTA_CD)) | ((d == -1) & (v[a] - cur[a] >= DELTA_CD))]
            if rev.size:
                confirm(rev)
                cur[rev], cur_i[rev] = v[rev], j
                direction[rev] = -direction[rev]

    out_cpd = np.zeros((n, m + 1))
    for rows, s, e, c in segs:                                # 구간 표시 (겹치면 합집합, cpd 는 최댓값)
        for r, a0, b0, cc in zip(rows.tolist(), s.tolist(), e.tolist(), c.tolist()):
            out[r, a0:min(b0, m)] = True
            out_cpd[r, a0:min(b0, m)] = np.maximum(out_cpd[r, a0:min(b0, m)], cc)
    if return_cpd:
        return out[:, :m], out_cpd[:, :m]
    return out[:, :m]


def stripe_masks(lum: np.ndarray) -> np.ndarray:
    """⑤ 휘도(T×H×W) → 프레임별 줄무늬 픽셀 (T×H×W bool). 12방향의 합집합.

    픽셀마다 모든 방향에서 잰 cpd 의 최댓값(= 줄에 수직인 방향의 cpd)이 PATTERN_CPD 안이어야 줄무늬로 인정.
    """
    t_len, h, w = lum.shape
    mask = np.zeros(lum.shape, dtype=bool)
    cpd_max = np.zeros(lum.shape, dtype=np.float32)
    for ang in PATTERN_ANGLES:
        mx, my, valid = _scan_maps(h, w, ang)
        prof = _sample(lum, mx, my, valid)                   # T×Hc×Wc
        hc, wc = prof.shape[1:]
        runs, cpd = stripe_runs(prof.reshape(t_len * hc, wc), pixels_per_degree(w, h, ang), return_cpd=True)
        runs = runs.reshape(t_len, hc, wc) & valid[None]
        cpd = cpd.reshape(t_len, hc, wc)
        ti, ii, jj = np.nonzero(runs)
        if ti.size:
            ys = np.clip(np.round(my[ii, jj]).astype(int), 0, h - 1)
            xs = np.clip(np.round(mx[ii, jj]).astype(int), 0, w - 1)
            mask[ti, ys, xs] = True
            np.maximum.at(cpd_max, (ti, ys, xs), cpd[ti, ii, jj].astype(np.float32))
    return mask & (cpd_max >= PATTERN_CPD[0]) & (cpd_max <= PATTERN_CPD[1])


def stripe_normal_angle(lum2d: np.ndarray, mask: np.ndarray) -> float:
    """줄무늬 영역의 '줄에 수직인 방향'(밝기가 바뀌는 방향)을 0~180° 사이, 15° 단위로 돌려준다.

    구조 텐서 J = Σ[gx², gx·gy; gx·gy, gy²] 의 주축 = 기울기 방향 = 줄의 법선.
    각도는 _scan_maps 와 같은 좌표(x 오른쪽, y 아래쪽)를 쓴다.
    """
    gy, gx = np.gradient(lum2d.astype(np.float64))
    m = mask if mask.any() else np.ones_like(mask, dtype=bool)
    jxx, jyy, jxy = (gx[m] ** 2).sum(), (gy[m] ** 2).sum(), (gx[m] * gy[m]).sum()
    ang = 0.5 * np.degrees(np.arctan2(2.0 * jxy, jxx - jyy))
    step = PATTERN_ANGLES[1] - PATTERN_ANGLES[0]
    return float((round(ang / step) * step) % 180)


def motion_energy(lum_win: np.ndarray, mask: np.ndarray, angle_deg: float) -> tuple[float, float]:
    """⑥ ±0.5초 창(T×H×W)에서 줄무늬 영역의 움직임을 잰다 → (시간 변화 에너지 비율, 방향 지수 DI).

    줄의 법선 방향(angle_deg)으로 영역을 가로지르는 직선 최대 MOTION_ROWS 개를 골라,
    직선마다 '위치 x × 시간 t' 휘도 단면을 만들고 평균을 뺀 뒤 Hann 창을 씌워 2차원 FFT 한다.
      시간 변화 에너지 비율 = Σ|F(ω≠0, k)|² / Σ|F|²
      DI = (E(ω·k > 0) − E(ω·k < 0)) / (E(ω·k > 0) + E(ω·k < 0))       (ω ≠ 0, k ≠ 0)
    한 방향 흐름은 L(x − vt) 꼴이라 에너지가 한쪽(ω·k 부호 하나)에 몰려 |DI| ≈ 1,
    진동·명암 반전·점멸은 정지파라 양쪽에 대칭으로 나뉘어 DI ≈ 0 (Adelson & Bergen 1985 운동 에너지).
    """
    t_len, h, w = lum_win.shape
    mx, my, valid = _scan_maps(h, w, angle_deg)
    ys = np.clip(np.round(my).astype(int), 0, h - 1)
    xs = np.clip(np.round(mx).astype(int), 0, w - 1)
    inreg = valid & mask[ys, xs]
    rows = np.flatnonzero(inreg.sum(axis=1) >= 8)
    if rows.size == 0 or t_len < 3:
        return 0.0, 0.0
    rows = rows[np.linspace(0, rows.size - 1, min(MOTION_ROWS, rows.size)).astype(int)]
    e_all = e_time = e_pos = e_neg = 0.0
    for r in rows:
        cols = np.flatnonzero(inreg[r])
        # 가장 긴 연속 구간만 쓴다 (영역이 끊긴 곳을 이어 붙이면 가짜 주파수가 생김)
        breaks = np.flatnonzero(np.diff(cols) > 1)
        seg = np.split(cols, breaks + 1)
        cols = max(seg, key=len)
        if cols.size < 8:
            continue
        a = _sample(lum_win, mx[r:r + 1, cols], my[r:r + 1, cols], valid[r:r + 1, cols])[:, 0, :]
        a = np.nan_to_num(a.astype(np.float64))
        a = a - a.mean()
        # 정지 성분 = 위치마다의 시간 평균. 나머지 = 시간에 따라 변한 성분 (창 없이 재야 정지 패턴이 0이 됨)
        resid = a - a.mean(axis=0, keepdims=True)
        e_all += (a ** 2).sum()
        e_time += (resid ** 2).sum()
        # 방향은 변한 성분에만 창을 씌워 FFT (정지 성분이 창 때문에 시간 주파수로 새는 것을 막음)
        b = resid * np.hanning(t_len)[:, None] * np.hanning(a.shape[1])[None, :]
        p = np.abs(np.fft.fft2(b)) ** 2
        prod = np.fft.fftfreq(t_len)[:, None] * np.fft.fftfreq(a.shape[1])[None, :]
        e_pos += p[prod > 0].sum()
        e_neg += p[prod < 0].sum()
    if e_all <= 0:
        return 0.0, 0.0
    di = (e_pos - e_neg) / (e_pos + e_neg) if (e_pos + e_neg) > 0 else 0.0
    return float(e_time / e_all), float(di)


def classify_motion(time_ratio: float, di: float) -> str:
    """⑥ 시간 에너지 비율과 방향 지수 → 'static' | 'drifting' | 'reversing'."""
    if time_ratio < MOTION_STATIC_MAX:
        return "static"
    if abs(di) >= MOTION_DRIFT_MIN_DI:
        return "drifting"
    return "reversing"


def analyze_pattern(frames_bgr: np.ndarray, fps: float, lum: np.ndarray | None = None,
                    signed: np.ndarray | None = None) -> dict:
    """줄무늬(규칙 패턴) 판정 ⑤~⑧. analyze() 와 같은 입력 검사, 같은 꼴의 정답(hazard_frames, hazard_mask).

    돌려주는 값
    - stripe_area     프레임별 줄무늬 픽셀 면적 비율
    - stripes         프레임별 줄무늬 픽셀 (T×H×W bool) — analyze_all 이 흐름 줄무늬를 휘도 판정에서 뺄 때 씀
    - motion          프레임별 움직임 분류 'static' | 'reversing' | 'drifting' (줄무늬 없으면 '') (⑥)
    - time_ratio      프레임별 시간 변화 에너지 비율 (정지 판정 근거)
    - direction_index 프레임별 방향 지수 DI (흐름 판정 근거)
    - dynamic         (호환용) motion == 'reversing'
    - candidate       프레임별 면적 조건 충족 여부 (⑦)
    - max_stripe_sec  아무 1초 창 안의 후보 프레임 시간(초) 최댓값
    - peak_stripe_area 가장 넓은 줄무늬 면적
    - dangerous       위험 여부 (⑧)
    - hazard_frames, hazard_mask  위험 창 안의 후보 프레임 / 그 프레임들의 줄무늬 픽셀
    """
    _check_input(frames_bgr, fps)
    if lum is None:
        lum = luminance_cd(frames_bgr)
    if signed is None:
        signed = pixel_transitions(lum)
    t_len = len(lum)
    w = max(1, int(round(fps)))
    stripes = stripe_masks(lum)
    area = stripes.mean(axis=(1, 2))

    # ⑥ 움직임 3분류: ±0.5초(반 창) 휘도 단면의 운동 에너지
    half = w // 2
    motion = np.full(t_len, "", dtype=object)
    time_ratio = np.zeros(t_len)
    direction_index = np.zeros(t_len)
    stride = max(1, w // 4)                                  # 계산량: 창 중심을 1/4초마다 옮겨 이웃 프레임에 같은 결과
    for t0 in range(0, t_len, stride):
        frames_here = [t for t in range(t0, min(t_len, t0 + stride)) if stripes[t].any()]
        if not frames_here:
            continue
        tc = frames_here[len(frames_here) // 2]
        lo_t, hi_t = max(0, tc - half), min(t_len, tc + half + 1)
        region = stripes[lo_t:hi_t].any(axis=0)
        ang = stripe_normal_angle(lum[tc], stripes[tc])
        ratio, di = motion_energy(lum[lo_t:hi_t], region, ang)
        cls = classify_motion(ratio, di)
        for t in frames_here:
            motion[t], time_ratio[t], direction_index[t] = cls, ratio, di
    dynamic = motion == "reversing"

    # ⑦ 면적: 정지 > 40%, 반전 > 25%, 흐름은 제외(원문) 또는 25%
    drift_limit = np.inf if PATTERN_DRIFT_EXEMPT else PATTERN_AREA_DYNAMIC
    limit = np.where(motion == "static", PATTERN_AREA_STATIC,
                     np.where(motion == "drifting", drift_limit, PATTERN_AREA_DYNAMIC))
    cand = area > limit

    # ⑧ 지속: 1초 창 [s, s+W) 안의 후보 프레임 수 ≥ ceil(0.5·W)
    need = int(np.ceil(PATTERN_MIN_SEC * w))
    cs = np.concatenate([[0], np.cumsum(cand)])
    starts = np.arange(t_len)
    counts = cs[np.minimum(starts + w, t_len)] - cs[:t_len]
    hazard_frames = np.zeros(t_len, dtype=bool)
    for s in np.flatnonzero(counts >= need):
        hazard_frames[s:s + w] = True
    hazard_frames &= cand                                   # 위험 창 안의 '후보' 프레임만
    hazard_mask = stripes[hazard_frames].any(axis=0) if hazard_frames.any() else np.zeros(lum.shape[1:], bool)
    return {
        "stripe_area": area,
        "stripes": stripes,
        "motion": motion,
        "time_ratio": time_ratio,
        "direction_index": direction_index,
        "dynamic": dynamic,
        "candidate": cand,
        "max_stripe_sec": float(counts.max()) / fps if t_len else 0.0,
        "peak_stripe_area": float(area.max()) if t_len else 0.0,
        "dangerous": bool((counts >= need).any()),
        "hazard_frames": hazard_frames,
        "hazard_mask": hazard_mask,
    }


def analyze_all(frames_bgr: np.ndarray, fps: float) -> dict:
    """휘도 깜빡임(analyze)·적색 깜빡임(analyze_red)·줄무늬(analyze_pattern)를 함께 판정. 위험 = 하나라도 위험.

    hazard_frames·hazard_mask 는 세 판정의 합집합. 휘도와 픽셀 전이는 한 번만 계산한다.

    흐름 줄무늬 제외 (2026-11-10 결정 A): 줄무늬 판정이 '흐름(drifting)'으로 분류한 프레임의 줄무늬 픽셀에서
    생긴 휘도 전이는 휘도 깜빡임으로 세지 않는다. 한 방향으로 흐르는 줄무늬는 각 픽셀이 주기적으로 밝아졌다
    어두워져 휘도 깜빡임 조건을 넘을 수 있지만, 표준은 줄무늬 규칙에서 이를 제외한다
    ("obviously flow smoothly across, into, or out of the screen in one direction", ITC 2001 → Ofcom → ITU-R BT.1702-3
    Att.1). Harding FPA 도 흐름·카메라 이동을 제외한다(벤더 매뉴얼). 실험 근거: Binnie 1985 흐름 PPR 0~0.04.
    analyze()(휘도만)는 줄무늬를 모르므로 이 제외가 없다 — 줄무늬가 있을 수 있는 영상은 analyze_all 로 판정한다.
    """
    lum = luminance_cd(frames_bgr)
    signed = pixel_transitions(lum)
    pattern = analyze_pattern(frames_bgr, fps, lum=lum, signed=signed)
    drift = (pattern["motion"] == "drifting")
    if drift.any():
        signed = signed.copy()
        signed[drift] = np.where(pattern["stripes"][drift], 0, signed[drift])
    flash = analyze(frames_bgr, fps, lum=lum, signed=signed)
    flash["drift_exempt_frames"] = int(drift.sum())
    red = analyze_red(frames_bgr, fps)
    return {
        "flash": flash,
        "red": red,
        "pattern": pattern,
        "dangerous": bool(flash["dangerous"] or red["dangerous"] or pattern["dangerous"]),
        "hazard_frames": flash["hazard_frames"] | red["hazard_frames"] | pattern["hazard_frames"],
        "hazard_mask": flash["hazard_mask"] | red["hazard_mask"] | pattern["hazard_mask"],
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
