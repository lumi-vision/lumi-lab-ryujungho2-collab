"""오라클(Oracle): 규칙으로 깜빡임을 '정확히 원하는 조건대로' 넣는 합성기.

한 줄 요약
    깨끗한 영상 Y 의 마스크 M 영역에, 선형 빛 공간에서 '빛을 더하는' 방식으로 깜빡임을 넣는다.
    X_lin[t,p] = min(1, Y_lin[t,p] + M[p] · w(t) · δ),   δ = ΔL / 200

왜 '빛을 더하는' 모델인가 (docs/ORACLE.md §2)
    - 선형 RGB 세 채널에 같은 δ 를 더하면 상대 휘도 Y = 0.2126R + 0.7152G + 0.0722B 가 정확히 δ 만큼 오른다
      (가중치 합이 1). 그래서 판정 규칙의 '밝기 변화 ΔL' 조건을 cd/m² 단위로 직접 조절할 수 있다.
    - 실제 위험 장면(번쩍이는 조명, 섬광, 폭발)도 물리적으로는 빛이 더해지는 현상이다.
    - 바탕의 질감(윤곽선)은 그대로 남는다. 곱하거나 덮어씌우는 방식과 달리 장면이 지워지지 않는다.
    - GAN(gan.py)도 같은 모델을 쓰되, 균일한 δ·w(t) 대신 '공간·시간적으로 변하는 빛'을 학습한다.

설계 의도(intent): pse-test-media 의 f/a/c/l 설계처럼 '어느 조건을 일부러 무너뜨릴지'를 정한다.
    hazard    : 세 조건 모두 넘김         (면적 > 25%, 깜빡임 > 3/s, ΔL ≥ 20)
    safe_area : 면적만 미달               (면적 < 25%)
    safe_count: 횟수만 미달               (깜빡임 ≤ 3/s)
    safe_lum  : 밝기만 미달               (ΔL < 20)
    → 같은 바탕에서 조건 하나만 다른 '경계 쌍'을 만들 수 있다.

정답(라벨)은 의도가 아니라 **판정 규칙(measure.analyze)의 결과**다. 의도와 판정이 다르면
intent_ok=False 로 기록한다(예: 바탕이 너무 밝아 δ 가 잘려 ΔL 이 모자란 경우). 이 비율이 오라클의 신뢰도다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from lumi_synth import measure

INTENTS = ("hazard", "safe_area", "safe_count", "safe_lum")


# ---- sRGB ↔ 선형 변환 (measure.luminance_cd 와 같은 IEC 61966-2-1 식) -------------------
def srgb_to_linear(u8: np.ndarray) -> np.ndarray:
    c = u8.astype(np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin, 0.0, 1.0)
    c = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return np.clip(np.rint(c * 255.0), 0, 255).astype(np.uint8)


# ---- 합성 조건 -----------------------------------------------------------------------------
@dataclass(frozen=True)
class OracleParams:
    """한 샘플의 합성 조건. 이 값만 있으면 같은 결과를 다시 만들 수 있다(재현성)."""
    intent: str            # INTENTS 중 하나 (설계 의도)
    area: float            # 마스크 면적 비율 (화면 대비)
    shape: str             # ellipse | rectangle | blobs | full
    center: tuple          # 마스크 중심 (x, y) 비율 0~1
    aspect: float          # 가로/세로 비
    freq_hz: float         # 깜빡임 주파수 (1초에 켜짐-꺼짐 주기 수 = 깜빡임 수)
    waveform: str          # square | sine | triangle
    phase: float           # 시작 위상 (0~1 주기)
    delta_cd: float        # 켜졌을 때 더하는 휘도 ΔL (cd/m²)
    seed: int


def sample_params(intent: str, rng: np.random.Generator) -> OracleParams:
    """설계 의도에 맞게 조건을 무작위로 뽑는다. 범위는 경계에서 '확실히' 떨어지게 잡았다(docs/ORACLE.md §3).

    - 면적: 넘김 0.32~0.70 / 미달 0.08~0.20  (경계 0.25 에서 ±0.05 이상 여유)
    - 주파수: 넘김 4~8 Hz / 미달 1~2.5 Hz     (경계 3 Hz. 2 프레임 단위 양자화를 고려한 여유)
    - ΔL: 넘김 35~80 cd/m² / 미달 6~14 cd/m²  (경계 20. 8비트 양자화 오차 약 1 cd/m² 를 고려)
    """
    if intent not in INTENTS:
        raise ValueError(f"intent 는 {INTENTS} 중 하나여야 합니다: {intent}")
    area = rng.uniform(0.32, 0.70)
    freq = rng.uniform(4.0, 8.0)
    delta = rng.uniform(35.0, 80.0)
    if intent == "safe_area":
        area = rng.uniform(0.08, 0.20)
    elif intent == "safe_count":
        freq = rng.uniform(1.0, 2.5)
    elif intent == "safe_lum":
        delta = rng.uniform(6.0, 14.0)
    shape = str(rng.choice(["ellipse", "rectangle", "blobs", "full"], p=[0.4, 0.3, 0.2, 0.1]))
    if shape == "full":
        area = 1.0 if intent != "safe_area" else area   # 화면 전체는 면적 미달 의도와 양립 불가
        if intent == "safe_area":
            shape = "ellipse"
    return OracleParams(
        intent=intent, area=float(area), shape=shape,
        center=(float(rng.uniform(0.3, 0.7)), float(rng.uniform(0.3, 0.7))),
        aspect=float(rng.uniform(0.6, 1.8)), freq_hz=float(freq),
        waveform=str(rng.choice(["square", "sine", "triangle"])), phase=float(rng.uniform(0, 1)),
        delta_cd=float(delta), seed=int(rng.integers(0, 2**31 - 1)),
    )


# ---- 공간 마스크 M ---------------------------------------------------------------------------
def _draw(shape: str, h: int, w: int, center, aspect: float, scale: float, seed: int) -> np.ndarray:
    """크기 배율 scale 로 모양을 그린다(0/1 uint8)."""
    m = np.zeros((h, w), np.uint8)
    cx, cy = int(center[0] * w), int(center[1] * h)
    ry = max(1, int(scale * h / 2))
    rx = max(1, int(ry * aspect))
    if shape == "full":
        m[:] = 1
    elif shape == "ellipse":
        cv2.ellipse(m, (cx, cy), (rx, ry), 0, 0, 360, 1, -1)
    elif shape == "rectangle":
        cv2.rectangle(m, (cx - rx, cy - ry), (cx + rx, cy + ry), 1, -1)
    elif shape == "blobs":                       # 원 3개 (여러 광원)
        # 원의 중심은 크기(scale)와 무관하게 고정한다. 중심까지 함께 커지면 원끼리 멀어져 면적이
        # scale 에 따라 늘지 않을 수 있고, 그러면 아래 make_mask 의 이분 탐색이 틀린다(2026-11-09 수정).
        r = np.random.default_rng(seed)
        for _ in range(3):
            ox, oy = r.uniform(-0.15, 0.15, 2)
            cv2.circle(m, (int(cx + ox * w), int(cy + oy * h)), max(1, int(0.6 * ry)), 1, -1)
    else:
        raise ValueError(f"알 수 없는 모양: {shape}")
    return m


def make_mask(p: OracleParams, h: int, w: int) -> np.ndarray:
    """면적이 p.area 에 가장 가깝도록 크기 배율을 이분 탐색한 마스크 M (bool, H×W).

    면적은 판정의 핵심 조건이라 모양을 그린 뒤 '실제 픽셀 수'로 맞춘다(그리는 식이 아니라 결과를 맞춤).
    """
    if p.shape == "full":
        return np.ones((h, w), bool)
    lo, hi = 0.0, 4.0
    best, best_err = None, np.inf
    for _ in range(30):
        mid = (lo + hi) / 2
        m = _draw(p.shape, h, w, p.center, p.aspect, mid, p.seed)
        err = abs(m.mean() - p.area)
        if err < best_err:                       # 시도한 것 중 목표 면적에 가장 가까운 것을 고른다
            best, best_err = m, err
        if m.mean() < p.area:
            lo = mid
        else:
            hi = mid
    return best.astype(bool)


# ---- 시간 파형 w(t) ------------------------------------------------------------------------
def wave(kind: str, freq_hz: float, n_frames: int, fps: float, phase: float = 0.0) -> np.ndarray:
    """0~1 사이 켜짐 정도 w(t). 1주기 = 켜짐 1번 + 꺼짐 1번 = 깜빡임 1번. kind: square | sine | triangle."""
    pos = freq_hz * np.arange(n_frames) / fps + phase
    frac = pos - np.floor(pos)
    if kind == "square":
        w = (frac < 0.5).astype(np.float32)
    elif kind == "sine":
        w = 0.5 - 0.5 * np.cos(2 * np.pi * frac)
    elif kind == "triangle":
        w = 1.0 - np.abs(2.0 * frac - 1.0)
    else:
        raise ValueError(f"알 수 없는 파형: {kind}")
    return w.astype(np.float32)


def waveform(p: OracleParams, n_frames: int, fps: float) -> np.ndarray:
    """OracleParams 의 파형·주파수·위상으로 w(t) (wave() 참고)."""
    return wave(p.waveform, p.freq_hz, n_frames, fps, p.phase)


# ---- 합성 -------------------------------------------------------------------------------------
def synthesize(clean_bgr: np.ndarray, p: OracleParams, fps: float) -> dict:
    """깨끗한 클립(T×H×W×3 uint8 BGR)에 깜빡임을 넣고, 판정 규칙으로 정답을 매긴다.

    돌려주는 값
    - flicker      합성 결과 X (uint8 BGR)
    - mask_allow   M: 빛을 더하도록 '허용한' 영역 (bool H×W)
    - mask_changed 실제로 픽셀 값이 바뀐 영역 (bool H×W) — δ 가 작거나 잘리면 M 보다 작을 수 있음
    - w            시간 파형 (T)
    - judge        measure.analyze(X) 결과 요약 — 이것이 정답
    - intent_ok    설계 의도와 판정이 맞는지 (hazard ↔ dangerous)
    - params       재현용 합성 조건
    """
    t_len, h, w_ = clean_bgr.shape[:3]
    m = make_mask(p, h, w_)
    w = waveform(p, t_len, fps)
    out = render_flicker(clean_bgr, m, w, p.delta_cd)
    changed = (out != clean_bgr).any(axis=(0, 3))
    r = measure.analyze(out, fps)
    judge = {k: r[k] for k in ("dangerous", "max_flashes", "peak_area")}
    judge["hazard_frames"] = r["hazard_frames"]
    judge["hazard_mask"] = r["hazard_mask"]
    expect = p.intent == "hazard"
    return {"flicker": out, "mask_allow": m, "mask_changed": changed, "w": w,
            "judge": judge, "intent_ok": bool(r["dangerous"]) == expect, "params": asdict(p)}


# =============================================================================================
# 생성 계열 확장 (2026-11-10): 휘도 깜빡임 그리기 분리 · 적색 깜빡임 · 줄무늬
#
# 위 synthesize() 는 '무작위 조건 → 휘도 깜빡임 → 판정'을 한 번에 한다. 아래 함수들은 그리기만 한다
# (판정은 부르는 쪽에서: 저장한 파일을 다시 읽어서 잰다 — PLAN §5-3). 조건 하나씩 경계를 가로질러 바꾸는
# 스윕(scripts/make_hazard_sweep.py)과 이후 GAN 데이터 생성에 쓴다.
#
#   휘도  render_flicker    X_lin = min(1, Y_lin + M·w(t)·ΔL/200)          (위 synthesize 와 같은 식)
#         render_flat_flash 영역을 단색 회색으로 덮고 L_lo ↔ L_hi (cd/m²) 교대 — '어두운 쪽 < 160' 조건 시험용
#   적색  red_pair          같은 휘도의 두 색 A(빨강, 선형 R/(R+G+B) = ρ) ↔ B(A 와 파랑을 선형으로 섞은 색)
#                           · 선형 RGB 혼합은 휘도를 보존하므로(가중합) 휘도 깜빡임 없이 색만 바뀐다
#                           · u′v′ 는 XYZ 의 사영변환이라 선형 혼합이 u′v′ 위 직선 → Δu′v′ 가 혼합비에 단조 → 이분 탐색
#                           · 상대 휘도 0.06(12 cd/m²): 이 밝기여야 ρ 0.6~0.95 전 범위에서 Δu′v′ 0.33 이상까지
#                             만들 수 있다(0.15 에서는 청록·초록 쪽 최대 0.13~0.30, 색역 한계 — 2026-11-10 계산)
#         render_red        영역을 A ↔ B 단색으로 덮음 (pse-test-media 적색 패턴과 같은 방식)
#   줄무늬 render_stripes   사각파 줄무늬, 방향 θ, 공간 주파수 cpd (56 cm·15인치 기준 measure.pixels_per_degree),
#                           밝은 줄·어두운 줄 휘도(cd/m²), 움직임 static | reversing(명암 반전 hz) | drifting(°/s)
#                           · 픽셀마다 4×4 부분 표본으로 밝은 줄이 덮는 비율을 구해 '선형 빛'으로 섞는다
#                             (화면이 실제로 내는 빛의 면적 평균. 높은 cpd 에서 명암이 줄어드는 것도 그대로 나타남)
#
# 줄무늬 정의 중 아직 정하지 않은 것(스윕에서 다루지 않음): 쌍 수(5 또는 6, 면적과 얽힘), 체커보드·동심원 등
# 다른 모양, 흐름 줄무늬를 휘도 판정에서도 뺄지(결정 A, 재검토 예정).
# =============================================================================================
def region_mask(shape: str, area: float, h: int, w: int, center=(0.5, 0.5), aspect: float = 16 / 9,
                seed: int = 0) -> np.ndarray:
    """모양·면적만 정한 마스크 (make_mask 와 같은 이분 탐색). aspect = 가로/세로 픽셀 비."""
    p = OracleParams(intent="hazard", area=float(area), shape=shape, center=tuple(center), aspect=float(aspect),
                     freq_hz=0.0, waveform="square", phase=0.0, delta_cd=0.0, seed=int(seed))
    return make_mask(p, h, w)


def render_flicker(clean_bgr: np.ndarray, mask: np.ndarray, w: np.ndarray, delta_cd: float) -> np.ndarray:
    """휘도 깜빡임: 선형 빛 더하기. 마스크 밖은 원본과 비트 단위로 같다."""
    lin = srgb_to_linear(clean_bgr)
    add = (w[:, None, None] * (delta_cd / measure.WHITE_CD))[..., None] * mask[None, :, :, None]
    out = linear_to_srgb(lin + add)
    return np.where(mask[None, :, :, None], out, clean_bgr)


def gray_u8(cd: float) -> int:
    """휘도 cd/m² 인 회색의 sRGB 8비트 값 (가장 가까운 값)."""
    return int(linear_to_srgb(np.array([cd / measure.WHITE_CD]))[0])


def render_flat_flash(clean_bgr: np.ndarray, mask: np.ndarray, w: np.ndarray, lo_cd: float, hi_cd: float) -> np.ndarray:
    """영역을 단색 회색으로 덮고 w(t)=0 이면 lo_cd, 1 이면 hi_cd (중간값은 선형 빛으로 보간)."""
    out = clean_bgr.copy()
    for t in range(len(out)):
        out[t][mask] = gray_u8(lo_cd + (hi_cd - lo_cd) * float(w[t]))
    return out


RED_PAIR_Y = 0.06            # 적색 색 쌍의 상대 휘도 (12 cd/m²) — 위 설명 참고
_BLUE = np.array([0.0, 0.0, 1.0])


def _red_measure(rgb_u8) -> tuple[float, float, float, bool]:
    """(R,G,B) 8비트 → (선형 R/(R+G+B), u′, v′, 포화 빨강 여부) — 판정기(measure.red_state)로 잰다."""
    bgr = np.array([[[rgb_u8[2], rgb_u8[1], rgb_u8[0]]]], np.uint8)
    sat, u, v = measure.red_state(bgr)
    lin = srgb_to_linear(np.array(rgb_u8, np.uint8))
    return float(lin[0] / max(float(lin.sum()), 1e-12)), float(u[0, 0]), float(v[0, 0]), bool(sat[0, 0])


def _lum_cd(rgb_u8) -> float:
    return float(measure.WHITE_CD * (srgb_to_linear(np.array(rgb_u8, np.uint8)) @ np.array([0.2126, 0.7152, 0.0722])))


def red_pair(sat_ratio: float, delta_uv: float, y_rel: float = RED_PAIR_Y) -> dict:
    """같은 휘도의 적색 색 쌍. A = 선형 (r, g, g) 로 R/(R+G+B) = sat_ratio, B = (1−s)·A + s·파랑(같은 휘도).

    s 를 이분 탐색해 8비트로 양자화한 뒤 판정기로 잰 Δu′v′ 가 delta_uv 에 가장 가깝게 한다.
    돌려주는 값: a_rgb, b_rgb (8비트 R,G,B), 실제로 잰 ratio_a, ratio_b, sat_a, sat_b, delta_uv, lum_a_cd, lum_b_cd.
    """
    g_over_r = (1.0 - sat_ratio) / (2.0 * sat_ratio)
    r = y_rel / (0.2126 + 0.7874 * g_over_r)
    a_lin = np.array([r, r * g_over_r, r * g_over_r])
    blue = _BLUE * (y_rel / 0.0722)
    if r > 1 or blue.max() > 1:
        raise ValueError(f"색역 밖: sat_ratio={sat_ratio}, y_rel={y_rel}")
    a_u8 = tuple(int(c) for c in linear_to_srgb(a_lin))
    ra, ua, va, sa = _red_measure(a_u8)

    def b_of(s):
        b_u8 = tuple(int(c) for c in linear_to_srgb((1 - s) * a_lin + s * blue))
        rb, ub, vb, sb = _red_measure(b_u8)
        return b_u8, float(np.hypot(ub - ua, vb - va)), rb, sb

    lo, hi = 0.0, 1.0
    best = None
    for _ in range(40):
        mid = (lo + hi) / 2
        cand = b_of(mid)
        if best is None or abs(cand[1] - delta_uv) < abs(best[1] - delta_uv):
            best = cand
        if cand[1] < delta_uv:
            lo = mid
        else:
            hi = mid
    b_u8, d, rb, sb = best
    return {"a_rgb": a_u8, "b_rgb": b_u8, "ratio_a": ra, "ratio_b": rb, "sat_a": sa, "sat_b": sb,
            "delta_uv": d, "lum_a_cd": _lum_cd(a_u8), "lum_b_cd": _lum_cd(b_u8)}


def render_red(clean_bgr: np.ndarray, mask: np.ndarray, w: np.ndarray, a_rgb, b_rgb) -> np.ndarray:
    """적색 깜빡임: 영역을 단색으로 덮고 w(t) < 0.5 이면 A, ≥ 0.5 이면 B. 마스크 밖은 원본 그대로."""
    out = clean_bgr.copy()
    a_bgr, b_bgr = tuple(a_rgb)[::-1], tuple(b_rgb)[::-1]
    for t in range(len(out)):
        out[t][mask] = b_bgr if w[t] >= 0.5 else a_bgr
    return out


STRIPE_MOTIONS = ("static", "reversing", "drifting")


def stripe_period_px(width: int, height: int, cpd: float, angle_deg: float) -> float:
    """줄무늬 한 주기(밝은 줄 + 어두운 줄)의 픽셀 길이. angle_deg = 줄에 수직인 방향(0° = 세로 줄)."""
    return measure.pixels_per_degree(width, height, angle_deg) / cpd


def render_stripes(clean_bgr: np.ndarray, mask: np.ndarray, fps: float, cpd: float, angle_deg: float = 0.0,
                   lo_cd: float = 10.0, hi_cd: float = 110.0, motion: str = "static", hz: float = 5.0,
                   speed_dps: float = 4.0, sub: int = 4) -> np.ndarray:
    """사각파 줄무늬로 영역을 덮는다. 마스크 밖은 원본 그대로.

    motion: static(정지) | reversing(hz 로 명암 반전: 반주기마다 밝은 줄 ↔ 어두운 줄) | drifting(speed_dps °/s 로 한 방향 흐름)
    """
    if motion not in STRIPE_MOTIONS:
        raise ValueError(f"motion 은 {STRIPE_MOTIONS} 중 하나: {motion}")
    t_len, h, w = clean_bgr.shape[:3]
    period = stripe_period_px(w, h, cpd, angle_deg)
    ppd = measure.pixels_per_degree(w, h, angle_deg)
    a = np.radians(angle_deg)
    off = (np.arange(sub) + 0.5) / sub - 0.5
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    pos = [(xs + ox) * np.cos(a) + (ys + oy) * np.sin(a) for oy in off for ox in off]   # 줄 수직 방향 좌표
    lo, hi = lo_cd / measure.WHITE_CD, hi_cd / measure.WHITE_CD
    out = clean_bgr.copy()
    cache: dict = {}                                       # 정지·반전은 상태가 1~2개뿐이라 한 번만 계산
    for t in range(t_len):
        shift = speed_dps * ppd * t / fps if motion == "drifting" else 0.0
        flip = motion == "reversing" and int(t * hz * 2 / fps) % 2 == 1
        if (shift, flip) not in cache:
            cov = np.mean([(((q - shift) / period) % 1.0) < 0.5 for q in pos], axis=0)
            cache[(shift, flip)] = linear_to_srgb(lo + (hi - lo) * (1.0 - cov if flip else cov))[mask][:, None]
        out[t][mask] = cache[(shift, flip)]
    return out


def stripe_pairs_in_mask(mask: np.ndarray, period_px: float, angle_deg: float) -> float:
    """마스크가 줄 수직 방향으로 차지하는 길이 ÷ 주기 = 들어가는 줄 쌍 수 (설계상 기대값 계산용)."""
    ys, xs = np.nonzero(mask)
    a = np.radians(angle_deg)
    q = xs * np.cos(a) + ys * np.sin(a)
    return float((q.max() - q.min() + 1) / period_px) if len(q) else 0.0
