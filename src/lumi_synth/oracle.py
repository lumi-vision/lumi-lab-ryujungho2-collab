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
def waveform(p: OracleParams, n_frames: int, fps: float) -> np.ndarray:
    """0~1 사이 켜짐 정도 w(t). 1주기 = 켜짐 1번 + 꺼짐 1번 = 깜빡임 1번."""
    pos = p.freq_hz * np.arange(n_frames) / fps + p.phase
    frac = pos - np.floor(pos)
    if p.waveform == "square":
        w = (frac < 0.5).astype(np.float32)
    elif p.waveform == "sine":
        w = 0.5 - 0.5 * np.cos(2 * np.pi * frac)
    elif p.waveform == "triangle":
        w = 1.0 - np.abs(2.0 * frac - 1.0)
    else:
        raise ValueError(f"알 수 없는 파형: {p.waveform}")
    return w.astype(np.float32)


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
    delta_rel = p.delta_cd / measure.WHITE_CD           # cd/m² → 상대 휘도
    lin = srgb_to_linear(clean_bgr)
    add = (w[:, None, None] * delta_rel)[..., None] * m[None, :, :, None]   # T×H×W×1
    out = linear_to_srgb(lin + add)
    out = np.where(m[None, :, :, None], out, clean_bgr)  # 마스크 밖은 원본과 비트 단위로 같게
    changed = (out != clean_bgr).any(axis=(0, 3))
    r = measure.analyze(out, fps)
    judge = {k: r[k] for k in ("dangerous", "max_flashes", "peak_area")}
    judge["hazard_frames"] = r["hazard_frames"]
    judge["hazard_mask"] = r["hazard_mask"]
    expect = p.intent == "hazard"
    return {"flicker": out, "mask_allow": m, "mask_changed": changed, "w": w,
            "judge": judge, "intent_ok": bool(r["dangerous"]) == expect, "params": asdict(p)}
