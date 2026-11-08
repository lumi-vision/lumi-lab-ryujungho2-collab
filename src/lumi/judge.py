"""휘도 깜빡임 판정 규칙 — 하드 판정(라벨용)과 같은 규칙의 미분 가능 판정(G 학습용).

규칙 (WCAG 2.x / ITU-R BT.1702의 일반 깜빡임을 픽셀 단위로 구현, '화면 25%' 면적 기준):
1. sRGB → 선형 → 상대 휘도 Y (0.2126 R + 0.7152 G + 0.0722 B).
2. 연속 두 프레임에서 픽셀의 |ΔY| ≥ 0.1 이고 어두운 쪽 Y < 0.8이면 그 픽셀은 전이한 것이다.
3. 같은 방향(밝아짐 또는 어두워짐)으로 전이한 픽셀이 화면의 25% 이상이면 그 프레임 간격은 전이다.
4. 같은 방향 전이가 바로 이어지면(간격 1) 하나로 합친다(서서히 밝아지는 것 = 전이 1회).
5. 1초 창 안의 전이 수가 6을 넘으면(깜빡임 3회 초과) 실패.

검증: pse-test-media(Jordan 2025) broadcast_30fps_01 40개, 30fps_alternating_01 16개를 모두 기대대로
판정했다(IRIS_LABELING_PLAN 5.2와 같은 세트). IRIS처럼 화면 평균으로 판정하지 않으므로 국소 깜빡임도 센다.
면적은 아직 시야각이 아니라 화면 비율이다. 이 판정은 표준 문턱 판정이지 임상 위험 판정이 아니다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DELTA_Y = 0.1
DARK_Y = 0.8
AREA = 0.25
MAX_TRANSITIONS = 6


def srgb_to_luminance_np(rgb_uint8: np.ndarray) -> np.ndarray:
    """[..., 3] uint8 RGB → 상대 휘도 [...] float32."""
    c = rgb_uint8.astype(np.float32) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]


@dataclass(frozen=True)
class FlashJudgement:
    max_transitions_1s: int
    fail: bool
    transition_frames: tuple[int, ...]   # 전이가 끝나는 프레임 번호(1부터, 프레임 i-1 → i)
    max_area: float


def judge_luminance_flash(rgb_uint8: np.ndarray, fps: float) -> FlashJudgement:
    """[T,H,W,3] uint8 RGB 클립의 휘도 깜빡임 하드 판정."""
    if rgb_uint8.ndim != 4 or rgb_uint8.shape[-1] != 3 or len(rgb_uint8) < 2:
        raise ValueError("입력은 [T,H,W,3] uint8, T ≥ 2여야 합니다.")
    y = srgb_to_luminance_np(rgb_uint8)
    d = np.diff(y, axis=0)
    dark = np.minimum(y[1:], y[:-1]) < DARK_Y
    pos = ((d >= DELTA_Y) & dark).mean(axis=(1, 2))
    neg = ((d <= -DELTA_Y) & dark).mean(axis=(1, 2))
    area = np.maximum(pos, neg)
    events = [(i, 1 if p >= n else -1) for i, (p, n) in enumerate(zip(pos, neg)) if max(p, n) >= AREA]
    merged: list[tuple[int, int]] = []
    for i, s in events:
        if merged and merged[-1][1] == s and i - merged[-1][0] <= 1:
            merged[-1] = (i, s)
            continue
        merged.append((i, s))
    t = np.array([i / fps for i, _ in merged])
    best = 0
    for k in range(len(t)):
        best = max(best, int(np.sum((t >= t[k] - 1e-9) & (t < t[k] + 1.0 - 1e-9))))
    return FlashJudgement(best, best > MAX_TRANSITIONS, tuple(i + 1 for i, _ in merged),
                          float(area.max()) if len(area) else 0.0)


# ---------------------------------------------------------------------------
# 미분 가능 판정 (G 학습용). 같은 규칙을 시그모이드로 완화했다.
# ---------------------------------------------------------------------------
try:
    import torch
    from torch.nn import functional as F
except ImportError:  # 판정만 쓰는 환경에서는 torch가 없어도 된다
    torch = None


def srgb_to_luminance_torch(x: "torch.Tensor") -> "torch.Tensor":
    """[B,3,T,H,W] 0~1 sRGB → [B,T,H,W] 상대 휘도 (미분 가능)."""
    lin = torch.where(x <= 0.04045, x / 12.92, ((x.clamp(min=0) + 0.055) / 1.055) ** 2.4)
    return 0.2126 * lin[:, 0] + 0.7152 * lin[:, 1] + 0.0722 * lin[:, 2]


def soft_transition_count(x: "torch.Tensor", fps: float, *, tau_y: float = 0.02,
                          tau_area: float = 0.03, beta: float = 5.0) -> dict:
    """[B,3,T,H,W] 0~1 영상의 1초 창 최대 전이 수(연속값)와 프레임 간 전이 강도.

    하드 판정의 각 단계(픽셀 문턱, 면적 문턱, 같은 방향 합치기, 1초 창 최대)를
    시그모이드·소프트 최댓값으로 바꾼 것이다. tau가 0에 가까울수록 하드 판정에 가까워진다.
    """
    y = srgb_to_luminance_torch(x)
    d = y[:, 1:] - y[:, :-1]
    dark = torch.sigmoid((DARK_Y - torch.minimum(y[:, 1:], y[:, :-1])) / tau_y)
    p = torch.sigmoid((d - DELTA_Y) / tau_y) * dark
    n = torch.sigmoid((-d - DELTA_Y) / tau_y) * dark
    ap, an = p.mean(dim=(2, 3)), n.mean(dim=(2, 3))                  # [B, T-1]
    ep = torch.sigmoid((ap - AREA) / tau_area)
    en = torch.sigmoid((an - AREA) / tau_area)
    prev_p = F.pad(ep[:, :-1], (1, 0))
    prev_n = F.pad(en[:, :-1], (1, 0))
    onset = ep * (1 - prev_p) + en * (1 - prev_n)                     # 새 전이(같은 방향 연속은 1회)
    w = max(1, int(round(fps)))
    kernel = torch.ones(1, 1, w, device=x.device, dtype=onset.dtype)
    counts = F.conv1d(onset.unsqueeze(1), kernel, padding=w - 1).squeeze(1)  # 모든 1초 창
    soft_max = torch.logsumexp(beta * counts, dim=1) / beta
    return {"count": soft_max, "onset": onset, "area_pos": ap, "area_neg": an}
