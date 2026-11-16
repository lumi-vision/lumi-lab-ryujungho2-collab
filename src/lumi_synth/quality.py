"""보정 결과의 화질 지표 (docs/GDR_VERIFY.md '평가 체계').

    Fidelity   원본과의 참조 비교: PSNR · SSIM · LPIPS — 영역(위험·복원 가능·정보 손실)별로
    Spatial    경계 기울기 불연속(roi.seam_ratio) · 마스크 밖 변경량
    Temporal   광학 흐름 워핑 오차: 원본의 움직임(흐름)으로 앞 프레임을 옮겨 지금 프레임과 비교 — 출력/원본 비율
               (1 = 원본과 같은 시간 일관성, > 1 = 깜빡임·떨림이 남음). 프레임 간 변화량 비율(roi.temporal_ratio)과 함께 본다.

주의 (사용자 결정 문서 2026-11-15)
    - LPIPS 는 '자연스러움'이 아니라 원본과의 지각적 유사도다. 원본이 지워진 영역에서 그럴듯하게 채워도 점수가 나쁠 수 있다.
    - 워핑 오차가 낮아도 영상이 흐려지거나 멈춘 실패일 수 있다 — 질감·시간 변화량과 함께 본다. 어느 지표도 단독 합격 기준이 아니다.
"""
from __future__ import annotations

import functools

import numpy as np

LOSS_ALPHA = 0.95      # α ≥ 이 값 = 심한 정보 감쇠(M_loss). 실험적 문턱 — 수학적 완전 소실 기준이 아님


def gray(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) @ np.array([0.114, 0.587, 0.299], np.float32)


def regions(tubes: np.ndarray, alpha: np.ndarray | None) -> dict:
    """튜브(3×T×H×W) + α(T×H×W 0~1 또는 None) → 평가 영역(T×H×W bool).

    hazard        위험 영역 M_hazard (세 종류 합집합)
    recoverable   위험 ∩ α < 0.95  (원본 신호가 남아 있음)
    lost_temporal 위험 ∩ α ≥ 0.95 이지만 같은 픽셀이 다른 프레임에서는 α < 0.95 (시간 정보로 복원 가능성)
    lost_never    위험 ∩ 모든 프레임에서 α ≥ 0.95 (원본이 입력 어디에도 없음 → 채우기만 가능)
    """
    hz = tubes.any(axis=0)
    if alpha is None:
        z = np.zeros_like(hz)
        return {"hazard": hz, "recoverable": hz, "lost_temporal": z, "lost_never": z}
    lost = alpha >= LOSS_ALPHA
    always = lost.all(axis=0)[None]
    return {"hazard": hz, "recoverable": hz & ~lost, "lost_temporal": hz & lost & ~always, "lost_never": hz & lost & always}


def psnr(a: np.ndarray, b: np.ndarray, where: np.ndarray) -> float:
    if not where.any():
        return float("nan")
    mse = float(((a.astype(np.float32) - b.astype(np.float32)) ** 2)[where].mean())
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """프레임별 SSIM 지도(회색, 가우시안 σ 1.5 창) → T×H×W."""
    import cv2
    x, y = gray(a), gray(b)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    out = np.empty(x.shape, np.float32)
    for t in range(len(x)):
        blur = lambda v: cv2.GaussianBlur(v, (0, 0), 1.5)       # noqa: E731
        mx, my = blur(x[t]), blur(y[t])
        sxx, syy, sxy = blur(x[t] * x[t]) - mx * mx, blur(y[t] * y[t]) - my * my, blur(x[t] * y[t]) - mx * my
        out[t] = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return out


@functools.lru_cache(maxsize=1)
def _lpips_net(device: str):
    import warnings

    import lpips
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return lpips.LPIPS(net="alex", spatial=True, verbose=False).to(device).eval()


def lpips_map(a: np.ndarray, b: np.ndarray, device: str = "cuda", chunk: int = 8) -> np.ndarray:
    """프레임별 LPIPS(AlexNet) 공간 지도 → T×H×W (낮을수록 원본과 지각적으로 비슷)."""
    import torch
    net = _lpips_net(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(a), chunk):
            ta = torch.from_numpy(a[i:i + chunk][..., ::-1].copy()).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1
            tb = torch.from_numpy(b[i:i + chunk][..., ::-1].copy()).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1
            out.append(net(ta, tb)[:, 0].float().cpu().numpy())
    return np.concatenate(out)


def clean_flow(clean: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """원본의 광학 흐름(Farnebäck): 프레임 t → t−1 로 가는 좌표 지도 (T−1)×H×W×2 (float32, 전체 해상도 픽셀)."""
    import cv2
    g = gray(clean)
    h, w = g.shape[1:]
    small = [cv2.resize(f, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) for f in g]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    maps = np.empty((len(g) - 1, h, w, 2), np.float32)
    for t in range(1, len(g)):
        f = cv2.calcOpticalFlowFarneback(small[t], small[t - 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) / scale
        maps[t - 1, ..., 0] = xx + f[..., 0]
        maps[t - 1, ..., 1] = yy + f[..., 1]
    return maps


def warp_error(x: np.ndarray, maps: np.ndarray, where: np.ndarray) -> float:
    """원본의 흐름으로 x 의 앞 프레임을 옮겨 지금 프레임과의 평균 절대 차(회색, 0~255). where 는 T×H×W."""
    import cv2
    g = gray(x)
    errs, n = 0.0, 0
    for t in range(1, len(g)):
        w = where[t] & where[t - 1]
        if not w.any():
            continue
        prev = cv2.remap(g[t - 1], maps[t - 1, ..., 0], maps[t - 1, ..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        d = np.abs(g[t] - prev)[w]
        errs += float(d.sum())
        n += d.size
    return errs / n if n else float("nan")


def masked_mean(m: np.ndarray, where: np.ndarray) -> float:
    return float(m[where].mean()) if where.any() else float("nan")
