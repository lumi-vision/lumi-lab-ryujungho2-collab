"""7종 위험 라벨 합성기 — 팀원 판별기(rockrovks B2 vdcorefs, stage3)의 출력 라벨 체계에 맞춘다.

| 라벨 | 핵심 조건 (팀원 config.PSEConfig) | 이 합성기가 만드는 방법 (목표 'hazard') |
|---|---|---|
| 휘도 플래시 | 전이 면적 > 25%, 반대 방향 전이 ≥ 7회/1 s, 플래시 시작 간격 < 334 ms | 영역(25~75%)에 빛 더하기·밝기 곱하기·단색, 반주기 1~3 프레임(24 fps → 4~12 Hz) |
| 빠른 장면 전환 | 전이 면적 ≥ 80%, 전이 ≥ 4회/1 s, 인접 전이 간격 < 334 ms (휘도 플래시는 제외) | 화면 전체: 어두운 장면 ↔ 밝은 다른 장면 컷, 또는 전체 밝기 곱하기. 반주기 4~6 프레임(4~6회/s) |
| 교집합 | 위 둘이 같은 프레임에서 성립 | 화면 전체 컷·전체 밝기, 반주기 1~3 프레임 |
| 적색 플래시 | 포화 적색 + 색도 거리 ≥ 0.2, 면적 > 25%, 전이 ≥ 7회/1 s | 영역을 적색 쌍(같은 휘도 빨강 ↔ 보라)으로 덮거나 장면 ↔ 빨강 물들이기, 반주기 1~3 프레임 |
| 휘도 연장 | 5 s 창 중 4 s 이상 전이 4~6회/1 s (휘도 플래시는 제외) | 영역(30~70%) 깜빡임, 반주기 4~6 프레임, 5.5 s 이상 지속 |
| 적색 연장 | 같은 조건, 적색 | 적색 쌍·물들이기, 반주기 4~6 프레임, 5.5 s 이상 |
| 정적 공간 패턴 | 1~10 cpd, ≥ 6쌍, 면적 > 40%, 대비 20 cd/m², 0.5 s 이상 | 무늬 7종(직선·동심원·방사형·나선·처프·체커·물결)을 영역(45~90%)에 1 s 이상 |

목표 'near' = 경계 바로 밖(반주기·지속·면적·대비·채도·cpd 중 하나를 살짝 모자라게) → 대부분 안전이거나 다른 라벨.
**정답은 의도가 아니라 판별기가 붙인다**(scripts/make_hazard7_dataset.py: mp4 로 저장 → 다시 디코드 → 판별).
⚠️ 결과는 광과민성 위험 자극이다. 메모리·파일로만 다루고 재생·미리보기하지 않는다.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from . import measure, oracle, stripe_freq

LABELS = ("휘도 플래시", "교집합", "빠른 장면 전환", "적색 플래시", "휘도 연장", "적색 연장", "정적 공간 패턴")
KINDS = ("lum_flash", "intersection", "scene", "red_flash", "lum_ext", "red_ext", "pattern")
KIND_LABEL = dict(zip(KINDS, LABELS))
PATTERNS = ("straight", "concentric", "radial", "spiral", "chirp", "checker", "wavy")
W_CD = measure.WHITE_CD


# ---------------- 시간 일정 ----------------

def toggle_state(n: int, t0: int, t1: int, halves, rng) -> np.ndarray:
    """프레임 t0~t1 사이에서 상태 0/1 을 바꾸는 일정. 반주기(프레임)는 halves 에서 매번 뽑는다(지터)."""
    s = np.zeros(n, np.float32)
    t, cur = t0, 1.0
    while t < t1:
        h = int(rng.choice(halves))
        s[t:min(t + h, t1)] = cur
        t += h
        cur = 1.0 - cur
    return s


def soften(s: np.ndarray, sigma: float) -> np.ndarray:
    """사각 상태를 시간으로 살짝 흐려 사인·삼각꼴 파형도 섞는다 (σ 프레임)."""
    if sigma <= 0:
        return s
    k = np.arange(-3, 4)
    g = np.exp(-k ** 2 / (2 * sigma ** 2)); g /= g.sum()
    return np.convolve(np.pad(s, 3, mode="edge"), g, mode="valid").astype(np.float32)


# ---------------- 영역 ----------------

def region(h: int, w: int, area: float, rng, shapes=("ellipse", "rectangle", "blobs")) -> np.ndarray:
    shape = str(rng.choice(shapes))
    c = (float(rng.uniform(0.3, 0.7)), float(rng.uniform(0.3, 0.7)))
    asp = float(rng.uniform(0.8, 2.4))
    return oracle.region_mask(shape, area, h, w, center=c, aspect=asp, seed=int(rng.integers(1 << 30)))


def moving(mask: np.ndarray, n: int, vel_px: tuple[float, float]) -> np.ndarray:
    """정지 마스크를 프레임마다 평행 이동 (화면 밖은 잘림) → T×H×W bool."""
    h, w = mask.shape
    out = np.empty((n, h, w), bool)
    m8 = mask.astype(np.uint8)
    for t in range(n):
        M = np.float32([[1, 0, vel_px[0] * t], [0, 1, vel_px[1] * t]])
        out[t] = cv2.warpAffine(m8, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0) > 0
    return out


def feather(m: np.ndarray, px: float) -> np.ndarray:
    """경계를 px 만큼 부드럽게 (0~1 float). 날카로운 경계는 G 의 알려진 한계였다."""
    f = m.astype(np.float32)
    if px <= 0:
        return f
    k = int(2 * round(2 * px) + 1)
    if f.ndim == 2:
        return cv2.GaussianBlur(f, (k, k), px)
    return np.stack([cv2.GaussianBlur(x, (k, k), px) for x in f])


# ---------------- 동적 마스크 (현실의 위험이 생기는 방식을 흉내) ----------------
#  static    고정 영역 (전광판·화면 속 화면)
#  drift     부드러운 무작위 이동·확대·회전 (카메라가 움직이는 장면 속 광원)
#  anchored  장면 내용에서 영역을 키우고(밝은 곳·비슷한 색) 광학 흐름으로 따라감 (폭발·창문·조명 기구·물체에 붙은 빛)
#  spot      한 점에서 퍼지는 빛 (플래시·섬광·번개) — 가우스 감쇠, 터질 때 커짐
#  multi     광원 여러 개가 따로 깜빡임 (경광등·조명 쇼) — 위상이 달라 동시 면적이 줄 수 있음
#  perspective (무늬) 평면에 붙은 무늬를 원근 변환 (벽·바닥·옷의 줄무늬)
#  (무늬는 static·anchored·perspective 만: 정적 공간 패턴 라벨은 멈춘 무늬다)

MASK_MODES = ("static", "drift", "anchored", "spot", "multi")
PATTERN_MASK_MODES = ("static", "anchored", "perspective")


def _smooth_walk(n: int, rng, scale: float, smooth: int = 24) -> np.ndarray:
    """부드러운 무작위 경로 (n,) — 흰 잡음을 길게 흐려 누적."""
    x = np.cumsum(rng.normal(0, 1, n + 2 * smooth))
    k = np.ones(smooth) / smooth
    x = np.convolve(x, k, mode="same")[smooth:smooth + n]
    x = x - x[0]
    return (x / (np.abs(x).max() + 1e-6) * scale).astype(np.float32)


def drift_mask(m: np.ndarray, n: int, rng) -> np.ndarray:
    """정지 마스크 → 프레임마다 이동(화면의 ±15%)·확대(0.8~1.25)·회전(±15°)."""
    h, w = m.shape
    tx, ty = _smooth_walk(n, rng, rng.uniform(0.02, 0.10) * w), _smooth_walk(n, rng, rng.uniform(0.02, 0.10) * h)
    sc = 1 + _smooth_walk(n, rng, rng.uniform(0.0, 0.25))
    rot = _smooth_walk(n, rng, rng.uniform(0, 15))
    ys, xs = np.nonzero(m)
    cx, cy = (xs.mean(), ys.mean()) if len(xs) else (w / 2, h / 2)
    m8 = m.astype(np.uint8)
    out = np.empty((n, h, w), bool)
    for t in range(n):
        M = cv2.getRotationMatrix2D((float(cx), float(cy)), float(rot[t]), float(sc[t]))
        M[0, 2] += tx[t]; M[1, 2] += ty[t]
        out[t] = cv2.warpAffine(m8, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0) > 0
    return out


def grow_region(frame: np.ndarray, area: float, rng) -> np.ndarray:
    """장면 내용에서 영역을 키운다: 씨앗(밝은 곳 위주) → Lab 색 거리 순으로 픽셀을 더해 목표 면적까지.

    물체·조명·하늘 같은 '장면 속 덩어리' 모양이 나오도록 색 거리에 공간 거리를 섞는다(SLIC 와 비슷한 생각).
    """
    h, w = frame.shape[:2]
    s = 4
    small = cv2.resize(frame, (w // s, h // s), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    hh, ww = lab.shape[:2]
    L = lab[..., 0]
    if rng.random() < 0.7:                                    # 밝은 곳에서 시작 (광원·창문·하늘)
        cand = np.argwhere(L >= np.quantile(L, 0.85))
    else:
        cand = np.argwhere(np.ones_like(L, bool))
    sy, sx = cand[rng.integers(len(cand))]
    yy, xx = np.mgrid[:hh, :ww]
    dcol = np.linalg.norm(lab - lab[sy, sx], axis=-1)
    dsp = np.hypot(yy - sy, xx - sx) / max(hh, ww) * 255
    d = dcol + rng.uniform(0.3, 1.2) * dsp
    d = cv2.GaussianBlur(d, (0, 0), 1.5)
    thr = np.quantile(d, min(0.995, area))
    m = (d <= thr).astype(np.uint8)
    n_lab, lab_img = cv2.connectedComponents(m)
    if n_lab > 2:                                             # 씨앗과 이어진 덩어리 + 큰 덩어리들만
        keep = np.zeros(n_lab, bool); keep[lab_img[sy, sx]] = True
        sizes = np.bincount(lab_img.ravel())
        keep |= sizes >= 0.15 * sizes[1:].max()
        keep[0] = False
        m = keep[lab_img].astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0


def track_mask(m0: np.ndarray, frames: np.ndarray, t_ref: int) -> np.ndarray:
    """프레임 t_ref 의 마스크를 광학 흐름(Farneback, 1/4 해상도)으로 앞뒤 프레임에 옮긴다 → T×H×W bool."""
    n, h, w = frames.shape[:3]
    s = 4
    g = [cv2.cvtColor(cv2.resize(f, (w // s, h // s), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY) for f in frames]
    hh, ww = g[0].shape
    gy, gx = np.mgrid[:hh, :ww].astype(np.float32)
    out = np.zeros((n, hh, ww), np.float32)
    out[t_ref] = cv2.resize(m0.astype(np.float32), (ww, hh), interpolation=cv2.INTER_AREA)
    for step in (1, -1):
        t = t_ref
        while 0 <= t + step < n:
            nt = t + step
            flow = cv2.calcOpticalFlowFarneback(g[nt], g[t], None, 0.5, 3, 15, 3, 5, 1.2, 0)   # nt → t 방향
            out[nt] = cv2.remap(out[t], gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderValue=0)
            t = nt
    up = np.stack([cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR) for x in out])
    return up >= 0.5


def spot_weight(h: int, w: int, n: int, area: float, rng, s: np.ndarray) -> np.ndarray:
    """한 점에서 퍼지는 빛: w = exp(−r²/2σ²). σ 는 '가중 ≥ 0.5' 면적이 area 가 되게. 켜질 때 조금 커짐."""
    cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    lo, hi = 1.0, float(max(h, w)) * 3
    for _ in range(30):                                      # 면적(빛이 닿는 곳 = 가우스 ≥ 0.3) 맞추기
        sig = (lo + hi) / 2
        if (np.exp(-r2 / (2 * sig ** 2)) >= 0.3).mean() < area:
            lo = sig
        else:
            hi = sig
    grow = 1 + rng.uniform(0, 0.25) * s                      # 켜진 동안 반경이 최대 25% 커짐
    vx, vy = _smooth_walk(n, rng, 0.08 * w), _smooth_walk(n, rng, 0.08 * h)
    out = np.empty((n, h, w), np.float32)
    for t in range(n):
        rr = (xx - cx - vx[t]) ** 2 + (yy - cy - vy[t]) ** 2
        out[t] = np.exp(-rr / (2 * (sig * grow[t]) ** 2))
    # 0.3 등고선 밖은 0 — 꼬리가 화면 전체로 퍼지면 '빛이 닿는 면적'이 목표보다 커져(80% 넘으면 장면 전환) 의도와 어긋난다
    return np.clip((out - 0.3) / 0.4, 0, 1)


def dynamic_region(r: dict, clean: np.ndarray, s: np.ndarray, rng) -> np.ndarray:
    """레시피의 mask_mode 대로 시간에 따라 변하는 영역 가중 T×H×W float32 (0~1)."""
    n, h, w = clean.shape[:3]
    mode, area = r.get("mask_mode", "static"), r["area"]
    t_ref = (r["t0"] + r["t1"]) // 2
    if mode == "spot":
        return spot_weight(h, w, n, area, rng, s)
    if mode == "anchored":
        m = track_mask(grow_region(clean[t_ref], area, rng), clean, t_ref)
    elif mode == "multi":
        k = int(rng.integers(2, 5))
        m = np.zeros((h, w), bool)
        for _ in range(k):
            m |= region(h, w, min(0.95, 1.35 * area / k), rng, shapes=("ellipse", "rectangle"))   # 겹침 보정
        m = drift_mask(m, n, rng) if rng.random() < 0.5 else np.broadcast_to(m, (n, h, w))
    else:
        m = region(h, w, area, rng)
        if mode == "drift":
            m = drift_mask(m, n, rng)
    return feather(np.asarray(m), r["feather"]) if np.asarray(m).ndim == 3 else feather(m, r["feather"])


def perspective_pattern(plum: np.ndarray, area: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """무늬를 평면에 붙여 원근 변환: 화면 안 사각형(면적 ≈ area)으로 보낸다 → (무늬 휘도, 영역 0/1)."""
    h, w = plum.shape
    side = math.sqrt(area)
    cx, cy = rng.uniform(0.35, 0.65) * w, rng.uniform(0.35, 0.65) * h
    hw, hh = side * w / 2, side * h / 2
    jit = lambda: rng.uniform(-0.25, 0.25)
    dst = np.float32([[cx - hw * (1 + jit()), cy - hh * (1 + jit())], [cx + hw * (1 + jit()), cy - hh * (1 + jit())],
                      [cx + hw * (1 + jit()), cy + hh * (1 + jit())], [cx - hw * (1 + jit()), cy + hh * (1 + jit())]])
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    Hm = cv2.getPerspectiveTransform(src, dst)
    pl = cv2.warpPerspective(plum, Hm, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    m = cv2.warpPerspective(np.ones((h, w), np.uint8), Hm, (w, h), flags=cv2.INTER_NEAREST) > 0
    return pl, m


# ---------------- 섞는 식 (선형 빛) ----------------

_LUT = oracle.srgb_to_linear(np.arange(256, dtype=np.uint8)).astype(np.float32)


def lin(u8: np.ndarray) -> np.ndarray:
    """uint8 sRGB → 선형 float32 (8 s 클립이 크므로 float64 를 피한다)."""
    return _LUT[u8]


def u8(x: np.ndarray) -> np.ndarray:
    return oracle.linear_to_srgb(np.clip(x, 0, 1))


def lum_modulate(clean: np.ndarray, a: np.ndarray, s: np.ndarray, mode: str, p: dict) -> np.ndarray:
    """a: T×H×W(또는 H×W) 0~1 영역 가중, s: T 상태 0~1. mode: add(빛 더하기) | gain(밝기 곱) | flat(단색 회색)."""
    x = lin(clean)
    a = a if a.ndim == 3 else np.broadcast_to(a, x.shape[:3])
    k = (a * s[:, None, None])[..., None]
    if mode == "add":
        col = np.asarray(p.get("light_bgr", (1.0, 1.0, 1.0)), np.float32)       # 빛의 색 (휘도 1 로 맞춘 선형 BGR)
        col = col / float(col @ np.array([0.0722, 0.7152, 0.2126], np.float32))
        y = x + k * (p["delta_cd"] / W_CD) * col
    elif mode == "gain":
        g = p["gain_lo"] + (p["gain_hi"] - p["gain_lo"]) * s[:, None, None, None]
        y = x * (1 - a[..., None]) + x * g * a[..., None] + a[..., None] * s[:, None, None, None] * p.get("lift", 0.0)
    elif mode == "flat":
        lo, hi = p["lo_cd"] / W_CD, p["hi_cd"] / W_CD
        v = lo + (hi - lo) * s[:, None, None, None]
        y = x * (1 - a[..., None]) + v * a[..., None]
    else:
        raise ValueError(mode)
    return u8(y)


def scene_cut(clean: np.ndarray, other: np.ndarray, s: np.ndarray, p: dict) -> np.ndarray:
    """화면 전체 컷: s=0 → 어둡게 한 장면 A, s=1 → 밝힌 다른 장면 B. 컷이 아닌 동안(일정 밖)은 A 원본."""
    a, b = lin(clean), lin(other)
    dark = a * p["dark_gain"]
    bright = np.clip(b * p["bright_gain"] + p["bright_lift"], 0, 1)
    active = p["active"][:, None, None, None]
    y = np.where(s[:, None, None, None] >= 0.5, bright, dark)
    y = active * y + (1 - active) * a
    return u8(y)


def red_modulate(clean: np.ndarray, a: np.ndarray, s: np.ndarray, mode: str, p: dict) -> np.ndarray:
    """적색: pair = 영역을 색 쌍 A(빨강) ↔ B(보라 쪽)로 덮기, tint = 장면 ↔ 장면 휘도의 포화 빨강으로 물들이기."""
    x = lin(clean)
    a = a if a.ndim == 3 else np.broadcast_to(a, x.shape[:3])
    w = a[..., None]
    if mode == "pair":
        ca = lin(np.array(p["a_rgb"][::-1], np.uint8))           # BGR
        cb = lin(np.array(p["b_rgb"][::-1], np.uint8))
        col = np.where(s[:, None, None, None] >= 0.5, cb, ca)
        y = x * (1 - w) + col * w
    elif mode == "tint":
        lum = (x @ np.array([0.0722, 0.7152, 0.2126]))[..., None]          # BGR 순서
        lum = np.maximum(lum, p["min_y"])
        ratio = p.get("tint_sat", p["sat"])
        g_over_r = (1 - ratio) / (2 * ratio)
        r = lum / (0.2126 + 0.7874 * g_over_r)
        red = np.concatenate([r * g_over_r, r * g_over_r, r], -1)           # B, G, R
        rho = (s[:, None, None, None] * w)
        # 꺼진 상태: 영역 안 장면의 색을 desat 만큼 뺀다(휘도 유지) — 장면이 이미 주황·갈색이면 빨강과 색도 차가 0.2 에 못 미친다
        gray = np.repeat(lum, 3, -1)
        act = p["active"][:, None, None, None] * w
        off = x * (1 - act * p["desat"]) + gray * act * p["desat"]
        y = off * (1 - rho) + np.clip(red, 0, 1) * rho
    else:
        raise ValueError(mode)
    return u8(y)


# ---------------- GPU(torch) 섞기 — 위 numpy 식과 같은 식. render(device=...) 가 쓴다 ----------------

def _tl(x_u8, dev):
    import torch
    lut = torch.as_tensor(_LUT, device=dev)
    return lut[torch.as_tensor(np.ascontiguousarray(x_u8), device=dev).long()]


def _tu8(y):
    import torch
    y = y.clamp(0, 1)
    c = torch.where(y <= 0.0031308, y * 12.92, 1.055 * y.pow(1 / 2.4) - 0.055)
    return (c * 255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()


def _ta(a, n, dev):
    import torch
    t = torch.as_tensor(np.ascontiguousarray(a, dtype=np.float32), device=dev)
    return t if t.ndim == 3 else t.expand(n, *t.shape)


def mix_torch(kind: str, mode: str, clean, a, s, p: dict, dev, other=None, plum=None) -> np.ndarray:
    """한 묶음(프레임 몇 장)을 GPU 에서 섞는다 → uint8 BGR. 식은 lum_modulate·scene_cut·red_modulate·pattern_overlay 와 같다."""
    import torch
    x = _tl(clean, dev)
    n = x.shape[0]
    st = torch.as_tensor(s, device=dev, dtype=torch.float32)[:, None, None, None]
    act = torch.as_tensor(p["active"], device=dev, dtype=torch.float32)[:, None, None, None]
    if kind == "scene_cut":
        b = _tl(other, dev)
        y = torch.where(st >= 0.5, (b * p["bright_gain"] + p["bright_lift"]).clamp(0, 1), x * p["dark_gain"])
        return _tu8(act * y + (1 - act) * x)
    w = _ta(a, n, dev)[..., None]
    if kind == "lum":
        if mode == "add":
            col = torch.tensor(p.get("light_bgr", (1.0, 1.0, 1.0)), device=dev, dtype=torch.float32)
            col = col / (col @ torch.tensor([0.0722, 0.7152, 0.2126], device=dev))
            y = x + w * st * (p["delta_cd"] / W_CD) * col
        elif mode == "gain":
            g = p["gain_lo"] + (p["gain_hi"] - p["gain_lo"]) * st
            y = x * (1 - w) + x * g * w + w * st * p.get("lift", 0.0)
        else:
            lo, hi = p["lo_cd"] / W_CD, p["hi_cd"] / W_CD
            y = x * (1 - w) + (lo + (hi - lo) * st) * w
        return _tu8(y)
    if kind == "red":
        if mode == "pair":
            ca = torch.as_tensor(_LUT[np.array(p["a_rgb"][::-1], np.uint8)], device=dev)
            cb = torch.as_tensor(_LUT[np.array(p["b_rgb"][::-1], np.uint8)], device=dev)
            return _tu8(x * (1 - w) + torch.where(st >= 0.5, cb, ca) * w)
        lum = (x @ torch.tensor([0.0722, 0.7152, 0.2126], device=dev))[..., None].clamp_min(p["min_y"])
        ratio = p.get("tint_sat", p["sat"])
        gr = (1 - ratio) / (2 * ratio)
        r = lum / (0.2126 + 0.7874 * gr)
        red = torch.cat([r * gr, r * gr, r], -1).clamp(0, 1)
        rho = st * w
        aw = act * w
        off = x * (1 - aw * p["desat"]) + lum.expand_as(x) * aw * p["desat"]
        return _tu8(off * (1 - rho) + red * rho)
    # pattern
    pat = torch.as_tensor(plum / W_CD, device=dev, dtype=torch.float32)[None, ..., None]
    base = (1 - p["keep"]) * pat + p["keep"] * x
    ww = w * st
    return _tu8(x * (1 - ww) + base * ww)


def pattern_lum(kind: str, h: int, w: int, cpd: float, lo: float, hi: float, rng) -> np.ndarray:
    """무늬 휘도(cd/m²) H×W. 물결(wavy)만 여기서 직접 만들고 나머지는 stripe_freq.render_pattern."""
    ppd = measure.pixels_per_degree(w, h, 0.0)
    center = (float(rng.uniform(0.3, 0.7)), float(rng.uniform(0.3, 0.7)))
    ang = float(rng.uniform(0, 180))
    if kind == "wavy":
        f = cpd / ppd
        yy, xx = np.mgrid[:h, :w].astype(np.float64)
        a = math.radians(ang)
        u = xx * math.cos(a) + yy * math.sin(a)
        v = -xx * math.sin(a) + yy * math.cos(a)
        amp, lam = rng.uniform(0.3, 1.2) / f, rng.uniform(4, 10) / f
        phi = 2 * np.pi * f * (u + amp * np.sin(2 * np.pi * v / lam))
        return np.where(np.sin(phi) >= 0, hi, lo).astype(np.float32)
    kw = {"angle": ang, "center": center}
    if kind in ("radial", "spiral"):
        kw["spokes"] = int(rng.integers(16, 48))
    if kind == "chirp":
        kw["chirp"] = float(rng.uniform(0.5, 2.0))
    lum, _ = stripe_freq.render_pattern(kind, h, w, ppd, cpd, lo, hi, **kw)
    return lum


def pattern_overlay(clean: np.ndarray, a: np.ndarray, plum: np.ndarray, on: np.ndarray, keep: float) -> np.ndarray:
    """무늬를 영역에 덮는다(on[t]=1 인 프레임만). keep = 장면 결을 남기는 비율(0~0.25)."""
    x = lin(clean)
    pat = (plum / W_CD)[None, ..., None]
    base = (1 - keep) * pat + keep * x
    w = (a[None] if a.ndim == 2 else a)[..., None] * on[:, None, None, None]
    return u8(x * (1 - w) + base * w)


# ---------------- 레시피 ----------------

def sample_recipe(kind: str, intent: str, seed: int, n: int, fps: float) -> dict:
    """종류·의도·시드 → 레시피(재현 가능). n = 클립 프레임 수(8 s · 24 fps = 192)."""
    rng = np.random.default_rng(seed)
    r: dict = {"kind": kind, "intent": intent, "seed": seed, "n": n, "fps": fps}
    fast, mid = [1, 2, 3], [4, 5, 6]
    sec = lambda s: int(round(s * fps))
    near_how = None
    if kind in ("lum_flash", "red_flash"):
        halves, dur, area = fast, rng.uniform(1.5, 4.0), rng.uniform(0.3, 0.62)
        if intent == "near":
            near_how = str(rng.choice(["slow", "area", "weak"]))
            if near_how == "slow":
                halves, dur = [5, 6, 7], rng.uniform(1.5, 3.0)          # 3~5회/s, 연장 되기엔 짧게
            elif near_how == "area":
                area = rng.uniform(0.12, 0.22)
        r.update(halves=halves, dur=dur, area=area, near=near_how)
    elif kind in ("scene", "intersection"):
        halves = mid if kind == "scene" else fast
        dur = rng.uniform(1.5, 3.5)
        area = 1.0
        if intent == "near":
            near_how = str(rng.choice(["slow", "area"]))
            if near_how == "slow":
                halves = [9, 10, 12] if kind == "scene" else mid         # 교집합 near = 장면 전환 속도 → 다른 라벨
            else:
                area = rng.uniform(0.55, 0.72)                           # 80% 미달 → 장면 전환 아님(휘도 플래시는 될 수 있음)
        r.update(halves=halves, dur=dur, area=area, near=near_how)
    elif kind in ("lum_ext", "red_ext"):
        halves, dur, area = mid, rng.uniform(5.8, 7.5), rng.uniform(0.3, 0.7)
        if intent == "near":
            near_how = str(rng.choice(["short", "slow"]))
            if near_how == "short":
                dur = rng.uniform(2.5, 3.6)                              # 5 s 창 중 4 s 미달
            else:
                halves = [9, 10, 12]                                     # 2~3회/s
        r.update(halves=halves, dur=dur, area=area, near=near_how)
    elif kind == "pattern":
        area, cpd, dur = rng.uniform(0.48, 0.9), float(np.exp(rng.uniform(np.log(1.6), np.log(7.0)))), rng.uniform(1.2, 4.0)
        lo = rng.uniform(4, 20); hi = lo + rng.uniform(40, 120)
        if intent == "near":
            near_how = str(rng.choice(["area", "cpd", "contrast", "short"]))
            if near_how == "area":
                area = rng.uniform(0.2, 0.33)
            elif near_how == "cpd":
                cpd = float(rng.uniform(0.4, 0.75))
            elif near_how == "contrast":
                hi = lo + rng.uniform(8, 15)
            else:
                dur = rng.uniform(0.15, 0.35)
        r.update(area=area, cpd=cpd, dur=dur, lo=lo, hi=hi, pattern=str(rng.choice(PATTERNS)),
                 keep=float(rng.uniform(0, 0.2)), near=near_how)
    else:
        raise ValueError(kind)
    r["dur"] = float(min(r["dur"], n / fps - 0.5))
    r["t0"] = int(rng.integers(sec(0.2), max(sec(0.21), n - sec(r["dur"]) - sec(0.2))))
    r["t1"] = min(n, r["t0"] + sec(r["dur"]))
    r["sigma"] = float(rng.choice([0, 0, 0.4, 0.7]))
    r["feather"] = float(rng.uniform(0, 6))
    if kind == "pattern":
        r["mask_mode"] = str(rng.choice(PATTERN_MASK_MODES, p=[0.3, 0.35, 0.35]))
    elif r["area"] < 1:
        r["mask_mode"] = str(rng.choice(MASK_MODES, p=[0.15, 0.25, 0.3, 0.2, 0.1]))
    else:
        r["mask_mode"] = "full"
    warm = rng.uniform(-1, 1)                                   # 빛 색: 따뜻한(주황) ↔ 차가운(파랑), 대부분 흰색 가까이
    r["light_bgr"] = [float(1 - 0.35 * max(warm, 0)), 1.0, float(1 - 0.35 * max(-warm, 0))] if rng.random() < 0.5 else [1.0, 1.0, 1.0]
    if kind in ("lum_flash", "lum_ext") or (kind in ("scene", "intersection") and r["area"] < 1):
        r["mode"] = str(rng.choice(["add", "gain", "flat"] if r.get("near") != "weak" else ["add", "flat"]))
        r["delta_cd"] = float(rng.uniform(45, 130)) if not (r.get("near") == "weak") else float(rng.uniform(8, 15))
        r["gain_lo"], r["gain_hi"] = float(rng.uniform(0.05, 0.3)), float(rng.uniform(1.0, 1.6))
        r["lift"] = float(rng.uniform(0.15, 0.4))
        lo = float(rng.uniform(2, 30))
        r["lo_cd"], r["hi_cd"] = lo, lo + (float(rng.uniform(40, 150)) if r.get("near") != "weak" else float(rng.uniform(8, 15)))
    elif kind in ("scene", "intersection"):
        r["mode"] = str(rng.choice(["cut", "gain"]))
        r["dark_gain"], r["bright_gain"], r["bright_lift"] = float(rng.uniform(0.02, 0.15)), float(rng.uniform(1.0, 1.4)), float(rng.uniform(0.2, 0.45))
        r["gain_lo"], r["gain_hi"], r["lift"] = r["dark_gain"], r["bright_gain"], r["bright_lift"]
    elif kind in ("red_flash", "red_ext"):
        r["mode"] = str(rng.choice(["pair", "tint"]))
        sat = float(rng.uniform(0.86, 0.97)) if r.get("near") != "weak" else float(rng.uniform(0.55, 0.7))
        r["sat"] = sat
        r["tint_sat"] = float(rng.uniform(0.985, 0.997)) if r.get("near") != "weak" else sat   # 물들이기는 더 진한 빨강(u′ ≈ 0.45 이상)
        r["desat"] = float(rng.uniform(0.85, 1.0))
        # 적색 전환은 한 프레임 안에: 팀원 판별기는 66.8 ms 안의 색도 변화만 전이로 센다(24 fps 에선 바로 옆 프레임뿐).
        # 시간 흐림(σ)을 주면 Δu′v′ 가 프레임 여럿에 나뉘어 0.2 를 못 넘는다(d3: 적색 실패 11개 모두 물들이기 + σ>0 또는 색도 차 0.19)
        r["sigma"] = 0.0
        r["min_y"] = float(rng.uniform(0.03, 0.12))
        try:
            pr = oracle.red_pair(sat, float(rng.uniform(0.24, 0.45)), y_rel=float(rng.uniform(0.04, 0.16)))
            r["a_rgb"], r["b_rgb"] = list(pr["a_rgb"]), list(pr["b_rgb"])
        except ValueError:
            r["mode"] = "tint"
    return r


def render(r: dict, clean: np.ndarray, other: np.ndarray | None = None, device=None) -> tuple[np.ndarray, np.ndarray]:
    """레시피 → (영상 T×H×W×3 uint8 BGR, 합성 영역 가중 T×H×W float — 바뀐 곳). clean 은 n 프레임 이상.

    device(torch, 예: 'cuda') 를 주면 섞기를 GPU 에서 한다(마스크·무늬 만들기는 CPU). numpy 경로와 sRGB 반올림 차이 ±1 이내.
    """
    rng = np.random.default_rng(r["seed"] + 7)
    n, (h, w) = r["n"], clean.shape[1:3]
    clean = clean[:n]
    kind = r["kind"]
    if kind == "pattern":
        s = np.zeros(n, np.float32); s[r["t0"]:r["t1"]] = 1
    else:
        s = soften(toggle_state(n, r["t0"], r["t1"], r["halves"], rng), r["sigma"])
    plum = None
    if r["area"] >= 1:
        a = np.ones((h, w), np.float32)
    elif kind == "pattern":
        plum = pattern_lum(r["pattern"], h, w, r["cpd"], r["lo"], r["hi"], rng)
        if r["mask_mode"] == "perspective":
            plum, m = perspective_pattern(plum, r["area"], rng)
        elif r["mask_mode"] == "anchored":
            m = grow_region(clean[(r["t0"] + r["t1"]) // 2], r["area"], rng)
        else:
            m = region(h, w, r["area"], rng)
        a = feather(m, min(r["feather"], 2.0))
    else:
        a = dynamic_region(r, clean, s, rng)
    active = np.zeros(n, np.float32); active[r["t0"]:r["t1"]] = 1
    oth = (other[:n] if other is not None else clean[::-1]) if kind in ("scene", "intersection") else None
    out = np.empty_like(clean)
    # 프레임 24장씩 섞는다 — 8 s 전체를 float 로 한꺼번에 만들면 프로세스당 수~십수 GB 를 써서 메모리가 모자랐다(2026-11-20)
    for c0 in range(0, n, 24):
        c = slice(c0, min(n, c0 + 24))
        ac = a[c] if a.ndim == 3 else a
        p = dict(r); p["active"] = active[c]
        if device is not None:                                  # GPU 섞기
            if kind in ("scene", "intersection") and r["mode"] == "cut":
                out[c] = mix_torch("scene_cut", "cut", clean[c], None, s[c], p, device, other=oth[c])
            elif kind in ("lum_flash", "lum_ext", "scene", "intersection"):
                out[c] = mix_torch("lum", r["mode"], clean[c], ac, s[c], p, device)
            elif kind in ("red_flash", "red_ext"):
                out[c] = mix_torch("red", r["mode"], clean[c], ac, s[c], p, device)
            else:
                out[c] = mix_torch("pattern", "", clean[c], ac, s[c], p, device, plum=plum)
            continue
        if kind in ("lum_flash", "lum_ext") or (kind in ("scene", "intersection") and r["mode"] in ("add", "gain", "flat")):
            out[c] = lum_modulate(clean[c], ac, s[c], r["mode"], r)
        elif kind in ("scene", "intersection"):
            out[c] = scene_cut(clean[c], oth[c], s[c], p)
        elif kind in ("red_flash", "red_ext"):
            out[c] = red_modulate(clean[c], ac, s[c], r["mode"], p)
        else:
            out[c] = pattern_overlay(clean[c], ac, plum, s[c], r["keep"])
    aa = a if a.ndim == 3 else np.broadcast_to(a, (n, h, w))
    return out, aa
