"""ROI 탐지기와 보정기 — G 가 만든 쌍(원본, 위험 영상)으로 '탐지 → 보정'이 되는지 검증하는 작은 모델.

단계와 합격 기준은 docs/GDR_VERIFY.md 에 있다. 이 파일은 그 단계들이 함께 쓰는 부품이다.

    judge_tubes    하드 판정기(measure, stripe_freq) → 종류별 A 라벨 '튜브'(T×H×W bool, 언제·어디가 위험한가)
    RoiNet         영상 → 종류별(휘도·적색·줄무늬) 위험 튜브 확률 (1/4 해상도)
    Corrector      영상 + 튜브 → 보정 영상. 튜브 밖은 구조적으로 입력 그대로 (G 의 '마스크 밖은 원본'과 같은 원리)
    rule_correct   딥러닝 없는 대조군: 튜브 안에서 시간 평균(휘도·적색) / 공간 흐림(줄무늬)
    지표           tube_iou, clip_decision, psnr, texture_ratio, temporal_ratio, seam_ratio

원칙 (PLAN.md §2·§4, generate.py)
    - 정답은 생성 의도가 아니라 하드 판정기의 결과다. 의도가 '위험'이어도 판정이 안전이면 튜브는 비어 있다.
    - 튜브는 '판정기가 위험하다고 한 창에서 화면 전이를 만든 픽셀'(휘도·적색) 또는 '위험 프레임의 줄무늬 픽셀'이다.
      G 가 바꾼 영역 전체(mask_changed)가 아니다 — 기준에 못 미치는 약한 변화는 튜브에 들지 않는다.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from lumi_synth import measure
from lumi_synth import stripe_freq as sf

HAZARDS = ("lum", "red", "stripe")
SCALE = 4                    # 탐지 출력 해상도 = 입력의 1/4 (768×432 → 192×108)
CLIP_AREA_MIN = 0.125        # 클립 판정: 어느 프레임이든 예측 면적이 화면의 12.5% 이상이면 '위험'으로 본다.
                             # 구현 선택 — 기준 면적(휘도·적색 25%, 줄무늬 40%)의 가장 작은 값의 절반
STRIPE_JUDGE_FPS = 10.0      # 줄무늬 판정은 초당 약 10장만 본다(플레이어 '빠른 판정'과 같은 방식)


# ============================================================================================
# 1. 라벨: 하드 판정기 → 튜브
# ============================================================================================
def judge_tubes(frames_bgr: np.ndarray, fps: float, stripes: bool = True) -> tuple[dict, dict]:
    """영상(T×H×W×3 uint8 BGR) → ({종류: 튜브 T×H×W bool}, 요약).

    휘도·적색: hazard_frames(위험한 1초 창의 프레임) ⊗ hazard_mask(그 창에서 화면 전이를 만든 픽셀).
    줄무늬: 쿼드트리+가버 판정의 프레임별 줄무늬 픽셀 ∧ hazard_frames. 초당 약 10장만 판정하고 사이 프레임은
    앞 결과를 이어 쓴다(지속 규칙도 그 표본률로) — make_playlist.detect(fast=True) 와 같다.
    stripes=False 면 줄무늬 판정을 건너뛴다(튜브는 비고 요약의 stripe 는 None).
    """
    t_len = len(frames_bgr)
    fl = measure.analyze(frames_bgr, fps)
    rd = measure.analyze_red(frames_bgr, fps)
    tubes = {"lum": fl["hazard_frames"][:, None, None] & fl["hazard_mask"][None],
             "red": rd["hazard_frames"][:, None, None] & rd["hazard_mask"][None]}
    summary = {"lum": {"dangerous": bool(fl["dangerous"]), "max_flashes": float(fl["max_flashes"]),
                       "peak_area": float(fl["peak_area"])},
               "red": {"dangerous": bool(rd["dangerous"]), "max_flashes": float(rd["max_flashes"]),
                       "peak_area": float(rd["peak_area"])}}
    if stripes:
        step = max(1, int(round(fps / STRIPE_JUDGE_FPS)))
        pt = sf.analyze_pattern_freq(frames_bgr[::step], fps / step, model=None, method="quadtree", refine=True)
        rep = lambda a: np.repeat(np.asarray(a), step, axis=0)[:t_len]          # noqa: E731
        tubes["stripe"] = rep(pt["stripes"]) & rep(pt["hazard_frames"])[:, None, None]
        summary["stripe"] = {"dangerous": bool(pt["dangerous"]), "peak_area": float(pt["peak_stripe_area"]),
                             "max_sec": float(pt["max_stripe_sec"])}
    else:
        tubes["stripe"] = np.zeros(tubes["lum"].shape, bool)
        summary["stripe"] = None
    summary["dangerous"] = any(v["dangerous"] for v in summary.values() if v)
    return tubes, summary


def stack_tubes(tubes: dict) -> np.ndarray:
    """{종류: T×H×W} → 3×T×H×W bool (HAZARDS 순서)."""
    return np.stack([tubes[k] for k in HAZARDS])


def pool_tubes(t: torch.Tensor, scale: int = SCALE) -> torch.Tensor:
    """[B,C,T,H,W] 0/1 → 1/scale 해상도의 면적 평균(0~1). 탐지기의 학습 목표."""
    return F.avg_pool3d(t.float(), (1, scale, scale))


# ============================================================================================
# 2. 모델
# ============================================================================================
def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(8, c), c)


class Block3d(nn.Module):
    """잔차 블록: 공간 3×3 → 시간 3(dilation). 시간 dilation 을 키워 1초(24프레임) 이상을 본다."""

    def __init__(self, c: int, dilation: int = 1):
        super().__init__()
        self.sp = nn.Conv3d(c, c, (1, 3, 3), padding=(0, 1, 1))
        self.tm = nn.Conv3d(c, c, (3, 1, 1), padding=(dilation, 0, 0), dilation=(dilation, 1, 1))
        self.n1, self.n2 = _gn(c), _gn(c)

    def forward(self, x):
        h = F.gelu(self.n1(self.sp(x)))
        h = self.n2(self.tm(h))
        return F.gelu(x + h)


class _Encoder(nn.Module):
    """프레임별 2D 줄기(1/2, 1/4) + 시간 블록. 줄기는 전체 해상도를 직접 본다(줄무늬 7 cpd = 주기 3.3 px)."""

    def __init__(self, c_in: int, width: int, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.s1 = nn.Conv3d(c_in, width // 2, (1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2))
        self.s2 = nn.Conv3d(width // 2, width, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.n1, self.n2 = _gn(width // 2), _gn(width)
        self.blocks = nn.Sequential(*[Block3d(width, d) for d in dilations])

    def forward(self, x):
        f2 = F.gelu(self.n1(self.s1(x)))            # 1/2
        f4 = F.gelu(self.n2(self.s2(f2)))           # 1/4
        return f2, self.blocks(f4)


N_COLOR_FEATS = 11
N_COLOR_FEATS2 = 15


def color_features(x: torch.Tensor, spans: tuple[int, ...] = (1,)) -> torch.Tensor:
    """sRGB BGR [B,3,T,H,W] 0~1 → 판정 규칙이 쓰는 양 11채널 (fp32 로 계산).

    BGR 3 · 선형 상대 휘도 Y · CIE u′ · v′ · 선형 R/(R+G+B) · R−G−B · 프레임 간 |ΔY| · |Δu′v′| · |Δ비율|.
    색도는 measure 와 같은 식(거의 검정은 D65). 규칙에 나오는 양을 미리 계산해 주어, 탐지기가 G 특유의 모양이 아니라
    규칙의 양(색도 차 0.2, 포화 비율 0.8, 휘도 변화)에 기대도록 한다(2026-11-15, 적색 일반화 실패 대응).
    spans=(1, 2, 4) ('color2'): 2·4 프레임 간 |ΔY|·|Δu′v′| 를 더한다(15채널). 1프레임 차만 주면 탐지기가 사각파처럼 한
    프레임에 크게 바뀌는 깜빡임에만 기대어, 사인·삼각파 휘도 깜빡임(프레임마다 조금씩 바뀜)을 놓쳤다(v4_roi_b)."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        s = x.float()
        lin = torch.where(s <= 0.04045, s / 12.92, ((s.clamp(min=0) + 0.055) / 1.055) ** 2.4)
        b, g, r = lin[:, 0], lin[:, 1], lin[:, 2]
        y = 0.0722 * b + 0.7152 * g + 0.2126 * r
        X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
        Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
        d = (X + 15 * y + 3 * Z).clamp(min=1e-6)
        ok = y >= measure.RED_BLACK_Y
        u = torch.where(ok, 4 * X / d, torch.full_like(y, measure.D65_UV[0]))
        v = torch.where(ok, 9 * y / d, torch.full_like(y, measure.D65_UV[1]))
        ratio = r / (r + g + b).clamp(min=1e-6)
        exc = r - g - b

        def dt(a, k=1):
            return F.pad((a[:, k:] - a[:, :-k]).abs(), (0, 0, 0, 0, k, 0))

        def duv(k=1):
            return F.pad(torch.sqrt((u[:, k:] - u[:, :-k]) ** 2 + (v[:, k:] - v[:, :-k]) ** 2 + 1e-12), (0, 0, 0, 0, k, 0))
        feats = [y * 4, (u - 0.2) * 5, (v - 0.47) * 5, ratio * 2 - 1, exc * 4, dt(y) * 8, duv() * 5, dt(ratio) * 2]
        for k in spans[1:]:
            feats += [dt(y, k) * 8, duv(k) * 5]
        return torch.cat([s] + [f[:, None] for f in feats], 1)


class RoiNet(nn.Module):
    """영상 [B,3,T,H,W](0~1, BGR) → 위험 튜브 로짓 [B,3,T,H/4,W/4] (HAZARDS 순서).

    면적 기준(화면의 25%·40%)은 화면 전체를 봐야 알 수 있어, 1/16 해상도까지 내려가고 프레임별 화면 평균을
    더해 준다(global context). 횟수 기준(1초에 3번 초과)은 시간 dilation 으로 본다.
    features='color': 입력을 color_features(11채널)로 바꿔 넣는다. 'rgb'(기본, v1~v4 의 roi_a): BGR 그대로.
    """

    def __init__(self, width: int = 32, features: str = "rgb", n_out: int = len(HAZARDS)):
        super().__init__()
        w = width
        self.features = features
        self.enc = _Encoder({"color": N_COLOR_FEATS, "color2": N_COLOR_FEATS2}.get(features, 3), w)
        self.d8 = nn.Conv3d(w, 2 * w, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.b8 = nn.Sequential(Block3d(2 * w, 1), Block3d(2 * w, 4))
        self.d16 = nn.Conv3d(2 * w, 2 * w, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.b16 = nn.Sequential(Block3d(2 * w, 2), Block3d(2 * w, 8))
        self.ctx = nn.Conv3d(4 * w, 2 * w, 1)
        self.u8 = nn.Conv3d(4 * w, 2 * w, 1)
        self.c8 = Block3d(2 * w, 1)
        self.u4 = nn.Conv3d(3 * w, w, 1)
        self.c4 = Block3d(w, 1)
        self.head = nn.Conv3d(w, n_out, 1)       # n_out=1: 정보 손실 추정기(M_loss, E3)

    def forward(self, x):
        if self.features == "color":
            x = color_features(x)
        elif self.features == "color2":
            x = color_features(x, (1, 2, 4))
        _, f4 = self.enc(x)
        f8 = self.b8(F.gelu(self.d8(f4)))
        f16 = self.b16(F.gelu(self.d16(f8)))
        g = f16.mean(dim=(3, 4), keepdim=True).expand_as(f16)                 # 프레임별 화면 평균
        f16 = F.gelu(self.ctx(torch.cat([f16, g], 1)))
        up = F.interpolate(f16, size=f8.shape[2:], mode="nearest")
        f8 = self.c8(F.gelu(self.u8(torch.cat([f8, up], 1))))
        up = F.interpolate(f8, size=f4.shape[2:], mode="nearest")
        f4 = self.c4(F.gelu(self.u4(torch.cat([f4, up], 1))))
        return self.head(f4)


class Corrector(nn.Module):
    """영상 x [B,3,T,H,W] + 종류별 튜브 m [B,3,T,H,W](0~1, HAZARDS 순서) → 보정 영상 y = x + max(m)·δ.

    튜브를 곱하므로 세 튜브가 모두 0 인 곳은 구조적으로 입력 그대로다. 종류를 따로 주는 이유: 종류마다 할 일이
    다르다(휘도는 밝기 변화를, 적색은 색을, 줄무늬는 무늬를 되돌린다). 화면 전체 평균 같은 전역 연산이 없어서(완전 합성곱)
    잘라 낸 조각으로 학습하고 전체 화면에 쓸 수 있다.
    """

    def __init__(self, width: int = 32, loss_in: bool = False):
        super().__init__()
        w = width
        self.loss_in = loss_in                   # E3: 정보 손실 지도 M_loss(0~1) 한 채널을 더 받는다
        c = 6 + (1 if loss_in else 0)
        self.enc = _Encoder(c, w, dilations=(1, 2, 4, 8, 1, 2))
        self.u2 = nn.Conv3d(w + w // 2, w // 2, (1, 3, 3), padding=(0, 1, 1))
        self.t2 = nn.Conv3d(w // 2, w // 2, (3, 1, 1), padding=(1, 0, 0))
        self.u1 = nn.Conv3d(w // 2 + c, w // 2, (1, 3, 3), padding=(0, 1, 1))
        self.out = nn.Conv3d(w // 2, 3, (1, 3, 3), padding=(0, 1, 1))
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)                                          # 처음에는 아무것도 바꾸지 않는다

    def forward(self, x, m, lmap=None):
        inp = torch.cat([x, m] + ([lmap if lmap is not None else torch.zeros_like(m[:, :1])] if self.loss_in else []), 1)
        f2, f4 = self.enc(inp)
        h = F.interpolate(f4, size=f2.shape[2:], mode="trilinear", align_corners=False)
        h = F.gelu(self.u2(torch.cat([h, f2], 1)))
        h = F.gelu(self.t2(h))
        h = F.interpolate(h, size=x.shape[2:], mode="trilinear", align_corners=False)
        h = F.gelu(self.u1(torch.cat([h, inp], 1)))
        delta = self.out(h)
        return (x + m.amax(dim=1, keepdim=True) * delta).clamp(0.0, 1.0)


def soften(m: torch.Tensor, grow: int = 2, blur: int = 2) -> torch.Tensor:
    """튜브 [B,C,T,H,W] 0/1 → 가장자리가 부드러운 튜브(0~1). 조금 넓힌 뒤(grow) 흐린다(blur).

    PLAN §1 '보정 경계는 부드럽게'. 넓히는 이유: 튜브 경계 바로 밖에도 기준에 못 미치는 약한 변화가 남아 있다.
    """
    b, c, t, h, w = m.shape
    x = m.reshape(b * c * t, 1, h, w).float()
    if grow:
        x = F.max_pool2d(x, 2 * grow + 1, stride=1, padding=grow)
    if blur:
        x = F.avg_pool2d(F.pad(x, (blur,) * 4, mode="replicate"), 2 * blur + 1, stride=1)
    return x.reshape(b, c, t, h, w)


# ============================================================================================
# 3. 손실
# ============================================================================================
def roi_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE(면적 평균 목표) + 종류별 소프트 다이스. target [B,3,T,h,w] 0~1."""
    bce = F.binary_cross_entropy_with_logits(logits.float(), target)
    p = torch.sigmoid(logits.float())
    inter = (p * target).sum(dim=(2, 3, 4))
    denom = p.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice = 1.0 - (2 * inter + 1.0) / (denom + 1.0)                           # 빈 목표·빈 예측이면 0
    return bce + dice.mean()


def correct_loss(y: torch.Tensor, clean: torch.Tensor, m: torch.Tensor, lambda_t: float = 1.0) -> torch.Tensor:
    """튜브 안에서 원본과의 L1 + 프레임 간 변화의 L1(깜빡임 잔여를 직접 벌줌)."""
    wsum = m.sum().clamp(min=1.0) * 3
    l1 = ((y - clean).abs() * m).sum() / wsum
    mt = torch.minimum(m[:, :, 1:], m[:, :, :-1])
    dt = (((y[:, :, 1:] - y[:, :, :-1]) - (clean[:, :, 1:] - clean[:, :, :-1])).abs() * mt).sum() / (mt.sum().clamp(min=1.0) * 3)
    return l1 + lambda_t * dt


# ============================================================================================
# 4. 지표 (numpy)
# ============================================================================================
def tube_iou(pred: np.ndarray, true: np.ndarray) -> float:
    """두 bool 배열의 IoU. 둘 다 비어 있으면 nan(평균에서 뺀다)."""
    u = np.logical_or(pred, true).sum()
    return float(np.logical_and(pred, true).sum() / u) if u else float("nan")


def clip_decision(prob: np.ndarray, thr: float = 0.5, area_min: float = CLIP_AREA_MIN) -> bool:
    """튜브 확률 T×h×w → 이 클립에 그 종류의 위험이 있는가 (어느 프레임이든 예측 면적 ≥ area_min)."""
    return bool(((prob >= thr).mean(axis=(1, 2)) >= area_min).any())


def clip_decision_rule(series, fps: float, area: float, min_sec: float) -> bool:
    """판정기와 같은 꼴의 클립 판정: 프레임별 예측 면적이 area 를 넘는 프레임이 1초 창 안에 min_sec 이상이면 위험.

    clip_decision(어느 한 프레임이라도 12.5%)은 종류와 무관한 느슨한 규칙이라 줄무늬(기준 40%·0.5초)의 경계 안전
    영상을 위험으로 잡았다(v7: 정밀도 0.31). 줄무늬 기준과 같은 꼴로 판정하면 원인이 '문턱'인지 '탐지'인지 가를 수 있다."""
    a = np.asarray(series, dtype=np.float64)
    cand = a > area
    w = measure.window_frames(fps)
    need = max(1, int(np.ceil(min_sec * fps - 1e-9)))
    cs = np.concatenate([[0], np.cumsum(cand)])
    t = len(cand)
    counts = cs[np.minimum(np.arange(t) + w, t)] - cs[:t]
    return bool((counts >= need).any())


def psnr(a: np.ndarray, b: np.ndarray, where: np.ndarray | None = None) -> float:
    """uint8 영상 두 개의 PSNR(dB). where(T×H×W bool)를 주면 그 픽셀만. 차이가 0 이면 inf."""
    d = (a.astype(np.float32) - b.astype(np.float32)) ** 2
    if where is not None:
        if not where.any():
            return float("nan")
        d = d[where]
    mse = float(d.mean())
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def _gray(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) @ np.array([0.114, 0.587, 0.299], np.float32)


def texture_ratio(x: np.ndarray, clean: np.ndarray, where: np.ndarray) -> float:
    """윤곽선 에너지(4-이웃 라플라시안 절댓값) 비율 출력/원본. 1 = 장면 그대로, < 1 = 뭉개짐, > 1 = 무늬가 남음."""
    def lap(v):
        g = _gray(v)
        out = np.zeros_like(g)
        out[:, 1:-1, 1:-1] = np.abs(4 * g[:, 1:-1, 1:-1] - g[:, :-2, 1:-1] - g[:, 2:, 1:-1] - g[:, 1:-1, :-2] - g[:, 1:-1, 2:])
        return out
    den = lap(clean)[where].mean() if where.any() else 0.0
    return float(lap(x)[where].mean() / den) if den > 0 else float("nan")


def temporal_ratio(x: np.ndarray, clean: np.ndarray, where: np.ndarray) -> float:
    """프레임 간 변화량 비율 출력/원본. 1 = 원본과 같은 움직임, > 1 = 깜빡임이 남음, < 1 = 움직임이 뭉개짐."""
    w = where[1:] & where[:-1]
    if not w.any():
        return float("nan")
    den = np.abs(np.diff(_gray(clean), axis=0))[w].mean()
    return float(np.abs(np.diff(_gray(x), axis=0))[w].mean() / max(den, 1e-6))


def seam_ratio(x: np.ndarray, clean: np.ndarray, where: np.ndarray) -> float:
    """튜브 경계를 가로지르는 밝기 차이의 비율 출력/원본. 1 = 이음새 없음, > 1 = 경계가 눈에 띔."""
    gx, gc = _gray(x), _gray(clean)
    eh = where[:, :, 1:] != where[:, :, :-1]
    ev = where[:, 1:] != where[:, :-1]
    if not (eh.any() or ev.any()):
        return float("nan")
    num = np.abs(np.diff(gx, axis=2))[eh].sum() + np.abs(np.diff(gx, axis=1))[ev].sum()
    den = np.abs(np.diff(gc, axis=2))[eh].sum() + np.abs(np.diff(gc, axis=1))[ev].sum()
    return float(num / max(den, 1e-6))


# ============================================================================================
# 5. 대조군: 규칙 보정 (PLAN §1 '판정기로 탐지 + 규칙으로 영역만 감광')
# ============================================================================================
def rule_correct(frames_bgr: np.ndarray, tubes: dict, fps: float) -> np.ndarray:
    """딥러닝 없는 보정. 원본을 모른다고 가정하고 입력 영상과 튜브만 쓴다.

    휘도·적색 튜브: 선형 빛에서 앞뒤 0.5초(합쳐 약 1초)의 시간 평균으로 바꾼다 → 깜빡임이 평균 밝기·평균 색이 된다.
    줄무늬 튜브: 프레임마다 공간 흐림(가우시안, σ = 1.3 cpd 주기의 절반)으로 줄무늬 대비를 없앤다.
    """
    import cv2
    lin = measure_linear(frames_bgr)
    out = lin.copy()
    flick = tubes["lum"] | tubes["red"]
    if flick.any():
        k = max(1, int(round(fps / 2)))
        cs = np.concatenate([np.zeros((1,) + lin.shape[1:], np.float32), np.cumsum(lin, axis=0, dtype=np.float32)])
        t = np.arange(len(lin))
        lo, hi = np.maximum(0, t - k), np.minimum(len(lin), t + k + 1)
        mean = (cs[hi] - cs[lo]) / (hi - lo)[:, None, None, None]
        out = np.where(flick[..., None], mean, out)
    if tubes["stripe"].any():
        ppd = measure.pixels_per_degree(frames_bgr.shape[2], frames_bgr.shape[1])
        sigma = 0.5 * ppd / 1.3
        for i in np.flatnonzero(tubes["stripe"].any(axis=(1, 2))):
            blur = cv2.GaussianBlur(out[i], (0, 0), sigma)
            out[i] = np.where(tubes["stripe"][i][..., None], blur, out[i])
    return linear_to_u8(out)


def measure_linear(frames_bgr: np.ndarray) -> np.ndarray:
    """uint8 sRGB → 선형 빛 float32 (IEC 61966-2-1)."""
    x = frames_bgr.astype(np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_u8(lin: np.ndarray) -> np.ndarray:
    x = np.clip(lin, 0.0, 1.0)
    s = np.where(x <= 0.0031308, x * 12.92, 1.055 * np.maximum(x, 1e-8) ** (1 / 2.4) - 0.055)
    return np.round(s * 255.0).astype(np.uint8)
