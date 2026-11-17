"""가벼운 위험 분류기 HazNet: 클립(2초) → 휘도 깜빡임 · 적색 깜빡임 · 공간 패턴 위험 여부.

목적: G 가 만든 데이터(정답 = 하드 판정기)로 학습해 **실제 위험 영상**에서 얼마나 통하는지 본다.
판정기를 대체하려는 것이 아니라 '우리 합성 데이터로 학습한 모델이 실제로 일반화하는가'를 재는 도구다.

구조 — 갈래마다 합성곱 5층(팀원의 정지 이미지 5층 CNN 과 같은 깊이):
  시간 갈래 (점멸): 48프레임 × 90×160, 입력 = 선형 RGB · 휘도 · 빨강 비율
      L1 공간 3×3 /2 → L2~L4 시간 합성곱(k5, 팽창 1·2·4, 수용 영역 29프레임 ≈ 1.2초) → L5 1×1 → 휘도·적색 지도
  공간 갈래 (무늬): 프레임 2장 × 360×640, 입력 = 로그 휘도 · 국소 대비
      L1 7×7 /2 → L2 5×5 /2 → L3 3×3 /2 → L4 5×5 팽창 3(수용 영역 약 120 px ≈ 6°) → L5 1×1 → 무늬 지도
  지도(45×80) → 프레임별 '위험 면적' = 화면 평균 → 시간 최대 → 클립 로짓 = a·(면적 − b)  (a, b 학습)
  판정 규칙(면적 문턱)을 그대로 닮은 머리라 해석이 쉽다: 지도는 '위험한 픽셀', 클립 판정은 '그 면적'.

학습 목표: 클립 라벨(판정기) BCE + 지도 보조 손실(판정기 튜브의 면적 비율, 합성 데이터에만 있음).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

HEADS = ("lum", "red", "pattern")
T, TH, TW = 48, 90, 160          # 시간 갈래 입력
P, PH, PW = 2, 360, 640          # 공간 갈래 입력 (프레임 2장)
MH, MW = 45, 80                  # 지도 해상도
PAT_IDX = (12, 36)               # 2초 클립에서 공간 갈래에 쓰는 프레임 (24 fps 기준 0.5 s, 1.5 s)
WHITE_CD = 200.0


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def tem_features(bgr_u8: torch.Tensor) -> torch.Tensor:
    """[B,T,H,W,3] uint8 BGR → [B,5,T,H,W]: 선형 R·G·B, 휘도 Y(=L/200), 빨강 비율 R/(R+G+B)."""
    x = srgb_to_linear(bgr_u8.float() / 255.0)
    b, g, r = x[..., 0], x[..., 1], x[..., 2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    red = r / (r + g + b + 1e-3)
    return torch.stack([r, g, b, y, red], 1)


def pat_features(bgr_u8: torch.Tensor) -> torch.Tensor:
    """[B,P,H,W,3] uint8 → [B*P,2,H,W]: 로그 휘도, 국소 대비(휘도 − 9×9 평균)/100 cd."""
    x = srgb_to_linear(bgr_u8.float() / 255.0)
    y = (0.2126 * x[..., 2] + 0.7152 * x[..., 1] + 0.0722 * x[..., 0]) * WHITE_CD    # cd/m²
    y = y.flatten(0, 1)[:, None]
    loc = y - F.avg_pool2d(y, 9, 1, 4, count_include_pad=False)
    return torch.cat([torch.log1p(y) / 5.3, loc / 100.0], 1)


def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(4, c)


class HazNet(nn.Module):
    def __init__(self, width: int = 24):
        super().__init__()
        w = width
        a = nn.GELU
        self.tem = nn.Sequential(
            nn.Conv3d(5, w, (1, 3, 3), (1, 2, 2), (0, 1, 1)), _gn(w), a(),                     # L1 공간
            nn.Conv3d(w, w, (5, 1, 1), padding=(2, 0, 0)), _gn(w), a(),                        # L2 시간
            nn.Conv3d(w, w, (5, 1, 1), padding=(4, 0, 0), dilation=(2, 1, 1)), _gn(w), a(),    # L3
            nn.Conv3d(w, w, (5, 1, 1), padding=(8, 0, 0), dilation=(4, 1, 1)), _gn(w), a(),    # L4
            nn.Conv3d(w, 2, 1))                                                                # L5 → 휘도·적색
        self.pat = nn.Sequential(
            nn.Conv2d(2, w, 7, 2, 3), _gn(w), a(),                                             # L1
            nn.Conv2d(w, w, 5, 2, 2), _gn(w), a(),                                             # L2
            nn.Conv2d(w, w, 3, 2, 1), _gn(w), a(),                                             # L3
            nn.Conv2d(w, w, 5, 1, 6, dilation=3), _gn(w), a(),                                 # L4
            nn.Conv2d(w, 1, 1))                                                                # L5 → 무늬
        # 클립 머리: logit = a·(최대 면적 − b). b 초기값 = 판정 규칙의 면적 문턱(25%, 40%)
        self.head_a = nn.Parameter(torch.full((3,), 20.0))
        self.head_b = nn.Parameter(torch.tensor([0.25, 0.25, 0.40]))

    def forward(self, tem_u8: torch.Tensor, pat_u8: torch.Tensor) -> dict:
        """tem_u8 [B,T,90,160,3], pat_u8 [B,P,360,640,3] (uint8 BGR) → {'logit' [B,3], 'tmap' [B,2,T,45,80], 'pmap' [B,P,45,80]}."""
        b = tem_u8.shape[0]
        tmap = self.tem(tem_features(tem_u8))                                  # [B,2,T,45,80]
        pmap = self.pat(pat_features(pat_u8)).view(b, -1, MH, MW)             # [B,P,45,80]
        area_t = torch.sigmoid(tmap).mean((-1, -2)).amax(-1)                   # [B,2]
        area_p = torch.sigmoid(pmap).mean((-1, -2)).amax(-1, keepdim=True)    # [B,1]
        area = torch.cat([area_t, area_p], 1)
        logit = self.head_a * (area - self.head_b)
        return {"logit": logit, "area": area, "tmap": tmap, "pmap": pmap}


def loss_fn(out: dict, y: torch.Tensor, tub_t: torch.Tensor | None, tub_p: torch.Tensor | None,
            aux: float = 1.0) -> tuple[torch.Tensor, dict]:
    """y [B,3] 0/1 · tub_t [B,2,T,45,80] · tub_p [B,P,45,80] (0~1 면적 비율, 없으면 None)."""
    cls = F.binary_cross_entropy_with_logits(out["logit"].float(), y.float())
    parts = {"cls": float(cls)}
    loss = cls
    if aux and tub_t is not None:
        lt = F.binary_cross_entropy_with_logits(out["tmap"].float(), tub_t.float())
        lp = F.binary_cross_entropy_with_logits(out["pmap"].float(), tub_p.float())
        loss = loss + aux * (lt + lp)
        parts["aux"] = float(lt + lp)
    return loss, parts


# ---------------- 증강 (GPU, 라벨을 바꾸지 않는 범위로만) ----------------

def _rot_grid(angle: torch.Tensor, h: int, w: int, n: int) -> torch.Tensor:
    """화면 비율을 고려한 회전 격자 (각 표본마다 다른 각도)."""
    c, s = torch.cos(angle), torch.sin(angle)
    theta = torch.zeros(n, 2, 3, device=angle.device)
    theta[:, 0, 0], theta[:, 0, 1] = c, -s * h / w
    theta[:, 1, 0], theta[:, 1, 1] = s * w / h, c
    return F.affine_grid(theta, (n, 1, h, w), align_corners=False)


def _warp(x: torch.Tensor, angle: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """x [B,C,H,W] float → 표본별 회전(반사 채움)."""
    grid = _rot_grid(angle, x.shape[-2], x.shape[-1], x.shape[0])
    return F.grid_sample(x, grid, mode=mode, padding_mode="reflection", align_corners=False)


def augment(tem: torch.Tensor, pat: torch.Tensor, tub_t: torch.Tensor | None, tub_p: torch.Tensor | None,
            g: torch.Generator, max_deg: float = 25.0, quality: bool = True):
    """tem [B,T,h,w,3] u8, pat [B,P,H,W,3] u8 → 같은 꼴(uint8). 튜브도 같은 기하 변환.

    - 좌우·상하 뒤집기, 회전 ±max_deg°(반사 채움): 판정 규칙(면적·횟수·cpd·대비)이 방향과 무관
    - 시간 뒤집기: 1초 창 전이 횟수가 같다
    - 화질: 축소→확대(0.6~1배), 가우스 흐림 σ ≤ 0.8, 잡음 σ ≤ 2/255, 블록 양자화(압축 흉내) — 무늬 대비를 크게 깎지 않게 약하게
    - 밝기 이득 0.94~1.06: 20 cd/m²·160 cd/m² 문턱 근처 표본의 라벨이 바뀌지 않도록 작게
    """
    dev = tem.device
    b = tem.shape[0]
    r = lambda *s: torch.rand(*s, generator=g, device="cpu").to(dev)
    tem = tem.float(); pat = pat.float()
    has_tub = tub_t is not None
    # 뒤집기
    for dim_t, dim_p, dim_tt, dim_tp in ((3, 3, 4, 3), (2, 2, 3, 2)):          # 가로, 세로
        m = r(b) < 0.5
        if m.any():
            tem[m] = tem[m].flip(dim_t - 1)
            pat[m] = pat[m].flip(dim_p - 1)
            if has_tub:
                tub_t[m] = tub_t[m].flip(dim_tt - 1)
                tub_p[m] = tub_p[m].flip(dim_tp - 1)
    # 시간 뒤집기
    m = r(b) < 0.5
    if m.any():
        tem[m] = tem[m].flip(1)
        if has_tub:
            tub_t[m] = tub_t[m].flip(2)
        pat[m] = pat[m].flip(1)
        if has_tub:
            tub_p[m] = tub_p[m].flip(1)
    # 회전 (표본의 절반)
    ang = (r(b) * 2 - 1) * math.radians(max_deg) * (r(b) < 0.5).float()
    if (ang != 0).any():
        bt, tt, h, w, c = tem.shape
        x = tem.permute(0, 1, 4, 2, 3).reshape(b, tt * c, h, w)
        tem = _warp(x, ang).view(b, tt, c, h, w).permute(0, 1, 3, 4, 2)
        _, pp, H, W, _ = pat.shape
        x = pat.permute(0, 1, 4, 2, 3).reshape(b, pp * c, H, W)
        pat = _warp(x, ang).view(b, pp, c, H, W).permute(0, 1, 3, 4, 2)
        if has_tub:
            k = tub_t.shape[1]
            tub_t = _warp(tub_t.flatten(1, 2), ang).view_as(tub_t)
            tub_p = _warp(tub_p, ang)
            del k
    if quality:
        # 축소→확대 (공간 갈래)
        m = (r(b) < 0.4).nonzero().flatten()
        for i in m.tolist():
            f = float(0.6 + 0.4 * r(1))
            x = pat[i].permute(0, 3, 1, 2)
            small = F.interpolate(x, scale_factor=f, mode="area")
            pat[i] = F.interpolate(small, size=x.shape[-2:], mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        # 가우스 흐림
        m = (r(b) < 0.3).nonzero().flatten()
        for i in m.tolist():
            sig = float(0.3 + 0.5 * r(1))
            k = torch.arange(-2, 3, device=dev, dtype=torch.float32)
            k = torch.exp(-k ** 2 / (2 * sig ** 2)); k = k / k.sum()
            x = pat[i].permute(0, 3, 1, 2).reshape(-1, 1, *pat.shape[2:4])
            x = F.conv2d(F.pad(x, (2, 2, 0, 0), mode="reflect"), k.view(1, 1, 1, 5))
            x = F.conv2d(F.pad(x, (0, 0, 2, 2), mode="reflect"), k.view(1, 1, 5, 1))
            pat[i] = x.view(pat.shape[1], 3, *pat.shape[2:4]).permute(0, 2, 3, 1)
        # 블록 양자화(압축 흉내): 8×8 블록 평균 쪽으로 조금 당김
        m = (r(b) < 0.3).nonzero().flatten()
        for i in m.tolist():
            x = pat[i].permute(0, 3, 1, 2)
            blk = F.interpolate(F.avg_pool2d(x, 8), scale_factor=8, mode="nearest")
            q = float(0.15 * r(1))
            pat[i] = ((1 - q) * x + q * blk).permute(0, 2, 3, 1)
        # 이득·잡음 (두 갈래 같게)
        gain = 0.94 + 0.12 * r(b)
        gain = torch.where(r(b) < 0.5, gain, torch.ones_like(gain))
        tem = tem * gain.view(b, 1, 1, 1, 1)
        pat = pat * gain.view(b, 1, 1, 1, 1)
        sig = 2.0 * r(b) * (r(b) < 0.5).float()
        tem = tem + torch.randn(tem.shape, device=dev) * sig.view(b, 1, 1, 1, 1)
        pat = pat + torch.randn(pat.shape, device=dev) * sig.view(b, 1, 1, 1, 1)
    tem = tem.round().clamp(0, 255).to(torch.uint8)
    pat = pat.round().clamp(0, 255).to(torch.uint8)
    return tem, pat, tub_t, tub_p
