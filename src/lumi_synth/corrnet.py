"""보정 모델 H (docs/CORRECTION_PROTOCOL.md §3.1) — 분석(인수분해 시공간 트랜스포머) → 물리 역산 → 정보 없음 채우기·정제.

입력  X: [B,T,3,H,W] sRGB 0~1 (BGR 순서, 학습 기본 T=192 · 216×384)
출력  dict
  Y      [B,T,3,H,W] 보정 결과 (선형 빛)
  A, B   [B,T,3,H/2,W/2] 예측한 층 (X_lin = A·Y_lin + B)
  haz    [B,T,1,H/2,W/2] 위험 지도 로짓 (팀원 Partial B2 픽셀 마스크로 지도)
  lost   [B,T,1,H,W] 정보 없음 가중(A 가 작은 곳)

구성과 근거
  1. 분석: 프레임별 CNN(1/8) → [공간 어텐션(프레임 안) ↔ 시간 어텐션(같은 위치, T 전체)] × depth.
     지도 먼저 → 복원 조건(VDFP, arXiv 2605.21079), 인수분해 시공간 어텐션(LPM, arXiv 2607.13460).
     시간 어텐션 길이 = 클립 전체(8 s) — 연장 기준 5 s 창을 직접 본다.
  2. 물리 역산: Y_phys = (X_lin − B̂)/Â  (Â ≥ 0.05). O = G + F 꼴의 열화 모델(BlazeBVD, Lei 외 2023)을 아핀으로 일반화.
  3. 채우기·정제: 시간 팽창 3D 합성곱 잔차망 — 입력 [X, Y_phys, Â, B̂, lost, 분석 특징]. 출력 = Y0 + 잔차(RMFAT 의 잔차 출력).
     정보 없음(Â < 0.05) 은 시간 이웃·분석 특징으로 채운다(Lei 외 아틀라스·BlazeBVD 이웃 질감의 발상, 학습으로 대체).
변형(절제, §3.4): use_attn=False → 시간 어텐션 대신 3D 합성곱(창 약 1 s) · use_phys=False → 역산 없이 잔차 직접 예측.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

A_MIN = 0.05


def srgb_to_lin(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).clamp_min(0) ** 2.4)


def lin_to_srgb(y: torch.Tensor) -> torch.Tensor:
    y = y.clamp(0, 1)
    return torch.where(y <= 0.0031308, y * 12.92, 1.055 * y.clamp_min(1e-8) ** (1 / 2.4) - 0.055)


def conv(ci, co, s=1, k=3):
    return nn.Sequential(nn.Conv2d(ci, co, k, s, k // 2), nn.GroupNorm(8, co), nn.GELU())


class AttnBlock(nn.Module):
    """토큰 [N, L, C] 에 사전 정규화 어텐션 + MLP."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.h = heads

    def forward(self, x, pos=None):
        n, l, c = x.shape
        y = self.n1(x if pos is None else x + pos)
        q, k, v = self.qkv(y).view(n, l, 3, self.h, c // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(a.transpose(1, 2).reshape(n, l, c))
        return x + self.mlp(self.n2(x))


class FactorizedST(nn.Module):
    """[B,T,C,h,w] → 공간 어텐션(프레임마다 h·w 토큰) ↔ 시간 어텐션(위치마다 T 토큰)."""

    def __init__(self, dim: int, heads: int, depth: int, t_max: int = 256, use_attn: bool = True):
        super().__init__()
        self.use_attn = use_attn
        self.sp = nn.ModuleList(AttnBlock(dim, heads) for _ in range(depth))
        if use_attn:
            self.tm = nn.ModuleList(AttnBlock(dim, heads) for _ in range(depth))
            self.tpos = nn.Parameter(torch.zeros(1, t_max, dim))
            nn.init.trunc_normal_(self.tpos, std=0.02)
        else:                                            # 절제 H−attn: 시간 3D 합성곱(커널 5, 팽창 1·2·3 → 약 1 s)
            self.tm = nn.ModuleList(nn.Conv1d(dim, dim, 5, padding=2 * (i % 3 + 1), dilation=i % 3 + 1) for i in range(depth))

    def forward(self, f):
        b, t, c, h, w = f.shape
        for i in range(len(self.sp)):
            x = f.permute(0, 1, 3, 4, 2).reshape(b * t, h * w, c)
            x = checkpoint(self.sp[i], x, use_reentrant=False) if self.training else self.sp[i](x)
            x = x.view(b, t, h * w, c).permute(0, 2, 1, 3).reshape(b * h * w, t, c)
            if self.use_attn:
                fn = lambda z, j=i: self.tm[j](z, self.tpos[:, :t])
                x = checkpoint(fn, x, use_reentrant=False) if self.training else fn(x)
            else:
                x = x + F.gelu(self.tm[i](x.transpose(1, 2))).transpose(1, 2)
            f = x.view(b, h, w, t, c).permute(0, 3, 4, 1, 2)
        return f


class Block21D(nn.Module):
    """(2+1)D 잔차 블록: 공간 3×3 → 시간 3(팽창 d). 3D 3×3×3 보다 연산·메모리가 약 1/3."""

    def __init__(self, w: int, d: int):
        super().__init__()
        self.s = nn.Conv3d(w, w, (1, 3, 3), padding=(0, 1, 1))
        self.t = nn.Conv3d(w, w, (3, 1, 1), padding=(d, 0, 0), dilation=(d, 1, 1))
        self.n = nn.GroupNorm(4, w)

    def forward(self, h):
        return h + F.gelu(self.n(self.t(F.gelu(self.s(h)))))


class Refine3D(nn.Module):
    """시간 팽창 (2+1)D 합성곱 잔차망 (채우기·정제). 시간 수용 영역 ±17 프레임."""

    def __init__(self, cin: int, width: int = 24):
        super().__init__()
        self.inp = nn.Conv3d(cin, width, (1, 3, 3), padding=(0, 1, 1))
        self.blocks = nn.ModuleList(Block21D(width, d) for d in (1, 2, 4, 8, 1))
        self.out = nn.Conv3d(width, 3, (1, 3, 3), padding=(0, 1, 1))
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):                                # x [B,C,T,H,W]
        h = F.gelu(self.inp(x))
        for blk in self.blocks:                          # 블록마다 체크포인트(학습 중 메모리 최고점을 낮춤)
            h = checkpoint(blk, h, use_reentrant=False) if self.training else blk(h)
        return self.out(h)


class CorrNet(nn.Module):
    def __init__(self, dim: int = 96, heads: int = 4, depth: int = 4, use_attn: bool = True, use_phys: bool = True):
        super().__init__()
        self.use_phys = use_phys
        self.e1, self.e2, self.e3 = conv(6, 32, 2), conv(32, 64, 2), conv(64, dim, 2)        # 1/2, 1/4, 1/8
        self.st = FactorizedST(dim, heads, depth, use_attn=use_attn)
        self.d2 = conv(dim + 64, 64)                                                          # 1/4
        self.d1 = conv(64 + 32, 48)                                                           # 1/2
        self.head = nn.Conv2d(48, 3 + 3 + 1 + 8, 3, padding=1)                                # A, B, 위험, 특징 8
        self.refine = Refine3D(3 + 3 + 3 + 3 + 1 + 8)

    def forward(self, X: torch.Tensor) -> dict:
        b, t, _, H, W = X.shape
        xl = srgb_to_lin(X)
        inp = torch.cat([X, xl], 2).flatten(0, 1)                                             # [BT,6,H,W]
        f1 = self.e1(inp); f2 = self.e2(f1); f3 = self.e3(f2)
        g = self.st(f3.view(b, t, *f3.shape[1:])).flatten(0, 1)
        u = self.d2(torch.cat([F.interpolate(g, size=f2.shape[-2:], mode="bilinear", align_corners=False), f2], 1))
        u = self.d1(torch.cat([F.interpolate(u, size=f1.shape[-2:], mode="bilinear", align_corners=False), f1], 1))
        o = self.head(u)                                                                      # [BT,15,H/2,W/2]
        A = torch.sigmoid(o[:, 0:3]) * 1.2
        Bv = F.softplus(o[:, 3:6]) * 0.25
        haz, feat = o[:, 6:7], o[:, 7:]
        up = lambda z: F.interpolate(z, size=(H, W), mode="bilinear", align_corners=False)
        Au, Bu, fu = up(A), up(Bv), up(feat)
        xlf = xl.flatten(0, 1)
        lost = torch.sigmoid((A_MIN - Au.mean(1, keepdim=True)) / 0.01)
        if self.use_phys:
            yp = ((xlf - Bu) / Au.clamp_min(A_MIN)).clamp(0, 1)
            y0 = (1 - lost) * yp + lost * xlf
        else:                                            # 절제 H−phys: 역산 없이 입력에서 잔차
            yp, y0 = xlf, xlf
        r_in = torch.cat([xlf, yp, Au, Bu, lost, fu], 1).view(b, t, -1, H, W).transpose(1, 2)
        res = self.refine(r_in)
        Y = (y0.view(b, t, 3, H, W) + res.transpose(1, 2)).clamp(0, 1)
        sh = lambda z: z.view(b, t, *z.shape[1:])
        return {"Y": Y, "A": sh(A), "B": sh(Bv), "haz": sh(haz), "lost": sh(lost), "Yphys": sh(yp)}
