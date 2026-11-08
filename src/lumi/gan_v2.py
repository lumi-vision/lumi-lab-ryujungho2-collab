"""G v2 — 판정 규칙(lumi.judge)을 학습 신호로 쓰는 국소 깜빡임 생성기.

E1(gan_synthesis.py)의 문제와 대응:
- Oracle 결과를 정답으로 흉내 내다 결정적이 됨 → Oracle 재구성 손실을 쓰지 않는다. 위험 수준은
  **미분 가능 판정기**(soft_transition_count)가 목표(문턱 넘기 / 문턱 근처 통과)에 맞추게 한다.
- 수용 영역 9복셀·복셀 단위 잡음 → U-Net(공간 1/4 축소) + 시간 팽창 합성곱 + **클립 단위 전역 z**(FiLM).
- 시간 파형을 조건으로 주지 않는다. 대신 여러 주파수의 사인·코사인 시간 특징을 주고, z가 이를 조합해
  깜빡임 방식을 고르게 한다.
- D는 **프레임 한 장(2D)**만 본다: 공간적 자연스러움만 판정하고, 깜빡임(프레임 간 변화)은 벌주지 않는다.
  D의 진짜는 실제 안전 영상 프레임이다(위험 영상은 시험 전용이라 쓰지 않음).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .judge import soft_transition_count

TIME_FREQS = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0)
TARGET_FAIL, TARGET_NEAR = 1.0, 0.0


def time_features(frames: int, fps: float, device=None) -> torch.Tensor:
    """[2·len(TIME_FREQS), T] 사인·코사인 시간 특징."""
    t = torch.arange(frames, device=device, dtype=torch.float32) / fps
    feats = [f(2 * math.pi * hz * t) for hz in TIME_FREQS for f in (torch.sin, torch.cos)]
    return torch.stack(feats)


def random_roi_masks(n: int, h: int, w: int, rng: np.random.Generator,
                     area_range=(0.3, 0.9)) -> np.ndarray:
    """[n,h,w] uint8 직사각형·타원 ROI. 면적 비율은 area_range에서 균등."""
    masks = np.zeros((n, h, w), np.uint8)
    yy, xx = np.mgrid[:h, :w]
    for i in range(n):
        a = rng.uniform(*area_range)
        aspect = rng.uniform(0.6, 1.6)
        if rng.random() < 0.5:   # 직사각형
            rw = min(w, int(round(np.sqrt(a * h * w * aspect))))
            rh = min(h, int(round(a * h * w / max(rw, 1))))
            x0, y0 = rng.integers(0, w - rw + 1), rng.integers(0, h - rh + 1)
            masks[i, y0:y0 + rh, x0:x0 + rw] = 1
        else:                    # 타원
            rx = min(w / 2, np.sqrt(a * h * w * aspect / np.pi))
            ry = min(h / 2, a * h * w / (np.pi * max(rx, 1)))
            cx, cy = rng.uniform(rx, w - rx), rng.uniform(ry, h - ry)
            masks[i] = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1).astype(np.uint8)
    return masks


class FiLMBlock(nn.Module):
    """Conv3d → GroupNorm → FiLM(z) → LeakyReLU."""

    def __init__(self, cin, cout, cond, kernel=(3, 3, 3), dilation=(1, 1, 1), stride=(1, 1, 1)):
        super().__init__()
        pad = tuple(d * (k // 2) for k, d in zip(kernel, dilation))
        self.conv = nn.Conv3d(cin, cout, kernel, stride=stride, padding=pad, dilation=dilation)
        self.norm = nn.GroupNorm(8, cout)
        self.film = nn.Linear(cond, 2 * cout)

    def forward(self, x, c):
        x = self.norm(self.conv(x))
        g, b = self.film(c).chunk(2, dim=1)
        x = x * (1 + g[:, :, None, None, None]) + b[:, :, None, None, None]
        return F.leaky_relu(x, 0.2)


class GeneratorV2(nn.Module):
    """입력: 원본 [B,3,T,H,W], ROI [B,1,T,H,W], 목표 [B], z [B,zdim]. 출력: 원본 + ROI·잔차."""

    def __init__(self, width: int = 32, zdim: int = 16, frames: int = 32, fps: float = 24.0,
                 output: str = "residual"):
        super().__init__()
        if output not in ("residual", "gain"):
            raise ValueError("output은 'residual' 또는 'gain'")
        self.zdim, self.fps, self.output = zdim, fps, output
        cond = zdim + 1
        cin = 3 + 1 + 1 + 2 * len(TIME_FREQS)
        w = width
        self.e1 = FiLMBlock(cin, w, cond)
        self.e2 = FiLMBlock(w, 2 * w, cond, stride=(1, 2, 2))
        self.e3 = FiLMBlock(2 * w, 4 * w, cond, stride=(1, 2, 2))
        # 시간 팽창: 수용 영역을 1초 이상으로 넓혀 깜빡임 횟수를 조율하게 한다
        self.t1 = FiLMBlock(4 * w, 4 * w, cond, kernel=(3, 1, 1), dilation=(2, 1, 1))
        self.t2 = FiLMBlock(4 * w, 4 * w, cond, kernel=(3, 1, 1), dilation=(4, 1, 1))
        self.t3 = FiLMBlock(4 * w, 4 * w, cond, kernel=(3, 1, 1), dilation=(8, 1, 1))
        self.d3 = FiLMBlock(4 * w + 2 * w, 2 * w, cond)
        self.d2 = FiLMBlock(2 * w + w, w, cond)
        self.head = nn.Conv3d(w, 3, 3, padding=1)
        # gain 출력: 1/4 해상도에서 밝기 이득(1채널)과 색 오프셋(3채널)을 내고 키워서 원본에 적용 → 질감 보존
        self.gain_head = nn.Conv3d(4 * w, 4, 3, padding=1)

    def forward(self, clean, mask, target, z):
        b, _, t, h, w = clean.shape
        cond = torch.cat([z, target[:, None]], dim=1)
        tf = time_features(t, self.fps, clean.device)[None, :, :, None, None].expand(b, -1, t, h, w)
        tgt = target[:, None, None, None, None].expand(b, 1, t, h, w)
        x = torch.cat([clean, mask, tgt, tf], dim=1)
        s1 = self.e1(x, cond)
        s2 = self.e2(s1, cond)
        x = self.e3(s2, cond)
        x = self.t3(self.t2(self.t1(x, cond), cond), cond)
        if self.output == "gain":
            g = F.interpolate(self.gain_head(x), size=(t, h, w), mode="trilinear", align_corners=False)
            gain = torch.exp(1.5 * torch.tanh(g[:, :1]))          # 약 0.22~4.5배 밝기
            offset = 0.5 * torch.tanh(g[:, 1:])                    # 색 오프셋 ±0.5
            altered = torch.clamp(clean * gain + offset, 0.0, 1.0)
            return clean + mask * (altered - clean)
        x = F.interpolate(x, size=s2.shape[2:], mode="trilinear", align_corners=False)
        x = self.d3(torch.cat([x, s2], 1), cond)
        x = F.interpolate(x, size=s1.shape[2:], mode="trilinear", align_corners=False)
        x = self.d2(torch.cat([x, s1], 1), cond)
        delta = torch.tanh(self.head(x))
        return torch.clamp(clean + mask * delta, 0.0, 1.0)


class FrameDiscriminator(nn.Module):
    """프레임 한 장 [N,3,H,W]의 공간적 자연스러움을 패치 단위로 판정."""

    def __init__(self, width: int = 32):
        super().__init__()
        w = width
        self.net = nn.Sequential(
            nn.Conv2d(3, w, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(w, 2 * w, 4, 2, 1), nn.GroupNorm(8, 2 * w), nn.LeakyReLU(0.2, True),
            nn.Conv2d(2 * w, 4 * w, 4, 2, 1), nn.GroupNorm(8, 4 * w), nn.LeakyReLU(0.2, True),
            nn.Conv2d(4 * w, 1, 3, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


def judge_loss(out: torch.Tensor, target: torch.Tensor, fps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """목표에 맞춘 판정 손실. 문턱 넘기: 소프트 전이 수 ≥ 8, 문턱 근처 통과: 2 ≤ 수 ≤ 5.5."""
    c = soft_transition_count(out.float(), fps)["count"]
    fail = target > 0.5
    loss_fail = F.relu(8.0 - c)
    loss_near = F.relu(c - 5.5) + F.relu(2.0 - c)
    return torch.where(fail, loss_fail, loss_near).mean(), c.detach()


def texture_loss(out: torch.Tensor, clean: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """ROI 안 질감 보존: 프레임별 라플라시안 에너지가 원본과 같도록 (밝기 배율로 정규화)."""
    k = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], device=out.device, dtype=torch.float32)
    def lap_energy(v):
        y = 0.2126 * v[:, 0] + 0.7152 * v[:, 1] + 0.0722 * v[:, 2]          # [B,T,H,W]
        b, t, h, w = y.shape
        l = F.conv2d(y.reshape(b * t, 1, h, w), k[None, None], padding=1).abs().reshape(b, t, h, w)
        m = mask[:, 0]
        return (l * m).sum((2, 3)) / (m.sum((2, 3)) + 1e-6), (y * m).sum((2, 3)) / (m.sum((2, 3)) + 1e-6)
    eo, yo = lap_energy(out.float())
    ec, yc = lap_energy(clean.float())
    # 밝기를 키우면 질감 대비도 비례해 커지는 것은 허용: 원본 질감 × (출력 밝기 / 원본 밝기) 기준
    expected = ec * (yo / (yc + 1e-3)).clamp(0.2, 5.0)
    return (torch.log(eo + 1e-4) - torch.log(expected.detach() + 1e-4)).abs().mean()


def frames_of(video: torch.Tensor, k: int) -> torch.Tensor:
    """[B,3,T,H,W] → 무작위 k개 프레임 [B·k,3,H,W]."""
    b, _, t, _, _ = video.shape
    idx = torch.randint(0, t, (b, k), device=video.device)
    return torch.stack([video[i, :, idx[i]] for i in range(b)]).transpose(1, 2).reshape(b * k, 3, *video.shape[3:])
