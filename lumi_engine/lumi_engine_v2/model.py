"""LUMI-7: 탐지와 국소 보정을 공동 학습하는 초경량 PyTorch 모델.

ConvGRU 기반 인과적 시공간 특징 추출과 Zero-init Global Context를 결합하여
단일 프레임 스트리밍(step)과 클립 배치 연산(forward)을 동일하게 지원한다.
"""
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ConvGRUCell(nn.Module):
    """두 합성곱으로 구현한 인과적 ConvGRU 셀."""

    def __init__(self, channels: int = 32):
        super().__init__()
        self.channels = channels
        self.gates = nn.Conv2d(2 * channels, 2 * channels, 3, padding=1)
        self.candidate = nn.Conv2d(2 * channels, channels, 3, padding=1)

    def forward(self, x: Tensor, h: Optional[Tensor] = None) -> Tensor:
        if h is None:
            h = torch.zeros_like(x)
        if h.shape != x.shape:
            raise ValueError("이전 은닉 상태의 크기가 현재 특징 맵과 다릅니다.")
        z, r = self.gates(torch.cat([x, h], dim=1)).sigmoid().chunk(2, dim=1)
        n = self.candidate(torch.cat([x, r * h], dim=1)).tanh()
        return (1 - z) * h + z * n


class LUMI7(nn.Module):
    """7개 학습 블록 + Zero-init Global Context로 구성된 통합 탐지·보정 모델."""

    def __init__(self, aux_channels: int = 9, residual_scale: float = 1.0):
        super().__init__()
        if aux_channels < 0 or residual_scale <= 0:
            raise ValueError("보조 채널 수 또는 보정 스케일이 잘못되었습니다.")
        self.aux_channels = aux_channels
        self.residual_scale = residual_scale

        # B1 ~ B3: 시공간 특징 추출
        self.b1 = nn.Conv2d(3 + aux_channels, 16, 3, padding=1)
        self.b2 = nn.Conv2d(16, 32, 3, padding=1)
        self.b3 = ConvGRUCell(32)

        # Global Context Projection (초기에는 기존 모델과 동일하게 동작하도록 Zero-init)
        self.global_proj = nn.Conv2d(32, 32, 1)
        nn.init.zeros_(self.global_proj.weight)
        nn.init.zeros_(self.global_proj.bias)

        # B4: 탐지 헤드 / B5 ~ B7: 국소 보정 디코더
        self.b4 = nn.Conv2d(32, 1, 1)
        self.b5 = nn.Conv2d(33, 32, 3, padding=1)
        self.b6 = nn.Conv2d(32, 16, 3, padding=1)
        self.b7 = nn.Conv2d(16, 3, 1)

    def step(
        self, rgb: Tensor, aux: Tensor, h: Optional[Tensor] = None
    ) -> dict[str, Tensor]:
        """한 프레임 처리. RGB: [B, 3, H, W], aux: [B, A, H, W]."""
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("RGB 입력은 [B,3,H,W]여야 합니다.")
        b, _, height, width = rgb.shape
        if aux.shape != (b, self.aux_channels, height, width):
            raise ValueError("보조 특징 맵의 크기가 올바르지 않습니다.")

        # B1 ~ B3: 시공간 특징 추출
        x = torch.cat([rgb, aux], dim=1)
        x = F.silu(self.b1(x))
        x = F.silu(self.b2(x))
        h = self.b3(x, h)

        # Global Context 추출 및 Broadcasting 융합
        global_ctx = F.adaptive_avg_pool2d(h, 1)
        global_ctx = self.global_proj(global_ctx)
        h_ctx = h + global_ctx

        # B4: 국소 + 전역 문맥 기반 탐지
        logits = self.b4(h_ctx)
        mask = logits.sigmoid()

        # B5 ~ B7: 국소 + 전역 문맥 기반 보정
        x = F.silu(self.b5(torch.cat([h_ctx, mask], dim=1)))
        x = F.silu(self.b6(x))
        delta = self.residual_scale * self.b7(x).tanh()
        candidate = (rgb + delta).clamp(0, 1)
        corrected = rgb + mask * (candidate - rgb)

        return {
            "corrected": corrected,
            "mask": mask,
            "mask_logits": logits,
            "delta": delta,
            "state": h,
        }

    def forward(
        self, video: Tensor, aux: Tensor, h: Optional[Tensor] = None
    ) -> dict[str, Tensor]:
        """클립 처리. video: [B, T, 3, H, W], aux: [B, T, A, H, W]."""
        if video.ndim != 5 or video.shape[2] != 3 or video.shape[1] == 0:
            raise ValueError("영상은 비어 있지 않은 [B,T,3,H,W]여야 합니다.")
        b, t, _, height, width = video.shape
        if aux.shape != (b, t, self.aux_channels, height, width):
            raise ValueError("보조 특징은 [B,T,A,H,W]여야 합니다.")

        outputs = {k: [] for k in ("corrected", "mask", "mask_logits", "delta")}
        for i in range(t):
            result = self.step(video[:, i], aux[:, i], h)
            h = result["state"]
            for k in outputs:
                outputs[k].append(result[k])
        return {**{k: torch.stack(v, dim=1) for k, v in outputs.items()}, "state": h}
