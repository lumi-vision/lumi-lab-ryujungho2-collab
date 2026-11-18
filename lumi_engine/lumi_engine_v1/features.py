"""LUMI-7용 9채널 인과적 시공간·전역 특징 추출기 (CausalAuxExtractor).

입력 RGB는 [0, 1] 범위이며, 출력 aux는 사전에 정규화된 9채널 맵이다.
학습 가능한 가중치가 없는 완전 미분 가능 모듈로, step(단일 프레임)과 forward(클립)를 모두 지원한다.
"""
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class CausalAuxExtractor(nn.Module):
    """9채널 인과적 시공간·전역 특징 추출기.

    채널 구성:
        Ch 0: ITU-R BT.709 선형 휘도 (Y_t)
        Ch 1: 시간축 휘도 차분 (dY_t)
        Ch 2: 포화 적색 차분 (dR_sat_t)
        Ch 3: 국소 공간 고주파 대비 (5x5)
        Ch 4: 중규모 섬광 포락선 (1/8 해상도 풀링)
        Ch 5: 광역 섬광 포락선 (1/32 해상도 풀링)
        Ch 6: 광역 줄무늬 밀도 (1/16 해상도 풀링)
        Ch 7: 전역 위험 면적 비율 (프레임 전체 단일값 브로드캐스트)
        Ch 8: 시간축 섬광 지속도 (EMA)
    """

    def __init__(self, flash_thresh: float = 0.1, ema_decay: float = 0.85):
        super().__init__()
        self.flash_thresh = flash_thresh
        self.ema_decay = ema_decay

    @staticmethod
    def _srgb_to_linear(rgb: Tensor) -> Tensor:
        low = rgb / 12.92
        high = ((rgb.clamp_min(0.04045) + 0.055) / 1.055) ** 2.4
        return torch.where(rgb <= 0.04045, low, high)

    @staticmethod
    def _pool_and_up(feat: Tensor, scale_div: int) -> Tensor:
        _, _, h, w = feat.shape
        target_hw = (max(1, h // scale_div), max(1, w // scale_div))
        pooled = F.adaptive_avg_pool2d(feat, target_hw)
        return F.interpolate(pooled, size=(h, w), mode="bilinear", align_corners=False)

    def step(
        self, rgb: Tensor, state: Optional[dict[str, Tensor]] = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """단일 프레임 특징 추출. rgb: [B, 3, H, W] -> (aux: [B, 9, H, W], next_state)."""
        lin = self._srgb_to_linear(rgb.clamp(0.0, 1.0))
        r, g, b = lin[:, 0:1], lin[:, 1:2], lin[:, 2:3]

        # Ch 0: ITU-R BT.709 선형 휘도
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b

        # 포화 적색 지수
        red_ratio = r / (r + g + b + 1e-6)
        red_excess = F.relu(r - torch.maximum(g, b))
        s_red = red_ratio * red_excess

        if state is None:
            prev_y = y
            prev_red = s_red
            prev_ema = torch.zeros_like(y)
        else:
            prev_y = state["prev_y"]
            prev_red = state["prev_red"]
            prev_ema = state["prev_ema"]

        # Ch 1 & Ch 2: 시간축 휘도 / 포화 적색 변화량
        dy = y - prev_y
        dred = s_red - prev_red
        flash_mag = torch.maximum(dy.abs(), dred.abs())

        # Ch 3: 국소 고주파 대비 (5x5 평균 대비 편차)
        y_blur = F.avg_pool2d(F.pad(y, (2, 2, 2, 2), mode="reflect"), 5, stride=1)
        local_contrast = ((y - y_blur).abs() * 2.0).clamp(0.0, 1.0)

        # Ch 4 & Ch 5: 다중 스케일 섬광 포락선 (1/8, 1/32 해상도)
        mid_flash = self._pool_and_up(flash_mag, 8)
        coarse_flash = self._pool_and_up(flash_mag, 32)

        # Ch 6: 광역 줄무늬 밀도 (1/16 해상도)
        coarse_pattern = self._pool_and_up(local_contrast, 16)

        # Ch 7: 전역 위험 면적 비율
        active_prob = torch.sigmoid(30.0 * (flash_mag - self.flash_thresh))
        global_area_ratio = active_prob.mean(dim=(2, 3), keepdim=True).expand_as(y)

        # Ch 8: 시간축 섬광 지속도 (EMA)
        ema = self.ema_decay * prev_ema + (1.0 - self.ema_decay) * flash_mag

        aux = torch.cat(
            [
                y,
                dy,
                dred,
                local_contrast,
                mid_flash,
                coarse_flash,
                coarse_pattern,
                global_area_ratio,
                ema,
            ],
            dim=1,
        )
        next_state = {"prev_y": y, "prev_red": s_red, "prev_ema": ema}
        return aux, next_state

    def forward(
        self, video: Tensor, state: Optional[dict[str, Tensor]] = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """클립 특징 추출. video: [B, T, 3, H, W] -> (aux: [B, T, 9, H, W], next_state)."""
        aux_list = []
        for i in range(video.shape[1]):
            aux_i, state = self.step(video[:, i], state)
            aux_list.append(aux_i)
        return torch.stack(aux_list, dim=1), state
