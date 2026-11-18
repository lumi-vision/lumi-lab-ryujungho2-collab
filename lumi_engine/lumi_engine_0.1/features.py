"""LUMI-7용 9채널 인과적 시공간·전역 특징 추출기 (CausalAuxExtractor).

입력 RGB는 [0, 1] 범위이며, 출력 aux는 사전에 정규화된 9채널 맵이다.
실제 영상의 FPS(dt = 1 / fps)를 반영하여 시간 감쇠율(EMA)과 프레임 간 변화량을 스케일링하고,
체크포인트 호환성 검증을 위해 추출기 버전 및 하이퍼파라미터 설정(get_config)을 제공한다.
"""
import math
from typing import Any, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class CausalAuxExtractor(nn.Module):
    """9채널 인과적 시공간·전역 특징 추출기 (FPS 인식형).

    채널 구성:
        Ch 0: ITU-R BT.709 선형 휘도 (Y_t)
        Ch 1: FPS 보정 시간축 휘도 차분 (dY_t)
        Ch 2: FPS 보정 포화 적색 차분 (dR_sat_t)
        Ch 3: 국소 공간 고주파 대비 (5x5)
        Ch 4: 중규모 섬광 포락선 (1/8 해상도 풀링)
        Ch 5: 광역 섬광 포락선 (1/32 해상도 풀링)
        Ch 6: 광역 줄무늬 밀도 (1/16 해상도 풀링)
        Ch 7: 전역 위험 면적 비율 (프레임 전체 단일값 브로드캐스트)
        Ch 8: 물리 시간 상수 기반 섬광 지속도 (EMA)
    """

    EXTRACTOR_VERSION = "v2.0"

    def __init__(
        self,
        flash_thresh: float = 0.1,
        ema_tau: float = 1.0 / 6.0,  # 3Hz 반주기(0.1667s) 기준 물리 시간 상수
        ref_fps: float = 30.0,
    ):
        super().__init__()
        if ema_tau <= 0 or ref_fps <= 0:
            raise ValueError("ema_tau와 ref_fps는 0보다 커야 합니다.")
        self.flash_thresh = float(flash_thresh)
        self.ema_tau = float(ema_tau)
        self.ref_fps = float(ref_fps)

    def get_config(self) -> dict[str, Any]:
        """체크포인트 저장 및 호환성 검증용 추출기 설정 딕셔너리를 반환한다."""
        return {
            "extractor_version": self.EXTRACTOR_VERSION,
            "flash_thresh": self.flash_thresh,
            "ema_tau": self.ema_tau,
            "ref_fps": self.ref_fps,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CausalAuxExtractor":
        """체크포인트에 저장된 설정으로부터 동일한 특징 추출기를 복원한다."""
        if not isinstance(config, dict):
            raise ValueError("유효하지 않은 extractor_config 형식입니다.")
        version = config.get("extractor_version")
        if version != cls.EXTRACTOR_VERSION:
            raise ValueError(
                f"특징 추출기 버전 불일치: 체크포인트={version}, 현재 코드={cls.EXTRACTOR_VERSION}"
            )
        return cls(
            flash_thresh=float(config["flash_thresh"]),
            ema_tau=float(config["ema_tau"]),
            ref_fps=float(config["ref_fps"]),
        )

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

    def _temporal_coeffs(self, fps: float) -> tuple[float, float]:
        """주어진 FPS로부터 물리 시간 기반 EMA 감쇠율과 차분 스케일 계수를 계산한다."""
        if isinstance(fps, Tensor):
            rates = fps.reshape(-1, 1, 1, 1)
            if not (torch.isfinite(rates) & (rates > 0)).all():
                raise ValueError("Invalid clip FPS")
            return torch.exp(-1 / (rates * self.ema_tau)), (rates / self.ref_fps).clamp(max=1)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"유효하지 않은 FPS입니다: {fps}")
        dt = 1.0 / fps
        ema_decay = math.exp(-dt / self.ema_tau)
        rate_scale = min(1.0, fps / self.ref_fps)
        return ema_decay, rate_scale

    def step(
        self,
        rgb: Tensor,
        state: Optional[dict[str, Tensor]] = None,
        fps: float = 30.0,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """단일 프레임 특징 추출. rgb: [B, 3, H, W] -> (aux: [B, 9, H, W], next_state)."""
        ema_decay, rate_scale = self._temporal_coeffs(fps)

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

        # Ch 1 & Ch 2: FPS 보정 시간축 휘도 / 포화 적색 변화량
        dy = (y - prev_y) * rate_scale
        dred = (s_red - prev_red) * rate_scale
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

        # Ch 8: 물리 시간 상수 기반 섬광 지속도 (EMA)
        ema = ema_decay * prev_ema + (1.0 - ema_decay) * flash_mag

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
        self,
        video: Tensor,
        state: Optional[dict[str, Tensor]] = None,
        fps: float = 30.0,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """클립 특징 추출. video: [B, T, 3, H, W] -> (aux: [B, T, 9, H, W], next_state)."""
        aux_list = []
        for i in range(video.shape[1]):
            aux_i, state = self.step(video[:, i], state, fps=fps)
            aux_list.append(aux_i)
        return torch.stack(aux_list, dim=1), state
