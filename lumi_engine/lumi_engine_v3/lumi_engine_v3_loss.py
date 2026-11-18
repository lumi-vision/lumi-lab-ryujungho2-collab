"""LUMI-7 공동 학습 손실 함수 (TBPTT 청크 경계 시간 일관성 포함)."""
from typing import Callable, Optional

import torch
from torch import Tensor
import torch.nn.functional as F


def lumi_loss(
    output: dict[str, Tensor],
    video: Tensor,
    clean: Optional[Tensor],
    target_mask: Tensor,
    safety_loss_fn: Optional[Callable[[Tensor], Tensor]] = None,
    weights: tuple[float, float, float, float, float] = (1.0, 1.0, 0.5, 0.2, 1.0),
    prev_y: Optional[Tensor] = None,
    prev_clean: Optional[Tensor] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """쌍을 이룬 정답 영상 및 영역 라벨용 공동 학습 손실.

    Args:
        output: LUMI7.forward()의 출력 딕셔너리
        video: 입력 위험 영상 [B, T, 3, H, W]
        clean: 정답 깨끗한 원본 영상 [B, T, 3, H, W] (없으면 None 전달 가능)
        target_mask: 정답 위험 영역 마스크 [B, T, 1, H, W]
        safety_loss_fn: 미분 가능한 안전 규칙 대리 손실 함수
        weights: (detect, recon, preserve, temporal, safety) 가중치 튜플
        prev_y: 직전 TBPTT 청크의 마지막 보정 프레임 [B, 1, 3, H, W] (청크 경계 시간 손실용)
        prev_clean: 직전 TBPTT 청크의 마지막 정답 프레임 [B, 1, 3, H, W]
    """
    y = output["corrected"]
    if y.shape != video.shape:
        raise ValueError("원본과 출력 영상의 크기가 같아야 합니다.")
    if target_mask.shape != output["mask_logits"].shape:
        raise ValueError("정답 마스크와 탐지 출력의 크기가 같아야 합니다.")

    target_mask = target_mask.to(device=y.device, dtype=y.dtype)

    # 1. 위험 영역 탐지 손실
    detect = F.binary_cross_entropy_with_logits(output["mask_logits"], target_mask)

    # 2. 정상 영역 보존 손실 (위험하지 않은 배경 픽셀은 원본 유지)
    normal = (1.0 - target_mask).expand_as(y)
    preserve = ((y - video).abs() * normal).sum() / normal.sum().clamp_min(1.0)

    # 3 & 4. 정답 원본 복원 및 시간축 일관성 손실 (이전 청크 경계 프레임 포함)
    if clean is not None:
        if clean.shape != video.shape:
            raise ValueError("원본과 정답 영상의 크기가 같아야 합니다.")
        recon = F.l1_loss(y, clean)

        if prev_y is not None and prev_clean is not None:
            y_seq = torch.cat([prev_y.detach(), y], dim=1)
            clean_seq = torch.cat([prev_clean.detach(), clean], dim=1)
        else:
            y_seq = y
            clean_seq = clean

        temporal = (
            F.l1_loss(y_seq[:, 1:] - y_seq[:, :-1], clean_seq[:, 1:] - clean_seq[:, :-1])
            if y_seq.shape[1] > 1
            else y.new_zeros(())
        )
    else:
        recon = y.new_zeros(())
        temporal = y.new_zeros(())

    # 5. 미분 가능 안전성 규칙 손실
    safety = safety_loss_fn(y) if safety_loss_fn is not None else y.new_zeros(())

    terms = {
        "detect": detect,
        "recon": recon,
        "preserve": preserve,
        "temporal": temporal,
        "safety": safety,
    }
    total = sum(w * term for w, term in zip(weights, terms.values()))
    return total, terms
