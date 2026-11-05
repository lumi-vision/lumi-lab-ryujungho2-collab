"""통제된 합성 데이터의 변형 분포를 학습하는 조건부 3D PatchGAN.

영상의 의학적 위험도나 임상적 안전성을 예측하는 모델이 아닙니다.
GAN이 합성한 프레임은 별도 검증 전까지 학습 후보로만 취급합니다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .dataset_synthesis import STIMULUS_TYPES


def mode_tensor(labels: list[str], *, device: torch.device | None = None) -> torch.Tensor:
    """Oracle 합성 변형 유형을 4개 원-핫 채널로 인코딩합니다."""
    values = torch.zeros(len(labels), len(STIMULUS_TYPES), 1, 1, 1, device=device)
    for i, label in enumerate(labels):
        values[i, STIMULUS_TYPES.index(label)] = 1.0
    return values


def to_tensor(clip: np.ndarray) -> torch.Tensor:
    """uint8 [T,H,W,C] → float [C,T,H,W]로 변환합니다."""
    return torch.from_numpy(clip.copy()).permute(3, 0, 1, 2).float() / 255.0


class OracleDataset(Dataset):
    """생성된 NPZ와 메타데이터를 읽는 학습 데이터셋입니다."""

    def __init__(self, root: Path, split: str = "train") -> None:
        self.root = Path(root)
        self.rows = []
        with (self.root / "manifest.jsonl").open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row["split"] == split:
                    self.rows.append(row)
        if not self.rows:
            raise ValueError(f"{split} 분할에 데이터가 없습니다. 원본 영상을 추가하세요.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        with np.load(self.root / row["file"], allow_pickle=False) as arr:
            clean = to_tensor(arr["clean"])
            affected = to_tensor(arr["affected"])
            mask = torch.from_numpy(arr["mask"].copy()).float().unsqueeze(0)
            activation = torch.from_numpy(arr["activation"].copy()).float()
        temporal = activation.view(1, -1, 1, 1).expand(1, *mask.shape[1:])
        mode = mode_tensor([row["type"]])[0]
        return {"clean": clean, "affected": affected, "mask": mask,
                "activation": temporal, "mode": mode, "type": row["type"],
                "id": row["id"], "source_id": row["source_id"]}


class ConvBlock(nn.Module):
    """작은 비디오 합성기로 사용하기 위한 시공간 컨볼루션 블록."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.Conv3d(in_channels, out_channels, 3, padding=1),
                                   nn.GroupNorm(4, out_channels), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalGenerator(nn.Module):
    """원본과 ROI·파형·유형·잡음에 조건화된 국소 RGB 잔차 생성기."""

    def __init__(self, width: int = 24, max_delta: float = 0.90) -> None:
        super().__init__()
        if width % 4:
            raise ValueError("채널 수는 4의 배수여야 합니다.")
        self.width = width
        self.max_delta = max_delta
        # clean 3 + ROI 1 + activation 1 + 유형 4 + noise 1 = 10
        self.encoder = nn.Sequential(ConvBlock(10, width), ConvBlock(width, width),
                                     ConvBlock(width, width))
        self.head = nn.Conv3d(width, 3, kernel_size=3, padding=1)

    def forward(self, clean: torch.Tensor, mask: torch.Tensor,
                activation: torch.Tensor, mode: torch.Tensor,
                noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(mask)
        b, _, t, h, w = clean.shape
        control = torch.cat((clean, mask, activation, mode.expand(b, -1, t, h, w), noise), dim=1)
        delta = self.max_delta * torch.tanh(self.head(self.encoder(control)))
        return torch.clamp(clean + mask * activation * delta, 0.0, 1.0)


class ConditionalDiscriminator(nn.Module):
    """원본과 변형 영상 쌍의 시공간 패치가 기준 분포를 따르는지 판별."""

    def __init__(self, width: int = 24) -> None:
        super().__init__()
        # clean 3 + augmented 3 + ROI 1 + activation 1 + 유형 4 = 12
        self.blocks = nn.Sequential(
            nn.Conv3d(12, width, kernel_size=3, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(width, width * 2, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(width * 2, width * 2, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(width * 2, 1, kernel_size=3, padding=1),
        )

    def forward(self, clean: torch.Tensor, altered: torch.Tensor,
                mask: torch.Tensor, activation: torch.Tensor,
                mode: torch.Tensor) -> torch.Tensor:
        b, _, t, h, w = clean.shape
        x = torch.cat((clean, altered, mask, activation, mode.expand(b, -1, t, h, w)), 1)
        return self.blocks(x)


def gan_step(generator: ConditionalGenerator, discriminator: ConditionalDiscriminator,
             batch: dict, g_optimizer: torch.optim.Optimizer,
             d_optimizer: torch.optim.Optimizer, *, device: torch.device,
             lambda_reconstruction: float = 20.0,
             amp_dtype: torch.dtype | None = None) -> dict[str, float]:
    """Hinge GAN 손실과 재구성 손실로 두 모델을 각 1회 갱신합니다.

    `amp_dtype=torch.bfloat16`이면 CUDA에서 순전파를 bf16 autocast로 실행합니다.
    bf16은 fp32와 지수 범위가 같아 GradScaler가 필요 없고, 손실은 fp32로 계산합니다.
    """
    non_blocking = device.type == "cuda"
    clean = batch["clean"].to(device, non_blocking=non_blocking)
    real = batch["affected"].to(device, non_blocking=non_blocking)
    mask = batch["mask"].to(device, non_blocking=non_blocking)
    activation = batch["activation"].to(device, non_blocking=non_blocking)
    mode = batch["mode"].to(device, non_blocking=non_blocking)
    autocast = torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None)

    # 판별기: 실제 Oracle 생성 변형본과 GAN 생성 변형본을 구분합니다.
    d_optimizer.zero_grad(set_to_none=True)
    with torch.no_grad(), autocast:
        fake = generator(clean, mask, activation, mode)
    with autocast:
        real_logits = discriminator(clean, real, mask, activation, mode).float()
        fake_logits = discriminator(clean, fake, mask, activation, mode).float()
    loss_d = (torch.relu(1 - real_logits).mean() + torch.relu(1 + fake_logits).mean())
    loss_d.backward()
    d_optimizer.step()

    # 생성기: 판별기를 속이되 Oracle 기준의 변형을 재현하도록 훈련합니다.
    g_optimizer.zero_grad(set_to_none=True)
    with autocast:
        fake = generator(clean, mask, activation, mode).float()
        fake_logits = discriminator(clean, fake, mask, activation, mode).float()
    loss_adv = -fake_logits.mean()
    # 전체 픽셀 평균 대신 ROI 면적을 정규화해 작은 위험 영역도 학습합니다.
    recon = (torch.abs(fake - real) * mask).sum() / (mask.sum() * 3 + 1e-6)
    # 정상 영상 샘플은 마스크가 없으므로 반드시 원본을 유지합니다.
    loss_g = loss_adv + lambda_reconstruction * recon
    loss_g.backward()
    g_optimizer.step()
    return {"d_loss": float(loss_d.detach()), "g_loss": float(loss_g.detach()),
            "roi_l1": float(recon.detach())}


def create_checkpoint(generator: ConditionalGenerator, discriminator: ConditionalDiscriminator,
                      epoch: int, cfg: dict) -> dict:
    """모델 구조를 재현하는 데 필요한 버전과 하이퍼파라미터까지 저장합니다."""
    return {"format_version": 1, "epoch": epoch, "config": cfg,
            "generator": generator.state_dict(), "discriminator": discriminator.state_dict()}


def save_checkpoint(path: Path, generator: ConditionalGenerator,
                    discriminator: ConditionalDiscriminator, epoch: int, cfg: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(create_checkpoint(generator, discriminator, epoch, cfg), path)


def load_generator(path: Path, device: torch.device) -> ConditionalGenerator:
    """학습 체크포인트로부터 생성기만 불러옵니다."""
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("지원되지 않는 GAN 체크포인트 버전입니다.")
    cfg = payload["config"]
    model = ConditionalGenerator(width=cfg["width"]).to(device)
    model.load_state_dict(payload["generator"])
    model.eval()
    return model
