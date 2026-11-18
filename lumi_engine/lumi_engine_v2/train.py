"""LUMI-7 TBPTT + 영상 단위 Gradient Accumulation 학습 스크립트.

특징:
    - 청크 간 경계 프레임(prev_y, prev_clean)을 전달하여 chunk_size=1에서도 시간 일관성 손실 반영
    - 데이터셋의 실제 FPS를 CausalAuxExtractor에 전달
    - 스모크 테스트에서 스트리밍 동치성, FPS 민감도, 청크 경계 손실, 미학습 가중치 차단을 모두 검증

실행 예시:
    1) 파이프라인 무결성 및 스모크 테스트:
       python train.py --smoke-test

    2) 실제 합성 데이터셋 연결 후 학습:
       python train.py --epochs 30 --chunk-size 16 --batch-size 4 --save-path lumi7_best.pth
"""
import argparse
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from features import CausalAuxExtractor
from loss import lumi_loss
from model import LUMI7


class DummySyntheticDataset(Dataset):
    """파이프라인 테스트용 더미 데이터셋.

    실제 학습 시에는 생성기(G)가 만든 (rgb, clean, target_mask, fps)를 반환하도록 교체하세요.
    반환 텐서 규격:
        rgb: [T, 3, H, W] (float32, 0~1 범위)
        clean: [T, 3, H, W] (float32, 0~1 범위)
        target_mask: [T, 1, H, W] (float32, 0~1 범위)
        fps: float (예: 24.0 또는 60.0)
    """

    def __init__(
        self,
        num_samples: int = 8,
        num_frames: int = 48,
        height: int = 128,
        width: int = 192,
        fps: float = 24.0,
    ):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.fps = fps

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, float]:
        clean = torch.rand(self.num_frames, 3, self.height, self.width) * 0.5 + 0.2
        target_mask = torch.zeros(self.num_frames, 1, self.height, self.width)
        target_mask[1::2, :, self.height // 4 : self.height // 2, self.width // 4 : self.width // 2] = 1.0
        rgb = (clean + 0.4 * target_mask).clamp(0.0, 1.0)
        return rgb, clean, target_mask, self.fps


def iter_video_chunks(
    video: Tensor, clean: Tensor, target_mask: Tensor, chunk_size: int
) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
    """[B, T, ...] 클립을 시간축 기준 chunk_size 단위로 분할한다."""
    total_frames = video.shape[1]
    for start in range(0, total_frames, chunk_size):
        end = min(start + chunk_size, total_frames)
        yield video[:, start:end], clean[:, start:end], target_mask[:, start:end]


def train_one_video(
    model: LUMI7,
    extractor: CausalAuxExtractor,
    optimizer: torch.optim.Optimizer,
    video: Tensor,
    clean: Tensor,
    target_mask: Tensor,
    fps: float = 30.0,
    chunk_size: int = 16,
    safety_loss_fn=None,
) -> dict[str, float]:
    """한 영상 배치에 대해 TBPTT + 영상 전체 Gradient Accumulation을 수행한다."""
    model.train()
    total_frames = video.shape[1]

    # 새로운 영상 시작 시 상태 및 경계 프레임 초기화
    aux_state = None
    model_state = None
    prev_y = None
    prev_clean = None

    optimizer.zero_grad(set_to_none=True)

    accum_metrics = {
        "total": 0.0,
        "detect": 0.0,
        "recon": 0.0,
        "preserve": 0.0,
        "temporal": 0.0,
        "safety": 0.0,
    }

    for rgb_chunk, clean_chunk, mask_chunk in iter_video_chunks(
        video, clean, target_mask, chunk_size
    ):
        # 고정 특징 추출기 (영상의 실제 FPS 전달)
        with torch.no_grad():
            aux_chunk, aux_state = extractor(rgb_chunk, aux_state, fps=fps)

        # 탐지 및 보정
        out = model(rgb_chunk, aux_chunk, model_state)

        loss, terms = lumi_loss(
            out,
            rgb_chunk,
            clean_chunk,
            mask_chunk,
            safety_loss_fn=safety_loss_fn,
            prev_y=prev_y,
            prev_clean=prev_clean,
        )

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss detected during TBPTT chunk")

        # 마지막 청크의 길이가 다른 경우도 고려한 가중치
        weight = rgb_chunk.shape[1] / total_frames
        (loss * weight).backward()

        accum_metrics["total"] += float(loss.detach()) * weight
        for k, v in terms.items():
            accum_metrics[k] += float(v.detach()) * weight

        # TBPTT: 연산 그래프만 절단하고 은닉 상태 및 경계 프레임은 다음 청크로 유지
        model_state = out["state"].detach()
        aux_state = {k: v.detach() for k, v in aux_state.items()}
        prev_y = out["corrected"][:, -1:].detach()
        prev_clean = clean_chunk[:, -1:].detach()

        del out, loss, terms

    # 전체 영상의 gradient를 누적한 후 한 번만 파라미터 업데이트
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
        error_if_nonfinite=True,
    )
    optimizer.step()

    return accum_metrics


def smoke_test() -> None:
    """스트리밍 동치성, FPS 민감도, 청크 경계 시간 손실, 미학습 가중치 차단을 모두 검증한다."""
    from infer import LUMIInferencer

    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor = CausalAuxExtractor().to(device).eval()
    model = LUMI7(aux_channels=9).to(device)

    # 교번 깜빡임이 있는 테스트 시퀀스 생성
    base = torch.rand(1, 1, 3, 32, 48, device=device) * 0.4
    video = base.repeat(1, 6, 1, 1, 1)
    video[:, 1::2] = (video[:, 1::2] + 0.5).clamp(0.0, 1.0)
    clean = base.repeat(1, 6, 1, 1, 1)
    target_mask = torch.ones(1, 6, 1, 32, 48, device=device)

    # 1. 클립 일괄 처리(forward) vs 프레임별 스트리밍(step) 동치성 검증
    with torch.no_grad():
        aux_full, _ = extractor(video, fps=24.0)
        full_out = model(video, aux_full)

        a_state, m_state, frames = None, None, []
        for i in range(video.shape[1]):
            aux_i, a_state = extractor.step(video[:, i], a_state, fps=24.0)
            out_i = model.step(video[:, i], aux_i, m_state)
            m_state = out_i["state"]
            frames.append(out_i["corrected"])

        torch.testing.assert_close(full_out["corrected"], torch.stack(frames, dim=1))

    # 2. FPS 민감도 검증 (동일 시퀀스라도 10fps vs 60fps 특징이 달라야 함)
    with torch.no_grad():
        aux_10, _ = extractor(video, fps=10.0)
        aux_60, _ = extractor(video, fps=60.0)
        diff_ema = (aux_60[:, -1, 8] - aux_10[:, -1, 8]).abs().mean().item()
        assert diff_ema > 0.01, f"FPS에 따른 EMA 차이가 충분하지 않습니다: {diff_ema}"

    # 3. chunk_size=1 TBPTT 경계 시간 손실 검증 (temporal > 0 이어야 함)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics_c1 = train_one_video(
        model, extractor, optimizer, video, clean, target_mask, fps=60.0, chunk_size=1
    )
    assert metrics_c1["temporal"] > 0.0, "chunk_size=1에서 청크 경계 시간 손실이 0입니다."

    metrics = train_one_video(
        model, extractor, optimizer, video, clean, target_mask, fps=60.0, chunk_size=3
    )

    # 4. 미학습(누락된) 가중치 로딩 거부 검증
    blocked = False
    try:
        LUMIInferencer(weights_path="definitely_missing_weights_12345.pth", allow_untrained=False)
    except FileNotFoundError:
        blocked = True
    assert blocked, "존재하지 않는 가중치 파일에 대해 추론기 초기화가 차단되지 않았습니다."

    print("[SMOKE TEST PASS]")
    print("device:", device)
    print("parameters:", sum(p.numel() for p in model.parameters()))
    print("fps_ema_diff (60fps vs 10fps):", round(diff_ema, 6))
    print("chunk_size=1 boundary temporal loss:", round(metrics_c1["temporal"], 6))
    print("chunk_size=3 metrics:", {k: round(v, 6) for k, v in metrics.items()})
    print("untrained weights guard: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="LUMI-7 Training Script")
    parser.add_argument("--smoke-test", action="store_true", help="스모크 테스트만 실행")
    parser.add_argument("--epochs", type=int, default=5, help="학습 에폭 수")
    parser.add_argument("--batch-size", type=int, default=2, help="배치 크기")
    parser.add_argument("--chunk-size", type=int, default=16, help="TBPTT 청크 프레임 길이")
    parser.add_argument("--fps", type=float, default=24.0, help="학습 영상의 FPS")
    parser.add_argument("--lr", type=float, default=1e-3, help="학습률")
    parser.add_argument("--save-path", type=str, default="lumi7_best.pth", help="가중치 저장 경로")
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = CausalAuxExtractor().to(device).eval()
    model = LUMI7(aux_channels=9).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # TODO: 실제 생성기(G) 기반 Dataset으로 교체
    dataset = DummySyntheticDataset(
        num_samples=8, num_frames=48, height=128, width=192, fps=args.fps
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    best_loss = float("inf")
    save_path = Path(args.save_path)

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for video, clean, target_mask, fps_batch in loader:
            video = video.to(device)
            clean = clean.to(device)
            target_mask = target_mask.to(device)
            batch_fps = float(fps_batch[0]) if isinstance(fps_batch, Tensor) else float(fps_batch)

            metrics = train_one_video(
                model=model,
                extractor=extractor,
                optimizer=optimizer,
                video=video,
                clean=clean,
                target_mask=target_mask,
                fps=batch_fps,
                chunk_size=args.chunk_size,
                safety_loss_fn=None,
            )
            epoch_loss += metrics["total"]

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch [{epoch}/{args.epochs}] - Total Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> Best model saved to {save_path} (loss={best_loss:.6f})")


if __name__ == "__main__":
    main()
