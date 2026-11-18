"""LUMI-7 TBPTT + 영상 단위 Gradient Accumulation 학습 스크립트.

안전한 체크포인트 관리 정책:
    - 실제 데이터셋 디렉토리(--data-dir) 없이 기본 학습 명령을 실행하면 오류로 차단된다.
      파이프라인 확인용 더미 학습은 반드시 --allow-dummy 플래그를 명시해야 하며,
      저장된 체크포인트에는 trained_on_dummy=True, validated=False 메타데이터가 기록되어
      API 서버가 이를 검증된 안전 모델로 오인하지 않도록 차단한다.
    - 체크포인트에는 model_config, extractor_config(버전·시간상수), 학습 메타데이터가 함께 저장된다.

실행 예시:
    1) 파이프라인 무결성 및 스모크 테스트:
       python train.py --smoke-test

    2) 실제 합성 데이터셋(.npz / .pt 클립 폴더) 학습:
       python train.py --data-dir ./dataset_clips --epochs 30 --chunk-size 16 --batch-size 4 --save-path lumi7_best.pth

    3) 더미 데이터 파이프라인 점검용(API 추론 불가 태그 부여):
       python train.py --allow-dummy --epochs 2 --save-path dummy_test.pth
"""
import argparse
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from features import CausalAuxExtractor
from loss import lumi_loss
from evaluation import evaluate, passes, ensure_disjoint
from model import LUMI7


class ClipFolderDataset(Dataset):
    """생성기(G)가 생성한 클립 파일(.npz 또는 .pt) 디렉토리 로더.

    각 파일은 다음 키를 포함해야 한다:
        - video (또는 rgb): [T, 3, H, W] float32 [0, 1]
        - clean: [T, 3, H, W] float32 [0, 1]
        - target_mask: [T, 1, H, W] float32 [0, 1]
        - fps (선택): float (없으면 default_fps 사용)
    """

    def __init__(self, data_dir: str, default_fps: float = 24.0):
        root = Path(data_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"데이터셋 디렉토리를 찾을 수 없습니다: {data_dir}")
        self.files = sorted(
            [p for p in root.rglob("*") if p.suffix.lower() in {".npz", ".pt"}]
        )
        if not self.files:
            raise ValueError(
                f"데이터셋 디렉토리에 .npz 또는 .pt 클립 파일이 없습니다: {data_dir}"
            )
        self.default_fps = float(default_fps)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, float]:
        path = self.files[idx]
        if path.suffix.lower() == ".npz":
            with np.load(path) as data:
                vid_arr = data["video"] if "video" in data else data["rgb"]
                video = torch.from_numpy(vid_arr).float()
                clean = torch.from_numpy(data["clean"]).float()
                target_mask = torch.from_numpy(data["target_mask"]).float()
                fps = float(data["fps"]) if "fps" in data else self.default_fps
        else:
            data = torch.load(path, map_location="cpu", weights_only=True)
            vid_t = data["video"] if "video" in data else data["rgb"]
            video = vid_t.float()
            clean = data["clean"].float()
            target_mask = data["target_mask"].float()
            fps = float(data.get("fps", self.default_fps))

        if video.ndim != 4 or video.shape[1] != 3:
            raise ValueError(f"잘못된 video 텐서 크기 ({path}): {tuple(video.shape)}")
        if clean.shape != video.shape:
            raise ValueError(f"clean과 video 크기가 다릅니다 ({path})")
        if target_mask.shape != (video.shape[0], 1, video.shape[2], video.shape[3]):
            raise ValueError(f"잘못된 target_mask 크기 ({path}): {tuple(target_mask.shape)}")

        return video, clean, target_mask, fps


class DummySyntheticDataset(Dataset):
    """파이프라인 디버깅 전용 더미 데이터셋 (--allow-dummy 명시 시에만 사용 가능)."""

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
        self.fps = float(fps)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, float]:
        clean = torch.rand(self.num_frames, 3, self.height, self.width) * 0.5 + 0.2
        target_mask = torch.zeros(self.num_frames, 1, self.height, self.width)
        target_mask[1::2, :, self.height // 4 : self.height // 2, self.width // 4 : self.width // 2] = 1.0
        rgb = (clean + 0.4 * target_mask).clamp(0.0, 1.0)
        return rgb, clean, target_mask, self.fps


def build_checkpoint(
    model: LUMI7,
    extractor: CausalAuxExtractor,
    *,
    trained_on_dummy: bool,
    validated: bool,
    trained_fps: list[float],
    epoch: int,
    best_loss: float,
) -> dict[str, Any]:
    """모델 가중치와 특징 추출기 설정·검증 메타데이터를 포함한 표준 체크포인트를 생성한다."""
    return {
        "checkpoint_format": "lumi7_0.1",
        "model_state_dict": model.state_dict(),
        "model_config": model.get_config(),
        "extractor_config": extractor.get_config(),
        "metadata": {
            "trained_on_dummy": bool(trained_on_dummy),
            "validated": bool(validated),
            "trained_fps": trained_fps,
            "epochs": int(epoch),
            "best_loss": float(best_loss),
        },
    }


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
        with torch.no_grad():
            aux_chunk, aux_state = extractor(rgb_chunk, aux_state, fps=fps)

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

        weight = rgb_chunk.shape[1] / total_frames
        (loss * weight).backward()

        accum_metrics["total"] += float(loss.detach()) * weight
        for k, v in terms.items():
            accum_metrics[k] += float(v.detach()) * weight

        model_state = out["state"].detach()
        aux_state = {k: v.detach() for k, v in aux_state.items()}
        prev_y = out["corrected"][:, -1:].detach()
        prev_clean = clean_chunk[:, -1:].detach()

        del out, loss, terms

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
        error_if_nonfinite=True,
    )
    optimizer.step()

    return accum_metrics


def smoke_test() -> None:
    """스트리밍 동치성, FPS 민감도, 청크 경계 시간 손실, 체크포인트/더미 가드를 검증한다."""
    import tempfile
    from infer import LUMIInferencer

    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor = CausalAuxExtractor().to(device).eval()
    model = LUMI7(aux_channels=9).to(device)

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

    # 2. FPS 민감도 검증
    with torch.no_grad():
        aux_10, _ = extractor(video, fps=10.0)
        aux_60, _ = extractor(video, fps=60.0)
        diff_ema = (aux_60[:, -1, 8] - aux_10[:, -1, 8]).abs().mean().item()
        assert diff_ema > 0.01, f"FPS에 따른 EMA 차이가 충분하지 않습니다: {diff_ema}"

    # 3. chunk_size=1 TBPTT 경계 시간 손실 검증
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics_c1 = train_one_video(
        model, extractor, optimizer, video, clean, target_mask, fps=60.0, chunk_size=1
    )
    assert metrics_c1["temporal"] > 0.0, "chunk_size=1에서 청크 경계 시간 손실이 0입니다."

    metrics = train_one_video(
        model, extractor, optimizer, video, clean, target_mask, fps=60.0, chunk_size=3
    )

    # 4. 체크포인트 호환성 및 더미/구버전(v1) 가중치 차단 검증
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # 4-a. 구버전(v1) raw state_dict 차단 확인
        v1_path = tmp / "v1_raw.pth"
        torch.save(model.state_dict(), v1_path)
        v1_blocked = False
        try:
            LUMIInferencer(weights_path=str(v1_path))
        except ValueError:
            v1_blocked = True
        assert v1_blocked, "추출기 설정이 없는 v1 체크포인트가 차단되지 않았습니다."

        # 4-b. 더미 학습 체크포인트 차단 확인
        dummy_path = tmp / "dummy.pth"
        torch.save(
            build_checkpoint(
                model,
                extractor,
                trained_on_dummy=True,
                validated=False,
                trained_fps=[24.0],
                epoch=1,
                best_loss=metrics["total"],
            ),
            dummy_path,
        )
        dummy_blocked = False
        try:
            LUMIInferencer(weights_path=str(dummy_path))
        except RuntimeError:
            dummy_blocked = True
        assert dummy_blocked, "더미 데이터로 학습된 체크포인트가 차단되지 않았습니다."

        # 4-c. 추출기 설정 불일치 체크포인트 차단 확인
        mismatch_path = tmp / "mismatch.pth"
        bad_ckpt = build_checkpoint(
            model,
            CausalAuxExtractor(ema_tau=0.5),
            trained_on_dummy=False,
            validated=True,
            trained_fps=[24.0],
            epoch=1,
            best_loss=metrics["total"],
        )
        torch.save(bad_ckpt, mismatch_path)
        mismatch_blocked = False
        try:
            LUMIInferencer(weights_path=str(mismatch_path))
        except ValueError:
            mismatch_blocked = True
        assert mismatch_blocked, "추출기 파라미터가 다른 체크포인트가 차단되지 않았습니다."

    print("[SMOKE TEST PASS]")
    print("device:", device)
    print("parameters:", sum(p.numel() for p in model.parameters()))
    print("fps_ema_diff (60fps vs 10fps):", round(diff_ema, 6))
    print("chunk_size=1 boundary temporal loss:", round(metrics_c1["temporal"], 6))
    print("chunk_size=3 metrics:", {k: round(v, 6) for k, v in metrics.items()})
    print("checkpoint compatibility & dummy/v1 guards: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="LUMI-7 Training Script")
    parser.add_argument("--smoke-test", action="store_true", help="스모크 테스트만 실행")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="실제 학습 클립(.npz/.pt) 디렉토리 경로 (미지정 시 기본 학습 차단)",
    )
    parser.add_argument(
        "--allow-dummy",
        action="store_true",
        help="파이프라인 테스트용 더미 데이터 학습 허용 (API 검증 모델로 사용 불가 태그 기록)",
    )
    parser.add_argument("--epochs", type=int, default=5, help="학습 에폭 수")
    parser.add_argument("--batch-size", type=int, default=2, help="배치 크기")
    parser.add_argument("--chunk-size", type=int, default=16, help="TBPTT 청크 프레임 길이")
    parser.add_argument("--fps", type=float, default=24.0, help="기본 학습 영상 FPS")
    parser.add_argument("--lr", type=float, default=1e-3, help="학습률")
    parser.add_argument("--save-path", type=str, default="lumi7_best.pth", help="가중치 저장 경로")
    parser.add_argument("--val-dir")
    parser.add_argument("--min-recall", type=float)
    parser.add_argument("--max-fpr", type=float)
    parser.add_argument("--max-hazard-mae", type=float)
    args = parser.parse_args()
    limits = (args.min_recall, args.max_fpr, args.max_hazard_mae)
    if args.val_dir:
        if not args.data_dir or not all(x is not None and 0 <= x <= 1 for x in limits):
            parser.error("Validation requires --data-dir and three evaluation limits in [0, 1]")
    elif any(x is not None for x in limits):
        parser.error("Evaluation limits require --val-dir")

    if args.smoke_test:
        smoke_test()
        return

    if args.data_dir is None and not args.allow_dummy:
        raise SystemExit(
            "[ERROR] --data-dir가 지정되지 않았습니다. 임의 더미 데이터로 검증된 것처럼 보이는 "
            "가중치가 생성되는 것을 막기 위해 기본 더미 학습은 차단되어 있습니다.\n"
            "  - 실제 데이터셋 학습: python train.py --data-dir <클립_폴더_경로> ...\n"
            "  - 파이프라인 점검용: python train.py --allow-dummy ..."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = CausalAuxExtractor().to(device).eval()
    model = LUMI7(aux_channels=9).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.data_dir is not None:
        dataset: Dataset = ClipFolderDataset(args.data_dir, default_fps=args.fps)
        trained_on_dummy = False
    else:
        print("[WARNING] --allow-dummy 모드로 실행 중입니다. 생성된 가중치는 API에서 거부됩니다.")
        dataset = DummySyntheticDataset(
            num_samples=8, num_frames=48, height=128, width=192, fps=args.fps
        )
        trained_on_dummy = True

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    best_loss = float("inf")
    best_approved = False
    save_path = Path(args.save_path)
    validation = None
    if args.val_dir:
        validation = ClipFolderDataset(args.val_dir, args.fps)
        ensure_disjoint(dataset.files, validation.files)
    seen_fps = set()

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for video, clean, target_mask, fps_batch in loader:
            video = video.to(device)
            clean = clean.to(device)
            target_mask = target_mask.to(device)
            seen_fps.update(fps_batch.tolist())
            batch_fps = fps_batch.to(device=device, dtype=video.dtype)

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

        report = evaluate(model, extractor, validation, device, limits) if validation else None
        approved = passes(report)
        if (approved and not best_approved) or (approved == best_approved and avg_loss < best_loss):
            best_loss = avg_loss
            best_approved = approved
            ckpt = build_checkpoint(
                model=model,
                extractor=extractor,
                trained_on_dummy=trained_on_dummy,
                validated=approved,
                trained_fps=sorted(seen_fps),
                epoch=epoch,
                best_loss=best_loss,
            )
            ckpt["evaluation"] = report
            torch.save(ckpt, save_path)
            print(
                f"  -> Best checkpoint saved to {save_path} "
                f"(loss={best_loss:.6f}, validated={approved})"
            )


if __name__ == "__main__":
    main()
