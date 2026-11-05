"""Oracle 합성 데이터를 학습하는 조건부 PatchGAN 훈련 스크립트."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lumi.device import configure_cuda, describe, resolve_device
from lumi.gan_synthesis import (OracleDataset, ConditionalGenerator,
                                ConditionalDiscriminator, gan_step, save_checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description="LUMI 조건부 3D PatchGAN 학습")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/oracle"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/lumi_cgan.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--width", type=int, default=24, help="모델 채널 수(4의 배수)")
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--amp", default="auto", choices=("auto", "bf16", "none"),
                        help="auto = CUDA에서 bf16 autocast, CPU에서는 끔")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader 작업 프로세스 수 (GPU PC 권장: 4)")
    parser.add_argument("--lambda-rec", type=float, default=20.0, help="ROI 재구성(L1) 손실 가중치")
    parser.add_argument("--log", type=Path, default=None, help="에폭별 손실 JSONL 기록 경로")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("에폭과 배치 크기는 1 이상이어야 합니다.")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    configure_cuda(device)
    use_bf16 = args.amp == "bf16" or (args.amp == "auto" and device.type == "cuda")
    amp_dtype = torch.bfloat16 if use_bf16 else None
    data = OracleDataset(args.dataset)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    generator = ConditionalGenerator(width=args.width).to(device)
    discriminator = ConditionalDiscriminator(width=args.width).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    print(f"학습 장치: {describe(device)} · AMP: {'bf16' if amp_dtype else 'off'} · 학습 클립: {len(data)}개")
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text("", encoding="utf-8")
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        sums = {"d_loss": 0.0, "g_loss": 0.0, "roi_l1": 0.0}
        for batch in loader:
            losses = gan_step(generator, discriminator, batch, opt_g, opt_d,
                              device=device, amp_dtype=amp_dtype,
                              lambda_reconstruction=args.lambda_rec)
            for key in sums:
                sums[key] += losses[key]
        means = {key: value / len(loader) for key, value in sums.items()}
        print(f"Epoch {epoch}/{args.epochs} " + " ".join(f"{k}={v:.4f}" for k, v in means.items()))
        if args.log:
            with args.log.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"epoch": epoch, **means,
                                    "elapsed_s": round(time.perf_counter() - start, 1)}) + "\n")
    if device.type == "cuda":
        print(f"GPU 최대 메모리 사용: {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB")
    save_checkpoint(args.checkpoint, generator, discriminator, args.epochs,
                    {"width": args.width, "lr": args.lr, "seed": args.seed,
                     "lambda_rec": args.lambda_rec, "batch_size": args.batch_size,
                     "amp": "bf16" if amp_dtype else "none", "device": str(device)})
    print(f"체크포인트 저장: {args.checkpoint}")
    print("학습 수치와 생성 결과는 PSE 위험 통과 여부를 의미하지 않습니다.")


if __name__ == "__main__":
    main()
