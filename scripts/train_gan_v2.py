"""G v2 학습 (실험 E2). 판정 규칙을 학습 신호로, 실제 안전 영상 프레임을 D의 진짜로 쓴다.

데이터: scripts/prep_real_clips.py 결과(<data>/train.npy, val.npy). test.npy는 쓰지 않는다.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from lumi.device import configure_cuda, describe, resolve_device
from lumi.gan_v2 import (FrameDiscriminator, GeneratorV2, frames_of, judge_loss, random_roi_masks,
                         texture_loss)


def main() -> None:
    ap = argparse.ArgumentParser(description="LUMI G v2 학습")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--zdim", type=int, default=16)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda-judge", type=float, default=1.0)
    ap.add_argument("--lambda-adv", type=float, default=0.1)
    ap.add_argument("--lambda-div", type=float, default=1.0)
    ap.add_argument("--lambda-tex", type=float, default=0.0, help="ROI 질감 보존 손실 가중치")
    ap.add_argument("--output", default="residual", choices=("residual", "gain"),
                    help="residual = RGB 잔차, gain = 저해상도 밝기 이득·색 오프셋(질감 보존)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    configure_cuda(device)
    train = np.load(args.data / "train.npy", mmap_mode="r")
    n, t, h, w, _ = train.shape
    G = GeneratorV2(args.width, args.zdim, t, args.fps, args.output).to(device)
    D = FrameDiscriminator(args.width).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    amp = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    print(f"장치 {describe(device)} · 배경 클립 {n}개 [{t}×{h}×{w}] · G 파라미터 {sum(p.numel() for p in G.parameters())/1e6:.2f}M")
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text("", encoding="utf-8")
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, n, args.batch_size)
        clean = torch.from_numpy(np.stack([train[i] for i in idx])).to(device).permute(0, 4, 1, 2, 3).float() / 255
        mask = torch.from_numpy(random_roi_masks(args.batch_size, h, w, rng)).to(device).float()
        mask = mask[:, None, None].expand(-1, 1, t, -1, -1)
        target = torch.from_numpy((rng.random(args.batch_size) < 0.5).astype(np.float32)).to(device)
        z1 = torch.randn(args.batch_size, args.zdim, device=device)
        z2 = torch.randn(args.batch_size, args.zdim, device=device)

        # --- D: 실제 안전 영상 프레임 대 G 출력 프레임 (hinge)
        with torch.no_grad(), amp:
            fake = G(clean, mask, target, z1)
        real_idx = rng.integers(0, n, args.batch_size)
        real = torch.from_numpy(np.stack([train[i] for i in real_idx])).to(device).permute(0, 4, 1, 2, 3).float() / 255
        opt_d.zero_grad(set_to_none=True)
        with amp:
            d_real = D(frames_of(real, 4)).float()
            d_fake = D(frames_of(fake, 4)).float()
        loss_d = F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean()
        loss_d.backward()
        opt_d.step()

        # --- G: 판정 목표 + 프레임 자연스러움 + 다양성(mode seeking)
        opt_g.zero_grad(set_to_none=True)
        with amp:
            out1 = G(clean, mask, target, z1)
            out2 = G(clean, mask, target, z2)
            adv = -D(frames_of(out1, 4)).float().mean()
        lj1, c1 = judge_loss(out1, target, args.fps)
        lj2, _ = judge_loss(out2, target, args.fps)
        roi = mask.sum() * 3 + 1e-6
        diff = ((out1.float() - out2.float()).abs() * mask).sum() / roi
        div = -diff / ((z1 - z2).abs().mean() + 1e-6)
        tex = texture_loss(out1, clean, mask) if args.lambda_tex > 0 else torch.zeros((), device=device)
        loss_g = (args.lambda_judge * (lj1 + lj2) / 2 + args.lambda_adv * adv + args.lambda_div * div
                  + args.lambda_tex * tex)
        loss_g.backward()
        opt_g.step()

        if step % 50 == 0 or step == 1:
            fail_m = target > 0.5
            rec = {"step": step, "loss_d": float(loss_d), "judge": float((lj1 + lj2) / 2), "adv": float(adv),
                   "tex": float(tex),
                   "div_l1": float(diff), "count_fail": float(c1[fail_m].mean()) if fail_m.any() else None,
                   "count_near": float(c1[~fail_m].mean()) if (~fail_m).any() else None,
                   "elapsed_s": round(time.perf_counter() - start, 1)}
            print(json.dumps(rec), flush=True)
            if args.log:
                with args.log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "gan_v2", "config": vars(args) | {"data": str(args.data), "checkpoint": str(args.checkpoint),
                "log": str(args.log), "frames": t, "height": h, "width_px": w},
                "generator": G.state_dict(), "discriminator": D.state_dict()}, args.checkpoint)
    print("저장:", args.checkpoint)


if __name__ == "__main__":
    main()
