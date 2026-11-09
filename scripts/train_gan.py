"""조건부 GAN 학습. 재료는 오라클 단계가 만든 깨끗한 바탕 클립(backgrounds_train.npz)뿐이다.

실행 예
    노트북(CPU, 동작 확인용):  python scripts/train_gan.py --steps 300 --batch-size 2
    집 PC(RTX 5080):           python scripts/train_gan.py --steps 4000 --batch-size 8 --width 32 --device cuda

한 스텝에서 일어나는 일 (docs/GAN.md §4 의 그림과 같은 순서)
    1. 바탕 Y 를 B 개 뽑는다. 마스크 M 은 오라클의 마스크 생성기(make_mask)로 면적 30~70% 를 뽑는다.
       목표 c 는 절반은 '위험', 절반은 '문턱 근처 안전'. 잡음 z1, z2 를 뽑는다.
    2. D 갱신: 진짜 = 깨끗한 바탕 프레임, 가짜 = G(Y,M,c,z1) 프레임. 힌지 손실.
    3. G 갱신: 손실 = λ_judge·판정 목표 + λ_adv·D 속이기 + λ_div·리듬 다양성
    4. 25 스텝마다 기록(JSON 한 줄): 손실, 목표별 소프트 전이 수 → checks/gan/train_log.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from lumi_synth import oracle
from lumi_synth.gan import (SOFT_JUDGE_VERSION, FrameDiscriminator, LightGenerator, judge_loss, luminance_rel, random_frames,
                            spectral_diversity, srgb_to_linear)

ROOT = Path(__file__).resolve().parents[1]


def sample_masks(n: int, h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """오라클과 같은 마스크 생성기. 'hazard' 의도의 면적 범위(0.32~0.70, 10% 는 화면 전체)를 써서, 위험/안전의 차이가
    면적이 아니라 '빛의 시간 변화'에서 나오게 한다(문턱 근처 안전 = 가장 어려운 음성 사례)."""
    return np.stack([oracle.make_mask(oracle.sample_params("hazard", rng), h, w) for _ in range(n)])


def rescale_clips(clips: np.ndarray, scale: float) -> np.ndarray:
    """(N,T,H,W,3) 클립을 면적 평균(INTER_AREA)으로 축소. scale=1 이면 그대로."""
    if scale == 1.0:
        return clips
    import cv2
    n, t, h, w, _ = clips.shape
    size = (max(8, int(round(w * scale))), max(8, int(round(h * scale))))
    return np.stack([[cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in c] for c in clips])


def to_tensor(clips_bgr: np.ndarray, device) -> torch.Tensor:
    """(B,T,H,W,3) uint8 → (B,3,T,H,W) float 0~1 (채널 순서는 BGR 그대로)."""
    return torch.from_numpy(clips_bgr).to(device).permute(0, 4, 1, 2, 3).float() / 255.0


def main() -> None:
    ap = argparse.ArgumentParser(description="LUMI 조건부 GAN 학습")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "oracle")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "data" / "gan" / "gan.pt")
    ap.add_argument("--log", type=Path, default=ROOT / "checks" / "gan" / "train_log.jsonl")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--zdim", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda-judge", type=float, default=1.0)
    ap.add_argument("--lambda-adv", type=float, default=0.1)
    ap.add_argument("--lambda-div", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--scale", type=float, default=1.0,
                    help="학습 해상도 배율(노트북 CPU 동작 확인은 0.5 → 64×36). 면적 비율은 해상도와 무관")
    args = ap.parse_args()
    if args.batch_size < 2 or args.batch_size % 2:
        raise SystemExit("--batch-size 는 2 이상의 짝수여야 합니다(목표 '위험'/'근처 안전'을 절반씩).")

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, (__import__("os").cpu_count() or 2)))
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA 를 요청했지만 사용할 수 없습니다.")
    bg = np.load(args.data / "backgrounds_train.npz")["clips"]          # (N,T,H,W,3) uint8 BGR
    if bg.ndim != 5 or len(bg) == 0:
        raise SystemExit(f"바탕 클립이 없습니다: {args.data}/backgrounds_train.npz (make_oracle_dataset.py 먼저)")
    bg = rescale_clips(bg, args.scale)
    n, t, h, w, _ = bg.shape
    fps = 24.0
    G = LightGenerator(args.width, args.zdim, fps).to(device)
    D = FrameDiscriminator(args.width).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    print(f"바탕 {n}개 [{t}×{h}×{w}] · G {sum(p.numel() for p in G.parameters()):,} · "
          f"D {sum(p.numel() for p in D.parameters()):,} 파라미터 · {device}", flush=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text("", encoding="utf-8")
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        b = args.batch_size
        clean = to_tensor(bg[rng.integers(0, n, b)], device)
        mask = torch.from_numpy(sample_masks(b, h, w, rng)).to(device).float()[:, None, None]  # [B,1,1,H,W]
        target = torch.from_numpy((np.arange(b) % 2 == 0).astype(np.float32)).to(device)        # 위험/근처 절반씩
        z1, z2 = torch.randn(b, args.zdim, device=device), torch.randn(b, args.zdim, device=device)

        # ---- 2. D: 진짜(깨끗한 바탕 프레임) 대 가짜(G 출력 프레임)
        with torch.no_grad():
            fake, _ = G(clean, mask, target, z1)
        real = to_tensor(bg[rng.integers(0, n, b)], device)
        opt_d.zero_grad(set_to_none=True)
        loss_d = F.relu(1 - D(random_frames(real, 4))).mean() + F.relu(1 + D(random_frames(fake, 4))).mean()
        loss_d.backward()
        opt_d.step()

        # ---- 3. G: 판정 목표 + 자연스러움 + 리듬 다양성
        opt_g.zero_grad(set_to_none=True)
        out1, d1 = G(clean, mask, target, z1)
        out2, d2 = G(clean, mask, target, z2)
        lj1, c1 = judge_loss(luminance_rel(srgb_to_linear(out1)), target, fps)
        lj2, _ = judge_loss(luminance_rel(srgb_to_linear(out2)), target, fps)
        adv = -D(random_frames(out1, 4)).mean()
        div = spectral_diversity(d1, d2, mask, z1, z2)
        loss_g = args.lambda_judge * (lj1 + lj2) / 2 + args.lambda_adv * adv + args.lambda_div * div
        loss_g.backward()
        opt_g.step()

        if step == 1 or step % 25 == 0 or step == args.steps:
            haz = target > 0.5
            rec = {"step": step, "loss_d": round(float(loss_d), 4), "judge": round(float((lj1 + lj2) / 2), 4),
                   "adv": round(float(adv), 4), "div": round(float(div), 4),
                   "soft_count_hazard": round(float(c1[haz].mean().detach()), 2),
                   "soft_count_near": round(float(c1[~haz].mean()), 2),
                   "elapsed_s": round(time.perf_counter() - start, 1)}
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            with args.log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "lumi_light_gan_v1", "config": {k: str(v) if isinstance(v, Path) else v
                                                          for k, v in vars(args).items()} | {"fps": fps,
                "soft_judge_version": SOFT_JUDGE_VERSION},
                "train_log": [json.loads(x) for x in args.log.read_text(encoding="utf-8").splitlines() if x],
                "generator": G.state_dict(), "discriminator": D.state_dict()}, args.checkpoint)
    print("저장:", args.checkpoint)


if __name__ == "__main__":
    main()
