"""조건부 GAN 학습. 재료는 오라클 단계가 만든 깨끗한 바탕 클립(backgrounds_train.npz)뿐이다.

실행 예
    집 PC(RTX 5080):  python scripts/train_gan.py --steps 4000 --batch-size 8 --width 32 --device cuda
    (판정·학습 해상도 768×432 통일 — 2026-11-09. 노트북 CPU 로는 이 크기 학습이 사실상 어렵다. 동작 확인은
     --scale 0.25 --debug-scale --steps 10 처럼 하고, 그 결과는 학습용 데이터에 쓰지 않는다.)

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


def load_backgrounds(data_dir: Path, split: str) -> tuple[np.ndarray, float]:
    """backgrounds_<split>.npz → (클립, fps). fps 가 없는 옛 파일은 거부한다."""
    npz = np.load(data_dir / f"backgrounds_{split}.npz")
    clips = npz["clips"]
    if clips.ndim != 5 or len(clips) == 0:
        raise SystemExit(f"{split} 바탕이 없습니다: {data_dir}/backgrounds_{split}.npz")
    if "fps" not in npz:
        raise SystemExit(f"{data_dir}/backgrounds_{split}.npz 에 fps 가 없습니다. make_oracle_dataset.py 로 다시 만드세요.")
    return clips, float(npz["fps"])


def check_track(cfg: dict, clips: np.ndarray, fps: float, allow_mismatch: bool) -> None:
    """G 를 학습한 해상도·fps 와 지금 넣으려는 바탕의 해상도·fps 가 같은지 확인한다.

    G 는 완전 합성곱이라 다른 해상도에도 오류 없이 돌아간다 → 조용히 다른 조건의 데이터가 만들어진다
    (2026-11-09: 64×36 으로 학습한 G 의 샘플이 128×72 오라클과 섞일 뻔함). 그래서 다르면 멈춘다.
    """
    want_hw = tuple(cfg.get("train_hw", ()))
    have_hw = tuple(clips.shape[2:4])
    problems = []
    if want_hw and want_hw != have_hw:
        problems.append(f"해상도: G 학습 {want_hw[1]}×{want_hw[0]} ≠ 바탕 {have_hw[1]}×{have_hw[0]}")
    if abs(float(cfg.get("fps", fps)) - fps) > 1e-6:
        problems.append(f"fps: G 학습 {cfg.get('fps')} ≠ 바탕 {fps}")
    if not want_hw:
        # 옛 체크포인트(2026-11-09 트랙 기록 이전): 체크포인트의 scale 을 그대로 썼다면 학습 때와 같은 해상도다.
        print("[경고] 체크포인트에 학습 해상도(train_hw)가 없음(옛 체크포인트) — 체크포인트의 scale 로 맞춘 것으로 간주")
    if problems and not allow_mismatch:
        raise SystemExit("트랙 불일치로 중단합니다:\n  - " + "\n  - ".join(problems)
                         + "\n의도한 것이면 --allow-mismatch (결과에 그대로 기록됨)")
    for p in problems:
        print(f"[경고] {p}")


def to_tensor(clips_bgr: np.ndarray, device) -> torch.Tensor:
    """(B,T,H,W,3) uint8 → (B,3,T,H,W) float 0~1 (채널 순서는 BGR 그대로)."""
    return torch.from_numpy(clips_bgr).to(device).permute(0, 4, 1, 2, 3).float() / 255.0


def main() -> None:
    ap = argparse.ArgumentParser(description="LUMI 조건부 GAN 학습")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "oracle_768x432_24fps",
                    help="오라클 단계의 데이터 폴더(트랙). backgrounds_train.npz 의 fps·해상도를 그대로 쓴다")
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
                    help="학습 해상도 배율. 판정 해상도 = 학습 해상도 원칙이라 1.0 만 허용(그 외는 --debug-scale)")
    ap.add_argument("--amp", action="store_true",
                    help="GPU 효율: G·D 합성곱만 bf16 (판정기·sRGB·다양성 손실은 fp32). 빛 Δ 수치 오차 약 0.2 cd/m² 수준")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="GPU 메모리: G 와 소프트 판정기를 그래디언트 체크포인팅(역전파 때 다시 계산, 결과 동일)")
    ap.add_argument("--micro-batch", type=int, default=0,
                    help="GPU 메모리: 배치를 이 크기로 나눠 그래디언트 누적(0 = 나누지 않음). GroupNorm 은 샘플별이라 결과 동일")
    ap.add_argument("--debug-scale", action="store_true",
                    help="배율 ≠ 1.0 을 동작 확인용으로만 허용. 결과는 학습용 데이터에 쓸 수 없다(check_track 이 막음)")
    args = ap.parse_args()
    if args.scale != 1.0 and not args.debug_scale:
        raise SystemExit(f"--scale {args.scale}: 판정 해상도와 학습 해상도를 통일하기로 했다(2026-11-09). "
                         "동작 확인만이라면 --debug-scale 를 함께 주세요.")
    if args.batch_size < 2 or args.batch_size % 2:
        raise SystemExit("--batch-size 는 2 이상의 짝수여야 합니다(목표 '위험'/'근처 안전'을 절반씩).")

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, (__import__("os").cpu_count() or 2)))
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA 를 요청했지만 사용할 수 없습니다.")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        # Windows 는 GPU 메모리가 넘치면 시스템 RAM 으로 흘려 수십 배 느려진다(2026-11-10 측정) → 넘치면 바로 오류로 멈춘다
        torch.cuda.set_per_process_memory_fraction(0.95)
    mb = args.micro_batch or args.batch_size
    if args.batch_size % mb:
        raise SystemExit("--batch-size 는 --micro-batch 의 배수여야 합니다.")
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda")
    from torch.utils.checkpoint import checkpoint

    def run_g(clean_, mask_, target_, z_):
        if args.grad_ckpt:
            return checkpoint(G, clean_, mask_, target_, z_, use_reentrant=False)
        return G(clean_, mask_, target_, z_)

    def run_judge(out_, target_):
        f = lambda o: judge_loss(luminance_rel(srgb_to_linear(o.float())), target_, fps)
        if args.grad_ckpt:
            return checkpoint(f, out_, use_reentrant=False)
        return f(out_)
    npz = np.load(args.data / "backgrounds_train.npz")
    bg = npz["clips"]                                                   # (N,T,H,W,3) uint8 BGR
    if bg.ndim != 5 or len(bg) == 0:
        raise SystemExit(f"바탕 클립이 없습니다: {args.data}/backgrounds_train.npz (make_oracle_dataset.py 먼저)")
    bg = rescale_clips(bg, args.scale)
    n, t, h, w, _ = bg.shape
    # fps 는 데이터에서 읽는다(이전에는 24.0 고정 → 30fps 트랙에서 1초 창을 틀리게 잴 뻔함)
    if "fps" not in npz:
        raise SystemExit("backgrounds_train.npz 에 fps 가 없습니다. make_oracle_dataset.py 로 다시 만드세요.")
    fps = float(npz["fps"])
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

        # ---- 2. D: 진짜(깨끗한 바탕 프레임) 대 가짜(G 출력 프레임) — 마이크로배치로 나눠 누적(평균은 같음)
        real = to_tensor(bg[rng.integers(0, n, b)], device)
        opt_d.zero_grad(set_to_none=True)
        loss_d_sum = 0.0
        for i in range(0, b, mb):
            sl = slice(i, i + mb)
            with torch.no_grad(), amp:
                fake, _ = G(clean[sl], mask[sl], target[sl], z1[sl])
            with amp:
                loss_d = (F.relu(1 - D(random_frames(real[sl], 4))).mean()
                          + F.relu(1 + D(random_frames(fake.float(), 4))).mean())
            (loss_d.float() * (mb / b)).backward()
            loss_d_sum += float(loss_d) * mb / b
        opt_d.step()
        loss_d = torch.tensor(loss_d_sum)

        # ---- 3. G: 판정 목표 + 자연스러움 + 리듬 다양성
        opt_g.zero_grad(set_to_none=True)
        lj_sum, adv_sum, div_sum, counts = 0.0, 0.0, 0.0, []
        for i in range(0, b, mb):
            sl = slice(i, i + mb)
            with amp:
                out1, d1 = run_g(clean[sl], mask[sl], target[sl], z1[sl])
                out2, d2 = run_g(clean[sl], mask[sl], target[sl], z2[sl])
                adv = -D(random_frames(out1, 4)).mean().float()
            lj1, c1 = run_judge(out1, target[sl])
            lj2, _ = run_judge(out2, target[sl])
            div = spectral_diversity(d1.float(), d2.float(), mask[sl], z1[sl], z2[sl])
            loss_g = args.lambda_judge * (lj1 + lj2) / 2 + args.lambda_adv * adv + args.lambda_div * div
            (loss_g * (mb / b)).backward()
            lj_sum += float((lj1 + lj2) / 2) * mb / b
            adv_sum += float(adv) * mb / b
            div_sum += float(div) * mb / b
            counts.append(c1.detach())
        opt_g.step()
        c1 = torch.cat(counts)
        lj1 = lj2 = torch.tensor(lj_sum)
        adv, div = torch.tensor(adv_sum), torch.tensor(div_sum)

        if step == 1 or step % 25 == 0 or step == args.steps:
            haz = target > 0.5
            rec = {"step": step, "loss_d": round(float(loss_d), 4), "judge": round(float((lj1 + lj2) / 2), 4),
                   "adv": round(float(adv), 4), "div": round(float(div), 4),
                   "soft_count_hazard": round(float(c1[haz].mean().detach()), 2),
                   "soft_count_near": round(float(c1[~haz].mean()), 2),
                   "elapsed_s": round(time.perf_counter() - start, 1)}
            if device.type == "cuda":
                rec["gpu_peak_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            with args.log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "lumi_light_gan_v1", "config": {k: str(v) if isinstance(v, Path) else v
                                                          for k, v in vars(args).items()} | {"fps": fps,
                "soft_judge_version": SOFT_JUDGE_VERSION, "train_hw": [h, w], "frames": t,
                "data_track": args.data.name},
                "train_log": [json.loads(x) for x in args.log.read_text(encoding="utf-8").splitlines() if x],
                "generator": G.state_dict(), "discriminator": D.state_dict()}, args.checkpoint)
    print("저장:", args.checkpoint)


if __name__ == "__main__":
    main()
