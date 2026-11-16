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
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from lumi_synth import measure, oracle
from lumi_synth.gan import (HAZARDS, SOFT_JUDGE_VERSION, STRIPE_LO, stripe_reversal, FrameDiscriminator, judge_loss, judge_loss_red, judge_loss_stripe,
                            luminance_rel, make_generator, palette_diversity, random_frames, red_crisp_loss, spectral_diversity, srgb_to_linear,
                            stripe_diversity)

ROOT = Path(__file__).resolve().parents[1]


def sample_masks(n: int, h: int, w: int, rng: np.random.Generator, hazard: str = "lum") -> np.ndarray:
    """오라클과 같은 마스크 생성기. 'hazard' 의도의 면적 범위(0.32~0.70, 10% 는 화면 전체)를 써서, 위험/안전의 차이가
    면적이 아니라 '빛의 시간 변화'에서 나오게 한다(문턱 근처 안전 = 가장 어려운 음성 사례).

    줄무늬(hazard='stripe')는 면적 기준이 40% 초과라 영역을 0.45~0.85 로 뽑는다. 위험/근처 안전의 차이는 G 가
    그 안에서 줄무늬를 얼마나 넓게·오래 덮느냐에서 나온다."""
    out = []
    for _ in range(n):
        p = oracle.sample_params("hazard", rng)
        if hazard == "stripe" and p.shape != "full":
            p = dataclasses.replace(p, area=float(rng.uniform(0.45, 0.85)))
        out.append(oracle.make_mask(p, h, w))
    return np.stack(out)


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
    ap.add_argument("--hazard", default="lum", choices=HAZARDS,
                    help="만들 위험 종류: lum 휘도 깜빡임(기본) · red 적색 깜빡임 · stripe 줄무늬. 종류마다 따로 학습한다(gan.py §3-2)")
    ap.add_argument("--lambda-judge", type=float, default=1.0)
    ap.add_argument("--lambda-adv", type=float, default=0.1)
    ap.add_argument("--lambda-div", type=float, default=0.05)
    ap.add_argument("--lambda-crisp", type=float, default=0.0,
                    help="적색 전용: ρ 를 한 프레임에 끝나는 켜짐·꺼짐으로 모는 손실의 가중치(gan.red_crisp_loss). 0 = 끔")
    ap.add_argument("--red-palette", nargs="?", const=True, default=False,
                    help="적색 v2: 빨강과 '빨강 아닌 상태'의 색·밝기·평평함을 G 가 고른다(gan.red_palette, 다양성 손실에 색 조건 차이를 더함). "
                         "'--red-palette z' 이면 v3: 색 조건을 z 에서 균등하게 뽑는다(--zdim 12 이상), "
                         "'--red-palette z2' 이면 v4: v3 + 등휘도 조건, 양 끝값이 자주 나오게")
    ap.add_argument("--stripe-cond", default="mlp", choices=("mlp", "z", "z2", "z2dc", "z3"),
                    help="줄무늬: mlp(첫 버전, 조건을 배움) | z(v2: 조건을 z 에서 균등하게 — 정지·반전·흐름, 줄 모양, 반전 파형. --zdim 8 이상) "
                         "| z2(v3: v2 + 덮는 정도의 시간 흐림 σ 0~12 프레임 — 진짜 정지 줄무늬. --zdim 10)")
    ap.add_argument("--lum-smooth", action="store_true",
                    help="휘도 전용: z 의 마지막 성분으로 시간 흐림(σ 0~2 프레임)을 정해 사인·삼각파꼴 깜빡임도 만들게 한다(gan.temporal_blur)")
    ap.add_argument("--div-norm", action="store_true",
                    help="다양성 손실에서 스펙트럼을 크기로 정규화(세기가 아니라 리듬 차이만 봄, 실험용 — gan.spectral_diversity)")
    ap.add_argument("--overwrite", action="store_true", help="같은 이름의 체크포인트가 있어도 덮어쓴다(기본: 멈춤 — 학습한 매개변수 보호)")
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

    if args.checkpoint.exists() and not args.overwrite:
        raise SystemExit(f"{args.checkpoint} 가 이미 있습니다(학습한 매개변수). 다른 --checkpoint 이름을 쓰거나 --overwrite")
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

    def run_judge(outs_, mask_, target_):
        """종류별 판정 손실. outs_ = G 의 출력 (X, 필드[, 줄무늬 조건]). 돌려주는 둘째 값: 휘도·적색 = 1초 전이 수, 줄무늬 = 지속(초)."""
        if args.hazard == "stripe":           # 대리 판정은 영상이 아니라 G 가 정한 값(덮는 정도·밝기)에서 계산
            par_ = outs_[2].float()
            hi_ = par_[:, 3]
            if par_.shape[1] > 4:             # v2: 반전으로 프레임마다 명암이 달라진다
                hi_ = STRIPE_LO + (hi_[:, None] - STRIPE_LO) * stripe_reversal(par_, outs_[1].shape[2], fps).abs()
            return judge_loss_stripe(mask_ * outs_[1].float(), hi_, target_, fps)
        if args.hazard == "red":
            f = lambda o: judge_loss_red(srgb_to_linear(o.float()), target_, fps)
        else:
            f = lambda o: judge_loss(luminance_rel(srgb_to_linear(o.float())), target_, fps)
        if args.grad_ckpt:
            return checkpoint(f, outs_[0], use_reentrant=False)
        return f(outs_[0])
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
    G = make_generator(args.hazard, args.width, args.zdim, fps, red_palette=args.red_palette, lum_smooth=args.lum_smooth,
                       stripe_cond=args.stripe_cond).to(device)
    D = FrameDiscriminator(args.width).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    print(f"종류 {args.hazard} · 바탕 {n}개 [{t}×{h}×{w}] · G {sum(p.numel() for p in G.parameters()):,} · "
          f"D {sum(p.numel() for p in D.parameters()):,} 파라미터 · {device}", flush=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text("", encoding="utf-8")
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        b = args.batch_size
        clean = to_tensor(bg[rng.integers(0, n, b)], device)
        mask = torch.from_numpy(sample_masks(b, h, w, rng, args.hazard)).to(device).float()[:, None, None]  # [B,1,1,H,W]
        target = torch.from_numpy((np.arange(b) % 2 == 0).astype(np.float32)).to(device)        # 위험/근처 절반씩
        z1, z2 = torch.randn(b, args.zdim, device=device), torch.randn(b, args.zdim, device=device)

        # ---- 2. D: 진짜(깨끗한 바탕 프레임) 대 가짜(G 출력 프레임) — 마이크로배치로 나눠 누적(평균은 같음)
        real = to_tensor(bg[rng.integers(0, n, b)], device)
        opt_d.zero_grad(set_to_none=True)
        loss_d_sum = 0.0
        for i in range(0, b, mb):
            sl = slice(i, i + mb)
            with torch.no_grad(), amp:
                fake = G(clean[sl], mask[sl], target[sl], z1[sl])[0]
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
                o1 = run_g(clean[sl], mask[sl], target[sl], z1[sl])
                o2 = run_g(clean[sl], mask[sl], target[sl], z2[sl])
                adv = -D(random_frames(o1[0], 4)).mean().float()
            lj1, c1 = run_judge(o1, mask[sl], target[sl])
            lj2, _ = run_judge(o2, mask[sl], target[sl])
            if args.hazard == "stripe" and args.stripe_cond in ("z", "z2", "z2dc", "z3"):   # v2: 조건은 z 가 정하므로 덮는 정도의 시간 리듬만
                div = spectral_diversity(o1[1].float(), o2[1].float(), mask[sl], z1[sl], z2[sl])
            elif args.hazard == "stripe":     # 줄무늬 조건(cpd·방향·속도·밝기)이 z 에 따라 달라지게
                div = stripe_diversity(o1[2].float(), o2[2].float(), z1[sl], z2[sl])
            else:                             # 휘도: 빛 Δ, 적색: 빨강 정도 ρ 의 시간 리듬이 z 에 따라 달라지게
                div = spectral_diversity(o1[1].float(), o2[1].float(), mask[sl], z1[sl], z2[sl], normalize=args.div_norm)
                if args.hazard == "red" and args.red_palette is True:   # 적색 v2: 색 조건도 z 에 따라 달라지게 (v3 는 z 가 곧 색)
                    div = div + palette_diversity(o1[2].float(), o2[2].float(), z1[sl], z2[sl])
            loss_g = args.lambda_judge * (lj1 + lj2) / 2 + args.lambda_adv * adv + args.lambda_div * div
            if args.hazard == "red" and args.lambda_crisp > 0:
                loss_g = loss_g + args.lambda_crisp * (red_crisp_loss(o1[1].float(), mask[sl]) + red_crisp_loss(o2[1].float(), mask[sl])) / 2
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
                # 학습 때의 판정 규칙(소프트 판정기가 measure 에서 읽는 값). 이 기록이 없는 체크포인트는
                # 2026-11-14 이전 규칙(동시성 창 2프레임, 1초 창 round)으로 학습한 것일 수 있다
                "judge_rule": {"d3_sync_ms": measure.D3_SYNC_MS, "window_1s": "ceil(fps)"},
                "data_track": args.data.name},
                "train_log": [json.loads(x) for x in args.log.read_text(encoding="utf-8").splitlines() if x],
                "generator": G.state_dict(), "discriminator": D.state_dict()}, args.checkpoint)
    print("저장:", args.checkpoint)


if __name__ == "__main__":
    main()
