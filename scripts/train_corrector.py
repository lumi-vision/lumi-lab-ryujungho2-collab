"""4단계: G 로 만든 쌍(원본, 위험 영상)으로 보정기(roi.Corrector)를 학습한다.

    python scripts/train_corrector.py --data data/gdr/v1/train --epochs 6 --run a

입력 = 위험 영상 + 판정기 튜브(가장자리를 부드럽게), 목표 = 원본. 판정기 튜브가 빈 표본은 쓰지 않는다
(튜브가 0 이면 보정기는 구조적으로 입력을 그대로 내므로 배울 것이 없다).
화면을 잘라(기본 256×384, 48프레임 전부) 학습한다 — 보정기는 완전 합성곱이라 전체 화면에 그대로 쓴다.
체크포인트: data/gdr/<이름>/cor_<run>.pt · 학습 기록: checks/gdr/train_log_cor_<run>.jsonl

주의: 목표가 '원본'이라는 것은 이 검증의 선택이다(PLAN §2 의 B 정답 = '가장 적게 고치기'는 아직 구현하지 않았다).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from gdr_common import ROOT, prefetch, read_manifest, require_split, to_x
from lumi_synth import roi


def pick_crop(union: np.ndarray, ch: int, cw: int, rng: np.random.Generator) -> tuple[int, int]:
    """튜브가 많이 들어오는 조각을 고른다(무작위 8번 중 튜브 비율이 가장 큰 것)."""
    h, w = union.shape
    best, best_frac = (0, 0), -1.0
    for _ in range(8):
        y0, x0 = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
        frac = float(union[y0:y0 + ch, x0:x0 + cw].mean())
        if frac > best_frac:
            best, best_frac = (y0, x0), frac
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--run", default="a")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--crop", type=int, nargs=2, default=(256, 384), metavar=("H", "W"))
    ap.add_argument("--lambda-t", type=float, default=1.0, help="프레임 간 변화 손실의 가중치")
    ap.add_argument("--no-mask", action="store_true", help="E0: 마스크 없이(튜브 = 화면 전체) — 보정기가 어디를 고칠지 스스로 정한다")
    ap.add_argument("--loss-input", action="store_true", help="E3: 정보 손실 지도 M_loss(α ≥ 0.95, 정답)를 한 채널 더 준다")
    ap.add_argument("--overwrite", action="store_true", help="같은 이름의 체크포인트가 있어도 덮어쓴다(기본: 멈춤 — 학습한 매개변수 보호)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    metas = [m for m in read_manifest(args.data) if sum(m["tube_frac"].values()) > 0]
    if args.limit:
        metas = metas[:args.limit]
    require_split(metas, "train")
    kinds = {k: sum(m["tube_frac"][k] > 0 for m in metas) for k in roi.HAZARDS}
    net = roi.Corrector(args.width, loss_in=args.loss_input).to(device)
    steps_total = args.epochs * (len(metas) // args.batch_size)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps_total, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    name = f"cor_{args.run}"
    ckpt_path = args.data.parent / f"{name}.pt"
    log_path = ROOT / "checks" / "gdr" / f"train_log_{args.data.parent.name}_{name}.jsonl"
    if ckpt_path.exists() and not args.overwrite and not args.limit:
        raise SystemExit(f"{ckpt_path} 가 이미 있습니다(학습한 매개변수). 다른 --run 이름을 쓰거나 --overwrite")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"튜브가 있는 표본 {len(metas)} ({kinds}) · 스텝 {steps_total} · {device}", flush=True)
    ch, cw = args.crop

    t0, step = time.time(), 0
    with log_path.open("w", encoding="utf-8") as lf:
        for epoch in range(args.epochs):
            net.train()
            order = [metas[i] for i in rng.permutation(len(metas))]
            xs, cs, ts, ls, losses, before = [], [], [], [], [], []
            for s in prefetch(order):
                y0, x0 = pick_crop(s["tubes"].any(axis=(0, 1)), ch, cw, rng)
                xs.append(s["x"][:, y0:y0 + ch, x0:x0 + cw])
                cs.append(s["clean"][:, y0:y0 + ch, x0:x0 + cw])
                ts.append(s["tubes"][:, :, y0:y0 + ch, x0:x0 + cw])
                al = s["alpha"] if s["alpha"] is not None else np.zeros(s["x"].shape[:3], np.float32)
                ls.append((al[:, y0:y0 + ch, x0:x0 + cw] >= 0.95)[None])
                if len(xs) < args.batch_size:
                    continue
                x, clean = to_x(xs, device), to_x(cs, device)
                tube = torch.from_numpy(np.stack(ts)).to(device)
                lmap = torch.from_numpy(np.stack(ls)).to(device).float()
                xs, cs, ts, ls = [], [], [], []
                if rng.random() < 0.5:
                    x, clean, tube, lmap = x.flip(-1), clean.flip(-1), tube.flip(-1), lmap.flip(-1)
                if args.no_mask:
                    m = torch.ones(tube.shape, device=device)
                else:
                    m = roi.soften(tube, grow=int(rng.integers(0, 5)), blur=2)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    y = net(x, m, lmap if args.loss_input else None)
                m1 = m.amax(dim=1, keepdim=True)
                if args.no_mask:                   # E0: 손실은 위험 영역(튜브) 안 + 밖(원본 그대로여야 함) 모두
                    m1 = torch.ones_like(m1)
                loss = roi.correct_loss(y.float(), clean, m1, args.lambda_t)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                step += 1
                losses.append(float(loss.detach()))
                before.append(float(roi.correct_loss(x, clean, m1, args.lambda_t)))
                if step % 25 == 0:
                    rec = {"step": step, "epoch": epoch, "loss": round(float(np.mean(losses[-25:])), 5),
                           "loss_no_correction": round(float(np.mean(before[-25:])), 5), "elapsed_s": round(time.time() - t0, 1)}
                    if device.type == "cuda":
                        rec["gpu_peak_GB"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)
                    lf.write(json.dumps(rec) + "\n")
                    lf.flush()
                    print(json.dumps(rec), flush=True)
            torch.save({"model": net.state_dict(),
                        "config": {"width": args.width, "epochs": args.epochs, "epoch_done": epoch + 1, "seed": args.seed,
                                   "data": str(args.data), "n_train": len(metas), "crop": list(args.crop),
                                   "lambda_t": args.lambda_t, "target": "clean", "lr": args.lr,
                                   "mask": "none" if args.no_mask else "tube", "loss_input": args.loss_input}}, ckpt_path)
    print(f"저장: {ckpt_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
