"""3단계: G 로 만든 데이터로 ROI 탐지기(roi.RoiNet)를 학습한다.

    python scripts/train_roi.py --data data/gdr/v1/train --val data/gdr/v1/val --epochs 8 --run a

입력 = 영상(위험·근처 안전·깨끗한 바탕), 목표 = 하드 판정기가 붙인 종류별 튜브(1/4 해상도 면적 평균).
체크포인트: data/gdr/<이름>/roi_<run>.pt · 학습 기록: checks/gdr/train_log_roi_<run>.jsonl
좌우 뒤집기만 쓴다(판정 규칙이 좌우 대칭 — 줄무늬 쿼드트리는 블록 경계 때문에 근사).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from gdr_common import ROOT, eval_roi, prefetch, read_manifest, require_split, summarize_roi, to_x
from lumi_synth import roi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--val", type=Path, default=None)
    ap.add_argument("--run", default="a")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--target", default="hazard", choices=("hazard", "loss"),
                    help="hazard: 종류별 위험 튜브(탐지기) · loss: 정보 손실 M_loss = α ≥ 0.95 (손실 추정기, E3)")
    ap.add_argument("--features", default="rgb", choices=("rgb", "color", "color2"), help="color: 판정 규칙의 양(휘도·색도·포화 비율·프레임 간 변화)을 입력에 더함")
    ap.add_argument("--overwrite", action="store_true", help="같은 이름의 체크포인트가 있어도 덮어쓴다(기본: 멈춤 — 학습한 매개변수 보호)")
    ap.add_argument("--oversample", default="", help="예: red:2 — 그 종류 위험 표본을 epoch 마다 k 번 넣는다(종류 비율 희석 검증용)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="표본을 N 개만 (동작 확인용)")
    ap.add_argument("--crop", type=int, nargs=2, default=None, metavar=("H", "W"), help="동작 확인용: 가운데를 잘라 학습")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    metas = read_manifest(args.data)
    val = read_manifest(args.val) if args.val else []
    if args.limit:
        metas, val = metas[:args.limit], val[:args.limit]
    require_split(metas, "train")
    if val:
        require_split(val, "eval")
    if args.target == "loss":                 # α 옆 파일이 있는 표본만(바탕 그대로는 손실 0 으로)
        metas = [m for m in metas if m["hazard"] is None or (Path(m["folder"]) / f"{m['id']}.alpha.npz").exists()]
        val = []
    if args.oversample:
        kind, k = args.oversample.split(":")
        extra = [m for m in metas if (m["judge"].get(kind) or {}).get("dangerous")] * (int(k) - 1)
        print(f"{kind} 위험 표본 {len(extra) // max(1, int(k) - 1)}개를 {k}배로 (+{len(extra)})", flush=True)
        metas = metas + extra
    net = roi.RoiNet(args.width, args.features, n_out=1 if args.target == "loss" else len(roi.HAZARDS)).to(device)
    steps_total = args.epochs * (len(metas) // args.batch_size)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps_total, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    name = f"{'loss' if args.target == 'loss' else 'roi'}_{args.run}"
    ckpt_path = args.data.parent / f"{name}.pt"
    log_path = ROOT / "checks" / "gdr" / f"train_log_{args.data.parent.name}_{name}.jsonl"
    if ckpt_path.exists() and not args.overwrite and not args.limit:
        raise SystemExit(f"{ckpt_path} 가 이미 있습니다(학습한 매개변수). 다른 --run 이름을 쓰거나 --overwrite")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"표본 {len(metas)} · 검증 {len(val)} · 스텝 {steps_total} · 매개변수 {n_par / 1e6:.2f}M · {device}", flush=True)

    def crop(a: np.ndarray, axes: tuple[int, int]) -> np.ndarray:
        if not args.crop:
            return a
        h, w = a.shape[axes[0]], a.shape[axes[1]]
        sl = [slice(None)] * a.ndim
        sl[axes[0]] = slice((h - args.crop[0]) // 2, (h - args.crop[0]) // 2 + args.crop[0])
        sl[axes[1]] = slice((w - args.crop[1]) // 2, (w - args.crop[1]) // 2 + args.crop[1])
        return a[tuple(sl)]

    t0, step = time.time(), 0
    with log_path.open("w", encoding="utf-8") as lf:
        for epoch in range(args.epochs):
            net.train()
            order = [metas[i] for i in rng.permutation(len(metas))]
            batch, losses = [], []
            for s in prefetch(order):
                batch.append(s)
                if len(batch) < args.batch_size:
                    continue
                x = to_x([crop(b["x"], (1, 2)) for b in batch], device)
                if args.target == "loss":
                    tube = torch.from_numpy(np.stack([crop((b["alpha"] if b["alpha"] is not None else np.zeros(b["x"].shape[:3], np.float32))
                                                            >= 0.95, (1, 2))[None] for b in batch])).to(device)
                else:
                    tube = torch.from_numpy(np.stack([crop(b["tubes"], (2, 3)) for b in batch])).to(device)
                batch = []
                if rng.random() < 0.5:
                    x, tube = x.flip(-1), tube.flip(-1)
                target = roi.pool_tubes(tube)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = net(x)
                loss = roi.roi_loss(logits, target)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                step += 1
                losses.append(float(loss.detach()))
                if step % 25 == 0:
                    rec = {"step": step, "epoch": epoch, "loss": round(float(np.mean(losses[-25:])), 4),
                           "elapsed_s": round(time.time() - t0, 1)}
                    if device.type == "cuda":
                        rec["gpu_peak_GB"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)
                    lf.write(json.dumps(rec) + "\n")
                    lf.flush()
                    print(json.dumps(rec), flush=True)
            rec = {"epoch_end": epoch, "loss": round(float(np.mean(losses)), 4), "elapsed_s": round(time.time() - t0, 1)}
            if val and not args.crop:
                su = summarize_roi(eval_roi(net, val, device))
                rec["val"] = {k: {m: (round(v, 3) if isinstance(v, float) else v) for m, v in su[k].items()} for k in su}
            lf.write(json.dumps(rec) + "\n")
            lf.flush()
            print(json.dumps(rec), flush=True)
            torch.save({"model": net.state_dict(),
                        "config": {"width": args.width, "epochs": args.epochs, "epoch_done": epoch + 1, "seed": args.seed,
                                   "data": str(args.data), "n_train": len(metas), "scale": roi.SCALE,
                                   "hazards": list(roi.HAZARDS), "lr": args.lr, "features": args.features, "target": args.target, "oversample": args.oversample, "batch_size": args.batch_size}}, ckpt_path)
    print(f"저장: {ckpt_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
