"""보정 모델 학습 (docs/CORRECTION_PROTOCOL.md §3).

    python scripts/train_correction.py --run h_s0                 # 제안 H
    python scripts/train_correction.py --run abl_noattn --no-attn  # 절제 H−attn
    python scripts/train_correction.py --run abl_nophys --no-phys  # 절제 H−phys
    python scripts/train_correction.py --run abl_nolayer --w-layer 0
    python scripts/train_correction.py --run abl_nosafe --w-safe 0

데이터: data/hazard7/<data>/{train,val} — 합성 mp4, 원본 mp4, 층 정답(layers.npz: A·B, 1/4), Partial B2 라벨(labels.npz, 있으면).
학습: 384×216 로 줄인 뒤 무작위로 잘라(기본 112×192) T=192 전체. 검증: val 전체 화면 384×216.
손실 (§3.2): 층 L1 + 복원 Charbonnier(A ≥ 0.05 가중 1, 정보 없음 0.2) + LPIPS(프레임 일부) + 시간 변화 일치 + 위험 지도 BCE + 안전(소프트 판정).
조기 종료: val 손실이 --patience epoch 동안 --min-delta 이상 줄지 않으면 멈추고 가장 좋은 epoch 로 되돌림.
결과: data/correction/<run>.pt · checks/correction/<run>_log.jsonl · <run>_curves.png
⚠️ 영상은 광과민성 위험 자극이다. 프레임은 메모리에서만 다루고 재생·저장하지 않는다.
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # 조각화로 '예약됐지만 빈' 메모리가 2.5 GB 쌓여 메모리 부족이 났다(2026-11-21)
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lumi_synth import gan  # noqa: E402
from lumi_synth.corrnet import A_MIN, CorrNet, lin_to_srgb, srgb_to_lin  # noqa: E402

FPS, TW, TH = 24.0, 384, 216


def decode_small(path: Path) -> np.ndarray:
    import av
    import cv2
    with av.open(str(path)) as c:
        return np.stack([cv2.resize(f.to_ndarray(format="bgr24"), (TW, TH), interpolation=cv2.INTER_AREA)
                         for f in c.decode(c.streams.video[0])])


def load_item(d: Path, sid: str) -> dict:
    x = decode_small(d / f"{sid}.mp4")
    y = decode_small(d / f"{sid}.clean.mp4")
    with np.load(d / f"{sid}.layers.npz") as z:
        A, B = z["A"].astype(np.float32), z["B"].astype(np.float32)          # [T,108,192,3] = 학습 해상도의 1/2
    haz = None
    lab = d / f"{sid}.labels.npz"
    if lab.exists():
        with np.load(lab) as z:
            if "weight" in z:
                haz = z["weight"].astype(np.float32) / 255.0                   # [T,108,192] Partial B2 픽셀 위험 마스크
    n = min(len(x), len(y), len(A))
    return {"x": x[:n], "y": y[:n], "A": A[:n], "B": B[:n], "haz": None if haz is None else haz[:n], "id": sid}


def items(d: Path) -> list[str]:
    if not (d / "gen.jsonl").exists():
        return []
    rows = [json.loads(l) for l in (d / "gen.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r["id"] for r in rows if "error" not in r and (d / f"{r['id']}.layers.npz").exists()]


def loader(d: Path, ids: list[str], workers: int = 4, depth: int = 6):
    """스레드로 디코드해 미리 채운다(PyAV·cv2 는 GIL 을 놓는다)."""
    import concurrent.futures as cf
    q: queue.Queue = queue.Queue(depth)

    def run():
        with cf.ThreadPoolExecutor(workers) as ex:
            for it in ex.map(lambda s: load_item(d, s), ids):
                q.put(it)
        q.put(None)

    threading.Thread(target=run, daemon=True).start()
    while (it := q.get()) is not None:
        yield it


def to_tensors(it: dict, dev, crop: tuple[int, int] | None, rng) -> dict:
    T, H, W = it["x"].shape[:3]
    if crop:
        ch, cw = crop
        y0 = int(rng.integers(0, (H - ch) // 2 + 1)) * 2
        x0 = int(rng.integers(0, (W - cw) // 2 + 1)) * 2
    else:
        ch, cw, y0, x0 = H, W, 0, 0
    sl = (slice(None), slice(y0, y0 + ch), slice(x0, x0 + cw))
    sl2 = (slice(None), slice(y0 // 2, (y0 + ch) // 2), slice(x0 // 2, (x0 + cw) // 2))
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    X = t(it["x"][sl]).permute(0, 3, 1, 2).float()[None] / 255.0                     # [1,T,3,h,w] sRGB
    Y = srgb_to_lin(t(it["y"][sl]).permute(0, 3, 1, 2).float()[None] / 255.0)        # 선형
    A = t(it["A"][sl2]).permute(0, 3, 1, 2)[None]
    B = t(it["B"][sl2]).permute(0, 3, 1, 2)[None]
    haz = None if it["haz"] is None else t(it["haz"][sl2])[None, :, None]
    if rng.random() < 0.5:                                                           # 좌우 뒤집기(판정 규칙 대칭)
        X, Y, A, B = X.flip(-1), Y.flip(-1), A.flip(-1), B.flip(-1)
        haz = None if haz is None else haz.flip(-1)
    return {"X": X, "Y": Y, "A": A, "B": B, "haz": haz}


def charbonnier(a, b, w=None, eps=1e-3):
    e = torch.sqrt((a - b) ** 2 + eps ** 2)
    return e.mean() if w is None else (e * w).sum() / w.sum().clamp_min(1.0)


def safety_loss(Y: torch.Tensor) -> torch.Tensor:
    """소프트 판정(휘도·적색)의 1초 창 최대 화면 전이 수가 5(하드 기준 7회 미만에 여유) 넘는 만큼 벌점."""
    lin = Y.transpose(1, 2)                                                          # [B,3,T,H,W] 선형 BGR
    yl = 0.0722 * lin[:, 0] + 0.7152 * lin[:, 1] + 0.2126 * lin[:, 2]
    c_l = gan.soft_judge(yl, FPS)["count"]
    c_r = gan.soft_judge_red(lin, FPS)["count"]
    return (F.softplus(c_l - 5.0) + F.softplus(c_r - 5.0)).mean()


def compute_loss(o: dict, b: dict, a: argparse.Namespace, lp=None) -> tuple[torch.Tensor, dict]:
    Yh, Y = o["Y"].float(), b["Y"]
    Agt = b["A"]
    rec_w = F.interpolate(Agt.mean(2, keepdim=True).flatten(0, 1), size=Y.shape[-2:], mode="nearest").view(*Y.shape[:2], 1, *Y.shape[-2:])
    rec_w = torch.where(rec_w >= A_MIN, 1.0, 0.2)                                    # 정보 없음은 약하게(지어내기 억제)
    parts = {}
    parts["rec"] = charbonnier(Yh, Y, rec_w.expand_as(Y))
    parts["layer"] = (o["A"].float() - Agt).abs().mean() + (o["B"].float() - b["B"]).abs().mean() * 4
    dY, dYh = Y[:, 1:] - Y[:, :-1], Yh[:, 1:] - Yh[:, :-1]
    parts["temp"] = (dYh - dY).abs().mean()                                          # 원본의 프레임 간 변화와 맞추기
    if b["haz"] is not None:
        parts["haz"] = F.binary_cross_entropy_with_logits(o["haz"].float(), b["haz"])
    if a.w_safe > 0:
        parts["safe"] = safety_loss(Yh)
    if lp is not None and a.w_lpips > 0:
        idx = torch.randperm(Y.shape[1], device=Y.device)[:6]
        s = lambda z: (lin_to_srgb(z[0, idx]).flip(1) * 2 - 1)                       # BGR → RGB, −1~1
        parts["lpips"] = lp(s(Yh), s(Y)).mean()
    w = {"rec": 1.0, "layer": a.w_layer, "temp": 0.5, "haz": 0.2, "safe": a.w_safe, "lpips": a.w_lpips}
    loss = sum(w[k] * v for k, v in parts.items())
    return loss, {k: float(v.detach()) for k, v in parts.items()}


@torch.no_grad()
def validate(net, d: Path, ids: list[str], dev, a) -> dict:
    net.eval()
    acc, ps = {}, []
    rng = np.random.default_rng(0)
    torch.cuda.empty_cache()
    for it in loader(d, ids, workers=4):
        try:
            b = to_tensors(it, dev, None, rng)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = net(b["X"])
        except torch.OutOfMemoryError:                     # 검증도 메모리 부족이면 그 표본만 건너뜀(수를 기록)
            torch.cuda.empty_cache(); acc.setdefault("skipped_oom", []).append(1.0)
            continue
        _, parts = compute_loss(o, b, argparse.Namespace(**{**vars(a), "w_lpips": 0.0}))
        for k, v in parts.items():
            acc.setdefault(k, []).append(v)
        rec = F.interpolate(b["A"].mean(2, keepdim=True).flatten(0, 1), size=b["Y"].shape[-2:], mode="nearest").view(*b["Y"].shape[:2], 1, *b["Y"].shape[-2:]) >= A_MIN
        mse = (((o["Y"].float() - b["Y"]) ** 2) * rec).sum() / (rec.sum() * 3).clamp_min(1)
        ps.append(float(-10 * torch.log10(mse.clamp_min(1e-10))))
    net.train()
    out = {k: round(float(np.mean(v)), 5) for k, v in acc.items()}
    out["psnr_rec"] = round(float(np.mean(ps)), 3)
    out["loss"] = round(out["rec"] + a.w_layer * out["layer"] + 0.5 * out["temp"] + 0.2 * out.get("haz", 0) + a.w_safe * out.get("safe", 0), 5)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="big1")
    ap.add_argument("--crop", type=int, nargs=2, default=[112, 192])
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--w-layer", type=float, default=1.0)
    ap.add_argument("--w-safe", type=float, default=0.05)
    ap.add_argument("--w-lpips", type=float, default=0.1)
    ap.add_argument("--no-attn", action="store_true")
    ap.add_argument("--no-phys", action="store_true")
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="표본 N 개만(동작 확인)")
    ap.add_argument("--epoch-size", type=int, default=600, help="한 epoch = 학습 표본 중 무작위 N 개 (2,800 개 전체는 1시간+)")
    ap.add_argument("--val-limit", type=int, default=70, help="검증 표본 수(고르게 뽑아 고정)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--resume", action="store_true", help="data/correction/<run>.last.pt 에서 이어 학습(옵티마이저·난수·기록 포함)")
    ap.add_argument("--mem-frac", type=float, default=0.85, help="GPU 메모리 상한 — 넘으면 시스템 메모리로 넘쳐 몇 배 느려지는 대신 오류를 내고 그 단계를 건너뛴다")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda")
    base = ROOT / "data" / "hazard7" / a.data
    tr_ids, va_ids = items(base / "train"), items(base / "val")
    if a.limit:
        tr_ids = tr_ids[:a.limit]
    if a.val_limit:
        va_ids = va_ids[::max(1, len(va_ids) // a.val_limit)][:a.val_limit]
    out_dir, chk = ROOT / "data" / "correction", ROOT / "checks" / "correction"
    out_dir.mkdir(parents=True, exist_ok=True); chk.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"{a.run}.pt"
    last = out_dir / f"{a.run}.last.pt"
    if ckpt.exists() and not a.overwrite and not a.limit and not a.resume:
        raise SystemExit(f"{ckpt} 가 이미 있습니다(--overwrite)")
    net = CorrNet(dim=a.dim, depth=a.depth, use_attn=not a.no_attn, use_phys=not a.no_phys).to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    lp = None
    if a.w_lpips > 0:
        import lpips
        lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
        for p in lp.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(1, a.patience // 2))
    print(f"학습 {len(tr_ids)} · 검증 {len(va_ids)} · 매개변수 {n_par / 1e6:.2f}M · 자르기 {a.crop} · 어텐션 {not a.no_attn} · 역산 {not a.no_phys}", flush=True)
    best, best_ep, bad, hist, start = float("inf"), 0, 0, [], 1
    if a.resume and last.exists():                         # 이어 하기: 마지막 epoch 끝 상태 그대로
        st = torch.load(last, map_location=dev, weights_only=False)
        net.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        best, best_ep, bad, hist, start = st["best"], st["best_ep"], st["bad"], st["hist"], st["epoch"] + 1
        rng.bit_generator.state = st["rng"]; torch.set_rng_state(st["torch_rng"].cpu())
        print(f"이어 하기: epoch {start} 부터 (가장 좋은 epoch {best_ep})", flush=True)
    log = (chk / f"{a.run}_log.jsonl").open("w", encoding="utf-8")
    for h in hist:
        log.write(json.dumps(h, ensure_ascii=False) + "\n")
    t0 = time.time() - (hist[-1]["elapsed_s"] if hist else 0)
    skipped = 0
    for ep in range(start, a.max_epochs + 1):
        if bad >= a.patience:
            break
        order = [tr_ids[i] for i in rng.permutation(len(tr_ids))[:a.epoch_size or None]]
        tl, tparts = [], {}
        t_wait = t_step = 0.0
        tw = time.time()
        for i, it in enumerate(loader(base / "train", order), 1):
            t_wait += time.time() - tw
            ts = time.time()
            try:
                b = to_tensors(it, dev, tuple(a.crop), rng)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = net(b["X"])
                loss, parts = compute_loss(o, b, a, lp)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            except torch.OutOfMemoryError:                 # 다른 프로그램이 GPU 를 쓰는 동안 등: 이 단계만 건너뜀
                opt.zero_grad(set_to_none=True); torch.cuda.empty_cache(); skipped += 1
                tw = time.time()
                continue
            torch.cuda.synchronize(); t_step += time.time() - ts; tw = time.time()
            tl.append(float(loss.detach()))
            for k, v in parts.items():
                tparts.setdefault(k, []).append(v)
            if i % 200 == 0:
                print(f"  ep{ep} {i}/{len(order)} 손실 {np.mean(tl[-200:]):.4f} ({time.time() - t0:.0f}초)", flush=True)
        va = validate(net, base / "val", va_ids, dev, a) if va_ids else {"loss": float(np.mean(tl)), "psnr_rec": float("nan")}
        sched.step(va["loss"])
        rec = {"epoch": ep, "train_loss": round(float(np.mean(tl)), 5), "train": {k: round(float(np.mean(v)), 5) for k, v in tparts.items()},
               "val": va, "lr": opt.param_groups[0]["lr"], "elapsed_s": round(time.time() - t0, 1),
               "gpu_peak_GB": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1),
               "sec_data_wait": round(t_wait, 1), "sec_gpu_step": round(t_step, 1), "skipped_oom_total": skipped}
        if va["loss"] < best - a.min_delta:
            best, best_ep, bad = va["loss"], ep, 0
            torch.save({"model": net.state_dict(), "config": vars(a) | {"epoch": ep, "n_params": n_par}}, ckpt)
            rec["best"] = True
        else:
            bad += 1
        hist.append(rec)
        log.write(json.dumps(rec, ensure_ascii=False) + "\n"); log.flush()
        print(f"[{ep}] 학습 {rec['train_loss']:.4f} · 검증 {va['loss']:.4f} · 복원 PSNR {va['psnr_rec']:.2f} dB"
              f"{' ★' if rec.get('best') else f' ({bad}/{a.patience})'} · {rec['elapsed_s']:.0f}초", flush=True)
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep,
                    "best": best, "best_ep": best_ep, "bad": bad, "hist": hist, "rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state()}, last)
        if bad >= a.patience:
            print(f"조기 종료 → epoch {best_ep}", flush=True)
            break
    plot(hist, chk / f"{a.run}_curves.png", best_ep)
    (out_dir / f"{a.run}.done").write_text(json.dumps({"best_epoch": best_ep, "epochs": len(hist)}), encoding="utf-8")
    print("저장:", ckpt, flush=True)


def plot(hist, path, best_ep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(ep, [h["train_loss"] for h in hist], label="train (crop)")
    ax[0].plot(ep, [h["val"]["loss"] for h in hist], label="val (full frame)")
    ax[0].axvline(best_ep, ls="--", c="gray", lw=1); ax[0].set(xlabel="epoch", ylabel="loss"); ax[0].legend()
    ax[1].plot(ep, [h["val"]["psnr_rec"] for h in hist]); ax[1].axvline(best_ep, ls="--", c="gray", lw=1)
    ax[1].set(xlabel="epoch", ylabel="PSNR (dB, recoverable region)", title="val")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
