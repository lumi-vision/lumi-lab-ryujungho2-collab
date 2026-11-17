"""HazNet(가벼운 위험 분류기) 학습: G 데이터로 학습 → 실제 위험 영상으로 검증·조기 종료 → 실제 영상 test 로 최종 평가.

    python scripts/train_hazcls.py --run a                 # 기본: 학습 data/hazcls/train, 검증 val·val_rule·real(val)
    python scripts/train_hazcls.py --run b --monitor real  # 조기 종료를 실제 영상 검증 손실로만

- 학습: 합성(G) 표본만. 실제 영상은 학습에 넣지 않는다(실제 영상은 검증·시험 전용 — South 위험 163개는 시험에만).
- 증강: 좌우·상하·시간 뒤집기, 회전 ±25°, 화질(축소→확대·흐림·블록 양자화·잡음), 약한 밝기 이득 (lumi_synth.hazcls.augment)
- 조기 종료: 검증 손실(--monitor both = 합성 val 분류 손실 + 실제 val 균형 손실)이 --patience epoch 동안 --min-delta 이상 줄지 않으면 멈추고
  가장 좋았던 epoch 의 매개변수로 되돌린다.
- 실제 영상의 정답은 영상 단위다(창 단위 정답 없음) → 영상 확률 = 창들 중 최대(다중 인스턴스).
결과: data/hazcls/<run>.pt · checks/hazcls/<run>.md · checks/hazcls/<run>_log.jsonl · checks/hazcls/<run>_curves.png · <run>_test_items.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import queue
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lumi_synth import hazcls  # noqa: E402

CACHE = ROOT / "data" / "hazcls"
CHECKS = ROOT / "checks" / "hazcls"


# ---------------- 데이터 ----------------

class Arr:
    """캐시 폴더 하나 (memmap)."""

    def __init__(self, name: str):
        d = CACHE / name
        self.name = name
        self.tem = np.load(d / "tem.npy", mmap_mode="r")
        self.pat = np.load(d / "pat.npy", mmap_mode="r")
        self.y = np.load(d / "y.npy") if (d / "y.npy").exists() else None
        self.tub_t = np.load(d / "tub_t.npy", mmap_mode="r") if (d / "tub_t.npy").exists() else None
        self.tub_p = np.load(d / "tub_p.npy", mmap_mode="r") if (d / "tub_p.npy").exists() else None
        self.rows = [json.loads(x) for x in (d / "rows.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        self.items = ([json.loads(x) for x in (d / "items.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
                      if (d / "items.jsonl").exists() else None)

    def __len__(self):
        return len(self.tem)

    def batch(self, idx):
        idx = np.sort(np.asarray(idx))
        out = {"tem": torch.from_numpy(np.ascontiguousarray(self.tem[idx])),
               "pat": torch.from_numpy(np.ascontiguousarray(self.pat[idx]))}
        if self.y is not None:
            out["y"] = torch.from_numpy(self.y[idx])
        if self.tub_t is not None:
            out["tub_t"] = torch.from_numpy(np.ascontiguousarray(self.tub_t[idx]))
            out["tub_p"] = torch.from_numpy(np.ascontiguousarray(self.tub_p[idx]))
        out["idx"] = idx
        return out


def prefetch(arr: Arr, batches, depth: int = 4):
    """다음 배치들을 스레드에서 미리 읽는다(memmap 읽기는 GIL 을 놓는다)."""
    q: queue.Queue = queue.Queue(depth)

    def work():
        for b in batches:
            q.put(arr.batch(b))
        q.put(None)

    th = threading.Thread(target=work, daemon=True)
    th.start()
    while (b := q.get()) is not None:
        yield b


def to_dev(b: dict, dev) -> dict:
    out = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
    if "tub_t" in out:
        out["tub_t"] = out["tub_t"].float() / 255.0
        out["tub_p"] = out["tub_p"].float() / 255.0
    return out


# ---------------- 지표 ----------------

def auc(score: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y).astype(bool)
    if y.all() or (~y).all():
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    # 동점은 평균 순위
    s = np.asarray(score)[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    npos = y.sum()
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * (~y).sum()))


def bal_bce(p: np.ndarray, y: np.ndarray) -> float:
    """양성·음성 평균을 따로 낸 뒤 평균(실제 영상 검증은 양성 16 · 음성 300 으로 치우침)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    y = np.asarray(y).astype(bool)
    parts = []
    if y.any():
        parts.append(-np.log(p[y]).mean())
    if (~y).any():
        parts.append(-np.log(1 - p[~y]).mean())
    return float(np.mean(parts))


@torch.no_grad()
def predict(net, arr: Arr, dev, bs: int = 8) -> np.ndarray:
    """창(표본)마다 머리 3개의 확률 → N×3."""
    net.eval()
    out = np.zeros((len(arr), 3), np.float32)
    batches = [list(range(i, min(i + bs, len(arr)))) for i in range(0, len(arr), bs)]
    for b in prefetch(arr, batches):
        b = to_dev(b, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            o = net(b["tem"], b["pat"])
        out[b["idx"]] = torch.sigmoid(o["logit"].float()).cpu().numpy()
    return out


def synth_eval(net, arr: Arr, dev) -> dict:
    p = predict(net, arr, dev)
    y = arr.y.astype(np.float32)
    pt = torch.from_numpy(np.clip(p, 1e-6, 1 - 1e-6))
    loss = float(F.binary_cross_entropy(pt, torch.from_numpy(y)))
    res = {"loss": round(loss, 4)}
    for k, h in enumerate(hazcls.HEADS):
        res[h] = {"auc": round(auc(p[:, k], y[:, k]), 4), "acc": round(float(((p[:, k] >= 0.5) == y[:, k]).mean()), 4),
                  "pos": int(y[:, k].sum())}
    anyp, anyy = p.max(1), y.max(1)
    res["any"] = {"auc": round(auc(anyp, anyy), 4), "acc": round(float(((anyp >= 0.5) == anyy).mean()), 4)}
    return res


def item_probs(win_p: np.ndarray, arr: Arr) -> np.ndarray:
    """창 확률 N×3 → 영상 확률 items×3 (창 중 최대). 창이 없는 영상은 nan."""
    out = np.full((len(arr.items), 3), np.nan, np.float32)
    for r, p in zip(arr.rows, win_p):
        k = r["item"]
        out[k] = p if np.isnan(out[k, 0]) else np.maximum(out[k], p)
    return out


def real_eval(net, arr: Arr, dev, split: str, win_p: np.ndarray | None = None) -> tuple[dict, np.ndarray]:
    if win_p is None:
        sel = [i for i, r in enumerate(arr.rows) if arr.items[r["item"]]["split"] == split]
        sub = _Sub(arr, sel)
        wp = predict(net, sub, dev)
        win_p = np.full((len(arr), 3), np.nan, np.float32)
        win_p[sel] = wp
    ip = item_probs(np.nan_to_num(win_p, nan=0.0), arr)
    ks = [k for k, it in enumerate(arr.items) if it["split"] == split and it.get("n_win")]
    anyp = ip[ks].max(1)
    y = np.array([arr.items[k]["any"] for k in ks])
    res = {"n": len(ks), "pos": int(y.sum()), "bal_loss": round(bal_bce(anyp, y), 4), "auc": round(auc(anyp, y), 4),
           "sens@0.5": round(float((anyp[y == 1] >= 0.5).mean()), 4) if y.any() else None,
           "spec@0.5": round(float((anyp[y == 0] < 0.5).mean()), 4) if (y == 0).any() else None}
    return res, ip


class _Sub(Arr):
    """Arr 의 일부 창만 (예측용)."""

    def __init__(self, base: Arr, sel):
        self.base, self.sel = base, np.asarray(sel)
        self.y = None; self.tub_t = None

    def __len__(self):
        return len(self.sel)

    def batch(self, idx):
        b = self.base.batch(self.sel[np.asarray(idx)])
        b["idx"] = np.asarray(idx)
        b.pop("y", None); b.pop("tub_t", None); b.pop("tub_p", None)
        return b


# ---------------- 학습 ----------------

def run(args) -> dict:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    dev = torch.device(args.device)
    CHECKS.mkdir(parents=True, exist_ok=True)
    ckpt = CACHE / f"{args.run}.pt"
    if ckpt.exists() and not args.overwrite:
        raise SystemExit(f"{ckpt} 가 이미 있습니다. 다른 --run 이름이나 --overwrite")
    tr = Arr(args.train)
    vals = {n: Arr(n) for n in args.val if (CACHE / n / "tem.npy").exists()}
    real = Arr(args.real) if (CACHE / args.real / "items.jsonl").exists() else None
    net = hazcls.HazNet(args.width).to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    pos = tr.y.mean(0)
    pos_w = torch.tensor(np.clip((1 - pos) / np.maximum(pos, 1e-3), 1, 5), dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(1, args.patience // 2))
    print(f"학습 {len(tr)} (위험 휘도 {int(tr.y[:, 0].sum())} · 적색 {int(tr.y[:, 1].sum())} · 무늬 {int(tr.y[:, 2].sum())}) · "
          f"검증 {list(vals)} · 실제 {'있음' if real else '없음'} · 매개변수 {n_par / 1e3:.1f}k · {dev}", flush=True)
    log_path = CHECKS / f"{args.run}_log.jsonl"
    best, best_ep, bad, hist = float("inf"), -1, 0, []
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as lf:
        for ep in range(args.max_epochs):
            net.train()
            order = rng.permutation(len(tr))
            batches = [order[i:i + args.batch_size] for i in range(0, len(order) - args.batch_size + 1, args.batch_size)]
            tl, parts_acc = [], []
            for b in prefetch(tr, batches):
                b = to_dev(b, dev)
                tem, pat, tt, tp = hazcls.augment(b["tem"], b["pat"], b["tub_t"], b["tub_p"], g,
                                                  quality=not args.no_quality, max_deg=args.max_deg)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    o = net(tem, pat)
                cls = F.binary_cross_entropy_with_logits(o["logit"].float(), b["y"].float(), pos_weight=pos_w)
                lt = F.binary_cross_entropy_with_logits(o["tmap"].float(), tt)
                lp = F.binary_cross_entropy_with_logits(o["pmap"].float(), tp)
                loss = cls + args.aux * (lt + lp)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                tl.append(float(loss.detach())); parts_acc.append((float(cls.detach()), float((lt + lp).detach())))
            rec = {"epoch": ep + 1, "train_loss": round(float(np.mean(tl)), 4),
                   "train_cls": round(float(np.mean([a for a, _ in parts_acc])), 4),
                   "train_aux": round(float(np.mean([b_ for _, b_ in parts_acc])), 4),
                   "lr": opt.param_groups[0]["lr"], "elapsed_s": round(time.time() - t0, 1)}
            for n, a in vals.items():
                rec[n] = synth_eval(net, a, dev)
            if real:
                rec["real_val"], _ = real_eval(net, real, dev, "val")
            syn = rec.get(args.val[0], {}).get("loss", 0.0)
            rv = rec.get("real_val", {}).get("bal_loss", 0.0)
            mon = {"both": syn + rv, "synth": syn, "real": rv}[args.monitor]
            rec["monitor"] = round(mon, 4)
            sched.step(mon)
            if mon < best - args.min_delta:
                best, best_ep, bad = mon, ep + 1, 0
                torch.save({"model": net.state_dict(), "config": vars(args) | {"epoch": ep + 1, "n_train": len(tr), "n_params": n_par}}, ckpt)
                rec["best"] = True
            else:
                bad += 1
            hist.append(rec)
            lf.write(json.dumps(rec, ensure_ascii=False) + "\n"); lf.flush()
            rv_s = f" · 실제 val 손실 {rv:.3f} AUC {rec['real_val']['auc']}" if real else ""
            print(f"[{ep + 1}] 학습 {rec['train_loss']:.3f} · 합성 val {syn:.3f}{rv_s} · 감시 {mon:.3f}"
                  f"{' ★' if rec.get('best') else f' (나아지지 않음 {bad}/{args.patience})'} · {rec['elapsed_s']:.0f}초", flush=True)
            if bad >= args.patience:
                print(f"조기 종료: {args.patience} epoch 동안 감시 손실이 줄지 않음 → epoch {best_ep} 로 되돌림", flush=True)
                break
    net.load_state_dict(torch.load(ckpt, map_location=dev)["model"])
    result = {"run": args.run, "best_epoch": best_ep, "epochs_run": len(hist), "history": hist}
    if real:
        result["test"] = final_test(net, real, dev, args.run)
    if (CACHE / args.pairs / "items.jsonl").exists():
        result["pairs"] = pair_test(net, Arr(args.pairs), dev)
    plot_curves(hist, args.run, best_ep)
    write_report(args, result, n_par, len(tr))
    return result


def final_test(net, real: Arr, dev, run: str) -> dict:
    sel = [i for i, r in enumerate(real.rows) if real.items[r["item"]]["split"] in ("test", "val")]
    wp = np.full((len(real), 3), np.nan, np.float32)
    wp[sel] = predict(net, _Sub(real, sel), dev)
    ip = item_probs(np.nan_to_num(wp, nan=0.0), real)
    # 문턱: 실제 val 에서 균형 정확도 최대 (test 를 보지 않고 정함)
    vk = [k for k, it in enumerate(real.items) if it["split"] == "val" and it.get("n_win")]
    vy = np.array([real.items[k]["any"] for k in vk]); vp = ip[vk].max(1)
    cands = np.unique(np.concatenate([[0.5], vp]))
    bal = [((vp[vy == 1] >= t).mean() + (vp[vy == 0] < t).mean()) / 2 for t in cands]
    thr = float(cands[int(np.argmax(bal))])
    tk = [k for k, it in enumerate(real.items) if it["split"] == "test" and it.get("n_win")]
    judges = load_judges()
    rows = []
    for k in tk:
        it = real.items[k]
        j = judges.get(it["file"].replace("/", "\\"), {})
        rows.append({**{x: it[x] for x in ("source", "file", "any", "lum", "red", "pattern")},
                     "p": [round(float(v), 4) for v in ip[k]], "p_any": round(float(ip[k].max()), 4),
                     "ours": j.get("ours"), "team": j.get("team")})
    with (CHECKS / f"{run}_test_items.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    y = np.array([r["any"] for r in rows]); p = np.array([r["p_any"] for r in rows])
    out = {"thr_from_val": round(thr, 4), "n": len(rows), "pos": int(y.sum()), "auc": round(auc(p, y), 4)}
    for name, t in (("0.5", 0.5), ("val_thr", thr)):
        out[f"sens@{name}"] = round(float((p[y == 1] >= t).mean()), 4)
        out[f"spec@{name}"] = round(float((p[y == 0] < t).mean()), 4)
    # 종류별 (South 라벨) — 해당 머리의 확률과 '아무 위험'
    sub = {}
    for kind, head in (("flash", 0), ("red", 1), ("pattern", 2)):
        key = {"flash": "lum"}.get(kind, kind)
        rs = [r for r in rows if r["source"] == "south" and r[key] == 1]
        if rs:
            sub[kind] = {"n": len(rs), "any@thr": round(float(np.mean([r["p_any"] >= thr for r in rs])), 4),
                         "head@thr": round(float(np.mean([r["p"][head] >= thr for r in rs])), 4)}
    rs = [r for r in rows if r["source"] == "kaya"]
    sub["kaya"] = {"n": len(rs), "any@thr": round(float(np.mean([r["p_any"] >= thr for r in rs])), 4) if rs else None}
    out["by_kind"] = sub
    # 같은 영상에서 판정기 둘과 비교 (판정 결과가 있는 영상만)
    cmp = {}
    for who in ("ours", "team"):
        rr = [r for r in rows if r[who] is not None]
        if rr:
            yy = np.array([r["any"] for r in rr]); jj = np.array([r[who] for r in rr]); pp = np.array([r["p_any"] >= thr for r in rr])
            cmp[who] = {"n": len(rr), "pos": int(yy.sum()),
                        "judge_sens": round(float(jj[yy == 1].mean()), 4) if yy.any() else None,
                        "judge_spec": round(float((~jj[yy == 0]).mean()), 4) if (yy == 0).any() else None,
                        "model_sens": round(float(pp[yy == 1].mean()), 4) if yy.any() else None,
                        "model_spec": round(float((~pp[yy == 0]).mean()), 4) if (yy == 0).any() else None}
    out["vs_judges"] = cmp
    return out


def pair_test(net, arr: Arr, dev) -> dict:
    """Kaya 원본 쪽(active) 대 Kaya 보정본 쪽(calm), 같은 파일·같은 처리. 보정 뒤 점수가 떨어지면 '위험'을 본다는 근거."""
    ip = item_probs(predict(net, arr, dev), arr)
    by = {}
    for k, it in enumerate(arr.items):
        if it.get("n_win"):
            by.setdefault(it["pair"], {})[it["half"]] = float(np.nanmax(ip[k]))
    pr = [(v["active"], v["calm"]) for v in by.values() if "active" in v and "calm" in v]
    a, c = np.array([x for x, _ in pr]), np.array([y for _, y in pr])
    return {"n": len(pr), "orig_mean": round(float(a.mean()), 4), "corr_mean": round(float(c.mean()), 4),
            "drop_frac": round(float((c < a).mean()), 4), "orig_pos@0.5": int((a >= 0.5).sum()), "corr_pos@0.5": int((c >= 0.5).sum()),
            "pairs": {int(k): {h: round(v, 4) for h, v in d.items()} for k, d in sorted(by.items())}}


def load_judges() -> dict:
    """data/real/eval_rows.jsonl (우리 판정기 · 팀원 판별기) → 파일 → {'ours': bool, 'team': bool}."""
    p = ROOT / "data" / "real" / "eval_rows.jsonl"
    out = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if "error" not in r:
                    out[r["file"]] = {"ours": bool(r["ours"]["any_core"]), "team": bool(r["team"]["dangerous"])}
    return out


def plot_curves(hist: list[dict], run: str, best_ep: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_cls"] for h in hist], label="train (cls, aug)")
    for n in ("val", "val_rule"):
        if n in hist[0]:
            ax[0].plot(ep, [h[n]["loss"] for h in hist], label=f"synth {n}")
    if "real_val" in hist[0]:
        ax[0].plot(ep, [h["real_val"]["bal_loss"] for h in hist], label="real val (balanced)")
    ax[0].axvline(best_ep, ls="--", c="gray", lw=1)
    ax[0].set(xlabel="epoch", ylabel="BCE", title="loss (dashed = best epoch, early stop)")
    ax[0].legend(fontsize=8)
    for n, key in (("val", "any"), ("val_rule", "any")):
        if n in hist[0]:
            ax[1].plot(ep, [h[n][key]["auc"] for h in hist], label=f"synth {n} AUC")
    if "real_val" in hist[0]:
        ax[1].plot(ep, [h["real_val"]["auc"] for h in hist], label="real val AUC")
    ax[1].axvline(best_ep, ls="--", c="gray", lw=1)
    ax[1].set(xlabel="epoch", ylabel="AUC", title="AUC (any hazard)", ylim=(0.4, 1.01))
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CHECKS / f"{run}_curves.png", dpi=130)
    plt.close(fig)


def write_report(args, res: dict, n_par: int, n_train: int) -> None:
    h = res["history"]
    best = next(x for x in h if x["epoch"] == res["best_epoch"])
    L = [f"# HazNet `{args.run}` — G 데이터로 학습, 실제 위험 영상으로 검증·시험", "",
         f"> {time.strftime('%Y-%m-%d %H:%M')} · `scripts/train_hazcls.py` · 매개변수 {n_par / 1e3:.1f}k · 학습 {n_train}개(합성만) · "
         f"감시 `{args.monitor}` · 인내 {args.patience} · 최대 {args.max_epochs} epoch · 증강 회전 ±{args.max_deg}° · 화질 {'끔' if args.no_quality else '켬'} · aux {args.aux}", "",
         f"![curves]({args.run}_curves.png)", "",
         f"- 돌린 epoch {res['epochs_run']} · **가장 좋은 epoch {res['best_epoch']}** (감시 손실 {best['monitor']})", ""]
    L += ["## 검증 (가장 좋은 epoch)", "", "| 검증 세트 | 손실 | 아무 위험 AUC | 휘도 AUC | 적색 AUC | 무늬 AUC |", "|---|---|---|---|---|---|"]
    for n in args.val:
        if n in best:
            v = best[n]
            L.append(f"| 합성 {n} | {v['loss']} | {v['any']['auc']} | {v['lum']['auc']} | {v['red']['auc']} | {v['pattern']['auc']} |")
    if "real_val" in best:
        v = best["real_val"]
        L.append(f"| 실제 val (Kaya {v['pos']} · South 안전 {v['n'] - v['pos']}) | {v['bal_loss']} (균형) | {v['auc']} | – | – | – |")
    if "test" in res:
        t = res["test"]
        L += ["", "## 실제 영상 test (학습·조기 종료에 쓰지 않음)", "",
              f"영상 {t['n']}개 (위험 {t['pos']}: South 163 + Kaya 6 · 안전 {t['n'] - t['pos']}: South). 문턱은 실제 val 에서 정함 = {t['thr_from_val']}", "",
              "| 지표 | 문턱 0.5 | 문턱(val) |", "|---|---|---|",
              f"| 민감도 (위험을 위험으로) | {t['sens@0.5']} | {t['sens@val_thr']} |",
              f"| 특이도 (안전을 안전으로) | {t['spec@0.5']} | {t['spec@val_thr']} |",
              f"| AUC | {t['auc']} | |", "",
              "| 종류 (데이터셋 라벨) | 개수 | 아무 머리라도 위험 | 해당 머리가 위험 |", "|---|---|---|---|"]
        for k, v in t["by_kind"].items():
            L.append(f"| {k} | {v['n']} | {v.get('any@thr')} | {v.get('head@thr', '–')} |")
        if t["vs_judges"]:
            L += ["", "같은 영상에서 판정기와 비교 (판정 결과가 있는 영상만):", "",
                  "| 비교 대상 | 영상 수 (위험) | 판정기 민감도 / 특이도 | HazNet 민감도 / 특이도 |", "|---|---|---|---|"]
            for who, v in t["vs_judges"].items():
                L.append(f"| {'우리 판정기' if who == 'ours' else '팀원 판별기 (B2)'} | {v['n']} ({v['pos']}) | {v['judge_sens']} / {v['judge_spec']} | {v['model_sens']} / {v['model_spec']} |")
    if "pairs" in res:
        p = res["pairs"]
        L += ["", "## Kaya 원본 대 Kaya 보정본 (같은 파일의 두 쪽, 같은 처리 · 정답 없음)", "",
              f"짝 {p['n']}개 · 평균 확률 원본 {p['orig_mean']} → 보정본 {p['corr_mean']} · 보정본 점수가 더 낮은 짝 {p['drop_frac']:.0%} · "
              f"0.5 이상: 원본 {p['orig_pos@0.5']} · 보정본 {p['corr_pos@0.5']}",
              "", "실제 val 양성이 Kaya 뿐이라 모델이 'Kaya 처럼 생긴 영상'을 배웠을 수 있다. 같은 영상의 보정 전·후에서 점수가 떨어지면 생김새가 아니라 깜빡임을 본다는 근거가 된다."]
    L += ["", "## 주의", "",
          "- 실제 영상 정답은 데이터셋 라벨이다(South: 저자 라벨 위험·깜빡임·적색·패턴, Kaya: 'PSE 유발' 데이터셋이라 전부 위험). 우리 판정 기준과 정의가 다르다.",
          "- 실제 영상 검증 양성은 Kaya 16개뿐이다 — 조기 종료가 이 16개에 맞춰질 수 있다. test 는 이와 겹치지 않는다.",
          "- 실제 영상은 2초 창(최대 South 8개 · Kaya 60개)으로 잘라 창 중 최대를 영상 확률로 썼다. 24 fps 로 다시 뽑아 12 Hz 넘는 깜빡임은 겹쳐 보일 수 있다."]
    (CHECKS / f"{args.run}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("보고서:", CHECKS / f"{args.run}.md", flush=True)


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="a")
    ap.add_argument("--train", default="train")
    ap.add_argument("--val", nargs="+", default=["val", "val_rule"], help="합성 검증 캐시(첫 번째가 감시용)")
    ap.add_argument("--real", default="real")
    ap.add_argument("--pairs", default="real_pair", help="Kaya 보정 전·후 짝 캐시 (hazcls_cache.py real --pairs)")
    ap.add_argument("--monitor", default="both", choices=("both", "synth", "real"))
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--min-delta", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--aux", type=float, default=1.0, help="지도(튜브) 보조 손실 가중치, 0 이면 클립 라벨만")
    ap.add_argument("--max-deg", type=float, default=25.0)
    ap.add_argument("--no-quality", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args(argv)


if __name__ == "__main__":
    run(parse())
