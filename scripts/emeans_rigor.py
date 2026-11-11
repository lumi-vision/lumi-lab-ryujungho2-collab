"""E-Means 검증 보강: (1) 본 데이터 양에 따른 학습 곡선(0 epoch = 초기값 포함), (2) 시드 10개 짝비교 통계.

노트북(notebooks/emeans_verify.ipynb)과 같은 데이터·분할·초기값 규칙을 쓴다.
  데이터: data/emeans/patch_feats.npy (LUMI 깜빡임 패치 232만 개, 26차원), 클립 단위 8:2 분할(rng 0), K = 64
  초기값: 시드마다 학습 패치에서 무작위 K 개 — 세 방법이 같은 초기값에서 출발(짝비교)
출력: checks/emeans/RIGOR.md, R1_data_seen.png, R2_paired.png (+ 발표용 사본), data/emeans/rigor_rows.csv
사용: python scripts/emeans_rigor.py [--seeds 10]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from faiss.contrib.e_means import NumpyEmeans
from matplotlib.ticker import FuncFormatter
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data" / "emeans", ROOT / "checks" / "emeans"
PRES = ROOT.parent / "reports" / "발표_그래프" / "emeans"
K, BATCH, ALPHA = 64, 4096, 0.05
BUDGETS = [1, 2, 3, 5, 10, 25]
CURVE = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 2, 3, 5, 10, 25]
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "mathtext.fontset": "dejavusans"})
COL = {"Lloyd k-means": "C1", "단순 EMA (우리 착안)": "C2", "E-Means": "C0"}


def load():
    X = np.load(DATA / "patch_feats.npy").astype("float32")
    clip = np.load(DATA / "patch_meta.npz")["clip"]
    rng = np.random.default_rng(0)
    clips = np.unique(clip)
    test = np.isin(clip, list(set(rng.choice(clips, len(clips) // 5, replace=False))))
    return np.ascontiguousarray(X[~test]), np.ascontiguousarray(X[test])


def mse(C, Xq):
    idx = faiss.IndexFlatL2(C.shape[1]); idx.add(np.ascontiguousarray(C, dtype="float32"))
    return float(idx.search(Xq, 1)[0].mean())


def init(Xtr, seed):
    return Xtr[np.random.RandomState(seed).permutation(len(Xtr))[:K]].copy()


def lloyd(Xtr, epochs, seed):
    if epochs < 1:                                  # Lloyd 는 한 번에 전체를 본다: 1 epoch 미만은 정의되지 않음
        return None
    km = faiss.Kmeans(Xtr.shape[1], K, niter=int(epochs), seed=seed, max_points_per_centroid=10**9)
    km.train(Xtr, init_centroids=init(Xtr, seed)); return km.centroids


def emeans(Xtr, epochs, seed):
    em = NumpyEmeans(Xtr.shape[1], K, nepochs=epochs, seed=seed)
    em.train(Xtr, init_centroids=init(Xtr, seed)); return em.centroids


def simple_ema(Xtr, epochs, seed, alpha=ALPHA):
    C = init(Xtr, seed); r = np.random.RandomState(seed)
    idx = faiss.IndexFlatL2(C.shape[1])
    total = int(round(epochs * len(Xtr)))
    seen = 0
    while seen < total:
        order = r.permutation(len(Xtr))
        for b in range(0, len(Xtr), BATCH):
            if seen >= total:
                break
            xb = Xtr[order[b:min(b + BATCH, b + total - seen)]]
            seen += len(xb)
            idx.reset(); idx.add(C); I = idx.search(xb, 1)[1][:, 0]
            cnt = np.bincount(I, minlength=K); s = np.zeros_like(C); np.add.at(s, I, xb)
            hit = cnt > 0
            C[hit] = (1 - alpha) * C[hit] + alpha * (s[hit] / cnt[hit, None])
    return C


ALGOS = {"Lloyd k-means": lloyd, "단순 EMA (우리 착안)": simple_ema, "E-Means": emeans}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--curve-seeds", type=int, default=3)
    ap.add_argument("--plot-only", action="store_true", help="저장된 data/emeans/rigor_rows.csv 로 그림·표만 다시")
    a = ap.parse_args()
    for d in (OUT, PRES):
        d.mkdir(parents=True, exist_ok=True)
    Xtr, Xte = load()
    print(f"학습 {len(Xtr):,} / 시험 {len(Xte):,}", flush=True)
    rows = []
    if a.plot_only:
        a.seeds = 0
    t0 = time.time()
    for s in range(a.seeds):
        rows.append({"seed": s, "algo": "초기값", "epochs": 0.0, "mse": mse(init(Xtr, s), Xte)})
        budgets = sorted(set(BUDGETS + (CURVE if s < a.curve_seeds else [])))
        for name, fn in ALGOS.items():
            for e in budgets:
                C = fn(Xtr, e, s)
                if C is not None:
                    rows.append({"seed": s, "algo": name, "epochs": float(e), "mse": mse(C, Xte)})
        print(f"시드 {s} 끝 ({time.time() - t0:.0f}초)", flush=True)
    if a.plot_only:
        df = pd.read_csv(DATA / "rigor_rows.csv", encoding="utf-8-sig")
        a.seeds = int(df.seed.nunique())
    else:
        df = pd.DataFrame(rows)
        df.to_csv(DATA / "rigor_rows.csv", index=False, encoding="utf-8-sig")
    ref = df[(df.algo == "Lloyd k-means") & (df.epochs == 25)].set_index("seed")["mse"]
    df["rel"] = 100 * (df["mse"] / df["seed"].map(ref) - 1)      # 같은 시드의 Lloyd 25 epoch 대비

    # ---- R1: 본 데이터 양(epoch) 대비 학습 곡선 (곡선용 시드들)
    cs = df[df.seed < a.curve_seeds]
    fig, ax = plt.subplots(figsize=(9, 5))
    init_rel = cs[cs.algo == "초기값"]["rel"].mean()
    ax.axhline(init_rel, color="#9AA5B1", ls=":", label=f"초기값(무작위 64점) +{init_rel:.0f}%")
    for name in ALGOS:
        g = cs[cs.algo == name].groupby("epochs")["rel"]
        ax.errorbar(g.mean().index, g.mean(), yerr=g.std(), marker="o", ms=4, capsize=2, color=COL[name], label=name)
    ax.axhline(0, ls="--", c="gray", lw=1)
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=5)
    tk = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 25]
    ax.set_xticks(tk, [f"{v:g}" for v in tk]); ax.minorticks_off()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%".replace("−", "-")))
    ax.set_xlabel("본 데이터 양 (epoch, 로그) — 0.1 = 학습 데이터의 10% 를 본 시점")
    ax.set_ylabel("같은 시드 Lloyd 25 epoch 대비 시험 MSE")
    ax.set_title(f"R1. 1 epoch 안에서 이미 수렴한다 (시드 {a.curve_seeds}개 평균 ± 표준편차)")
    ax.grid(alpha=.3, which="both"); ax.legend(fontsize=9)
    for d in (OUT, PRES):
        fig.savefig(d / "R1_data_seen.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- R2: 시드 10개 짝비교
    piv = df.pivot_table(index="seed", columns=["algo", "epochs"], values="mse")
    comps = [("E-Means 1 epoch 대 Lloyd 25 epoch", ("E-Means", 1.0), ("Lloyd k-means", 25.0)),
             ("E-Means 25 대 Lloyd 25", ("E-Means", 25.0), ("Lloyd k-means", 25.0)),
             ("단순 EMA 25 대 Lloyd 25", ("단순 EMA (우리 착안)", 25.0), ("Lloyd k-means", 25.0)),
             ("단순 EMA 25 대 E-Means 25", ("단순 EMA (우리 착안)", 25.0), ("E-Means", 25.0)),
             ("단순 EMA 2 대 Lloyd 25", ("단순 EMA (우리 착안)", 2.0), ("Lloyd k-means", 25.0))]
    res = []
    for label, a_, b_ in comps:
        d = 100 * (piv[a_] / piv[b_] - 1)
        n = len(d); m, sd = d.mean(), d.std(ddof=1)
        half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        w = stats.wilcoxon(d) if n >= 6 else None
        res.append({"비교": label, "n": n, "평균 차이(%)": round(m, 2), "95% CI": f"[{m - half:.2f}, {m + half:.2f}]",
                    "음수인 시드": f"{int((d < 0).sum())}/{n}", "Wilcoxon p": round(float(w.pvalue), 4) if w else None,
                    "_d": d.values, "_lo": m - half, "_hi": m + half, "_m": m})
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, r in enumerate(res):
        ax.scatter(r["_d"], np.full(len(r["_d"]), i) + np.random.default_rng(i).uniform(-.12, .12, len(r["_d"])),
                   color="#9AA5B1", s=18, zorder=2)
        ax.errorbar(r["_m"], i, xerr=[[r["_m"] - r["_lo"]], [r["_hi"] - r["_m"]]], fmt="o", color="#14213D",
                    capsize=5, zorder=3)
        ax.text(max(r["_hi"], r["_d"].max()) + 0.4, i, f"평균 {r['_m']:+.1f}% · p={r['Wilcoxon p']}", va="center", fontsize=10)
    ax.axvline(0, color="gray", ls="--")
    ax.set_yticks(range(len(res)), [r["비교"] for r in res]); ax.invert_yaxis()
    ax.set_xlabel("짝비교 차이 (%) — 0 보다 왼쪽이면 앞쪽 방법이 더 좋음 (점 = 시드, 막대 = 95% 신뢰구간)")
    ax.set_title(f"R2. 같은 초기값 짝비교, 시드 {a.seeds}개")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    lo = min(r["_d"].min() for r in res); hi = max(r["_d"].max() for r in res)
    ax.set_xlim(lo - 1, hi + 6); ax.grid(alpha=.3, axis="x")
    for d in (OUT, PRES):
        fig.savefig(d / "R2_paired.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    tab = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in res])
    upd = {"Lloyd k-means": 1, "E-Means": None, "단순 EMA (우리 착안)": int(np.ceil(len(Xtr) / BATCH))}
    md = ["# E-Means 검증 보강: 학습 곡선과 시드 10개 짝비교 (2026-11-12)", "",
          f"> 데이터·분할·초기값은 노트북과 같음. 학습 {len(Xtr):,} / 시험 {len(Xte):,} 패치, K = {K}, CPU. 스크립트 `scripts/emeans_rigor.py`. "
          "기준 = 같은 시드 Lloyd 25 epoch 의 시험 MSE.", "",
          "## 1 epoch 의 의미가 방법마다 다르다", "",
          f"- Lloyd: 1 epoch = 중심 갱신 1번. 단순 EMA: 1 epoch = 갱신 {upd['단순 EMA (우리 착안)']}번(배치 {BATCH}). "
          "E-Means: 1 epoch = 갱신 약 113번(법칙이 고른 배치 16,384) + 마지막 평균 확정.",
          "- 그래서 E-Means 는 '처음부터 낮은' 것이 아니라 첫 epoch 안에서 이미 수렴한다. R1 의 0.02~1 epoch 구간이 그 과정.", "",
          f"![R1](R1_data_seen.png)", "", "## 짝비교 (같은 초기값, 시드 {})".format(a.seeds), "",
          tab.to_markdown(index=False), "", "![R2](R2_paired.png)", ""]
    (OUT / "RIGOR.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(tab.to_string(index=False))
    print("곡선 요약(평균 rel %):")
    print(cs.groupby(["algo", "epochs"])["rel"].mean().round(2).unstack(0).to_string())


if __name__ == "__main__":
    main()
