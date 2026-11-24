"""로지스틱 회귀 실험 보고서용 그림·표 (docs/PREREG_FFT_RULES_ML_2026-11-24.md 의 실험을 그대로 다시 계산해 그림으로).

출력 (OUT = 첫 인자, 기본 data/report_lr)
  fig1_roc_flash.png        M0–M5 ROC (flash)
  fig2_delta_auc.png        가설별 ΔAUC 와 95 % CI (flash · dangerous)
  fig3_local_vs_mean.png    화면 평균 FFT 진폭 vs 격자 FFT 진폭 — 라벨별 분포(왜 격자가 나은가)
  fig4_ablation.png         M4 에서 특징 묶음을 하나씩 뺐을 때 AUC (탐색적)
  fig5_threshold.png        M4 OOF 확률 분포와 Youden 문턱, 혼동표
  oof_predictions.csv       영상별 OOF 확률(M0–M5)·라벨·주요 특징 (집에서 하나씩 확인용)
  features.csv              영상별 전체 특징
  numbers.json              보고서에 쓴 모든 숫자
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "report_lr"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": ["Malgun Gothic", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
# 범주 색(고정 순서, Okabe–Ito 계열 — 색각 이상에서도 구별)
COL = {"M0": "#7a7a7a", "M1": "#e69f00", "M2": "#0072b2", "M3": "#009e73", "M4": "#d55e00", "M5": "#cc79a7"}
NAME = {"M0": "M0 규칙 판정 그대로", "M1": "M1 화면 평균 FFT (원문)", "M2": "M2 격자 FFT", "M3": "M3 규칙 여유값",
        "M4": "M4 = M3 + M2", "M5": "M5 = M4 + CorrNet"}

R = [json.loads(l) for l in (ROOT / "data/fftml/feats.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
n_err = sum("error" in r for r in R)
R = [r for r in R if "error" not in r]
probe = {}
for l in (ROOT / "data/probe/feats.jsonl").read_text(encoding="utf-8").splitlines():
    if l.strip():
        q = json.loads(l)
        if "A" in q:
            probe[q["id"]] = q["A"]
R = [r for r in R if r["id"] in probe]
names = sorted(R[0]["f"])
sets = {"M1": [n for n in names if n.startswith("m1_")], "M2": [n for n in names if n.startswith("m2_")],
        "M3": [n for n in names if n.startswith("m3_")]}
sets["M4"] = sets["M3"] + sets["M2"]
F = {n: np.array([r["f"][n] for r in R], float) for n in names}
A = np.array([probe[r["id"]] for r in R], float)


def X_of(cols, extra=None):
    X = np.stack([F[n] for n in cols], 1)
    return X if extra is None else np.hstack([X, extra])


def oof(X, y):
    P = np.zeros(len(y))
    for s in range(10):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
        P += cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=s), method="predict_proba")[:, 1]
    return P / 10


rng = np.random.default_rng(0)


def boot(y, a, b, n=2000):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    d = np.empty(n)
    for i in range(n):
        ix = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        d[i] = roc_auc_score(y[ix], a[ix]) - roc_auc_score(y[ix], b[ix])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((np.sum(d <= 0) + 1) / (n + 1))


num = {"n": len(R), "n_decode_error": n_err, "features": {k: v for k, v in sets.items()}, "labels": {}}
P_all = {}
for lab, pos in (("flash", "flash"), ("dangerous", "dangerous")):
    y = np.array([r["y"][lab] == pos for r in R], int)
    P = {"M0": F["m0_score"], "M1": oof(X_of(sets["M1"]), y), "M2": oof(X_of(sets["M2"]), y), "M3": oof(X_of(sets["M3"]), y),
         "M4": oof(X_of(sets["M4"]), y), "M5": oof(X_of(sets["M4"], A), y)}
    P_all[lab] = (y, P)
    auc = {m: float(roc_auc_score(y, p)) for m, p in P.items()}
    tests = {}
    for h, a, b in (("H1", "M2", "M1"), ("H2a", "M4", "M3"), ("H2b", "M4", "M0"), ("H3", "M5", "M4")):
        lo, hi, p = boot(y, P[a], P[b])
        tests[h] = {"a": a, "b": b, "delta": auc[a] - auc[b], "ci": [lo, hi], "p": p}
    order = sorted(tests, key=lambda h: tests[h]["p"]); run = 0.0
    for k, h in enumerate(order):
        run = max(run, min(1.0, (len(order) - k) * tests[h]["p"])); tests[h]["holm"] = run
    num["labels"][lab] = {"pos": int(y.sum()), "auc": auc, "tests": tests}

# ---- 그림 1: ROC (flash)
y, P = P_all["flash"]
fig, ax = plt.subplots(figsize=(5.6, 5))
for m in ("M0", "M1", "M2", "M3", "M4", "M5"):
    fpr, tpr, _ = roc_curve(y, P[m])
    ax.plot(fpr, tpr, color=COL[m], lw=2 if m in ("M4", "M1", "M0") else 1.4, label=f"{NAME[m]}  AUC {num['labels']['flash']['auc'][m]:.3f}")
ax.plot([0, 1], [0, 1], color="#bbb", lw=1, ls=":")
ax.set_xlabel("거짓 양성 비율 (라벨 '깜빡임 없음' 중 위험이라 한 비율)"); ax.set_ylabel("참 양성 비율 (라벨 '깜빡임' 중 잡은 비율)")
ax.set_title(f"그림 1. flash 라벨 ROC (South {len(y)} · 양성 {y.sum()}, 5겹×10회 OOF)", fontsize=10)
ax.legend(fontsize=8, loc="lower right"); fig.tight_layout(); fig.savefig(OUT / "fig1_roc_flash.png", dpi=150); plt.close(fig)

# ---- 그림 2: ΔAUC 숲 그림
fig, ax = plt.subplots(figsize=(7, 3.6))
labels_h = {"H1": "H1 격자 FFT − 평균 FFT", "H2a": "H2a (규칙 여유+FFT) − 규칙 여유", "H2b": "H2b 학습 가중 − 규칙 그대로", "H3": "H3 +CorrNet − M4"}
ys = []
for i, h in enumerate(("H1", "H2a", "H2b", "H3")):
    for j, (lab, mk, c) in enumerate((("flash", "o", "#0072b2"), ("dangerous", "s", "#7a7a7a"))):
        t = num["labels"][lab]["tests"][h]; yy = i + (0.15 if j == 0 else -0.15)
        ax.errorbar(t["delta"], yy, xerr=[[t["delta"] - t["ci"][0]], [t["ci"][1] - t["delta"]]], fmt=mk, color=c, ms=6, capsize=3,
                    label=("flash (주)" if j == 0 else "dangerous (보조·탐색)") if i == 0 else None)
    ys.append(i)
ax.axvline(0, color="#444", lw=1); ax.axvline(0.02, color="#d55e00", lw=1, ls="--", label="사전 기준 Δ = 0.02")
ax.set_yticks(ys); ax.set_yticklabels([labels_h[h] for h in ("H1", "H2a", "H2b", "H3")]); ax.invert_yaxis()
ax.set_xlabel("ΔAUC (점) 과 95 % 부트스트랩 신뢰구간 (막대)"); ax.set_title("그림 2. 가설별 차이 — 막대가 0.02 선 오른쪽 + Holm p < 0.05 이면 '개선'", fontsize=10)
ax.legend(fontsize=8, loc="lower right"); fig.tight_layout(); fig.savefig(OUT / "fig2_delta_auc.png", dpi=150); plt.close(fig)

# ---- 그림 3: 화면 평균 vs 격자 진폭
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, key, ttl in ((axs[0], "m1_haz_amp", "화면 평균 밝기의 ≥3 Hz 진폭 (M1)"), (axs[1], "m2_haz_area_ge5", "격자 칸 중 ≥3 Hz 진폭 ≥ 5 cd/m² 비율 (M2)")):
    v = F[key]
    bins = np.linspace(0, np.percentile(v, 99.5), 40)
    ax.hist(v[y == 0], bins=bins, color="#9a9a9a", alpha=0.8, density=True, label=f"라벨 '깜빡임 없음' ({(y == 0).sum()})")
    ax.hist(v[y == 1], bins=bins, color="#0072b2", alpha=0.75, density=True, label=f"라벨 '깜빡임' ({y.sum()})")
    ax.set_title(ttl, fontsize=9.5); ax.set_ylabel("밀도"); ax.legend(fontsize=8)
    ax.text(0.98, 0.6, f"단일 특징 AUC {roc_auc_score(y, v):.3f}", transform=ax.transAxes, ha="right", fontsize=9)
axs[0].set_xlabel("cd/m²"); axs[1].set_xlabel("칸 비율")
fig.suptitle("그림 3. 국소 깜빡임은 화면 평균에서 묽어진다 — 같은 영상들의 두 특징 분포", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig3_local_vs_mean.png", dpi=150); plt.close(fig)

# ---- 그림 4: 묶음 제거(탐색적)
groups = {"휘도 규칙 여유": [n for n in sets["M3"] if n.startswith("m3_lum")], "적색 규칙 여유": [n for n in sets["M3"] if n.startswith("m3_red")],
          "격자 휘도 FFT ≥3 Hz": [n for n in sets["M2"] if "haz" in n and "red" not in n], "격자 휘도 FFT 1–3 Hz·우세 주파수": [n for n in sets["M2"] if ("low" in n or "dom" in n)],
          "격자 적색 FFT": [n for n in sets["M2"] if "red" in n]}
full = num["labels"]["flash"]["auc"]["M4"]
abl = {}
for g, cols in groups.items():
    keep = [n for n in sets["M4"] if n not in cols]
    abl[g] = float(roc_auc_score(y, oof(X_of(keep), y)))
num["ablation_flash"] = {"full": full, "drop": abl}
fig, ax = plt.subplots(figsize=(6.4, 3.2))
gs = sorted(abl, key=lambda g: abl[g])
ax.barh(range(len(gs)), [full - abl[g] for g in gs], color="#0072b2", height=0.55)
ax.set_yticks(range(len(gs))); ax.set_yticklabels([f"{g} 빼면 → {abl[g]:.3f}" for g in gs], fontsize=8.5)
ax.set_xlabel(f"AUC 감소 (전체 M4 = {full:.3f})"); ax.invert_yaxis()
ax.set_title("그림 4. 특징 묶음을 하나씩 뺐을 때 (탐색적 — 사전 등록 밖)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig4_ablation.png", dpi=150); plt.close(fig)

# ---- 그림 5: 문턱·혼동표
p4 = P["M4"]
fpr, tpr, th = roc_curve(y, p4); j = int(np.argmax(tpr - fpr)); thr = float(th[j])
pred = (p4 >= thr).astype(int)
cm = {"TP": int(((pred == 1) & (y == 1)).sum()), "FN": int(((pred == 0) & (y == 1)).sum()), "FP": int(((pred == 1) & (y == 0)).sum()), "TN": int(((pred == 0) & (y == 0)).sum())}
num["threshold_M4_flash"] = {"youden_threshold": thr, **cm}
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1.6, 1]})
bins = np.linspace(0, 1, 41)
axs[0].hist(p4[y == 0], bins=bins, color="#9a9a9a", alpha=0.85, label="라벨 '깜빡임 없음'"); axs[0].hist(p4[y == 1], bins=bins, color="#0072b2", alpha=0.8, label="라벨 '깜빡임'")
axs[0].axvline(thr, color="#d55e00", ls="--", label=f"Youden 문턱 {thr:.2f}"); axs[0].set_yscale("log")
axs[0].set_xlabel("M4 OOF 확률 (학습에 안 쓴 겹에서 낸 값)"); axs[0].set_ylabel("영상 수 (로그)"); axs[0].legend(fontsize=8)
axs[1].axis("off")
axs[1].table(cellText=[[cm["TP"], cm["FN"]], [cm["FP"], cm["TN"]]], rowLabels=["라벨 깜빡임", "라벨 없음"], colLabels=["M4 위험", "M4 안전"],
             loc="center", cellLoc="center").scale(1, 1.8)
axs[1].set_title("혼동표 (문턱에서)", fontsize=9.5)
fig.suptitle("그림 5. M4 확률 분포와 판정 문턱 — 문턱도 CV 결과로 고른 것이라 낙관적일 수 있음", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig5_threshold.png", dpi=150); plt.close(fig)

# ---- 전체 데이터 계수(해석 참고)
clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)).fit(X_of(sets["M4"]), y)
num["coef_M4_flash"] = {n: float(c) for n, c in zip(sets["M4"], clf[-1].coef_[0])}
num["intercept_M4_flash"] = float(clf[-1].intercept_[0])
num["scaler"] = {n: [float(m), float(s)] for n, m, s in zip(sets["M4"], clf[0].mean_, clf[0].scale_)}

# ---- 영상별 표
with (OUT / "oof_predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    yd, Pd = P_all["dangerous"]
    w.writerow(["id", "label_flash", "label_dangerous", "label_red", "label_pattern"] + [f"flash_{m}" for m in P] + [f"dangerous_{m}" for m in Pd]
               + ["m0_score", "m3_lum_flashes_at_area25", "m2_haz_area_ge5", "m1_haz_amp"])
    for i, r in enumerate(R):
        w.writerow([r["id"], r["y"]["flash"], r["y"]["dangerous"], r["y"]["red"], r["y"]["pattern"]] + [round(float(P[m][i]), 4) for m in P]
                   + [round(float(Pd[m][i]), 4) for m in Pd] + [F["m0_score"][i], F["m3_lum_flashes_at_area25"][i], round(F["m2_haz_area_ge5"][i], 4), round(F["m1_haz_amp"][i], 3)])
with (OUT / "features.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["id", "fps", "frames"] + names + ["label_flash", "label_dangerous"])
    for r in R:
        w.writerow([r["id"], round(r["fps"], 3), r["T"]] + [r["f"][n] for n in names] + [r["y"]["flash"], r["y"]["dangerous"]])
(OUT / "numbers.json").write_text(json.dumps(num, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: num[k] for k in ("n", "ablation_flash", "threshold_M4_flash")}, ensure_ascii=False, indent=1))
