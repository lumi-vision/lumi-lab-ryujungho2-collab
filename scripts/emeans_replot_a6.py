"""A6 α 민감도 그래프를 저장된 CSV 로 다시 그린다(노트북 재실행 없이 글꼴 문제만 고칠 때).

맑은 고딕에 아래첨자 '₀' 와 로그 눈금의 마이너스 기호가 없어 네모로 나오던 문제를 고친 판.
사용: python scripts/emeans_replot_a6.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
OUT, PRES = ROOT / "checks" / "emeans", ROOT.parent / "reports" / "발표_그래프" / "emeans"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
pct = FuncFormatter(lambda v, _: f"{v:+.0f}%" if abs(v) >= 1 or v == 0 else f"{v:+.1f}%")

sens = pd.read_csv(OUT / "A6_alpha.csv")
res = pd.read_csv(OUT / "A_convergence.csv")
auto = {e: res[(res["알고리즘"] == "E-Means") & (res["epoch"] == e)]["상대 MSE(%)"].mean() for e in (3, 25)}
alphas = sorted(sens["α"].unique())
fig, axs = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
for ax, e in zip(axs, (3, 25)):
    for meth, label, c in (("단순 EMA", "단순 EMA (우리 착안)", "C2"), ("E-Means (α₀ 지정)", "E-Means (α0 직접 지정)", "C0")):
        g = sens[(sens["방법"] == meth) & (sens["epoch"] == e)].groupby("α")["상대 MSE(%)"]
        ax.errorbar(g.mean().index, g.mean(), yerr=g.std(), marker="o", capsize=3, color=c, label=label)
    ax.axhline(auto[e], color="C0", ls=":", label="E-Means 자동 α0 (법칙이 고른 값 = 1.0)")
    ax.axhline(0, ls="--", c="gray", lw=1)
    ax.set_xscale("log"); ax.set_xticks(alphas, [f"{a:g}" for a in alphas]); ax.minorticks_off()
    ax.set_xlabel("α (단순 EMA) 또는 α0 (E-Means 초기값)"); ax.yaxis.set_major_locator(MultipleLocator(2)); ax.yaxis.set_major_formatter(pct)
    ax.grid(alpha=.3); ax.set_title(f"{e} epoch"); ax.legend(fontsize=8)
axs[0].set_ylabel("Lloyd 25 epoch 대비 시험 MSE (아래가 좋음)")
fig.suptitle("A6. α 민감도: 같은 α 를 주면 단순 EMA 와 E-Means 는 어떻게 다른가 (시드 3개 평균 ± 표준편차)")
for d in (OUT, PRES):
    fig.savefig(d / "A6_alpha_sensitivity.png", dpi=200, bbox_inches="tight")
print("저장:", OUT / "A6_alpha_sensitivity.png")
