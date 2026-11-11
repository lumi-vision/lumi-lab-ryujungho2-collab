"""발표용 GAN 그래프 2장 (checks/gan/u_sweep_SUMMARY.md 의 실측값, 평가 로그에서 다시 읽음).

G1: λ_div 별 조절력(근처 안전 성공률)과 다양성(z 주파수 표준편차), 시드 3개
G2: 500 → 2000 스텝에서 다양성 변화 (λ 0.05 붕괴, λ 0.5 유지)
사용: python scripts/gan_figures.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "data" / "gan" / "logs"
OUT, PRES = ROOT / "checks" / "gan", ROOT.parent / "reports" / "발표_그래프" / "gan"
PRES.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})


def stats(name):
    t = (LOGS / f"eval_{name}.log").read_text(encoding="utf-8")
    j = json.loads(t[t.index("{"):])
    return j["trained"], j["untrained"]


def save(fig, name):
    for d in (OUT, PRES):
        fig.savefig(d / f"{name}.png", dpi=200, bbox_inches="tight")


lams = ["0.05", "0.5", "2.0", "5.0"]
hz, near, haz = [], [], []
for l in lams:
    rs = [stats(f"u_div{l}_s{s}")[0] for s in (0, 1, 2)]
    hz.append([r["within_std_hz"] for r in rs])
    near.append(100 * sum(r["near_ok"][0] for r in rs) / sum(r["near_ok"][1] for r in rs))
    haz.append(100 * sum(r["hazard_ok"][0] for r in rs) / sum(r["hazard_ok"][1] for r in rs))
untrained_hz = stats("u_div0.05_s0")[1]["within_std_hz"]

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
x = np.arange(len(lams))
m = [np.mean(h) for h in hz]
ax[0].bar(x, m, color=["#9AA5B1", "#9AA5B1", "#2F6FB0", "#9AA5B1"])
for k, h in enumerate(hz):
    ax[0].scatter([k] * 3, h, color="#14213D", zorder=3, s=18)
    ax[0].text(k, max(h) + .08, f"{m[k]:.2f}", ha="center", fontsize=12)
ax[0].axhline(untrained_hz, ls="--", c="gray"); ax[0].text(3.45, untrained_hz + .05, f"학습 안 한 G {untrained_hz:.2f}", ha="right", color="gray")
ax[0].set_xticks(x, [f"λ {l}" for l in lams]); ax[0].set_ylabel("z 만 바꿨을 때 주파수 표준편차 (Hz)")
ax[0].set_title("다양성 (높을수록 다양, 점 = 시드)"); ax[0].set_ylim(0, 3.1); ax[0].grid(alpha=.3, axis="y")
ax[1].plot(x, haz, marker="o", label="목표 '위험' → 실제 위험", color="#E07A2E")
ax[1].plot(x, near, marker="s", label="목표 '근처 안전' → 안전·1~3회/초", color="#2F6FB0")
for k in range(len(lams)):
    ax[1].text(k, near[k] - 2.2, f"{near[k]:.0f}%", ha="center", color="#2F6FB0")
ax[1].set_xticks(x, [f"λ {l}" for l in lams]); ax[1].set_ylim(80, 102); ax[1].set_ylabel("성공률 (%)")
ax[1].set_title("조절력 (목표 판정에 맞추는 능력)"); ax[1].legend(loc="lower left"); ax[1].grid(alpha=.3)
fig.suptitle("GAN 다양성 손실 λ_div: λ 2 가 조절력을 지키며 다양성 최고, λ 5 는 조절력이 무너짐 (768×432, 500스텝, 시드 3개)")
save(fig, "G1_lambda_div"); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 4.4))
pairs = [("λ 0.05", "u_div0.05_s0", "u2k_div0.05_s0"), ("λ 0.5", "u_div0.5_s0", "u2k_div0.5_s0")]
w = .35
for k, (lab, a, b) in enumerate(pairs):
    va, vb = stats(a)[0]["within_std_hz"], stats(b)[0]["within_std_hz"]
    ax.bar(k - w / 2, va, w, color="#9AA5B1", label="500스텝" if k == 0 else None)
    ax.bar(k + w / 2, vb, w, color="#C0392B" if vb < .5 else "#2F6FB0", label="2000스텝" if k == 0 else None)
    ax.text(k - w / 2, va + .03, f"{va:.2f}", ha="center"); ax.text(k + w / 2, vb + .03, f"{vb:.2f}", ha="center")
ax.text(-w / 2 + w, .6, "붕괴", color="#C0392B", ha="center", fontsize=14, fontweight="bold")
ax.set_xticks(range(2), [p[0] for p in pairs]); ax.set_ylabel("z 주파수 표준편차 (Hz)"); ax.set_ylim(0, 1.4)
ax.set_title("오래 학습하면: 다양성 손실이 약한 λ 0.05 는 한 리듬으로 붕괴 (시드 0)"); ax.legend(); ax.grid(alpha=.3, axis="y")
save(fig, "G2_long_training_collapse"); plt.close(fig)
print("저장:", PRES)
