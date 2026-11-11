"""발표용: Lloyd · 단순 EMA · E-Means 의 갱신식 비교 그림 (수식은 arXiv:2611.09501 원문 식 2·5·6·7, 부록 H.1, 공식 구현 기준).

사용: python scripts/emeans_formula_figure.py → checks/emeans/F1_formulas.png (+ 발표용 사본)
"""
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTS = [ROOT / "checks" / "emeans", ROOT.parent / "reports" / "발표_그래프" / "emeans"]
plt.rcParams.update({"font.family": "Malgun Gothic", "mathtext.fontset": "cm", "axes.unicode_minus": False})

ROWS = [  # (구분, 색, 이름, 수식, 뜻)
    ("기존", "#6B7280", "Lloyd k-means",
     r"$c_j \;=\; \frac{1}{|S_j|}\;\Sigma_{\,x \in S_j}\; x$",
     "매 epoch 전체 데이터를 다시 훑어 평균으로 교체"),
    ("우리 착안", "#2C9A3F", "단순 EMA",
     r"$c_j \;\leftarrow\; (1-\alpha)\,c_j \;+\; \alpha\,\frac{s_j}{n_j}$",
     "배치마다 배치 평균 쪽으로 α 만큼 (α 고정, 사람이 정함)"),
    ("E-Means ①", "#2F6FB0", "질량·합의 EMA",
     r"$N_j \leftarrow \gamma N_j + (1-\gamma)\,n_j,\quad S_j \leftarrow \gamma S_j + (1-\gamma)\,s_j,\quad c_j = S_j / N_j$",
     "배정 수(질량)까지 이동평균 → 자주 쓰이는 군집은 무겁게, 안 쓰이면 가벼워짐"),
    ("E-Means ②", "#2F6FB0", "코사인 감쇠",
     r"$\alpha(t) = \alpha_0\left[\varepsilon + (1-\varepsilon)\,\frac{1}{2}\left(1+\cos\frac{\pi t}{M}\right)\right],\;\; \varepsilon = 0.1$",
     "처음엔 크게 움직이고 끝으로 갈수록 1/10 까지 줄임"),
    ("E-Means ③", "#2F6FB0", "스케일링 법칙",
     r"$\alpha_0 = 0.824\,k^{-0.462} B^{0.900} n^{-0.450} T^{-0.106},\quad B = 0.0351\,k^{0.135} n^{0.876} T^{0.813}$",
     "데이터 수 n · 군집 수 k · epoch T 로 α0 와 배치 B 를 자동 결정"),
    ("E-Means ④", "#2F6FB0", "죽은 군집 복구",
     r"$N_j < 0.1\,\bar{N} \;\Rightarrow\; c_{dead},\, c_{donor} = c_{donor} \odot (1 \mp \epsilon),\;\; N = N_{donor}/2$",
     "5배치마다, 빈 군집을 큰 군집(질량에 비례해 뽑음)을 쪼개 되살림"),
    ("E-Means ⑤", "#2F6FB0", "마무리",
     r"$c_j = \mathrm{mean}\{\, x : x \in S_j \text{ in the last epoch} \}$",
     "마지막 epoch 에 배정된 점들의 평균으로 확정"),
]

fig = plt.figure(figsize=(17, 8.6), dpi=200)
fig.patch.set_facecolor("#FFFFFF")
n = len(ROWS)
top, step = 0.93, 0.128
for i, (tag, col, name, eq, mean) in enumerate(ROWS):
    y = top - i * step
    if i in (1, 2):
        fig.add_artist(plt.Line2D([0.01, 0.99], [y + step / 2, y + step / 2], color="#C9CED6", lw=1.2))
    fig.text(0.012, y + 0.018, tag, fontsize=14, color=col, fontweight="bold", va="center")
    fig.text(0.012, y - 0.022, name, fontsize=17, color="#14213D", fontweight="bold", va="center")
    fig.text(0.175, y, eq, fontsize=22, color="#14213D", va="center")
    fig.text(0.175, y - 0.046, mean, fontsize=14, color="#3D4A5C", va="center")
fig.text(0.012, 0.012, "s_j, n_j = 이번 배치에서 군집 j 에 배정된 점의 합·개수 · α = 1-γ · M = 전체 갱신 횟수 · "
         "우리 데이터(n=186만, k=64, 26차원, T=1)에서 법칙이 고른 값: B = 16,384, α0 = 1.0",
         fontsize=12.5, color="#6B7280")
for d in OUTS:
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / "F1_formulas.png", dpi=200, facecolor=fig.get_facecolor())
print("저장:", OUTS[-1] / "F1_formulas.png")
