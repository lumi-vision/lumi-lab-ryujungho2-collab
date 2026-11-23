"""보정 모델(CorrNet) 1 차 실험 그림 — 발표·기술 보고서용 (docs/figures/correction/*.png).

원자료만 읽는다(학습 기록 checks/correction/*_log.jsonl, 평가 행 eval_*_rows*.jsonl). 숫자를 손으로 넣지 않는다.
색: 범주 순서 고정(파랑 = H, 주황 = B2 팀원 감광, 그다음 aqua·yellow·magenta·green·violet·red), 한 축만, 얇은 선.

    python scripts/make_correction_figures.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CK = ROOT / "checks" / "correction"
OUT = ROOT / "docs" / "figures" / "correction"
OUT.mkdir(parents=True, exist_ok=True)

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4",
     "green": "#008300", "violet": "#6250d6", "red": "#e34948", "gray": "#9a9994"}
plt.rcParams.update({"font.family": ["Malgun Gothic", "DejaVu Sans"], "axes.unicode_minus": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 2, "axes.axisbelow": True})

H60 = ["h_s0_e60", "h_s1_e60", "h_s2_e60"]
ABL = [("abl_noattn_e60", "− 시간 어텐션", C["aqua"]), ("abl_nophys_e60", "− 물리 역산", C["yellow"]),
       ("abl_nolayer_e60", "− 층 손실", C["magenta"]), ("abl_nosafe_e60", "− 안전 손실", C["violet"]),
       ("cnn_e0_e60", "E0 단순 CNN", C["red"])]
BASE = [("B0", "B0 무처리", C["gray"]), ("B1", "B1 전체 시간평균", C["green"]), ("B2", "B2 팀원 감광", C["orange"]),
        ("B3", "B3 국소 시간평균", C["aqua"]), ("B4", "B4 FlashGuard 재구현", C["red"])]


def log(run):
    p = CK / f"{run}_log.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)
    print("그림:", OUT / name)


# ---------- 1. H 학습 곡선 (3 시드) ----------
def fig_training():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    shades = ["#2a78d6", "#6da3e6", "#0d4f9e"]
    for run, col in zip(H60, shades):
        L = log(run)
        ep = [h["epoch"] for h in L]
        ax[0].plot(ep, [h["train_loss"] for h in L], color=col, lw=1.2, alpha=0.55)
        ax[0].plot(ep, [h["val"]["loss"] for h in L], color=col, lw=2, label=run.replace("_e60", ""))
        ax[1].plot(ep, [h["val"]["psnr_rec"] for h in L], color=col, lw=2, label=run.replace("_e60", ""))
    for a in ax:
        a.axvline(30, color=INK2, lw=1, ls="--")
        a.text(30.5, a.get_ylim()[1], " 계획한 30 epoch", va="top", color=INK2, fontsize=9)
        a.set_xlabel("epoch")
    ax[0].set_title("손실 (굵은 선 = 검증, 옅은 선 = 학습)")
    ax[0].set_ylabel("가중 합 손실")
    ax[1].set_title("검증 복원 PSNR (복원 가능 영역)")
    ax[1].set_ylabel("dB")
    ax[0].legend(); ax[1].legend()
    save(fig, "01_training_curves_H.png")


# ---------- 2. 손실 항목별 (h_s0_e60, 학습) ----------
def fig_components():
    L = log("h_s0_e60")
    ep = [h["epoch"] for h in L]
    keys = [("rec", "복원 (Charbonnier)"), ("layer", "층 (A·B L1)"), ("temp", "시간 변화 일치"),
            ("haz", "위험 지도 BCE"), ("safe", "안전 (소프트 판정)"), ("lpips", "LPIPS")]
    fig, ax = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for a, (k, t) in zip(ax.flat, keys):
        a.plot(ep, [h["train"].get(k, np.nan) for h in L], color=C["blue"], lw=1.5, label="학습")
        if k in L[0]["val"]:
            a.plot(ep, [h["val"].get(k, np.nan) for h in L], color=C["orange"], lw=1.5, label="검증")
        a.set_title(t, fontsize=11)
        a.axvline(30, color=INK2, lw=0.8, ls="--")
    ax[0, 0].legend()
    for a in ax[1]:
        a.set_xlabel("epoch")
    fig.suptitle("H(시드 0) 손실 항목별 추이 — 가중치: 복원 1 · 층 1 · 시간 0.5 · 위험 0.2 · 안전 0.05 · LPIPS 0.1", color=INK, fontsize=12)
    save(fig, "02_loss_components_h_s0.png")


# ---------- 3. 같은 학습량 절제 (검증 PSNR) ----------
def fig_ablation_curves():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    Hs = [log(r) for r in H60]
    n = min(len(h) for h in Hs)
    ps = np.array([[h[i]["val"]["psnr_rec"] for i in range(n)] for h in Hs])
    ep = np.arange(1, n + 1)
    ax.fill_between(ep, ps.min(0), ps.max(0), color=C["blue"], alpha=0.15, lw=0)
    ax.plot(ep, ps.mean(0), color=C["blue"], lw=2.4, label="H (3 시드 평균, 띠 = 최소~최대)")
    for run, lab, col in ABL:
        L = log(run)
        ax.plot([h["epoch"] for h in L], [h["val"]["psnr_rec"] for h in L], color=col, lw=1.5, label=lab)
    ax.axvline(30, color=INK2, lw=1, ls="--")
    ax.set_xlabel("epoch"); ax.set_ylabel("검증 복원 PSNR (dB)")
    ax.set_title("같은 학습량(60 epoch)에서 절제 — 차이가 시드 편차 안")
    ax.legend(ncol=2, fontsize=9, loc="lower right")
    save(fig, "03_ablation_val_psnr.png")


def rows(path):
    out = {}
    for q in ([path] if path.exists() else []):
        for l in q.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                if "error" not in r:
                    out[(r.get("id") or r.get("file"), r["method"])] = r
    return out


# ---------- 4. test: 안전 vs 화질 (핵심 그림) ----------
def fig_tradeoff():
    R = rows(CK / "eval_test_all_rows.merged.jsonl")
    if not R:
        return
    ids = sorted({k[0] for k in R})
    haz = [i for i in ids if R[(i, "B0")]["hazard_before"]]
    hz6 = [i for i in haz if set(R[(i, "B0")]["labels_before"] or []) - {"정적 공간 패턴"}]

    def stat(m):
        s7 = np.mean([R[(i, m)]["safe_team"] for i in haz]) * 100
        s6 = np.mean([not (set(R[(i, m)]["left_labels"]) - {"정적 공간 패턴"}) for i in hz6]) * 100
        lp = np.median([R[(i, m)]["lpips_changed"] for i in haz])
        return s7, s6, lp
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)
    pts = [(m, lab, col, 80) for m, lab, col in BASE] + [(m, "H 60 epoch", C["blue"], 150) for m in H60]
    off = {"B0": (8, 4), "B1": (8, -14), "B2": (8, 6), "B3": (8, 6), "B4": (8, 6), "h_s0_e60": (12, -3)}
    for a, which, title in [(ax[0], 0, "7 라벨 전부 (사전 등록 A1 기준)"), (ax[1], 1, "깜빡임 6 라벨 (정적 공간 패턴 제외, 보조)")]:
        for m, lab, col, sz in pts:
            s_ = stat(m)
            a.scatter(s_[which], s_[2], s=sz, color=col, edgecolor=SURF, linewidth=2, zorder=3)
            if m in off:
                a.annotate("H (3 시드)" if m == "h_s0_e60" else lab, (s_[which], s_[2]), xytext=off[m], textcoords="offset points",
                           fontsize=9.5, color=INK, ha="left")
        a.axvline(95, color=INK2, lw=1, ls=":")
        a.text(95, 1.0, "A1 합격선 95 %", transform=a.get_xaxis_transform(), color=INK2, fontsize=9, ha="right", va="bottom")
        a.set_xlim(-5, 115)
        a.set_xlabel("보정 뒤 안전 비율 (%) — 오른쪽일수록 안전")
        a.set_title(title, pad=16)
    ax[0].set_ylabel("위험 영역 LPIPS (중앙값)" + chr(10) + "아래일수록 원본에 가까움")
    ax[0].invert_yaxis()
    fig.suptitle("합성 test 140 (위험 76): 같은 안전에서 덜 망가뜨리는가", color=INK, fontsize=13)
    save(fig, "04_test_safety_vs_fidelity.png")


# ---------- 5. 실패 원인: 종류별 실패 · 남은 라벨 ----------
def fig_failures():
    R = rows(CK / "eval_test_all_rows.merged.jsonl")
    if not R:
        return
    ids = sorted({k[0] for k in R})
    haz = [i for i in ids if R[(i, "B0")]["hazard_before"]]
    kinds = ["lum_flash", "intersection", "scene", "red_flash", "lum_ext", "red_ext", "pattern"]
    names = {"lum_flash": "휘도 플래시", "intersection": "교집합", "scene": "장면 전환", "red_flash": "적색 플래시",
             "lum_ext": "휘도 연장", "red_ext": "적색 연장", "pattern": "정적 공간 패턴"}
    fail, tot, left = Counter(), Counter(), Counter()
    for m in H60:
        for i in haz:
            k = R[(i, m)]["kind"]; tot[k] += 1; fail[k] += not R[(i, m)]["safe_team"]
            for l in R[(i, m)]["left_labels"]:
                left[l] += 1
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    rate = [100 * fail[k] / max(tot[k], 1) for k in kinds]
    cols = [C["red"] if k == "pattern" else C["blue"] for k in kinds]
    ax[0].barh([names[k] for k in kinds], rate, color=cols, height=0.6)
    for y, (k, r) in enumerate(zip(kinds, rate)):
        ax[0].text(r + 1, y, f"{fail[k]}/{tot[k]}", va="center", fontsize=9, color=INK)
    ax[0].invert_yaxis(); ax[0].set_xlabel("보정 뒤에도 위험 (%) — H 3 시드 합")
    ax[0].set_title("합성 종류별 실패")
    labs = left.most_common()
    ax[1].barh([l for l, _ in labs], [c for _, c in labs], color=[C["red"] if l == "정적 공간 패턴" else C["blue"] for l, _ in labs], height=0.5)
    ax[1].invert_yaxis(); ax[1].set_xlabel("건수"); ax[1].set_title(f"보정 뒤 남은 라벨 ({sum(left.values())} 건)")
    save(fig, "05_test_failures_by_kind.png")


# ---------- 6. 30 → 60 epoch 변화 (test) ----------
def fig_epochs():
    R = rows(CK / "eval_test_all_rows.merged.jsonl")
    if not R:
        return
    ids = sorted({k[0] for k in R})
    haz = [i for i in ids if R[(i, "B0")]["hazard_before"]]
    models = [("h_s0", "H s0"), ("h_s1", "H s1"), ("h_s2", "H s2"), ("abl_noattn", "− 어텐션"), ("abl_nophys", "− 역산"),
              ("abl_nolayer", "− 층"), ("abl_nosafe", "− 안전"), ("cnn_e0", "E0")]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for j, (m, lab) in enumerate(models):
        for a, f in [(ax[0], lambda q: 100 * np.mean([R[(i, q)]["safe_team"] for i in haz])),
                     (ax[1], lambda q: np.median([R[(i, q)]["psnr_changed"] for i in haz]))]:
            v30, v60 = f(m), f(m + "_e60")
            a.plot([v30, v60], [j, j], color=GRID, lw=3, zorder=1)
            a.scatter([v30], [j], color=C["gray"], s=50, zorder=2)
            a.scatter([v60], [j], color=C["blue"], s=70, zorder=3)
    for a in ax:
        a.set_yticks(range(len(models))); a.set_yticklabels([l for _, l in models]); a.invert_yaxis()
    ax[0].set_xlabel("안전 비율 (%)"); ax[0].set_title("안전 (7 라벨)")
    ax[1].set_xlabel("바뀐 곳 PSNR (dB)"); ax[1].set_title("화질")
    ax[1].legend(handles=[plt.Line2D([], [], marker="o", ls="", color=C["gray"], label="30 epoch"),
                          plt.Line2D([], [], marker="o", ls="", color=C["blue"], label="60 epoch")], loc="lower right")
    fig.suptitle("학습량이 결론을 바꿨다 — 30 → 60 epoch (합성 test)", color=INK, fontsize=13)
    save(fig, "06_test_30_vs_60_epochs.png")


# ---------- 7. Kaya 실영상 ----------
def fig_kaya():
    p = CK / "eval_kaya_all_rows.merged.jsonl"
    R = rows(p if p.exists() else CK / "eval_kaya_all_rows.jsonl")
    if not R:
        return
    files = sorted({k[0] for k in R})
    meth = [("B0", "B0 무처리", C["gray"]), ("B1", "B1", C["green"]), ("B2", "B2 팀원 감광", C["orange"]), ("B3", "B3", C["aqua"]),
            ("B4", "B4 FlashGuard", C["red"])] + [(m, m.replace("_e60", ""), C["blue"]) for m in H60]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for j, (m, lab, col) in enumerate(meth):
        rs = [R[(f, m)] for f in files if (f, m) in R]
        if not rs:
            continue
        s6 = 100 * np.mean([r["safe_team_flash"] for r in rs])
        ax[0].barh(j, s6, color=col, height=0.6)
        ax[0].text(s6 + 1, j, f"{sum(r['safe_team_flash'] for r in rs)}/{len(rs)}", va="center", fontsize=9)
        ax[1].barh(j, np.median([r["lpips_vs_input"] for r in rs]), color=col, height=0.6)
    for a in ax:
        a.set_yticks(range(len(meth))); a.set_yticklabels([l for _, l, _ in meth]); a.invert_yaxis()
    ax[0].set_xlabel("보정 뒤 깜빡임 6 라벨 안전 (%)"); ax[0].set_title(f"Kaya 실영상 {len(files)} 개 — 안전")
    ax[1].set_xlabel("입력 대비 LPIPS (중앙값) — 작을수록 덜 바꿈"); ax[1].set_title("얼마나 바꿨나")
    save(fig, "07_kaya_real_videos.png")


# ---------- 8. 데이터: 합성 적중률 · 바탕 패턴 오탐 ----------
def fig_data():
    kinds = [("휘도 플래시", 191, 200), ("교집합", 196, 200), ("장면 전환", 200, 200), ("적색 플래시", 153, 200),
             ("휘도 연장", 180, 200), ("적색 연장", 148, 200), ("정적 공간 패턴", 163, 200)]
    # 위 숫자는 checks/hazard7/big1_train.md (생성 보고서)의 hazard 적중 열 그대로
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    ax[0].barh([k for k, _, _ in kinds], [100 * a / b for _, a, b in kinds], color=C["blue"], height=0.6)
    for y, (_, a, b) in enumerate(kinds):
        ax[0].text(100 * a / b + 1, y, f"{a}/{b}", va="center", fontsize=9)
    ax[0].invert_yaxis(); ax[0].set_xlim(0, 112); ax[0].set_xlabel("의도한 라벨이 판정기에서 나온 비율 (%)")
    ax[0].set_title("합성 7 종 적중 (train 2,800)")
    ax[1].bar(["바탕에서\n정적 공간 패턴", "바탕에서\n휘도 플래시"], [1020, 20], color=[C["red"], C["gray"]], width=0.55)
    for x, v in enumerate([1020, 20]):
        ax[1].text(x, v + 20, f"{v}/2800", ha="center", fontsize=10)
    ax[1].set_title("합성 전 '안전' 바탕을 판정했더니")
    ax[1].set_ylabel("클립 수")
    save(fig, "08_data_hits_and_judge_fp.png")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for f in (fig_training, fig_components, fig_ablation_curves, fig_tradeoff, fig_failures, fig_epochs, fig_kaya, fig_data):
        f()
