"""기존 판정기(11-17 실행)의 South 2021 실제 영상 결과를 사람 라벨과 비교 — 기준선 기록.

원자료: data/real/eval_rows.jsonl (scripts/eval_real_hazards.py, 11-17 23:23 완료)
  우리 판정기 = lumi_synth.measure (휘도·적색·연장·줄무늬), 팀원 판정기 = rockrovks **Fullscreen/B2 vdcorefs**(당시 판).
  영상은 가로 640 이하로 줄여 원래 fps 로 판정, 60 s 조각(6 s 겹침).
라벨: data/real/south2021/social-media-study.csv — 사람 라벨 dangerous / flash / red / pattern (South et al. 2021).

    python scripts/analyze_judge_south.py   → checks/judge/south_baseline_2026-11-17.md, docs/figures/judge/01_south_baseline.png
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTMD = ROOT / "checks" / "judge" / "south_baseline_2026-11-17.md"
OUTFIG = ROOT / "docs" / "figures" / "judge"
TYPES = [("flash", "플래시", ["휘도 플래시", "교집합", "빠른 장면 전환"], ["lum"]),
         ("red", "적색", ["적색 플래시"], ["red"]),
         ("pattern", "패턴", ["정적 공간 패턴"], ["pattern"])]


def key(r):
    return r["file"].replace("\\", "/").split("/")[-1].rsplit(".", 1)[0]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    lab = {r["label"]: r for r in csv.DictReader(open(ROOT / "data/real/south2021/social-media-study.csv", encoding="utf-8", errors="ignore"))}
    rows = [json.loads(l) for l in (ROOT / "data/real/eval_rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    errs = [r for r in rows if "error" in r]
    S = {}
    for r in rows:
        if r["source"] == "south" and "team" in r and "ours" in r and key(r) in lab:
            S[key(r)] = r                                            # 같은 파일은 마지막 실행
    K = {}
    for r in rows:
        if r["source"] == "kaya" and "team" in r:
            K[key(r)] = r
    n = len(S)
    cnt = Counter(lab[k]["dangerous"] for k in S)
    L = ["# 기존 판정기의 South 2021 실제 영상 결과 (사람 라벨 대비) — 기준선", "",
         "> 판정 실행 2026-11-17 (`scripts/eval_real_hazards.py`, 원자료 `data/real/eval_rows.jsonl`) · 분석 `scripts/analyze_judge_south.py`",
         "> 팀원 판정기 = **Fullscreen/B2 `vdcorefs`(당시 판)**, 우리 판정기 = `lumi_synth.measure`. 영상은 가로 ≤ 640, 원래 fps.",
         "> ⚠️ South 라벨은 사람이 본 '위험해 보임'(연구자 라벨)이고 ITU-R BT.1702 정의와 다르다 → 아래는 **사람 판단과의 일치도**이지 정답 대비 성능이 아니다.", "",
         f"판정 표본 {n} (South 2,000 중 위험 {cnt['dangerous']} 전부 + 안전 {cnt['safe']}) · 판정 실패 {len([e for e in errs if e['source']=='south'])} 건 제외", "",
         "## 1. 종류별 일치", "",
         "| 종류 | 사람 라벨 양성 | 판정기 | TP | FP | FN | TN | 재현율 | 정밀도 |", "|---|---|---|---|---|---|---|---|---|"]
    stats = {}
    for typ, name, tl, ol in TYPES:
        pos = sum(lab[k][typ] == typ for k in S)
        for who in ("team", "ours"):
            tp = fp = fn = tn = 0
            for k, r in S.items():
                y = lab[k][typ] == typ
                p = any(l in (r["team"].get("labels") or []) for l in tl) if who == "team" else any(r["ours"].get(o) for o in ol)
                tp += y and p; fp += (not y) and p; fn += y and not p; tn += (not y) and not p
            rec, pre = tp / max(tp + fn, 1), (tp / (tp + fp) if tp + fp else float("nan"))
            stats[(typ, who)] = (tp, fp, fn, tn, rec, pre)
            L.append(f"| {name} | {pos} | {'팀원 B2' if who == 'team' else '우리'} | {tp} | {fp} | {fn} | {tn} | {rec:.2f} | {pre:.2f} |")
    # 위험 여부 전체
    L += ["", "## 2. 위험 여부(아무 라벨)", "", "| 판정기 | 위험 영상 163 중 위험 판정 | 안전 영상 중 위험 판정(오탐) |", "|---|---|---|"]
    for who in ("team", "ours"):
        dpos = [k for k in S if lab[k]["dangerous"] == "dangerous"]
        dneg = [k for k in S if lab[k]["dangerous"] == "safe"]
        f = (lambda r: bool(r["team"].get("labels"))) if who == "team" else (lambda r: bool(r["ours"].get("any")))
        f_flash = (lambda r: bool([l for l in r["team"].get("labels") or [] if l != "정적 공간 패턴"])) if who == "team" else (lambda r: bool(r["ours"].get("any_core")))
        L.append(f"| {'팀원 B2' if who == 'team' else '우리'} | {sum(f(S[k]) for k in dpos)}/{len(dpos)} (패턴 제외 {sum(f_flash(S[k]) for k in dpos)}) | "
                 f"{sum(f(S[k]) for k in dneg)}/{len(dneg)} (패턴 제외 {sum(f_flash(S[k]) for k in dneg)}) |")
    # 플래시 놓침 분석: 우리 판정기의 1초 최대 깜빡임 수 분포
    fl = [S[k]["ours"]["lum_max_flashes"] for k in S if lab[k]["flash"] == "flash"]
    nf = [S[k]["ours"]["lum_max_flashes"] for k in S if lab[k]["flash"] != "flash"]
    fps_f = [S[k]["fps"] for k in S if lab[k]["flash"] == "flash"]
    L += ["", "## 3. 플래시를 왜 놓치나 — 단서", "",
          f"- 사람이 '플래시' 라고 한 {len(fl)} 개에서 우리 판정기의 **1 초 최대 깜빡임 수** 중앙값 {np.median(fl):.1f} (기준: 3 초과), 3 초과는 {sum(v > 3 for v in fl)} 개. 플래시 아님 중앙값 {np.median(nf):.1f}.",
          f"- 그 {len(fl)} 개의 fps 중앙값 {np.median(fps_f):.1f}, 12 fps 이하 {sum(v <= 12 for v in fps_f)} 개 — GIF 는 fps 가 낮아 1 초에 셀 수 있는 전이 수가 적다.",
          f"- 해석(가설): South 의 '플래시' 는 사람이 깜빡인다고 느낀 것(느린 점멸·작은 영역 포함)이고, 기준(1 초 3 회 초과 · 면적 25 % · 20 cd/m²)보다 넓다. 판정기 v2 에서 **후보 모드(넓게)** 와 **규격 모드(ITU 그대로)** 를 나누는 근거.",
          "", "## 4. 패턴 오탐", "",
          f"- 팀원 판정기는 패턴을 사람 라벨 양성 {stats[('pattern','team')][0] + stats[('pattern','team')][2]} 중 {stats[('pattern','team')][0]} 을 잡지만 **오탐 {stats[('pattern','team')][1]}** — 정밀도 {stats[('pattern','team')][5]:.2f}.",
          "- 같은 현상을 합성 데이터에서도 봤다: 합성 전 '안전' 바탕 2,800 중 1,020(36 %)이 정적 공간 패턴(`checks/hazard7/big1_train.md`).",
          "- 우리 판정기의 줄무늬 판정은 반대로 거의 안 냄(재현율 0.01) — 두 판정기의 패턴 규칙이 정반대로 치우침.",
          "", "## 5. Kaya 2025 (모두 위험 영상)", "",
          f"- 팀원 B2 위험 판정 {sum(bool(r['team'].get('labels')) for r in K.values())}/{len(K)} · 우리 {sum(bool(r['ours'].get('any')) for r in K.values() if 'ours' in r)}/{len(K)}",
          "", "## 6. 이 기준선이 말하는 것", "",
          "1. 실제 영상에서 **플래시 재현율 0.06–0.07, 적색 0.00** — 사람이 위험하다고 본 깜빡임 대부분을 규칙 판정기가 놓친다(정의 차이 포함).",
          "2. **패턴은 팀원 판정기가 과하게(정밀도 0.11), 우리 판정기는 거의 안(재현율 0.01)** 낸다.",
          "3. → 보정·게이트·에이전트 모두 판정기 위에 서므로 **판정기 v2 가 먼저**. 최신 판(Partial B2)으로 South 2,000 전체를 다시 재는 것이 다음 단계."]
    OUTMD.parent.mkdir(parents=True, exist_ok=True)
    OUTMD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    fig_baseline(stats)


def fig_baseline(stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.family": ["Malgun Gothic", "DejaVu Sans"], "axes.unicode_minus": False, "figure.facecolor": SURF,
                         "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlecolor": INK,
                         "legend.frameon": False})
    OUTFIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    names = [n for _, n, _, _ in TYPES]
    x = np.arange(len(names))
    for a, idx, title in [(ax[0], 4, "재현율 — 사람이 위험하다고 본 것 중 잡은 비율"), (ax[1], 5, "정밀도 — 위험하다고 한 것 중 맞은 비율")]:
        for j, (who, lab_, col) in enumerate([("team", "팀원 B2 (Fullscreen, 11-17)", "#eb6834"), ("ours", "우리 판정기", "#2a78d6")]):
            raw = [stats[(t, who)][idx] for t, _, _, _ in TYPES]
            v = [np.nan_to_num(q) for q in raw]
            a.bar(x + (j - 0.5) * 0.36, v, width=0.34, color=col, label=lab_)
            for xi, vi, q in zip(x, v, raw):
                a.text(xi + (j - 0.5) * 0.36, vi + 0.02, "판정 0건" if np.isnan(q) else f"{vi:.2f}", ha="center", fontsize=8.5 if np.isnan(q) else 9, color=INK)
        a.set_xticks(x); a.set_xticklabels(names); a.set_ylim(0, 1.05); a.set_title(title, fontsize=11)
    ax[0].legend(loc="upper left")
    fig.suptitle("기존 규칙 판정기 vs 사람 라벨 — South 2021 실제 영상 1,055", color=INK, fontsize=12)
    fig.tight_layout(); fig.savefig(OUTFIG / "01_south_baseline.png", dpi=160); plt.close(fig)
    print("그림:", OUTFIG / "01_south_baseline.png")


if __name__ == "__main__":
    main()
