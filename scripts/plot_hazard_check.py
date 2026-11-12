"""생성 확인 13개(종류별 위험·안전 짝 11개 + 적색 하한 2개)의 판정 결과 그림 2장: 판정 표(우리 대 팀원)와 프레임별 위험 표시.

입력: data/hazard_sweep/manifest.json, data/hazard_sweep/results/gray__<id>.{ours,team}.json (eval_hazard_sweep.py 결과)
출력: checks/hazard_sweep/fig_check_matrix.png, fig_check_frames.png  (수치 그림만, 영상 프레임 없음)

사용: python scripts\\plot_hazard_check.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data/hazard_sweep/results"
OUT = ROOT / "checks/hazard_sweep"
IDS = ["c00_clean", "l_area_0.4", "l_area_0.2", "r_delta_uv_0.27", "r_delta_uv_0.19", "r_sat_ratio_0.78",
       "r_floor_r8_71", "r_floor_r8_70",
       "s_area_static_0.5", "s_area_static_0.36", "s_area_reversing_0.32", "s_area_reversing_0.22", "s_drift_dps_4"]
LABEL = {"c00_clean": "원본 (합성 없음)", "l_area_0.4": "휘도 40%·5 Hz·+50", "l_area_0.2": "휘도 20% (면적 미달)",
         "r_delta_uv_0.27": "적색 Δu′v′ 0.27", "r_delta_uv_0.19": "적색 Δu′v′ 0.19 (미달)",
         "r_sat_ratio_0.78": "적색 ρ 0.78 (포화 아님)", "r_floor_r8_71": "어두운 빨강 R 71 ↔ 검정",
         "r_floor_r8_70": "어두운 빨강 R 70 ↔ 검정 (하한 미달)", "s_area_static_0.5": "줄무늬 정지 50%",
         "s_area_static_0.36": "줄무늬 정지 36% (40% 미달)", "s_area_reversing_0.32": "줄무늬 반전 32% (40% 미달)",
         "s_area_reversing_0.22": "줄무늬 반전 22% (40% 미달)", "s_drift_dps_4": "줄무늬 흐름 50% (제외 없음)"}


def main() -> None:
    plt.rcParams["font.family"] = ["Malgun Gothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    OUT.mkdir(parents=True, exist_ok=True)
    items = {it["id"]: it for it in json.loads((ROOT / "data/hazard_sweep/manifest.json").read_text(encoding="utf-8"))["items"]}
    rows = []
    for i in IDS:
        o = json.loads((RES / f"gray__{i}.ours.json").read_text(encoding="utf-8"))
        t = json.loads((RES / f"gray__{i}.team.json").read_text(encoding="utf-8"))
        rows.append((i, items[i], o, t))

    # 그림 1: 판정 표. 열 = 우리 기준 기대 / 우리 휘도·적색·줄무늬 / 팀원 기준 기대 / 팀원 휘도·적색·줄무늬·고대비
    # 각 판정기는 자기 기준의 기대와 비교한다 (우리 = expected_ours, 팀원 = expected: 팀원 보고서 stage3 기준)
    cols = ["우리 기준\n기대", "우리\n휘도", "우리\n적색", "우리\n줄무늬", "팀원 기준\n기대", "팀원\n휘도", "팀원\n적색",
            "팀원\n줄무늬", "팀원\n고대비"]
    fig, ax = plt.subplots(figsize=(11, 6.4))
    for r, (i, it, o, t) in enumerate(rows):
        exp_o, exp_t = it.get("expected_ours", it["expected"]), it["expected"]
        vals = [exp_o, o["lum"], o["red"], o["stripe"], exp_t, t["lum"], t["red"], t["stripe"], t["contrast"]]
        types = [None, "lum", "red", "stripe", None, "lum", "red", "stripe", "contrast"]
        for c, (v, ty) in enumerate(zip(vals, types)):
            if ty is None:
                txt, fc = ("위험" if v else "안전"), ("#F6D5C7" if v else "#E8EEF6")
            else:
                exp = exp_o if c < 4 else exp_t
                ok = (v == (exp and it["expected_type"] == ty)) if ty != "contrast" else True
                txt = "●" if v else "○"
                fc = "#FFFFFF" if ok else "#FFE8A3"
            ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor=fc, edgecolor="#D0D6DE"))
            ax.text(c + 0.5, r + 0.5, txt, ha="center", va="center", fontsize=11,
                    color="#B4441C" if (ty is not None and v) else "#1C2430")
    ax.set_xlim(0, len(cols))
    ax.set_ylim(len(rows), 0)
    ax.set_xticks(np.arange(len(cols)) + 0.5, cols, fontsize=9)
    ax.set_yticks(np.arange(len(rows)) + 0.5, [LABEL[i] for i, *_ in rows], fontsize=9)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("생성 확인 13개 (회색 바탕, 768×432·30 fps·2초): ● 위험 판정, 노란 칸 = 자기 기준 기대와 다름", fontsize=10, pad=34)
    fig.tight_layout()
    fig.savefig(OUT / "fig_check_matrix.png", dpi=120)
    plt.close(fig)

    # 그림 2: 위험 의도 4개의 프레임별 표시 (우리 위험 프레임 대 팀원 프레임 판정)
    reps = ["l_area_0.4", "r_delta_uv_0.27", "s_area_static_0.5", "s_area_reversing_0.32"]
    fig, axs = plt.subplots(len(reps), 1, figsize=(10, 2.0 * len(reps)), sharex=True)
    for ax, i in zip(axs, reps):
        _, it, o, t = next(r for r in rows if r[0] == i)
        series = [("우리 휘도", o["lum_frames"]), ("우리 적색", o["red_frames"]), ("우리 줄무늬", o["stripe_frames"]),
                  ("팀원 휘도", t["lum_flash_danger"]), ("팀원 적색", t["red_flash_danger"]),
                  ("팀원 줄무늬", t["stripe_danger"]), ("팀원 고대비", t["hc_danger"])]
        for k, (name, s) in enumerate(series):
            s = np.array(s)
            y = len(series) - 1 - k
            ax.fill_between(np.arange(len(s)), y - 0.35, y + 0.35, where=s > 0, step="mid",
                            color="#1F5AA6" if name.startswith("우리") else "#B4441C")
        ax.set_yticks(range(len(series)), [n for n, _ in reversed(series)], fontsize=8)
        ax.set_xlim(0, 59)
        ax.set_title(f"{LABEL[i]} — 우리: 위험 창 안의 프레임 / 팀원: 프레임 단위 판정(구간 판정 전)", fontsize=9)
    axs[-1].set_xlabel("프레임 (30 fps)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_check_frames.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
