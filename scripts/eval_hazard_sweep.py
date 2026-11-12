"""생성 스윕(make_hazard_sweep.py) 영상을 우리 판정기와 팀원 판별기로 검사하고 그래프·표를 만든다.

- 우리: lumi_synth.measure.analyze_all (휘도·적색·줄무늬, 흐름 줄무늬 휘도 제외 = 결정 A)
- 팀원: lumi-lab-rockrovks Fullscreen/A1/vdcoreops_v2.py 의 analyze_video_frames 를 **수정 없이** 기본 설정으로 호출
        (laptop 56 cm, 15인치, 긴 변 720, 334 ms). 구간 판정 = evaluate_video_pse_risk 와 같은 trigger_intervals.
- 둘 다 저장된 파일을 다시 읽어서 잰다. 결과는 클립마다 data/hazard_sweep/results/ 에 캐시(중단 후 이어서 실행 가능).
- 출력: checks/hazard_sweep/README.md, results.csv, fig_*.png (수치 그래프만. 영상 프레임 그림은 만들지 않음)

사용
    python scripts\\eval_hazard_sweep.py --only ours
    python scripts\\eval_hazard_sweep.py --only team --team-dir <lumi-lab-rockrovks 경로>
    python scripts\\eval_hazard_sweep.py --only report [--team-dir ...]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "data/hazard_sweep"
RES = SWEEP / "results"
CHECKS = ROOT / "checks/hazard_sweep"
TEAM_TRIG = {"휘도 플래시": "lum", "적색 플래시": "red", "줄무늬 패턴": "stripe", "지속 플래시": "sustained", "고대비": "contrast"}
TYPE_KO = {"lum": "휘도", "red": "적색", "stripe": "줄무늬", "none": "없음"}


def key(it: dict) -> str:
    return f"{it['background']}__{it['id']}"


def run_ours(items: list[dict]) -> None:
    from lumi_synth import measure
    from lumi_synth.data import iter_frames_bgr
    for it in items:
        out = RES / f"{key(it)}.ours.json"
        if out.exists():
            continue
        frames = np.stack(list(iter_frames_bgr(SWEEP / it["file"])))
        t0 = time.time()
        r = measure.analyze_all(frames, 30.0)
        f, red, pat = r["flash"], r["red"], r["pattern"]
        labels = [m for m in pat["motion"] if m]
        res = {
            "lum": bool(f["dangerous"]), "lum_flashes": float(f["max_flashes"]), "lum_area": float(f["peak_area"]),
            "red": bool(red["dangerous"]), "red_flashes": float(red["max_flashes"]), "red_area": float(red["peak_area"]),
            "red_duv": float(red["max_delta_uv"]),
            "stripe": bool(pat["dangerous"]), "stripe_area": float(pat["peak_stripe_area"]),
            "stripe_motion": max(set(labels), key=labels.count) if labels else "",
            "stripe_sec": float(pat["max_stripe_sec"]),
            "lum_frames": f["hazard_frames"].astype(int).tolist(), "red_frames": red["hazard_frames"].astype(int).tolist(),
            "stripe_frames": pat["hazard_frames"].astype(int).tolist(),
            "sec": time.time() - t0,
        }
        out.write_text(json.dumps(res), encoding="utf-8")
        print("ours", key(it), res["lum"], res["red"], res["stripe"], f"{res['sec']:.0f}s", flush=True)


def load_team(team_dir: Path):
    sys.path.insert(0, str(team_dir / "Fullscreen" / "A1"))
    import vdcoreops_v2 as team_mod  # noqa: E402  (팀원 코드, 수정하지 않음)
    commit = subprocess.run(["git", "-C", str(team_dir), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    return team_mod, commit


def run_team(items: list[dict], team_dir: Path) -> None:
    team_mod, commit = load_team(team_dir)
    cache: dict = {}
    for it in items:
        out = RES / f"{key(it)}.team.json"
        if out.exists():
            continue
        t0 = time.time()
        a = team_mod.analyze_video_frames(str(SWEEP / it["file"]), device="laptop", display_area=15, scan_cache=cache)
        h = a["history"]
        res = {"commit": commit, "sec": time.time() - t0}
        for ko, en in TEAM_TRIG.items():
            res[en] = bool(a["trigger_intervals"][ko])
            res[f"{en}_intervals"] = [list(map(int, iv)) for iv in a["trigger_intervals"][ko]]
        for k in ("lum_flash_danger", "red_flash_danger", "stripe_danger", "hc_danger", "sustained_danger"):
            res[k] = [int(bool(x)) for x in h[k]]
        out.write_text(json.dumps(res), encoding="utf-8")
        print("team", key(it), res["lum"], res["red"], res["stripe"], f"{res['sec']:.1f}s", flush=True)


# ---- 보고서 -------------------------------------------------------------------------------------
def detected(r: dict | None, typ: str) -> bool | None:
    if r is None:
        return None
    return bool(r.get(typ, False)) if typ in ("lum", "red", "stripe") else None


def want(it: dict, who: str) -> bool:
    """그 판정기 기준의 설계상 기대. 팀원 = expected(팀원 stage3 기준), 우리 = expected_ours (옛 manifest 는 expected)."""
    return it["expected"] if who == "team" else it.get("expected_ours", it["expected"])


def report(meta: dict, team_commit: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Malgun Gothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    CHECKS.mkdir(parents=True, exist_ok=True)
    items = meta["items"]
    rows = []
    for it in items:
        po, pt = RES / f"{key(it)}.ours.json", RES / f"{key(it)}.team.json"
        o = json.loads(po.read_text(encoding="utf-8")) if po.exists() else None
        t = json.loads(pt.read_text(encoding="utf-8")) if pt.exists() else None
        rows.append((it, o, t))

    # results.csv
    import csv
    with open(CHECKS / "results.csv", "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.writer(fh)
        wr.writerow(["background", "id", "family", "sweep", "value", "expected", "expected_type", "broken",
                     "expected_ours", "broken_ours",
                     "ours_lum", "ours_red", "ours_stripe", "ours_lum_flashes", "ours_lum_area", "ours_red_duv",
                     "ours_red_area", "ours_stripe_area", "ours_stripe_motion", "ours_sec",
                     "team_lum", "team_red", "team_stripe", "team_sustained", "team_contrast", "team_sec"])
        for it, o, t in rows:
            o = o or {}
            t = t or {}
            wr.writerow([it["background"], it["id"], it["family"], it["sweep"], it["value"], it["expected"],
                         it["expected_type"], it["broken"], want(it, "ours"), it.get("broken_ours", it["broken"]),
                         o.get("lum"), o.get("red"), o.get("stripe"), o.get("lum_flashes"), o.get("lum_area"),
                         o.get("red_duv"), o.get("red_area"), o.get("stripe_area"), o.get("stripe_motion"),
                         round(o["sec"], 1) if "sec" in o else None,
                         t.get("lum"), t.get("red"), t.get("stripe"), t.get("sustained"), t.get("contrast"),
                         round(t["sec"], 1) if "sec" in t else None])

    bgs = sorted({it["background"] for it in items}, key=lambda b: (b != "gray", b))
    BG_KO = {"gray": "회색", "bbb80": "BBB"}

    # 일치율: 각 판정기를 '자기 기준의 기대'와 비교 (우리 ↔ expected_ours, 팀원 ↔ expected), 그 종류 판정만
    def agree(fam: str, who: str, bg: str):
        n = k = 0
        for it, o, t in rows:
            if it["family"] != fam or it["background"] != bg:
                continue
            r = o if who == "ours" else t
            d = detected(r, fam)
            if d is None:
                continue
            n += 1
            k += int(d == want(it, who))
        return k, n

    # 그림 1~3: 계열별 스윕 판정 지도
    SW_KO = {"delta_cd": "ΔL (cd/m²)", "area": "면적", "hz": "주파수 (Hz)", "wave": "파형", "dark_cd": "어두운 쪽 (cd/m²)",
             "delta_uv": "Δu′v′", "sat_ratio": "A의 R/(R+G+B)", "cpd": "공간 주파수 (cpd)", "area_static": "면적 (정지)",
             "area_reversing": "면적 (반전 5 Hz)", "contrast_cd": "밝기 차 ΔL (cd/m²)", "angle": "방향 (°)", "drift_dps": "흐름 속도 (°/s)",
             "floor_r8": "순수 빨강의 R 값 (8비트, ↔ 검정)"}
    THR = {("lum", "delta_cd"): 20, ("lum", "area"): 0.25, ("lum", "hz"): 3, ("lum", "dark_cd"): 160,
           ("red", "delta_uv"): 0.2, ("red", "sat_ratio"): 0.8, ("red", "area"): 0.25, ("red", "hz"): 3,
           ("stripe", "cpd"): (1, 10), ("red", "floor_r8"): 71, ("stripe", "area_static"): 0.4, ("stripe", "area_reversing"): 0.25,
           ("stripe", "contrast_cd"): 20}
    for fam in ("lum", "red", "stripe"):
        sweeps = list(dict.fromkeys(it["sweep"] for it, _, _ in rows if it["family"] == fam))
        ncol = 3
        nrow = int(np.ceil(len(sweeps) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.9 * nrow), squeeze=False)
        lanes = [(who, bg) for who in ("ours", "team") for bg in bgs]
        for ax, sw in zip(axes.flat, sweeps):
            pts = [(it, o, t) for it, o, t in rows if it["family"] == fam and it["sweep"] == sw]
            vals = list(dict.fromkeys(it["value"] for it, _, _ in pts))
            cat = isinstance(vals[0], str)
            xs = {v: (i if cat else v) for i, v in enumerate(vals)}
            for li, (who, bg) in enumerate(lanes):
                y = len(lanes) - 1 - li
                for it, o, t in pts:
                    if it["background"] != bg:
                        continue
                    d = detected(o if who == "ours" else t, fam)
                    if d is None:
                        continue
                    ok = d == want(it, who)
                    ax.scatter(xs[it["value"]], y, s=70, marker="o",
                               facecolors=("#B4441C" if d else "white"),
                               edgecolors=("#1C2430" if ok else "#E0A800"), linewidths=(1.2 if ok else 2.6), zorder=3)
            exp_x = [xs[it["value"]] for it, _, _ in pts if it["background"] == bgs[0] and it["expected"]]
            for x in exp_x:
                ax.scatter(x, len(lanes) - 0.35, marker="v", s=28, color="#1F5AA6", zorder=3)
            thr = THR.get((fam, sw))
            if thr is not None and not cat:
                for tv in (thr if isinstance(thr, tuple) else (thr,)):
                    ax.axvline(tv, ls="--", color="#6B7280", lw=1)
            if cat:
                ax.set_xticks(range(len(vals)), vals)
            ax.set_yticks(range(len(lanes)), [f"{'팀원' if w == 'team' else '우리'}·{BG_KO.get(b, b)}"
                                              for w, b in reversed(lanes)])
            ax.set_ylim(-0.6, len(lanes) - 0.1)
            ax.set_title(SW_KO.get(sw, sw), fontsize=11)
            ax.grid(axis="x", color="#E1E6EC")
        for ax in list(axes.flat)[len(sweeps):]:
            ax.axis("off")
        fig.suptitle(f"{TYPE_KO[fam]} 생성 스윕: 채운 점 = '{TYPE_KO[fam]}' 위험 판정, 노란 테두리 = 생성 의도와 불일치, "
                     f"▼ = 의도상 위험, 점선 = 기준값", fontsize=11)
        fig.tight_layout()
        fig.savefig(CHECKS / f"fig_sweep_{fam}.png", dpi=110)
        plt.close(fig)

    # 그림 4: 일치율 막대
    fig, ax = plt.subplots(figsize=(8, 3.6))
    fams = ["lum", "red", "stripe"]
    combos = [(w, b) for w in ("ours", "team") for b in bgs]
    colors = {("ours", "gray"): "#1F5AA6", ("ours", "bbb80"): "#7FA6D6", ("team", "gray"): "#B4441C", ("team", "bbb80"): "#E3A68C"}
    width = 0.8 / len(combos)
    for ci, (w, b) in enumerate(combos):
        vals, labels = [], []
        for fam in fams:
            k, n = agree(fam, w, b)
            vals.append(k / n if n else np.nan)
            labels.append(f"{k}/{n}")
        x = np.arange(len(fams)) + (ci - (len(combos) - 1) / 2) * width
        bars = ax.bar(x, vals, width, color=colors.get((w, b), "#999"), label=f"{'팀원' if w == 'team' else '우리'}·{BG_KO.get(b, b)}")
        for xi, v, lab in zip(x, vals, labels):
            if not np.isnan(v):
                ax.text(xi, v + 0.02, lab, ha="center", fontsize=8)
    ax.set_xticks(range(len(fams)), [TYPE_KO[f] for f in fams])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("생성 의도와 판정 일치율")
    ax.legend(fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHECKS / "fig_agreement.png", dpi=110)
    plt.close(fig)

    # 그림 5: 생성 조건 대 우리 측정값 (생성기 제어 확인)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    specs = [("lum", "area", "lum_area", "설계 면적", "측정: 같은 방향 동시 전이 최대 면적"),
             ("red", "delta_uv", "red_duv", "설계 Δu′v′ (8비트 양자화 후)", "측정: 적색 전이의 최대 Δu′v′"),
             ("stripe", "area_static", "stripe_area", "설계 면적 (정지 줄무늬)", "측정: 최대 줄무늬 면적")]
    for ax, (fam, sw, mk, xl, yl) in zip(axs, specs):
        for b in bgs:
            xs_, ys_ = [], []
            for it, o, _ in rows:
                if it["family"] == fam and it["sweep"] == sw and it["background"] == b and o:
                    xs_.append(it["params"]["pair"]["delta_uv"] if fam == "red" else it["params"]["area_actual"])
                    ys_.append(o[mk])
            ax.plot(xs_, ys_, "o-", label=BG_KO.get(b, b), color="#1F5AA6" if b == "gray" else "#B4441C")
        lim = ax.get_xlim()
        ax.plot(lim, lim, ls=":", color="#6B7280", lw=1)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl, fontsize=9)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHECKS / "fig_measured_vs_design.png", dpi=110)
    plt.close(fig)

    # 그림 6: 대표 클립의 프레임별 판정 (팀원 프레임 플래그 vs 우리 위험 프레임)
    reps = [("lum", "l_area_0.4"), ("red", "r_area_0.4"), ("stripe", "s_area_static_0.5")]
    fig, axs = plt.subplots(len(reps), len(bgs), figsize=(13, 2.2 * len(reps)), squeeze=False)
    for ri, (fam, rid) in enumerate(reps):
        for bi, b in enumerate(bgs):
            ax = axs[ri][bi]
            hit = [(it, o, t) for it, o, t in rows if it["id"] == rid and it["background"] == b]
            if not hit or not hit[0][1] or not hit[0][2]:
                ax.axis("off")
                continue
            it, o, t = hit[0]
            series = [("우리 휘도", o["lum_frames"]), ("우리 적색", o["red_frames"]), ("우리 줄무늬", o["stripe_frames"]),
                      ("팀원 휘도 프레임", t["lum_flash_danger"]), ("팀원 적색 프레임", t["red_flash_danger"]),
                      ("팀원 줄무늬 프레임", t["stripe_danger"]), ("팀원 고대비 프레임", t["hc_danger"])]
            for si, (name, s) in enumerate(series):
                s = np.array(s)
                y = len(series) - 1 - si
                ax.fill_between(np.arange(len(s)), y - 0.35, y + 0.35, where=s > 0, step="mid",
                                color="#1F5AA6" if name.startswith("우리") else "#B4441C", alpha=0.85)
            ax.set_yticks(range(len(series)), [n for n, _ in reversed(series)], fontsize=8)
            ax.set_xlim(0, 59)
            ax.set_title(f"{rid} · {BG_KO.get(b, b)} (의도: {TYPE_KO[fam]} 위험)", fontsize=9)
            if ri == len(reps) - 1:
                ax.set_xlabel("프레임 (30 fps)")
    fig.tight_layout()
    fig.savefig(CHECKS / "fig_timeseries.png", dpi=110)
    plt.close(fig)

    # README
    ppd = meta["ppd"]
    L = ["# 위험 3종 생성 스윕: 우리 판정기 대 팀원 판별기", "",
         f"> 만든 날 {time.strftime('%Y-%m-%d')} · 생성 `scripts/make_hazard_sweep.py` (영상은 Git 제외, `data/hazard_sweep/`) · 평가 `scripts/eval_hazard_sweep.py`",
         f"> 768×432, 30 fps, 60프레임(2초), 1도당 {ppd:.1f} px (56 cm·15인치). 바탕: 균일 회색(sRGB 128) / BBB 1080p 80초~",
         f"> 팀원 판별기: lumi-lab-rockrovks `{team_commit}` `vdcoreops_v2.analyze_video_frames` 수정 없이 기본 설정",
         "> ⚠️ 영상은 광과민성 위험 자극이라 재생하지 않았습니다. 모든 그림은 수치 그래프입니다.", "",
         "> 기대는 둘입니다: 팀원 판별기는 팀원 보고서(stage3) 기준 기대와, 우리 판정기는 우리 기준(ITU-R BT.1702-3 + "
         "docs/STANDARDS.md) 기대와 비교합니다. 두 기대가 갈리는 클립은 표에 '우리 기준' 열로 따로 적었습니다.", "",
         "## 일치율 (각자 기준의 기대 대비, 그 종류의 판정만)", "", "| 종류 | " + " | ".join(f"{'팀원' if w == 'team' else '우리'}·{BG_KO.get(b, b)}" for w, b in combos) + " |",
         "|---|" + "---|" * len(combos)]
    for fam in fams:
        cells = []
        for w, b in combos:
            k, n = agree(fam, w, b)
            cells.append(f"{k}/{n}" if n else "-")
        L.append(f"| {TYPE_KO[fam]} | " + " | ".join(cells) + " |")
    L += ["", "![일치율](fig_agreement.png)", "", "![휘도](fig_sweep_lum.png)", "", "![적색](fig_sweep_red.png)", "",
          "![줄무늬](fig_sweep_stripe.png)", "", "![측정 대 설계](fig_measured_vs_design.png)", "",
          "![프레임별](fig_timeseries.png)", "", "## 클립별 결과", "",
          "| 바탕 | 클립 | 기대 (팀원 기준) | 무너뜨린 조건 | 우리 기준이 다르면 | 우리 휘도/적색/줄무늬 | 팀원 휘도/적색/줄무늬 (+지속/고대비) |",
          "|---|---|---|---|---|---|---|"]
    yn = lambda v: "-" if v is None else ("●" if v else "○")
    for it, o, t in rows:
        o = o or {}
        t = t or {}
        extra = ",".join(n for n, k in (("지속", "sustained"), ("고대비", "contrast")) if t.get(k))
        L.append(f"| {BG_KO.get(it['background'], it['background'])} | `{it['id']}` | "
                 f"{'위험' if it['expected'] else '안전'}({TYPE_KO.get(it['expected_type'], '')}) | {it['broken']} | "
                 f"{'' if want(it, 'ours') == it['expected'] else ('위험' if want(it, 'ours') else '안전: ' + it['broken_ours'])} | "
                 f"{yn(o.get('lum'))}{yn(o.get('red'))}{yn(o.get('stripe'))} | "
                 f"{yn(t.get('lum'))}{yn(t.get('red'))}{yn(t.get('stripe'))} {extra} |")
    L += ["", "● 위험 판정, ○ 안전. 원본 자료: [results.csv](results.csv)", ""]
    (CHECKS / "README.md").write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["ours", "team", "report", "all"], default="all")
    ap.add_argument("--team-dir", type=Path, default=None)
    ap.add_argument("--ids", default=None, help="쉼표로 구분한 클립 id 만 평가·보고 (생략하면 전부)")
    args = ap.parse_args()
    meta = json.loads((SWEEP / "manifest.json").read_text(encoding="utf-8"))
    if args.ids:
        keep = set(args.ids.split(","))
        meta["items"] = [it for it in meta["items"] if it["id"] in keep]
    RES.mkdir(parents=True, exist_ok=True)
    if args.only in ("ours", "all"):
        run_ours(meta["items"])
    if args.only in ("team", "all"):
        run_team(meta["items"], args.team_dir)
    if args.only in ("report", "all"):
        commit = load_team(args.team_dir)[1] if args.team_dir else "?"
        report(meta, commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
