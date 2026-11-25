"""판정 '논리' 단계 그림 — 팀원 판(v1)과 G 가 같은 프레임에서 무엇을 어떻게 셌는지 단계별로.

팀원 판 기본 경로(enhanced_spatial_pattern=False, spatial.py StaticPatternDetector.inspect_frame 710–805):
  1 대비 지도: 픽셀마다 반경 pool_radius(≈ 0.55°) 정사각 창의 최대−최소 ≥ 20 cd/m² (어두운 < 160) → fine_area
  2 고정 직선 108 개(12 방향 × 9 줄, 0.5 px 간격) — 선마다 히스테리시스 극값 → 밝기 차·1–10 cpd 를 만족하는 '연속 구간'의 쌍
  3 모든 선(+ 고리·바큇살 주사, FFT 대체)의 쌍 수 최댓값 하나
  4 최댓값 ≥ 6 → 패턴, 면적 = fine_area (쌍이 나온 위치와 무관)
G (grid_pattern.py): 후보 칸 → 이웃 연결 영역 → 영역마다 씨앗 ≤ 12 → 법선 경로 단면 → 같은 쌍 규칙 + 이웃 주기 규칙성 → 면적 = 영역 줄무늬 픽셀 × 확인 비율

    python scripts/make_logic_figs.py [out_dir]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from lumi_judge.cpu_limit import limit_cpu  # noqa: E402
limit_cpu()
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import lumi_judge as J  # noqa: E402
from lumi_judge import grid_pattern as GP  # noqa: E402
from lumi_judge.config import PSEConfig, ViewingConfig  # noqa: E402
from lumi_judge.gpu_kernel import hyst_runs  # noqa: E402
from lumi_judge.grid_flash import luminance_cd  # noqa: E402
from lumi_judge.photometry import PhotometricConverter  # noqa: E402
from lumi_judge.spatial import extract_hysteresis_extrema_1d  # noqa: E402
from lumi_judge.spatial_mask import PartialStaticPatternDetector  # noqa: E402
from make_before_after import g_inspect, trace_pts  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "before_after"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": ["Malgun Gothic", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 11})
C_V1, C_G, C_OK, C_BAD = "#C25E00", "#2F6FD6", "#1B9E5A", "#C0392B"
CLIP = "train_red_flash_near_140.clean.mp4"
NUM = {}


def v1_best_line(det, lum, pse):
    """팀원 판 선형 주사를 그대로 재현: 108 선 중 쌍이 가장 많은 선과 그 연속 구간 위치."""
    w, h = det.geom.width, det.geom.height
    prof = torch.nn.functional.grid_sample(lum[None, None], det._linear_grid, mode="bilinear", padding_mode="border", align_corners=True)[0, 0].cpu().numpy()
    grid = det._linear_grid[0].cpu().numpy()                                   # [lines, L, 2] 정규화 좌표
    best = None
    for i in range(prof.shape[0]):
        v = det._linear_valid[i]
        if v.sum() < 12 or np.ptp(prof[i]) < pse.pattern_hysteresis_cd_m2:
            continue
        vals, th = prof[i, v], det._linear_theta[i, v]
        et, ev = extract_hysteresis_extrema_1d(vals, th, pse.pattern_hysteresis_cd_m2)
        if len(ev) < 2:
            continue
        lo, hi = np.minimum(ev[:-1], ev[1:]), np.maximum(ev[:-1], ev[1:])
        ok = np.where(lo < pse.dark_lum_limit_cd_m2, (hi - lo) >= pse.delta_lum_min_cd_m2, (hi - lo) / np.maximum(hi + lo, 1e-6) > pse.bright_michelson_min)
        cpd = np.where(np.diff(et) > 1e-6, 0.5 / np.maximum(np.diff(et), 1e-6), 0)
        valid = ok & (cpd >= pse.pattern_cpd_min) & (cpd <= pse.pattern_cpd_max)
        pad = np.concatenate(([False], valid, [False])); e = np.flatnonzero(pad[1:] != pad[:-1])
        if not len(e):
            continue
        lens = e[1::2] - e[0::2]; b = int(np.argmax(lens)); pairs = (int(lens[b]) + 1) // 2
        if best is None or pairs > best["pairs"]:
            xy = grid[i][v]
            best = {"line": i, "pairs": pairs, "vals": vals, "th": th, "et": et, "ev": ev, "rs": int(e[2 * b]), "re": int(e[2 * b + 1]),
                    "x": (xy[:, 0] + 1) / 2 * (w - 1), "y": (xy[:, 1] + 1) / 2 * (h - 1)}
    return best


def half_period_cv(et):
    d = np.diff(et)
    return float(np.std(d) / max(np.mean(d), 1e-9)) if len(d) > 1 else float("nan")


def all_region_pairs(G, lum):
    """G 의 모든 영역에서 씨앗 경로마다 '규칙성 검사 전' 쌍 수 — G 가 왜 탈락시켰는지 분해용."""
    pse = GP._sdr(PSEConfig())
    k, amp, th, cp = G.bank.respond(lum); cand = (2 * amp >= GP.CAND_FRAC * 20.0)
    frac, thc, fc, coh = GP._cells(cand, th, cp, G.rows, G.cols); lab, n = GP._regions(frac >= GP.CELL_MIN_FRAC, thc, fc, coh)
    ch, cw = G.h / G.rows, G.w / G.cols; cn = cand.cpu().numpy(); an = amp.cpu().numpy()
    pairs, biggest = [], (0, 0.0, 0)
    for rid in range(n):
        cells = np.argwhere(lab == rid); reg = np.zeros((G.h, G.w), bool)
        for r, c in cells:
            reg[int(r * ch):int((r + 1) * ch), int(c * cw):int((c + 1) * cw)] = True
        core = an[reg & cn]
        if core.size == 0:
            continue
        a90 = np.percentile(core, 90); rd = reg.copy()
        for r, c in cells:
            rd[max(0, int((r - 1) * ch)):int((r + 2) * ch), max(0, int((c - 1) * cw)):int((c + 2) * cw)] = True
        pat = rd & cn & (an >= GP.HALF_AMP * a90)
        seeds = []
        for ci in np.linspace(0, len(cells) - 1, min(12, len(cells))).round().astype(int):
            r, c = cells[ci]; y0, y1, x0, x1 = int(r * ch), int((r + 1) * ch), int(c * cw), int((c + 1) * cw)
            b = np.where(pat[y0:y1, x0:x1], an[y0:y1, x0:x1], -1)
            if b.max() > 0:
                yy, xx = np.unravel_index(int(np.argmax(b)), b.shape); seeds.append((y0 + yy, x0 + xx))
        pr = GP._trace_profiles(lum, th, pat, seeds, G.ppd)
        if not pr:
            continue
        L = max(len(q[0]) for q in pr); V = np.zeros((len(pr), L)); T = np.zeros_like(V); ln = np.array([len(q[0]) for q in pr])
        for i, (v, t) in enumerate(pr):
            V[i, :len(v)] = v; T[i, :len(t)] = t
        P = hyst_runs(V, T, ln, pse)[0].get()
        pairs += [int(q) for q in P]
        if len(cells) > biggest[0]:
            biggest = (len(cells), float(pat.mean()), int((P >= GP.MIN_PAIRS).sum()), len(P))
    return np.array(pairs), biggest


def main():
    from make_hazard7_dataset import decode
    x = decode(ROOT / "data/hazard7/big1/train" / CLIP)
    fr = x[len(x) // 2]
    # ---------------- v1
    vc = ViewingConfig()
    s = vc.analysis_max_width / fr.shape[1]
    fr1 = cv2.resize(fr, (vc.analysis_max_width, int(round(fr.shape[0] * s))), interpolation=cv2.INTER_AREA)
    h1, w1 = fr1.shape[:2]
    geom = vc.resolve_geometry(w1, h1); pse = PSEConfig()
    det = PartialStaticPatternDetector(geometry=geom, pse=pse, device="cuda", mask_config=J.PartialMaskConfig())
    lum1 = PhotometricConverter(l_max_cd_m2=200.0, pse=pse, device="cuda").convert_bgr(fr1)[1]
    cm = det._compute_contrast_mask(lum1[None, None]).squeeze().bool().cpu().numpy()
    isp, area, pairs, cpd = det.inspect_frame(lum1)
    bl = v1_best_line(det, lum1, pse)
    rgb1 = cv2.cvtColor(fr1, cv2.COLOR_BGR2RGB)
    run_t = bl["et"][bl["rs"]:bl["re"] + 1]
    reg_v1 = GP._best_run_regularity(np.asarray(bl["et"], float), np.asarray(bl["ev"], float), len(bl["et"]), GP._sdr(pse))
    NUM["v1"] = {"pool_radius_px": det.pool_radius, "px_per_deg": geom.px_per_deg_x, "fine_area": float(cm.mean()), "is_pat": bool(isp),
                 "pairs_total": int(pairs), "best_line_pairs": bl["pairs"], "run_halfperiod_cv": half_period_cv(run_t), "G_regularity_on_v1_run": reg_v1}

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.82])
    # 1 대비 지도
    a = fig.add_subplot(gs[0, 0]); a.imshow(rgb1)
    ov = np.zeros((*cm.shape, 4)); ov[cm] = (*matplotlib.colors.to_rgb(C_V1), 0.5); a.imshow(ov)
    cy, cx = int(h1 * 0.35), int(w1 * 0.2); r = det.pool_radius
    a.add_patch(Rectangle((cx - r, cy - r), 2 * r, 2 * r, fill=False, ec="white", lw=2.5))
    a.annotate(f"창 {2 * r}×{2 * r} px (≈1.1°)\n안 최대−최소 ≥ 20 cd/m²\n→ 가운데 픽셀 = '대비'", (cx + r, cy), (cx + 3 * r, cy - 2 * r), color="white", fontsize=10,
               arrowprops=dict(arrowstyle="->", color="white"), bbox=dict(fc=(0, 0, 0, 0.55), ec="none"))
    a.set_title(f"① 대비 지도 — 주황 {100 * cm.mean():.0f}% = 판정에 쓰는 '면적'", fontsize=12); a.axis("off")
    # 2 고정 직선 108
    b = fig.add_subplot(gs[0, 1]); b.imshow(rgb1)
    grid = det._linear_grid[0].cpu().numpy()
    for i in range(grid.shape[0]):
        v = det._linear_valid[i]; xy = grid[i][v]
        b.plot((xy[:, 0] + 1) / 2 * (w1 - 1), (xy[:, 1] + 1) / 2 * (h1 - 1), color="white", lw=0.5, alpha=0.45)
    b.plot(bl["x"], bl["y"], color=C_V1, lw=2)
    # 셈에 쓴 연속 구간을 굵게
    th = bl["th"]; i0 = int(np.searchsorted(th, run_t[0])); i1 = int(np.searchsorted(th, run_t[-1]))
    b.plot(bl["x"][i0:i1], bl["y"][i0:i1], color="#FFE14D", lw=5)
    b.set_xlim(0, w1); b.set_ylim(h1, 0)
    b.set_title(f"② 고정 직선 108개(흰색) — 내용과 무관하게 같은 자리\n주황 = 쌍이 가장 많은 선, 노랑 = 센 구간 ({bl['pairs']}쌍)", fontsize=12); b.axis("off")
    # 3 결정
    c = fig.add_subplot(gs[0, 2]); c.axis("off")
    txt = ("③ 판정 규칙\n\n"
           f"쌍  = 108개 선(+고리·바큇살·FFT) 중\n       최댓값 하나 → {int(pairs)}쌍 ≥ 6\n\n"
           f"면적 = ① 대비 지도 {100 * cm.mean():.0f}% > 40%\n\n→ '정지 패턴' 위험\n\n"
           "✗ 쌍이 나온 곳(노랑)과 면적을 잰 곳(주황)을\n   잇는 검사가 없다\n"
           "✗ 줄 간격이 규칙적인지 보지 않는다\n   (1–10 cpd 안이면 어떤 간격이든 통과)\n"
           "✗ 선이 고정 — 줄무늬를 비스듬히 자르면\n   간격이 늘어나고, 휜 줄무늬는 따라가지 못함")
    c.text(0.0, 1.0, txt, va="top", fontsize=12.5, linespacing=1.45, bbox=dict(fc="#FFF4E6", ec=C_V1, boxstyle="round,pad=0.8"))
    # 4 단면
    d = fig.add_subplot(gs[1, :])
    d.plot(bl["th"], bl["vals"], color="#555", lw=1.2, label="주황 선을 따라 읽은 밝기 (cd/m²)")
    d.axvspan(run_t[0], run_t[-1], color="#FFE14D", alpha=0.35, label=f"센 연속 구간: {bl['pairs']}쌍")
    d.plot(bl["et"], bl["ev"], "o", color=C_V1, ms=5, label="히스테리시스 극값 (±20 cd/m²)")
    d.set_xlabel("선을 따른 거리 (시각도 °)"); d.set_ylabel("cd/m²")
    d.set_title(f"④ 그 선의 밝기 단면 — 나뭇잎과 하늘이 번갈아 {bl['pairs']}쌍이 '연속'으로 잡힘. 간격은 제각각(반주기 변동계수 {half_period_cv(run_t):.2f}, 규칙적 줄무늬 ≈ 0)이지만 이 검사는 없다", fontsize=12)
    d.legend(fontsize=10, loc="lower right")
    fig.suptitle("팀원 판(Partial B2 기본 경로)은 이렇게 판정한다 — 같은 Big Buck Bunny 프레임", fontsize=15)
    fig.tight_layout(); fig.savefig(OUT / "logic_v1.png", dpi=120); plt.close(fig)

    # ---------------- G
    lum = luminance_cd(torch.from_numpy(fr[None]).cuda())[0]
    G = GP.GridPatternJudge(fr.shape[0], fr.shape[1])
    g = g_inspect(G, lum)
    rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB); H, W = rgb.shape[:2]
    # 영역 번호(시각화용)
    k, amp, thp, cp = G.bank.respond(lum)
    cand = (2.0 * amp >= GP.CAND_FRAC * 20.0)
    frac, th_c, f_c, coh = GP._cells(cand, thp, cp, G.rows, G.cols)
    lab, n = GP._regions(frac >= GP.CELL_MIN_FRAC, th_c, f_c, coh)
    sizes = sorted(((int(np.sum(lab == q)), q) for q in range(n)), reverse=True)
    # 한 경로의 단면(실패 사례)과 규격 줄무늬 경로의 단면(성공 사례)
    pse_u = GP._sdr(PSEConfig())
    fail_prof = None
    for pts, ok in zip(g["all_paths"], g["all_ok"]):
        if len(pts) > 200 and not ok:
            fail_prof = np.array(pts); break
    import pattern_conformance as PC
    Ls, _ = PC.stripes(kind="concentric", area=0.6)
    u8 = PC.to_u8(Ls); lum_s = luminance_cd(torch.from_numpy(u8[None]).cuda())[0]
    Gs = GP.GridPatternJudge(PC.H, PC.W); gs_ = g_inspect(Gs, lum_s)
    ok_prof = np.array(next(p for p, ok in zip(gs_["all_paths"], gs_["all_ok"]) if ok and len(p) > 200))

    def profile(lumt, pts, Gx):
        Ln = lumt.cpu().numpy(); ys, xs = pts[:, 0], pts[:, 1]
        v = Ln[np.clip(ys.round().astype(int), 0, Ln.shape[0] - 1), np.clip(xs.round().astype(int), 0, Ln.shape[1] - 1)]
        seg = np.hypot(np.diff(ys), np.diff(xs)); t = np.concatenate([[0], np.cumsum(seg / Gx.ppd)])
        P, C, _, NE, ET, EV = hyst_runs(v[None].astype(np.float64), t[None], np.array([len(v)]), pse_u)
        ne = int(NE.get()[0]); et, ev = ET.get()[0][:ne], EV.get()[0][:ne]
        reg = GP._best_run_regularity(ET.get()[0], EV.get()[0], ne, pse_u)
        return t, v, et, ev, int(P.get()[0]), reg
    fig = plt.figure(figsize=(18, 10))
    gs2 = fig.add_gridspec(2, 3, height_ratios=[1, 0.82])
    a = fig.add_subplot(gs2[0, 0]); a.imshow(rgb)
    ch, cw = H / G.rows, W / G.cols
    cols = [C_G, "#7B4FD6", "#00A3A3"]
    for j, (_, q) in enumerate(sizes[:3]):
        for (r, c) in np.argwhere(lab == q):
            a.add_patch(Rectangle((c * cw, r * ch), cw, ch, fill=True, fc=(*matplotlib.colors.to_rgb(cols[j]), 0.35), ec="none"))
    for (r, c) in g["on_cells"]:
        a.add_patch(Rectangle((c * cw, r * ch), cw, ch, fill=False, ec="#FFD27A", lw=0.5))
    a.set_title(f"① 후보 칸 {len(g['on_cells'])}개(노란 테두리) → 이웃끼리 방향·주파수가\n비슷하면 같은 영역 (색 = 큰 영역 3개)", fontsize=12); a.axis("off")
    b = fig.add_subplot(gs2[0, 1]); b.imshow(rgb)
    for pts, ok in zip(g["all_paths"], g["all_ok"]):
        if len(pts):
            p = np.array(pts); b.plot(p[:, 1], p[:, 0], color=(C_OK if ok else C_BAD), lw=2)
    for (y, xx), ok in zip(g["all_seeds"], g["all_ok"]):
        b.plot(xx, y, "o", color="white", ms=6, mec="black")
    if fail_prof is not None:
        b.plot(fail_prof[:, 1], fail_prof[:, 0], color="#FFE14D", lw=4)
    b.set_xlim(0, W); b.set_ylim(H, 0)
    b.set_title(f"② 영역마다 씨앗 ≤ 12개를 칸에 고르게(흰 점, 합 {len(g['all_seeds'])}개)\n씨앗에서 줄무늬 법선을 따라 경로 — 영역 밖으로 나가면 멈춤", fontsize=12); b.axis("off")
    c = fig.add_subplot(gs2[0, 2]); c.axis("off")
    txt = ("③ 판정 규칙\n\n"
           "경로 하나 = 단면 하나: 팀원 판과 같은 쌍 규칙\n(±20 cd/m², 어두운 < 160, 연속 구간)\n"
           "+ 규칙성: 이웃 주기 상대 변화 중앙값 ≤ 0.25\n\n"
           f"확인 = 6쌍 이상 & 규칙적 → {int(sum(g['all_ok']))}/{len(g['all_ok'])}\n\n"
           "면적 = 그 영역의 줄무늬 픽셀 × 확인 비율\n"
           "  가장 큰 후보 영역(411칸) 원면적 42.7%\n"
           "  — 후보만 썼다면 40% 초과로 위험\n"
           f"  × 확인 0/12 = {100 * g['area']:.1f}% → 안전\n\n"
           "✓ 쌍을 센 곳과 면적을 잰 곳이 같은 영역\n✓ 경로가 줄무늬 법선을 따라 휘고 영역에서 멈춘다\n✓ 간격이 제각각이면 탈락")
    c.text(0.0, 1.0, txt, va="top", fontsize=12.5, linespacing=1.45, bbox=dict(fc="#EAF1FC", ec=C_G, boxstyle="round,pad=0.8"))
    rp, big = all_region_pairs(G, lum)
    h0 = fig.add_subplot(gs2[1, 0])
    h0.hist(rp, bins=np.arange(-0.5, max(10, rp.max() + 1.5)), color=C_G, rwidth=0.85, label=f"G 경로 {len(rp)}개 (모든 영역)")
    h0.axvline(5.5, color=C_BAD, ls="--", lw=1.5); h0.text(5.7, h0.get_ylim()[1] * 0.9, "6쌍 문턱", color=C_BAD, fontsize=10)
    h0.axvline(bl["pairs"], color=C_V1, lw=3); h0.text(bl["pairs"] + 0.15, h0.get_ylim()[1] * 0.6, f"팀원 판\n고정 직선\n{bl['pairs']}쌍", color=C_V1, fontsize=10)
    h0.set_xlabel("경로 하나의 쌍 수 (규칙성 검사 전)"); h0.set_ylabel("경로 수")
    h0.set_title(f"④ 결정적 차이 1 — 줄무늬 법선 경로에서는\n최대 {rp.max()}쌍: 6쌍에 이른 경로 0/{len(rp)}", fontsize=12)
    d = fig.add_subplot(gs2[1, 1])
    seg = (bl["th"] >= run_t[0] - 0.2) & (bl["th"] <= run_t[-1] + 0.2)
    d.plot(bl["th"][seg], bl["vals"][seg], color="#555", lw=1.1)
    d.plot(run_t, bl["ev"][bl["rs"]:bl["re"] + 1], "o", color=C_V1, ms=5)
    d.set_xlabel("팀원 판 선을 따른 거리 (°)")
    d.set_title(f"⑤ 보조 — 팀원 판 9쌍 구간에 G 규칙성 적용:\n{reg_v1:.2f} > 0.25 (아슬아슬하게 탈락)", fontsize=12)
    NUM["G_rule_on_v1_run"] = reg_v1; NUM["G_all_paths"] = {"n": len(rp), "max_pairs": int(rp.max()), "ge6": int((rp >= 6).sum())}
    NUM["G_biggest_region"] = {"cells": big[0], "raw_area": big[1], "verified": f"{big[2]}/{big[3]}"}
    e = fig.add_subplot(gs2[1, 2])
    t, v, et, ev, P, reg = profile(lum_s, ok_prof, Gs)
    e.plot(t, v, color="#555", lw=1.2); e.plot(et, ev, "o", color=C_OK, ms=4)
    e.set_xlim(0, min(t[-1], 3)); e.set_title(f"비교: 실제 줄무늬(동심원) 경로\n쌍 {P} · 규칙성 {reg:.2f} → ✓", fontsize=12)
    e.set_xlabel("경로를 따른 거리 (°)")
    NUM["G"] = {"on_cells": len(g["on_cells"]), "seeds": len(g["all_seeds"]), "verified": int(sum(g["all_ok"])), "area": g["area"],
                "stripe_path": {"pairs": P, "regularity": reg}}
    fig.suptitle("G 는 이렇게 판정한다 — 같은 Big Buck Bunny 프레임", fontsize=15)
    fig.tight_layout(); fig.savefig(OUT / "logic_g.png", dpi=120); plt.close(fig)
    (OUT / "logic_numbers.json").write_text(json.dumps(NUM, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(NUM, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
