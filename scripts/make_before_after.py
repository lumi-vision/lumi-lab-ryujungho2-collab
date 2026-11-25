"""팀원 판(v1) vs 격자 판정기 G — 같은 입력에서 '무엇을 보고 어떻게 판정했나' 전/후 그림.

    python scripts/make_before_after.py [out_dir]   (기본 data/before_after)

그림 (모두 정지 이미지·그래프 — 영상 재생 없음)
  ba1_scene_fp.png      실제 장면(Big Buck Bunny, CC-BY Blender Foundation) — v1 이 패턴이라 한 진단 오탐 클립의 한 프레임
                        v1: 국소 대비 지도(판정 면적 fine_area) · 고정 직선 주사선 · 최종 판정
                        G : 줄무늬 후보 · 영역 칸 · 씨앗(확인 ✓/실패 ✗)과 법선 경로 · 확인 면적 · 판정
  ba2_area39.png        규격 경계 '면적 39 %' 직선 줄무늬 — 두 판정기가 잰 면적
  ba3_drift_vs_osc.png  흐르는 줄무늬 vs 진동 줄무늬 — v1 프레임별 판정 vs G 칸별 위상 속도
  ba4_reverse.png       대비 반전 줄무늬 30 % — 깜빡임 규칙의 방향별 면적(15 %) vs P2 변화 면적
  ba5_scoreboard.png    전/후 점수표
  ba_numbers.json       그림에 쓴 숫자
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
from lumi_judge.grid_flash import judge_luminance, luminance_cd  # noqa: E402
from lumi_judge.grid_motion import AREA_P2, _responses, judge_pattern_motion  # noqa: E402
from lumi_judge.photometry import PhotometricConverter  # noqa: E402
from lumi_judge.spatial_mask import PartialStaticPatternDetector  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "before_after"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": ["Malgun Gothic", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 11})
C_V1, C_G, C_OK, C_BAD = "#C25E00", "#2F6FD6", "#1B9E5A", "#C0392B"
NUM: dict = {}


# ------------------------------------------------------------------ v1 내부
def v1_inspect(frame_bgr):
    """팀원 판 정지 패턴 검출기를 원본 설정(분석 폭 640, laptop)으로 한 프레임에 → 대비 지도, 판정, 마스크, 주사선."""
    vc = ViewingConfig()
    h0, w0 = frame_bgr.shape[:2]
    if w0 > vc.analysis_max_width:
        s = vc.analysis_max_width / w0
        fr = cv2.resize(frame_bgr, (vc.analysis_max_width, max(2, int(round(h0 * s)))), interpolation=cv2.INTER_AREA)
    else:
        fr = frame_bgr
    h, w = fr.shape[:2]
    geom = vc.resolve_geometry(w, h)
    pse = PSEConfig()
    conv = PhotometricConverter(l_max_cd_m2=vc.l_max_cd_m2, pse=pse, device="cuda")
    det = PartialStaticPatternDetector(geometry=geom, pse=pse, device="cuda", mask_config=J.PartialMaskConfig())
    lum = conv.convert_bgr(fr)[1]
    cmask = det._compute_contrast_mask(lum[None, None])[0, 0] if det._compute_contrast_mask(lum[None, None]).dim() == 4 else det._compute_contrast_mask(lum[None, None])
    cmask = cmask.squeeze().bool().cpu().numpy()
    is_pat, area, pairs, cpd, pmask = det.inspect_frame_with_mask(lum)
    lines = []
    cx, cy = det._lin_cx, det._lin_cy
    for (c_p, s_p, _p, offsets, _o) in det._lin_angle_meta:
        nx, ny = -s_p, c_p
        for off in offsets:
            x0, y0 = cx + off * nx, cy + off * ny
            L = math.hypot(w, h)
            lines.append(((x0 - L * c_p) / w, (y0 - L * s_p) / h, (x0 + L * c_p) / w, (y0 + L * s_p) / h))
    return {"contrast": cmask, "fine_area": float(cmask.mean()), "is_pat": bool(is_pat), "area": float(area), "pairs": int(pairs),
            "cpd": float(cpd), "mask": pmask.cpu().numpy().astype(bool), "lines": lines}


# ------------------------------------------------------------------ G 내부 (grid_pattern.frame 과 같은 절차를 펼쳐 중간 결과를 남김)
def trace_pts(lum, th_px, inreg, seed, step=GP.TRACE_STEP, max_len=16000):
    h, w = lum.shape
    TH = th_px.cpu().numpy()
    out = []
    for sgn in (-1.0, 1.0):
        y, x = float(seed[0]), float(seed[1]); pv = None; pts = []
        for it in range(max_len):
            yi, xi = int(round(y)), int(round(x))
            if not (0 <= yi < h and 0 <= xi < w) or not inreg[yi, xi]:
                break
            if it * step > 30 and math.hypot(y - seed[0], x - seed[1]) < 2.0:
                break
            a = math.radians(float(TH[yi, xi])); d = np.array([math.sin(a), math.cos(a)])
            if pv is not None and float(d @ pv) < 0:
                d = -d
            if pv is None:
                d = d * sgn
            pv = d; pts.append((y, x)); y += step * d[0]; x += step * d[1]
        out.append(pts[::-1] if sgn < 0 else pts[1:])
    return out[0] + out[1]


def g_inspect(G, lum, area_min=GP.AREA_MIN, n_seeds=12):
    pse = GP._sdr(PSEConfig())
    k, amp, th, cp = G.bank.respond(lum)
    cand = (2.0 * amp >= GP.CAND_FRAC * 20.0)
    frac, th_c, f_c, coh = GP._cells(cand, th, cp, G.rows, G.cols)
    on = frac >= GP.CELL_MIN_FRAC
    lab, n = GP._regions(on, th_c, f_c, coh)
    cand_np = cand.cpu().numpy(); amp_np = amp.cpu().numpy()
    ch, cw = G.h / G.rows, G.w / G.cols
    best = {"cand": cand_np, "cells": [], "seeds": [], "paths": [], "ok": [], "area_raw": 0.0, "area": 0.0, "verified": 0.0, "pairs": 0, "is_pat": False,
            "on_cells": [(int(r), int(c)) for r, c in np.argwhere(on)], "all_seeds": [], "all_paths": [], "all_ok": []}
    sizes = sorted(((np.sum(lab == r), r) for r in range(n)), reverse=True)
    for _, rid in sizes[:3]:
        cells = np.argwhere(lab == rid)
        reg = np.zeros((G.h, G.w), bool)
        for r, c in cells:
            reg[int(r * ch):int((r + 1) * ch), int(c * cw):int((c + 1) * cw)] = True
        core = amp_np[reg & cand_np]
        if core.size == 0:
            continue
        a_core = float(np.percentile(core, 90))
        reg_d = reg.copy()
        for r, c in cells:
            reg_d[max(0, int((r - 1) * ch)):int((r + 2) * ch), max(0, int((c - 1) * cw)):int((c + 2) * cw)] = True
        pat = reg_d & cand_np & (amp_np >= GP.HALF_AMP * a_core)
        area_raw = float(pat.mean())
        seeds = []
        for ci in np.linspace(0, len(cells) - 1, min(n_seeds, len(cells))).round().astype(int):
            r, c = cells[ci]
            y0, y1, x0, x1 = int(r * ch), int((r + 1) * ch), int(c * cw), int((c + 1) * cw)
            blk = np.where(pat[y0:y1, x0:x1], amp_np[y0:y1, x0:x1], -1)
            if blk.max() > 0:
                yy, xx = np.unravel_index(int(np.argmax(blk)), blk.shape); seeds.append((y0 + yy, x0 + xx))
        profs = GP._trace_profiles(lum, th, pat, seeds, G.ppd)
        paths = [trace_pts(lum, th, pat, s) for s in seeds]
        if not profs:
            continue
        Lm = max(len(p[0]) for p in profs)
        V = np.zeros((len(profs), Lm)); T = np.zeros_like(V); lens = np.array([len(p[0]) for p in profs])
        for i, (v, t) in enumerate(profs):
            V[i, :len(v)] = v; T[i, :len(t)] = t
        P, C, _, NE, ET, EV = hyst_runs(V, T, lens, pse)
        P, NE, ET, EV = P.get(), NE.get(), ET.get(), EV.get()
        reg_ok = np.array([GP._best_run_regularity(ET[i], EV[i], int(NE[i]), pse) <= GP.LOCAL_REG_MAX for i in range(len(P))])
        Pq = np.where(reg_ok, P, 0); ok = Pq >= GP.MIN_PAIRS
        kept = [pp for pp in paths if len(pp) >= 12]
        best["all_seeds"] += seeds[:len(ok)]; best["all_paths"] += kept[:len(ok)]; best["all_ok"] += ok.tolist()
        if area_raw * ok.mean() >= best["area"] or not best["cells"]:
            best.update(cells=[(int(r), int(c)) for r, c in cells], seeds=seeds[:len(ok)], paths=kept[:len(ok)], ok=ok.tolist(),
                        area_raw=area_raw, verified=float(ok.mean()), area=area_raw * float(ok.mean()), pairs=int(Pq.max()) if len(Pq) else 0,
                        is_pat=bool(Pq.max() >= GP.MIN_PAIRS and area_raw * ok.mean() > area_min), pat=pat)
    return best


def rgb(fr):
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)


def overlay(ax, img, mask, color, alpha=0.45):
    ax.imshow(img)
    m = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    ov = np.zeros((*m.shape, 4)); c = matplotlib.colors.to_rgb(color)
    ov[m] = (*c, alpha); ax.imshow(ov)


def verdict(ax, ok_safe, text):
    ax.text(0.02, 0.97, text, transform=ax.transAxes, va="top", ha="left", fontsize=11.5, color="white",
            bbox=dict(boxstyle="round,pad=0.35", fc=(C_OK if ok_safe else C_BAD), ec="none", alpha=0.92))


# ------------------------------------------------------------------ 그림 1: 실제 장면 오탐
def fig_scene():
    from make_hazard7_dataset import decode
    R = [json.loads(l) for l in (ROOT / "data/judge/pattern_diag_rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rec = {}
    import glob
    for f in glob.glob(str(ROOT / "data/hazard7/big1/train/manifest.s*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l); rec[str(r.get("id")).replace(".mp4", "")] = r
    fp = [r["id"] for r in R if r["grp"] == "fp" and r["cfg"] == "default" and rec[r["id"].replace(".clean.mp4", "")]["bg"].get("source") == "movie"]
    cands = []
    for cid in fp:
        x = decode(ROOT / "data/hazard7/big1/train" / cid)
        fr = x[len(x) // 2]
        v1 = v1_inspect(fr)
        if v1["is_pat"]:
            cands.append((v1["fine_area"], cid, fr, v1))
    cands.sort(key=lambda t: -t[0])
    _, cid, fr, v1 = cands[min(2, len(cands) - 1)]                 # 대비 면적이 큰 편이면서 극단이 아닌 것
    lum = luminance_cd(torch.from_numpy(fr[None]).cuda())[0]
    G = GP.GridPatternJudge(fr.shape[0], fr.shape[1])
    g = g_inspect(G, lum)
    gf = G.frame(lum)
    img = rgb(fr); H, W = img.shape[:2]
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.9))
    axs[0].imshow(img); axs[0].set_title("입력 프레임 (Big Buck Bunny 장면, 합성 전 안전 바탕)", fontsize=11.5)
    overlay(axs[1], img, v1["contrast"], C_V1, 0.5)
    for (x0, y0, x1, y1) in v1["lines"][::3]:
        axs[1].plot([x0 * W, x1 * W], [y0 * H, y1 * H], color="white", lw=0.5, alpha=0.5)
    axs[1].set_xlim(0, W); axs[1].set_ylim(H, 0)
    axs[1].set_title("팀원 판(v1): 주황 = 국소 대비 지도 · 흰 선 = 고정 직선 주사(일부)", fontsize=11.5)
    verdict(axs[1], False, f"판정 면적(fine_area) {100 * v1['fine_area']:.0f}% > 40%\n쌍 {v1['pairs']} → '정지 패턴' 위험")
    axs[2].imshow(img)
    ch, cw = H / G.rows, W / G.cols
    for (r, c) in g["on_cells"]:
        axs[2].add_patch(Rectangle((c * cw, r * ch), cw, ch, fill=True, fc=(1, 0.82, 0.3, 0.28), ec="#FFD27A", lw=0.8))
    for pts, ok in zip(g["all_paths"], g["all_ok"]):
        if pts:
            pts = np.array(pts); axs[2].plot(pts[:, 1], pts[:, 0], color=(C_OK if ok else C_BAD), lw=2.2)
    for (y, x), ok in zip(g["all_seeds"], g["all_ok"]):
        axs[2].plot(x, y, marker="o" if ok else "x", color=(C_OK if ok else C_BAD), ms=10, mew=3)
    axs[2].set_xlim(0, W); axs[2].set_ylim(H, 0)
    axs[2].set_title("G: 노란 칸 = 줄무늬 후보 칸 · 선 = 법선 경로 (초록 확인 / 빨강 ✗ 실패)", fontsize=11.5)
    ok_n = int(sum(g["all_ok"]))
    verdict(axs[2], True, f"후보 칸 {len(g['on_cells'])}개 → 경로 {len(g['all_ok'])}개 중 규격 확인 {ok_n}개\n확인 면적 {100 * g['area']:.1f}% ≤ 40% → 안전")
    for a in axs:
        a.axis("off")
    fig.suptitle("① 같은 장면, 다른 판단 — 팀원 판은 '대비가 있는 곳'을, G 는 '규격대로 확인된 줄무늬'를 면적으로 잰다", fontsize=13.5, y=0.99)
    fig.tight_layout(); fig.savefig(OUT / "ba1_scene_fp.png", dpi=130); plt.close(fig)
    NUM["scene"] = {"clip": cid, "v1_fine_area": v1["fine_area"], "v1_pairs": v1["pairs"], "v1_is_pat": v1["is_pat"], "G_area_raw": g["area_raw"],
                    "G_verified": f"{ok_n}/{len(g['all_ok'])}", "G_on_cells": len(g["on_cells"]), "G_area": g["area"], "G_frame_is_pat": gf["is_pat"], "n_movie_fp_flagged": len(cands)}


# ------------------------------------------------------------------ 그림 2: 면적 39 %
def fig_area39():
    import pattern_conformance as PC
    L, ar = PC.stripes(area=0.39)
    u8 = PC.to_u8(L)
    v1 = v1_inspect(u8)
    lum = luminance_cd(torch.from_numpy(u8[None]).cuda())[0]
    G = GP.GridPatternJudge(PC.H, PC.W)
    g = g_inspect(G, lum)
    img = rgb(u8)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
    overlay(axs[0], img, v1["contrast"], C_V1, 0.55)
    verdict(axs[0], False, f"팀원 판: 대비 면적 {100 * v1['fine_area']:.0f}% > 40% → 위험 (틀림)")
    overlay(axs[1], img, g.get("pat", g["cand"]), C_G, 0.5)
    verdict(axs[1], True, f"G: 줄무늬 픽셀 {100 * g['area_raw']:.0f}% × 확인 {int(sum(g['ok']))}/{len(g['ok'])} = {100 * g['area']:.0f}% ≤ 40% → 안전 (맞음)")
    for a in axs:
        a.axis("off")
    fig.suptitle(f"② 규격 경계 시험 '면적 39 %' (실제 줄무늬 {100 * ar:.0f}%, 정답 = 안전) — 국소 대비 지도는 줄무늬 가장자리 바깥까지 번진다", fontsize=12.5)
    fig.tight_layout(); fig.savefig(OUT / "ba2_area39.png", dpi=130); plt.close(fig)
    NUM["area39"] = {"real": ar, "v1_fine_area": v1["fine_area"], "v1_is_pat": v1["is_pat"], "G_area": g["area"]}


# ------------------------------------------------------------------ 그림 3: 흐름 vs 진동
def cell_phase_speed(x, fps, G):
    T = len(x); a = T // 2; half = int(round(0.5 * fps))
    lums = [luminance_cd(torch.from_numpy(x[j:j + 1]).cuda())[0] for j in range(a - half, a + half + 1)]
    b = G.frame(lums[half], area_min=AREA_P2, maps=True)
    mask = torch.from_numpy(b["mask"]).cuda(); kmap = b["k_map"]
    ks, inv = torch.unique(kmap[mask], return_inverse=True)
    ys, xs = torch.nonzero(mask, as_tuple=True)
    cid = (torch.div(ys, G.h / G.rows, rounding_mode="floor").long() * G.cols + torch.div(xs, G.w / G.cols, rounding_mode="floor").long())
    z = torch.stack([_responses(G.bank, L, ks)[inv, ys, xs] for L in lums])
    prod = z[1:] * z[:-1].conj()
    cells = torch.unique(cid)[:: max(1, len(torch.unique(cid)) // 8)][:8]
    out = []
    for c in cells:
        m = cid == c
        v = torch.atan2(prod[:, m].imag.sum(1), prod[:, m].real.sum(1)).cpu().numpy()
        out.append(v * fps / (2 * math.pi))                            # 주기/초
    return np.array(out), np.arange(1, len(lums)) / fps


def fig_motion():
    import pattern_motion_conformance as PM
    G = GP.GridPatternJudge(PM.H, PM.W)
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.4), sharey=True)
    res = {}
    for ax, (name, kw) in zip(axs, (("흐름 60 % (한 방향 1 주기/초)", dict(motion="drift", area=0.60, speed=1.0)),
                                    ("진동 30 % (2 Hz 앞뒤)", dict(motion="osc", area=0.30, freq=2.0)))):
        x, _ = PM.clip(**kw)
        v, t = cell_phase_speed(x, PM.FPS, G)
        for row in v:
            ax.plot(t, row, color=C_G, alpha=0.75, lw=1.6)
        ax.axhline(0, color="#444", lw=1)
        v1 = PM.v1_pattern(x)
        g = judge_pattern_motion(x, PM.FPS, G=G)
        safe = not (g["p1"] or g["p2"])
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("1 초 창 안의 시각 (초)")
        txt = (f"팀원 판: 프레임마다 정지 규칙 → {'위험' if v1 else '안전'}\n"
               f"G: 칸 위상 속도가 {'한 방향 일정 → 흐름 면제 → 안전' if safe else '부호가 바뀜 → 변화 패턴 → 위험'}")
        ax.text(0.02, 0.97, txt, transform=ax.transAxes, va="top", fontsize=10.5, bbox=dict(boxstyle="round", fc="white", ec="#bbb"))
        res[kw["motion"]] = {"v1": v1, "G_dangerous": not safe}
    axs[0].set_ylabel("칸별 줄무늬 이동 속도 (주기/초)")
    fig.suptitle("③ 움직이는 줄무늬 — 팀원 판은 움직임을 보지 않고, G 는 칸마다 복소 필터 위상의 변화로 '흐름'과 '진동'을 가른다 (선 = 격자 칸 8 개)", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "ba3_drift_vs_osc.png", dpi=130); plt.close(fig)
    NUM["motion"] = res


# ------------------------------------------------------------------ 그림 4: 대비 반전
def fig_reverse():
    import pattern_motion_conformance as PM
    x, _ = PM.clip(motion="reverse", area=0.30, freq=2.0)
    f1 = judge_luminance(x, PM.FPS)
    lum = luminance_cd(torch.from_numpy(x).cuda())
    mean = lum.mean((1, 2)).cpu().numpy()
    g = judge_pattern_motion(x, PM.FPS)
    t = np.arange(len(x)) / PM.FPS
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    axs[0].plot(t, 100 * f1["area_up"], color=C_V1, lw=2, label="밝아진 픽셀 면적 (%)")
    axs[0].plot(t, 100 * f1["area_down"], color=C_G, lw=2, ls="--", label="어두워진 픽셀 면적 (%)")
    axs[0].axhline(25, color=C_BAD, lw=1.5, ls=":", label="깜빡임 규칙 문턱 25 %")
    axs[0].set_ylim(0, 40); axs[0].set_xlabel("초"); axs[0].set_ylabel("화면 대비 %"); axs[0].legend(fontsize=9.5, loc="upper right")
    axs[0].set_title(f"깜빡임 규칙(F1): 반전마다 위 15 % · 아래 15 % → 문턱 미달 → {'위험' if f1['dangerous'] else '안전'}", fontsize=11)
    ax2 = axs[0].twinx(); ax2.plot(t, mean, color="#777", lw=1); ax2.set_ylim(0, 120); ax2.set_ylabel("화면 평균 밝기 cd/m² (회색)", color="#777")
    b = g["best2"] or {}
    vals = [100 * b.get("area_change", 0), 100 * b.get("area_static", 0), 100 * b.get("area_flow", 0)]
    axs[1].bar(["변화", "정지", "흐름"], vals, color=[C_BAD, "#9AA0A8", C_OK])
    axs[1].axhline(25, color=C_BAD, lw=1.5, ls=":"); axs[1].set_ylim(0, 40); axs[1].set_ylabel("화면 대비 %")
    axs[1].set_title(f"G P2: 칸별 분류 — 변화 {vals[0]:.0f}% > 25% ({b.get('reason', '')}) → {'위험' if g['p2'] else '안전'}", fontsize=11)
    fig.suptitle("④ 대비 반전 줄무늬 30 % (초당 4 번 반전) — 절반은 밝아지고 절반은 어두워져 평균도, 방향별 면적도 그대로. 깜빡임 규칙으로는 원리적으로 못 잡는다", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "ba4_reverse.png", dpi=130); plt.close(fig)
    NUM["reverse"] = {"F1_dangerous": bool(f1["dangerous"]), "F1_max_up": float(f1["area_up"].max()), "F1_max_down": float(f1["area_down"].max()),
                      "G_p2": bool(g["p2"]), "G_change_area": b.get("area_change")}


# ------------------------------------------------------------------ 그림 5: 점수표
def fig_score():
    rows = [("정지 패턴 규격 경계 시험 (21, 정답 수)", 16, 21, 21, True),
            ("움직이는 패턴 규격 경계 시험 (16, 정답 수)", 9, 16, 16, True),
            ("팀원 판이 패턴이라 한 안전 바탕 (41, 오탐 수)", 41, 0, 41, False),
            ("합성 정지 무늬 위험 (42, 탐지 수; G = BT.1702 축)", None, 36, 42, True)]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for i, (name, a, b, n, hi) in enumerate(rows):
        y = len(rows) - 1 - i
        if a is not None:
            ax.barh(y + 0.18, a / n, height=0.34, color=C_V1)
            ax.text(a / n + 0.01, y + 0.18, f"팀원 판 {a}/{n}", va="center", fontsize=10.5)
        else:
            ax.text(0.01, y + 0.18, "팀원 판: 이 세트 미측정 (1–10 cpd 판 G = 29/42)", va="center", fontsize=10, color="#777")
        ax.barh(y - 0.18, b / n, height=0.34, color=C_G)
        ax.text(b / n + 0.01, y - 0.18, f"G {b}/{n}" + ("  (낮을수록 좋음)" if not hi else ""), va="center", fontsize=10.5, fontweight="bold")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=10.5)
    ax.set_xlim(0, 1.25); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_title("⑤ 전/후 — 같은 시험 세트, 같은 정답 (일상 영상 379 개에서 G 오탐: 정지 0 · 변화 0 · BT.1702 축 경계 1)", fontsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "ba5_scoreboard.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    for f in (fig_scene, fig_area39, fig_motion, fig_reverse, fig_score):
        if only and f.__name__ not in only:
            continue
        f(); torch.cuda.empty_cache(); print(f.__name__, "ok", flush=True)
    if only:
        sys.exit(0)
    (OUT / "ba_numbers.json").write_text(json.dumps(NUM, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(NUM, ensure_ascii=False, indent=1, default=float))
