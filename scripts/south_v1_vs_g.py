"""South '안전'(PhotosensitivityPal 라벨) 영상 중 팀원 판(v1)이 '정적 공간 패턴'이라 한 840 개 — 클립마다 이유를 분해.

각 클립: v1 이 패턴이라 한 첫 구간의 가운데 프레임 하나에서
  v1 : 대비 면적(fine_area) · 6 쌍 이상이 나온 경로(직선 주사 / 고리·바큇살 / FFT 대체) · 최대 쌍 · (직선이면) 그 구간의 G 규칙성 값
  G  : 같은 프레임에서 어느 단계에서 멈췄나 (기본 G = 1–10 cpd, 32 칸; BT.1702 축 판정도 함께)
       G0 줄무늬 칸 없음 · G1 가장 큰 후보 영역 원면적 ≤ 40 % · G2 영역 > 40 % 이지만 법선 경로에 6 쌍 없음
       G3 6 쌍 경로는 있으나 규칙성 실패로 확인 면적 ≤ 40 % · G4 G 도 이 프레임은 패턴(클립 판정은 지속 조건으로 갈릴 수 있음)
  클립 판정: G(P1, 초당 8 프레임, 0.5 초 지속).
출력: data/judge/south_v1g.jsonl · checks/judge/south_v1_vs_g.md · data/viewer/south_v1g/ (3 칸 근거 그림 jpg + index.html, 로컬·비공개 전용)
⚠️ South 재배포 금지 — 공개 저장소에 그림을 넣지 않는다.
"""
from __future__ import annotations

import collections
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
import lumi_judge as J  # noqa: E402
from lumi_judge import grid_pattern as GP  # noqa: E402
from lumi_judge.config import PSEConfig, ViewingConfig  # noqa: E402
from lumi_judge.gpu_kernel import hyst_runs  # noqa: E402
from lumi_judge.grid_flash import luminance_cd  # noqa: E402
from lumi_judge.photometry import PhotometricConverter  # noqa: E402
from lumi_judge.spatial_mask import PartialStaticPatternDetector  # noqa: E402
from eval_real_hazards import decode as dec_any  # noqa: E402
from grid_pattern_eval import judge_clip  # noqa: E402
from make_logic_figs import v1_best_line  # noqa: E402

OUT_V = ROOT / "data" / "viewer" / "south_v1g"
REASON = {"G0": "줄무늬 칸 없음", "G1": "후보 영역 원면적 ≤ 40 %", "G2": "영역 > 40 % 이나 법선 경로 6쌍 없음",
          "G3": "6쌍 경로는 있으나 규칙성 실패", "G4": "G 도 이 프레임은 패턴"}
_DET: dict = {}


def v1_decompose(fr):
    vc = ViewingConfig()
    h0, w0 = fr.shape[:2]
    if w0 > vc.analysis_max_width:
        s = vc.analysis_max_width / w0
        fr = cv2.resize(fr, (vc.analysis_max_width, max(2, int(round(h0 * s)))), interpolation=cv2.INTER_AREA)
    h, w = fr.shape[:2]
    if (h, w) not in _DET:
        if len(_DET) > 6:
            _DET.clear(); torch.cuda.empty_cache()
        geom = vc.resolve_geometry(w, h); pse = PSEConfig()
        _DET[(h, w)] = (PartialStaticPatternDetector(geometry=geom, pse=pse, device="cuda", mask_config=J.PartialMaskConfig()),
                        PhotometricConverter(l_max_cd_m2=200.0, pse=pse, device="cuda"), pse)
    det, conv, pse = _DET[(h, w)]
    det.reset()
    lum = conv.convert_bgr(fr)[1]
    l4 = lum[None, None]
    cm4 = det._compute_contrast_mask(l4)
    cm = cm4.squeeze().bool()
    fine = float(cm.float().mean())
    isp, area, pairs, cpd = det.inspect_frame(lum)
    bl = v1_best_line(det, lum, pse)
    lin = bl["pairs"] if bl else 0
    centers = det._estimate_candidate_centers(l4, cm4)
    try:
        rc = det._scan_centers_batched(l4, centers, check_spokes=True)[0]
    except Exception:                                                   # noqa: BLE001
        rc = 0
    fft = det._evaluate_2d_fft_fallback(lum)[0] if max(lin, rc) < pse.pattern_min_pairs else 0
    src = "직선 주사" if lin >= max(rc, fft) and lin >= 6 else ("고리·바큇살" if rc >= max(lin, fft) and rc >= 6 else ("FFT 대체" if fft >= 6 else "없음"))
    reg = None
    if bl and src == "직선 주사":
        reg = float(GP._best_run_regularity(np.asarray(bl["et"], float), np.asarray(bl["ev"], float), len(bl["et"]), GP._sdr(pse)))
    return {"is_pat": bool(isp), "fine_area": fine, "pairs": int(pairs), "src": src, "line_pairs": int(lin), "ring_pairs": int(rc),
            "fft_pairs": int(fft), "line_reg_G": reg, "_cm": cm.cpu().numpy(), "_bl": bl, "_wh": (w, h)}


def g_decompose(G, lum):
    pse = GP._sdr(PSEConfig())
    k, amp, th, cp = G.bank.respond(lum)
    cand = (2.0 * amp >= GP.CAND_FRAC * 20.0)
    frac, thc, fc, coh = GP._cells(cand, th, cp, G.rows, G.cols)
    on = frac >= GP.CELL_MIN_FRAC
    lab, n = GP._regions(on, thc, fc, coh)
    ch, cw = G.h / G.rows, G.w / G.cols
    cn = cand.cpu().numpy(); an = amp.cpu().numpy()
    regs, paths_draw = [], []
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
        raw = float(pat.mean())
        if raw <= GP.AREA_MIN:
            regs.append({"cells": len(cells), "raw": raw}); continue
        seeds = []
        for ci in np.linspace(0, len(cells) - 1, min(12, len(cells))).round().astype(int):
            r, c = cells[ci]; y0, y1, x0, x1 = int(r * ch), int((r + 1) * ch), int(c * cw), int((c + 1) * cw)
            b = np.where(pat[y0:y1, x0:x1], an[y0:y1, x0:x1], -1)
            if b.max() > 0:
                yy, xx = np.unravel_index(int(np.argmax(b)), b.shape); seeds.append((y0 + yy, x0 + xx))
        pr = GP._trace_profiles(lum, th, pat, seeds, G.ppd)
        if not pr:
            regs.append({"cells": len(cells), "raw": raw, "n": 0, "ge6": 0, "ok": 0}); continue
        L = max(len(q[0]) for q in pr); V = np.zeros((len(pr), L)); T = np.zeros_like(V); ln = np.array([len(q[0]) for q in pr])
        for i, (v, t) in enumerate(pr):
            V[i, :len(v)] = v; T[i, :len(t)] = t
        P, C, _, NE, ET, EV = hyst_runs(V, T, ln, pse)
        P, NE, ET, EV = P.get(), NE.get(), ET.get(), EV.get()
        regok = np.array([GP._best_run_regularity(ET[i], EV[i], int(NE[i]), pse) <= GP.LOCAL_REG_MAX for i in range(len(P))])
        ok = (P >= GP.MIN_PAIRS) & regok
        regs.append({"cells": len(cells), "raw": raw, "n": int(len(P)), "ge6": int((P >= GP.MIN_PAIRS).sum()), "ok": int(ok.sum()),
                     "verified_area": raw * float(ok.mean())})
        from make_before_after import trace_pts
        for s, good in zip(seeds, ok.tolist() + [False] * (len(seeds) - len(ok))):
            paths_draw.append((trace_pts(lum, th, pat, s), bool(good)))
    big = max(regs, key=lambda q: q["raw"]) if regs else None
    tested = [q for q in regs if "n" in q]
    if not regs or not on.any():
        why = "G0"
    elif not tested:
        why = "G1"
    elif any(q.get("verified_area", 0) > GP.AREA_MIN for q in tested):
        why = "G4"
    elif all(q["ge6"] == 0 for q in tested):
        why = "G2"
    else:
        why = "G3"
    return {"why": why, "max_raw": big["raw"] if big else 0.0, "n_on": int(on.sum()), "regions_tested": tested,
            "_on": [(int(r), int(c)) for r, c in np.argwhere(on)], "_paths": paths_draw, "_shape": (G.rows, G.cols)}


def render(fr, v1, g, path):
    H, W = fr.shape[:2]; s = 300 / W
    img = cv2.resize(fr, (300, max(2, int(round(H * s)))), interpolation=cv2.INTER_AREA); h = img.shape[0]
    a = img.copy()
    cm = cv2.resize(v1["_cm"].astype(np.uint8), (300, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    a[cm] = (0.5 * a[cm] + 0.5 * np.array([0, 94, 194])).astype(np.uint8)
    bl = v1["_bl"]
    if bl and v1["src"] == "직선 주사":
        w1, h1 = v1["_wh"]; sx, sy = 300 / w1, h / h1
        pts = np.stack([bl["x"] * sx, bl["y"] * sy], 1).astype(np.int32)
        cv2.polylines(a, [pts], False, (0, 94, 194), 1)
        th = bl["th"]; rt = bl["et"][bl["rs"]:bl["re"] + 1]
        i0, i1 = int(np.searchsorted(th, rt[0])), int(np.searchsorted(th, rt[-1]))
        cv2.polylines(a, [pts[i0:i1]], False, (77, 225, 255), 3)
    b = img.copy(); rows, cols = g["_shape"]; chh, cww = h / rows, 300 / cols
    for r, c in g["_on"]:
        cv2.rectangle(b, (int(c * cww), int(r * chh)), (int((c + 1) * cww), int((r + 1) * chh)), (122, 210, 255), 1)
    for pts, good in g["_paths"]:
        if pts:
            p = (np.array(pts)[:, ::-1] * np.array([300 / W, h / H])).astype(np.int32)
            cv2.polylines(b, [p], False, (90, 158, 27) if good else (43, 57, 192), 2)
    bar = np.full((18, 900, 3), 24, np.uint8)
    cv2.putText(bar, "input", (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1)
    cv2.putText(bar, f"v1: contrast {100 * v1['fine_area']:.0f}%  {v1['pairs']} pairs", (304, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1)
    cv2.putText(bar, f"G: {g['why']}  max region {100 * g['max_raw']:.0f}%", (604, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1)
    cv2.imwrite(str(path), np.vstack([bar, np.hstack([img, a, b])]), [cv2.IMWRITE_JPEG_QUALITY, 82])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT_V.mkdir(parents=True, exist_ok=True)
    R = [json.loads(l) for l in open(ROOT / "data/judge/south_partial/rows.jsonl", encoding="utf-8") if l.strip()]
    R = [r for r in R if "error" not in r and r["human"]["dangerous"] == "safe" and "정적 공간 패턴" in r["labels"]]
    src = ROOT / "data/real/south2021/social-media-study-gifs"; files = {p.stem: p for p in src.iterdir()}
    cache = ROOT / "data/judge/south_v1g.jsonl"
    done = {json.loads(l)["id"] for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()} if cache.exists() else set()
    Gs: dict = {}
    with cache.open("a", encoding="utf-8") as f:
        for i, r in enumerate(R, 1):
            if r["label"] in done:
                continue
            try:
                x, fps = dec_any(files[r["label"]])
                iv = r["intervals_by_label"]["정적 공간 패턴"][0]; t = min(len(x) - 1, (iv[0] + iv[1]) // 2)
                fr = x[t]
                v1 = v1_decompose(fr)
                h, w = fr.shape[:2]
                if (h, w) not in Gs:
                    if len(Gs) > 6:
                        Gs.clear(); torch.cuda.empty_cache()
                    Gs[(h, w)] = (GP.GridPatternJudge(h, w), GP.GridPatternJudge(h, w, cpd_lo=0.15, cpd_hi=1e3))
                G, Gc = Gs[(h, w)]
                lum = luminance_cd(torch.from_numpy(fr[None]).cuda())[0]
                g = g_decompose(G, lum)
                clip_G = judge_clip(x, fps)["danger"]
                c_frame = Gc.frame(lum)["is_pat"]
                render(fr, v1, g, OUT_V / f"{r['label']}.jpg")
                row = {"id": r["label"], "source": r["human"]["source"], "t_frame": t, "fps": fps, "w": w, "h": h,
                       "v1": {k: v for k, v in v1.items() if not k.startswith("_")},
                       "G": {k: v for k, v in g.items() if not k.startswith("_")}, "G_clip": clip_G, "G_bt1702_frame": bool(c_frame)}
            except Exception as e:  # noqa: BLE001
                row = {"id": r["label"], "error": f"{type(e).__name__}: {e}"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            torch.cuda.empty_cache()
            if i % 50 == 0:
                print(f"  {i}/{len(R)}", flush=True)
    report(cache)


def report(cache):
    rows = [json.loads(l) for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()]
    err = [r for r in rows if "error" in r]; rows = [r for r in rows if "error" not in r]
    ct = collections.Counter((r["v1"]["src"], r["G"]["why"]) for r in rows)
    srcs = ["직선 주사", "고리·바큇살", "FFT 대체", "없음"]; whys = ["G0", "G1", "G2", "G3", "G4"]
    L = ["# South '안전' 중 팀원 판이 정지 패턴이라 한 영상 — 이유 분해 (scripts/south_v1_vs_g.py)", "",
         f"> {len(rows)} 클립 (오류 {len(err)}) · 라벨은 PhotosensitivityPal 출력(사람 판정 아님) · 프레임 = v1 첫 패턴 구간 가운데", "",
         "## v1 이 6 쌍을 얻은 경로 × G 가 멈춘 단계", "", "| v1 경로 \\ G | " + " | ".join(f"{w} {REASON[w]}" for w in whys) + " | 합 |",
         "|---" * (len(whys) + 2) + "|"]
    for s in srcs:
        n = [ct[(s, w)] for w in whys]
        if sum(n):
            L.append(f"| {s} | " + " | ".join(str(v) for v in n) + f" | {sum(n)} |")
    L.append("| 합 | " + " | ".join(str(sum(ct[(s, w)] for s in srcs)) for w in whys) + f" | {len(rows)} |")
    fa = np.array([r["v1"]["fine_area"] for r in rows]); mr = np.array([r["G"]["max_raw"] for r in rows])
    lr = [r["v1"]["line_reg_G"] for r in rows if r["v1"]["line_reg_G"] is not None]
    L += ["", "## 수치", "", f"- v1 대비 면적(fine_area) 중앙값 {100 * np.median(fa):.0f}% · G 가장 큰 후보 영역 원면적 중앙값 {100 * np.median(mr):.1f}%",
          f"- v1 직선 주사 구간에 G 규칙성 검사: {len(lr)} 개 중 {sum(v > 0.25 for v in lr)} 개 탈락(> 0.25), 중앙값 {np.median(lr):.2f}" if lr else "",
          f"- G 클립 판정(P1, 기본) 위험: {sum(r['G_clip'] for r in rows)} / {len(rows)} · BT.1702 축 프레임 판정 위험: {sum(r['G_bt1702_frame'] for r in rows)}",
          f"- 출처별: " + ", ".join(f"{k} {v}" for k, v in collections.Counter(r['source'] for r in rows).items())]
    (ROOT / "checks/judge/south_v1_vs_g.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
