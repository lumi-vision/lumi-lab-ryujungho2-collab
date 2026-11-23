"""정지 패턴 적합성 시험 세트 — 규격 조건의 경계 바로 위·아래를 직접 만들어 판정기들을 비교한다 (판정기와 독립된 정답).

정답 근거(원문): ITC/Ofcom Annex 1 §5·5.1, ITU-R BT.1702-3 Att.1 — 정지 줄무늬, 밝고 어두운 쌍 > 5, 어두운 막대 < 160 cd/m² 이고
밝은 막대와 ≥ 20 cd/m² 차이, 면적 > 40 %. 모양은 '평행·방사·곡선·직선, 점 열 포함'.
영상: 768×432, 24 fps, 1 초(정지), 바탕 40 cd/m² 균일. 밝기 → sRGB 8 비트(흰색 200 cd/m²).
비교: v1 = 팀원 Partial B2 원본 설정 · v2 = 면적 수정(+방사) · G = 격자 판정기(grid_pattern).
⚠️ 정지 줄무늬 영상(깜빡임 없음) — 메모리에서만 다루고 저장·재생하지 않는다.

    python scripts/pattern_conformance.py
"""
from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from lumi_judge.cpu_limit import limit_cpu  # noqa: E402
limit_cpu()
import lumi_judge as J  # noqa: E402
from lumi_judge.grid_flash import luminance_cd  # noqa: E402
from lumi_judge.grid_pattern import GridPatternJudge, pixels_per_degree  # noqa: E402

H, W, FPS, BG = 432, 768, 24.0, 40.0


def to_u8(L: np.ndarray) -> np.ndarray:
    y = np.clip(L / 200.0, 0, 1)
    s = np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055)
    g = np.round(s * 255).astype(np.uint8)
    return np.repeat(g[..., None], 3, -1)


def stripes(kind="straight", cpd=3.0, lo=20.0, hi=120.0, area=0.6, pairs=None, angle=0.0):
    """정지 줄무늬 밝기 지도(cd/m²). area = 화면 대비 패턴 영역 비율(가운데 정사각에 가까운 사각형/원), pairs 를 주면 그 쌍 수만."""
    ppd = pixels_per_degree(W, H)
    period = ppd / cpd
    yy, xx = np.mgrid[:H, :W].astype(np.float64)
    cy, cx = H / 2, W / 2
    a = math.radians(angle)
    if kind == "straight":
        u = (xx - cx) * math.cos(a) + (yy - cy) * math.sin(a)
        phase = u / period
    elif kind == "concentric":
        phase = np.hypot(xx - cx, yy - cy) / period
    elif kind == "radial":                        # 바큇살: 반지름 r 에서의 각 주기가 period 가 되는 살 수를 영역 가장자리 기준으로
        r_ref = math.sqrt(area * H * W / math.pi) * 0.7
        n_sp = max(8, int(round(2 * math.pi * r_ref / period)))
        phase = (np.arctan2(yy - cy, xx - cx) / (2 * math.pi)) * n_sp
    elif kind == "checker":
        phase = None
        q = (np.floor((xx - cx) / (period / 2)) + np.floor((yy - cy) / (period / 2))) % 2
    elif kind == "dots":
        phase = None
        fx = ((xx - cx) / period) % 1.0; fy = ((yy - cy) / period) % 1.0
        q = (np.hypot(fx - 0.5, fy - 0.5) < 0.3).astype(float)
    if phase is not None:
        q = (np.mod(phase, 1.0) < 0.5).astype(float)
    L = np.where(q > 0.5, hi, lo)
    # 영역
    if kind in ("concentric", "radial"):
        R = math.sqrt(area * H * W / math.pi)
        reg = np.hypot(xx - cx, yy - cy) <= R
    else:
        rw = math.sqrt(area * W * H * (W / H)); rh = area * W * H / rw
        rw, rh = min(rw, W), min(rh, H)
        if rw >= W:
            rh = area * H
        reg = (np.abs(xx - cx) <= rw / 2) & (np.abs(yy - cy) <= rh / 2)
    if pairs is not None and kind == "straight":
        u = (xx - cx) * math.cos(a) + (yy - cy) * math.sin(a)
        reg &= np.abs(u) <= pairs * period / 2
    return np.where(reg, L, BG), float(reg.mean())


def truth_of(kw, area_real):
    """정답 = 규격: 실제 패턴 영역 > 40 %, 어두운 막대 < 160 이고 차 ≥ 20 cd/m² (SDR), 쌍 > 5 · 1–10 cpd(문헌 근거 구현 선택).
    1–10 cpd·이 시청 조건에서 면적 > 40 % 인 줄무늬는 쌍이 항상 5 를 넘으므로, 쌍 제한 사례는 면적으로 갈린다(면적이 작음)."""
    lo, hi = kw.get("lo", 20.0), kw.get("hi", 120.0)
    return area_real > 0.40 and lo < 160 and (hi - lo) >= 20


def cases():
    C = []
    add = lambda name, truth, **kw: C.append((name, truth, kw))
    for p in (4, 5, 6, 8):
        add(f"쌍 {p} (1.5 cpd 띠)", None, pairs=p, area=0.9, cpd=1.5)
    for d in (19, 21, 60):
        add(f"밝기 차 {d}", d >= 20, lo=40.0, hi=40.0 + d)
    add("어두운 막대 155 (차 30)", True, lo=155.0, hi=185.0)
    add("어두운 막대 165 (차 30)", False, lo=165.0, hi=195.0)
    for ar in (0.35, 0.39, 0.42, 0.6):
        add(f"면적 {int(ar*100)} %", ar > 0.40, area=ar)
    for k in ("straight", "concentric", "radial", "checker", "dots"):
        add(f"모양 {k}", True, kind=k, area=0.6)
    add("직선 45°", True, angle=45.0)
    for c in (2.0, 6.0):
        add(f"{c} cpd", True, cpd=c)
    return C


def judge_v(frames, variant):
    m0 = J.PartialMaskConfig()
    m = {"v1": m0, "v2": dataclasses.replace(m0, pattern_area_from_mask=True, pattern_area_radial=True)}[variant]
    r = J.analyze_frames(list(frames), FPS, device="cuda", mask_config=m)
    return "정적 공간 패턴" in r.active_labels


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    G = GridPatternJudge(H, W)
    rows = []
    for name, truth, kw in cases():
        L, area_real = stripes(**kw)
        truth = truth_of(kw, area_real)
        name = f"{name} [실면적 {100*area_real:.0f}%]"
        u8 = to_u8(L)
        frames = np.repeat(u8[None], int(FPS), 0)
        lum = luminance_cd(torch.from_numpy(u8[None]).cuda())[0]
        g = G.frame(lum)
        v1 = judge_v(frames, "v1"); v2 = judge_v(frames, "v2")
        rows.append({"case": name, "truth": truth, "v1": v1, "v2": v2, "G": g["is_pat"], "G_area": round(100 * g["area"], 1),
                     "G_pairs": g["pairs"], "G_kind": g["kind"]})
        print(f"  {name:22s} 정답 {'위험' if truth else '안전'} | v1 {'위험' if v1 else '안전'} v2 {'위험' if v2 else '안전'} "
              f"G {'위험' if g['is_pat'] else '안전'} (면적 {100*g['area']:.0f}% 쌍 {g['pairs']} {g['kind']})", flush=True)
    acc = {k: sum(r[k] == r["truth"] for r in rows) for k in ("v1", "v2", "G")}
    L = ["# 정지 패턴 적합성 시험 (경계 위·아래, 직접 생성)", "",
         f"> 정답 = 규격(ITC/Ofcom Annex 1, BT.1702-3 Att.1) · {len(rows)} 사례 · 정답 수: v1 {acc['v1']} · v2 {acc['v2']} · G {acc['G']}", "",
         "| 사례 | 정답 | v1 원본 | v2 면적 수정 | G 격자 | G 면적 % | G 쌍 | G 모양 |", "|---|---|---|---|---|---|---|---|"]
    yn = lambda b: "위험" if b else "안전"
    for r in rows:
        mark = lambda k: yn(r[k]) + ("" if r[k] == r["truth"] else " ✗")
        L.append(f"| {r['case']} | {yn(r['truth'])} | {mark('v1')} | {mark('v2')} | {mark('G')} | {r['G_area']} | {r['G_pairs']} | {r['G_kind'] or '-'} |")
    out = ROOT / "checks" / "judge" / "pattern_conformance.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    (ROOT / "data" / "judge" / "pattern_conformance.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L[:3]))


if __name__ == "__main__":
    main()
