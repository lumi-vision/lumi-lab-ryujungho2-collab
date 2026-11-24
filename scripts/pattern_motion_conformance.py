"""움직이는 패턴(P2·흐름 면제) 적합성 시험 — 사례와 정답은 실행 전에 이 파일로 고정(커밋)한다.

정답 근거(원문, docs/STANDARDS.md §3-1): ITU-R BT.1702-3 Att.1 = Ofcom Annex 1 —
  정지 > 40 % · 방향 전환/진동/점멸/대비 반전 > 25 % · "obviously flow smoothly across, into, or out of the screen in one direction" 면제.
  쌍 > 5, ΔL ≥ 20 cd/m², 어두운 막대 < 160 (모든 사례 3 cpd·쌍 > 5 를 만족하게 만듦).
정답은 '패턴 규칙' 기준(정지 또는 변화 패턴 위험). 깜빡임 규칙(F1)이 같은 영상을 잡는지는 따로 표시한다.
회전(바큇살 회전)은 원문 '가로질러·안으로·밖으로' 에 들지 않아 규격이 모호 → 채점하지 않고 참고로만.
영상: 768×432, 24 fps, 2 초, 바탕 40 cd/m², 막대 20/120 cd/m²(달리 적지 않으면), 3 cpd.
⚠️ 움직이는/깜빡이는 줄무늬 — 메모리에서만 다루고 저장·재생하지 않는다.

    python scripts/pattern_motion_conformance.py
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
from lumi_judge.grid_flash import judge_luminance  # noqa: E402
from lumi_judge.grid_motion import judge_pattern_motion  # noqa: E402
from lumi_judge.grid_pattern import GridPatternJudge, pixels_per_degree  # noqa: E402

H, W, FPS, SEC, BG = 432, 768, 24.0, 2.0, 40.0
PPD = pixels_per_degree(W, H)


def to_u8(L):
    y = np.clip(L / 200.0, 0, 1)
    s = np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055)
    return np.repeat(np.round(s * 255).astype(np.uint8)[..., None], 3, -1)


yy, xx = np.mgrid[:H, :W].astype(np.float64)
cy, cx = H / 2, W / 2


def region(area, round_=False):
    if round_:
        return np.hypot(xx - cx, yy - cy) <= math.sqrt(area * H * W / math.pi)
    rw = min(math.sqrt(area * W * H * (W / H)), W); rh = min(area * W * H / rw, H)
    return (np.abs(xx - cx) <= rw / 2) & (np.abs(yy - cy) <= rh / 2)


def clip(motion="static", area=0.3, lo=20.0, hi=120.0, cpd=3.0, speed=1.0, freq=2.0):
    """motion: static | drift | osc | reverse | flash | orient | zoom | drift_back | rotate. speed = 주기/초, freq = Hz."""
    period = PPD / cpd
    reg = region(area, round_=motion in ("zoom", "rotate"))
    out = []
    for i in range(int(FPS * SEC)):
        t = i / FPS
        ang, ph, on, inv = 0.0, 0.0, True, False
        if motion == "drift":
            ph = speed * t
        elif motion == "drift_back":                       # 0.5 초마다 흐름 방향이 바뀜 = '방향 전환'
            k = int(t / 0.5); r = t - 0.5 * k
            ph = speed * (r if k % 2 == 0 else 0.5 - r)    # 연속: 짝수 구간 앞으로, 홀수 구간 뒤로
        elif motion == "osc":
            ph = 0.5 * math.sin(2 * math.pi * freq * t)
        elif motion == "reverse":
            inv = int(t * 2 * freq) % 2 == 1               # freq Hz 반전 주기(초당 2·freq 번 반전)
        elif motion == "flash":
            on = int(t * 2 * freq) % 2 == 0
        elif motion == "orient":
            ang = 90.0 if int(t * 2 * freq) % 2 else 0.0
        if motion == "zoom":
            u = np.hypot(xx - cx, yy - cy) / period - speed * t
        elif motion == "rotate":
            n_sp = max(8, int(round(2 * math.pi * math.sqrt(area * H * W / math.pi) * 0.7 / period)))
            u = (np.arctan2(yy - cy, xx - cx) / (2 * math.pi)) * n_sp - speed * t
        else:
            a = math.radians(ang)
            u = ((xx - cx) * math.cos(a) + (yy - cy) * math.sin(a)) / period + ph
        q = np.mod(u, 1.0) < 0.5
        if inv:
            q = ~q
        L = np.where(q, hi, lo) if on else np.full_like(xx, (hi + lo) / 2)
        out.append(to_u8(np.where(reg, L, BG)))
    return np.stack(out), float(reg.mean())


CASES = [  # (이름, 정답 위험?, 채점?, kwargs)
    ("정지 30 %", False, True, dict(motion="static", area=0.30)),
    ("정지 60 %", True, True, dict(motion="static", area=0.60)),
    ("흐름 30 % (1 주기/초)", False, True, dict(motion="drift", area=0.30, speed=1.0)),
    ("흐름 60 % (1 주기/초)", False, True, dict(motion="drift", area=0.60, speed=1.0)),
    ("흐름 60 % (4 주기/초)", False, True, dict(motion="drift", area=0.60, speed=4.0)),
    ("동심원 줌 60 % (밖으로 1 주기/초)", False, True, dict(motion="zoom", area=0.60, speed=1.0)),
    ("진동 30 % (2 Hz)", True, True, dict(motion="osc", area=0.30, freq=2.0)),
    ("진동 20 % (2 Hz)", False, True, dict(motion="osc", area=0.20, freq=2.0)),
    ("진동 30 % ΔL 19", False, True, dict(motion="osc", area=0.30, lo=40.0, hi=59.0)),
    ("진동 30 % ΔL 21", True, True, dict(motion="osc", area=0.30, lo=40.0, hi=61.0)),
    ("대비 반전 30 % (1 Hz)", True, True, dict(motion="reverse", area=0.30, freq=1.0)),
    ("대비 반전 30 % (2 Hz)", True, True, dict(motion="reverse", area=0.30, freq=2.0)),
    ("점멸 30 % (1 Hz)", True, True, dict(motion="flash", area=0.30, freq=1.0)),
    ("방향 바뀜 0°↔90° 30 % (1 Hz)", True, True, dict(motion="orient", area=0.30, freq=1.0)),
    ("흐름 방향 전환 30 % (0.5 초마다)", True, True, dict(motion="drift_back", area=0.30, speed=2.0)),
    ("반전 20 % (2 Hz)", False, True, dict(motion="reverse", area=0.20, freq=2.0)),
    ("회전 바큇살 30 % (참고)", None, False, dict(motion="rotate", area=0.30, speed=0.5)),
]


def v1_pattern(frames):
    r = J.analyze_frames(list(frames), FPS, device="cuda", mask_config=J.PartialMaskConfig())
    return "정적 공간 패턴" in r.active_labels


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    G = GridPatternJudge(H, W)
    rows = []
    for name, truth, scored, kw in CASES:
        x, ar = clip(**kw)
        g = judge_pattern_motion(x, FPS, G=G)
        f1 = judge_luminance(x, FPS)["dangerous"]
        v1 = v1_pattern(x)
        gd = g["p1"] or g["p2"]
        b = g["best2"] or g["best1"]
        why = (f"정지 {100*b['area_static']:.0f}% · 변화 {100*b['area_change']:.0f}% · 흐름 {100*b['area_flow']:.0f}%" + (f" ({b['reason']})" if b["reason"] else "")) if b else "-"
        rows.append({"case": name, "real_area": round(100 * ar, 1), "truth": truth, "scored": scored, "G": gd, "G_p1": g["p1"], "G_p2": g["p2"],
                     "v1": v1, "F1": f1, "why": why})
        print(f"  {name:28s} 정답 {truth} | G {gd} (P1 {g['p1']} P2 {g['p2']}; {why}) | v1 {v1} | F1 깜빡임 {f1}", flush=True)
    S = [r for r in rows if r["scored"]]
    acc = {k: sum(r[k] == r["truth"] for r in S) for k in ("G", "v1")}
    yn = lambda b: "-" if b is None else ("위험" if b else "안전")
    L = ["# 움직이는 패턴 적합성 시험 (P2 · 흐름 면제)", "",
         f"> 정답 = 규격 패턴 규칙(BT.1702-3 Att.1 / Ofcom Annex 1) · 채점 {len(S)} 사례 · 정답 수: **G {acc['G']}/{len(S)}** · v1(팀원 원본) {acc['v1']}/{len(S)} · "
         "F1 = 같은 영상에 깜빡임 규칙이 걸리는지(참고)", "",
         "| 사례 | 실면적 % | 정답 | G | G 근거 | v1 | F1 깜빡임 |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        mk = lambda k: yn(r[k]) + ("" if not r["scored"] or r[k] == r["truth"] else " ✗")
        L.append(f"| {r['case']} | {r['real_area']} | {yn(r['truth'])} | {mk('G')} | {r['why']} | {mk('v1')} | {yn(r['F1'])} |")
    (ROOT / "checks" / "judge" / "pattern_motion_conformance.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (ROOT / "data" / "judge" / "pattern_motion_conformance.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L[:3]))


if __name__ == "__main__":
    main()
