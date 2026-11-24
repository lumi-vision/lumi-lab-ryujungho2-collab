"""격자 판정기 G — 통합 판정 + 근거 문장 (에이전트 도구가 부를 함수). docs/GRID_JUDGE_DESIGN.md §4.

    from lumi_judge.grid_judge import judge_video
    r = judge_video(frames_bgr_uint8, fps)        # → {"dangerous", "labels", "findings":[{label, clause, sentence, evidence}], "grid_png"(선택)}

포함: F1 휘도 깜빡임 · F2 적색 깜빡임(measure.py 와 판정 동일) · F3 연장 깜빡임 · P1 정지 패턴(흐름 면제 적용) · P2 변화 패턴.
png_dir 를 주면 판정마다 대표 정지 프레임 위에 근거 칸을 칠한 PNG 를 만든다(grid_render).
없는 것은 결과에 '미구현' 으로 명시한다(없는 것을 안전이라 말하지 않음).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .grid_flash import judge_luminance, judge_red, luminance_cd
from .grid_motion import CLAUSE_P2, judge_pattern_motion
from .grid_pattern import CLAUSE_P1, GridPatternJudge

NOT_IMPLEMENTED = ["빠른 장면 전환·교집합(팀원 판의 별도 라벨 — G 에서는 휘도 깜빡임 규칙에 포함)"]
CLAUSE_F3 = "연장 깜빡임: 5 초 창 안에서 1 초 전이 4–6 회(2–3 Hz)가 4 초 이상 (EA IRIS 파라미터 — 팀원 판과 같음; ITU 문구 'more than five seconds' 의 구현)"


def _extended(r, fps, name):
    """F3: F1/F2 의 화면 전이 열에 measure.extended_flash(IRIS 규칙)를 그대로 적용 — 같은 함수라 판정 동일."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lumi_synth.measure import extended_flash
    x = extended_flash(r["transitions"], fps)
    if not x["extended_dangerous"]:
        return None
    t0 = x["first_fail_s"]
    return {"label": name, "clause": CLAUSE_F3,
            "sentence": f"{t0:.1f}초부터 1초에 2–3회 깜빡임이 4초 넘게 이어짐 → 연장 깜빡임 기준 넘음",
            "evidence": {"first_fail_s": t0, "fail_frames": int(x["extended_frames"].sum())}}


def _sec(s):
    return f"{s:.1f}초"


def _flash_sentence(name, r):
    ev = r["evidence"]
    seg = ev["segments"][0] if ev["segments"] else None
    if not r["dangerous"] or seg is None:
        return None
    dl = ev.get("delta_cd_m2_median")
    return (f"{_sec(seg['start_s'])}–{_sec(seg['end_s'])}, 화면의 {ev['hazard_area_pct']:.0f}%(지도의 칸 {len(ev['cells'])}개)에서 "
            f"1초에 최대 {seg['max_flashes_per_s']:.1f}회 {name}"
            + (f", 밝기 변화 약 {dl:.0f} cd/m²" if dl else "")
            + (f", Δu′v′ 최대 {ev['max_delta_uv']:.2f}" if "max_delta_uv" in ev else "")
            + " → 기준(면적 25 % 초과, 1초 3회 초과) 넘음")


@torch.no_grad()
def judge_pattern_clip(frames_bgr: np.ndarray, fps: float, sample_fps: float = 8.0) -> dict:
    h, w = frames_bgr.shape[1:3]
    G = GridPatternJudge(h, w)
    step = max(1, int(round(fps / sample_fps)))
    pat = np.zeros(len(frames_bgr), bool)
    best = None
    for i in range(0, len(frames_bgr), step):
        lum = luminance_cd(torch.from_numpy(np.ascontiguousarray(frames_bgr[i:i + 1])).cuda())[0]
        r = G.frame(lum)
        pat[i:i + step] = r["is_pat"]
        if r["is_pat"] and (best is None or r["area"] > best["area"]):
            best = dict(r, t=i / fps)
    win = max(1, int(math.ceil(fps - 1e-9))); need = int(math.ceil(0.5 * fps - 1e-9))
    cs = np.concatenate([[0], np.cumsum(pat)]); n = len(pat)
    counts = cs[np.minimum(np.arange(n) + win, n)] - cs[:n]
    danger = bool((counts >= need).any()) if n else False
    return {"dangerous": danger, "best": best, "frames": pat}


def _rep_frame(r, fps):
    """깜빡임 대표 프레임: 위험 구간 안에서 화면 전이 면적이 가장 큰 프레임."""
    hz = np.flatnonzero(r["hazard_frames"])
    if not hz.size:
        return 0
    ar = np.maximum(r["area_up"], r["area_down"])
    return int(hz[np.argmax(ar[hz])])


def judge_video(frames_bgr: np.ndarray, fps: float, png_dir=None, name: str = "clip") -> dict:
    """uint8 BGR [T,H,W,3] + fps → 라벨·조항·근거 문장 (+ png_dir 가 있으면 근거 그림)."""
    from pathlib import Path
    f1 = judge_luminance(frames_bgr, fps)
    f2 = judge_red(frames_bgr, fps)
    pm = judge_pattern_motion(frames_bgr, fps)
    findings = []
    s = _flash_sentence("휘도 깜빡임", f1)
    if s:
        findings.append({"label": "휘도 플래시", "clause": f1["evidence"]["clause"], "sentence": s, "evidence": f1["evidence"],
                         "_frame": _rep_frame(f1, fps), "_cells": f1["evidence"]["cells"], "_shape": f1["evidence"]["grid_shape"]})
    s = _flash_sentence("적색 깜빡임", f2)
    if s:
        findings.append({"label": "적색 플래시", "clause": f2["evidence"]["clause"], "sentence": s, "evidence": f2["evidence"],
                         "_frame": _rep_frame(f2, fps), "_cells": f2["evidence"]["cells"], "_shape": f2["evidence"]["grid_shape"]})
    for r_, nm in ((f1, "휘도 연장"), (f2, "적색 연장")):
        fx = None if r_["dangerous"] else _extended(r_, fps, nm)       # 1 초 기준 위반이 없을 때만(팀원 판과 같은 우선순위)
        if fx:
            fx.update(_frame=int(fx["evidence"]["first_fail_s"] * fps), _cells=[], _shape=r_["evidence"]["grid_shape"])
            findings.append(fx)
    h, w = frames_bgr.shape[1:3]
    shp = [max(1, int(round(32 * h / w))), 32]
    if pm["p1"] and pm["best1"]:
        b = pm["best1"]
        findings.append({"label": "정적 공간 패턴", "clause": CLAUSE_P1,
                         "sentence": (f"{_sec(b['t'])} 무렵, 화면의 {100 * (b['area_static'] + b['area_change']):.0f}%에 {b['kind']} 줄무늬 {b['pairs']}쌍"
                                      f"(약 {b['cpd']:.1f} cpd)이 멈춰 있음(흐름 면제 {100 * b['area_flow']:.0f}% 제외), 영역 곳곳 {b['verified_seeds']} 지점에서 규격대로 확인 "
                                      "→ 기준(5쌍 초과, 면적 40 % 초과, 0.5초 이상) 넘음"),
                         "evidence": {k: b[k] for k in ("t", "pairs", "cpd", "kind", "verified_seeds", "area_static", "area_change", "area_flow")},
                         "_frame": int(b["t"] * fps), "_cells": b["cells"]["static"] + b["cells"]["change"], "_flow": b["cells"]["flow"], "_shape": shp})
    if pm["p2"] and pm["best2"]:
        b = pm["best2"]
        findings.append({"label": "변화 공간 패턴", "clause": CLAUSE_P2,
                         "sentence": (f"{_sec(b['t'])} 무렵, 화면의 {100 * b['area_change']:.0f}%에서 줄무늬 {b['pairs']}쌍(약 {b['cpd']:.1f} cpd)이 "
                                      f"{b['reason']} → 기준(5쌍 초과, 면적 25 % 초과, 한 방향 흐름 아님) 넘음"),
                         "evidence": {k: b[k] for k in ("t", "pairs", "cpd", "kind", "verified_seeds", "area_static", "area_change", "area_flow", "reason")},
                         "_frame": int(b["t"] * fps), "_cells": b["cells"]["change"], "_flow": b["cells"]["flow"], "_shape": shp})
    if png_dir is not None:
        from .grid_render import draw
        for i, f in enumerate(findings):
            t = min(max(0, f["_frame"]), len(frames_bgr) - 1)
            layers = [(f["label"], f["_cells"], tuple(f["_shape"]))]
            if f.get("_flow"):
                layers.append(("흐름(면제)", f["_flow"], tuple(f["_shape"])))
            f["grid_png"] = draw(frames_bgr[t], layers, f"{name}  t={t / fps:.2f}s  {f['label']}", Path(png_dir) / f"{name}_{i}_{t / fps:.2f}s.png")
    for f in findings:
        for k in [k for k in f if k.startswith("_")]:
            f.pop(k)
    notes = []
    ba = pm.get("best_any")
    if ba and ba["area_flow"] > 0.25 and not (pm["p1"] or pm["p2"]):          # 면제로 안전이 된 경우도 이유를 남긴다(설명 가능성)
        notes.append(f"{_sec(ba['t'])} 무렵 화면의 {100 * ba['area_flow']:.0f}%에 줄무늬 {ba['pairs']}쌍이 있으나 한 방향으로 매끄럽게 흐름 "
                     "→ 규격의 흐름 면제(BT.1702-3 Att.1)로 패턴 위험 아님")
    return {"dangerous": bool(findings), "labels": [f["label"] for f in findings], "findings": findings, "notes": notes,
            "not_implemented": NOT_IMPLEMENTED,
            "summary": ("위험: " + " / ".join(f["sentence"] for f in findings)) if findings else
                       "기준을 넘는 휘도·적색 깜빡임, 연장 깜빡임, 정지·변화 패턴이 없음"}
