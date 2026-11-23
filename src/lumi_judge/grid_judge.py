"""격자 판정기 G — 통합 판정 + 근거 문장 (에이전트 도구가 부를 함수). docs/GRID_JUDGE_DESIGN.md §4.

    from lumi_judge.grid_judge import judge_video
    r = judge_video(frames_bgr_uint8, fps)        # → {"dangerous", "labels", "findings":[{label, clause, sentence, evidence}], "grid_png"(선택)}

현재 포함: F1 휘도 깜빡임 · F2 적색 깜빡임(measure.py 와 판정 동일) · P1 정지 패턴(적합성 시험 21/21).
아직 없음: F3 연장 깜빡임 · P2 변화 패턴·흐름 면제 — 결과에 '미구현' 으로 명시한다(없는 것을 안전이라 말하지 않음).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .grid_flash import judge_luminance, judge_red, luminance_cd
from .grid_pattern import CLAUSE_P1, GridPatternJudge

NOT_IMPLEMENTED = ["변화 패턴·흐름 면제(P2)", "빠른 장면 전환·교집합(팀원 판의 별도 라벨 — G 에서는 휘도 깜빡임 규칙에 포함)"]
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


def judge_video(frames_bgr: np.ndarray, fps: float) -> dict:
    """uint8 BGR [T,H,W,3] + fps → 라벨·조항·근거 문장."""
    f1 = judge_luminance(frames_bgr, fps)
    f2 = judge_red(frames_bgr, fps)
    p1 = judge_pattern_clip(frames_bgr, fps)
    findings = []
    s = _flash_sentence("휘도 깜빡임", f1)
    if s:
        findings.append({"label": "휘도 플래시", "clause": f1["evidence"]["clause"], "sentence": s, "evidence": f1["evidence"]})
    s = _flash_sentence("적색 깜빡임", f2)
    if s:
        findings.append({"label": "적색 플래시", "clause": f2["evidence"]["clause"], "sentence": s, "evidence": f2["evidence"]})
    for r_, nm in ((f1, "휘도 연장"), (f2, "적색 연장")):
        fx = None if r_["dangerous"] else _extended(r_, fps, nm)       # 1 초 기준 위반이 없을 때만(팀원 판과 같은 우선순위)
        if fx:
            findings.append(fx)
    if p1["dangerous"] and p1["best"]:
        b = p1["best"]
        findings.append({"label": "정적 공간 패턴", "clause": CLAUSE_P1,
                         "sentence": (f"{_sec(b['t'])} 무렵, 화면의 {100 * b['area']:.0f}%에 {b['kind']} 줄무늬 {b['pairs']}쌍(약 {b['cpd']:.1f} cpd), "
                                      f"영역 곳곳 {b.get('verified_seeds', '?')} 지점에서 규격대로 확인 → 기준(5쌍 초과, 면적 40 % 초과, 0.5초 이상) 넘음"),
                         "evidence": {k: b[k] for k in ("area", "pairs", "cpd", "kind", "cells", "verified_seeds", "t") if k in b}})
    return {"dangerous": bool(findings), "labels": [f["label"] for f in findings], "findings": findings,
            "not_implemented": NOT_IMPLEMENTED,
            "summary": ("위험: " + " / ".join(f["sentence"] for f in findings)) if findings else
                       "기준을 넘는 휘도·적색 깜빡임, 연장 깜빡임, 정지 패턴이 없음 (변화 패턴은 아직 판정하지 않음)"}
